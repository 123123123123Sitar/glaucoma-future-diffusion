"""Training loop for the single-image discrete-time glaucoma-risk model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from glaucoma_forecast.data.highres import preprocess_high_resolution
from glaucoma_forecast.evaluation.survival_metrics import binary_auc
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.risk_model import build_dual_view_risk_model
from glaucoma_forecast.training.survival import discrete_time_survival_loss
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


REQUIRED_COLUMNS = {
    "patient_id",
    "eye_id",
    "image_path",
    "event_observed",
    "event_or_censor_time_years",
}


@dataclass(frozen=True)
class RiskTrainingConfig:
    manifest: str
    output_dir: str
    image_size: int = 512
    years: int = 10
    backbone: str = "compact"
    feature_dim: int = 256
    batch_size: int = 2
    epochs: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    seed: int = 20260720
    val_fraction: float = 0.2
    current_loss_weight: float = 0.25
    vcdr_loss_weight: float = 0.10
    mask_loss_weight: float = 0.20
    freeze_backbone_epochs: int = 0


def validate_risk_manifest(frame: pd.DataFrame, years: int = 10) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Risk manifest missing columns: {missing}")
    if frame.empty:
        raise ValueError("Risk manifest is empty")
    times = pd.to_numeric(frame["event_or_censor_time_years"], errors="coerce")
    events = pd.to_numeric(frame["event_observed"], errors="coerce")
    eligible = (
        pd.to_numeric(frame["incidence_eligible"], errors="coerce").fillna(1.0)
        if "incidence_eligible" in frame
        else pd.Series(1.0, index=frame.index)
    )
    if times.isna().any() or (times < 0).any():
        raise ValueError("event_or_censor_time_years must be non-negative numeric values")
    if not events.isin([0, 1]).all():
        raise ValueError("event_observed must contain only 0/1")
    if not eligible.isin([0, 1]).all():
        raise ValueError("incidence_eligible must contain only 0/1 when supplied")
    event_times = times[(events == 1) & (eligible == 1)]
    if not event_times.empty and (event_times > years).any():
        raise ValueError(f"Observed event times must not exceed the {years}-year model horizon")
    duplicates = frame.duplicated(["patient_id", "eye_id"]).sum()
    if duplicates:
        raise ValueError(
            "Use one baseline photograph per eye for incidence training; "
            f"found {int(duplicates)} duplicate patient/eye rows"
        )


def _image_tensor(image: Image.Image):
    torch = require_torch()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


class RiskCohortDataset:
    def __init__(self, frame: pd.DataFrame, image_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        processed = preprocess_high_resolution(row["image_path"], self.image_size)
        incidence_eligible = row.get("incidence_eligible", 1.0)
        if pd.isna(incidence_eligible):
            incidence_eligible = 1.0
        mask = np.full((2, self.image_size, self.image_size), np.nan, dtype=np.float32)
        for channel, column in enumerate(["disc_mask_path", "cup_mask_path"]):
            mask_path = row.get(column)
            if isinstance(mask_path, str) and mask_path:
                with Image.open(mask_path) as source:
                    resized = source.convert("L").resize(
                        (self.image_size, self.image_size), Image.Resampling.NEAREST
                    )
                mask[channel] = np.asarray(resized, dtype=np.float32) / 255.0
        item = {
            "full_field": _image_tensor(processed.full_field),
            "optic_disc": _image_tensor(processed.optic_disc),
            "time": float(row["event_or_censor_time_years"]),
            "event": float(row["event_observed"]),
            "incidence_eligible": float(incidence_eligible),
            "current_label": float(row.get("current_glaucoma_label", np.nan)),
            "vcdr": float(row.get("vcdr", np.nan)),
            "anatomy_masks": require_torch().from_numpy(mask),
        }
        return item


def _collate(batch: list[dict]):
    torch = require_torch()
    return {
        "full_field": torch.stack([item["full_field"] for item in batch]),
        "optic_disc": torch.stack([item["optic_disc"] for item in batch]),
        "time": torch.tensor([item["time"] for item in batch], dtype=torch.float32),
        "event": torch.tensor([item["event"] for item in batch], dtype=torch.float32),
        "incidence_eligible": torch.tensor(
            [item["incidence_eligible"] for item in batch], dtype=torch.float32
        ),
        "current_label": torch.tensor([item["current_label"] for item in batch], dtype=torch.float32),
        "vcdr": torch.tensor([item["vcdr"] for item in batch], dtype=torch.float32),
        "anatomy_masks": torch.stack([item["anatomy_masks"] for item in batch]),
    }


def _patient_split(frame: pd.DataFrame, fraction: float, seed: int):
    patients = frame["patient_id"].astype(str).unique()
    if len(patients) < 5:
        raise ValueError("Need at least five patients for a patient-level train/validation split")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * fraction)))
    val_ids = set(patients[:n_val])
    train = frame[~frame["patient_id"].astype(str).isin(val_ids)].reset_index(drop=True)
    val = frame[frame["patient_id"].astype(str).isin(val_ids)].reset_index(drop=True)
    return train, val


def _optional_masked_loss(prediction, target, kind: str, pos_weight=None):
    torch = require_torch()
    available = torch.isfinite(target)
    if not available.any():
        return prediction.sum() * 0.0
    if kind == "bce":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[available], target[available], pos_weight=pos_weight
        )
    return torch.nn.functional.smooth_l1_loss(prediction[available], target[available])


def _optional_segmentation_loss(logits, target):
    torch = require_torch()
    available = torch.isfinite(target).flatten(1).all(dim=1)
    if not available.any():
        return logits.sum() * 0.0
    prediction = logits[available]
    truth = target[available]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(prediction, truth)
    probability = torch.sigmoid(prediction)
    intersection = (probability * truth).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + truth.sum(dim=(2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_loss


def train_risk_model(config: RiskTrainingConfig) -> dict[str, object]:
    torch = require_torch()
    seed_everything(config.seed)
    device = select_device(config.device)
    frame = pd.read_csv(config.manifest)
    validate_risk_manifest(frame, config.years)
    train_frame, val_frame = _patient_split(frame, config.val_fraction, config.seed)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(out / "train_manifest.csv", index=False)
    val_frame.to_csv(out / "val_manifest.csv", index=False)
    train_data = RiskCohortDataset(train_frame, config.image_size)
    val_data = RiskCohortDataset(val_frame, config.image_size)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=_collate
    )
    model = build_dual_view_risk_model(config.years, config.feature_dim, config.backbone).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_val = float("inf")
    history = []
    current_values = (
        pd.to_numeric(frame["current_glaucoma_label"], errors="coerce").dropna()
        if "current_glaucoma_label" in frame
        else pd.Series(dtype=float)
    )
    current_head_trained = bool(set(current_values.unique()) >= {0.0, 1.0})
    vcdr_head_trained = bool(
        "vcdr" in frame and pd.to_numeric(frame["vcdr"], errors="coerce").notna().any()
    )
    mask_head_trained = bool(
        {"disc_mask_path", "cup_mask_path"}.issubset(frame.columns)
        and frame[["disc_mask_path", "cup_mask_path"]].notna().all(axis=1).any()
    )
    eligible_values = (
        pd.to_numeric(frame["incidence_eligible"], errors="coerce").fillna(1.0)
        if "incidence_eligible" in frame
        else pd.Series(1.0, index=frame.index)
    )
    eligible_frame = frame[eligible_values == 1]
    hazard_head_trained = bool(
        not eligible_frame.empty
        and pd.to_numeric(eligible_frame["event_observed"], errors="coerce").isin([0, 1]).all()
        and pd.to_numeric(eligible_frame["event_observed"], errors="coerce").nunique() == 2
    )
    current_positive = int((current_values == 1).sum())
    current_negative = int((current_values == 0).sum())
    current_pos_weight = (
        torch.tensor(
            [current_negative / current_positive], dtype=torch.float32, device=device
        )
        if current_positive and current_negative
        else None
    )
    for epoch in range(1, config.epochs + 1):
        backbone_trainable = epoch > config.freeze_backbone_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad = backbone_trainable
        model.train()
        train_losses = []
        for batch in train_loader:
            full = batch["full_field"].to(device)
            disc = batch["optic_disc"].to(device)
            outputs = model(full, disc)
            survival = discrete_time_survival_loss(
                outputs["hazard_logits"],
                batch["time"].to(device),
                batch["event"].to(device),
                batch["incidence_eligible"].to(device),
            )
            current = _optional_masked_loss(
                outputs["current_glaucoma_logit"],
                batch["current_label"].to(device),
                "bce",
                pos_weight=current_pos_weight,
            )
            vcdr = _optional_masked_loss(outputs["vcdr"], batch["vcdr"].to(device), "l1")
            segmentation = _optional_segmentation_loss(
                outputs["disc_cup_mask_logits"], batch["anatomy_masks"].to(device)
            )
            loss = (
                survival
                + config.current_loss_weight * current
                + config.vcdr_loss_weight * vcdr
                + config.mask_loss_weight * segmentation
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        val_current_losses = []
        val_current_labels = []
        val_current_scores = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch["full_field"].to(device), batch["optic_disc"].to(device)
                )
                val_losses.append(
                    float(
                        discrete_time_survival_loss(
                            outputs["hazard_logits"],
                            batch["time"].to(device),
                            batch["event"].to(device),
                            batch["incidence_eligible"].to(device),
                        ).cpu()
                    )
                )
                labels = batch["current_label"].to(device)
                available = torch.isfinite(labels)
                if available.any():
                    logits = outputs["current_glaucoma_logit"][available]
                    val_current_losses.append(
                        float(
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                logits,
                                labels[available],
                                pos_weight=current_pos_weight,
                            ).cpu()
                        )
                    )
                    val_current_labels.extend(labels[available].cpu().tolist())
                    val_current_scores.extend(torch.sigmoid(logits).cpu().tolist())
        val_current_loss = (
            float(np.mean(val_current_losses)) if val_current_losses else float("nan")
        )
        val_current_auc = (
            binary_auc(
                np.asarray(val_current_labels, dtype=int),
                np.asarray(val_current_scores, dtype=float),
            )
            if val_current_labels
            else float("nan")
        )
        metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_survival_loss": float(np.mean(val_losses)),
            "val_current_loss": val_current_loss,
            "val_current_auc": val_current_auc,
        }
        history.append(metrics)
        checkpoint = {
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "metrics": metrics,
            "temperature": 1.0,
            "model_version": f"risk-v2-epoch-{epoch}",
            "current_head_trained": current_head_trained,
            "vcdr_head_trained": vcdr_head_trained,
            "mask_head_trained": mask_head_trained,
            "hazard_head_trained": hazard_head_trained,
        }
        torch.save(checkpoint, out / "latest.pt")
        selection_value = (
            metrics["val_survival_loss"] if hazard_head_trained else metrics["val_current_loss"]
        )
        if selection_value < best_val:
            best_val = selection_value
            torch.save(checkpoint, out / "best.pt")
        print(metrics, flush=True)
    summary = {
        "status": "trained_research_model",
        "train_eyes": len(train_frame),
        "val_eyes": len(val_frame),
        "events": int(
            (
                frame["event_observed"]
                * frame.get("incidence_eligible", pd.Series(1.0, index=frame.index))
            ).sum()
        ),
        "max_followup_years": float(frame["event_or_censor_time_years"].max()),
        "selection_metric": (
            "val_survival_loss" if hazard_head_trained else "val_current_loss"
        ),
        "best_val_loss": best_val,
        "current_head_trained": current_head_trained,
        "hazard_head_trained": hazard_head_trained,
        "checkpoint": str(out / "best.pt"),
        "history": history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
