"""Nested patient-grouped training for the three-year multimodal endpoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from glaucoma_forecast.data.grape_multimodal import (
    CLINICAL_COLUMNS,
    clinical_vector,
)
from glaucoma_forecast.data.grape_progression import patient_stratified_folds
from glaucoma_forecast.data.highres import preprocess_high_resolution
from glaucoma_forecast.evaluation.calibration import expected_calibration_error
from glaucoma_forecast.evaluation.progression_metrics import (
    balanced_accuracy,
    binary_classification_metrics,
)
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.multimodal_progression import (
    build_multimodal_progression_model,
)
from glaucoma_forecast.training.risk_trainer import _image_tensor
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


WARNING = (
    "Research-only approximately-three-year visual-field progression estimate "
    "for eyes with known glaucoma. Not validated for patient care."
)


@dataclass(frozen=True)
class MultimodalProgressionConfig:
    manifest: str
    output_dir: str
    variant: str = "multimodal"
    image_size: int = 512
    backbone: str = "compact"
    feature_dim: int = 128
    folds: int = 5
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    freeze_image_encoder: bool = True
    device: str = "auto"
    seed: int = 20260730
    bootstrap_samples: int = 500
    initialize_retinal_encoder_from: str | None = None


def _categorical(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            1.0 if str(row.get("sex", "")).upper() == "M" else 0.0,
            1.0 if str(row.get("glaucoma_type", "")).upper() == "OAG" else 0.0,
            1.0 if str(row.get("laterality", "")).upper() == "OD" else 0.0,
        ],
        dtype=np.float32,
    )


def _fit_scaler(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = []
    masks = []
    for _, row in frame.iterrows():
        value, mask = clinical_vector(row)
        values.append(value)
        masks.append(mask)
    value_array = np.stack(values)
    mask_array = np.stack(masks)
    denominator = np.maximum(mask_array.sum(axis=0), 1.0)
    mean = (value_array * mask_array).sum(axis=0) / denominator
    variance = (((value_array - mean) * mask_array) ** 2).sum(axis=0) / denominator
    scale = np.sqrt(np.maximum(variance, 1e-6))
    return mean.astype(np.float32), scale.astype(np.float32)


def encode_clinical(
    row: pd.Series | dict[str, object],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values, mask = clinical_vector(row)
    normalized = ((values - mean) / scale) * mask
    return np.concatenate([normalized, mask, _categorical(pd.Series(row))]).astype(
        np.float32
    )


class MultimodalProgressionDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        mean: np.ndarray,
        scale: np.ndarray,
        load_images: bool = True,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.mean = mean
        self.scale = scale
        self.load_images = load_images
        self._image_cache: dict[str, tuple[object, object]] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch = require_torch()
        row = self.frame.iloc[index]
        if self.load_images:
            image_path = str(row["image_path"])
            if image_path not in self._image_cache:
                processed = preprocess_high_resolution(image_path, self.image_size)
                self._image_cache[image_path] = (
                    _image_tensor(processed.full_field),
                    _image_tensor(processed.optic_disc),
                )
            full_field, optic_disc = self._image_cache[image_path]
        else:
            full_field = torch.zeros(
                (3, self.image_size, self.image_size), dtype=torch.float32
            )
            optic_disc = full_field.clone()
        return {
            "full_field": full_field,
            "optic_disc": optic_disc,
            "clinical": torch.from_numpy(
                encode_clinical(row, self.mean, self.scale)
            ),
            "label": float(row["primary_progression_label"]),
        }


def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
    torch = require_torch()
    return {
        "full_field": torch.stack([item["full_field"] for item in batch]),
        "optic_disc": torch.stack([item["optic_disc"] for item in batch]),
        "clinical": torch.stack([item["clinical"] for item in batch]),
        "label": torch.tensor(
            [item["label"] for item in batch], dtype=torch.float32
        ),
    }


def _temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.geomspace(0.25, 8.0, 121)
    losses = []
    for value in candidates:
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits / value, -30, 30)))
        loss = -np.mean(
            labels * np.log(probability + 1e-8)
            + (1 - labels) * np.log(1 - probability + 1e-8)
        )
        losses.append(float(loss))
    return float(candidates[int(np.argmin(losses))])


def _threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], probability]))
    return float(
        max(
            candidates,
            key=lambda value: balanced_accuracy(labels, probability >= value),
        )
    )


def _loader(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    mean: np.ndarray,
    scale: np.ndarray,
    shuffle: bool,
    load_images: bool = True,
):
    torch = require_torch()
    return torch.utils.data.DataLoader(
        MultimodalProgressionDataset(
            frame, image_size, mean, scale, load_images=load_images
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate,
    )


def _predict(model, loader, device: str) -> np.ndarray:
    torch = require_torch()
    logits: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["full_field"].to(device),
                batch["optic_disc"].to(device),
                batch["clinical"].to(device),
            )
            logits.extend(output["progression_logit"].cpu().tolist())
    return np.asarray(logits, dtype=float)


def _patient_bootstrap(
    frame: pd.DataFrame, samples: int, seed: int
) -> dict[str, list[float]]:
    patients = frame["patient_id"].astype(str).unique()
    if samples <= 0 or len(patients) < 2:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auroc": [], "auprc": []}
    grouped = {patient: frame[frame["patient_id"].astype(str) == patient] for patient in patients}
    for _ in range(samples):
        selected = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([grouped[patient] for patient in selected], ignore_index=True)
        if sample["label"].nunique() < 2:
            continue
        metrics = binary_classification_metrics(
            sample["label"].to_numpy(dtype=int),
            sample["probability"].to_numpy(dtype=float),
            threshold=0.5,
        )
        values["auroc"].append(metrics["auroc"])
        values["auprc"].append(metrics["auprc"])
    return {
        key: [
            float(np.quantile(metric_values, 0.025)),
            float(np.quantile(metric_values, 0.975)),
        ]
        for key, metric_values in values.items()
        if metric_values
    }


def _validate_manifest(frame: pd.DataFrame, folds: int) -> pd.DataFrame:
    required = {
        "patient_id",
        "eye_id",
        "image_path",
        "primary_progression_label",
        "outer_fold",
        *CLINICAL_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Three-year progression manifest missing columns: {missing}")
    result = frame.copy()
    result["primary_progression_label"] = pd.to_numeric(
        result["primary_progression_label"], errors="raise"
    ).astype(int)
    if result["outer_fold"].nunique() != folds:
        raise ValueError("Frozen outer folds do not match requested folds")
    if result.groupby("patient_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("A patient appears in more than one outer fold")
    return result


def train_multimodal_progression_cv(
    config: MultimodalProgressionConfig,
) -> dict[str, object]:
    torch = require_torch()
    seed_everything(config.seed)
    device = select_device(config.device)
    frame = _validate_manifest(pd.read_csv(config.manifest), config.folds)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_rows: list[dict[str, object]] = []
    fold_reports: list[dict[str, object]] = []
    clinical_dim = len(CLINICAL_COLUMNS) * 2 + 3

    for fold in range(config.folds):
        seed_everything(config.seed + fold)
        test = frame[frame["outer_fold"] == fold].reset_index(drop=True)
        development = frame[frame["outer_fold"] != fold].reset_index(drop=True)
        inner_folds = patient_stratified_folds(
            development,
            folds=min(4, development["patient_id"].nunique()),
            seed=config.seed + 100 + fold,
        )
        validation = development[inner_folds == 0].reset_index(drop=True)
        train = development[inner_folds != 0].reset_index(drop=True)
        mean, scale = _fit_scaler(train)
        model = build_multimodal_progression_model(
            clinical_dim=clinical_dim,
            feature_dim=config.feature_dim,
            backbone=config.backbone,
            variant=config.variant,
        ).to(device)
        if config.initialize_retinal_encoder_from:
            transfer = torch.load(
                config.initialize_retinal_encoder_from,
                map_location=device,
                weights_only=False,
            )
            source = transfer["model"]
            destination = model.state_dict()
            reusable = {}
            for key, value in source.items():
                destination_key = f"image_model.{key}"
                if (
                    (
                        key.startswith("encoder.")
                        or key.startswith("fusion.")
                        or key in {"image_mean", "image_std"}
                    )
                    and destination_key in destination
                    and destination[destination_key].shape == value.shape
                ):
                    reusable[destination_key] = value
            if not any(key.startswith("image_model.encoder.") for key in reusable):
                raise ValueError(
                    "Transfer checkpoint has no compatible retinal encoder weights"
                )
            model.load_state_dict(reusable, strict=False)
        if config.freeze_image_encoder:
            for parameter in model.image_model.encoder.parameters():
                parameter.requires_grad = False
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        positives = int(train["primary_progression_label"].sum())
        negatives = len(train) - positives
        pos_weight = torch.tensor(
            [negatives / max(positives, 1)], dtype=torch.float32, device=device
        )
        train_loader = _loader(
            train,
            config.image_size,
            config.batch_size,
            mean,
            scale,
            shuffle=True,
            load_images=config.variant != "clinical_only",
        )
        validation_loader = _loader(
            validation,
            config.image_size,
            config.batch_size,
            mean,
            scale,
            shuffle=False,
            load_images=config.variant != "clinical_only",
        )
        best_loss = float("inf")
        best_state = None
        history = []
        for epoch in range(1, config.epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                output = model(
                    batch["full_field"].to(device),
                    batch["optic_disc"].to(device),
                    batch["clinical"].to(device),
                )
                labels = batch["label"].to(device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["progression_logit"], labels, pos_weight=pos_weight
                )
                if config.variant == "multimodal":
                    loss = loss + 0.15 * (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            output["image_logit"], labels, pos_weight=pos_weight
                        )
                        + torch.nn.functional.binary_cross_entropy_with_logits(
                            output["clinical_logit"], labels, pos_weight=pos_weight
                        )
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_logits = _predict(model, validation_loader, device)
            validation_labels = validation[
                "primary_progression_label"
            ].to_numpy(dtype=int)
            validation_loss = float(
                np.mean(
                    np.maximum(validation_logits, 0)
                    - validation_logits * validation_labels
                    + np.log1p(np.exp(-np.abs(validation_logits)))
                )
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if best_state is None:
            raise RuntimeError("No multimodal checkpoint was selected")
        model.load_state_dict(best_state)
        validation_logits = _predict(model, validation_loader, device)
        validation_labels = validation["primary_progression_label"].to_numpy(
            dtype=int
        )
        temperature = _temperature(validation_logits, validation_labels)
        validation_probability = 1.0 / (
            1.0 + np.exp(-np.clip(validation_logits / temperature, -30, 30))
        )
        threshold = _threshold(validation_labels, validation_probability)
        test_logits = _predict(
            model,
            _loader(
                test,
                config.image_size,
                config.batch_size,
                mean,
                scale,
                shuffle=False,
                load_images=config.variant != "clinical_only",
            ),
            device,
        )
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(test_logits / temperature, -30, 30))
        )
        labels = test["primary_progression_label"].to_numpy(dtype=int)
        metrics = binary_classification_metrics(labels, probability, threshold)
        metrics["ece"] = expected_calibration_error(labels, probability, bins=5)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        checkpoint = {
            "schema_version": "3.0",
            "model": best_state,
            "config": asdict(config),
            "clinical_columns": list(CLINICAL_COLUMNS),
            "clinical_mean": mean,
            "clinical_scale": scale,
            "clinical_dim": clinical_dim,
            "temperature": temperature,
            "decision_threshold": threshold,
            "outer_fold": fold,
            "endpoint": "plr2_progression_complete_followup_2p5_to_3p5_years",
            "model_version": f"progression-3y-{config.variant}-fold-{fold}",
            "warning": WARNING,
        }
        torch.save(checkpoint, fold_dir / "best.pt")
        fold_report = {
            "fold": fold,
            "train_eyes": len(train),
            "validation_eyes": len(validation),
            "test_eyes": len(test),
            "metrics": metrics,
            "history": history,
        }
        (fold_dir / "report.json").write_text(json.dumps(fold_report, indent=2))
        fold_reports.append(fold_report)
        for row, logit, score in zip(
            test.to_dict("records"), test_logits, probability
        ):
            oof_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "eye_id": row["eye_id"],
                    "outer_fold": fold,
                    "label": int(row["primary_progression_label"]),
                    "logit": float(logit),
                    "probability": float(score),
                    "decision_threshold": threshold,
                    "predicted_label": int(score >= threshold),
                }
            )

    oof = pd.DataFrame(oof_rows).sort_values(
        ["outer_fold", "patient_id", "eye_id"]
    )
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    aggregate = binary_classification_metrics(
        oof["label"].to_numpy(dtype=int),
        oof["probability"].to_numpy(dtype=float),
        threshold=0.5,
    )
    aggregate["fold_specific_balanced_accuracy"] = balanced_accuracy(
        oof["label"].to_numpy(dtype=int),
        oof["predicted_label"].to_numpy(dtype=int),
    )
    aggregate["ece_5_bins"] = expected_calibration_error(
        oof["label"].to_numpy(dtype=int),
        oof["probability"].to_numpy(dtype=float),
        bins=5,
    )
    report = {
        "status": "cross_validation_complete",
        "schema_version": "3.0",
        "endpoint": "approximately_three_year_plr2_progression",
        "variant": config.variant,
        "eyes": len(frame),
        "patients": int(frame["patient_id"].nunique()),
        "progressors": int(frame["primary_progression_label"].sum()),
        "aggregate_oof": aggregate,
        "patient_bootstrap_95ci": _patient_bootstrap(
            oof, config.bootstrap_samples, config.seed
        ),
        "folds": fold_reports,
        "config": asdict(config),
        "warning": WARNING,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
