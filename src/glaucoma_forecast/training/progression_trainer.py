"""Patient-grouped cross-validation for baseline glaucoma progression."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from glaucoma_forecast.data.grape_progression import patient_stratified_folds
from glaucoma_forecast.evaluation.calibration import expected_calibration_error
from glaucoma_forecast.evaluation.progression_metrics import (
    balanced_accuracy,
    binary_classification_metrics,
)
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.risk_model import build_dual_view_risk_model
from glaucoma_forecast.training.risk_trainer import RiskCohortDataset, _collate
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


WARNING = (
    "Research-only progression susceptibility for eyes already diagnosed with "
    "glaucoma. Not fixed-horizon risk, diagnosis, prognosis, or patient care."
)


@dataclass(frozen=True)
class ProgressionCVConfig:
    manifest: str
    output_dir: str
    image_size: int = 512
    backbone: str = "compact"
    feature_dim: int = 128
    folds: int = 5
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    freeze_encoder: bool = False
    label_column: str = "primary_progression_label"
    device: str = "auto"
    seed: int = 20260729
    initialize_retinal_encoder_from: str | None = None


def _prepare(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    required = {"patient_id", "eye_id", "image_path", label_column, "outer_fold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Progression cohort missing columns: {missing}")
    result = frame.dropna(subset=[label_column]).copy()
    result[label_column] = pd.to_numeric(result[label_column], errors="raise").astype(int)
    if not result[label_column].isin([0, 1]).all():
        raise ValueError("Progression labels must contain only 0/1")
    if result.duplicated(["patient_id", "eye_id"]).any():
        raise ValueError("Progression cohort must contain one baseline image per eye")
    # Reuse the dual-view loader without claiming survival supervision.
    result["event_observed"] = result[label_column]
    result["event_or_censor_time_years"] = 1.0
    result["incidence_eligible"] = 0.0
    result["current_glaucoma_label"] = np.nan
    return result


def _temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.geomspace(0.25, 8.0, 121)
    best = 1.0
    best_loss = float("inf")
    for value in candidates:
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits / value, -30, 30)))
        loss = -np.mean(
            labels * np.log(probability + 1e-8)
            + (1 - labels) * np.log(1 - probability + 1e-8)
        )
        if loss < best_loss:
            best_loss = float(loss)
            best = float(value)
    return best


def _threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], probability]))
    return float(
        max(
            candidates,
            key=lambda value: balanced_accuracy(labels, probability >= value),
        )
    )


def _loader(frame: pd.DataFrame, image_size: int, batch_size: int, shuffle: bool):
    torch = require_torch()
    return torch.utils.data.DataLoader(
        RiskCohortDataset(frame, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate,
    )


def _predict(model, loader, device: str):
    torch = require_torch()
    logits: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["full_field"].to(device), batch["optic_disc"].to(device)
            )
            logits.extend(output["current_glaucoma_logit"].cpu().tolist())
    return np.asarray(logits, dtype=float)


def train_progression_cross_validation(
    config: ProgressionCVConfig,
) -> dict[str, object]:
    torch = require_torch()
    seed_everything(config.seed)
    device = select_device(config.device)
    frame = _prepare(pd.read_csv(config.manifest), config.label_column)
    if frame["outer_fold"].nunique() != config.folds:
        raise ValueError("Frozen outer folds do not match requested fold count")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    oof_rows: list[dict[str, object]] = []
    fold_reports = []
    for fold in range(config.folds):
        seed_everything(config.seed + fold)
        test = frame[frame["outer_fold"] == fold].reset_index(drop=True)
        development = frame[frame["outer_fold"] != fold].reset_index(drop=True)
        inner = patient_stratified_folds(
            development,
            folds=min(5, development["patient_id"].nunique()),
            seed=config.seed + 100 + fold,
            label_column=config.label_column,
        )
        validation = development[inner == 0].reset_index(drop=True)
        train = development[inner != 0].reset_index(drop=True)
        model = build_dual_view_risk_model(
            years=1, feature_dim=config.feature_dim, backbone=config.backbone
        ).to(device)
        if config.initialize_retinal_encoder_from:
            transfer = torch.load(
                config.initialize_retinal_encoder_from,
                map_location=device,
                weights_only=False,
            )
            source = transfer["model"]
            destination = model.state_dict()
            reusable = {
                key: value
                for key, value in source.items()
                if (
                    key.startswith("encoder.")
                    or key.startswith("fusion.")
                    or key in {"image_mean", "image_std"}
                )
                and key in destination
                and destination[key].shape == value.shape
            }
            if not any(key.startswith("encoder.") for key in reusable):
                raise ValueError(
                    "Transfer checkpoint has no shape-compatible encoder weights"
                )
            model.load_state_dict(reusable, strict=False)
        if config.freeze_encoder:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
        positive = int(train[config.label_column].sum())
        negative = len(train) - positive
        pos_weight = torch.tensor(
            [negative / max(positive, 1)], dtype=torch.float32, device=device
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        train_loader = _loader(train, config.image_size, config.batch_size, True)
        validation_loader = _loader(
            validation, config.image_size, config.batch_size, False
        )
        best_loss = float("inf")
        best_state = None
        history = []
        for epoch in range(1, config.epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                logits = model(
                    batch["full_field"].to(device),
                    batch["optic_disc"].to(device),
                )["current_glaucoma_logit"]
                labels = batch["event"].to(device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_logits = _predict(model, validation_loader, device)
            validation_labels = validation[config.label_column].to_numpy(dtype=int)
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
        assert best_state is not None
        model.load_state_dict(best_state)
        validation_logits = _predict(model, validation_loader, device)
        validation_labels = validation[config.label_column].to_numpy(dtype=int)
        temperature = _temperature(validation_logits, validation_labels)
        validation_probability = 1.0 / (
            1.0 + np.exp(-np.clip(validation_logits / temperature, -30, 30))
        )
        decision_threshold = _threshold(
            validation_labels, validation_probability
        )
        test_logits = _predict(
            model, _loader(test, config.image_size, config.batch_size, False), device
        )
        test_probability = 1.0 / (
            1.0 + np.exp(-np.clip(test_logits / temperature, -30, 30))
        )
        labels = test[config.label_column].to_numpy(dtype=int)
        metrics = binary_classification_metrics(
            labels, test_probability, decision_threshold
        )
        metrics["ece"] = expected_calibration_error(labels, test_probability, bins=5)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "model": best_state,
                "config": asdict(config),
                "outer_fold": fold,
                "temperature": temperature,
                "decision_threshold": decision_threshold,
                "warning": WARNING,
                "endpoint": "progression_during_observed_followup",
            },
            fold_dir / "best.pt",
        )
        fold_report = {
            "fold": fold,
            "train_eyes": len(train),
            "validation_eyes": len(validation),
            "test_eyes": len(test),
            "temperature": temperature,
            "decision_threshold": decision_threshold,
            "history": history,
            "metrics": metrics,
        }
        (fold_dir / "report.json").write_text(json.dumps(fold_report, indent=2))
        fold_reports.append(fold_report)
        for row, logit, probability in zip(
            test.to_dict("records"), test_logits, test_probability
        ):
            oof_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "eye_id": row["eye_id"],
                    "outer_fold": fold,
                    "label": int(row[config.label_column]),
                    "logit": float(logit),
                    "probability": float(probability),
                    "decision_threshold": decision_threshold,
                    "predicted_label": int(probability >= decision_threshold),
                }
            )
    oof = pd.DataFrame(oof_rows).sort_values(["outer_fold", "patient_id", "eye_id"])
    oof.to_csv(output / "oof_predictions.csv", index=False)
    labels = oof["label"].to_numpy(dtype=int)
    probability = oof["probability"].to_numpy(dtype=float)
    prediction = oof["predicted_label"].to_numpy(dtype=int)
    aggregate = binary_classification_metrics(labels, probability, threshold=0.5)
    aggregate["fold_specific_balanced_accuracy"] = balanced_accuracy(labels, prediction)
    aggregate["ece_5_bins"] = expected_calibration_error(labels, probability, bins=5)
    report = {
        "status": "cross_validation_complete",
        "endpoint": "documented_progression_during_observed_study_followup",
        "fixed_horizon_claim": False,
        "model": "dual_view_progression_susceptibility",
        "initialization": (
            "retinal_current_glaucoma_transfer"
            if config.initialize_retinal_encoder_from
            else "configured_backbone_default"
        ),
        "config": asdict(config),
        "aggregate_oof": aggregate,
        "folds": fold_reports,
        "warning": WARNING,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report
