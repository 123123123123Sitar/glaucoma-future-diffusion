"""Fixed-split Harvard-GDP visual-field progression benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from glaucoma_forecast.evaluation.calibration import expected_calibration_error
from glaucoma_forecast.evaluation.progression_metrics import (
    balanced_accuracy,
    binary_classification_metrics,
)
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


@dataclass(frozen=True)
class HarvardGDPConfig:
    manifest: str
    output_dir: str
    label_column: str = "progression.md"
    epochs: int = 300
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    calibration_fraction: float = 0.20
    bootstrap_samples: int = 2000
    device: str = "auto"
    seed: int = 20260730
    known_glaucoma_only: bool = True


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    td_columns = sorted(
        [column for column in frame if column.startswith("td")],
        key=lambda value: int(value[2:]),
    )
    return ["age", "md", *td_columns]


def _matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    race_categories: list[str],
) -> np.ndarray:
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    numeric_mask = np.isfinite(numeric).astype(np.float32)
    numeric = np.nan_to_num(numeric, nan=0.0)
    gender = (
        frame["gender"].astype(str).str.lower().eq("male").to_numpy(dtype=np.float32)
    )[:, None]
    hispanic = (
        frame["hispanic"].astype(str).str.lower().eq("yes").to_numpy(dtype=np.float32)
    )[:, None]
    race = np.stack(
        [
            frame["race"]
            .astype(str)
            .str.lower()
            .eq(category)
            .to_numpy(dtype=np.float32)
            for category in race_categories
        ],
        axis=1,
    )
    return np.concatenate([numeric, numeric_mask, gender, hispanic, race], axis=1)


def _stratified_calibration_indices(
    labels: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    calibration = []
    development = []
    for label in [0, 1]:
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        calibration.extend(indices[:count])
        development.extend(indices[count:])
    return np.asarray(development), np.asarray(calibration)


def _platt_parameters(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Fit an affine sigmoid on calibration data without touching test labels."""

    slopes = np.geomspace(0.125, 4.0, 81)
    biases = np.linspace(-3.0, 3.0, 121)
    best = (1.0, 0.0)
    best_loss = float("inf")
    for slope in slopes:
        transformed = slope * logits[:, None] + biases[None, :]
        probability = 1.0 / (1.0 + np.exp(-np.clip(transformed, -30, 30)))
        loss = -np.mean(
            labels[:, None] * np.log(probability + 1e-8)
            + (1 - labels[:, None]) * np.log(1 - probability + 1e-8),
            axis=0,
        )
        index = int(np.argmin(loss))
        if float(loss[index]) < best_loss:
            best_loss = float(loss[index])
            best = (float(slope), float(biases[index]))
    return best


def _threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], probability]))
    return float(
        max(
            candidates,
            key=lambda value: balanced_accuracy(labels, probability >= value),
        )
    )


def _bootstrap(
    labels: np.ndarray, probability: np.ndarray, samples: int, seed: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    metrics = {"auroc": [], "auprc": []}
    for _ in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[indices]).size < 2:
            continue
        values = binary_classification_metrics(
            labels[indices], probability[indices], threshold=0.5
        )
        metrics["auroc"].append(values["auroc"])
        metrics["auprc"].append(values["auprc"])
    return {
        key: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for key, values in metrics.items()
    }


def train_harvard_gdp_progression(
    config: HarvardGDPConfig,
) -> dict[str, object]:
    torch = require_torch()
    seed_everything(config.seed)
    device = select_device(config.device)
    frame = pd.read_csv(config.manifest)
    required = {
        "filename",
        "gender",
        "age",
        "race",
        "hispanic",
        "md",
        "progression_forecasting_use",
        config.label_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Harvard-GDP summary missing columns: {missing}")
    frame = frame.dropna(
        subset=["progression_forecasting_use", config.label_column]
    ).copy()
    if config.known_glaucoma_only:
        frame = frame[pd.to_numeric(frame["glaucoma"], errors="coerce") == 1].copy()
    train_frame = frame[
        frame["progression_forecasting_use"] == "training"
    ].reset_index(drop=True)
    test_frame = frame[
        frame["progression_forecasting_use"] == "test"
    ].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError("Harvard-GDP published training/test split is missing")
    feature_columns = _feature_columns(frame)
    race_categories = sorted(
        train_frame["race"].astype(str).str.lower().unique().tolist()
    )
    train_matrix = _matrix(train_frame, feature_columns, race_categories)
    test_matrix = _matrix(test_frame, feature_columns, race_categories)
    train_labels = train_frame[config.label_column].to_numpy(dtype=np.float32)
    test_labels = test_frame[config.label_column].to_numpy(dtype=int)
    development_indices, calibration_indices = _stratified_calibration_indices(
        train_labels.astype(int), config.calibration_fraction, config.seed
    )
    mean = train_matrix[development_indices].mean(axis=0)
    scale = np.maximum(train_matrix[development_indices].std(axis=0), 1e-6)
    train_matrix = (train_matrix - mean) / scale
    test_matrix = (test_matrix - mean) / scale

    class ProgressionMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(train_matrix.shape[1], config.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.25),
                torch.nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, features):
            return self.network(features).squeeze(1)

    model = ProgressionMLP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    development_x = torch.tensor(
        train_matrix[development_indices], dtype=torch.float32, device=device
    )
    development_y = torch.tensor(
        train_labels[development_indices], dtype=torch.float32, device=device
    )
    positives = float(development_y.sum().cpu())
    negatives = float(len(development_y) - positives)
    pos_weight = torch.tensor(
        [negatives / max(positives, 1.0)], dtype=torch.float32, device=device
    )
    best_loss = float("inf")
    best_state = None
    history = []
    calibration_x = torch.tensor(
        train_matrix[calibration_indices], dtype=torch.float32, device=device
    )
    calibration_y = train_labels[calibration_indices].astype(int)
    for epoch in range(1, config.epochs + 1):
        model.train()
        logits = model(development_x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, development_y, pos_weight=pos_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            calibration_logits = model(calibration_x).cpu().numpy()
        calibration_loss = float(
            np.mean(
                np.maximum(calibration_logits, 0)
                - calibration_logits * calibration_y
                + np.log1p(np.exp(-np.abs(calibration_logits)))
            )
        )
        history.append(
            {
                "epoch": epoch,
                "development_loss": float(loss.detach().cpu()),
                "calibration_loss": calibration_loss,
            }
        )
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("No Harvard-GDP checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        calibration_logits = model(calibration_x).cpu().numpy()
        test_logits = model(
            torch.tensor(test_matrix, dtype=torch.float32, device=device)
        ).cpu().numpy()
    calibration_slope, calibration_bias = _platt_parameters(
        calibration_logits, calibration_y
    )
    calibration_probability = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                calibration_slope * calibration_logits + calibration_bias,
                -30,
                30,
            )
        )
    )
    threshold = _threshold(calibration_y, calibration_probability)
    probability = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                calibration_slope * test_logits + calibration_bias, -30, 30
            )
        )
    )
    metrics = binary_classification_metrics(test_labels, probability, threshold)
    metrics["ece"] = expected_calibration_error(test_labels, probability, bins=10)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "1.0",
        "model": best_state,
        "config": asdict(config),
        "feature_columns": feature_columns,
        "race_categories": race_categories,
        "mean": mean,
        "scale": scale,
        "calibration_slope": calibration_slope,
        "calibration_bias": calibration_bias,
        "decision_threshold": threshold,
        "endpoint": config.label_column,
        "population": (
            "known_glaucoma" if config.known_glaucoma_only else "mixed_glaucoma_status"
        ),
        "warning": (
            "Harvard-GDP OCT/VF progression research only. The downloaded summary "
            "does not contain fundus photographs and is not a three-year endpoint."
        ),
    }
    torch.save(checkpoint, output / "best.pt")
    predictions = test_frame[["filename", config.label_column]].copy()
    predictions["probability"] = probability
    predictions.to_csv(output / "test_predictions.csv", index=False)
    report = {
        "status": "published_test_evaluated_once",
        "dataset": "Harvard-GDP",
        "endpoint": config.label_column,
        "population": checkpoint["population"],
        "development_patients": int(len(development_indices)),
        "calibration_patients": int(len(calibration_indices)),
        "test_patients": len(test_frame),
        "test_progressors": int(test_labels.sum()),
        "metrics": metrics,
        "bootstrap_95ci": _bootstrap(
            test_labels, probability, config.bootstrap_samples, config.seed
        ),
        "config": asdict(config),
        "warning": checkpoint["warning"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
