"""Exploratory Harvard-GDP RNFL thickness-map progression benchmark.

The publisher's fixed test split has already been examined by this project, so
this benchmark is explicitly post-hoc.  It is useful for deciding whether OCT
adds signal, but it is not independent confirmatory validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from glaucoma_forecast.evaluation.calibration import expected_calibration_error
from glaucoma_forecast.evaluation.progression_metrics import (
    binary_classification_metrics,
)
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.training.harvard_gdp_trainer import (
    _bootstrap,
    _feature_columns,
    _matrix,
    _platt_parameters,
    _stratified_calibration_indices,
    _threshold,
)
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


@dataclass(frozen=True)
class HarvardGDPOCTConfig:
    manifest: str
    rnflt_dir: str
    output_dir: str
    label_column: str = "progression.td_pointwise_no_p_cut"
    variant: str = "multimodal"
    epochs: int = 120
    batch_size: int = 16
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 3e-3
    calibration_fraction: float = 0.20
    patience: int = 20
    bootstrap_samples: int = 2000
    device: str = "auto"
    seed: int = 20260730
    known_glaucoma_only: bool = True


def _load_rnflt(frame: pd.DataFrame, directory: str) -> np.ndarray:
    maps = []
    root = Path(directory)
    for filename in frame["filename"].astype(str):
        path = root / f"{filename}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing RNFLT map: {path}")
        with np.load(path) as archive:
            value = np.asarray(archive["rnflt"], dtype=np.float32)
        if value.shape != (225, 225):
            raise ValueError(f"Unexpected RNFLT shape {value.shape} in {path}")
        # Preserve the full central 224x224 field; -2 is the publisher's
        # outside-scan sentinel and becomes a separate validity channel.
        maps.append(value[:224, :224])
    return np.stack(maps)


def _normalise_rnflt(
    train: np.ndarray, test: np.ndarray, development_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    valid = train[development_indices] >= 0
    values = train[development_indices][valid]
    mean = float(values.mean())
    scale = float(max(values.std(), 1e-6))

    def transform(array: np.ndarray) -> np.ndarray:
        mask = array >= 0
        normalised = np.where(mask, (array - mean) / scale, 0.0)
        normalised = np.clip(normalised, -5.0, 5.0)
        return np.stack([normalised, mask.astype(np.float32)], axis=1).astype(
            np.float32
        )

    return transform(train), transform(test), mean, scale


def train_harvard_gdp_oct(
    config: HarvardGDPOCTConfig,
) -> dict[str, object]:
    if config.variant not in {"rnflt_only", "multimodal"}:
        raise ValueError("variant must be rnflt_only or multimodal")
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
        "glaucoma",
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
    train_labels = train_frame[config.label_column].to_numpy(dtype=np.float32)
    test_labels = test_frame[config.label_column].to_numpy(dtype=int)
    development_indices, calibration_indices = _stratified_calibration_indices(
        train_labels.astype(int), config.calibration_fraction, config.seed
    )

    feature_columns = _feature_columns(frame)
    race_categories = sorted(
        train_frame["race"].astype(str).str.lower().unique().tolist()
    )
    train_clinical = _matrix(train_frame, feature_columns, race_categories)
    test_clinical = _matrix(test_frame, feature_columns, race_categories)
    clinical_mean = train_clinical[development_indices].mean(axis=0)
    clinical_scale = np.maximum(
        train_clinical[development_indices].std(axis=0), 1e-6
    )
    train_clinical = (train_clinical - clinical_mean) / clinical_scale
    test_clinical = (test_clinical - clinical_mean) / clinical_scale

    train_maps, test_maps, rnflt_mean, rnflt_scale = _normalise_rnflt(
        _load_rnflt(train_frame, config.rnflt_dir),
        _load_rnflt(test_frame, config.rnflt_dir),
        development_indices,
    )

    class OCTProgressionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image = torch.nn.Sequential(
                torch.nn.Conv2d(2, 12, 5, stride=2, padding=2),
                torch.nn.BatchNorm2d(12),
                torch.nn.GELU(),
                torch.nn.Conv2d(12, 24, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(24),
                torch.nn.GELU(),
                torch.nn.Conv2d(24, 48, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(48),
                torch.nn.GELU(),
                torch.nn.Conv2d(48, 64, 3, stride=2, padding=1),
                torch.nn.GELU(),
                # The 14x14 activation divides cleanly into 2x2.  This also
                # avoids an unsupported non-divisible adaptive pool on MPS.
                torch.nn.AdaptiveAvgPool2d((2, 2)),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 2 * 2, config.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.30),
            )
            if config.variant == "multimodal":
                self.clinical = torch.nn.Sequential(
                    torch.nn.Linear(train_clinical.shape[1], config.hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.25),
                )
                output_dim = config.hidden_dim * 2
            else:
                self.clinical = None
                output_dim = config.hidden_dim
            self.output = torch.nn.Linear(output_dim, 1)

        def forward(self, image, clinical):
            embedding = self.image(image)
            if self.clinical is not None:
                embedding = torch.cat([embedding, self.clinical(clinical)], dim=1)
            return self.output(embedding).squeeze(1)

    model = OCTProgressionModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    development_labels = train_labels[development_indices]
    positive = float(development_labels.sum())
    negative = float(len(development_labels) - positive)
    pos_weight = torch.tensor(
        [negative / max(positive, 1.0)], dtype=torch.float32, device=device
    )
    generator = torch.Generator().manual_seed(config.seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_maps[development_indices]),
        torch.from_numpy(train_clinical[development_indices]).float(),
        torch.from_numpy(development_labels).float(),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    calibration_image = torch.from_numpy(train_maps[calibration_indices]).to(device)
    calibration_clinical = (
        torch.from_numpy(train_clinical[calibration_indices]).float().to(device)
    )
    calibration_y = train_labels[calibration_indices].astype(int)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    stale = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for image, clinical, label in loader:
            logits = model(image.to(device), clinical.to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, label.to(device), pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            calibration_logits = model(
                calibration_image, calibration_clinical
            ).cpu().numpy()
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
                "development_loss": float(np.mean(losses)),
                "calibration_loss": calibration_loss,
            }
        )
        if calibration_loss < best_loss - 1e-5:
            best_loss = calibration_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("No OCT checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        calibration_logits = model(
            calibration_image, calibration_clinical
        ).cpu().numpy()
        test_logits = model(
            torch.from_numpy(test_maps).to(device),
            torch.from_numpy(test_clinical).float().to(device),
        ).cpu().numpy()
    slope, bias = _platt_parameters(calibration_logits, calibration_y)
    calibration_probability = 1.0 / (
        1.0 + np.exp(-np.clip(slope * calibration_logits + bias, -30, 30))
    )
    threshold = _threshold(calibration_y, calibration_probability)
    probability = 1.0 / (
        1.0 + np.exp(-np.clip(slope * test_logits + bias, -30, 30))
    )
    metrics = binary_classification_metrics(test_labels, probability, threshold)
    metrics["ece"] = expected_calibration_error(test_labels, probability, bins=10)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warning = (
        "Exploratory post-hoc analysis: this project's investigators had already "
        "seen results on the publisher test split before this OCT model was built. "
        "This is not independent confirmation and is not a three-year fundus model."
    )
    checkpoint = {
        "schema_version": "1.0",
        "model": best_state,
        "config": asdict(config),
        "feature_columns": feature_columns,
        "race_categories": race_categories,
        "clinical_mean": clinical_mean,
        "clinical_scale": clinical_scale,
        "rnflt_mean": rnflt_mean,
        "rnflt_scale": rnflt_scale,
        "calibration_slope": slope,
        "calibration_bias": bias,
        "decision_threshold": threshold,
        "best_epoch": best_epoch,
        "warning": warning,
    }
    torch.save(checkpoint, output / "best.pt")
    predictions = test_frame[["filename", config.label_column]].copy()
    predictions["probability"] = probability
    predictions.to_csv(output / "test_predictions.csv", index=False)
    report = {
        "status": "exploratory_posthoc_test_evaluation",
        "dataset": "Harvard-GDP",
        "modality": (
            "RNFL thickness map plus VF/demographics"
            if config.variant == "multimodal"
            else "RNFL thickness map"
        ),
        "endpoint": config.label_column,
        "population": (
            "known_glaucoma" if config.known_glaucoma_only else "mixed_glaucoma_status"
        ),
        "development_patients": int(len(development_indices)),
        "calibration_patients": int(len(calibration_indices)),
        "test_patients": int(len(test_frame)),
        "test_progressors": int(test_labels.sum()),
        "best_epoch": best_epoch,
        "metrics": metrics,
        "bootstrap_95ci": _bootstrap(
            test_labels, probability, config.bootstrap_samples, config.seed
        ),
        "config": asdict(config),
        "warning": warning,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "history.json").write_text(json.dumps(history, indent=2))
    return report
