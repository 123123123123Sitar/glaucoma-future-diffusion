#!/usr/bin/env python3
"""Fit scalar temperature calibration for a held-out current-glaucoma gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from glaucoma_forecast.evaluation.calibration import expected_calibration_error
from glaucoma_forecast.evaluation.survival_metrics import binary_auc
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.risk_model import build_dual_view_risk_model
from glaucoma_forecast.training.risk_trainer import RiskCohortDataset, _collate
from glaucoma_forecast.utils.reproducibility import select_device


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def _bce(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probability = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(
        -np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch = require_torch()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = build_dual_view_risk_model(
        int(config.get("years", 10)),
        int(config.get("feature_dim", 256)),
        str(config.get("backbone", "compact")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    import pandas as pd

    frame = pd.read_csv(args.validation_manifest)
    loader = torch.utils.data.DataLoader(
        RiskCohortDataset(frame, int(config.get("image_size", 512))),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    logits = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["full_field"].to(device), batch["optic_disc"].to(device))
            available = torch.isfinite(batch["current_label"])
            logits.extend(output["current_glaucoma_logit"][available.to(device)].cpu().tolist())
            labels.extend(batch["current_label"][available].cpu().tolist())
    logits_array = np.asarray(logits, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    temperatures = np.geomspace(0.05, 10.0, 1000)
    losses = np.asarray(
        [_bce(labels_array, _sigmoid(logits_array / value)) for value in temperatures]
    )
    temperature = float(temperatures[int(np.argmin(losses))])
    raw = _sigmoid(logits_array)
    calibrated = _sigmoid(logits_array / temperature)
    candidates = np.unique(calibrated)
    operating_points = []
    for threshold in candidates:
        predicted = calibrated >= threshold
        sensitivity = float(predicted[labels_array == 1].mean())
        specificity = float((~predicted[labels_array == 0]).mean())
        if sensitivity >= 0.90:
            operating_points.append((specificity, float(threshold), sensitivity))
    specificity, current_threshold, sensitivity = max(
        operating_points, default=(0.0, 0.5, 0.0)
    )
    report = {
        "checkpoint": args.checkpoint,
        "validation_images": len(labels_array),
        "validation_patients": int(frame["patient_id"].nunique()),
        "temperature": temperature,
        "auc": binary_auc(labels_array, calibrated),
        "raw_bce": _bce(labels_array, raw),
        "calibrated_bce": _bce(labels_array, calibrated),
        "raw_brier": float(np.mean((raw - labels_array) ** 2)),
        "calibrated_brier": float(np.mean((calibrated - labels_array) ** 2)),
        "raw_ece": expected_calibration_error(labels_array, raw),
        "calibrated_ece": expected_calibration_error(labels_array, calibrated),
        "current_threshold": current_threshold,
        "sensitivity_at_threshold": sensitivity,
        "specificity_at_threshold": specificity,
        "threshold_policy": "maximum held-out specificity subject to sensitivity >= 0.90",
        "scope": "current_glaucoma_gate_only_not_incident_risk",
    }
    checkpoint["current_temperature"] = temperature
    checkpoint["current_threshold"] = current_threshold
    checkpoint["calibration"] = report
    checkpoint["model_version"] = (
        str(checkpoint.get("model_version", "risk-v2")) + "-current-calibrated"
    )
    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
