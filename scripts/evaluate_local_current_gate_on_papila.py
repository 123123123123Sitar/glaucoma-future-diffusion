#!/usr/bin/env python3
"""Fairly compare the locally trained current-glaucoma gate on PAPILA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

import _bootstrap  # noqa: F401
from evaluate_external_glaucoma_classifier import patient_bootstrap_auc
from glaucoma_forecast.data.highres import preprocess_high_resolution
from glaucoma_forecast.models.risk_model import build_dual_view_risk_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    import torch

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    model = build_dual_view_risk_model(
        years=int(config["years"]),
        feature_dim=int(config["feature_dim"]),
        backbone=str(config["backbone"]),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    manifest = pd.read_csv(args.manifest)
    evaluable = manifest[manifest["diagnosis"].isin(["healthy", "glaucoma"])].copy()
    probabilities: list[float] = []
    temperature = float(checkpoint.get("current_temperature", 1.0))
    threshold = float(checkpoint.get("current_threshold", 0.5))

    with torch.inference_mode():
        for path in evaluable["image_path"]:
            processed = preprocess_high_resolution(path, int(config["image_size"]))
            tensors = []
            for image in (processed.full_field, processed.optic_disc):
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
                tensors.append(tensor.unsqueeze(0).to(args.device))
            logit = model(tensors[0], tensors[1])["current_glaucoma_logit"]
            probabilities.append(float(torch.sigmoid(logit / temperature)[0].cpu()))

    labels = evaluable["diagnosis_code"].to_numpy(dtype=int)
    scores = np.asarray(probabilities)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    evaluable["local_glaucoma_probability"] = scores
    evaluable["local_predicted_label"] = predicted

    report = {
        "status": "external_cross_dataset_evaluation_complete",
        "model": "local_hygd_convnext_tiny_current_glaucoma_gate",
        "evaluation_dataset": "PAPILA_v1.1",
        "evaluable_images": int(len(evaluable)),
        "evaluable_patients": int(evaluable["patient_id"].nunique()),
        "original_hygd_calibrated_threshold": threshold,
        "metrics_without_papila_recalibration": {
            "auroc": float(roc_auc_score(labels, scores)),
            "patient_bootstrap_auroc_95ci": patient_bootstrap_auc(
                labels,
                scores,
                evaluable["patient_id"].astype(str).to_numpy(),
                args.bootstrap_samples,
                args.seed,
            ),
            "accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)),
            "brier": float(brier_score_loss(labels, scores)),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "warning": "External research evaluation only; no PAPILA threshold fitting was performed.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluable.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
