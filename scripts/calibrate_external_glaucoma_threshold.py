#!/usr/bin/env python3
"""Fit a screening threshold on one patient fold and evaluate on the remainder."""

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
from sklearn.model_selection import StratifiedGroupKFold

from evaluate_external_glaucoma_classifier import patient_bootstrap_auc


def operating_point(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    frame = frame[frame["diagnosis"].isin(["healthy", "glaucoma"])].reset_index(drop=True)
    labels = frame["diagnosis_code"].to_numpy(dtype=int)
    scores = frame["external_glaucoma_probability"].to_numpy(dtype=float)
    groups = frame["patient_id"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    test_indices, calibration_indices = next(splitter.split(frame, labels, groups))
    calibration_labels = labels[calibration_indices]
    calibration_scores = scores[calibration_indices]

    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], calibration_scores)))
    feasible = []
    for threshold in candidates:
        point = operating_point(calibration_labels, calibration_scores, float(threshold))
        if point["sensitivity"] >= args.target_sensitivity:
            feasible.append(point)
    selected = max(feasible, key=lambda point: (point["specificity"], point["threshold"]))

    test_labels = labels[test_indices]
    test_scores = scores[test_indices]
    test_point = operating_point(test_labels, test_scores, selected["threshold"])
    report = {
        "status": "patient_separated_threshold_calibration_complete",
        "model_repository": "pamixsun/swinv2_tiny_for_glaucoma_classification",
        "weights_changed": False,
        "threshold_policy": (
            "maximum calibration-fold specificity subject to sensitivity "
            f">= {args.target_sensitivity:.2f}"
        ),
        "calibration": {
            "patients": int(frame.iloc[calibration_indices]["patient_id"].nunique()),
            "images": int(len(calibration_indices)),
            **selected,
        },
        "held_out_test": {
            "patients": int(frame.iloc[test_indices]["patient_id"].nunique()),
            "images": int(len(test_indices)),
            "auroc": float(roc_auc_score(test_labels, test_scores)),
            "patient_bootstrap_auroc_95ci": patient_bootstrap_auc(
                test_labels,
                test_scores,
                groups[test_indices],
                5000,
                args.seed,
            ),
            "brier": float(brier_score_loss(test_labels, test_scores)),
            **test_point,
        },
        "limitations": [
            "PAPILA provides cross-sectional diagnosis, not longitudinal progression.",
            "This threshold is PAPILA-specific and needs independent multi-camera validation.",
            "The model is a research screening component, not a diagnostic medical device.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
