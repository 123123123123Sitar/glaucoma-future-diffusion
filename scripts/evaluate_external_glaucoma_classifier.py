#!/usr/bin/env python3
"""Evaluate a downloaded Hugging Face glaucoma classifier on PAPILA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_bootstrap_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patients: np.ndarray,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patients)
    estimates: list[float] = []
    patient_indices = {
        patient: np.flatnonzero(patients == patient) for patient in unique_patients
    }
    for _ in range(samples):
        draw = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([patient_indices[patient] for patient in draw])
        if np.unique(labels[indices]).size == 2:
            estimates.append(float(roc_auc_score(labels[indices], probabilities[indices])))
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(args.device)
    model.eval()

    manifest = pd.read_csv(args.manifest)
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(manifest), args.batch_size):
            paths = manifest.iloc[start : start + args.batch_size]["image_path"]
            images = [Image.open(path).convert("RGB") for path in paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            probabilities.extend(
                torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
            )

    predictions = manifest.copy()
    predictions["external_glaucoma_probability"] = probabilities
    predictions["external_predicted_label"] = (
        predictions["external_glaucoma_probability"] >= 0.5
    ).astype(int)

    evaluable = predictions[predictions["diagnosis"].isin(["healthy", "glaucoma"])]
    labels = evaluable["diagnosis_code"].to_numpy(dtype=int)
    scores = evaluable["external_glaucoma_probability"].to_numpy(dtype=float)
    predicted = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()

    report = {
        "status": "external_cross_dataset_evaluation_complete",
        "model_repository": "pamixsun/swinv2_tiny_for_glaucoma_classification",
        "pinned_revision": "a25a03d9ff23d6fbaf6cd7a329373253f4671c63",
        "license": "Apache-2.0",
        "weights_sha256": sha256(args.model_dir / "model.safetensors"),
        "evaluation_dataset": "PAPILA_v1.1",
        "evaluable_images": int(len(evaluable)),
        "evaluable_patients": int(evaluable["patient_id"].nunique()),
        "healthy_images": int((labels == 0).sum()),
        "glaucoma_images": int((labels == 1).sum()),
        "suspect_images_excluded_from_binary_metrics": int(
            (predictions["diagnosis"] == "suspect").sum()
        ),
        "metrics_at_documented_default_threshold_0p5": {
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
        "limitations": [
            "PAPILA is cross-sectional and cannot validate future progression.",
            "PAPILA may overlap conceptually with public fundus benchmarks, but the model card only documents REFUGE training.",
            "The downloaded model card does not provide numerical test metrics or confidence intervals.",
            "This is research evaluation, not clinical validation or regulatory clearance.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
