#!/usr/bin/env python3
"""Evaluate a registered GRAPE checkpoint against last-observation persistence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from glaucoma_forecast.evaluation.image_metrics import psnr, simple_ssim
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.training.grape_glaucoma_trainer import (
    DISCLAIMER,
    RegisteredDiscPairDataset,
    build_compact_denoiser,
)
from glaucoma_forecast.utils.reproducibility import select_device


def _edge_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = reference.astype(np.float32) / 255.0
    pred = prediction.astype(np.float32) / 255.0
    return float(
        np.mean(np.abs(np.diff(ref, axis=1) - np.diff(pred, axis=1)))
        + np.mean(np.abs(np.diff(ref, axis=0) - np.diff(pred, axis=0)))
    )


def _as_uint8(tensor) -> np.ndarray:
    return (np.transpose(tensor.detach().cpu().clamp(0, 1).numpy(), (1, 2, 0)) * 255).astype(np.uint8)


def _bootstrap_patient_ci(frame: pd.DataFrame, column: str, seed: int, iterations: int = 1000) -> list[float]:
    rng = np.random.default_rng(seed)
    patient_means = frame.groupby("patient_id")[column].mean().to_numpy(dtype=float)
    if patient_means.size < 2:
        value = float(patient_means.mean())
        return [value, value]
    samples = [float(rng.choice(patient_means, size=len(patient_means), replace=True).mean()) for _ in range(iterations)]
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="Held-out pair CSV produced by training.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    torch = require_torch()
    device = select_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = build_compact_denoiser(int(config.get("base_channels", 32))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pairs = pd.read_csv(args.pairs)
    dataset = RegisteredDiscPairDataset(
        pairs,
        image_size=int(config.get("image_size", 128)),
        crop_fraction=float(config.get("crop_fraction", 0.5)),
    )
    rows: list[dict[str, object]] = []
    preview_dir = out / "previews"
    preview_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            context = item["context"].unsqueeze(0).to(device)
            target = item["target"].unsqueeze(0).to(device)
            horizon = torch.tensor([min(max(float(item["horizon"]), 0.0), 5.0) / 5.0], device=device)
            prediction = model.predict_future(context, horizon, float(config.get("max_change", 0.18)))[0]
            context_np = _as_uint8(context[0])
            target_np = _as_uint8(target[0])
            prediction_np = _as_uint8(prediction)
            row = pairs.iloc[index]
            metrics = {
                "patient_id": str(row["patient_id"]),
                "eye_id": str(row["eye_id"]),
                "context_visit_id": str(row["context_visit_id"]),
                "target_visit_id": str(row["target_visit_id"]),
                "forecast_horizon": float(row["forecast_horizon"]),
                "registration_confidence": float(item["registration_confidence"]),
                "model_mae": float(np.mean(np.abs(prediction_np.astype(float) - target_np.astype(float))) / 255.0),
                "persistence_mae": float(np.mean(np.abs(context_np.astype(float) - target_np.astype(float))) / 255.0),
                "model_psnr": psnr(target_np, prediction_np),
                "persistence_psnr": psnr(target_np, context_np),
                "model_ssim": simple_ssim(target_np, prediction_np),
                "persistence_ssim": simple_ssim(target_np, context_np),
                "model_edge_error": _edge_error(target_np, prediction_np),
                "persistence_edge_error": _edge_error(target_np, context_np),
            }
            metrics["mae_improvement"] = metrics["persistence_mae"] - metrics["model_mae"]
            metrics["psnr_improvement"] = metrics["model_psnr"] - metrics["persistence_psnr"]
            metrics["ssim_improvement"] = metrics["model_ssim"] - metrics["persistence_ssim"]
            metrics["edge_improvement"] = metrics["persistence_edge_error"] - metrics["model_edge_error"]
            rows.append(metrics)
            if index < 12:
                Image.fromarray(np.concatenate([context_np, target_np, prediction_np], axis=1)).save(preview_dir / f"pair_{index:03d}.png")

    results = pd.DataFrame(rows)
    results.to_csv(out / "per_pair_metrics.csv", index=False)
    differences = ["mae_improvement", "psnr_improvement", "ssim_improvement", "edge_improvement"]
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(checkpoint["step"]),
        "held_out_patients": int(results["patient_id"].nunique()),
        "held_out_pairs": len(results),
        "mean_metrics": {column: float(results[column].mean()) for column in results.columns if column not in {"patient_id", "eye_id", "context_visit_id", "target_visit_id"}},
        "patient_bootstrap_95_ci": {column: _bootstrap_patient_ci(results, column, args.seed) for column in differences},
        "beats_persistence": {
            "mae": bool(results["mae_improvement"].mean() > 0),
            "psnr": bool(results["psnr_improvement"].mean() > 0),
            "ssim": bool(results["ssim_improvement"].mean() > 0),
            "edge_error": bool(results["edge_improvement"].mean() > 0),
        },
        "disclaimer": DISCLAIMER,
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = [
        "# Held-out persistence comparison",
        "",
        DISCLAIMER,
        "",
        f"- Checkpoint step: {summary['checkpoint_step']}",
        f"- Held-out patients: {summary['held_out_patients']}",
        f"- Held-out pairs: {summary['held_out_pairs']}",
        f"- Mean MAE improvement: {summary['mean_metrics']['mae_improvement']:.6f}",
        f"- Mean PSNR improvement: {summary['mean_metrics']['psnr_improvement']:.4f} dB",
        f"- Mean SSIM improvement: {summary['mean_metrics']['ssim_improvement']:.6f}",
        f"- Mean edge-error improvement: {summary['mean_metrics']['edge_improvement']:.6f}",
        "",
        "Positive improvement favors the model. Small gains do not establish clinical utility.",
    ]
    (out / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
