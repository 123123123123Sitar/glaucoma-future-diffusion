#!/usr/bin/env python3
"""Evaluate sampled residual trajectories against a locked patient test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from glaucoma_forecast.evaluation.image_metrics import psnr, simple_ssim
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.residual_diffusion import (
    build_residual_diffusion,
    sample_residual_trajectory,
)
from glaucoma_forecast.training.residual_diffusion_trainer import (
    RegisteredResidualPairDataset,
    _collate,
)
from glaucoma_forecast.utils.reproducibility import select_device


def horizon_band(years: float) -> str:
    if years <= 1.5:
        return "approximately_1_year"
    if years <= 3.5:
        return "approximately_3_years"
    return "approximately_5_years"


def bootstrap_patient_mae_difference(rows, seed: int, samples: int = 2000):
    by_patient = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], []).append(
            row["model_mae"] - row["persistence_mae"]
        )
    patient_differences = np.asarray(
        [np.mean(values) for values in by_patient.values()], dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            np.mean(generator.choice(patient_differences, len(patient_differences), replace=True))
            for _ in range(samples)
        ]
    )
    return [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]


def binary_auc(rows) -> float | None:
    usable = [
        row for row in rows if row.get("progression_probability") is not None
    ]
    positives = [row for row in usable if row["future_glaucoma_label"] == 1]
    negatives = [row for row in usable if row["future_glaucoma_label"] == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0
        if positive["progression_probability"] > negative["progression_probability"]
        else 0.5
        if positive["progression_probability"] == negative["progression_probability"]
        else 0.0
        for positive in positives
        for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def main() -> int:
    torch = require_torch()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=25)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--change-scale", type=float, default=1.0)
    args = parser.parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    clinical_dim = 1 if config.get("conditioning_manifest") else 0
    structural_feature_mode = str(config.get("structural_feature_mode", "vessel"))
    structural_channels = 3 if structural_feature_mode == "glaucoma" else 1
    model = build_residual_diffusion(
        int(config["base_channels"]),
        clinical_dim,
        structural_channels=structural_channels,
        progression_head=float(config.get("progression_loss_weight", 0.0)) > 0,
    ).to(device)
    model.load_state_dict(checkpoint.get("best_state") or checkpoint["model"])
    pairs = __import__("pandas").read_csv(args.pairs)
    if args.max_pairs:
        pairs = pairs.iloc[: args.max_pairs].copy()
    residual_size = int(config["residual_size"])
    max_change = float(config["max_change"])
    max_horizon = float(config["max_supported_horizon"])
    dataset = RegisteredResidualPairDataset(
        pairs,
        int(config["image_size"]),
        residual_size,
        max_change,
        max_horizon,
        str(config.get("view_mode", "full_field")),
        float(config.get("disc_crop_fraction", 0.50)),
        clinical_dim,
        structural_feature_mode,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate
    )
    rows = []
    for index, batch in enumerate(loader):
        baseline = batch["baseline_low"].to(device)
        vessel = batch["vessel_low"].to(device)
        target_residual = batch["target_residual"].to(device)
        horizon = float(batch["horizon"][0]) * max_horizon
        sampled = sample_residual_trajectory(
            model,
            baseline,
            baseline,
            vessel,
            [horizon],
            max_horizon,
            max_change,
            trajectories=args.trajectories,
            diffusion_steps=args.sampling_steps,
            seed=int(config["seed"]) + index,
            change_scale=args.change_scale,
            clinical_condition=batch["clinical_condition"].to(device),
        )[0, :, 0]
        prediction = sampled.mean(dim=0)
        progression_probability = None
        if getattr(model, "progression_head_enabled", False):
            with torch.no_grad():
                progression_probability = float(
                    torch.sigmoid(
                        model.predict_progression(
                            baseline,
                            vessel,
                            batch["horizon"].to(device),
                        )
                    )[0].cpu()
                )
        target = (baseline[0] + target_residual[0] * max_change).clamp(0, 1)
        persistence = baseline[0]
        prediction_array = (
            prediction.permute(1, 2, 0).detach().cpu().numpy() * 255
        )
        target_array = target.permute(1, 2, 0).detach().cpu().numpy() * 255
        persistence_array = (
            persistence.permute(1, 2, 0).detach().cpu().numpy() * 255
        )
        rows.append(
            {
                "patient_id": batch["patient_id"][0],
                "eye_id": batch["eye_id"][0],
                "cohort": str(
                    pairs.iloc[index].get("cohort", "longitudinal_unspecified")
                ),
                "diagnosis": str(pairs.iloc[index].get("diagnosis", "unspecified")),
                "target_type": str(
                    pairs.iloc[index].get("target_type", "observed_longitudinal_change")
                ),
                "baseline_glaucoma_label": int(
                    pairs.iloc[index].get("baseline_glaucoma_label", -1)
                ),
                "future_glaucoma_label": int(
                    pairs.iloc[index].get("future_glaucoma_label", -1)
                ),
                "transition": (
                    f"{int(pairs.iloc[index].get('baseline_glaucoma_label', -1))}"
                    f"->{int(pairs.iloc[index].get('future_glaucoma_label', -1))}"
                ),
                "horizon_years": horizon,
                "horizon_band": horizon_band(horizon),
                "progression_probability": progression_probability,
                "model_mae": float(np.mean(np.abs(prediction_array - target_array)) / 255),
                "persistence_mae": float(
                    np.mean(np.abs(persistence_array - target_array)) / 255
                ),
                "model_psnr": psnr(target_array, prediction_array),
                "persistence_psnr": psnr(target_array, persistence_array),
                "model_ssim": simple_ssim(target_array, prediction_array),
                "persistence_ssim": simple_ssim(target_array, persistence_array),
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_pair.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def aggregate(selected):
        if not selected:
            return {"pairs": 0, "patients": 0}
        model_mae = float(np.mean([row["model_mae"] for row in selected]))
        persistence_mae = float(
            np.mean([row["persistence_mae"] for row in selected])
        )
        return {
            "pairs": len(selected),
            "patients": len({row["patient_id"] for row in selected}),
            "model_mae": model_mae,
            "persistence_mae": persistence_mae,
            "model_minus_persistence_mae": float(
                np.mean(
                    [
                        row["model_mae"] - row["persistence_mae"]
                        for row in selected
                    ]
                )
            ),
            "mae_reduction_percent": float(
                100.0 * (persistence_mae - model_mae) / max(persistence_mae, 1e-12)
            ),
            "model_ssim": float(np.mean([row["model_ssim"] for row in selected])),
            "persistence_ssim": float(
                np.mean([row["persistence_ssim"] for row in selected])
            ),
        }

    report = {
        "checkpoint": args.checkpoint,
        "locked_pairs_evaluated": len(rows),
        "patient_count": len({row["patient_id"] for row in rows}),
        "model_mae": float(np.mean([row["model_mae"] for row in rows])),
        "persistence_mae": float(
            np.mean([row["persistence_mae"] for row in rows])
        ),
        "model_psnr": float(np.mean([row["model_psnr"] for row in rows])),
        "persistence_psnr": float(
            np.mean([row["persistence_psnr"] for row in rows])
        ),
        "model_ssim": float(np.mean([row["model_ssim"] for row in rows])),
        "persistence_ssim": float(
            np.mean([row["persistence_ssim"] for row in rows])
        ),
        "beats_persistence_mae": bool(
            np.mean([row["model_mae"] for row in rows])
            < np.mean([row["persistence_mae"] for row in rows])
        ),
        "future_glaucoma_auc": binary_auc(rows),
        "patient_bootstrap_95_ci_model_minus_persistence_mae": (
            bootstrap_patient_mae_difference(rows, int(config["seed"]))
        ),
        "recursive_rollout": False,
        "view_mode": str(config.get("view_mode", "full_field")),
        "validation_fitted_change_scale": args.change_scale,
        "stratified": {
            cohort: aggregate([row for row in rows if row["cohort"] == cohort])
            for cohort in sorted({row["cohort"] for row in rows})
        },
        "by_transition": {
            transition: aggregate(
                [row for row in rows if row["transition"] == transition]
            )
            for transition in sorted({row["transition"] for row in rows})
        },
        "by_horizon": {
            band: aggregate([row for row in rows if row["horizon_band"] == band])
            for band in (
                "approximately_1_year",
                "approximately_3_years",
                "approximately_5_years",
            )
        },
        "warning": "Research-only simulation; no clinical validity.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
