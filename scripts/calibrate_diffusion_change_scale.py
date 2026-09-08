#!/usr/bin/env python3
"""Fit one validation-only shrinkage factor for sampled diffusion change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
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


def main() -> int:
    torch = require_torch()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectories", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=15)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--device", default="auto")
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
    max_change = float(config["max_change"])
    max_horizon = float(config["max_supported_horizon"])
    residual_size = int(config["residual_size"])
    pairs = pd.read_csv(args.pairs)
    if args.max_pairs:
        pairs = pairs.iloc[: args.max_pairs].copy()
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
    numerator = denominator = target_energy = 0.0
    for index, batch in enumerate(loader):
        baseline = batch["baseline_low"].to(device)
        vessel = batch["vessel_low"].to(device)
        target_change = batch["target_residual"].to(device) * max_change
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
            clinical_condition=batch["clinical_condition"].to(device),
        )[0, :, 0].mean(dim=0)
        predicted_change = sampled - baseline[0]
        numerator += float((predicted_change * target_change[0]).sum().cpu())
        denominator += float(predicted_change.square().sum().cpu())
        target_energy += float(target_change.square().sum().cpu())
    unbounded = numerator / max(denominator, 1e-12)
    scale = max(0.0, min(1.0, unbounded))
    report = {
        "checkpoint": args.checkpoint,
        "validation_pairs": len(dataset),
        "change_scale": scale,
        "unbounded_least_squares_scale": unbounded,
        "predicted_target_change_correlation_numerator": numerator,
        "predicted_change_energy": denominator,
        "target_change_energy": target_energy,
        "trajectories": args.trajectories,
        "sampling_steps": args.sampling_steps,
        "fit_partition": "validation_only",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
