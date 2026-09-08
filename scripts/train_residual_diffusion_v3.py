#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.residual_diffusion_trainer import (
    ResidualTrainingConfig,
    train_registered_residual_diffusion,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train direct registered residual diffusion on GRAPE."
    )
    parser.add_argument("--manifest", default="data/grape/manifest.csv")
    parser.add_argument(
        "--pair-manifest",
        help=(
            "Optional pre-split pair manifest with train/validation/test labels. "
            "Use this to preserve a frozen patient split."
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs/training/grape_residual_diffusion_v3"
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--residual-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--max-change", type=float, default=0.18)
    parser.add_argument("--max-supported-horizon", type=float, default=4.84)
    parser.add_argument("--validation-patients", type=int, default=20)
    parser.add_argument("--test-patients", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from")
    parser.add_argument(
        "--view-mode",
        choices=["full_field", "optic_disc"],
        default="full_field",
    )
    parser.add_argument("--disc-crop-fraction", type=float, default=0.50)
    parser.add_argument("--conditioning-manifest")
    parser.add_argument(
        "--conditioning-column", default="progression_probability_3y"
    )
    parser.add_argument("--residual-loss-weight", type=float, default=0.75)
    parser.add_argument("--edge-loss-weight", type=float, default=0.2)
    parser.add_argument("--identity-loss-weight", type=float, default=0.1)
    parser.add_argument("--anatomy-loss-weight", type=float, default=0.0)
    parser.add_argument("--anatomy-sector-emphasis", type=float, default=3.0)
    parser.add_argument("--anatomy-inner-radius", type=float, default=0.06)
    parser.add_argument("--anatomy-outer-radius", type=float, default=0.30)
    parser.add_argument("--segmenter-checkpoint")
    parser.add_argument("--disc-cup-loss-weight", type=float, default=0.75)
    parser.add_argument("--cup-ratio-loss-weight", type=float, default=1.5)
    parser.add_argument("--annotated-anatomy-loss-weight", type=float, default=2.0)
    parser.add_argument("--outer-disc-preservation-weight", type=float, default=1.0)
    parser.add_argument("--stability-loss-weight", type=float, default=1.0)
    parser.add_argument("--minimum-segmenter-disc-dice", type=float, default=0.90)
    parser.add_argument("--minimum-segmenter-cup-dice", type=float, default=0.75)
    parser.add_argument("--allow-pseudo-anatomy", action="store_true")
    parser.add_argument(
        "--structural-feature-mode",
        choices=["vessel", "glaucoma"],
        default="vessel",
    )
    parser.add_argument("--transition-balance-power", type=float, default=0.0)
    parser.add_argument("--structural-focus-loss-weight", type=float, default=0.0)
    parser.add_argument("--superior-inferior-emphasis", type=float, default=2.0)
    parser.add_argument("--feature-reconstruction-loss-weight", type=float, default=0.0)
    parser.add_argument("--progression-loss-weight", type=float, default=0.0)
    args = parser.parse_args()
    print(train_registered_residual_diffusion(ResidualTrainingConfig(**vars(args))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
