#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.risk_trainer import RiskTrainingConfig, train_risk_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the 512px single-fundus glaucoma-risk model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument(
        "--backbone",
        choices=[
            "compact",
            "convnext_tiny",
            "convnext_tiny_imagenet",
            "convnext_small",
            "convnext_small_imagenet",
        ],
        default="convnext_small_imagenet",
    )
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--current-loss-weight", type=float, default=0.25)
    parser.add_argument("--vcdr-loss-weight", type=float, default=0.10)
    parser.add_argument("--mask-loss-weight", type=float, default=0.20)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    args = parser.parse_args()
    summary = train_risk_model(RiskTrainingConfig(**vars(args)))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
