#!/usr/bin/env python3
"""Train patient-grouped cross-validated GRAPE progression models."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.progression_trainer import (
    ProgressionCVConfig,
    train_progression_cross_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/grape/progression_baseline_cohort_v1.csv"
    )
    parser.add_argument(
        "--output-dir", default="outputs/training/grape_progression_cv_v1"
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--backbone",
        choices=[
            "compact",
            "convnext_tiny",
            "convnext_tiny_imagenet",
            "convnext_small",
            "convnext_small_imagenet",
        ],
        default="convnext_tiny_imagenet",
    )
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--label-column", default="primary_progression_label")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--initialize-retinal-encoder-from")
    args = parser.parse_args()
    train_progression_cross_validation(ProgressionCVConfig(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
