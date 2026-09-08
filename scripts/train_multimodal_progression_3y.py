#!/usr/bin/env python3
"""Train nested-CV three-year glaucoma progression models."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.multimodal_progression_trainer import (
    MultimodalProgressionConfig,
    train_multimodal_progression_cv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/grape/progression_3y_multimodal_v1.csv"
    )
    parser.add_argument(
        "--output-dir", default="outputs/training/progression_3y_multimodal"
    )
    parser.add_argument(
        "--variant",
        choices=["clinical_only", "image_only", "multimodal"],
        default="multimodal",
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--train-image-encoder",
        dest="freeze_image_encoder",
        action="store_false",
    )
    parser.set_defaults(freeze_image_encoder=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--initialize-retinal-encoder-from")
    args = parser.parse_args()
    report = train_multimodal_progression_cv(
        MultimodalProgressionConfig(**vars(args))
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
