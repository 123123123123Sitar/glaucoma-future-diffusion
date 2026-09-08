#!/usr/bin/env python3
"""Train expert-supervised optic-disc geometry deformation."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.anatomy_deformation_trainer import (
    AnatomyDeformationConfig,
    train_anatomy_deformation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/annotations/grape_expert_anatomy_v1/manifest.csv")
    parser.add_argument("--output-dir", default="outputs/training/grape_anatomy_deformation_v1")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--max-displacement", type=float, default=18.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--evaluation-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = train_anatomy_deformation(AnatomyDeformationConfig(**vars(args)))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
