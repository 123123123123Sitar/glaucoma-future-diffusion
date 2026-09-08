#!/usr/bin/env python3
"""Train the fixed-split Harvard-GDP VF/demographic benchmark."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.harvard_gdp_trainer import (
    HarvardGDPConfig,
    train_harvard_gdp_progression,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/external/harvard_gdp/data_summary.csv"
    )
    parser.add_argument(
        "--output-dir", default="outputs/training/harvard_gdp_progression"
    )
    parser.add_argument("--label-column", default="progression.md")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--include-non-glaucoma",
        dest="known_glaucoma_only",
        action="store_false",
    )
    parser.set_defaults(known_glaucoma_only=True)
    args = parser.parse_args()
    report = train_harvard_gdp_progression(HarvardGDPConfig(**vars(args)))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
