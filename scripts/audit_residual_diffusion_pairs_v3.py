#!/usr/bin/env python3
"""Audit the longitudinal support available to residual diffusion V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.residual_diffusion_trainer import (
    make_all_future_pairs,
    patient_split,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/grape/manifest.csv")
    parser.add_argument(
        "--output-dir", default="outputs/audit/grape_residual_diffusion_v3"
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--validation-patients", type=int, default=20)
    parser.add_argument("--test-patients", type=int, default=20)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    missing_images = [
        path for path in manifest["image_path"] if not Path(str(path)).is_file()
    ]
    pairs = make_all_future_pairs(manifest)
    train, validation, test = patient_split(
        pairs, args.seed, args.validation_patients, args.test_patients
    )
    patient_sets = {
        name: set(frame["patient_id"].astype(str))
        for name, frame in {
            "train": train,
            "validation": validation,
            "test": test,
        }.items()
    }
    overlap = {
        "train_validation": sorted(patient_sets["train"] & patient_sets["validation"]),
        "train_test": sorted(patient_sets["train"] & patient_sets["test"]),
        "validation_test": sorted(
            patient_sets["validation"] & patient_sets["test"]
        ),
    }
    bins = [0, 0.5, 1, 2, 3, 4, 5]
    histogram = pd.cut(
        pairs["horizon_years"], bins=bins, right=True, include_lowest=False
    ).value_counts(sort=False)
    report = {
        "manifest_rows": len(manifest),
        "manifest_patients": int(manifest["patient_id"].nunique()),
        "longitudinal_patients": int(pairs["patient_id"].nunique()),
        "all_real_earlier_later_pairs": len(pairs),
        "maximum_observed_horizon_years": float(pairs["horizon_years"].max()),
        "minimum_observed_horizon_years": float(pairs["horizon_years"].min()),
        "horizon_pair_counts": {
            str(interval): int(value) for interval, value in histogram.items()
        },
        "partitions": {
            "train": {
                "pairs": len(train),
                "patients": len(patient_sets["train"]),
            },
            "validation": {
                "pairs": len(validation),
                "patients": len(patient_sets["validation"]),
            },
            "locked_test": {
                "pairs": len(test),
                "patients": len(patient_sets["test"]),
            },
        },
        "patient_overlap": overlap,
        "missing_images": missing_images,
        "training_safe": not missing_images
        and not any(overlap.values())
        and not pairs.empty,
        "years_5_to_10_supported": False,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["training_safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
