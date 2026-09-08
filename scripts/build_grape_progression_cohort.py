#!/usr/bin/env python3
"""Build the baseline-only GRAPE progression cohort and frozen patient folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.grape_progression import (
    build_grape_progression_cohort,
    patient_stratified_folds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/grape/manifest.csv")
    parser.add_argument("--metadata", default="data/grape/metadata_extra.csv")
    parser.add_argument(
        "--output", default="data/grape/progression_baseline_cohort_v1.csv"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    cohort = build_grape_progression_cohort(
        pd.read_csv(args.manifest), pd.read_csv(args.metadata)
    )
    cohort["outer_fold"] = patient_stratified_folds(
        cohort, folds=args.folds, seed=args.seed
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    report = {
        "schema_version": "1.0",
        "endpoint": "documented_progression_during_observed_study_followup",
        "primary_label": "progression_plr2",
        "fixed_horizon_claim": False,
        "eyes": len(cohort),
        "patients": int(cohort["patient_id"].nunique()),
        "plr2_positive_eyes": int(cohort["progression_plr2"].sum()),
        "plr3_positive_eyes": int(cohort["progression_plr3"].sum()),
        "md_positive_eyes": int(cohort["progression_md"].sum()),
        "maximum_observed_cfp_followup_years": float(
            cohort["observed_cfp_followup_years"].max()
        ),
        "folds": {
            str(fold): {
                "eyes": int(len(group)),
                "patients": int(group["patient_id"].nunique()),
                "plr2_positive_eyes": int(group["progression_plr2"].sum()),
            }
            for fold, group in cohort.groupby("outer_fold")
        },
        "warning": (
            "Eye-level progression labels do not provide event dates. This cohort "
            "supports progression susceptibility, not fixed-horizon survival risk."
        ),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
