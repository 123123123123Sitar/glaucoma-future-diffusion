#!/usr/bin/env python3
"""Build the frozen GRAPE 2.5--3.5-year multimodal progression cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.grape_multimodal import (
    build_three_year_progression_cohort,
    load_grape_examinations,
)
from glaucoma_forecast.data.grape_progression import patient_stratified_folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook", default="data/grape/VF and clinical information.xlsx"
    )
    parser.add_argument("--image-dir", default="data/grape/images")
    parser.add_argument(
        "--output", default="data/grape/progression_3y_multimodal_v1.csv"
    )
    parser.add_argument("--minimum-followup", type=float, default=2.5)
    parser.add_argument("--maximum-followup", type=float, default=3.5)
    parser.add_argument("--minimum-visits", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    baseline, visits = load_grape_examinations(args.workbook)
    cohort = build_three_year_progression_cohort(
        baseline,
        visits,
        args.image_dir,
        minimum_followup_years=args.minimum_followup,
        maximum_followup_years=args.maximum_followup,
        minimum_visits=args.minimum_visits,
    )
    cohort["outer_fold"] = patient_stratified_folds(
        cohort,
        folds=args.folds,
        seed=args.seed,
        label_column="primary_progression_label",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    report = {
        "schema_version": "1.0",
        "endpoint": str(cohort["endpoint"].iloc[0]),
        "forecast_horizon_years": 3.0,
        "eyes": len(cohort),
        "patients": int(cohort["patient_id"].nunique()),
        "progressors": int(cohort["primary_progression_label"].sum()),
        "visual_field_visits": int(visits.shape[0]),
        "minimum_complete_followup_years": float(
            cohort["maximum_followup_years"].min()
        ),
        "maximum_complete_followup_years": float(
            cohort["maximum_followup_years"].max()
        ),
        "warning": (
            "GRAPE provides eye-level PLR2 status rather than an event date. This "
            "small fixed-window cohort supports preliminary approximately-three-year "
            "risk research, not a clinical prognosis claim."
        ),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

