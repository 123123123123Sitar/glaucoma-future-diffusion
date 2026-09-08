#!/usr/bin/env python3
"""Convert an existing leakage-safe GRAPE pair split to V2 scenario format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs)
    required = {
        "patient_id",
        "context_image_path",
        "target_image_path",
        "forecast_horizon",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair CSV missing columns: {missing}")
    scenario = pd.DataFrame(
        {
            "patient_id": pairs["patient_id"],
            "baseline_image_path": pairs["context_image_path"],
            "future_image_path": pairs["target_image_path"],
            "future_year": pairs["forecast_horizon"].clip(upper=10.0),
            # GRAPE contains diagnosed glaucoma eyes only. This constant is a
            # disease-state condition, not an incident-risk prediction.
            "risk_model_cumulative_risk": 1.0,
            "conditioning_scope": "existing_glaucoma_only",
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scenario.to_csv(output, index=False)
    print(
        {
            "output": str(output),
            "pairs": len(scenario),
            "patients": int(scenario["patient_id"].nunique()),
            "max_observed_horizon_years": float(scenario["future_year"].max()),
            "scope": "existing_glaucoma_only_not_incident_risk",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
