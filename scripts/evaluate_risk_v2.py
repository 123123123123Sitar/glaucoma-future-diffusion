#!/usr/bin/env python3
"""Evaluate already-exported V2 risks without rerunning image inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from glaucoma_forecast.evaluation.survival_metrics import horizon_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="CSV containing event_or_censor_time_years, event_observed, risk_2y, risk_5y, risk_10y.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    required = {
        "event_or_censor_time_years",
        "event_observed",
        "risk_2y",
        "risk_5y",
        "risk_10y",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {missing}")
    reports = []
    for horizon in [2, 5, 10]:
        reports.append(
            horizon_metrics(
                frame["event_or_censor_time_years"].to_numpy(),
                frame["event_observed"].to_numpy(),
                frame[f"risk_{horizon}y"].to_numpy(),
                float(horizon),
            )
        )
    report = {
        "rows": len(frame),
        "horizons": reports,
        "note": (
            "Dependency-light complete-case horizon metrics. A clinical report must "
            "also include IPCW/time-dependent metrics, confidence intervals, calibration "
            "plots, decision curves, subgroup analyses, and an external-site evaluation."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
