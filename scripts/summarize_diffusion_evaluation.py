#!/usr/bin/env python3
"""Add paired uncertainty and horizon strata to a diffusion evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_bootstrap(values: np.ndarray, seed: int, samples: int = 10_000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "bootstrap_samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    root = Path(args.evaluation_dir)
    frame = pd.read_csv(root / "per_pair.csv")
    difference = (
        frame["model_mae"].to_numpy(float)
        - frame["persistence_mae"].to_numpy(float)
    )
    strata = []
    boundaries = [(0.0, 2.0), (2.0, 3.0), (3.0, float("inf"))]
    for lower, upper in boundaries:
        selected = frame[
            (frame["horizon_years"] > lower)
            & (frame["horizon_years"] <= upper)
        ]
        if selected.empty:
            continue
        strata.append(
            {
                "horizon_years": f"({lower:g}, {upper:g}]",
                "n": len(selected),
                "model_mae": float(selected["model_mae"].mean()),
                "persistence_mae": float(selected["persistence_mae"].mean()),
                "paired_model_minus_persistence_mae": paired_bootstrap(
                    (
                        selected["model_mae"].to_numpy(float)
                        - selected["persistence_mae"].to_numpy(float)
                    ),
                    args.seed + len(strata) + 1,
                ),
            }
        )
    report_path = root / "report.json"
    report = json.loads(report_path.read_text())
    report["paired_model_minus_persistence_mae"] = paired_bootstrap(
        difference, args.seed
    )
    report["pairwise_model_win_fraction"] = float(np.mean(difference < 0))
    if "cohort" in frame.columns:
        for cohort, selected in frame.groupby("cohort"):
            cohort_difference = (
                selected["model_mae"].to_numpy(float)
                - selected["persistence_mae"].to_numpy(float)
            )
            report.setdefault("stratified", {}).setdefault(cohort, {})[
                "paired_model_minus_persistence_mae"
            ] = paired_bootstrap(
                cohort_difference,
                args.seed + sum(ord(char) for char in str(cohort)),
            )
            report["stratified"][cohort]["pairwise_model_win_fraction"] = float(
                np.mean(cohort_difference < 0)
            )
    report["horizon_strata"] = strata
    report["interpretation"] = (
        "model_better_than_persistence"
        if report["paired_model_minus_persistence_mae"]["ci_95"][1] < 0
        else "model_not_demonstrated_better_than_persistence"
    )
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
