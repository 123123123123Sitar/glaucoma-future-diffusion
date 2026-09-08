#!/usr/bin/env python3
"""Create an auditable release gate for the balanced formative image pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument(
        "--data-report",
        default="data/derived/balanced_stability_pairs_v1_report.json",
    )
    parser.add_argument(
        "--output", default="outputs/release/balanced_diffusion_pilot_gate.json"
    )
    parser.add_argument("--maximum-healthy-change-mae", type=float, default=0.005)
    args = parser.parse_args()

    evaluation = json.loads(Path(args.evaluation).read_text())
    data = json.loads(Path(args.data_report).read_text())
    glaucoma = evaluation["stratified"]["grape_glaucoma_longitudinal"]
    healthy = evaluation["stratified"]["papila_healthy_stability_control"]
    checks = {
        "exact_class_balance": data["class_counts"] == {
            "glaucoma": 648,
            "healthy": 648,
        },
        "no_patient_leakage": not data["patient_leakage"],
        "glaucoma_beats_persistence_with_95ci": (
            glaucoma["paired_model_minus_persistence_mae"]["ci_95"][1] < 0
        ),
        "healthy_generated_change_below_limit": (
            healthy["model_mae"] <= args.maximum_healthy_change_mae
        ),
    }
    pilot_passed = all(checks.values())
    report = {
        "status": (
            "balanced_formative_image_pilot_released"
            if pilot_passed
            else "balanced_formative_image_pilot_withheld"
        ),
        "checks": checks,
        "pilot_passed": pilot_passed,
        "patient_specific_forecast_released": False,
        "glaucoma_locked_metrics": glaucoma,
        "healthy_stability_locked_metrics": healthy,
        "healthy_change_limit_mae": args.maximum_healthy_change_mae,
        "scope": (
            "Release applies only to blinded realism, class-preservation, and "
            "healthy-stability evaluation. It does not release diagnosis or prognosis."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if pilot_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
