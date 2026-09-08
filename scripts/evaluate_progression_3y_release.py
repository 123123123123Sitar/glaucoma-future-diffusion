#!/usr/bin/env python3
"""Evaluate frozen release gates without retraining or threshold tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from glaucoma_forecast.evaluation.release_gate import (
    build_release_report,
    load_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clinical",
        default="outputs/training/progression_3y_clinical_full/report.json",
    )
    parser.add_argument(
        "--image",
        default="outputs/training/progression_3y_image_hygd_transfer/report.json",
    )
    parser.add_argument(
        "--multimodal",
        default="outputs/training/progression_3y_multimodal_hygd_transfer/report.json",
    )
    parser.add_argument(
        "--harvard-known-glaucoma",
        default=(
            "outputs/training/harvard_gdp_known_glaucoma_"
            "td_pointwise_no_p_cut/report.json"
        ),
    )
    parser.add_argument(
        "--harvard-oct-known-glaucoma",
        default="outputs/training/harvard_gdp_oct_multimodal/report.json",
    )
    parser.add_argument(
        "--output", default="outputs/release/progression_3y_gate.json"
    )
    args = parser.parse_args()
    candidates = load_reports(
        {
            "grape_clinical": args.clinical,
            "grape_image": args.image,
            "grape_multimodal": args.multimodal,
            "harvard_known_glaucoma_secondary": args.harvard_known_glaucoma,
            "harvard_oct_known_glaucoma_posthoc": (
                args.harvard_oct_known_glaucoma
            ),
        }
    )
    report = build_release_report(
        candidates,
        primary_candidate_names=[
            "grape_clinical",
            "grape_image",
            "grape_multimodal",
        ],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["progression_output_released"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
