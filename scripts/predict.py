#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from glaucoma_forecast.inference.predictor import predict_research_forecast


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only longitudinal glaucoma forecast.")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--times", nargs="+", type=float, required=True)
    parser.add_argument("--horizon-years", type=float, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["forecasting", "controlled_synthesis"], default="forecasting")
    parser.add_argument("--controlled-class", choices=["normal", "glaucoma"])
    args = parser.parse_args()
    supplied = {"controlled_class"} if args.controlled_class else set()
    report = predict_research_forecast(
        args.images,
        args.times,
        args.horizon_years,
        args.num_samples,
        args.output_dir,
        mode=args.mode,
        supplied_fields=supplied,
    )
    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}")
    print(f"Wrote research-only prediction package to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
