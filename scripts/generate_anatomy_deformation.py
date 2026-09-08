#!/usr/bin/env python3
"""Generate one held-out original-to-five-year anatomy deformation preview."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from glaucoma_forecast.inference.anatomy_deformation_predictor import generate_anatomy_deformation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/training/grape_anatomy_deformation_v1/best.pt")
    parser.add_argument("--manifest", default="outputs/annotations/grape_expert_anatomy_v1/manifest.csv")
    parser.add_argument("--output-dir", default="outputs/diffusion_forecast/grape_best_expert_progressor_v1")
    parser.add_argument("--annotation-id")
    parser.add_argument("--horizon-years", type=float, default=5.0)
    parser.add_argument("--progression-strength", type=float)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = generate_anatomy_deformation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        annotation_id=args.annotation_id,
        horizon_years=args.horizon_years,
        progression_strength=args.progression_strength,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
