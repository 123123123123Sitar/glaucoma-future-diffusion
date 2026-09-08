#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from glaucoma_forecast.inference.risk_predictor import predict_single_fundus_risk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quality-gated 512px single-fundus glaucoma-risk research inference."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--risk-checkpoints", nargs="*", default=[])
    parser.add_argument("--scenario-checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-trajectories", type=int, default=8)
    args = parser.parse_args()
    report = predict_single_fundus_risk(
        image_path=args.image,
        output_dir=args.output_dir,
        risk_checkpoints=args.risk_checkpoints,
        scenario_checkpoint=args.scenario_checkpoint,
        device=args.device,
        num_trajectories=args.num_trajectories,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
