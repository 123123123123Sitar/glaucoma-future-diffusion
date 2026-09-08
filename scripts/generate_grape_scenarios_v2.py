#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from glaucoma_forecast.inference.scenario_predictor import (
    generate_existing_glaucoma_scenarios,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate research-only 512px scenarios for an already-glaucomatous eye."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_existing_glaucoma_scenarios(
                image_path=args.image,
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                num_trajectories=args.num_trajectories,
                sampling_steps=args.sampling_steps,
                seed=args.seed,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
