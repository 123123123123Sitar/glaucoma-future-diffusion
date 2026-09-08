#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from glaucoma_forecast.inference.residual_diffusion_predictor import (
    generate_direct_trajectory,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate direct, non-recursive retinal progression scenarios."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--detail-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--years", nargs="+", type=float)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--global-change-scale", type=float, default=1.0)
    parser.add_argument("--detail-change-scale", type=float, default=1.0)
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument("--interpolate-final-detail", action="store_true")
    parser.add_argument("--laterality", choices=["OD", "OS"])
    parser.add_argument(
        "--clinical-condition",
        type=float,
        help="Required by conditioned checkpoints: 0=healthy, 1=glaucoma.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            generate_direct_trajectory(
                image_path=args.image,
                checkpoint_path=args.checkpoint,
                detail_checkpoint_path=args.detail_checkpoint,
                output_dir=args.output_dir,
                years=args.years,
                trajectories=args.trajectories,
                sampling_steps=args.sampling_steps,
                device=args.device,
                seed=args.seed,
                global_change_scale=args.global_change_scale,
                detail_change_scale=args.detail_change_scale,
                clinical_condition=args.clinical_condition,
                save_candidates=args.save_candidates,
                interpolate_final_detail=args.interpolate_final_detail,
                laterality=args.laterality,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
