#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.grape_glaucoma_trainer import TrainingConfig, train_grape_glaucoma_denoiser
from glaucoma_forecast.training.diffusion_trainer import run_one_batch_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Train compact longitudinal latent diffusion model.")
    parser.add_argument("--config", default="configs/sigf_small.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir", default="outputs/training/grape_glaucoma")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--crop-fraction", type=float, default=0.50)
    parser.add_argument("--max-change", type=float, default=0.18)
    args = parser.parse_args()
    if args.dry_run:
        print(run_one_batch_smoke(args.device))
        return 0
    if not args.manifest:
        raise RuntimeError("Pass --manifest data/grape/manifest.csv for GRAPE glaucoma-eye training.")
    summary = train_grape_glaucoma_denoiser(
        TrainingConfig(
            manifest=args.manifest,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            validate_every=args.validate_every,
            base_channels=args.base_channels,
            crop_fraction=args.crop_fraction,
            max_change=args.max_change,
        )
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
