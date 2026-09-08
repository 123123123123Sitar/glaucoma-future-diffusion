#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from glaucoma_forecast.training.scenario_trainer import (
    ScenarioTrainingConfig,
    train_scenario_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the 512px visual-scenario model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--identity-weight", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    print(train_scenario_model(ScenarioTrainingConfig(**vars(args))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
