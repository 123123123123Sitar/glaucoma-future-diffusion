#!/usr/bin/env python3
"""Print or run the frozen three-year model ladder."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--initialize-retinal-encoder-from",
        default="outputs/training/hygd_current_gate_convnext_tiny_512_mps/best_calibrated.pt",
    )
    args = parser.parse_args()
    for variant in ["clinical_only", "image_only", "multimodal"]:
        command = [
            sys.executable,
            "scripts/train_multimodal_progression_3y.py",
            "--variant",
            variant,
            "--output-dir",
            f"outputs/training/progression_3y_{variant}",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
        ]
        if variant != "clinical_only" and args.initialize_retinal_encoder_from:
            command.extend(
                [
                    "--initialize-retinal-encoder-from",
                    args.initialize_retinal_encoder_from,
                ]
            )
        print(" ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
