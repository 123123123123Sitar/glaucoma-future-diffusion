#!/usr/bin/env python3
"""Predict approximately-three-year progression from a baseline-exam JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from glaucoma_forecast.inference.contracts import BaselineExam
from glaucoma_forecast.inference.progression_predictor import predict_progression_3y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam-json", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    exam = BaselineExam(**json.loads(Path(args.exam_json).read_text()))
    report = predict_progression_3y(
        exam,
        args.output_dir,
        checkpoints=args.checkpoint,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

