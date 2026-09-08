#!/usr/bin/env python3
"""Run the pinned external SwinV2 glaucoma screening checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/external/pamixsun_swinv2_glaucoma"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            "outputs/evaluation/pamixsun_swinv2_on_papila/"
            "threshold_calibration.json"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    threshold = float(calibration["calibration"]["threshold"])
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(args.device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    inputs = {
        key: value.to(args.device)
        for key, value in processor(images=image, return_tensors="pt").items()
    }
    with torch.inference_mode():
        probability = float(
            torch.softmax(model(**inputs).logits, dim=1)[0, 1].detach().cpu()
        )

    report = {
        "model_repository": "pamixsun/swinv2_tiny_for_glaucoma_classification",
        "pinned_revision": "a25a03d9ff23d6fbaf6cd7a329373253f4671c63",
        "input_image": str(args.image),
        "glaucoma_probability": probability,
        "screen_positive": probability >= threshold,
        "papila_calibrated_screening_threshold": threshold,
        "scope": "current_glaucoma_screening_only",
        "progression_prediction": None,
        "warning": (
            "Research-only output. Not a diagnosis, not validated for patient care, "
            "and not a prediction of future glaucoma progression."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
