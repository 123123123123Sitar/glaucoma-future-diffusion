#!/usr/bin/env python3
"""Audit disease-class preservation in the frozen balanced clinician pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private-key",
        default="outputs/clinician_balanced_v3/private-answer-key.json",
        type=Path,
    )
    parser.add_argument(
        "--public-root",
        default="clinician-evaluation/public/cases-balanced-v3",
        type=Path,
    )
    parser.add_argument(
        "--model-dir",
        default="models/external/pamixsun_swinv2_glaucoma",
        type=Path,
    )
    parser.add_argument(
        "--calibration",
        default=(
            "outputs/evaluation/pamixsun_swinv2_on_papila/"
            "threshold_calibration.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluation/balanced_clinician_external_detector/report.json",
        type=Path,
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    private = json.loads(args.private_key.read_text())
    calibration = json.loads(args.calibration.read_text())
    threshold = float(calibration["calibration"]["threshold"])
    processor = AutoImageProcessor.from_pretrained(
        args.model_dir, local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(args.device)
    model.eval()

    records = []
    for case in private["cases"]:
        case_dir = args.public_root / case["caseId"]
        baseline = Image.open(case_dir / "reference.webp").convert("RGB")
        generated_side = "B" if case["referenceSide"] == "A" else "A"
        generated = Image.open(
            case_dir / f"candidate-{generated_side.lower()}.webp"
        ).convert("RGB")
        inputs = processor(
            images=[baseline, generated], return_tensors="pt"
        )
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = (
                torch.softmax(model(**inputs).logits, dim=1)[:, 1]
                .detach()
                .cpu()
                .numpy()
            )
        truth = int(case["diagnosis"] == "glaucoma")
        records.append(
            {
                "caseId": case["caseId"],
                "diagnosis": case["diagnosis"],
                "baselineProbability": float(probabilities[0]),
                "generatedProbability": float(probabilities[1]),
                "absoluteProbabilityChange": float(
                    abs(probabilities[1] - probabilities[0])
                ),
                "baselineCorrect": bool((probabilities[0] >= threshold) == truth),
                "generatedCorrect": bool((probabilities[1] >= threshold) == truth),
            }
        )

    report = {
        "status": "balanced_pack_external_detector_audit_complete",
        "model": "pamixsun/swinv2_tiny_for_glaucoma_classification",
        "threshold": threshold,
        "cases": len(records),
        "baselineClassAccuracy": float(
            np.mean([record["baselineCorrect"] for record in records])
        ),
        "generatedClassAccuracy": float(
            np.mean([record["generatedCorrect"] for record in records])
        ),
        "meanAbsoluteProbabilityChange": float(
            np.mean([record["absoluteProbabilityChange"] for record in records])
        ),
        "byDiagnosis": {
            diagnosis: {
                "cases": len(selected),
                "baselineClassAccuracy": float(
                    np.mean([record["baselineCorrect"] for record in selected])
                ),
                "generatedClassAccuracy": float(
                    np.mean([record["generatedCorrect"] for record in selected])
                ),
                "meanAbsoluteProbabilityChange": float(
                    np.mean(
                        [record["absoluteProbabilityChange"] for record in selected]
                    )
                ),
            }
            for diagnosis in ("healthy", "glaucoma")
            if (
                selected := [
                    record for record in records if record["diagnosis"] == diagnosis
                ]
            )
        },
        "records": records,
        "interpretation": (
            "This frozen external detector is a class-consistency audit only. "
            "It does not establish future-image accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
