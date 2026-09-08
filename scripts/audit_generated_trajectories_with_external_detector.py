#!/usr/bin/env python3
"""Score generated trajectories and build slide-ready visual difference audits."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.highres import preprocess_high_resolution


def clean_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    # Generated files have a 28px research watermark. Apply the same crop to
    # baselines and generated images so the classifier comparison is fair.
    return image.crop((0, 28, image.width, image.height)).resize(
        (512, 512), Image.Resampling.BICUBIC
    )


def year_from_path(path: Path) -> float:
    match = re.search(r"trajectory_year_(.+)\.png$", path.name)
    if not match:
        raise ValueError(f"Cannot parse trajectory year from {path}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--difference-amplification", type=float, default=12.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    threshold_report = json.loads(args.calibration.read_text(encoding="utf-8"))
    threshold = float(threshold_report["calibration"]["threshold"])
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(args.device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    case_images: dict[str, list[tuple[float, Path, Image.Image]]] = {}

    for case_dir in sorted(path for path in args.trajectory_root.iterdir() if path.is_dir()):
        items = [(0.0, case_dir / "input_full_field_512.png")]
        items.extend(
            sorted(
                ((year_from_path(path), path) for path in case_dir.glob("trajectory_year_*.png")),
                key=lambda item: item[0],
            )
        )
        cleaned = [(year, path, clean_image(path)) for year, path in items]
        case_images[case_dir.name] = cleaned
        inputs = processor(images=[image for _, _, image in cleaned], return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = (
                torch.softmax(model(**inputs).logits, dim=1)[:, 1].detach().cpu().numpy()
            )
        baseline_probability = float(probabilities[0])
        baseline_array = np.asarray(cleaned[0][2], dtype=np.float32)
        for (year, path, image), probability in zip(cleaned, probabilities):
            mean_absolute_change = float(
                np.abs(np.asarray(image, dtype=np.float32) - baseline_array).mean()
            )
            records.append(
                {
                    "case": case_dir.name,
                    "year": year,
                    "image_path": str(path),
                    "glaucoma_probability": float(probability),
                    "probability_change_from_baseline": float(
                        probability - baseline_probability
                    ),
                    "screen_positive": bool(probability >= threshold),
                    "mean_absolute_pixel_change_0_to_255": mean_absolute_change,
                }
            )

    scores = pd.DataFrame(records)
    scores.to_csv(args.output_dir / "trajectory_detector_scores.csv", index=False)

    case_reports = []
    for case, images in case_images.items():
        case_scores = scores[scores["case"] == case].sort_values("year")
        actual_images = [image for _, _, image in images]
        baseline_array = np.asarray(actual_images[0], dtype=np.float32)
        disc_images = []
        for index, image in enumerate(actual_images):
            temporary = args.output_dir / f".{case}_{index}.png"
            image.save(temporary)
            disc_images.append(preprocess_high_resolution(temporary, 512).optic_disc)
            temporary.unlink()

        fig, axes = plt.subplots(3, len(images), figsize=(16, 11), dpi=160)
        fig.patch.set_facecolor("#07111f")
        fig.suptitle(
            f"{case.replace('_', ' ')} · External detector trajectory audit",
            color="white",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        for column, ((year, _, image), disc) in enumerate(zip(images, disc_images)):
            row = case_scores.iloc[column]
            label = "Baseline" if year == 0 else f"Year {year:g}"
            axes[0, column].imshow(image)
            axes[0, column].set_title(
                f"{label}\nGlaucoma score {row.glaucoma_probability:.3f}",
                color="white",
                fontsize=11,
            )
            axes[1, column].imshow(disc)
            axes[1, column].set_title("Optic-disc detail", color="#b9d7ff", fontsize=10)
            if column == 0:
                axes[2, column].text(
                    0.5,
                    0.5,
                    "Difference maps are\namplified for visibility",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=11,
                )
                axes[2, column].set_facecolor("#0d2037")
            else:
                difference = np.mean(
                    np.abs(np.asarray(image, dtype=np.float32) - baseline_array), axis=2
                )
                axes[2, column].imshow(
                    np.clip(difference * args.difference_amplification, 0, 255),
                    cmap="magma",
                    vmin=0,
                    vmax=255,
                )
                axes[2, column].set_title(
                    f"{args.difference_amplification:g}× absolute difference\n"
                    f"mean Δ {difference.mean():.2f}/255",
                    color="#ffcc80",
                    fontsize=10,
                )
            for row_index in range(3):
                axes[row_index, column].axis("off")
        fig.text(
            0.5,
            0.015,
            "Research-only simulated images. Detector-score change is a consistency check, not proof of future accuracy.",
            ha="center",
            color="#9fb1c7",
            fontsize=10,
        )
        fig.tight_layout(rect=(0.01, 0.04, 0.99, 0.95))
        figure_path = args.output_dir / f"{case}_slide_ready_audit.png"
        fig.savefig(figure_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        future_scores = case_scores[case_scores["year"] > 0]
        ordered_probabilities = case_scores["glaucoma_probability"].to_numpy()
        case_reports.append(
            {
                "case": case,
                "baseline_probability": float(ordered_probabilities[0]),
                "maximum_absolute_probability_change": float(
                    future_scores["probability_change_from_baseline"].abs().max()
                ),
                "score_monotonic_non_decreasing": bool(
                    np.all(np.diff(ordered_probabilities) >= 0)
                ),
                "slide_ready_audit": str(figure_path),
            }
        )

    report = {
        "status": "trajectory_detector_audit_complete",
        "external_detector": "pamixsun/swinv2_tiny_for_glaucoma_classification",
        "threshold": threshold,
        "cases": case_reports,
        "interpretation": (
            "Detector scores evaluate classifier consistency only. They do not establish "
            "that a generated image matches the patient's true future anatomy."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
