"""Inference orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageChops

from glaucoma_forecast.data.pairing import assert_no_target_label_leakage
from glaucoma_forecast.data.preprocessing import preprocess_image
from glaucoma_forecast.data.quality import assess_image_quality
from glaucoma_forecast.evaluation.reports import DISCLAIMER


def predict_research_forecast(
    image_paths: list[str],
    times: list[float],
    horizon_years: float,
    num_samples: int,
    output_dir: str | Path,
    mode: str = "forecasting",
    supplied_fields: set[str] | None = None,
) -> dict:
    if len(image_paths) != len(times):
        raise ValueError("--images and --times must have the same length")
    if len(image_paths) < 1:
        raise ValueError("At least one image is required")
    assert_no_target_label_leakage(mode, supplied_fields or set())
    if len(image_paths) == 1:
        single_warning = "Single-image forecast has less patient-specific longitudinal information."
    else:
        single_warning = None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    preprocessed = []
    quality_warnings = []
    for idx, path in enumerate(image_paths):
        quality = assess_image_quality(path)
        if quality.is_low_quality:
            quality_warnings.append({"image": path, "reason": quality.reason, "score": quality.score})
        result = preprocess_image(path, output_size=256)
        pp = out / f"preprocessed_input_{idx:02d}.png"
        result.image.save(pp)
        preprocessed.append(str(pp))

    latest = Image.open(preprocessed[-1]).convert("RGB")
    samples = []
    # Deterministic persistence samples until a trained checkpoint is supplied.
    # This is explicitly marked as a baseline, not a generative forecast.
    for i in range(num_samples):
        sample_path = out / f"sample_{i:03d}.png"
        latest.save(sample_path)
        samples.append(str(sample_path))
    mean_path = out / "mean_or_medoid.png"
    latest.save(mean_path)
    uncertainty = ImageChops.constant(latest, 0)
    uncertainty_path = out / "pixel_uncertainty_map.png"
    uncertainty.save(uncertainty_path)

    report = {
        "disclaimer": DISCLAIMER,
        "mode": mode,
        "baseline_used": "last_observation_persistence_until_trained_checkpoint_is_provided",
        "images": image_paths,
        "times": times,
        "horizon_years": horizon_years,
        "num_samples": num_samples,
        "preprocessed_inputs": preprocessed,
        "samples": samples,
        "mean_or_medoid": str(mean_path),
        "pixel_uncertainty_map": str(uncertainty_path),
        "predicted_vcdr_distribution": None,
        "research_model_estimated_progression_probability": None,
        "quality_warnings": quality_warnings,
        "warnings": [single_warning] if single_warning else [],
        "model_checkpoint": None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
