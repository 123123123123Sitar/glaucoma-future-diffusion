"""Abstention-aware inference for known-glaucoma three-year progression."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from glaucoma_forecast.data.grape_multimodal import CLINICAL_COLUMNS
from glaucoma_forecast.data.highres import (
    preprocess_high_resolution,
    save_high_resolution_artifacts,
)
from glaucoma_forecast.inference.contracts import (
    BaselineExam,
    ProgressionForecastReport,
)
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.multimodal_progression import (
    build_multimodal_progression_model,
)
from glaucoma_forecast.training.multimodal_progression_trainer import encode_clinical
from glaucoma_forecast.training.risk_trainer import _image_tensor
from glaucoma_forecast.utils.reproducibility import select_device


def _load_member(path: str | Path, device: str):
    torch = require_torch()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    expected_columns = list(checkpoint.get("clinical_columns", []))
    if expected_columns != list(CLINICAL_COLUMNS):
        raise ValueError("Checkpoint clinical schema does not match inference schema")
    model = build_multimodal_progression_model(
        clinical_dim=int(checkpoint["clinical_dim"]),
        feature_dim=int(config["feature_dim"]),
        backbone=str(config["backbone"]),
        variant=str(config["variant"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def predict_progression_3y(
    exam: BaselineExam,
    output_dir: str | Path,
    checkpoints: list[str],
    device: str = "auto",
    image_size: int = 512,
) -> dict[str, object]:
    """Run the calibrated ensemble; withhold output on unsupported inputs."""

    torch = require_torch()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = ProgressionForecastReport()
    validation_errors = exam.validate()
    if validation_errors:
        report.model_status = "invalid_input"
        report.abstention_reasons.extend(validation_errors)
        result = report.to_dict()
        (output / "report.json").write_text(json.dumps(result, indent=2))
        return result
    if not checkpoints:
        report.model_status = "progression_checkpoint_required"
        report.abstention_reasons.append("no_progression_checkpoint")
        result = report.to_dict()
        (output / "report.json").write_text(json.dumps(result, indent=2))
        return result

    processed = preprocess_high_resolution(exam.fundus_image, image_size)
    artifacts = save_high_resolution_artifacts(processed, output)
    report.preprocessing = {**processed.metadata(), **artifacts}
    report.gradability = "reject" if processed.quality.is_low_quality else "pass"
    report.ood_status = "out_of_distribution" if processed.quality.ood_score >= 0.4 else "pass"
    if report.gradability != "pass":
        report.abstention_reasons.append("image_quality_failure")
    if report.ood_status != "pass":
        report.abstention_reasons.append("image_out_of_distribution")
    if not exam.known_glaucoma:
        report.abstention_reasons.append("known_glaucoma_required")

    row = pd.Series(exam.to_feature_row())
    observed = pd.to_numeric(row[list(CLINICAL_COLUMNS)], errors="coerce").notna()
    report.clinical_observed_fraction = float(observed.mean())
    if report.clinical_observed_fraction < 0.70:
        report.abstention_reasons.append("insufficient_multimodal_baseline")
    if report.abstention_reasons:
        report.model_status = "abstained"
        report.applicability = "unsupported_input"
        result = report.to_dict()
        (output / "report.json").write_text(json.dumps(result, indent=2))
        return result

    selected_device = select_device(device)
    full = _image_tensor(processed.full_field).unsqueeze(0).to(selected_device)
    disc = _image_tensor(processed.optic_disc).unsqueeze(0).to(selected_device)
    probabilities = []
    thresholds = []
    versions = []
    with torch.no_grad():
        for checkpoint_path in checkpoints:
            model, checkpoint = _load_member(checkpoint_path, selected_device)
            mean = np.asarray(checkpoint["clinical_mean"], dtype=np.float32)
            scale = np.asarray(checkpoint["clinical_scale"], dtype=np.float32)
            clinical = (
                torch.from_numpy(encode_clinical(row, mean, scale))
                .unsqueeze(0)
                .to(selected_device)
            )
            logit = model(full, disc, clinical)["progression_logit"]
            temperature = max(float(checkpoint["temperature"]), 1e-3)
            probabilities.append(float(torch.sigmoid(logit / temperature)[0].cpu()))
            thresholds.append(float(checkpoint["decision_threshold"]))
            versions.append(str(checkpoint["model_version"]))
    values = np.asarray(probabilities, dtype=float)
    report.model_status = "research_prediction"
    report.applicability = "known_glaucoma_multimodal_baseline"
    report.model_version = "+".join(versions)
    report.progression_probability_3y = float(values.mean())
    report.ensemble_interval = [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]
    report.decision_threshold = float(np.mean(thresholds))
    report.warnings.append(
        "The interval reflects variation across cross-validation members; it is "
        "not a patient-level clinical confidence interval."
    )
    result = report.to_dict()
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return result

