"""End-to-end, abstention-aware single-fundus glaucoma-risk inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from glaucoma_forecast.data.highres import (
    preprocess_high_resolution,
    save_high_resolution_artifacts,
)
from glaucoma_forecast.inference.contracts import RiskForecastReport
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.risk_model import build_dual_view_risk_model
from glaucoma_forecast.models.scenario_diffusion import (
    build_scenario_diffusion,
    sample_scenario_trajectories,
)
from glaucoma_forecast.utils.reproducibility import select_device


def _tensor(image: Image.Image, device: str, grayscale_to_rgb: bool = False):
    torch = require_torch()
    converted = image.convert("RGB") if grayscale_to_rgb else image.convert("RGB")
    array = np.asarray(converted, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).unsqueeze(0).to(device)


def _save_mask(mask, path: Path) -> str:
    array = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)
    return str(path)


def _load_risk_member(path: str, device: str):
    torch = require_torch()
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = build_dual_view_risk_model(
        years=int(config.get("years", 10)),
        feature_dim=int(config.get("feature_dim", 256)),
        backbone=str(config.get("backbone", "compact")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _save_scenario_outputs(trajectories, out: Path):
    torch = require_torch()
    # trajectories: [batch=1, paths, years, channels, height, width]
    data = trajectories[0]
    medoids = data.mean(dim=0)
    uncertainty = data.std(dim=0).mean(dim=1)
    scenario_paths = []
    uncertainty_paths = []
    for year in range(10):
        image_array = (
            medoids[year].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255
        ).astype(np.uint8)
        image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
        draw.text((6, 6), f"YEAR {year + 1} SIMULATED SCENARIO - NOT A DIAGNOSIS", fill="white")
        path = out / f"scenario_year_{year + 1:02d}.png"
        image.save(path)
        scenario_paths.append(str(path))
        uncertainty_paths.append(
            _save_mask(
                uncertainty[year].clamp(0, 1),
                out / f"scenario_uncertainty_year_{year + 1:02d}.png",
            )
        )
    return scenario_paths, uncertainty_paths


def predict_single_fundus_risk(
    image_path: str,
    output_dir: str,
    risk_checkpoints: list[str] | None = None,
    scenario_checkpoint: str | None = None,
    device: str = "auto",
    image_size: int = 512,
    num_trajectories: int = 8,
) -> dict:
    torch = require_torch()
    selected_device = select_device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = preprocess_high_resolution(image_path, image_size)
    artifacts = save_high_resolution_artifacts(processed, out)
    quality = processed.quality
    warnings = [] if quality.reason == "ok" else quality.reason.split(";")
    report = RiskForecastReport(
        input_image=str(image_path),
        quality_status="reject" if quality.is_low_quality or quality.ood_score >= 0.4 else "pass",
        quality_warnings=warnings,
        quality_metrics={
            "score": quality.score,
            "sharpness_score": quality.sharpness_score,
            "field_coverage": quality.field_coverage,
            "source_width": quality.width,
            "source_height": quality.height,
        },
        ood_score=quality.ood_score,
        preprocessing={**processed.metadata(), **artifacts},
    )
    checkpoints = risk_checkpoints or []
    if not checkpoints:
        report.model_status = "risk_checkpoint_required"
        report.warnings.append(
            "No v2 incident-risk checkpoint was supplied; no patient risk or visual "
            "scenario has been produced."
        )
        result = report.to_dict()
        (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if report.quality_status != "pass":
        report.model_status = "abstained_input_quality"
        report.warnings.append("Prediction withheld because the input failed quality/OOD checks.")
        result = report.to_dict()
        (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    full = _tensor(processed.full_field, selected_device)
    disc = _tensor(processed.optic_disc, selected_device)
    member_risks = []
    member_hazards = []
    member_current = []
    member_vcdr = []
    masks = []
    versions = []
    current_head_trained = True
    vcdr_head_trained = True
    mask_head_trained = True
    hazard_head_trained = True
    current_thresholds = []
    with torch.no_grad():
        for checkpoint_path in checkpoints:
            model, checkpoint = _load_risk_member(checkpoint_path, selected_device)
            outputs = model(full, disc)
            temperature = max(float(checkpoint.get("temperature", 1.0)), 1e-3)
            hazards = torch.sigmoid(outputs["hazard_logits"] / temperature)
            risks = 1.0 - torch.cumprod(1.0 - hazards, dim=1)
            member_risks.append(risks[0])
            member_hazards.append(hazards[0])
            current_temperature = max(
                float(checkpoint.get("current_temperature", 1.0)), 1e-3
            )
            current_thresholds.append(float(checkpoint.get("current_threshold", 0.5)))
            member_current.append(
                torch.sigmoid(outputs["current_glaucoma_logit"] / current_temperature)[0]
            )
            member_vcdr.append(outputs["vcdr"][0])
            masks.append(torch.sigmoid(outputs["disc_cup_mask_logits"])[0])
            versions.append(str(checkpoint.get("model_version", Path(checkpoint_path).stem)))
            current_head_trained = current_head_trained and bool(
                checkpoint.get("current_head_trained", False)
            )
            vcdr_head_trained = vcdr_head_trained and bool(
                checkpoint.get("vcdr_head_trained", False)
            )
            mask_head_trained = mask_head_trained and bool(
                checkpoint.get("mask_head_trained", False)
            )
            hazard_head_trained = hazard_head_trained and bool(
                checkpoint.get("hazard_head_trained", False)
            )
    risk_stack = torch.stack(member_risks)
    mean_risk = risk_stack.mean(dim=0)
    report.model_status = (
        "research_prediction" if hazard_head_trained else "current_glaucoma_gate_only"
    )
    report.model_version = "+".join(versions)
    if current_head_trained:
        report.possible_current_glaucoma_probability = float(
            torch.stack(member_current).mean().cpu()
        )
        current_threshold = float(np.mean(current_thresholds))
        report.forecast_applicable = (
            report.possible_current_glaucoma_probability < current_threshold
        )
        report.preprocessing["current_glaucoma_operating_threshold"] = current_threshold
    else:
        report.possible_current_glaucoma_probability = None
        report.forecast_applicable = False
        report.warnings.append(
            "Current-glaucoma head lacks both positive and negative supervision; incident "
            "risk was withheld because existing disease cannot be excluded."
        )
    if not hazard_head_trained:
        report.forecast_applicable = False
        report.warnings.append(
            "Annual survival head lacks longitudinal incidence supervision; 2/5/10-year "
            "risk was withheld."
        )
    if vcdr_head_trained:
        report.vcdr_estimate = float(torch.stack(member_vcdr).mean().cpu())
    else:
        report.warnings.append("VCDR head was not supervised; VCDR output was withheld.")
    if mask_head_trained:
        mask = torch.stack(masks).mean(dim=0)
        report.disc_cup_masks = {
            "disc_probability": _save_mask(mask[0], out / "disc_probability_mask.png"),
            "cup_probability": _save_mask(mask[1], out / "cup_probability_mask.png"),
        }
    else:
        report.warnings.append(
            "Disc/cup segmentation head was not supervised; anatomy masks were withheld."
        )
    if current_head_trained and hazard_head_trained and not report.forecast_applicable:
        report.warnings.append(
            "Possible existing glaucoma exceeds the configured threshold; incident-onset "
            "forecast is not applicable and specialist assessment is recommended."
        )
    elif report.forecast_applicable:
        mean_hazards = torch.stack(member_hazards).mean(dim=0)
        report.annual_hazard_years_1_to_10 = [
            float(value) for value in mean_hazards.cpu()
        ]
        report.risk_2y = float(mean_risk[1].cpu())
        report.risk_5y = float(mean_risk[4].cpu())
        report.risk_10y = float(mean_risk[9].cpu())
        for label, index in [("risk_2y", 1), ("risk_5y", 4), ("risk_10y", 9)]:
            values = risk_stack[:, index]
            report.confidence_intervals[label] = [
                float(torch.quantile(values, 0.025).cpu()),
                float(torch.quantile(values, 0.975).cpu()),
            ]
    if scenario_checkpoint and report.forecast_applicable:
        checkpoint = torch.load(scenario_checkpoint, map_location=selected_device)
        scenario_config = checkpoint.get("config", {})
        scenario_model = build_scenario_diffusion(
            latent_channels=int(scenario_config.get("latent_channels", 8)),
            base_channels=int(scenario_config.get("base_channels", 64)),
        ).to(selected_device)
        scenario_model.load_state_dict(checkpoint["model"])
        vessel = _tensor(processed.vessel_map, selected_device, grayscale_to_rgb=True)
        trajectories = sample_scenario_trajectories(
            scenario_model,
            full,
            vessel,
            mean_risk[None],
            num_trajectories=num_trajectories,
        )
        (
            report.annual_scenario_images_512px,
            report.scenario_uncertainty_maps,
        ) = _save_scenario_outputs(trajectories, out)
    elif report.forecast_applicable:
        report.warnings.append(
            "No v2 scenario checkpoint supplied; risk was computed without generating images."
        )
    result = report.to_dict()
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
