"""Standalone visual scenarios for an eye already known to have glaucoma."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from glaucoma_forecast.data.highres import (
    preprocess_high_resolution,
    save_high_resolution_artifacts,
)
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.scenario_diffusion import (
    build_scenario_diffusion,
    sample_scenario_trajectories,
)
from glaucoma_forecast.utils.reproducibility import select_device


def _tensor(image: Image.Image, device: str):
    torch = require_torch()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).unsqueeze(0).to(device)


def generate_existing_glaucoma_scenarios(
    image_path: str,
    checkpoint_path: str,
    output_dir: str,
    device: str = "auto",
    num_trajectories: int = 8,
    sampling_steps: int = 50,
    seed: int = 20260720,
) -> dict[str, object]:
    torch = require_torch()
    selected_device = select_device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = preprocess_high_resolution(image_path, 512)
    artifacts = save_high_resolution_artifacts(processed, out)
    if processed.quality.is_low_quality or processed.quality.ood_score >= 0.4:
        raise ValueError(f"Input failed quality/OOD checks: {processed.quality.reason}")
    checkpoint = torch.load(checkpoint_path, map_location=selected_device)
    config = checkpoint.get("config", {})
    model = build_scenario_diffusion(
        int(config.get("latent_channels", 8)),
        int(config.get("base_channels", 64)),
    ).to(selected_device)
    model.load_state_dict(checkpoint["model"])
    baseline = _tensor(processed.full_field, selected_device)
    vessel = _tensor(processed.vessel_map, selected_device)
    # This model was trained only on diagnosed GRAPE eyes. One is a cohort
    # condition, not a predicted probability of future onset.
    existing_glaucoma_condition = torch.ones((1, 10), device=selected_device)
    trajectories = sample_scenario_trajectories(
        model,
        baseline,
        vessel,
        existing_glaucoma_condition,
        num_trajectories=num_trajectories,
        diffusion_steps=sampling_steps,
        seed=seed,
    )[0]
    representative = trajectories.mean(dim=0)
    uncertainty = trajectories.std(dim=0).mean(dim=1)
    scenario_paths = []
    uncertainty_paths = []
    for index in range(10):
        array = (
            representative[index].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
            * 255
        ).astype(np.uint8)
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        label = f"YEAR {index + 1}: SIMULATED - NOT A DIAGNOSIS"
        if index + 1 > 4:
            label += " - EXTRAPOLATION"
        draw.rectangle((0, 0, image.width, 25), fill=(0, 0, 0))
        draw.text((5, 6), label, fill=(255, 255, 255))
        path = out / f"scenario_year_{index + 1:02d}.png"
        image.save(path)
        scenario_paths.append(str(path))
        uncertainty_array = (
            uncertainty[index].detach().cpu().numpy().clip(0, 1) * 255
        ).astype(np.uint8)
        uncertainty_path = out / f"uncertainty_year_{index + 1:02d}.png"
        Image.fromarray(uncertainty_array).save(uncertainty_path)
        uncertainty_paths.append(str(uncertainty_path))
    report = {
        "disclaimer": (
            "Research visual scenario only. Not validated for diagnosis, screening, "
            "prognosis, treatment, or patient-care decisions."
        ),
        "scope": "eyes_already_diagnosed_with_glaucoma",
        "not_an_incident_risk_model": True,
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint.get("step"),
        "observed_training_horizon_max_years": 4.427710843373494,
        "years_5_to_10": "unsupported_extrapolation",
        "recursive_rollout_used": False,
        "trajectory_method": "all years conditioned directly on the real baseline with shared path noise",
        "input_artifacts": artifacts,
        "scenario_images": scenario_paths,
        "uncertainty_maps": uncertainty_paths,
        "num_trajectories": num_trajectories,
        "sampling_steps": sampling_steps,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
