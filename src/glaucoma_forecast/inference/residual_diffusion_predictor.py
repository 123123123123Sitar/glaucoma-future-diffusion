"""Inference packaging for direct baseline-conditioned retinal trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from glaucoma_forecast.data.highres import (
    estimate_glaucoma_structural_maps,
    estimate_vessel_map,
    preprocess_high_resolution,
    save_high_resolution_artifacts,
)
from glaucoma_forecast.data.preprocessing import mirror_left_eye
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.detail_diffusion import sample_dual_scale_trajectory
from glaucoma_forecast.models.residual_diffusion import (
    build_residual_diffusion,
    sample_residual_trajectory,
)
from glaucoma_forecast.utils.reproducibility import select_device


def _tensor(image: Image.Image, device: str, grayscale: bool = False):
    torch = require_torch()
    converted = image.convert("L" if grayscale else "RGB")
    array = np.asarray(converted, dtype=np.float32) / 255.0
    if grayscale:
        array = array[None]
    else:
        array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array.copy()).unsqueeze(0).to(device)


def _structural_tensor(image: Image.Image, mode: str, size: int, device: str):
    if mode == "glaucoma":
        channels = [
            _tensor(
                feature.resize((size, size), Image.Resampling.BILINEAR),
                device,
                grayscale=True,
            )
            for feature in estimate_glaucoma_structural_maps(image)
        ]
        return require_torch().cat(channels, dim=1)
    return _tensor(
        estimate_vessel_map(image).resize((size, size), Image.Resampling.BILINEAR),
        device,
        grayscale=True,
    )


def _infer_laterality(image_path: str, supplied: str | None) -> str | None:
    if supplied:
        return supplied
    token = Path(image_path).stem.upper().rsplit("_", 1)[-1]
    return token if token in {"OD", "OS"} else None


def generate_direct_trajectory(
    image_path: str,
    checkpoint_path: str,
    output_dir: str,
    detail_checkpoint_path: str | None = None,
    years: list[float] | None = None,
    trajectories: int = 8,
    sampling_steps: int = 50,
    device: str = "auto",
    seed: int = 20260729,
    global_change_scale: float = 1.0,
    detail_change_scale: float = 1.0,
    clinical_condition: float | None = None,
    save_candidates: bool = False,
    interpolate_final_detail: bool = False,
    laterality: str | None = None,
) -> dict[str, object]:
    torch = require_torch()
    selected_device = select_device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=selected_device, weights_only=False
    )
    config = checkpoint["config"]
    clinical_dim = 1 if config.get("conditioning_manifest") else 0
    structural_feature_mode = str(config.get("structural_feature_mode", "vessel"))
    structural_channels = 3 if structural_feature_mode == "glaucoma" else 1
    model = build_residual_diffusion(
        int(config.get("base_channels", 32)),
        clinical_dim,
        structural_channels=structural_channels,
        progression_head=float(config.get("progression_loss_weight", 0.0)) > 0,
    ).to(
        selected_device
    )
    state = checkpoint.get("best_state") or checkpoint["model"]
    model.load_state_dict(state)
    image_size = int(config.get("image_size", 512))
    residual_size = int(config.get("residual_size", 256))
    max_change = float(config.get("max_change", 0.18))
    max_horizon = float(config.get("max_supported_horizon", 4.5))
    requested_years = years or [float(value) for value in range(1, 11)]
    if not requested_years or any(value <= 0 for value in requested_years):
        raise ValueError("All requested years must be positive")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    processed = preprocess_high_resolution(
        image_path,
        image_size,
        crop_fraction=float(config.get("disc_crop_fraction", 0.50)),
    )
    artifacts = save_high_resolution_artifacts(processed, output)
    if processed.quality.is_low_quality:
        raise ValueError(f"Input failed quality gate: {processed.quality.reason}")
    view_mode = str(config.get("view_mode", "full_field"))
    resolved_laterality = _infer_laterality(image_path, laterality)
    baseline_image = processed.optic_disc if view_mode == "optic_disc" else processed.full_field
    if view_mode == "optic_disc":
        baseline_image, _ = mirror_left_eye(baseline_image, resolved_laterality)
    baseline_full = _tensor(baseline_image, selected_device)
    baseline_low_image = baseline_image.resize(
        (residual_size, residual_size), Image.Resampling.LANCZOS
    )
    baseline_low = _tensor(baseline_low_image, selected_device)
    vessel_low = _structural_tensor(
        baseline_image,
        structural_feature_mode,
        residual_size,
        selected_device,
    )
    condition_tensor = None
    if clinical_dim:
        if clinical_condition is None:
            raise ValueError(
                "This checkpoint requires clinical_condition (0=healthy, 1=glaucoma)"
            )
        condition_tensor = torch.tensor(
            [[float(clinical_condition)]],
            device=selected_device,
            dtype=baseline_low.dtype,
        )
    detail_checkpoint_step = None
    detail_samples = None
    if detail_checkpoint_path:
        detail_checkpoint = torch.load(
            detail_checkpoint_path,
            map_location=selected_device,
            weights_only=False,
        )
        detail_config = detail_checkpoint["config"]
        if str(detail_config.get("view_mode")) != "optic_disc":
            raise ValueError("Detail checkpoint must be trained with view_mode=optic_disc")
        if int(detail_config.get("image_size", image_size)) != image_size:
            raise ValueError("Global and detail checkpoint image sizes must match")
        detail_residual_size = int(detail_config.get("residual_size", residual_size))
        detail_clinical_dim = 1 if detail_config.get("conditioning_manifest") else 0
        detail_model = build_residual_diffusion(
            int(detail_config.get("base_channels", 32)), detail_clinical_dim
        ).to(selected_device)
        detail_model.load_state_dict(
            detail_checkpoint.get("best_state") or detail_checkpoint["model"]
        )
        detail_baseline_full = _tensor(processed.optic_disc, selected_device)
        detail_baseline_low_image = processed.optic_disc.resize(
            (detail_residual_size, detail_residual_size), Image.Resampling.LANCZOS
        )
        detail_vessel_image = estimate_vessel_map(processed.optic_disc).resize(
            (detail_residual_size, detail_residual_size), Image.Resampling.BILINEAR
        )
        detail_condition_tensor = None
        if detail_clinical_dim:
            if clinical_condition is None:
                raise ValueError(
                    "The detail checkpoint requires clinical_condition (0=healthy, 1=glaucoma)"
                )
            detail_condition_tensor = torch.tensor(
                [[float(clinical_condition)]],
                device=selected_device,
                dtype=baseline_low.dtype,
            )
        combined_batch, detail_batch = sample_dual_scale_trajectory(
            model,
            detail_model,
            baseline_full,
            baseline_low,
            vessel_low,
            detail_baseline_full,
            _tensor(detail_baseline_low_image, selected_device),
            _tensor(detail_vessel_image, selected_device, grayscale=True),
            processed.disc_center_normalized,
            requested_years,
            max_horizon,
            max_change,
            trajectories=trajectories,
            diffusion_steps=sampling_steps,
            seed=seed,
            crop_fraction=float(detail_config.get("disc_crop_fraction", 0.50)),
            global_change_scale=global_change_scale,
            detail_change_scale=detail_change_scale,
            clinical_condition=condition_tensor,
            detail_clinical_condition=detail_condition_tensor,
            return_detail_samples=True,
            interpolate_final_detail=interpolate_final_detail,
        )
        samples = combined_batch[0]
        detail_samples = detail_batch[0]
        detail_checkpoint_step = detail_checkpoint.get("step")
    else:
        samples = sample_residual_trajectory(
            model,
            baseline_full,
            baseline_low,
            vessel_low,
            requested_years,
            max_horizon,
            max_change,
            trajectories=trajectories,
            diffusion_steps=sampling_steps,
            seed=seed,
            change_scale=global_change_scale,
            clinical_condition=condition_tensor,
        )[0]
    representative = samples.mean(dim=0)
    uncertainty = samples.std(dim=0, unbiased=False).mean(dim=1)
    image_paths = []
    uncertainty_paths = []
    contact_images = [baseline_image.copy()]
    for index, year in enumerate(requested_years):
        array = (
            representative[index]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .clip(0, 1)
            * 255
        ).astype(np.uint8)
        image = Image.fromarray(array)
        support = "SUPPORTED RANGE" if year <= max_horizon else "EXTRAPOLATION"
        path = output / f"trajectory_year_{year:g}.png"
        image.save(path)
        image_paths.append(str(path))
        labelled = image.copy()
        draw = ImageDraw.Draw(labelled)
        draw.rectangle((0, 0, labelled.width, 28), fill=(0, 0, 0))
        draw.text(
            (6, 7),
            f"YEAR {year:g} · SIMULATED · {support}",
            fill=(255, 255, 255),
        )
        contact_images.append(labelled)
        uncertainty_array = (
            uncertainty[index].detach().cpu().numpy().clip(0, 0.25) / 0.25 * 255
        ).astype(np.uint8)
        uncertainty_path = output / f"uncertainty_year_{year:g}.png"
        Image.fromarray(uncertainty_array).save(uncertainty_path)
        uncertainty_paths.append(str(uncertainty_path))

    candidate_sequences = []
    if save_candidates:
        for candidate_index, candidate in enumerate(samples, start=1):
            candidate_paths = []
            candidate_dir = output / "candidates" / f"candidate_{candidate_index:02d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for year_index, year in enumerate(requested_years):
                candidate_array = (
                    candidate[year_index]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    .clip(0, 1)
                    * 255
                ).astype(np.uint8)
                candidate_path = candidate_dir / f"year_{year:g}.png"
                Image.fromarray(candidate_array).save(candidate_path)
                candidate_paths.append(str(candidate_path))
            candidate_sequences.append(
                {
                    "candidate_id": f"candidate_{candidate_index:02d}",
                    "images": candidate_paths,
                }
            )

    columns = min(4, len(contact_images))
    rows = (len(contact_images) + columns - 1) // columns
    contact = Image.new(
        "RGB", (columns * image_size, rows * image_size), (0, 0, 0)
    )
    for index, image in enumerate(contact_images):
        contact.paste(
            image,
            ((index % columns) * image_size, (index // columns) * image_size),
        )
    contact_path = output / "trajectory_contact_sheet.png"
    contact.save(contact_path)
    detail_image_paths = []
    detail_contact_path = None
    if detail_samples is not None:
        representative_detail = detail_samples.mean(dim=0)
        detail_contact_images = [processed.optic_disc.copy()]
        for index, year in enumerate(requested_years):
            detail_array = (
                representative_detail[index]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .clip(0, 1)
                * 255
            ).astype(np.uint8)
            detail_image = Image.fromarray(detail_array)
            detail_path = output / f"disc_trajectory_year_{year:g}.png"
            detail_image.save(detail_path)
            detail_image_paths.append(str(detail_path))
            detail_contact_images.append(detail_image)
        detail_contact = Image.new(
            "RGB", (len(detail_contact_images) * image_size, image_size), (0, 0, 0)
        )
        for index, image in enumerate(detail_contact_images):
            detail_contact.paste(image, (index * image_size, 0))
        detail_contact_path = output / "disc_trajectory_contact_sheet.png"
        detail_contact.save(detail_contact_path)
    report = {
        "schema_version": "3.0",
        "model": (
            "dual_scale_registered_residual_diffusion_v4"
            if detail_checkpoint_path
            else "registered_residual_diffusion_v3"
        ),
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint.get("step"),
        "detail_checkpoint": detail_checkpoint_path,
        "detail_checkpoint_step": detail_checkpoint_step,
        "optic_disc_detail_branch_used": bool(detail_checkpoint_path),
        "global_change_scale": global_change_scale,
        "detail_change_scale": (
            detail_change_scale if detail_checkpoint_path else None
        ),
        "clinical_condition": clinical_condition,
        "view_mode": view_mode,
        "structural_feature_mode": structural_feature_mode,
        "laterality": resolved_laterality,
        "disclaimer": (
            "Research-only simulated possibilities. These are not diagnoses, "
            "patient-specific forecasts, or evidence that glaucoma will progress."
        ),
        "scope": "progression_scenarios_for_eyes_already_diagnosed_with_glaucoma",
        "direct_baseline_conditioning": True,
        "recursive_rollout_used": False,
        "observed_training_horizon_max_years": max_horizon,
        "unsupported_years": [
            year for year in requested_years if year > max_horizon
        ],
        "requested_years": requested_years,
        "num_trajectories": trajectories,
        "sampling_steps": sampling_steps,
        "final_detail_interpolation_used": interpolate_final_detail,
        "input_artifacts": artifacts,
        "disc_center_normalized": list(processed.disc_center_normalized),
        "representative_images": image_paths,
        "representative_disc_images": detail_image_paths,
        "candidate_sequences": candidate_sequences,
        "uncertainty_maps": uncertainty_paths,
        "contact_sheet": str(contact_path),
        "disc_contact_sheet": str(detail_contact_path) if detail_contact_path else None,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
