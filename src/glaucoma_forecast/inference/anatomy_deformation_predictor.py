"""Native-resolution rendering and release metrics for anatomy deformation checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from glaucoma_forecast.evaluation.anatomy_metrics import vcdr_from_masks
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.anatomy_deformation import (
    build_anatomy_deformation_model,
    displacement_jacobian,
    soft_anatomy_envelope,
    warp_tensor,
)
from glaucoma_forecast.utils.reproducibility import select_device


DISCLAIMER = "Research-only simulated possibility. Not a diagnosis or patient-specific prognosis."


def _image_tensor(image: Image.Image, size: int | None = None):
    torch = require_torch()
    source = image.convert("RGB")
    if size is not None:
        source = source.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(source, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).unsqueeze(0)


def _mask_tensor(path: str, size: int):
    torch = require_torch()
    with Image.open(path) as source:
        mask = source.convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = (np.asarray(mask) > 0).astype(np.float32)
    return torch.from_numpy(array[None, None].copy())


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.sum() + second.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(first, second).sum() / denominator)


def _to_image(tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1).cpu().numpy()[0]
    return Image.fromarray((np.transpose(array, (1, 2, 0)) * 255).round().astype(np.uint8))


def automatic_progression_strength(frame: pd.DataFrame) -> float:
    development = frame[
        (frame["split"] != "test") & (frame["progression_label"] == "progressor")
    ]["delta_vcdr"].astype(float)
    if development.empty:
        raise ValueError("No development progressors are available for automatic strength")
    return float(np.clip(development.max(), 0.05, 0.12))


def _validation_calibration(model, frame: pd.DataFrame, image_size: int, device: str) -> float:
    torch = require_torch()
    ratios = []
    validation = frame[
        (frame["split"] == "validation") & (frame["progression_label"] == "progressor")
    ]
    with torch.no_grad():
        for _, row in validation.iterrows():
            target = float(row["delta_vcdr"])
            with Image.open(str(row["baseline_image_path"])) as source:
                baseline = _image_tensor(source.convert("RGB"), image_size).to(device)
            disc = _mask_tensor(str(row["baseline_disc_mask_path"]), image_size).to(device)
            cup = _mask_tensor(str(row["baseline_cup_mask_path"]), image_size).to(device)
            result = model(
                baseline,
                disc,
                cup,
                torch.tensor([float(row["horizon_years"]) / 5.0], device=device),
                torch.tensor([target], device=device),
            )
            baseline_disc = disc[0, 0].cpu().numpy() >= 0.5
            baseline_cup = cup[0, 0].cpu().numpy() >= 0.5
            generated_disc = result["warped_disc"][0, 0].cpu().numpy() >= 0.5
            generated_cup = result["warped_cup"][0, 0].cpu().numpy() >= 0.5
            generated_delta = (
                vcdr_from_masks(generated_cup, generated_disc)
                - vcdr_from_masks(baseline_cup, baseline_disc)
            )
            if generated_delta > 0.01:
                ratios.append(target / generated_delta)
    return 1.0 if not ratios else float(np.clip(np.median(ratios), 1.0, 1.6))


def generate_anatomy_deformation(
    checkpoint_path: str,
    manifest_path: str,
    output_dir: str,
    annotation_id: str | None = None,
    horizon_years: float = 5.0,
    progression_strength: float | None = None,
    device: str = "auto",
) -> dict:
    torch = require_torch()
    resolved_device = select_device(device)
    frame = pd.read_csv(manifest_path)
    if annotation_id is None:
        report_path = Path(manifest_path).with_name("report.json")
        report = json.loads(report_path.read_text())
        annotation_id = report["recommended_demo_annotation_id"]
    selected = frame[frame["annotation_id"].astype(str) == str(annotation_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected one manifest row for {annotation_id}")
    row = selected.iloc[0]
    if str(row["split"]) != "test":
        raise ValueError("Final visual demo must use a held-out test eye")
    target_strength = automatic_progression_strength(frame) if progression_strength is None else float(progression_strength)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config = checkpoint["config"]
    image_size = int(config["image_size"])
    model = build_anatomy_deformation_model(
        int(config["base_channels"]), float(config["max_displacement"])
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    calibration = _validation_calibration(model, frame, image_size, resolved_device)
    conditioning_strength = min(0.15, target_strength * calibration)
    with Image.open(str(row["baseline_image_path"])) as source:
        native_image = source.convert("RGB")
    baseline_low = _image_tensor(native_image, image_size).to(resolved_device)
    baseline_disc = _mask_tensor(str(row["baseline_disc_mask_path"]), image_size).to(resolved_device)
    baseline_cup = _mask_tensor(str(row["baseline_cup_mask_path"]), image_size).to(resolved_device)
    horizon = torch.tensor([horizon_years / 5.0], dtype=torch.float32, device=resolved_device)
    strength_tensor = torch.tensor([conditioning_strength], dtype=torch.float32, device=resolved_device)
    with torch.no_grad():
        result = model(baseline_low, baseline_disc, baseline_cup, horizon, strength_tensor)
        native = _image_tensor(native_image).to(resolved_device)
        native_height, native_width = native.shape[-2:]
        displacement = torch.nn.functional.interpolate(
            result["displacement"], size=(native_height, native_width), mode="bilinear", align_corners=False
        )
        displacement[:, 0] *= native_width / image_size
        displacement[:, 1] *= native_height / image_size
        warped_native = warp_tensor(native, displacement)
        residual = torch.nn.functional.interpolate(
            result["appearance_residual"], size=(native_height, native_width), mode="bilinear", align_corners=False
        )
        envelope = torch.nn.functional.interpolate(
            soft_anatomy_envelope(baseline_disc),
            size=(native_height, native_width), mode="bilinear", align_corners=False,
        )
        generated_native = (warped_native + residual * envelope).clamp(0, 1)

    warped_disc = result["warped_disc"][0, 0].detach().cpu().numpy() >= 0.5
    warped_cup = result["warped_cup"][0, 0].detach().cpu().numpy() >= 0.5
    baseline_disc_np = baseline_disc[0, 0].detach().cpu().numpy() >= 0.5
    baseline_cup_np = baseline_cup[0, 0].detach().cpu().numpy() >= 0.5
    baseline_vcdr = vcdr_from_masks(baseline_cup_np, baseline_disc_np)
    generated_vcdr = vcdr_from_masks(warped_cup, warped_disc)
    jacobian = displacement_jacobian(result["displacement"]).detach().cpu().numpy()
    outside = ~np.asarray(
        Image.fromarray(baseline_disc_np.astype(np.uint8) * 255).resize(
            native_image.size, Image.Resampling.NEAREST
        ), dtype=np.uint8
    ).astype(bool)
    original_array = np.asarray(native_image, dtype=np.float32) / 255.0
    generated_array = np.asarray(_to_image(generated_native), dtype=np.float32) / 255.0
    outside_mae = float(np.abs(original_array - generated_array)[outside].mean())
    metrics = {
        "baseline_vcdr": baseline_vcdr,
        "generated_vcdr": generated_vcdr,
        "generated_delta_vcdr": generated_vcdr - baseline_vcdr,
        "requested_delta_vcdr": target_strength,
        "target_recovery_fraction": (generated_vcdr - baseline_vcdr) / max(target_strength, 1e-6),
        "outer_disc_dice": _dice(baseline_disc_np, warped_disc),
        "outside_disc_mae": outside_mae,
        "positive_jacobian_fraction": float((jacobian > 0).mean()),
        "cup_identity_dice": _dice(baseline_cup_np, warped_cup),
    }
    gates = {
        "visible_vcdr_change": metrics["generated_delta_vcdr"] >= 0.05,
        "target_recovery": 0.75 <= metrics["target_recovery_fraction"] <= 1.25,
        "outer_disc_preserved": metrics["outer_disc_dice"] >= 0.95,
        "outside_texture_preserved": metrics["outside_disc_mae"] <= 0.03,
        "deformation_topology": metrics["positive_jacobian_fraction"] >= 0.995,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original_path = output / "original.png"
    generated_path = output / "generated_5_year.png"
    native_image.save(original_path)
    _to_image(generated_native).save(generated_path)
    gap = 12
    comparison = Image.new("RGB", (native_image.width * 2 + gap, native_image.height), "white")
    comparison.paste(native_image, (0, 0))
    comparison.paste(_to_image(generated_native), (native_image.width + gap, 0))
    comparison_path = output / "original_then_generated.png"
    comparison.save(comparison_path)
    report = {
        "schema_version": "1.0",
        "architecture": checkpoint["architecture"],
        "checkpoint": checkpoint_path,
        "checkpoint_step": int(checkpoint["step"]),
        "annotation_id": str(annotation_id),
        "held_out_split": str(row["split"]),
        "horizon_years": horizon_years,
        "target_progression_strength": target_strength,
        "validation_calibration_factor": calibration,
        "model_conditioning_strength": conditioning_strength,
        "future_image_used_for_generation": False,
        "native_resolution_rendering": True,
        "metrics": metrics,
        "gates": gates,
        "all_automatic_gates_pass": all(gates.values()),
        "original": str(original_path),
        "generated": str(generated_path),
        "comparison": str(comparison_path),
        "disclaimer": DISCLAIMER,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
