#!/usr/bin/env python3
"""Generate research-only future fundus variants with the compact GRAPE model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import _bootstrap  # noqa: F401
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.data.preprocessing import optic_disc_crop, preprocess_image
from glaucoma_forecast.training.grape_glaucoma_trainer import build_compact_denoiser
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


DISCLAIMER = "Research prototype only. Not validated for patient care, diagnosis, triage, or treatment decisions."


def load_context(
    path: Path,
    image_size: int,
    device: str,
    input_mode: str,
    laterality: str,
    disc_center_normalized: tuple[float, float] | None = None,
):
    torch = require_torch()
    if input_mode == "full_field":
        full = preprocess_image(path, output_size=256, laterality=laterality, mirror_left=True).image
        center = None
        if disc_center_normalized is not None:
            center = (
                int(round(disc_center_normalized[0] * full.width)),
                int(round(disc_center_normalized[1] * full.height)),
            )
        image = optic_disc_crop(full, center=center, output_size=image_size)
    else:
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("RGB"), (image_size, image_size), method=Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device), image


def tensor_to_image(tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray((arr * 255).astype(np.uint8))


def deterministic_denoiser_edit(model, context, horizon_years: float, device: str, edit_scale: float):
    torch = require_torch()
    horizon = torch.tensor([min(max(horizon_years, 0.0), 5.0) / 5.0], dtype=torch.float32, device=device)
    timestep = torch.zeros(1, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        residual = model(context, context, horizon, timestep)
        horizon_scale = min(max(horizon_years, 0.0), 3.0) / 3.0
        edited = context - residual * edit_scale * horizon_scale
    return edited[0]


def change_forecast(
    model,
    context,
    horizon_years: float,
    device: str,
    max_change: float,
    detail_preservation: float,
):
    torch = require_torch()
    horizon = torch.tensor([min(max(horizon_years, 0.0), 5.0) / 5.0], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        prediction = model.predict_future(context, horizon, max_change=max_change)
        if detail_preservation > 0:
            pool = torch.nn.functional.avg_pool2d
            context_high = context - pool(context, kernel_size=9, stride=1, padding=4)
            prediction_high = prediction - pool(prediction, kernel_size=9, stride=1, padding=4)
            prediction = torch.clamp(
                prediction + detail_preservation * (context_high - prediction_high),
                0.0,
                1.0,
            )
        return prediction[0]


def sample_future(
    model,
    context,
    horizon_years: float,
    device: str,
    diffusion_steps: int,
    seed: int,
    init_strength: float = 0.35,
):
    torch = require_torch()
    seed_everything(seed)
    betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    start_idx = max(1, min(diffusion_steps - 1, int(diffusion_steps * init_strength)))
    start_alpha_bar = alpha_bars[start_idx]
    x = torch.sqrt(start_alpha_bar) * context + torch.sqrt(1.0 - start_alpha_bar) * torch.randn_like(context)
    horizon = torch.tensor([min(max(horizon_years, 0.0), 5.0) / 5.0], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for idx in reversed(range(start_idx + 1)):
            t_idx = torch.tensor([idx], dtype=torch.long, device=device)
            t = t_idx.float() / max(1, diffusion_steps - 1)
            pred_noise = model(x, context, horizon, t)
            alpha = alphas[idx]
            alpha_bar = alpha_bars[idx]
            beta = betas[idx]
            x0 = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            if idx > 0:
                prev_alpha_bar = alpha_bars[idx - 1]
                direction = torch.sqrt(1.0 - prev_alpha_bar) * pred_noise
                x = torch.sqrt(prev_alpha_bar) * x0 + direction
                # small stochastic term; deterministic enough with seed, but keeps samples varied
                x = x + torch.sqrt(beta) * 0.15 * torch.randn_like(x)
            else:
                x = x0
    return x[0]


def make_contact_sheet(input_image: Image.Image, generated: list[tuple[float, Image.Image]], output: Path) -> None:
    tile = input_image.size[0]
    labels = ["input"] + [f"{h:g}y" for h, _ in generated]
    images = [input_image] + [img.resize((tile, tile), Image.Resampling.BICUBIC) for _, img in generated]
    sheet = Image.new("RGB", (tile * len(images), tile + 28), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(zip(labels, images)):
        sheet.paste(image, (i * tile, 28))
        draw.text((i * tile + 6, 6), label, fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def make_change_visualizations(
    input_image: Image.Image,
    generated: list[tuple[float, Image.Image]],
    output_dir: Path,
    amplification: float = 5.0,
) -> tuple[Path, Path]:
    """Save explicitly labeled amplified residuals and absolute-difference maps."""

    tile = max(256, input_image.width)
    display_input = input_image.resize((tile, tile), Image.Resampling.LANCZOS)
    source = np.asarray(display_input.convert("RGB"), dtype=np.float32)
    amplified_sheet = Image.new("RGB", (tile * len(generated), tile + 40), (0, 0, 0))
    difference_sheet = Image.new("RGB", (tile * len(generated), tile + 40), (0, 0, 0))
    amplified_draw = ImageDraw.Draw(amplified_sheet)
    difference_draw = ImageDraw.Draw(difference_sheet)
    for index, (horizon, image) in enumerate(generated):
        current = np.asarray(image.resize((tile, tile), Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32)
        residual = current - source
        amplified = np.clip(source + amplification * residual, 0, 255).astype(np.uint8)
        magnitude = np.mean(np.abs(residual), axis=2)
        scaled = np.clip(magnitude * 20.0, 0, 255).astype(np.uint8)
        heatmap = np.stack([scaled, np.zeros_like(scaled), 255 - scaled], axis=2)
        x = index * tile
        amplified_sheet.paste(Image.fromarray(amplified), (x, 40))
        difference_sheet.paste(Image.fromarray(heatmap), (x, 40))
        amplified_draw.text((x + 5, 5), f"{horizon:g}y: {amplification:g}x change (NOT forecast)", fill=(255, 255, 255))
        difference_draw.text((x + 5, 5), f"{horizon:g}y absolute change map", fill=(255, 255, 255))
    amplified_path = output_dir / "amplified_change_visualization_not_forecast.png"
    difference_path = output_dir / "absolute_difference_maps.png"
    amplified_sheet.save(amplified_path)
    difference_sheet.save(difference_path)
    return amplified_path, difference_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compact diffusion research variants from one fundus image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="outputs/training/grape_registered_disc_mps_1000/best.pt")
    parser.add_argument("--output-dir", default="outputs/diffusion_forecast/latest_screenshot")
    parser.add_argument("--horizons", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--init-strength", type=float, default=0.35, help="Image-to-image noise strength in [0, 1].")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--mode", choices=["change_forecast", "img2img", "deterministic_edit"], default="change_forecast")
    parser.add_argument("--edit-scale", type=float, default=0.18)
    parser.add_argument("--input-mode", choices=["disc_crop", "full_field"], default="disc_crop")
    parser.add_argument("--laterality", default="OD", help="OD/OS; used only for full-field input.")
    parser.add_argument(
        "--detail-preservation",
        type=float,
        default=0.5,
        help="Fraction of input high-frequency vessel detail retained in change_forecast mode.",
    )
    args = parser.parse_args()

    torch = require_torch()
    device = select_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    base_channels = int(checkpoint_config.get("base_channels", 32))
    max_change = float(checkpoint_config.get("max_change", 0.18))
    model = build_compact_denoiser(base_channels).to(device)
    model.load_state_dict(checkpoint["model"])

    context, context_image = load_context(Path(args.image), args.image_size, device, args.input_mode, args.laterality)
    context_image.save(out / "input_preprocessed.png")

    generated: list[tuple[float, Image.Image]] = []
    sample_paths: list[str] = []
    for i, horizon in enumerate(args.horizons):
        if args.mode == "change_forecast":
            sample = change_forecast(model, context, horizon, device, max_change, args.detail_preservation)
        elif args.mode == "deterministic_edit":
            sample = deterministic_denoiser_edit(model, context, horizon, device, args.edit_scale)
        else:
            sample = sample_future(model, context, horizon, device, args.diffusion_steps, args.seed + i, args.init_strength)
        image = tensor_to_image(sample)
        path = out / f"diffusion_horizon_{horizon:g}y.png"
        image.save(path)
        generated.append((horizon, image))
        sample_paths.append(str(path))

    contact = out / "contact_sheet.png"
    make_contact_sheet(context_image, generated, contact)
    amplified_path, difference_path = make_change_visualizations(context_image, generated, out)
    report = {
        "disclaimer": DISCLAIMER,
        "input_image": str(Path(args.image)),
        "checkpoint": args.checkpoint,
        "checkpoint_note": f"Best held-out persistence-comparison checkpoint at training step {checkpoint.get('step')}.",
        "device": device,
        "init_strength": args.init_strength,
        "mode": args.mode,
        "edit_scale": args.edit_scale,
        "max_change": max_change,
        "input_mode": args.input_mode,
        "detail_preservation": args.detail_preservation,
        "horizons_years": args.horizons,
        "samples": sample_paths,
        "contact_sheet": str(contact),
        "amplified_change_visualization_not_forecast": str(amplified_path),
        "absolute_difference_maps": str(difference_path),
        "limitations": [
            "Generated with a compact GRAPE glaucoma-eye prototype.",
            "Not trained on healthy-control to glaucoma conversion.",
            "Single-image input has limited patient-specific longitudinal information.",
            "Watermarks, screenshots, or non-GRAPE camera sources are outside the model training distribution.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
