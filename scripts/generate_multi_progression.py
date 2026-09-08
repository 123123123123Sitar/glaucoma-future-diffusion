#!/usr/bin/env python3
"""Generate a multi-input progression sheet with intensity differences below."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from generate_diffusion_forecast import change_forecast, load_context, tensor_to_image
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.training.grape_glaucoma_trainer import DISCLAIMER, build_compact_denoiser
from glaucoma_forecast.utils.reproducibility import select_device


def _difference_heatmap(source: Image.Image, forecast: Image.Image, scale: float) -> tuple[Image.Image, float]:
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32)
    forecast_array = np.asarray(forecast.convert("RGB"), dtype=np.float32)
    difference = np.mean(np.abs(forecast_array - source_array), axis=2)
    scaled = np.clip(difference * scale, 0, 255).astype(np.uint8)
    heatmap = np.stack([scaled, np.zeros_like(scaled), 255 - scaled], axis=2)
    return Image.fromarray(heatmap), float(difference.mean())


def _labelled_tile(image: Image.Image, label: str, tile_size: int) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size + 30), (0, 0, 0))
    canvas.paste(image.resize((tile_size, tile_size), Image.Resampling.LANCZOS), (0, 30))
    ImageDraw.Draw(canvas).text((7, 8), label, fill=(255, 255, 255))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs=3, required=True)
    parser.add_argument("--input-modes", nargs=3, choices=["disc_crop", "full_field"], required=True)
    parser.add_argument("--lateralities", nargs=3, default=["OD", "OD", "OD"])
    parser.add_argument(
        "--disc-centers",
        nargs=3,
        default=["auto", "auto", "auto"],
        help="Normalized x,y centers after laterality normalization, or 'auto'.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", nargs=3, type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--detail-preservation", type=float, default=0.5)
    parser.add_argument("--difference-scale", type=float, default=20.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch = require_torch()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    max_change = float(config.get("max_change", 0.18))
    model = build_compact_denoiser(int(config.get("base_channels", 32))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progressive_rows: list[list[Image.Image]] = []
    difference_rows: list[list[Image.Image]] = []
    records: list[dict[str, object]] = []

    for index, (image_path, input_mode, laterality, center_value) in enumerate(
        zip(args.images, args.input_modes, args.lateralities, args.disc_centers), start=1
    ):
        disc_center = None
        if center_value.lower() != "auto":
            pieces = center_value.split(",")
            if len(pieces) != 2:
                raise ValueError("disc centers must be 'auto' or normalized x,y")
            disc_center = (float(pieces[0]), float(pieces[1]))
        context, input_image = load_context(
            Path(image_path), args.image_size, device, input_mode, laterality, disc_center
        )
        input_image.save(output_dir / f"screenshot_{index}_input.png")
        forecasts: list[Image.Image] = []
        mean_differences: list[float] = []
        difference_maps: list[Image.Image] = []
        for horizon in args.horizons:
            prediction = change_forecast(
                model,
                context,
                horizon,
                device,
                max_change,
                args.detail_preservation,
            )
            forecast = tensor_to_image(prediction)
            forecast.save(output_dir / f"screenshot_{index}_{horizon:g}y.png")
            heatmap, mean_difference = _difference_heatmap(input_image, forecast, args.difference_scale)
            heatmap.save(output_dir / f"screenshot_{index}_{horizon:g}y_intensity_difference.png")
            forecasts.append(forecast)
            difference_maps.append(heatmap)
            mean_differences.append(mean_difference)

        progressive_rows.append(
            [_labelled_tile(input_image, f"Screenshot {index}: input", args.tile_size)]
            + [_labelled_tile(image, f"{horizon:g}-year forecast", args.tile_size) for horizon, image in zip(args.horizons, forecasts)]
        )
        grayscale = input_image.convert("L").convert("RGB")
        difference_rows.append(
            [_labelled_tile(grayscale, f"Screenshot {index}: baseline intensity", args.tile_size)]
            + [
                _labelled_tile(image, f"|delta {horizon:g}y| x{args.difference_scale:g}; mean={mean:.2f}/255", args.tile_size)
                for horizon, image, mean in zip(args.horizons, difference_maps, mean_differences)
            ]
        )
        records.append(
            {
                "screenshot": index,
                "input_path": image_path,
                "input_mode": input_mode,
                "laterality": laterality,
                "disc_center_normalized_after_orientation": disc_center,
                "mean_absolute_intensity_difference_8bit": dict(zip([str(value) for value in args.horizons], mean_differences)),
            }
        )

    rows = progressive_rows + difference_rows
    row_height = args.tile_size + 30
    sheet = Image.new("RGB", (args.tile_size * 4, row_height * len(rows)), (0, 0, 0))
    for row_index, row in enumerate(rows):
        for column_index, tile in enumerate(row):
            sheet.paste(tile, (column_index * args.tile_size, row_index * row_height))
    sheet_path = output_dir / "three_screenshot_progression_with_intensity_differences.png"
    sheet.save(sheet_path)

    report = {
        "disclaimer": DISCLAIMER,
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "sheet": str(sheet_path),
        "horizons_years": args.horizons,
        "difference_map_scale": args.difference_scale,
        "difference_map_note": "Only the bottom intensity-difference maps are scaled. Forecast images are not amplified.",
        "inputs": records,
        "limitations": [
            "Single screenshots are outside the preferred multi-visit input setting.",
            "These sources may differ from GRAPE camera and image distributions.",
            "The model's validated improvement over persistence is small and not clinically established.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
