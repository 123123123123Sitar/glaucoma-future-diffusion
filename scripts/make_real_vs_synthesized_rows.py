#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.preprocessing import optic_disc_crop
from glaucoma_forecast.evaluation.anatomy_metrics import vcdr_from_masks
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.disc_cup_segmenter import build_disc_cup_segmenter
from glaucoma_forecast.utils.reproducibility import select_device


BLUE = (30, 144, 255)
BACKGROUND = (248, 250, 253)
TEXT = (20, 31, 48)
TILE = 360


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def official_overlay(image_path: Path, annotation_path: Path) -> tuple[Image.Image, float]:
    image = Image.open(image_path).convert("RGB")
    annotation = json.loads(annotation_path.read_text())
    shapes = {shape["label"]: shape["points"] for shape in annotation["shapes"]}
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    cup_points = [tuple(point) for point in shapes["OC"]]
    draw.line(cup_points + [cup_points[0]], fill=BLUE, width=2, joint="curve")
    disc_mask = Image.new("1", image.size)
    cup_mask = Image.new("1", image.size)
    ImageDraw.Draw(disc_mask).polygon(shapes["OD"], fill=1)
    ImageDraw.Draw(cup_mask).polygon(shapes["OC"], fill=1)
    vcdr = vcdr_from_masks(np.asarray(cup_mask, dtype=bool), np.asarray(disc_mask, dtype=bool))
    return ImageOps.fit(canvas, (TILE, TILE), method=Image.Resampling.LANCZOS), vcdr


def tensor(image: Image.Image, device: str):
    torch = require_torch()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).unsqueeze(0).to(device)


def predicted_overlay(model, image: Image.Image, device: str) -> tuple[Image.Image, float]:
    torch = require_torch()
    model_input = image.resize((256, 256), Image.Resampling.LANCZOS)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor(model_input, device)))[0].cpu().numpy()
    disc = probability[0] >= 0.5
    cup = (probability[1] >= 0.5) & disc
    cup = clean_cup_mask(cup)
    vcdr = vcdr_from_masks(cup, disc)
    edge = cup ^ (
        np.roll(cup, 1, 0)
        & np.roll(cup, -1, 0)
        & np.roll(cup, 1, 1)
        & np.roll(cup, -1, 1)
    )
    overlay = np.asarray(model_input).copy()
    overlay[edge] = BLUE
    return Image.fromarray(overlay).resize((TILE, TILE), Image.Resampling.LANCZOS), vcdr


def largest_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        if len(component) > len(best):
            best = component
    output = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        output[y, x] = True
    return output


def clean_cup_mask(mask: np.ndarray) -> np.ndarray:
    opened = mask.copy()
    for _ in range(3):
        opened = (
            opened
            & np.roll(opened, 1, 0)
            & np.roll(opened, -1, 0)
            & np.roll(opened, 1, 1)
            & np.roll(opened, -1, 1)
        )
    opened = largest_component(opened)
    restored = opened.copy()
    for _ in range(3):
        restored = (
            restored
            | np.roll(restored, 1, 0)
            | np.roll(restored, -1, 0)
            | np.roll(restored, 1, 1)
            | np.roll(restored, -1, 1)
        )
    return restored & mask


def place_tile(sheet: Image.Image, image: Image.Image, x: int, y: int, label: str, vcdr: float) -> None:
    sheet.paste(image, (x, y))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((x, y + TILE, x + TILE, y + TILE + 54), fill=(255, 255, 255))
    draw.text((x + 10, y + TILE + 7), label, fill=TEXT, font=font(19, True))
    draw.text((x + 10, y + TILE + 30), f"Outlined VCDR: {vcdr:.3f}", fill=(66, 78, 96), font=font(15))


def main() -> int:
    case_dir = Path("outputs/progression/heldout_20_eyes_20260812/GRAPE_60_GRAPE_60_OS_same_baseline")
    report = json.loads((case_dir / "report.json").read_text())
    device = select_device("mps")
    torch = require_torch()
    checkpoint = torch.load(
        "outputs/training/papila_disc_cup_segmenter_v3_weighted/best.pt",
        map_location=device,
        weights_only=False,
    )
    model = build_disc_cup_segmenter(int(checkpoint.get("config", {}).get("base_channels", 24))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    real = [
        official_overlay(
            Path("data/external/grape_official/roi/60_OS_1.jpg"),
            Path("data/external/grape_official/json/60_OS_1.json"),
        ),
        official_overlay(
            Path("data/external/grape_official/roi/60_OS_4.jpg"),
            Path("data/external/grape_official/json/60_OS_4.json"),
        ),
    ]

    center = tuple(float(value) for value in report["disc_center_normalized"])
    synthesized_paths = [Path(path) for path in report["representative_images"]]
    synthesized = [real[0]]
    for path in synthesized_paths:
        with Image.open(path) as source:
            pixel_center = (round(center[0] * source.width), round(center[1] * source.height))
            crop = optic_disc_crop(source.convert("RGB"), pixel_center, 0.30, 256)
        synthesized.append(predicted_overlay(model, crop, device))

    margin = 28
    label_width = 225
    gap = 18
    row_header = 46
    row_height = row_header + TILE + 54
    width = margin * 2 + label_width + 4 * TILE + 3 * gap
    height = margin * 2 + 72 + 2 * row_height + 26
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), "Real vs. Synthesized Glaucoma Progression", fill=TEXT, font=font(30, True))
    draw.text(
        (margin, margin + 40),
        "Thin blue line = optic-cup boundary; it is an overlay, not part of the photograph.",
        fill=(66, 78, 96),
        font=font(17),
    )

    first_y = margin + 72
    second_y = first_y + row_height + 26
    draw.rounded_rectangle((margin, first_y, margin + label_width - 16, first_y + row_height - 8), 14, fill=(226, 239, 255))
    draw.text((margin + 18, first_y + 26), "REAL\nOBSERVED\nPROGRESSION", fill=(15, 86, 170), font=font(23, True), spacing=8)
    draw.text((margin + 18, first_y + 142), "Official GRAPE\nexpert outlines", fill=(50, 71, 99), font=font(16), spacing=5)
    draw.rounded_rectangle((margin, second_y, margin + label_width - 16, second_y + row_height - 8), 14, fill=(231, 246, 238))
    draw.text((margin + 18, second_y + 26), "SYNTHESIZED\nPROJECTION", fill=(20, 116, 67), font=font(23, True), spacing=8)
    draw.text((margin + 18, second_y + 115), "Model-generated images\nEstimated outlines", fill=(50, 71, 99), font=font(16), spacing=5)

    start_x = margin + label_width
    draw.text((start_x, first_y + 10), "Same held-out eye: actual photographs", fill=TEXT, font=font(20, True))
    real_start = start_x + (2 * TILE + gap) // 2
    place_tile(sheet, real[0][0], real_start, first_y + row_header, "Baseline (real)", real[0][1])
    place_tile(sheet, real[1][0], real_start + TILE + gap, first_y + row_header, "3.56 years (real)", real[1][1])

    draw.text((start_x, second_y + 10), "Identical baseline tile and outline; model projections follow", fill=TEXT, font=font(20, True))
    labels = ["Baseline", "Year 1", "Year 3", "Year 5*"]
    for index, ((image, vcdr), label) in enumerate(zip(synthesized, labels)):
        place_tile(sheet, image, start_x + index * (TILE + gap), second_y + row_header, label, vcdr)

    draw.text(
        (margin, height - 24),
        "*Year 5 is a slight extrapolation beyond the 4.84-year training horizon. Research-only simulation; not a patient-specific forecast.",
        fill=(101, 61, 15),
        font=font(14),
    )
    output = case_dir / "real_vs_synthesized_blue_outline_rows.png"
    sheet.save(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
