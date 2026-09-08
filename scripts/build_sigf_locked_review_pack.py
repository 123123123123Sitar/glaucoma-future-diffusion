#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.highres import (
    estimate_glaucoma_structural_maps,
    estimate_vessel_map,
    preprocess_high_resolution,
)
from glaucoma_forecast.data.preprocessing import (
    match_color_statistics,
    mirror_left_eye,
    optic_disc_crop,
)
from glaucoma_forecast.evaluation.anatomy_metrics import vcdr_from_masks
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.disc_cup_segmenter import build_disc_cup_segmenter
from glaucoma_forecast.models.residual_diffusion import (
    build_residual_diffusion,
    sample_residual_trajectory,
)
from glaucoma_forecast.utils.reproducibility import select_device


BLUE = (20, 120, 255)
TEXT = (19, 29, 45)
MUTED = (75, 88, 108)
TILE = 300
TARGET_YEARS = (1.0, 3.0, 5.0)


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tensor(image: Image.Image, device: str, grayscale: bool = False):
    torch = require_torch()
    array = np.asarray(image.convert("L" if grayscale else "RGB"), dtype=np.float32) / 255.0
    if grayscale:
        array = array[None]
    else:
        array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array.copy()).unsqueeze(0).to(device)


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
            for next_y, next_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
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


def mask_edge(mask: np.ndarray) -> np.ndarray:
    interior = (
        mask
        & np.roll(mask, 1, 0)
        & np.roll(mask, -1, 0)
        & np.roll(mask, 1, 1)
        & np.roll(mask, -1, 1)
    )
    return mask ^ interior


def fitted_ellipse(mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(largest_component(mask))
    if len(coordinates) < 10:
        return np.zeros_like(mask, dtype=bool)
    y0, x0 = np.quantile(coordinates, 0.02, axis=0)
    y1, x1 = np.quantile(coordinates, 0.98, axis=0)
    center_y, center_x = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    radius_y, radius_x = max((y1 - y0) / 2.0, 1.0), max((x1 - x0) / 2.0, 1.0)
    grid_y, grid_x = np.ogrid[: mask.shape[0], : mask.shape[1]]
    return (
        ((grid_y - center_y) / radius_y) ** 2
        + ((grid_x - center_x) / radius_x) ** 2
        <= 1.0
    )


def papila_style_contour_mask(mask: np.ndarray) -> np.ndarray:
    """Clean a PAPILA-segmenter component while retaining its natural contour."""

    component = largest_component(mask)
    if component.sum() < 10:
        return component
    image = Image.fromarray((component * 255).astype(np.uint8))
    image = image.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    image = image.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
    return largest_component(np.asarray(image, dtype=np.uint8) >= 128)


def outlined_crop(
    segmenter,
    image: Image.Image,
    center,
    device: str,
    already_disc_crop=False,
    segmentation_reference: Image.Image | None = None,
):
    torch = require_torch()
    if already_disc_crop:
        crop = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    else:
        pixel_center = (round(center[0] * image.width), round(center[1] * image.height))
        crop = optic_disc_crop(image.convert("RGB"), pixel_center, 0.30, 256)
    normalized = (
        match_color_statistics(crop, segmentation_reference.resize(crop.size))
        if segmentation_reference is not None
        else crop
    )
    with torch.no_grad():
        original_probability = torch.sigmoid(segmenter(tensor(crop, device)))[0]
        normalized_probability = torch.sigmoid(segmenter(tensor(normalized, device)))[0]
        probability = ((original_probability + normalized_probability) / 2.0).cpu().numpy()
    disc = papila_style_contour_mask(probability[0] >= 0.5)
    cup = papila_style_contour_mask((probability[1] >= 0.5) & disc) & disc
    overlay = np.asarray(crop).copy()
    overlay[mask_edge(cup)] = BLUE
    vcdr = vcdr_from_masks(cup, disc) if disc.any() and cup.any() else float("nan")
    return Image.fromarray(overlay).resize((TILE, TILE), Image.Resampling.LANCZOS), vcdr


def select_cases(manifest: pd.DataFrame, count: int):
    candidates = []
    for eye_id, visits in manifest[manifest["split"].eq("test")].groupby("eye_id"):
        visits = visits.sort_values("time_from_baseline_years").reset_index(drop=True)
        for baseline_index, baseline in visits.iloc[:-1].iterrows():
            if int(baseline["glaucoma_label"]) != 0:
                continue
            later = visits.iloc[baseline_index + 1 :].copy()
            later["relative_years"] = (
                later["time_from_baseline_years"] - baseline["time_from_baseline_years"]
            )
            later = later[(later["relative_years"] > 0) & (later["relative_years"] <= 5.0)]
            if len(later) < 3 or not later["glaucoma_label"].eq(1).any():
                continue
            selected = []
            unused = set(later.index)
            error = 0.0
            for target in TARGET_YEARS:
                index = min(unused, key=lambda value: abs(float(later.loc[value, "relative_years"]) - target))
                unused.remove(index)
                selected.append(later.loc[index])
                error += abs(float(later.loc[index, "relative_years"]) - target)
            candidates.append((error, str(eye_id), baseline, selected))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected_cases = []
    used_patients = set()
    for candidate in candidates:
        patient_id = str(candidate[2]["patient_id"])
        if patient_id in used_patients:
            continue
        selected_cases.append(candidate)
        used_patients.add(patient_id)
        if len(selected_cases) == count:
            break
    if len(selected_cases) < count:
        raise RuntimeError(f"Only {len(selected_cases)} qualifying held-out cases found")
    return selected_cases


def add_tile(sheet, tile, x, y, title, subtitle, vcdr):
    sheet.paste(tile, (x, y))
    draw = ImageDraw.Draw(sheet)
    draw.text((x + 6, y + TILE + 6), title, fill=TEXT, font=font(17, True))
    draw.text((x + 6, y + TILE + 29), subtitle, fill=MUTED, font=font(13))
    value = "unavailable" if not np.isfinite(vcdr) else f"{vcdr:.3f}"
    draw.text((x + 6, y + TILE + 47), f"Automatic VCDR: {value}", fill=MUTED, font=font(13))


def make_sheet(case_id, baseline, actual, generated, blinded_order=None):
    margin, header, label_width, gap = 24, 82, 160, 14
    row_height = TILE + 70
    width = margin * 2 + label_width + 4 * TILE + 3 * gap
    height = margin * 2 + header + 2 * row_height + 20
    sheet = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), f"SIGF Locked Test Case {case_id}", fill=TEXT, font=font(27, True))
    draw.text(
        (margin, margin + 38),
        "Thin blue contour = smoothed PAPILA-style automatic cup boundary.",
        fill=MUTED,
        font=font(15),
    )
    rows = [("REAL", actual), ("GENERATED", generated)]
    if blinded_order is not None:
        rows = [("ROW A", actual if blinded_order[0] == "real" else generated),
                ("ROW B", actual if blinded_order[1] == "real" else generated)]
    for row_index, (row_label, row_tiles) in enumerate(rows):
        y = margin + header + row_index * row_height
        draw.rounded_rectangle((margin, y, margin + label_width - 14, y + TILE - 4), 12, fill=(229, 238, 251))
        draw.text((margin + 18, y + 24), row_label, fill=(26, 86, 157), font=font(20, True))
        x0 = margin + label_width
        all_tiles = [baseline] + row_tiles
        for index, item in enumerate(all_tiles):
            tile, years, status, vcdr = item
            label = "Baseline" if index == 0 else f"Target year {TARGET_YEARS[index - 1]:g}"
            add_tile(sheet, tile, x0 + index * (TILE + gap), y, label, f"{years}; {status}", vcdr)
    draw.text(
        (margin, height - 25),
        "Exploratory research image. The automatic contour is not expert annotation or clinical evidence.",
        fill=(126, 66, 24),
        font=font(13),
    )
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description="Build held-out SIGF real/generated review rows.")
    parser.add_argument("--manifest", default="data/sigf/manifest.csv")
    parser.add_argument("--checkpoint", default="outputs/training/sigf_residual_diffusion_v1_20260816/best.pt")
    parser.add_argument("--segmenter", default="outputs/training/papila_disc_cup_segmenter_v3_weighted/best.pt")
    parser.add_argument("--output-dir", default="outputs/review/sigf_locked_1_3_5_20260816")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=25)
    parser.add_argument("--change-scale", type=float, default=1.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    torch = require_torch()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    structural_feature_mode = str(config.get("structural_feature_mode", "vessel"))
    structural_channels = 3 if structural_feature_mode == "glaucoma" else 1
    model = build_residual_diffusion(
        int(config["base_channels"]),
        0,
        structural_channels=structural_channels,
        progression_head=float(config.get("progression_loss_weight", 0.0)) > 0,
    ).to(device)
    model.load_state_dict(checkpoint.get("best_state") or checkpoint["model"])
    model.eval()
    segmenter_checkpoint = torch.load(args.segmenter, map_location=device, weights_only=False)
    segmenter = build_disc_cup_segmenter(
        int(segmenter_checkpoint.get("config", {}).get("base_channels", 24))
    ).to(device)
    segmenter.load_state_dict(segmenter_checkpoint["model"])
    segmenter.eval()

    manifest = pd.read_csv(args.manifest)
    cases = select_cases(manifest, args.cases)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(args.seed)
    answer_key = []
    review_rows = []
    summaries = []
    for case_index, (_, eye_id, baseline_row, actual_rows) in enumerate(cases, start=1):
        case_id = f"C{case_index:02d}"
        processed = preprocess_high_resolution(
            str(baseline_row["image_path"]),
            int(config["image_size"]),
            crop_fraction=float(config.get("disc_crop_fraction", 0.50)),
        )
        view_mode = str(config.get("view_mode", "full_field"))
        baseline_image = processed.optic_disc if view_mode == "optic_disc" else processed.full_field
        if view_mode == "optic_disc":
            baseline_image, _ = mirror_left_eye(baseline_image, str(baseline_row["laterality"]))
        baseline_full = tensor(baseline_image, device)
        baseline_low = tensor(
            baseline_image.resize((int(config["residual_size"]), int(config["residual_size"])), Image.Resampling.LANCZOS),
            device,
        )
        structural_images = (
            estimate_glaucoma_structural_maps(baseline_image)
            if structural_feature_mode == "glaucoma"
            else (estimate_vessel_map(baseline_image),)
        )
        vessel_low = torch.cat(
            [
                tensor(
                    feature.resize((int(config["residual_size"]), int(config["residual_size"])), Image.Resampling.BILINEAR),
                    device,
                    grayscale=True,
                )
                for feature in structural_images
            ],
            dim=1,
        )
        samples = sample_residual_trajectory(
            model,
            baseline_full,
            baseline_low,
            vessel_low,
            list(TARGET_YEARS),
            float(config["max_supported_horizon"]),
            float(config["max_change"]),
            trajectories=args.trajectories,
            diffusion_steps=args.sampling_steps,
            seed=args.seed + case_index,
            change_scale=args.change_scale,
        )[0]
        representative = samples.mean(dim=0)
        center = (0.5, 0.5) if view_mode == "optic_disc" else processed.disc_center_normalized
        baseline_reference = baseline_image.resize((256, 256), Image.Resampling.LANCZOS)
        baseline_overlay, baseline_vcdr = outlined_crop(
            segmenter,
            baseline_image,
            center,
            device,
            already_disc_crop=view_mode == "optic_disc",
            segmentation_reference=baseline_reference,
        )
        baseline_item = (baseline_overlay, "observed year 0", "label 0", baseline_vcdr)
        actual_items = []
        generated_items = []
        case_dir = output / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for target_index, (target_year, actual_row) in enumerate(zip(TARGET_YEARS, actual_rows)):
            actual_processed = preprocess_high_resolution(
                str(actual_row["image_path"]),
                int(config["image_size"]),
                crop_fraction=float(config.get("disc_crop_fraction", 0.50)),
            )
            actual_image = (
                actual_processed.optic_disc
                if view_mode == "optic_disc"
                else actual_processed.full_field
            )
            if view_mode == "optic_disc":
                actual_image, _ = mirror_left_eye(actual_image, str(actual_row["laterality"]))
            actual_overlay, actual_vcdr = outlined_crop(
                segmenter,
                actual_image,
                (0.5, 0.5) if view_mode == "optic_disc" else actual_processed.disc_center_normalized,
                device,
                already_disc_crop=view_mode == "optic_disc",
                segmentation_reference=baseline_reference,
            )
            observed_year = float(actual_row["time_from_baseline_years"] - baseline_row["time_from_baseline_years"])
            actual_items.append(
                (actual_overlay, f"observed {observed_year:.2f}y", f"label {int(actual_row['glaucoma_label'])}", actual_vcdr)
            )
            generated_array = (
                representative[target_index].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255
            ).astype(np.uint8)
            generated_image = Image.fromarray(generated_array)
            generated_image.save(case_dir / f"generated_year_{target_year:g}.png")
            generated_overlay, generated_vcdr = outlined_crop(
                segmenter,
                generated_image,
                center,
                device,
                already_disc_crop=view_mode == "optic_disc",
                segmentation_reference=baseline_reference,
            )
            generated_items.append(
                (generated_overlay, f"generated {target_year:g}y", "status unknown", generated_vcdr)
            )
        explicit = make_sheet(case_id, baseline_item, actual_items, generated_items)
        explicit.save(case_dir / "real_vs_generated_rows.png")
        order = ["real", "generated"]
        randomizer.shuffle(order)
        blinded = make_sheet(case_id, baseline_item, actual_items, generated_items, order)
        blinded.save(case_dir / "blinded_rows.png")
        answer_key.append({"case_id": case_id, "row_a": order[0], "row_b": order[1]})
        review_rows.append(
            {
                "case_id": case_id,
                "more_believable_row": "",
                "glaucoma_visible_row_a": "",
                "glaucoma_visible_row_b": "",
                "worsening_visible_row_a": "",
                "worsening_visible_row_b": "",
                "comments": "",
                "dr_gee_analysis": "[add dr. gee analysis results]",
            }
        )
        summaries.append(
            {
                "case_id": case_id,
                "patient_id": str(baseline_row["patient_id"]),
                "eye_id": eye_id,
                "baseline_image_path": str(baseline_row["image_path"]),
                "actual_followups": [
                    {
                        "image_path": str(row["image_path"]),
                        "years": float(row["time_from_baseline_years"] - baseline_row["time_from_baseline_years"]),
                        "glaucoma_label": int(row["glaucoma_label"]),
                    }
                    for row in actual_rows
                ],
                "automatic_baseline_vcdr": baseline_vcdr,
                "automatic_actual_vcdr": [item[3] for item in actual_items],
                "automatic_generated_vcdr": [item[3] for item in generated_items],
            }
        )
    with (output / "dr_gee_review_form.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    (output / "answer_key.json").write_text(json.dumps(answer_key, indent=2) + "\n")
    (output / "case_metadata.json").write_text(json.dumps(summaries, indent=2) + "\n")
    outline_errors = []
    direction_matches = []
    for summary in summaries:
        baseline_vcdr = float(summary["automatic_baseline_vcdr"])
        for actual_vcdr, generated_vcdr in zip(
            summary["automatic_actual_vcdr"], summary["automatic_generated_vcdr"]
        ):
            if not all(np.isfinite(value) for value in (baseline_vcdr, actual_vcdr, generated_vcdr)):
                continue
            outline_errors.append(abs(float(generated_vcdr) - float(actual_vcdr)))
            actual_direction = np.sign(float(actual_vcdr) - baseline_vcdr)
            generated_direction = np.sign(float(generated_vcdr) - baseline_vcdr)
            direction_matches.append(bool(actual_direction == generated_direction))
    report = {
        "cases": len(summaries),
        "output_dir": str(output),
        "automatic_contour_comparisons": len(outline_errors),
        "automatic_contour_vcdr_mae": (
            float(np.mean(outline_errors)) if outline_errors else None
        ),
        "automatic_change_direction_agreement": (
            float(np.mean(direction_matches)) if direction_matches else None
        ),
        "segmenter_validation_disc_dice": 0.9005,
        "segmenter_validation_cup_dice": 0.5617,
        "warning": (
            "Cross-dataset PAPILA-style automatic contours are exploratory and cannot replace "
            "SIGF expert cup/disc annotations or Dr. Gee's masked review."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
