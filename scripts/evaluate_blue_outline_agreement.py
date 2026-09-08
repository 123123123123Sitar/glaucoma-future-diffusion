#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.preprocessing import optic_disc_crop
from glaucoma_forecast.evaluation.anatomy_metrics import vcdr_from_masks
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.disc_cup_segmenter import build_disc_cup_segmenter
from glaucoma_forecast.utils.reproducibility import select_device


RUN_DIR = Path("outputs/training/balanced_residual_diffusion_v7_overnight_20260812")
PROGRESSION_DIR = Path("outputs/progression/heldout_20_eyes_20260812")
OUTPUT_DIR = Path("outputs/evaluation/heldout_20_blue_outline_agreement_20260812")
SEGMENTER = Path("outputs/training/papila_disc_cup_segmenter_v3_weighted/best.pt")
ANNOTATIONS = Path("data/external/grape_official/json")


def tensor(image: Image.Image, device: str):
    torch = require_torch()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).unsqueeze(0).to(device)


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
                if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not visited[next_y, next_x]:
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        if len(component) > len(best):
            best = component
    output = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        output[y, x] = True
    return output


def clean_cup(mask: np.ndarray) -> np.ndarray:
    opened = mask.copy()
    for _ in range(3):
        opened &= np.roll(opened, 1, 0) & np.roll(opened, -1, 0) & np.roll(opened, 1, 1) & np.roll(opened, -1, 1)
    opened = largest_component(opened)
    restored = opened.copy()
    for _ in range(3):
        restored |= np.roll(restored, 1, 0) | np.roll(restored, -1, 0) | np.roll(restored, 1, 1) | np.roll(restored, -1, 1)
    return restored & mask


def generated_vcdr(model, image_path: Path, center: tuple[float, float], device: str) -> float:
    torch = require_torch()
    with Image.open(image_path) as source:
        pixel_center = (round(center[0] * source.width), round(center[1] * source.height))
        crop = optic_disc_crop(source.convert("RGB"), pixel_center, 0.30, 256)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor(crop, device)))[0].cpu().numpy()
    disc = largest_component(probability[0] >= 0.5)
    cup = clean_cup((probability[1] >= 0.5) & disc)
    return vcdr_from_masks(cup, disc)


def expert_vcdr(image_path: str) -> float:
    annotation = json.loads((ANNOTATIONS / f"{Path(image_path).stem}.json").read_text())
    shapes = {shape["label"]: shape["points"] for shape in annotation["shapes"]}
    size = (annotation["imageWidth"], annotation["imageHeight"])
    disc = Image.new("1", size)
    cup = Image.new("1", size)
    ImageDraw.Draw(disc).polygon(shapes["OD"], fill=1)
    ImageDraw.Draw(cup).polygon(shapes["OC"], fill=1)
    return vcdr_from_masks(np.asarray(cup, dtype=bool), np.asarray(disc, dtype=bool))


def main() -> int:
    torch = require_torch()
    device = select_device("mps")
    checkpoint = torch.load(SEGMENTER, map_location=device, weights_only=False)
    model = build_disc_cup_segmenter(int(checkpoint.get("config", {}).get("base_channels", 24))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pairs = pd.read_csv(RUN_DIR / "locked_test_pairs.csv")
    selected = pd.read_csv(
        RUN_DIR / "heldout_progression_inputs.tsv",
        sep="\t",
        names=["patient_id", "eye_id", "baseline_image_path"],
    )
    rows = []
    for selection in selected.itertuples(index=False):
        eligible = pairs[
            pairs["patient_id"].eq(selection.patient_id)
            & pairs["eye_id"].eq(selection.eye_id)
            & pairs["baseline_image_path"].eq(selection.baseline_image_path)
        ].sort_values("horizon_years", ascending=False)
        if eligible.empty:
            raise RuntimeError(f"No held-out follow-up for {selection.patient_id} {selection.eye_id}")
        pair = eligible.iloc[0]
        report_path = PROGRESSION_DIR / f"{selection.patient_id}_{selection.eye_id}" / "report.json"
        report = json.loads(report_path.read_text())
        requested_years = np.asarray(report["requested_years"], dtype=float)
        generated_index = int(np.argmin(np.abs(requested_years - float(pair["horizon_years"]))))
        generated_year = float(requested_years[generated_index])
        real = expert_vcdr(str(pair["future_image_path"]))
        predicted = generated_vcdr(
            model,
            Path(report["representative_images"][generated_index]),
            tuple(report["disc_center_normalized"]),
            device,
        )
        absolute_error = abs(predicted - real)
        relative_agreement = max(0.0, 1.0 - absolute_error / real) * 100.0
        rows.append(
            {
                "patient_id": selection.patient_id,
                "eye_id": selection.eye_id,
                "real_followup_years": float(pair["horizon_years"]),
                "closest_generated_year": generated_year,
                "real_expert_vcdr": real,
                "generated_outline_vcdr": predicted,
                "absolute_vcdr_error": absolute_error,
                "relative_vcdr_agreement_percent": relative_agreement,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "per_patient.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "patients": len(rows),
        "selection": "one preselected held-out eye per patient; longest real follow-up from the generated baseline; closest of generated Years 1, 3, and 5",
        "mean_absolute_vcdr_error": float(np.mean([row["absolute_vcdr_error"] for row in rows])),
        "median_absolute_vcdr_error": float(np.median([row["absolute_vcdr_error"] for row in rows])),
        "mean_relative_vcdr_agreement_percent": float(np.mean([row["relative_vcdr_agreement_percent"] for row in rows])),
        "median_relative_vcdr_agreement_percent": float(np.median([row["relative_vcdr_agreement_percent"] for row in rows])),
        "definition": "relative agreement = 100 * max(0, 1 - abs(generated VCDR - real VCDR) / real VCDR)",
        "real_outline": "official GRAPE expert optic-cup polygon",
        "generated_outline": "frozen disc/cup segmenter with connected-component cleanup",
        "warning": "This measures blue-outline VCDR agreement, not clinical predictiveness or diagnostic accuracy.",
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
