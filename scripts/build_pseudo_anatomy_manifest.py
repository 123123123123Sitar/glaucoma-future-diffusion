#!/usr/bin/env python3
"""Build conservative automatic anatomy labels for preference-guided experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from glaucoma_forecast.evaluation.anatomy_metrics import vcdr_from_masks
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.disc_cup_segmenter import build_disc_cup_segmenter
from glaucoma_forecast.utils.reproducibility import select_device


def _tensor(image: Image.Image, device: str):
    torch = require_torch()
    array = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).unsqueeze(0).to(device)


def _segment(model, path: Path, device: str):
    torch = require_torch()
    with Image.open(path) as source, torch.no_grad():
        probability = torch.sigmoid(model(_tensor(source, device)))[0].cpu().numpy()
    disc = probability[0] >= 0.5
    cup = (probability[1] >= 0.5) & disc
    disc_area = float(disc.mean())
    cup_area = float(cup.mean())
    plausible = 0.08 <= disc_area <= 0.75 and 0.01 <= cup_area < disc_area
    return disc, cup, plausible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", default="outputs/annotations/grape_longitudinal_v1")
    parser.add_argument(
        "--segmenter-checkpoint",
        default="outputs/training/papila_disc_cup_segmenter_v3_weighted/best.pt",
    )
    parser.add_argument("--output-dir", default="outputs/annotations/grape_pseudo_anatomy_v1")
    parser.add_argument("--progressor-delta", type=float, default=0.08)
    parser.add_argument("--stable-delta", type=float, default=0.03)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    torch = require_torch()
    device = select_device(args.device)
    checkpoint = torch.load(args.segmenter_checkpoint, map_location=device, weights_only=False)
    model = build_disc_cup_segmenter(int(checkpoint.get("config", {}).get("base_channels", 24))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    batch_dir = Path(args.batch_dir)
    batch = json.loads((batch_dir / "batch.json").read_text())
    output = Path(args.output_dir)
    masks_dir = output / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for record in batch["records"]:
        eye_id = str(record["eye_id"])
        if eye_id in seen or record.get("repeat_group"):
            continue
        seen.add(eye_id)
        baseline_disc, baseline_cup, baseline_ok = _segment(
            model, batch_dir / record["baseline_asset"], device
        )
        future_disc, future_cup, future_ok = _segment(
            model, batch_dir / record["future_asset"], device
        )
        if not baseline_ok or not future_ok:
            continue
        baseline_vcdr = vcdr_from_masks(baseline_cup, baseline_disc)
        future_vcdr = vcdr_from_masks(future_cup, future_disc)
        delta = future_vcdr - baseline_vcdr
        if delta >= args.progressor_delta:
            label = "progressor"
        elif abs(delta) <= args.stable_delta:
            label = "stable"
        else:
            continue
        mask_paths = {}
        for name, mask in (
            ("baseline_disc", baseline_disc),
            ("baseline_cup", baseline_cup),
            ("future_disc", future_disc),
            ("future_cup", future_cup),
        ):
            path = (masks_dir / f"{record['annotation_id']}_{name}.png").resolve()
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            mask_paths[f"{name}_mask_path"] = str(path)
        rows.append(
            {
                **record,
                **mask_paths,
                "progression_label": label,
                "quality_acceptable": True,
                "baseline_vcdr": baseline_vcdr,
                "future_vcdr": future_vcdr,
                "delta_vcdr": delta,
                "label_source": "automatic_segmenter_pseudo_label_not_expert_ground_truth",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame["progression_label"]) != {"progressor", "stable"}:
        raise ValueError("Automatic thresholds did not yield both progressor and stable examples")
    split_note = "preserved_source_patient_splits"
    if set(frame["split"].astype(str)) != {"train", "validation", "test"}:
        patients = frame["patient_id"].drop_duplicates().astype(str).tolist()
        rng = np.random.default_rng(args.seed)
        rng.shuffle(patients)
        test_count = max(1, round(len(patients) * 0.15))
        validation_count = max(1, round(len(patients) * 0.15))
        test = set(patients[:test_count])
        validation = set(patients[test_count : test_count + validation_count])
        frame["split"] = frame["patient_id"].astype(str).map(
            lambda patient: "test" if patient in test else "validation" if patient in validation else "train"
        )
        split_note = "new_patient_grouped_exploratory_split_no_locked_release_claim"
    manifest = output / "manifest.csv"
    frame.to_csv(manifest, index=False)
    report = {
        "manifest": str(manifest),
        "source_cases": len(seen),
        "retained_cases": len(frame),
        "counts": frame["progression_label"].value_counts().to_dict(),
        "split_counts": frame["split"].value_counts().to_dict(),
        "split_note": split_note,
        "warning": "Automatic pseudo-labels are auxiliary and are not expert annotations.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
