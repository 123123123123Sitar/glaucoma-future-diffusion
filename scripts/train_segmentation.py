#!/usr/bin/env python3
"""Train disc/cup segmentation from PAPILA expert contours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.disc_cup_segmenter import build_disc_cup_segmenter
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


def _standard_contour(contours: Path, image_id: str, structure: str, expert: int) -> Path:
    exact = contours / f"{image_id}_{structure}_exp{expert}.txt"
    if exact.exists():
        return exact
    matches = sorted(contours.glob(f"{image_id}_{structure}_exp{expert}*.txt"))
    if not matches:
        raise FileNotFoundError(f"Missing {structure} expert {expert} contour for {image_id}")
    return matches[0]


def _points(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Invalid contour: {path}")
    return values


def _crop_box(disc_points: list[np.ndarray], image_size: tuple[int, int], fraction: float):
    combined = np.concatenate(disc_points, axis=0)
    center_x, center_y = combined.mean(axis=0)
    width, height = image_size
    side = min(width, height) * fraction
    left = min(max(0.0, center_x - side / 2), width - side)
    top = min(max(0.0, center_y - side / 2), height - side)
    return left, top, left + side, top + side


def _soft_mask(points_by_expert: list[np.ndarray], box, output_size: int) -> np.ndarray:
    left, top, right, bottom = box
    masks = []
    for points in points_by_expert:
        scaled = [
            (
                float((x - left) / (right - left) * output_size),
                float((y - top) / (bottom - top) * output_size),
            )
            for x, y in points
        ]
        canvas = Image.new("L", (output_size, output_size), 0)
        ImageDraw.Draw(canvas).polygon(scaled, fill=255)
        masks.append(np.asarray(canvas, dtype=np.float32) / 255.0)
    return np.mean(masks, axis=0)


class PapilaSegmentationDataset:
    def __init__(self, records, output_size: int, crop_fraction: float, augment: bool):
        self.records = records
        self.output_size = output_size
        self.crop_fraction = crop_fraction
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        torch = require_torch()
        image_path, contour_dir, image_id = self.records[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        disc = [_points(_standard_contour(contour_dir, image_id, "disc", expert)) for expert in (1, 2)]
        cup = [_points(_standard_contour(contour_dir, image_id, "cup", expert)) for expert in (1, 2)]
        box = _crop_box(disc, image.size, self.crop_fraction)
        crop = image.crop(tuple(int(round(value)) for value in box)).resize(
            (self.output_size, self.output_size), Image.Resampling.LANCZOS
        )
        image_array = np.asarray(crop, dtype=np.float32) / 255.0
        masks = np.stack(
            [
                _soft_mask(disc, box, self.output_size),
                _soft_mask(cup, box, self.output_size),
            ]
        )
        image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)).copy())
        mask_tensor = torch.from_numpy(masks.copy())
        if self.augment and bool(torch.rand(()) < 0.5):
            image_tensor = image_tensor.flip(-1)
            mask_tensor = mask_tensor.flip(-1)
        return image_tensor, mask_tensor


class LongitudinalSegmentationDataset:
    """Use human-traced, registered GRAPE disc crops without pseudo-labels."""

    def __init__(self, records, output_size: int, augment: bool):
        self.records = records
        self.output_size = output_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        torch = require_torch()
        image_path, disc_mask_path, cup_mask_path = self.records[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(
                (self.output_size, self.output_size), Image.Resampling.LANCZOS
            )
        masks = []
        for mask_path in (disc_mask_path, cup_mask_path):
            with Image.open(mask_path) as source:
                masks.append(
                    np.asarray(
                        source.convert("L").resize(
                            (self.output_size, self.output_size), Image.Resampling.NEAREST
                        ),
                        dtype=np.float32,
                    )
                    / 255.0
                )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)).copy())
        mask_tensor = torch.from_numpy(np.stack(masks).copy())
        if self.augment and bool(torch.rand(()) < 0.5):
            image_tensor = image_tensor.flip(-1)
            mask_tensor = mask_tensor.flip(-1)
        return image_tensor, mask_tensor


def _longitudinal_records(manifest_path: str, split: str):
    path = Path(manifest_path)
    frame = pd.read_csv(path)
    frame = frame[
        (frame["split"].astype(str) == split)
        & frame["quality_acceptable"].astype(bool)
    ]
    records = []
    for row in frame.to_dict("records"):
        for visit in ("baseline", "future"):
            image_key = (
                f"{visit}_asset"
                if f"{visit}_asset" in row
                else f"{visit}_image_path"
            )
            image = Path(str(row[image_key]))
            if not image.is_absolute() and not image.exists():
                image = path.parent / image
            records.append(
                (
                    image,
                    Path(str(row[f"{visit}_disc_mask_path"])),
                    Path(str(row[f"{visit}_cup_mask_path"])),
                )
            )
    return records


def _dice(probability, target):
    intersection = (probability * target).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    return (2.0 * intersection + 1.0) / (denominator + 1.0)


def main() -> int:
    torch = require_torch()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papila-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-fraction", type=float, default=0.50)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--longitudinal-annotations")
    parser.add_argument("--initialize-from")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    root = Path(args.papila_root)
    images = root / "FundusImages"
    contours = root / "ExpertsSegmentations" / "Contours"
    records = [(path, contours, path.stem) for path in sorted(images.glob("*.jpg"))]
    patient_ids = sorted({record[2][3:6] for record in records})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patient_ids)
    validation_patients = set(patient_ids[: max(1, len(patient_ids) // 5)])
    train_records = [record for record in records if record[2][3:6] not in validation_patients]
    validation_records = [record for record in records if record[2][3:6] in validation_patients]
    train_data = PapilaSegmentationDataset(train_records, args.image_size, args.crop_fraction, True)
    validation_data = PapilaSegmentationDataset(validation_records, args.image_size, args.crop_fraction, False)
    grape_train_records = []
    grape_validation_records = []
    if args.longitudinal_annotations:
        grape_train_records = _longitudinal_records(args.longitudinal_annotations, "train")
        grape_validation_records = _longitudinal_records(
            args.longitudinal_annotations, "validation"
        )
        if grape_train_records:
            train_data = torch.utils.data.ConcatDataset(
                [
                    train_data,
                    LongitudinalSegmentationDataset(
                        grape_train_records, args.image_size, True
                    ),
                ]
            )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    grape_validation_loader = None
    if grape_validation_records:
        grape_validation_loader = torch.utils.data.DataLoader(
            LongitudinalSegmentationDataset(
                grape_validation_records, args.image_size, False
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    model = build_disc_cup_segmenter(args.base_channels).to(device)
    if args.initialize_from:
        state = torch.load(args.initialize_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_cup_dice = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images_batch, masks_batch in train_loader:
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)
            logits = model(images_batch)
            pixel_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, masks_batch, reduction="none"
            )
            positive_weight = torch.tensor(
                [3.0, 12.0], device=device, dtype=logits.dtype
            )[None, :, None, None]
            bce = (pixel_bce * (1.0 + (positive_weight - 1.0) * masks_batch)).mean()
            channel_dice_loss = 1.0 - _dice(torch.sigmoid(logits), masks_batch)
            dice_loss = (channel_dice_loss[0] + 2.0 * channel_dice_loss[1]) / 3.0
            loss = bce + dice_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        dice_sum = torch.zeros(2, device=device)
        examples = 0
        with torch.no_grad():
            for images_batch, masks_batch in validation_loader:
                probabilities = torch.sigmoid(model(images_batch.to(device)))
                target = masks_batch.to(device)
                dice_sum += _dice(probabilities, target) * images_batch.shape[0]
                examples += images_batch.shape[0]
        dice = dice_sum / examples
        grape_dice = None
        if grape_validation_loader is not None:
            grape_sum = torch.zeros(2, device=device)
            grape_examples = 0
            with torch.no_grad():
                for images_batch, masks_batch in grape_validation_loader:
                    probabilities = torch.sigmoid(model(images_batch.to(device)))
                    target = masks_batch.to(device)
                    grape_sum += _dice(probabilities, target) * images_batch.shape[0]
                    grape_examples += images_batch.shape[0]
            grape_dice = grape_sum / grape_examples
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_disc_dice": float(dice[0].cpu()),
            "validation_cup_dice": float(dice[1].cpu()),
            "grape_validation_disc_dice": (
                float(grape_dice[0].cpu()) if grape_dice is not None else None
            ),
            "grape_validation_cup_dice": (
                float(grape_dice[1].cpu()) if grape_dice is not None else None
            ),
        }
        history.append(row)
        print(row, flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "config": vars(args),
            "epoch": epoch,
            "metrics": row,
            "training_source": "PAPILA_v1.1_two_expert_contours",
        }
        torch.save(checkpoint, output / "latest.pt")
        selection_dice = (
            row["grape_validation_cup_dice"]
            if row["grape_validation_cup_dice"] is not None
            else row["validation_cup_dice"]
        )
        if selection_dice > best_cup_dice:
            best_cup_dice = selection_dice
            torch.save(checkpoint, output / "best.pt")
    report = {
        "train_eyes": len(train_records),
        "validation_eyes": len(validation_records),
        "grape_train_images": len(grape_train_records),
        "grape_validation_images": len(grape_validation_records),
        "patient_grouped_split": True,
        "best_validation_cup_dice": best_cup_dice,
        "history": history,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
