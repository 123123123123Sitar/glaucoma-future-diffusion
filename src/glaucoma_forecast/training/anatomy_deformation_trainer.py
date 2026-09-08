"""Training loop for expert-supervised optic-disc geometry deformation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.anatomy_deformation import (
    build_anatomy_deformation_model,
    displacement_jacobian,
    soft_anatomy_envelope,
)
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


@dataclass
class AnatomyDeformationConfig:
    manifest: str
    output_dir: str
    steps: int = 2500
    batch_size: int = 4
    image_size: int = 256
    base_channels: int = 24
    max_displacement: float = 18.0
    learning_rate: float = 2e-4
    evaluation_interval: int = 250
    seed: int = 20260811
    device: str = "auto"


def _image_tensor(path: str, size: int):
    torch = require_torch()
    with Image.open(path) as source:
        image = source.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


def _mask_tensor(path: str, size: int):
    torch = require_torch()
    with Image.open(path) as source:
        mask = source.convert("L").resize((size, size), Image.Resampling.NEAREST)
    return torch.from_numpy((np.asarray(mask) > 0).astype(np.float32)[None].copy())


class ExpertPairDataset:
    def __init__(self, frame: pd.DataFrame, image_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        return {
            "baseline": _image_tensor(str(row["baseline_image_path"]), self.image_size),
            "future": _image_tensor(str(row["future_image_path"]), self.image_size),
            "baseline_disc": _mask_tensor(str(row["baseline_disc_mask_path"]), self.image_size),
            "baseline_cup": _mask_tensor(str(row["baseline_cup_mask_path"]), self.image_size),
            "future_disc": _mask_tensor(str(row["future_disc_mask_path"]), self.image_size),
            "future_cup": _mask_tensor(str(row["future_cup_mask_path"]), self.image_size),
            "horizon": float(row["horizon_years"]) / 5.0,
            "strength": max(0.0, float(row["delta_vcdr"])),
            "stable": str(row["progression_label"]) == "stable",
        }


def _soft_dice(prediction, target):
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _soft_vcdr(cup, disc):
    cup_height = cup.amax(dim=3).sum(dim=2).squeeze(1)
    disc_height = disc.amax(dim=3).sum(dim=2).squeeze(1).clamp_min(1.0)
    return cup_height / disc_height


def _rim_sectors(disc, cup):
    torch = require_torch()
    height, width = disc.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=disc.device),
        torch.arange(width, device=disc.device),
        indexing="ij",
    )
    vertical = (y - (height - 1) / 2).abs() >= (x - (width - 1) / 2).abs()
    sectors = (
        vertical & (y >= (height - 1) / 2),
        vertical & (y < (height - 1) / 2),
        ~vertical & (x < (width - 1) / 2),
        ~vertical & (x >= (width - 1) / 2),
    )
    rim = (disc - cup).clamp_min(0.0)
    return torch.stack([
        (rim * sector).sum(dim=(2, 3)) / sector.sum().clamp_min(1)
        for sector in sectors
    ], dim=2).squeeze(1)


def _moment_match(source, reference):
    source_mean = source.mean(dim=(2, 3), keepdim=True)
    source_std = source.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    reference_mean = reference.mean(dim=(2, 3), keepdim=True)
    reference_std = reference.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    return ((source - source_mean) / source_std * reference_std + reference_mean).clamp(0, 1)


def _losses(output, batch):
    torch = require_torch()
    target_future = _moment_match(batch["future"], batch["baseline"])
    cup_dice = _soft_dice(output["warped_cup"], batch["future_cup"])
    outer_disc_dice = _soft_dice(output["warped_disc"], batch["baseline_disc"])
    predicted_vcdr = _soft_vcdr(output["warped_cup"], output["warped_disc"])
    target_vcdr = _soft_vcdr(batch["future_cup"], batch["future_disc"])
    vcdr_loss = torch.nn.functional.smooth_l1_loss(predicted_vcdr, target_vcdr)
    sector_loss = torch.nn.functional.smooth_l1_loss(
        _rim_sectors(output["warped_disc"], output["warped_cup"]),
        _rim_sectors(batch["future_disc"], batch["future_cup"]),
    )
    envelope = soft_anatomy_envelope(batch["baseline_disc"])
    outside = 1.0 - envelope
    outside_loss = ((output["generated"] - batch["baseline"]).abs() * outside).sum() / (
        outside.sum() * 3.0
    ).clamp_min(1.0)
    photo_loss = ((output["generated"] - target_future).abs() * envelope).sum() / (
        envelope.sum() * 3.0
    ).clamp_min(1.0)
    velocity = output["velocity"]
    smoothness = (
        (velocity[:, :, 1:] - velocity[:, :, :-1]).abs().mean()
        + (velocity[:, :, :, 1:] - velocity[:, :, :, :-1]).abs().mean()
    )
    jacobian = displacement_jacobian(output["displacement"])
    folding_loss = torch.relu(0.05 - jacobian).mean()
    stable = batch["stable"].bool()
    stability_loss = output["generated"].sum() * 0.0
    if stable.any():
        stability_loss = (
            output["displacement"][stable].abs().mean()
            + (output["warped_cup"][stable] - batch["baseline_cup"][stable]).abs().mean()
        )
    total = (
        8.0 * (1.0 - cup_dice)
        + 12.0 * vcdr_loss
        + 4.0 * sector_loss
        + 5.0 * (1.0 - outer_disc_dice)
        + 2.0 * outside_loss
        + 0.5 * photo_loss
        + 0.5 * output["appearance_residual"].abs().mean()
        + 0.05 * smoothness
        + 10.0 * folding_loss
        + 3.0 * stability_loss
    )
    return total, {
        "cup_dice": cup_dice,
        "outer_disc_dice": outer_disc_dice,
        "vcdr_mae": (predicted_vcdr - target_vcdr).abs().mean(),
        "outside_mae": outside_loss,
        "positive_jacobian_fraction": (jacobian > 0).float().mean(),
    }


def _collate(records, device: str):
    torch = require_torch()
    result = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values).to(device)
        elif isinstance(values[0], bool):
            result[key] = torch.tensor(values, dtype=torch.bool, device=device)
        else:
            result[key] = torch.tensor(values, dtype=torch.float32, device=device)
    return result


def _evaluate(model, dataset, device: str):
    torch = require_torch()
    totals = []
    metrics = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = _collate([dataset[index]], device)
            output = model(
                batch["baseline"], batch["baseline_disc"], batch["baseline_cup"],
                batch["horizon"], batch["strength"],
            )
            loss, values = _losses(output, batch)
            totals.append(float(loss.cpu()))
            metrics.append({name: float(value.cpu()) for name, value in values.items()})
    return {
        "loss": float(np.mean(totals)),
        **{name: float(np.mean([item[name] for item in metrics])) for name in metrics[0]},
    }


def train_anatomy_deformation(config: AnatomyDeformationConfig) -> dict:
    torch = require_torch()
    seed_everything(config.seed)
    device = select_device(config.device)
    frame = pd.read_csv(config.manifest)
    frame = frame[frame["progression_label"].isin(["progressor", "stable"])].copy()
    train_frame = frame[frame["split"] == "train"]
    validation_frame = frame[frame["split"] == "validation"]
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Expert deformation training requires train and validation pairs")
    train_dataset = ExpertPairDataset(train_frame, config.image_size)
    validation_dataset = ExpertPairDataset(validation_frame, config.image_size)
    weights = np.where(train_frame["progression_label"].to_numpy() == "progressor", 3.0, 1.0)
    weights = weights / weights.sum()
    rng = np.random.default_rng(config.seed)
    model = build_anatomy_deformation_model(config.base_channels, config.max_displacement).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = float("inf")
    for step in range(1, config.steps + 1):
        indices = rng.choice(len(train_dataset), size=config.batch_size, replace=True, p=weights)
        batch = _collate([train_dataset[int(index)] for index in indices], device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(
            batch["baseline"], batch["baseline_disc"], batch["baseline_cup"],
            batch["horizon"], batch["strength"],
        )
        loss, train_metrics = _losses(result, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.evaluation_interval == 0 or step == config.steps:
            validation = _evaluate(model, validation_dataset, device)
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                **{f"train_{name}": float(value.detach().cpu()) for name, value in train_metrics.items()},
                **{f"validation_{name}": value for name, value in validation.items()},
            }
            history.append(record)
            checkpoint = {
                "model": model.state_dict(),
                "step": step,
                "config": asdict(config),
                "validation": validation,
                "architecture": "time_conditioned_diffeomorphic_optic_disc_deformation_v1",
                "label_source": "official_grape_ophthalmologist_polygon",
            }
            torch.save(checkpoint, output_dir / f"checkpoint_step_{step:06d}.pt")
            torch.save(checkpoint, output_dir / "latest.pt")
            score = validation["loss"]
            if score < best_score:
                best_score = score
                torch.save(checkpoint, output_dir / "best.pt")
            print(json.dumps(record), flush=True)
    summary = {
        "status": "training_complete",
        "architecture": "time_conditioned_diffeomorphic_optic_disc_deformation_v1",
        "device": device,
        "train_pairs": int(len(train_frame)),
        "validation_pairs": int(len(validation_frame)),
        "steps": config.steps,
        "best_validation_loss": best_score,
        "best_checkpoint": str(output_dir / "best.pt"),
        "history": history,
        "warning": "RESEARCH-ONLY SIMULATION. NOT A DIAGNOSIS OR PATIENT-SPECIFIC PROGNOSIS.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
