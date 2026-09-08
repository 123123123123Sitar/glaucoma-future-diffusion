"""Registered optic-disc forecasting for the glaucoma-only GRAPE cohort.

The model remains a compact conditional diffusion denoiser, but it also has an
explicit bounded change head. The change head starts at the persistence
baseline and is trained to add only evidence-supported longitudinal change.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from glaucoma_forecast.data.preprocessing import match_color_statistics, optic_disc_crop, preprocess_image
from glaucoma_forecast.data.registration import apply_translation, estimate_translation
from glaucoma_forecast.data.schema import normalize_manifest, validate_manifest
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


DISCLAIMER = "Research prototype only. Not validated for patient care, diagnosis, triage, or treatment decisions."


@dataclass(frozen=True)
class TrainingConfig:
    manifest: str
    output_dir: str
    image_size: int = 128
    batch_size: int = 2
    max_steps: int = 100
    learning_rate: float = 1e-4
    seed: int = 20260720
    device: str = "auto"
    diffusion_steps: int = 100
    checkpoint_every: int = 50
    validate_every: int = 25
    num_val_patients: int = 16
    base_channels: int = 32
    crop_fraction: float = 0.50
    max_change: float = 0.18
    diffusion_weight: float = 0.25
    forecast_weight: float = 4.0
    edge_weight: float = 1.0
    identity_weight: float = 0.5


def _make_adjacent_pairs(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = manifest.sort_values(["eye_id", "time_from_baseline_years", "visit_index"])
    for eye_id, group in ordered.groupby("eye_id"):
        group = group.reset_index(drop=True)
        for idx in range(1, len(group)):
            context = group.iloc[idx - 1]
            target = group.iloc[idx]
            horizon = float(target["time_from_baseline_years"]) - float(context["time_from_baseline_years"])
            if horizon <= 0:
                continue
            rows.append(
                {
                    "patient_id": str(target["patient_id"]),
                    "eye_id": str(eye_id),
                    "laterality": str(target.get("laterality", "")),
                    "context_visit_id": str(context["visit_id"]),
                    "target_visit_id": str(target["visit_id"]),
                    "context_image_path": str(context["image_path"]),
                    "target_image_path": str(target["image_path"]),
                    "context_time": float(context["time_from_baseline_years"]),
                    "target_time": float(target["time_from_baseline_years"]),
                    "forecast_horizon": horizon,
                }
            )
    return pd.DataFrame(rows)


def _patient_train_val_pairs(pairs: pd.DataFrame, seed: int, num_val_patients: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = sorted(pairs["patient_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = min(max(1, num_val_patients), max(1, len(patients) // 5))
    val_patients = set(patients[:n_val])
    train = pairs[~pairs["patient_id"].isin(val_patients)].reset_index(drop=True)
    val = pairs[pairs["patient_id"].isin(val_patients)].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("Could not create non-empty patient-level train/val pair split")
    return train, val


def _pil_to_tensor(image: Image.Image):
    torch = require_torch()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


class RegisteredDiscPairDataset:
    """Load paired, oriented, optic-disc-centered, translation-aligned visits."""

    def __init__(self, pairs: pd.DataFrame, image_size: int, crop_fraction: float) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.image_size = image_size
        self.crop_fraction = crop_fraction
        self._cache: dict[tuple[str, str], Image.Image] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_crop(self, path: str, laterality: str) -> Image.Image:
        key = (path, laterality)
        if key not in self._cache:
            standardized = preprocess_image(path, output_size=256, laterality=laterality, mirror_left=True).image
            self._cache[key] = optic_disc_crop(
                standardized,
                crop_fraction=self.crop_fraction,
                output_size=self.image_size,
            )
        return self._cache[key].copy()

    def __getitem__(self, index: int):
        row = self.pairs.iloc[index]
        laterality = str(row["laterality"])
        context_image = self._load_crop(str(row["context_image_path"]), laterality)
        target_image = self._load_crop(str(row["target_image_path"]), laterality)
        shift_x, shift_y, confidence = estimate_translation(
            context_image,
            target_image,
            max_shift=max(3, self.image_size // 12),
        )
        if confidence >= 3.0:
            target_image = apply_translation(target_image, shift_x, shift_y)
        else:
            shift_x = shift_y = 0
        target_image = match_color_statistics(target_image, context_image)
        return {
            "context": _pil_to_tensor(context_image),
            "target": _pil_to_tensor(target_image),
            "horizon": float(row["forecast_horizon"]),
            "shift": (shift_x, shift_y),
            "registration_confidence": confidence,
        }


def _collate(batch: list[dict]):
    torch = require_torch()
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "horizon": torch.tensor([item["horizon"] for item in batch], dtype=torch.float32),
        "shift": torch.tensor([item["shift"] for item in batch], dtype=torch.float32),
        "registration_confidence": torch.tensor([item["registration_confidence"] for item in batch], dtype=torch.float32),
    }


def build_compact_denoiser(base_channels: int = 32):
    """Build a conditional denoiser with a zero-initialized residual forecast head."""

    torch = require_torch()
    nn = torch.nn

    class CompactChangeDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_channels = 8
            self.down1 = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.SiLU(),
                nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.SiLU(),
            )
            self.down2 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1), nn.SiLU(),
            )
            self.mid = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1), nn.SiLU(),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1), nn.SiLU(),
            )
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(base_channels * 2, base_channels, 3, padding=1), nn.SiLU(),
            )
            self.noise_out = nn.Conv2d(base_channels * 2, 3, 3, padding=1)
            self.change_out = nn.Conv2d(base_channels * 2, 3, 3, padding=1)
            nn.init.zeros_(self.change_out.weight)
            nn.init.zeros_(self.change_out.bias)

        def _features(self, noisy, context, horizon, timestep):
            horizon_map = horizon[:, None, None, None].expand(-1, 1, noisy.shape[-2], noisy.shape[-1])
            timestep_map = timestep[:, None, None, None].expand(-1, 1, noisy.shape[-2], noisy.shape[-1])
            skip = self.down1(torch.cat([noisy, context, horizon_map, timestep_map], dim=1))
            encoded = self.mid(self.down2(skip))
            return torch.cat([self.up(encoded), skip], dim=1)

        def forward(self, noisy, context, horizon, timestep):
            return self.noise_out(self._features(noisy, context, horizon, timestep))

        def predict_future(self, context, horizon, max_change: float = 0.18):
            timestep = torch.zeros_like(horizon)
            features = self._features(context, context, horizon, timestep)
            residual = torch.tanh(self.change_out(features))
            horizon_scale = torch.clamp(horizon * 5.0, min=0.0, max=3.0)[:, None, None, None] / 3.0
            return torch.clamp(context + max_change * horizon_scale * residual, 0.0, 1.0)

    return CompactChangeDenoiser()


def _alpha_bars(diffusion_steps: int, device: str):
    torch = require_torch()
    betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


def _spatial_masks(size: int, device: str):
    torch = require_torch()
    axis = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius = torch.sqrt(xx**2 + yy**2)
    disc = (radius <= 0.48).float()[None, None]
    outside = (radius >= 0.62).float()[None, None]
    return disc, outside


def _edge_l1(prediction, target):
    torch = require_torch()
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return torch.nn.functional.l1_loss(pred_dx, target_dx) + torch.nn.functional.l1_loss(pred_dy, target_dy)


def _save_image_grid(context, target, prediction, path: Path) -> None:
    arrays = []
    for tensor in [context, target, prediction]:
        arr = np.transpose(tensor.detach().cpu().clamp(0, 1).numpy(), (1, 2, 0))
        arrays.append((arr * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(arrays, axis=1)).save(path)


def _psnr_from_mse(mse: float) -> float:
    return float("inf") if mse <= 0 else float(10.0 * math.log10(1.0 / mse))


def train_grape_glaucoma_denoiser(config: TrainingConfig) -> dict[str, object]:
    torch = require_torch()
    seed_everything(config.seed)
    device = select_device(config.device)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = normalize_manifest(pd.read_csv(config.manifest), source_dataset="GRAPE")
    validate_manifest(manifest, require_images=True)
    pairs = _make_adjacent_pairs(manifest)
    if pairs.empty:
        raise ValueError("No adjacent future-image pairs found. Need at least two visits per eye.")
    train_pairs, val_pairs = _patient_train_val_pairs(pairs, config.seed, config.num_val_patients)
    train_pairs.to_csv(out / "train_pairs.csv", index=False)
    val_pairs.to_csv(out / "val_pairs.csv", index=False)

    train_dataset = RegisteredDiscPairDataset(train_pairs, config.image_size, config.crop_fraction)
    val_dataset = RegisteredDiscPairDataset(val_pairs, config.image_size, config.crop_fraction)
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
        collate_fn=_collate, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate,
    )
    if len(loader) == 0:
        raise ValueError("Training loader is empty; lower batch size or check data.")

    model = build_compact_denoiser(config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    alpha_bars = _alpha_bars(config.diffusion_steps, device)
    disc_mask, outside_mask = _spatial_masks(config.image_size, device)
    log_fields = [
        "step", "train_total", "train_diffusion", "train_forecast", "train_edge", "train_identity",
        "val_model_mae", "val_persistence_mae", "val_improvement", "val_model_psnr", "val_persistence_psnr",
    ]
    log_path = out / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=log_fields).writeheader()

    best_model_mae = math.inf
    best_step = 0
    best_improvement = -math.inf

    def checkpoint(path: Path, step: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "config": asdict(config), "metrics": metrics, "disclaimer": DISCLAIMER,
            },
            path,
        )

    def validate(step: int) -> dict[str, float]:
        model.eval()
        model_abs = persistence_abs = model_sq = persistence_sq = pixels = 0.0
        confidence_values: list[float] = []
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                context = batch["context"].to(device)
                target = batch["target"].to(device)
                horizon = batch["horizon"].to(device).clamp(0, 5) / 5.0
                prediction = model.predict_future(context, horizon, config.max_change)
                model_error = prediction - target
                persistence_error = context - target
                model_abs += float(model_error.abs().sum().cpu())
                persistence_abs += float(persistence_error.abs().sum().cpu())
                model_sq += float(model_error.square().sum().cpu())
                persistence_sq += float(persistence_error.square().sum().cpu())
                pixels += float(target.numel())
                confidence_values.extend(batch["registration_confidence"].tolist())
                if index == 0:
                    _save_image_grid(context[0], target[0], prediction[0], out / f"preview_step_{step:06d}.png")
        model_mae = model_abs / pixels
        persistence_mae = persistence_abs / pixels
        metrics = {
            "val_model_mae": model_mae,
            "val_persistence_mae": persistence_mae,
            "val_improvement": persistence_mae - model_mae,
            "val_model_psnr": _psnr_from_mse(model_sq / pixels),
            "val_persistence_psnr": _psnr_from_mse(persistence_sq / pixels),
            "registration_confidence_mean": float(np.mean(confidence_values)),
        }
        model.train()
        return metrics

    step = 0
    last_losses = {"train_total": math.nan, "train_diffusion": math.nan, "train_forecast": math.nan, "train_edge": math.nan, "train_identity": math.nan}
    last_metrics: dict[str, float] = {}
    model.train()
    while step < config.max_steps:
        for batch in loader:
            step += 1
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            horizon = batch["horizon"].to(device).clamp(0, 5) / 5.0
            timestep_idx = torch.randint(0, config.diffusion_steps, (target.shape[0],), device=device)
            timestep = timestep_idx.float() / max(1, config.diffusion_steps - 1)
            alpha_bar = alpha_bars[timestep_idx][:, None, None, None]
            noise = torch.randn_like(target)
            noisy = torch.sqrt(alpha_bar) * target + torch.sqrt(1.0 - alpha_bar) * noise
            predicted_noise = model(noisy, context, horizon, timestep)
            prediction = model.predict_future(context, horizon, config.max_change)

            diffusion_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            weights = 1.0 + 2.0 * disc_mask
            forecast_loss = ((prediction - target).abs() * weights).mean()
            edge_loss = _edge_l1(prediction, target)
            identity_loss = ((prediction - context).abs() * outside_mask).sum() / max(1.0, float(outside_mask.sum() * prediction.shape[0] * prediction.shape[1]))
            loss = (
                config.diffusion_weight * diffusion_loss
                + config.forecast_weight * forecast_loss
                + config.edge_weight * edge_loss
                + config.identity_weight * identity_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_losses = {
                "train_total": float(loss.detach().cpu()),
                "train_diffusion": float(diffusion_loss.detach().cpu()),
                "train_forecast": float(forecast_loss.detach().cpu()),
                "train_edge": float(edge_loss.detach().cpu()),
                "train_identity": float(identity_loss.detach().cpu()),
            }

            if step % config.validate_every == 0 or step == 1:
                last_metrics = validate(step)
                row = {"step": step, **last_losses, **{key: last_metrics[key] for key in log_fields if key in last_metrics}}
                with log_path.open("a", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=log_fields, extrasaction="ignore").writerow(row)
                print({"step": step, **last_losses, **last_metrics, "device": device}, flush=True)
                if last_metrics["val_model_mae"] < best_model_mae:
                    best_model_mae = last_metrics["val_model_mae"]
                    best_improvement = last_metrics["val_improvement"]
                    best_step = step
                    checkpoint(out / "best.pt", step, last_metrics)

            if step % config.checkpoint_every == 0:
                checkpoint(out / f"checkpoint_step_{step:06d}.pt", step, last_metrics)
                checkpoint(out / "latest.pt", step, last_metrics)
            if step >= config.max_steps:
                break

    checkpoint(out / "latest.pt", step, last_metrics)
    summary = {
        "mode": "registered_optic_disc_change_conditioned_diffusion",
        "device": device,
        "steps": step,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "best_step": best_step,
        "best_val_model_mae": best_model_mae,
        "best_val_improvement_over_persistence_mae": best_improvement,
        "beats_persistence": bool(best_improvement > 0),
        "best_checkpoint": str(out / "best.pt"),
        "latest_checkpoint": str(out / "latest.pt"),
        "preprocessing": [
            "retinal-FOV crop", "OS laterality mirroring", "optic-disc-centered crop",
            "bounded phase-correlation translation", "target-to-context color-statistic matching",
        ],
        "disclaimer": DISCLAIMER,
        "limitations": [
            "GRAPE contains glaucoma eyes only; this does not model normal-to-glaucoma conversion.",
            "The optic-disc locator and registration are heuristic and require manual review.",
            "No visit-level progression labels or cup/disc masks are available for supervision.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
