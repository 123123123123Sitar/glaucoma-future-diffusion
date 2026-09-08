"""Training loop for visual scenarios from real registered future visits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from glaucoma_forecast.data.highres import preprocess_high_resolution
from glaucoma_forecast.data.registration import apply_translation, estimate_translation
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.scenario_diffusion import (
    build_scenario_diffusion,
    scenario_diffusion_loss,
)
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


SCENARIO_COLUMNS = {
    "patient_id",
    "baseline_image_path",
    "future_image_path",
    "future_year",
    "risk_model_cumulative_risk",
}


@dataclass(frozen=True)
class ScenarioTrainingConfig:
    manifest: str
    output_dir: str
    val_manifest: str | None = None
    image_size: int = 512
    latent_channels: int = 8
    base_channels: int = 64
    batch_size: int = 1
    epochs: int = 10
    learning_rate: float = 1e-4
    diffusion_steps: int = 100
    identity_weight: float = 0.25
    device: str = "auto"
    seed: int = 20260720
    max_steps: int | None = None
    checkpoint_every: int = 100


def validate_scenario_manifest(frame: pd.DataFrame) -> None:
    missing = sorted(SCENARIO_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Scenario manifest missing columns: {missing}")
    years = pd.to_numeric(frame["future_year"], errors="coerce")
    risks = pd.to_numeric(frame["risk_model_cumulative_risk"], errors="coerce")
    if years.isna().any() or not years.between(0.0, 10.0, inclusive="right").all():
        raise ValueError("future_year must be in (0, 10]")
    if risks.isna().any() or not risks.between(0.0, 1.0).all():
        raise ValueError("risk_model_cumulative_risk must be in [0, 1]")


def _tensor(image: Image.Image):
    torch = require_torch()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


class ScenarioPairDataset:
    def __init__(self, frame: pd.DataFrame, image_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        baseline = preprocess_high_resolution(row["baseline_image_path"], self.image_size)
        future = preprocess_high_resolution(row["future_image_path"], self.image_size)
        shift_x, shift_y, confidence = estimate_translation(
            baseline.full_field,
            future.full_field,
            max_shift=max(4, self.image_size // 12),
        )
        future_image = (
            apply_translation(future.full_field, shift_x, shift_y)
            if confidence >= 3.0
            else future.full_field
        )
        return {
            "baseline": _tensor(baseline.full_field),
            "future": _tensor(future_image),
            "vessel": _tensor(baseline.vessel_map),
            "year": float(row["future_year"]),
            "risk": float(row["risk_model_cumulative_risk"]),
        }


def _collate(batch: list[dict]):
    torch = require_torch()
    return {
        key: torch.stack([item[key] for item in batch])
        for key in ["baseline", "future", "vessel"]
    } | {
        "year": torch.tensor([item["year"] for item in batch], dtype=torch.float32),
        "risk": torch.tensor([item["risk"] for item in batch], dtype=torch.float32),
    }


def train_scenario_model(config: ScenarioTrainingConfig) -> dict[str, object]:
    torch = require_torch()
    seed_everything(config.seed)
    device = select_device(config.device)
    frame = pd.read_csv(config.manifest)
    validate_scenario_manifest(frame)
    val_frame = pd.read_csv(config.val_manifest) if config.val_manifest else None
    if val_frame is not None:
        validate_scenario_manifest(val_frame)
    dataset = ScenarioPairDataset(frame, config.image_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=_collate
    )
    val_loader = None
    if val_frame is not None:
        val_loader = torch.utils.data.DataLoader(
            ScenarioPairDataset(val_frame, config.image_size),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )
    model = build_scenario_diffusion(config.latent_channels, config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    best_val = float("inf")
    for epoch in range(1, config.epochs + 1):
        losses = []
        model.train()
        for batch in loader:
            global_step += 1
            loss = scenario_diffusion_loss(
                model,
                batch["baseline"].to(device),
                batch["future"].to(device),
                batch["vessel"].to(device),
                batch["year"].to(device),
                batch["risk"].to(device),
                config.diffusion_steps,
                config.identity_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if global_step % config.checkpoint_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(config),
                        "epoch": epoch,
                        "step": global_step,
                        "model_version": f"scenario-v2-step-{global_step}",
                        "warning": "SIMULATED SCENARIO - NOT A DIAGNOSIS",
                    },
                    out / f"checkpoint_step_{global_step:06d}.pt",
                )
                print(
                    {
                        "step": global_step,
                        "recent_train_loss": float(np.mean(losses[-min(25, len(losses)) :])),
                        "device": device,
                    },
                    flush=True,
                )
            if config.max_steps is not None and global_step >= config.max_steps:
                break
        value = float(np.mean(losses))
        val_value = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_losses.append(
                        float(
                            scenario_diffusion_loss(
                                model,
                                val_batch["baseline"].to(device),
                                val_batch["future"].to(device),
                                val_batch["vessel"].to(device),
                                val_batch["year"].to(device),
                                val_batch["risk"].to(device),
                                config.diffusion_steps,
                                config.identity_weight,
                            ).cpu()
                        )
                    )
            val_value = float(np.mean(val_losses))
        history.append(
            {"epoch": epoch, "step": global_step, "train_loss": value, "val_loss": val_value}
        )
        checkpoint = {
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "step": global_step,
            "model_version": f"scenario-v2-step-{global_step}",
            "warning": "SIMULATED SCENARIO - NOT A DIAGNOSIS",
        }
        torch.save(checkpoint, out / "latest.pt")
        if val_value is not None and val_value < best_val:
            best_val = val_value
            torch.save(checkpoint, out / "best.pt")
        print(history[-1], flush=True)
        if config.max_steps is not None and global_step >= config.max_steps:
            break
    summary = {
        "status": "trained_visual_scenario_research_model",
        "pairs": len(frame),
        "val_pairs": 0 if val_frame is None else len(val_frame),
        "steps": global_step,
        "checkpoint": str(out / "latest.pt"),
        "best_checkpoint": str(out / "best.pt") if val_frame is not None else None,
        "history": history,
        "warning": "Generated images are scenarios and must not feed the risk model.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
