"""Leakage-safe training for direct registered residual diffusion."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

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
from glaucoma_forecast.data.registration import apply_translation, estimate_translation
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.residual_diffusion import (
    build_residual_diffusion,
    diffusion_schedule,
    reconstruct_clean,
)
from glaucoma_forecast.models.disc_cup_segmenter import build_disc_cup_segmenter
from glaucoma_forecast.training.anatomy_losses import (
    disc_sector_masks,
    rnfl_decline_weights,
    sector_weighted_residual_loss,
)
from glaucoma_forecast.training.disc_cup_losses import (
    annotated_disc_cup_loss,
    frozen_disc_cup_loss,
)
from glaucoma_forecast.utils.reproducibility import seed_everything, select_device


WARNING = (
    "RESEARCH-ONLY SIMULATED TRAJECTORY. NOT A DIAGNOSIS OR PATIENT-SPECIFIC "
    "PROGNOSIS."
)


@dataclass(frozen=True)
class ResidualTrainingConfig:
    manifest: str
    output_dir: str
    pair_manifest: str | None = None
    image_size: int = 512
    residual_size: int = 256
    base_channels: int = 32
    batch_size: int = 1
    max_steps: int = 3000
    learning_rate: float = 1e-4
    diffusion_steps: int = 100
    max_change: float = 0.18
    max_supported_horizon: float = 4.84
    validation_patients: int = 20
    test_patients: int = 20
    checkpoint_every: int = 100
    validate_every: int = 100
    device: str = "auto"
    seed: int = 20260729
    resume: str | None = None
    initialize_from: str | None = None
    view_mode: str = "full_field"
    disc_crop_fraction: float = 0.50
    conditioning_manifest: str | None = None
    conditioning_column: str = "progression_probability_3y"
    residual_loss_weight: float = 0.75
    edge_loss_weight: float = 0.2
    identity_loss_weight: float = 0.1
    anatomy_loss_weight: float = 0.0
    anatomy_sector_emphasis: float = 3.0
    anatomy_inner_radius: float = 0.06
    anatomy_outer_radius: float = 0.30
    segmenter_checkpoint: str | None = None
    disc_cup_loss_weight: float = 0.75
    cup_ratio_loss_weight: float = 1.5
    annotated_anatomy_loss_weight: float = 2.0
    outer_disc_preservation_weight: float = 1.0
    stability_loss_weight: float = 1.0
    minimum_segmenter_disc_dice: float = 0.90
    minimum_segmenter_cup_dice: float = 0.75
    allow_pseudo_anatomy: bool = False
    structural_feature_mode: str = "vessel"
    transition_balance_power: float = 0.0
    structural_focus_loss_weight: float = 0.0
    superior_inferior_emphasis: float = 2.0
    feature_reconstruction_loss_weight: float = 0.0
    progression_loss_weight: float = 0.0


def make_all_future_pairs(manifest: pd.DataFrame) -> pd.DataFrame:
    """Use all real earlier→later visits, never a synthetic intermediate."""

    required = {
        "patient_id",
        "eye_id",
        "laterality",
        "visit_id",
        "time_from_baseline_years",
        "image_path",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"GRAPE manifest missing columns: {missing}")
    rows = []
    ordered = manifest.dropna(subset=["time_from_baseline_years"]).sort_values(
        ["eye_id", "time_from_baseline_years", "visit_index"]
    )
    for eye_id, group in ordered.groupby("eye_id"):
        visits = group.to_dict("records")
        for baseline_index, baseline in enumerate(visits[:-1]):
            for future in visits[baseline_index + 1 :]:
                horizon = float(future["time_from_baseline_years"]) - float(
                    baseline["time_from_baseline_years"]
                )
                if horizon <= 0:
                    continue
                rows.append(
                    {
                        "patient_id": str(baseline["patient_id"]),
                        "eye_id": str(eye_id),
                        "laterality": str(baseline["laterality"]),
                        "baseline_visit_id": str(baseline["visit_id"]),
                        "future_visit_id": str(future["visit_id"]),
                        "baseline_image_path": str(baseline["image_path"]),
                        "future_image_path": str(future["image_path"]),
                        "horizon_years": horizon,
                        **{
                            f"baseline_rnfl_{sector}": baseline.get(f"rnfl_{sector}", np.nan)
                            for sector in ("inferior", "superior", "nasal", "temporal")
                        },
                        **{
                            f"future_rnfl_{sector}": future.get(f"rnfl_{sector}", np.nan)
                            for sector in ("inferior", "superior", "nasal", "temporal")
                        },
                    }
                )
    return pd.DataFrame(rows)


def patient_split(
    pairs: pd.DataFrame,
    seed: int,
    validation_patients: int,
    test_patients: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = sorted(pairs["patient_id"].unique().tolist())
    if len(patients) < 5:
        raise ValueError("At least five patients are required for grouped splits")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    maximum_holdout = max(1, len(patients) // 5)
    n_val = min(validation_patients, maximum_holdout)
    n_test = min(test_patients, maximum_holdout)
    validation = set(patients[:n_val])
    test = set(patients[n_val : n_val + n_test])
    train = pairs[~pairs["patient_id"].isin(validation | test)].reset_index(drop=True)
    val = pairs[pairs["patient_id"].isin(validation)].reset_index(drop=True)
    held_out = pairs[pairs["patient_id"].isin(test)].reset_index(drop=True)
    if train.empty or val.empty or held_out.empty:
        raise ValueError("Grouped split produced an empty partition")
    return train, val, held_out


def _tensor(image: Image.Image, channels: int = 3):
    torch = require_torch()
    converted = image.convert("L" if channels == 1 else "RGB")
    array = np.asarray(converted, dtype=np.float32) / 255.0
    if channels == 1:
        array = array[None]
    else:
        array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array.copy())


class RegisteredResidualPairDataset:
    def __init__(
        self,
        pairs: pd.DataFrame,
        image_size: int,
        residual_size: int,
        max_change: float,
        max_supported_horizon: float,
        view_mode: str = "full_field",
        disc_crop_fraction: float = 0.50,
        clinical_dim: int = 0,
        structural_feature_mode: str = "vessel",
    ) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.image_size = image_size
        self.residual_size = residual_size
        self.max_change = max_change
        self.max_supported_horizon = max_supported_horizon
        if view_mode not in {"full_field", "optic_disc"}:
            raise ValueError("view_mode must be full_field or optic_disc")
        self.view_mode = view_mode
        self.disc_crop_fraction = disc_crop_fraction
        self.clinical_dim = clinical_dim
        if structural_feature_mode not in {"vessel", "glaucoma"}:
            raise ValueError("structural_feature_mode must be vessel or glaucoma")
        self.structural_feature_mode = structural_feature_mode
        self._cache: dict[str, object] = {}
        self._pair_cache: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, path: str):
        if path not in self._cache:
            self._cache[path] = preprocess_high_resolution(
                path,
                self.image_size,
                crop_fraction=self.disc_crop_fraction,
            )
        return self._cache[path]

    def _structural_tensor(self, image: Image.Image):
        if self.structural_feature_mode == "glaucoma":
            return require_torch().cat(
                [_tensor(feature, channels=1) for feature in estimate_glaucoma_structural_maps(image)],
                dim=0,
            )
        return _tensor(estimate_vessel_map(image), channels=1)

    def __getitem__(self, index: int):
        if index in self._pair_cache:
            return self._pair_cache[index]
        row = self.pairs.iloc[index]
        baseline = self._load(str(row["baseline_image_path"]))
        future = self._load(str(row["future_image_path"]))
        if self.view_mode == "optic_disc":
            baseline_image = baseline.optic_disc
            future_image = future.optic_disc
            baseline_image, _ = mirror_left_eye(baseline_image, str(row["laterality"]))
            future_image, _ = mirror_left_eye(future_image, str(row["laterality"]))
            shift_x, shift_y, confidence = estimate_translation(
                baseline_image,
                future_image,
                max_shift=max(4, self.image_size // 12),
            )
            if confidence >= 3.0:
                future_image = apply_translation(future_image, shift_x, shift_y)
            future_image = match_color_statistics(future_image, baseline_image)
            baseline_structural = self._structural_tensor(baseline_image)
        else:
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
            future_image = match_color_statistics(future_image, baseline.full_field)
            baseline_image = baseline.full_field
            baseline_structural = self._structural_tensor(baseline_image)
        baseline_low = baseline_image.resize(
            (self.residual_size, self.residual_size), Image.Resampling.LANCZOS
        )
        future_low = future_image.resize(
            (self.residual_size, self.residual_size), Image.Resampling.LANCZOS
        )
        structural_low = require_torch().nn.functional.interpolate(
            baseline_structural.unsqueeze(0),
            size=(self.residual_size, self.residual_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        baseline_tensor = _tensor(baseline_low)
        future_tensor = _tensor(future_low)
        residual = ((future_tensor - baseline_tensor) / self.max_change).clamp(
            -1.0, 1.0
        )
        annotation_columns = [
            "baseline_disc_mask_path",
            "baseline_cup_mask_path",
            "future_disc_mask_path",
            "future_cup_mask_path",
        ]
        annotation_available = all(
            isinstance(row.get(column), str)
            and bool(str(row.get(column)).strip())
            and Path(str(row.get(column))).is_file()
            for column in annotation_columns
        )
        annotation_masks = np.zeros((4, self.residual_size, self.residual_size), dtype=np.float32)
        if annotation_available:
            for mask_index, column in enumerate(annotation_columns):
                with Image.open(str(row[column])) as source:
                    annotation_masks[mask_index] = (
                        np.asarray(
                            source.convert("L").resize(
                                (self.residual_size, self.residual_size),
                                Image.Resampling.NEAREST,
                            ),
                            dtype=np.float32,
                        )
                        / 255.0
                    )
        item = {
            "baseline_low": baseline_tensor,
            "vessel_low": structural_low,
            "target_residual": residual,
            "horizon": float(row["horizon_years"]) / self.max_supported_horizon,
            "registration_confidence": float(confidence),
            "patient_id": str(row["patient_id"]),
            "eye_id": str(row["eye_id"]),
            "laterality": str(row["laterality"]),
            "disc_center": require_torch().tensor(
                (0.5, 0.5) if self.view_mode == "optic_disc" else baseline.disc_center_normalized,
                dtype=require_torch().float32,
            ),
            "baseline_rnfl": require_torch().tensor(
                [float(row.get(f"baseline_rnfl_{sector}", np.nan)) for sector in ("inferior", "superior", "nasal", "temporal")],
                dtype=require_torch().float32,
            ),
            "future_rnfl": require_torch().tensor(
                [float(row.get(f"future_rnfl_{sector}", np.nan)) for sector in ("inferior", "superior", "nasal", "temporal")],
                dtype=require_torch().float32,
            ),
            "clinical_condition": require_torch().tensor(
                [
                    float(row.get("progression_probability_3y", 0.0))
                    for _ in range(self.clinical_dim)
                ],
                dtype=require_torch().float32,
            ),
            "annotation_masks": require_torch().from_numpy(annotation_masks),
            "annotation_available": annotation_available,
            "stable_label": str(row.get("progression_label", "")) == "stable",
            "future_glaucoma_label": float(row.get("future_glaucoma_label", np.nan)),
        }
        self._pair_cache[index] = item
        return item


def _collate(batch: list[dict]):
    torch = require_torch()
    return {
        "baseline_low": torch.stack([item["baseline_low"] for item in batch]),
        "vessel_low": torch.stack([item["vessel_low"] for item in batch]),
        "target_residual": torch.stack(
            [item["target_residual"] for item in batch]
        ),
        "horizon": torch.tensor(
            [item["horizon"] for item in batch], dtype=torch.float32
        ),
        "registration_confidence": torch.tensor(
            [item["registration_confidence"] for item in batch],
            dtype=torch.float32,
        ),
        "patient_id": [item["patient_id"] for item in batch],
        "eye_id": [item["eye_id"] for item in batch],
        "laterality": [item["laterality"] for item in batch],
        "disc_center": torch.stack([item["disc_center"] for item in batch]),
        "baseline_rnfl": torch.stack([item["baseline_rnfl"] for item in batch]),
        "future_rnfl": torch.stack([item["future_rnfl"] for item in batch]),
        "clinical_condition": torch.stack(
            [item["clinical_condition"] for item in batch]
        ),
        "annotation_masks": torch.stack([item["annotation_masks"] for item in batch]),
        "annotation_available": torch.tensor(
            [item["annotation_available"] for item in batch], dtype=torch.bool
        ),
        "stable_label": torch.tensor(
            [item["stable_label"] for item in batch], dtype=torch.bool
        ),
        "future_glaucoma_label": torch.tensor(
            [item["future_glaucoma_label"] for item in batch], dtype=torch.float32
        ),
    }


def _gradient_l1(left, right):
    torch = require_torch()
    return torch.nn.functional.l1_loss(
        left[..., 1:] - left[..., :-1],
        right[..., 1:] - right[..., :-1],
    ) + torch.nn.functional.l1_loss(
        left[..., 1:, :] - left[..., :-1, :],
        right[..., 1:, :] - right[..., :-1, :],
    )


def _structural_focus_l1(prediction, target, centers, superior_inferior_emphasis: float):
    torch = require_torch()
    height, width = target.shape[-2:]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=target.device),
        torch.linspace(0.0, 1.0, width, device=target.device),
        indexing="ij",
    )
    delta_x = grid_x[None] - centers[:, 0, None, None]
    delta_y = grid_y[None] - centers[:, 1, None, None]
    radius = torch.sqrt(delta_x.square() + delta_y.square())
    disc_and_rim = torch.exp(-radius.square() / (2.0 * 0.22**2))
    peripapillary = torch.exp(-((radius - 0.32) ** 2) / (2.0 * 0.09**2))
    vertical_sector = (delta_y.abs() >= delta_x.abs()).float()
    weights = 1.0 + disc_and_rim + peripapillary
    weights = weights * (
        1.0 + (superior_inferior_emphasis - 1.0) * vertical_sector * disc_and_rim
    )
    error = (prediction - target).abs().mean(dim=1)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def _photographic_feature_loss(prediction, target):
    torch = require_torch()

    def descriptors(image):
        green = image[:, 1:2]
        gray = image.mean(dim=1, keepdim=True)
        local = torch.nn.functional.avg_pool2d(green, 3, 1, 1)
        background = torch.nn.functional.avg_pool2d(green, 15, 1, 7)
        rnfl_contrast = torch.relu(background - local)
        horizontal = torch.nn.functional.pad(
            green[..., 1:] - green[..., :-1], (0, 1, 0, 0)
        ).abs()
        vertical = torch.nn.functional.pad(
            green[..., 1:, :] - green[..., :-1, :], (0, 0, 0, 1)
        ).abs()
        vessel_edges = horizontal + vertical
        return torch.cat([rnfl_contrast, vessel_edges, gray], dim=1)

    return torch.nn.functional.l1_loss(descriptors(prediction), descriptors(target))


def _binary_auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def train_registered_residual_diffusion(
    config: ResidualTrainingConfig,
) -> dict[str, object]:
    torch = require_torch()
    seed_everything(config.seed)
    device = select_device(config.device)
    if config.residual_size % 4:
        raise ValueError("residual_size must be divisible by four")
    if not 0.0 <= config.transition_balance_power <= 1.0:
        raise ValueError("transition_balance_power must be between zero and one")
    if config.superior_inferior_emphasis < 1.0:
        raise ValueError("superior_inferior_emphasis must be at least one")
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if config.pair_manifest:
        pairs = pd.read_csv(config.pair_manifest)
        required_pairs = {
            "patient_id",
            "eye_id",
            "laterality",
            "baseline_visit_id",
            "future_visit_id",
            "baseline_image_path",
            "future_image_path",
            "horizon_years",
            "split",
        }
        missing_pairs = sorted(required_pairs - set(pairs.columns))
        if missing_pairs:
            raise ValueError(f"Pair manifest missing columns: {missing_pairs}")
    else:
        manifest = pd.read_csv(config.manifest)
        pairs = make_all_future_pairs(manifest)
    clinical_dim = 0
    if config.conditioning_manifest:
        conditioning = pd.read_csv(config.conditioning_manifest)
        required_conditioning = {"eye_id", config.conditioning_column}
        missing = sorted(required_conditioning - set(conditioning.columns))
        if missing:
            raise ValueError(f"Conditioning manifest missing columns: {missing}")
        conditioning = conditioning[["eye_id", config.conditioning_column]].rename(
            columns={config.conditioning_column: "progression_probability_3y"}
        )
        if conditioning.duplicated("eye_id").any():
            raise ValueError("Conditioning manifest must have one row per eye")
        pairs = pairs.merge(conditioning, on="eye_id", how="inner")
        pairs["progression_probability_3y"] = pd.to_numeric(
            pairs["progression_probability_3y"], errors="coerce"
        )
        if pairs.empty or pairs["progression_probability_3y"].isna().any():
            raise ValueError(
                "Conditioned diffusion requires at least one longitudinal pair "
                "with a finite out-of-fold progression score"
            )
        clinical_dim = 1
    annotation_columns = {
        "baseline_disc_mask_path",
        "baseline_cup_mask_path",
        "future_disc_mask_path",
        "future_cup_mask_path",
        "progression_label",
        "quality_acceptable",
    }
    annotated_training = annotation_columns.issubset(pairs.columns)
    if annotated_training:
        pairs = pairs[pairs["quality_acceptable"].astype(bool)].reset_index(drop=True)
        unknown_labels = sorted(
            set(pairs["progression_label"].astype(str))
            - {"progressor", "stable", "uncertain"}
        )
        if unknown_labels:
            raise ValueError(f"Unknown progression labels: {unknown_labels}")
    if config.pair_manifest:
        allowed_splits = {"train", "validation", "test"}
        unknown_splits = sorted(set(pairs["split"].astype(str)) - allowed_splits)
        if unknown_splits:
            raise ValueError(f"Pair manifest contains unknown splits: {unknown_splits}")
        patient_split_counts = (
            pairs.groupby("patient_id")["split"].nunique().sort_values(ascending=False)
        )
        if int(patient_split_counts.max()) > 1:
            raise ValueError("A patient appears in more than one pair-manifest split")
        train_pairs = pairs[pairs["split"] == "train"].reset_index(drop=True)
        val_pairs = pairs[pairs["split"] == "validation"].reset_index(drop=True)
        test_pairs = pairs[pairs["split"] == "test"].reset_index(drop=True)
        if train_pairs.empty or val_pairs.empty or test_pairs.empty:
            raise ValueError("Pair manifest produced an empty split")
    else:
        train_pairs, val_pairs, test_pairs = patient_split(
            pairs,
            config.seed,
            config.validation_patients,
            config.test_patients,
        )
    train_pairs.to_csv(out / "train_pairs.csv", index=False)
    val_pairs.to_csv(out / "validation_pairs.csv", index=False)
    test_pairs.to_csv(out / "locked_test_pairs.csv", index=False)
    rnfl_columns = [
        f"{visit}_rnfl_{sector}"
        for visit in ("baseline", "future")
        for sector in ("inferior", "superior", "nasal", "temporal")
    ]
    anatomy_supervised_pairs = (
        int(pairs[rnfl_columns].notna().all(axis=1).sum())
        if set(rnfl_columns).issubset(pairs.columns)
        else 0
    )
    dataset_args = (
        config.image_size,
        config.residual_size,
        config.max_change,
        config.max_supported_horizon,
        config.view_mode,
        config.disc_crop_fraction,
        clinical_dim,
        config.structural_feature_mode,
    )
    train_dataset = RegisteredResidualPairDataset(train_pairs, *dataset_args)
    val_dataset = RegisteredResidualPairDataset(val_pairs, *dataset_args)
    sampler = None
    transition_counts = {}
    if config.transition_balance_power > 0:
        label_columns = {"baseline_glaucoma_label", "future_glaucoma_label"}
        if not label_columns.issubset(train_pairs.columns):
            raise ValueError(
                "Transition-balanced sampling requires baseline and future glaucoma labels"
            )
        transitions = (
            train_pairs["baseline_glaucoma_label"].astype(int).astype(str)
            + "->"
            + train_pairs["future_glaucoma_label"].astype(int).astype(str)
        )
        transition_counts = transitions.value_counts().sort_index().to_dict()
        sample_weights = transitions.map(
            lambda label: float(transition_counts[label])
            ** (-config.transition_balance_power)
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(config.seed)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(sample_weights.to_numpy(), dtype=torch.double),
            num_samples=len(train_pairs),
            replacement=True,
            generator=sampler_generator,
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    structural_channels = 3 if config.structural_feature_mode == "glaucoma" else 1
    model = build_residual_diffusion(
        config.base_channels,
        clinical_dim,
        structural_channels=structural_channels,
        progression_head=config.progression_loss_weight > 0,
    ).to(device)
    segmenter = None
    if config.segmenter_checkpoint:
        segmenter_state = torch.load(
            config.segmenter_checkpoint, map_location=device, weights_only=False
        )
        if annotated_training and not config.allow_pseudo_anatomy:
            segmenter_metrics = segmenter_state.get("metrics", {})
            disc_dice = segmenter_metrics.get("grape_validation_disc_dice")
            cup_dice = segmenter_metrics.get("grape_validation_cup_dice")
            if disc_dice is None or cup_dice is None:
                raise ValueError(
                    "Annotated training requires held-out GRAPE segmenter metrics"
                )
            if float(disc_dice) < config.minimum_segmenter_disc_dice:
                raise ValueError("GRAPE disc Dice is below the anatomy release threshold")
            if float(cup_dice) < config.minimum_segmenter_cup_dice:
                raise ValueError("GRAPE cup Dice is below the anatomy release threshold")
        segmenter_base_channels = int(
            segmenter_state.get("config", {}).get("base_channels", 24)
        )
        segmenter = build_disc_cup_segmenter(segmenter_base_channels).to(device)
        segmenter.load_state_dict(segmenter_state["model"])
        segmenter.eval()
        for parameter in segmenter.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    _, alpha_bars = diffusion_schedule(config.diffusion_steps, device)
    start_step = 0
    history = []
    best_val = math.inf
    best_state = None
    if config.resume:
        state = torch.load(config.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        history = state.get("history", [])
        best_val = float(state.get("best_val_residual_mae", math.inf))
        best_state = state.get("best_state")
    elif config.initialize_from:
        state = torch.load(
            config.initialize_from, map_location=device, weights_only=False
        )
        source_state = state.get("best_state") or state["model"]
        current_state = model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)

    def save(path: Path, step: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "history": history,
                "best_val_residual_mae": best_val,
                "best_state": best_state,
                "config": asdict(config),
                "model_version": (
                    f"registered-residual-diffusion-v11-multitask-{config.view_mode}-step-{step}"
                ),
                "warning": WARNING,
            },
            path,
        )

    def validate() -> dict[str, float]:
        model.eval()
        model_error = persistence_error = count = 0.0
        progression_scores: list[float] = []
        progression_labels: list[int] = []
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed + 1)
        with torch.no_grad():
            for batch in val_loader:
                baseline = batch["baseline_low"].to(device)
                vessel = batch["vessel_low"].to(device)
                target = batch["target_residual"].to(device)
                horizon = batch["horizon"].to(device)
                clinical_condition = batch["clinical_condition"].to(device)
                timestep_index = torch.full(
                    (target.shape[0],),
                    config.diffusion_steps // 2,
                    device=device,
                    dtype=torch.long,
                )
                alpha = alpha_bars[timestep_index][:, None, None, None]
                noise = torch.randn(
                    target.shape, generator=generator, device=device
                )
                noisy = torch.sqrt(alpha) * target + torch.sqrt(1 - alpha) * noise
                predicted_noise = model(
                    noisy,
                    baseline,
                    vessel,
                    horizon,
                    timestep_index.float() / max(1, config.diffusion_steps - 1),
                    clinical_condition,
                )
                clean = reconstruct_clean(noisy, predicted_noise, alpha).clamp(
                    -1, 1
                )
                model_error += float((clean - target).abs().sum().cpu())
                persistence_error += float(target.abs().sum().cpu())
                count += float(target.numel())
                if config.progression_loss_weight > 0:
                    labels = batch["future_glaucoma_label"].to(device)
                    valid = torch.isfinite(labels)
                    if valid.any():
                        logits = model.predict_progression(
                            baseline,
                            vessel,
                            horizon,
                        )
                        progression_scores.extend(
                            torch.sigmoid(logits[valid]).detach().cpu().tolist()
                        )
                        progression_labels.extend(
                            labels[valid].to(torch.int64).cpu().tolist()
                        )
        model.train()
        metrics = {
            "val_residual_mae": model_error / count,
            "val_persistence_residual_mae": persistence_error / count,
            "val_improvement_over_persistence": (
                persistence_error - model_error
            )
            / count,
        }
        if progression_scores:
            metrics["val_progression_auc"] = _binary_auc(
                progression_scores, progression_labels
            )
        return metrics

    step = start_step
    model.train()
    while step < config.max_steps:
        for batch in train_loader:
            step += 1
            baseline = batch["baseline_low"].to(device)
            vessel = batch["vessel_low"].to(device)
            target = batch["target_residual"].to(device)
            horizon = batch["horizon"].to(device)
            clinical_condition = batch["clinical_condition"].to(device)
            indices = torch.randint(
                0, config.diffusion_steps, (target.shape[0],), device=device
            )
            alpha = alpha_bars[indices][:, None, None, None]
            noise = torch.randn_like(target)
            noisy = torch.sqrt(alpha) * target + torch.sqrt(1 - alpha) * noise
            predicted_noise = model(
                noisy,
                baseline,
                vessel,
                horizon,
                indices.float() / max(1, config.diffusion_steps - 1),
                clinical_condition,
            )
            clean = reconstruct_clean(noisy, predicted_noise, alpha).clamp(-1, 1)
            noise_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            residual_loss = torch.nn.functional.l1_loss(clean, target)
            edge_loss = _gradient_l1(clean, target)
            if config.structural_focus_loss_weight > 0:
                structural_focus_loss = _structural_focus_l1(
                    clean,
                    target,
                    batch["disc_center"].to(device),
                    config.superior_inferior_emphasis,
                )
            else:
                structural_focus_loss = clean.sum() * 0.0
            if config.anatomy_loss_weight > 0:
                sector_masks = disc_sector_masks(
                    target.shape[-2], target.shape[-1], batch["disc_center"].to(device),
                    batch["laterality"], config.anatomy_inner_radius, config.anatomy_outer_radius,
                )
                anatomy_weights, anatomy_available = rnfl_decline_weights(
                    batch["baseline_rnfl"].to(device), batch["future_rnfl"].to(device),
                    torch.ones(target.shape[0], dtype=torch.bool, device=device),
                    config.anatomy_sector_emphasis,
                )
                anatomy_loss = sector_weighted_residual_loss(
                    clean, target, sector_masks, anatomy_weights, anatomy_available,
                )
            else:
                anatomy_loss = clean.sum() * 0.0
                anatomy_available = torch.zeros(
                    target.shape[0], dtype=torch.bool, device=device
                )
            predicted_future = (baseline + clean * config.max_change).clamp(0.0, 1.0)
            target_future = (baseline + target * config.max_change).clamp(0.0, 1.0)
            if config.feature_reconstruction_loss_weight > 0:
                feature_reconstruction_loss = _photographic_feature_loss(
                    predicted_future,
                    target_future,
                )
            else:
                feature_reconstruction_loss = predicted_future.sum() * 0.0
            if config.progression_loss_weight > 0:
                progression_labels = batch["future_glaucoma_label"].to(device)
                progression_valid = torch.isfinite(progression_labels)
                if progression_valid.any():
                    progression_logits = model.predict_progression(
                        baseline,
                        vessel,
                        horizon,
                    )
                    progression_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        progression_logits[progression_valid],
                        progression_labels[progression_valid],
                    )
                else:
                    progression_loss = predicted_future.sum() * 0.0
            else:
                progression_loss = predicted_future.sum() * 0.0
            if segmenter is not None:
                disc_cup_loss, cup_ratio_loss, segmenter_available = frozen_disc_cup_loss(
                    segmenter,
                    predicted_future,
                    target_future,
                    batch["disc_center"].to(device),
                    config.view_mode,
                )
            else:
                disc_cup_loss = predicted_future.sum() * 0.0
                cup_ratio_loss = predicted_future.sum() * 0.0
                segmenter_available = torch.zeros(
                    target.shape[0], dtype=torch.bool, device=device
                )
            annotation_available = batch["annotation_available"].to(device)
            if segmenter is not None and annotation_available.any():
                annotation_masks = batch["annotation_masks"].to(device)
                (
                    annotated_anatomy_loss,
                    annotated_ratio_loss,
                    outer_disc_loss,
                    stability_loss,
                ) = annotated_disc_cup_loss(
                    segmenter,
                    predicted_future,
                    batch["disc_center"].to(device),
                    config.view_mode,
                    annotation_masks[:, 0:2],
                    annotation_masks[:, 2:4],
                    annotation_available,
                    batch["stable_label"].to(device),
                )
            else:
                annotated_anatomy_loss = predicted_future.sum() * 0.0
                annotated_ratio_loss = predicted_future.sum() * 0.0
                outer_disc_loss = predicted_future.sum() * 0.0
                stability_loss = predicted_future.sum() * 0.0

            # Synthetic zero-horizon examples enforce exact baseline identity.
            zero_noise = torch.randn_like(target)
            zero_indices = torch.randint(
                0, config.diffusion_steps, (target.shape[0],), device=device
            )
            zero_alpha = alpha_bars[zero_indices][:, None, None, None]
            zero_noisy = torch.sqrt(1 - zero_alpha) * zero_noise
            zero_prediction = model(
                zero_noisy,
                baseline,
                vessel,
                torch.zeros_like(horizon),
                zero_indices.float() / max(1, config.diffusion_steps - 1),
                clinical_condition,
            )
            identity_loss = torch.nn.functional.mse_loss(
                zero_prediction, zero_noise
            )
            loss = (
                noise_loss
                + config.residual_loss_weight * residual_loss
                + config.edge_loss_weight * edge_loss
                + config.structural_focus_loss_weight * structural_focus_loss
                + config.feature_reconstruction_loss_weight * feature_reconstruction_loss
                + config.progression_loss_weight * progression_loss
                + config.identity_loss_weight * identity_loss
                + config.anatomy_loss_weight * anatomy_loss
                + config.disc_cup_loss_weight * disc_cup_loss
                + config.cup_ratio_loss_weight * cup_ratio_loss
                + config.annotated_anatomy_loss_weight
                * (annotated_anatomy_loss + annotated_ratio_loss)
                + config.outer_disc_preservation_weight * outer_disc_loss
                + config.stability_loss_weight * stability_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % config.validate_every == 0:
                metrics = validate()
                row = {
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "train_noise_loss": float(noise_loss.detach().cpu()),
                    "train_residual_loss": float(residual_loss.detach().cpu()),
                    "train_structural_focus_loss": float(
                        structural_focus_loss.detach().cpu()
                    ),
                    "train_feature_reconstruction_loss": float(
                        feature_reconstruction_loss.detach().cpu()
                    ),
                    "train_progression_loss": float(progression_loss.detach().cpu()),
                    "train_anatomy_loss": float(anatomy_loss.detach().cpu()),
                    "anatomy_supervised_pairs": int(anatomy_available.sum().cpu()),
                    "train_disc_cup_loss": float(disc_cup_loss.detach().cpu()),
                    "train_cup_ratio_loss": float(cup_ratio_loss.detach().cpu()),
                    "segmenter_supervised_pairs": int(segmenter_available.sum().cpu()),
                    "annotated_supervised_pairs": int(annotation_available.sum().cpu()),
                    "train_annotated_anatomy_loss": float(
                        annotated_anatomy_loss.detach().cpu()
                    ),
                    "train_outer_disc_loss": float(outer_disc_loss.detach().cpu()),
                    "train_stability_loss": float(stability_loss.detach().cpu()),
                    **metrics,
                }
                history.append(row)
                print(row, flush=True)
                if metrics["val_residual_mae"] < best_val:
                    best_val = metrics["val_residual_mae"]
                    best_state = {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    }
                    save(out / "best.pt", step)
            if step % config.checkpoint_every == 0:
                save(out / "latest.pt", step)
                save(out / f"checkpoint_step_{step:06d}.pt", step)
            if step >= config.max_steps:
                break
    save(out / "latest.pt", step)
    summary = {
        "status": "training_complete",
        "architecture": f"registered_residual_diffusion_v11_multitask_{config.view_mode}",
        "steps": step,
        "all_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "locked_test_pairs": len(test_pairs),
        "patients": int(pairs["patient_id"].nunique()),
        "pair_manifest": config.pair_manifest,
        "structural_feature_mode": config.structural_feature_mode,
        "structural_channels": structural_channels,
        "transition_balance_power": config.transition_balance_power,
        "training_transition_counts": transition_counts,
        "feature_reconstruction_loss_weight": config.feature_reconstruction_loss_weight,
        "progression_loss_weight": config.progression_loss_weight,
        "cohort_counts": (
            pairs["cohort"].value_counts().to_dict()
            if "cohort" in pairs.columns
            else {"longitudinal": len(pairs)}
        ),
        "maximum_observed_horizon_years": float(pairs["horizon_years"].max()),
        "repeated_rnfl_rows_available_but_not_used": anatomy_supervised_pairs,
        "anatomy_supervision": (
            "automatic_longitudinal_pseudo_masks_plus_frozen_segmenter"
            if config.allow_pseudo_anatomy and annotated_training
            else "human_longitudinal_contours_plus_frozen_segmenter"
            if config.segmenter_checkpoint
            and annotated_training
            else "frozen_PAPILA_expert_contour_segmenter_target_matching"
            if config.segmenter_checkpoint
            else "none"
        ),
        "segmenter_checkpoint": config.segmenter_checkpoint,
        "pseudo_anatomy_labels": config.allow_pseudo_anatomy,
        "best_validation_residual_mae": best_val,
        "best_checkpoint": str(out / "best.pt"),
        "latest_checkpoint": str(out / "latest.pt"),
        "recursive_rollout": False,
        "warning": WARNING,
        "history": history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
