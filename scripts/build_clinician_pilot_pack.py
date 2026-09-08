#!/usr/bin/env python3
"""Build a blinded, validation-only clinician pilot image pack.

The locked test split is deliberately not used. Cases are selected by an
auditable rule: the longest well-registered interval per patient, ordered by
elapsed time, with a minimum observed structural-change threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.highres import preprocess_high_resolution
from glaucoma_forecast.data.preprocessing import match_color_statistics
from glaucoma_forecast.data.registration import apply_translation, estimate_translation
from glaucoma_forecast.evaluation.image_metrics import psnr, simple_ssim
from glaucoma_forecast.models._torch import require_torch
from glaucoma_forecast.models.residual_diffusion import (
    build_residual_diffusion,
    sample_residual_trajectory,
)
from glaucoma_forecast.training.residual_diffusion_trainer import _tensor
from glaucoma_forecast.utils.reproducibility import select_device


def _image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _mae(left: Image.Image, right: Image.Image) -> float:
    return float(np.mean(np.abs(_image_array(left) - _image_array(right))) / 255.0)


def _opaque_case_id(eye_id: str, visit_id: str) -> str:
    digest = hashlib.sha256(f"{eye_id}:{visit_id}:scope-pilot-v1".encode()).hexdigest()
    return f"R-{digest[:6].upper()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=25)
    parser.add_argument("--minimum-horizon", type=float, default=1.75)
    parser.add_argument("--minimum-observed-mae", type=float, default=0.02)
    parser.add_argument("--minimum-registration-confidence", type=float, default=3.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    torch = require_torch()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    image_size = int(config["image_size"])
    residual_size = int(config["residual_size"])
    max_change = float(config["max_change"])
    max_horizon = float(config["max_supported_horizon"])
    model = build_residual_diffusion(int(config["base_channels"])).to(device)
    model.load_state_dict(checkpoint.get("best_state") or checkpoint["model"])
    model.eval()

    pairs = pd.read_csv(args.pairs)
    screened: list[dict[str, object]] = []
    for row in pairs.to_dict("records"):
        baseline = preprocess_high_resolution(row["baseline_image_path"], image_size)
        future = preprocess_high_resolution(row["future_image_path"], image_size)
        shift_x, shift_y, registration_confidence = estimate_translation(
            baseline.full_field,
            future.full_field,
            max_shift=max(4, image_size // 12),
        )
        registered = (
            apply_translation(future.full_field, shift_x, shift_y)
            if registration_confidence >= args.minimum_registration_confidence
            else future.full_field
        )
        observed = match_color_statistics(registered, baseline.full_field)
        observed_mae = _mae(baseline.full_field, observed)
        screened.append(
            {
                **row,
                "registration_confidence": float(registration_confidence),
                "observed_mae": observed_mae,
                "_baseline": baseline,
                "_observed": observed,
            }
        )

    eligible = [
        row
        for row in screened
        if float(row["horizon_years"]) >= args.minimum_horizon
        and float(row["horizon_years"]) <= max_horizon
        and float(row["registration_confidence"])
        >= args.minimum_registration_confidence
        and float(row["observed_mae"]) >= args.minimum_observed_mae
    ]
    eligible.sort(
        key=lambda row: (float(row["horizon_years"]), float(row["observed_mae"])),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    seen_patients: set[str] = set()
    for row in eligible:
        patient = str(row["patient_id"])
        if patient in seen_patients:
            continue
        seen_patients.add(patient)
        selected.append(row)
        if len(selected) == args.cases:
            break
    if len(selected) < args.cases:
        raise RuntimeError(
            f"Only {len(selected)} eligible unique-patient cases; requested {args.cases}"
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for index, row in enumerate(selected):
        case_id = _opaque_case_id(str(row["eye_id"]), str(row["future_visit_id"]))
        case_dir = output / case_id
        case_dir.mkdir(exist_ok=True)
        baseline = row["_baseline"]
        observed = row["_observed"]
        horizon = float(row["horizon_years"])
        baseline.full_field.save(case_dir / "baseline.webp", "WEBP", quality=95)
        observed.save(case_dir / "observed.webp", "WEBP", quality=95)

        low = baseline.full_field.resize(
            (residual_size, residual_size), Image.Resampling.LANCZOS
        )
        vessel = baseline.vessel_map.resize(
            (residual_size, residual_size), Image.Resampling.BILINEAR
        )
        with torch.no_grad():
            samples = sample_residual_trajectory(
                model,
                _tensor(baseline.full_field).unsqueeze(0).to(device),
                _tensor(low).unsqueeze(0).to(device),
                _tensor(vessel, channels=1).unsqueeze(0).to(device),
                [horizon],
                max_horizon,
                max_change,
                trajectories=args.trajectories,
                diffusion_steps=args.sampling_steps,
                seed=args.seed + index,
            )[0, :, 0]
        generated_tensor = samples.mean(dim=0)
        generated_array = (
            generated_tensor.permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .clip(0, 1)
            * 255
        ).astype(np.uint8)
        generated = Image.fromarray(generated_array)
        generated.save(case_dir / "generated.webp", "WEBP", quality=95)
        uncertainty = samples.std(dim=0).mean(dim=0).detach().cpu().numpy()
        uncertainty_mean = float(uncertainty.mean())

        observed_array = _image_array(observed)
        generated_float = _image_array(generated)
        baseline_array = _image_array(baseline.full_field)
        manifest.append(
            {
                "caseId": case_id,
                "horizonYears": round(horizon, 2),
                "laterality": str(row["laterality"]),
                "baseline": f"/cases/{case_id}/baseline.webp",
                "observed": f"/cases/{case_id}/observed.webp",
                "generated": f"/cases/{case_id}/generated.webp",
                "quality": {
                    "registrationConfidence": round(
                        float(row["registration_confidence"]), 3
                    ),
                    "observedChangeMae": round(
                        float(np.mean(np.abs(observed_array - baseline_array)) / 255),
                        5,
                    ),
                    "generatedChangeMae": round(
                        float(np.mean(np.abs(generated_float - baseline_array)) / 255),
                        5,
                    ),
                    "generatedVsObservedMae": round(
                        float(np.mean(np.abs(generated_float - observed_array)) / 255),
                        5,
                    ),
                    "generatedVsObservedPsnr": round(
                        float(psnr(observed_array, generated_float)), 3
                    ),
                    "generatedVsObservedSsim": round(
                        float(simple_ssim(observed_array, generated_float)), 5
                    ),
                    "uncertaintyMean": round(uncertainty_mean, 5),
                },
            }
        )

    report = {
        "schemaVersion": "1.0",
        "studyType": "validation-only clinician pilot",
        "lockedTestUsed": False,
        "selectionRule": {
            "oneLongestEligibleIntervalPerPatient": True,
            "minimumHorizonYears": args.minimum_horizon,
            "minimumObservedMae": args.minimum_observed_mae,
            "minimumRegistrationConfidence": args.minimum_registration_confidence,
            "ordering": "horizon_descending_then_observed_change_descending",
        },
        "checkpointStep": int(checkpoint.get("step", 0)),
        "trajectories": args.trajectories,
        "samplingSteps": args.sampling_steps,
        "warning": (
            "Research-only simulated progression scenarios for already-diagnosed "
            "glaucoma eyes. Not for diagnosis, prognosis, or patient care."
        ),
        "cases": manifest,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
