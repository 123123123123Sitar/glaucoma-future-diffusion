#!/usr/bin/env python3
"""Build a blinded 1:1 glaucoma/healthy scenario-realism clinician study.

Glaucoma cases use real GRAPE validation follow-ups as the reference candidate.
Healthy PAPILA cases use an unchanged baseline copy as a disclosed
pseudo-longitudinal stability reference because PAPILA is cross-sectional.
The public manifest contains no diagnosis or answer key.
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
from glaucoma_forecast.inference.residual_diffusion_predictor import (
    generate_direct_trajectory,
)


def _case_id(source: str, eye_id: str, index: int) -> str:
    digest = hashlib.sha256(
        f"balanced-v3:{source}:{eye_id}:{index}".encode()
    ).hexdigest()
    return f"B-{digest[:7].upper()}"


def _select_unique(
    frame: pd.DataFrame, count: int, minimum_horizon: float = 0.0
) -> pd.DataFrame:
    eligible = frame[frame["horizon_years"] >= minimum_horizon].copy()
    eligible = eligible.sort_values(
        ["horizon_years", "patient_id"], ascending=[False, True]
    )
    return eligible.drop_duplicates("patient_id").head(count).reset_index(drop=True)


def _observed_followup(row: dict[str, object], image_size: int) -> Image.Image:
    baseline = preprocess_high_resolution(str(row["baseline_image_path"]), image_size)
    future = preprocess_high_resolution(str(row["future_image_path"]), image_size)
    shift_x, shift_y, confidence = estimate_translation(
        baseline.full_field,
        future.full_field,
        max_shift=max(4, image_size // 12),
    )
    registered = (
        apply_translation(future.full_field, shift_x, shift_y)
        if confidence >= 3.0
        else future.full_field
    )
    return match_color_statistics(registered, baseline.full_field)


def _mae(left: Image.Image, right: Image.Image) -> float:
    a = np.asarray(left.convert("RGB"), dtype=np.float32)
    b = np.asarray(right.convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def _blind_clean(image: Image.Image) -> Image.Image:
    """Apply the same crop to every study image and remove generator banners."""

    rgb = image.convert("RGB")
    return rgb.crop((0, 28, rgb.width, rgb.height)).resize(
        (512, 512), Image.Resampling.BICUBIC
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-manifest", default="data/derived/balanced_stability_pairs_v1.csv"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--detail-checkpoint")
    parser.add_argument(
        "--study-output", default="outputs/clinician_balanced_v3"
    )
    parser.add_argument(
        "--public-output", default="clinician-evaluation/public/cases-balanced-v3"
    )
    parser.add_argument(
        "--public-manifest", default="clinician-evaluation/public/cases/manifest.json"
    )
    parser.add_argument("--cases-per-class", type=int, default=10)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=25)
    parser.add_argument("--global-change-scale", type=float, default=1.0)
    parser.add_argument("--detail-change-scale", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pair_manifest)
    validation = pairs[pairs["split"] == "validation"].copy()
    glaucoma = _select_unique(
        validation[
            validation["cohort"] == "grape_glaucoma_longitudinal"
        ],
        args.cases_per_class,
        minimum_horizon=2.0,
    )
    healthy_pool = validation[
        validation["cohort"] == "papila_healthy_stability_control"
    ].copy()
    healthy_pool = healthy_pool.sort_values(
        ["horizon_years", "patient_id"], ascending=[False, True]
    )
    healthy = healthy_pool.drop_duplicates("patient_id").head(
        args.cases_per_class
    )
    if len(glaucoma) != args.cases_per_class or len(healthy) != args.cases_per_class:
        raise RuntimeError("Not enough unique validation patients for balanced study")

    study_output = Path(args.study_output)
    raw_output = study_output / "raw"
    public_output = Path(args.public_output)
    raw_output.mkdir(parents=True, exist_ok=True)
    public_output.mkdir(parents=True, exist_ok=True)
    public_cases: list[dict[str, object]] = []
    private_cases: list[dict[str, object]] = []

    selected = [
        ("glaucoma", row)
        for row in glaucoma.to_dict("records")
    ] + [
        ("healthy", row)
        for row in healthy.to_dict("records")
    ]
    for index, (diagnosis, row) in enumerate(selected):
        case_id = _case_id(diagnosis, str(row["eye_id"]), index)
        raw_case = raw_output / case_id
        public_case = public_output / case_id
        raw_case.mkdir(exist_ok=True)
        public_case.mkdir(exist_ok=True)
        horizon = float(row["horizon_years"])
        baseline_artifact = preprocess_high_resolution(
            str(row["baseline_image_path"]), 512
        ).full_field
        if diagnosis == "glaucoma":
            reference_candidate = _observed_followup(row, 512)
            reference_type = "observed_longitudinal_followup"
        else:
            reference_candidate = baseline_artifact.copy()
            reference_type = "pseudo_longitudinal_unchanged_stability_control"

        generation = generate_direct_trajectory(
            image_path=str(row["baseline_image_path"]),
            checkpoint_path=args.checkpoint,
            detail_checkpoint_path=args.detail_checkpoint,
            output_dir=str(raw_case / "generation"),
            years=[horizon],
            trajectories=args.trajectories,
            sampling_steps=args.sampling_steps,
            device=args.device,
            seed=args.seed + index,
            global_change_scale=args.global_change_scale,
            detail_change_scale=args.detail_change_scale,
            clinical_condition=1.0 if diagnosis == "glaucoma" else 0.0,
        )
        generated = Image.open(generation["representative_images"][0]).convert("RGB")
        baseline_artifact = _blind_clean(baseline_artifact)
        reference_candidate = _blind_clean(reference_candidate)
        generated = _blind_clean(generated)
        baseline_artifact.save(public_case / "reference.webp", "WEBP", quality=95)
        reference_side = "A" if index % 2 == 0 else "B"
        if reference_side == "A":
            candidate_a, candidate_b = reference_candidate, generated
        else:
            candidate_a, candidate_b = generated, reference_candidate
        candidate_a.save(public_case / "candidate-a.webp", "WEBP", quality=95)
        candidate_b.save(public_case / "candidate-b.webp", "WEBP", quality=95)

        public_cases.append(
            {
                "caseId": case_id,
                "horizonYears": round(horizon, 2),
                "laterality": str(row["laterality"]),
                "baseline": f"/cases-balanced-v3/{case_id}/reference.webp",
                "candidateA": f"/cases-balanced-v3/{case_id}/candidate-a.webp",
                "candidateB": f"/cases-balanced-v3/{case_id}/candidate-b.webp",
            }
        )
        private_cases.append(
            {
                "caseId": case_id,
                "diagnosis": diagnosis,
                "referenceSide": reference_side,
                "referenceType": reference_type,
                "sourcePatient": str(row["patient_id"]),
                "sourceEye": str(row["eye_id"]),
                "horizonYears": horizon,
                "generatedChangeMae": _mae(baseline_artifact, generated),
                "referenceChangeMae": _mae(
                    baseline_artifact, reference_candidate
                ),
                "generatedVsReferenceMae": _mae(generated, reference_candidate),
            }
        )

    public_manifest = {
        "schemaVersion": "3.0",
        "studyType": "balanced blinded scenario-realism and stability pilot",
        "lockedTestUsed": False,
        "caseCount": len(public_cases),
        "warning": (
            "Research-only simulations. Glaucoma cases use observed validation "
            "follow-ups; healthy cases use disclosed unchanged stability controls. "
            "This study does not validate patient-specific prognosis."
        ),
        "cases": public_cases,
    }
    Path(args.public_manifest).write_text(
        json.dumps(public_manifest, indent=2), encoding="utf-8"
    )
    private_report = {
        "schemaVersion": "3.0",
        "studyVersion": "retina-progress-balanced-v3",
        "checkpoint": args.checkpoint,
        "detailCheckpoint": args.detail_checkpoint,
        "counts": {"glaucoma": len(glaucoma), "healthy": len(healthy)},
        "referenceSideCounts": pd.Series(
            [case["referenceSide"] for case in private_cases]
        ).value_counts().to_dict(),
        "cases": private_cases,
        "limitations": [
            "PAPILA healthy controls are cross-sectional zero-change controls.",
            "Only GRAPE glaucoma cases have true longitudinal follow-ups.",
            "Validation patients are used; locked test patients remain untouched.",
            "Clinician ratings measure realism and class preservation, not prognosis.",
        ],
    }
    (study_output / "private-answer-key.json").write_text(
        json.dumps(private_report, indent=2), encoding="utf-8"
    )
    print(json.dumps(private_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
