#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="outputs/training/balanced_residual_diffusion_v7_overnight_20260812"
OUTPUT_DIR="outputs/progression/heldout_20_eyes_20260812"
LOG="$RUN_DIR/logs/heldout_20_generation.log"
CHECKPOINT="$RUN_DIR/best.pt"
mkdir -p "$OUTPUT_DIR"

PYTHONPATH=src python3 - <<'PY'
from pathlib import Path

import pandas as pd

from glaucoma_forecast.data.quality import assess_image_quality

run_dir = Path("outputs/training/balanced_residual_diffusion_v7_overnight_20260812")
pairs = pd.read_csv(run_dir / "locked_test_pairs.csv")
glaucoma = pairs[pairs["cohort"].eq("grape_glaucoma_longitudinal")]
selected = []
for patient_id, group in glaucoma.groupby("patient_id"):
    candidates = (
        group[["eye_id", "baseline_visit_id", "baseline_image_path"]]
        .drop_duplicates()
        .sort_values(["baseline_visit_id", "eye_id"])
    )
    accepted = None
    for row in candidates.itertuples(index=False):
        quality = assess_image_quality(row.baseline_image_path)
        if not quality.is_low_quality:
            accepted = row
            break
    if accepted is None:
        raise RuntimeError(f"No quality-passing baseline for {patient_id}")
    selected.append(
        {
            "patient_id": patient_id,
            "eye_id": accepted.eye_id,
            "image_path": accepted.baseline_image_path,
        }
    )

selection = pd.DataFrame(selected).sort_values("patient_id")
assert len(selection) == 20
selection.to_csv(run_dir / "heldout_progression_inputs.tsv", sep="\t", index=False, header=False)
PY

: > "$LOG"
while IFS=$'\t' read -r patient_id eye_id image_path; do
  case_dir="$OUTPUT_DIR/${patient_id}_${eye_id}"
  if [[ -f "$case_dir/report.json" ]]; then
    continue
  fi
  python3 scripts/generate_residual_trajectory_v3.py \
    --image "$image_path" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$case_dir" \
    --years 1 3 5 \
    --trajectories 8 \
    --sampling-steps 50 \
    --device mps \
    --clinical-condition 1 \
    --save-candidates \
    2>&1 | tee -a "$LOG"
done < "$RUN_DIR/heldout_progression_inputs.tsv"

date -u +%FT%TZ > "$OUTPUT_DIR/GENERATION_COMPLETE"
