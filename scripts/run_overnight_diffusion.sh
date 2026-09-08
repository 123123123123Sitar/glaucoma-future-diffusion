#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="outputs/training/balanced_residual_diffusion_v7_overnight_20260812"
SOURCE_CHECKPOINT="outputs/training/balanced_residual_diffusion_v7_disc_cup_detail/latest.pt"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ -f "$RUN_DIR/latest.pt" ]]; then
  SOURCE_CHECKPOINT="$RUN_DIR/latest.pt"
fi

python3 scripts/train_residual_diffusion_v3.py \
  --manifest data/grape/manifest.csv \
  --pair-manifest data/derived/balanced_stability_pairs_v2_isnt.csv \
  --output-dir "$RUN_DIR" \
  --image-size 512 \
  --residual-size 256 \
  --base-channels 32 \
  --batch-size 1 \
  --max-steps 18000 \
  --learning-rate 2e-5 \
  --diffusion-steps 100 \
  --max-change 0.18 \
  --max-supported-horizon 4.84 \
  --validation-patients 20 \
  --test-patients 20 \
  --checkpoint-every 250 \
  --validate-every 250 \
  --device mps \
  --seed 20260729 \
  --resume "$SOURCE_CHECKPOINT" \
  --view-mode optic_disc \
  --disc-crop-fraction 0.5 \
  --conditioning-manifest data/derived/balanced_stability_conditioning_v2_isnt.csv \
  --conditioning-column diagnosis_condition \
  --residual-loss-weight 0.75 \
  --edge-loss-weight 0.2 \
  --identity-loss-weight 0.1 \
  --anatomy-loss-weight 0 \
  --anatomy-sector-emphasis 3 \
  --anatomy-inner-radius 0.06 \
  --anatomy-outer-radius 0.3 \
  --segmenter-checkpoint outputs/training/papila_disc_cup_segmenter_v3_weighted/best.pt \
  --disc-cup-loss-weight 0.75 \
  --cup-ratio-loss-weight 1.5 \
  2>&1 | tee "$LOG_DIR/training.log"

CHECKPOINT="$RUN_DIR/best.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  CHECKPOINT="$RUN_DIR/latest.pt"
fi

python3 scripts/evaluate_residual_diffusion_v3.py \
  --checkpoint "$CHECKPOINT" \
  --pairs "$RUN_DIR/locked_test_pairs.csv" \
  --output-dir outputs/evaluation/balanced_residual_diffusion_v7_overnight_20260812 \
  --trajectories 4 \
  --sampling-steps 25 \
  --device mps \
  2>&1 | tee "$LOG_DIR/locked_test.log"

python3 scripts/generate_residual_trajectory_v3.py \
  --image data/grape/images/45_OD_1.jpg \
  --checkpoint "$CHECKPOINT" \
  --output-dir outputs/progression/balanced_residual_diffusion_v7_overnight_20260812_GRAPE_45_OD \
  --years 1 3 5 \
  --trajectories 8 \
  --sampling-steps 50 \
  --device mps \
  --clinical-condition 1 \
  --save-candidates \
  2>&1 | tee "$LOG_DIR/generation.log"

python3 - <<'PY'
from pathlib import Path
import pandas as pd

run_dir = Path("outputs/training/balanced_residual_diffusion_v7_overnight_20260812")
pairs = pd.read_csv(run_dir / "locked_test_pairs.csv")
selected = (
    pairs[pairs["cohort"].eq("grape_glaucoma_longitudinal")]
    .sort_values("horizon_years", ascending=False)
    .drop_duplicates("patient_id")
    .head(20)
)
assert len(selected) == 20
selected[["patient_id", "eye_id", "baseline_image_path"]].to_csv(
    run_dir / "heldout_progression_inputs.tsv", sep="\t", index=False, header=False
)
PY

while IFS=$'\t' read -r patient_id eye_id image_path; do
  python3 scripts/generate_residual_trajectory_v3.py \
    --image "$image_path" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "outputs/progression/heldout_20_eyes_20260812/${patient_id}_${eye_id}" \
    --years 1 3 5 \
    --trajectories 8 \
    --sampling-steps 50 \
    --device mps \
    --clinical-condition 1 \
    --save-candidates \
    2>&1 | tee -a "$LOG_DIR/heldout_20_generation.log"
done < "$RUN_DIR/heldout_progression_inputs.tsv"

date -u +%FT%TZ > "$RUN_DIR/OVERNIGHT_COMPLETE"
