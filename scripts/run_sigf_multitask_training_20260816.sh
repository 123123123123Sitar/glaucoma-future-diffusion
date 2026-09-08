#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/.."

output_dir="outputs/training/sigf_multitask_diffusion_v3_20260816"
mkdir -p "$output_dir"

resume_args=(
  --initialize-from outputs/training/sigf_structural_diffusion_v2_20260816/best.pt
)
if [[ -f "$output_dir/latest.pt" ]]; then
  resume_args=(--resume "$output_dir/latest.pt")
fi

python3 scripts/train_residual_diffusion_v3.py \
  --manifest data/sigf/manifest.csv \
  --pair-manifest data/sigf/pairs_5y_leakage_safe.csv \
  --output-dir "$output_dir" \
  --image-size 256 \
  --residual-size 128 \
  --base-channels 32 \
  --batch-size 1 \
  --max-steps 12000 \
  --learning-rate 1e-5 \
  --diffusion-steps 100 \
  --max-change 0.18 \
  --max-supported-horizon 5.0 \
  --checkpoint-every 250 \
  --validate-every 2000 \
  --device mps \
  --seed 20260816 \
  --view-mode optic_disc \
  --disc-crop-fraction 0.36 \
  --structural-feature-mode glaucoma \
  --transition-balance-power 0.75 \
  --structural-focus-loss-weight 0.75 \
  --superior-inferior-emphasis 2.5 \
  --feature-reconstruction-loss-weight 0.5 \
  --progression-loss-weight 0.75 \
  --residual-loss-weight 0.75 \
  --edge-loss-weight 0.2 \
  --identity-loss-weight 0.05 \
  --anatomy-loss-weight 0 \
  --disc-cup-loss-weight 0 \
  --cup-ratio-loss-weight 0 \
  "${resume_args[@]}" 2>&1 | tee -a "$output_dir/training.log"
