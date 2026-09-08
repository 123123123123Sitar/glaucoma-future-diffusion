# V3 registered residual diffusion

This branch generates research-only visual progression scenarios for eyes
already diagnosed with glaucoma. It is not an incident-glaucoma detector and
is not validated for diagnosis, prognosis, screening, treatment, or
reassurance.

## Why residual diffusion

The earlier 512px autoencoder regenerated the entire photograph and produced
broad texture artifacts. V3 preserves the observed 512px baseline exactly and
diffuses only the registered longitudinal change at 256px. The sampled change
is bounded and added back to the real baseline, so existing vessels, rim
texture, and camera detail are not needlessly synthesized.

Every requested year is conditioned directly on the real input image,
continuous elapsed time, and its vessel estimate. A generated year is never
fed back as a new observation.

## Data contract

- Source: GRAPE eyes already diagnosed with glaucoma.
- All 648 valid earlier-to-later pairs are constructed from real visits.
- 106 patients have at least two usable photographic visits.
- The maximum observed interval is 4.8339 years.
- Patients—not photographs or eyes—are assigned to train, validation, and
  locked test partitions.
- Target visits are translation-registered and color-matched to the baseline
  before the change target is calculated.

Years 1–4 fall within observed temporal support. Years 5–10 are emitted only
as prominently marked extrapolations; they are not evidence-backed forecasts.

## Train

```bash
python scripts/train_residual_diffusion_v3.py \
  --manifest data/grape/manifest.csv \
  --output-dir outputs/training/grape_residual_diffusion_v3_512_mps \
  --image-size 512 \
  --residual-size 256 \
  --base-channels 32 \
  --batch-size 1 \
  --max-steps 3000 \
  --checkpoint-every 100 \
  --validate-every 100 \
  --device mps
```

Training writes immutable pair splits, `latest.pt`, `best.pt`, periodic
checkpoints, and a summary. Resume with `--resume .../latest.pt`.

## Locked evaluation

```bash
python scripts/evaluate_residual_diffusion_v3.py \
  --checkpoint outputs/training/grape_residual_diffusion_v3_512_mps/best.pt \
  --pairs outputs/training/grape_residual_diffusion_v3_512_mps/locked_test_pairs.csv \
  --output-dir outputs/evaluation/grape_residual_diffusion_v3 \
  --device mps
```

The model must be compared with the persistence baseline; visual similarity
alone is insufficient. Ophthalmologist review of identity preservation,
vessels, disc/rim anatomy, artifacts, and temporal ordering remains required.

## Generate years 1–10

```bash
python scripts/generate_residual_trajectory_v3.py \
  --image example.jpg \
  --checkpoint outputs/training/grape_residual_diffusion_v3_512_mps/best.pt \
  --output-dir outputs/example_residual_v3 \
  --years 1 2 3 4 5 6 7 8 9 10 \
  --trajectories 8 \
  --sampling-steps 50 \
  --device mps
```

The output includes representative scenarios, uncertainty maps, a contact
sheet, and a JSON report recording temporal support and the non-recursive
generation contract.
