# Implementation plan

Research prototype only. Not validated for patient care, diagnosis, triage, or
treatment decisions.

## 1. Discovered dataset structure

Workspace inspected on 2026-07-20 in `/Users/sitareswar`.

- Requested project directory did not exist before this implementation:
  `/Users/sitareswar/glaucoma-future-diffusion`.
- Existing unrelated/earlier project found:
  `/Users/sitareswar/glaucoma-diffusion`.
- That older project contains class-labeled PNG files under
  `/Users/sitareswar/glaucoma-diffusion/data/train`, with names such as
  `g_00465_2.png`. These look like isolated class-labeled images, not validated
  longitudinal SIGF/GRAPE sequences with patient IDs, eyes, visit dates, and
  interval metadata.
- No local `SIGF`, `GRAPE`, longitudinal fundus manifest, or timing metadata was
  found in the workspace search.

The official DeepGF repository describes SIGF as request-only and documents a
legacy layout: `data/train(test)/image(label)/all/`. The adapter therefore
prefers an explicit `manifest.csv` and only falls back to conservative image
discovery for that legacy layout. Fallback discovery still refuses training if
timing metadata is missing.

## 2. Missing files or metadata

Missing for actual training:

- approved SIGF files or GRAPE files;
- patient identifiers;
- eye identifiers and laterality;
- acquisition dates or continuous `time_from_baseline_years`;
- longitudinal diagnosis/progression labels;
- split definitions if official SIGF splits are available;
- disc/cup masks for segmentation/VCDR supervision;
- trained quality, segmentation, autoencoder, diffusion, and progression
  checkpoints.

The code must stop if timing, labels required for a selected experiment, or
image files are absent. It must not infer private clinical facts from filenames.

## 3. Proposed common manifest

The common manifest columns are:

```text
patient_id, eye_id, laterality, visit_id, visit_index, acquisition_date,
time_from_baseline_years, image_path, glaucoma_label, progression_label,
camera_device, image_width, image_height, quality_score, iop, rnfl_mean,
rnfl_superior, rnfl_inferior, rnfl_nasal, rnfl_temporal, visual_field_md,
source_dataset
```

Missing optional fields remain null. Required fields for forecasting examples
are `patient_id`, `eye_id`, `visit_id`, `image_path`, and
`time_from_baseline_years`.

## 4. Exact model architecture

Initial compact model target: 50–150M trainable parameters when fully widened,
with smaller smoke-test widths.

- Latent autoencoder:
  - RGB input at 256×256 initially;
  - convolutional encoder to a lower-resolution latent;
  - convolutional decoder;
  - reconstruction validation with PSNR, SSIM, LPIPS when installed,
    disc/cup Dice, and VCDR error when masks exist.
- Shared spatial encoder:
  - convolutional multiscale features per visit;
  - same weights across visits.
- Temporal conditioner:
  - Transformer encoder;
  - inputs: visit features, continuous times, forecast horizon, availability
    mask;
  - preserves visit order and accepts variable-length padded sequences.
- Diffusion denoiser:
  - compact latent 2D U-Net with factorized temporal conditioning for the first
    implementation;
  - objective configurable as epsilon or velocity;
  - EMA weights maintained during training.
- Progression head:
  - predicts progression probability from temporal representation, not from
    generated image alone;
  - focal/class-balanced loss in converter-sparse training.
- Anatomy heads:
  - VCDR regression and optional disc/cup segmentation when masks exist.

Forecasting mode excludes target labels. Controlled synthesis mode may accept a
requested class, and output directories/report filenames must keep it distinct.

## 5. Estimated memory requirements

Smoke configuration:

- 256×256 images;
- batch size 1;
- gradient accumulation 4;
- 64 base channels;
- 2 temporal layers;
- suitable for CPU/MPS forward/backward smoke tests if PyTorch is installed.

Full CUDA configuration:

- 384×384 images;
- batch size 8 initially, with automatic reduction on OOM;
- 128 base channels;
- 6 temporal layers;
- gradient checkpointing and mixed precision on CUDA only by default.

The Mac/MPS path must not attempt a 900M-parameter model. It is for audit,
preprocessing, small model debugging, and one-batch smoke tests.

## 6. Smoke-test plan

1. Run unit tests.
2. Run `scripts/audit_dataset.py` and verify clean setup error when private data
   is absent.
3. With any valid local manifest, run preprocessing on 8 images and inspect
   overlays.
4. If PyTorch is installed, run:

   ```bash
   python scripts/train_diffusion.py --config configs/sigf_small.yaml --dry-run
   ```

5. Run `scripts/predict.py` on sample images only as a persistence-baseline
   packaging test until trained checkpoints exist.

## 7. Full-training plan

1. Obtain SIGF via approved request procedure.
2. Build a common manifest with real dates/intervals and labels.
3. Run audit and resolve corrupt files, missing dates, duplicate images, low
   quality images, both-eye leakage risks, and split leakage.
4. Freeze immutable patient-level train/validation/test split CSVs.
5. Preprocess full-field and optic-disc-centered crops.
6. Train/validate quality and disc/cup segmentation models.
7. Train persistence, VCDR extrapolation, deterministic U-Net, CNN+temporal
   progression, single-image latent diffusion, and full longitudinal diffusion.
8. Evaluate on held-out patients by horizon, sequence length, stable controls,
   converters, and metadata subgroups with sample counts.
9. Export final reports and blinded ophthalmologist review packages.

## 8. Scientific risks and leakage risks

- SIGF has few converting sequences, so variance and calibration uncertainty are
  expected.
- Visits are irregular; rounding intervals to integer years would distort the
  task.
- Camera/device changes can be learned as shortcuts.
- Treatment can produce non-monotonic anatomy/clinical measurements.
- Target diagnosis labels in forecasting mode are label leakage and are blocked
  by code.
- Future visits must never appear in historical context during evaluation.
- Fellow eyes from the same patient must stay in the same split.
- Single-image prediction lacks patient-specific temporal trajectory and must be
  evaluated separately.
- Controlled synthesis is not unbiased forecasting and must be reported
  separately.
