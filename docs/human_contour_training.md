# Expert-annotated progression training

RetinaProgress does not treat brightness, crop, or scale differences as glaucoma progression. The primary workflow now uses the ophthalmologist OD/OC polygons distributed with GRAPE. Manual contour tracing remains optional, and automatic pseudo-masks are not used for the geometry release.

## 1. Build the official expert manifest

```bash
python3 scripts/build_grape_expert_anatomy_manifest.py
```

The importer rasterizes the official LabelMe OD/OC polygons, validates cup containment, selects one longest pair of at least three years per eye, computes VCDR and ISNT-sector rim measurements, and preserves patient-isolated train/validation/test splits. Patient 12 is frozen as the sharp held-out visual demo.

## 2. Fine-tune segmentation

```bash
python3 scripts/train_segmentation.py \
  --papila-root data/external/papila/PAPILA \
  --longitudinal-annotations outputs/annotations/grape_expert_anatomy_v1/manifest.csv \
  --initialize-from outputs/training/papila_disc_cup_segmenter_v3_weighted/best.pt \
  --output-dir outputs/training/grape_expert_disc_cup_segmenter_v2
```

The completed fine-tune reaches GRAPE validation disc Dice `0.920` and cup Dice `0.848`, exceeding the `0.90` and `0.75` gates.

## 3. Train geometry-first progression

The deformation model predicts a smooth stationary velocity field plus a tightly bounded appearance residual. A VCDR-conditioned radial geometry prior guarantees that requested progression enlarges the cup rather than changing brightness. Stable controls penalize invented motion, and the same integrated field supplies every time point.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_anatomy_deformation.py \
  --manifest outputs/annotations/grape_expert_anatomy_v1/manifest.csv \
  --output-dir outputs/training/grape_anatomy_deformation_v2_sharp_holdout \
  --steps 700 --image-size 192 --base-channels 16 --max-displacement 8
```

## 4. Generate the held-out preview

```bash
python3 scripts/generate_anatomy_deformation.py \
  --checkpoint outputs/training/grape_anatomy_deformation_v2_sharp_holdout/best.pt \
  --output-dir outputs/diffusion_forecast/grape_best_expert_progressor_final
```

The generator automatically selects `GRAPE_12_OD_3.400y`, renders the original native pixels, and writes only one original-to-five-year comparison. Validation-only calibration controls the requested progression strength; the held-out future photograph is never used to generate the image.

## 5. Release interpretation

The final report records generated VCDR change `+0.109`, target recovery `1.226`, outer-disc Dice `0.966`, outside-disc MAE `0.0039`, and positive-Jacobian fraction `1.000`. Passing these gates means the output is a target-grounded research simulation with measurable visible change. It does not establish patient-specific prognosis or clinical validity.
