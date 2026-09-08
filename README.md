# glaucoma-future-diffusion

Research-grade prototype for longitudinal glaucoma forecasting from color fundus
photographs. This repository is not validated for patient care and must not be
used for clinical diagnosis, triage, or treatment decisions.

## RetinaProgress-3Y multimodal endpoint

The current primary research endpoint is approximately three-year
visual-field progression in eyes with known glaucoma. The official GRAPE
workbook is retained in full: 1,115 visual-field visits produce a frozen cohort
of 72 eyes from 38 patients, including 16 PLR2 progressors whose complete
released follow-up lies between 2.5 and 3.5 years.

```bash
python scripts/build_three_year_multimodal_cohort.py

# Frozen model ladder. Omit --execute to inspect commands.
python scripts/run_progression_3y_matrix.py

# Direct risk-conditioned diffusion uses out-of-fold scores, never true labels.
python scripts/train_residual_diffusion_v3.py \
  --conditioning-manifest outputs/training/progression_3y_multimodal/oof_predictions.csv \
  --conditioning-column probability \
  --output-dir outputs/training/grape_residual_diffusion_3y_conditioned
```

See `docs/progression_3y_protocol.md` for the endpoint, release gates, and
single-optometrist reader-study limitations. A failed gate causes abstention;
it is not repaired by repeatedly tuning against held-out patients.

The project supports two clearly separated modes:

- Forecasting mode: uses historical images, historical times, and a requested
  future horizon. Target-visit diagnosis labels are prohibited.
- Controlled synthesis mode: optionally accepts an explicit disease condition
  for counterfactual generation or augmentation experiments. These outputs must
  not be reported as genuine forecasts.

## Dataset status

SIGF is supported through an adapter, but the official DeepGF project states
that SIGF is available upon request. Place approved data under `data/sigf/` and
include either a common manifest CSV or the legacy DeepGF folder layout.

GRAPE support is optional. Place downloaded GRAPE files under `data/grape/` with
metadata CSV/XLSX files and fundus images.

Missing fields are stored as null. The code never fabricates dates, labels,
patient identifiers, images, or clinical measurements.

## Quick checks

```bash
python scripts/audit_dataset.py --dataset sigf --root data/sigf --output-dir outputs/audit
python scripts/preprocess.py --manifest outputs/audit/manifest.csv --output-dir outputs/preprocessed --limit 8
python -m unittest discover -s tests
```

If private dataset files are absent, scripts stop cleanly and print setup
instructions.

## Example inference interface

```bash
python scripts/predict.py \
  --images visit_1.jpg visit_2.jpg visit_3.jpg \
  --times 0.0 0.9 2.1 \
  --horizon-years 1.0 \
  --num-samples 16 \
  --output-dir outputs/example_multi
```

The prediction interface writes a machine-readable JSON report with a
research-only disclaimer. Single-image forecasts are supported as a fallback and
emit a warning because they contain less patient-specific longitudinal context.

## Current implementation slice

Implemented now:

- common manifest schema and validation;
- SIGF/GRAPE dataset adapters with defensive setup errors;
- dataset audit outputs: JSON, CSV, HTML, and normalized manifest;
- patient-level splitting with leakage checks and grouped folds;
- sequence pairing with irregular continuous time handling;
- preprocessing primitives for field-of-view crop, orientation handling, and
  registration overlay placeholders;
- model/training/evaluation entrypoints with safe PyTorch dependency checks;
- unit tests for schema, splitting, timing, leakage, pairing, orientation, and
  label-leakage guards.

## V2: 512px single-image incidence-risk pipeline

The repository now also contains a quality-gated 512×512 pipeline designed for
future incident-glaucoma research. It uses a dual full-field/optic-disc model,
discrete annual survival hazards, explicit right-censoring, optional supervised
anatomy heads, ensemble uncertainty, and a separate visual-scenario generator.

The existing GRAPE checkpoint is **not** a V2 risk checkpoint. With no approved
incident-glaucoma checkpoint, V2 preprocesses the image and explicitly withholds
risk instead of returning an untrained prediction.

```bash
python scripts/predict_risk_v2.py \
  --image data/grape/images/1_OD_1.jpg \
  --output-dir outputs/v2_preprocessing_check
```

See [docs/v2_glaucoma_risk.md](docs/v2_glaucoma_risk.md) for cohort contracts,
training commands, inference outputs, and release gates.

## V3: fine-detail longitudinal residual diffusion

V3 directly addresses the washed-out detail of the earlier scenario
autoencoder. It keeps the real 512px baseline photograph intact and learns only
registered longitudinal change with a 256px residual diffusion model. All
requested years are generated directly from the real baseline; synthetic
images are never recursively reused as observations.

The local GRAPE release produces 648 real earlier-to-later pairs from 106
patients with observed intervals up to 4.8339 years. Years 5–10 remain visibly
marked unsupported extrapolations.

```bash
python scripts/train_residual_diffusion_v3.py \
  --manifest data/grape/manifest.csv \
  --output-dir outputs/training/grape_residual_diffusion_v3_512_mps \
  --device mps
```

See [docs/v3_residual_diffusion.md](docs/v3_residual_diffusion.md) for the
architecture, evaluation, generation commands, and safety contract.

## Geometry-first expert progression

The current visible-progression experiment replaces automatic longitudinal
pseudo-masks with the official GRAPE ophthalmologist OD/OC polygons. A
time-conditioned diffeomorphic model expands the optic cup through a smooth
geometry field and renders by warping the original native-resolution ROI; a
small appearance branch cannot create the structural endpoint.

```bash
python3 scripts/build_grape_expert_anatomy_manifest.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_anatomy_deformation.py \
  --manifest outputs/annotations/grape_expert_anatomy_v1/manifest.csv \
  --output-dir outputs/training/grape_anatomy_deformation_v2_sharp_holdout \
  --steps 700 --image-size 192 --base-channels 16 --max-displacement 8
python3 scripts/generate_anatomy_deformation.py \
  --checkpoint outputs/training/grape_anatomy_deformation_v2_sharp_holdout/best.pt \
  --output-dir outputs/diffusion_forecast/grape_best_expert_progressor_final
```

The final held-out preview is
`outputs/diffusion_forecast/grape_best_expert_progressor_final/original_then_generated.png`.
Its report records a generated VCDR change of `+0.109`, outer-disc Dice `0.966`,
positive-Jacobian fraction `1.000`, and passage of all automatic gates. This is
a research simulation rather than a clinical prognosis.

## SCOPE-Progress: integrated study

The current project connects three separately evaluated components: the
quality/current-disease gate, a patient-grouped GRAPE progression-
susceptibility classifier, and V4 dual-scale residual diffusion. The detail
branch models an optic-disc crop at twice the effective spatial sampling
density while retaining the nonrecursive direct-from-baseline contract.

The exact supported claim, endpoints, frozen cohort counts, clinician-study
design, and manuscript completion gates are in
[docs/scope_progress_project.md](docs/scope_progress_project.md).
An evidence-aligned AMIA-style abstract is in
[docs/draft_abstract.md](docs/draft_abstract.md).

```bash
python scripts/train_grape_progression_cv.py \
  --output-dir outputs/training/grape_progression_cv_full \
  --image-size 256 \
  --backbone convnext_tiny_imagenet \
  --epochs 8 \
  --device mps

python scripts/train_residual_diffusion_v3.py \
  --output-dir outputs/training/grape_residual_diffusion_v4_detail \
  --view-mode optic_disc \
  --initialize-from outputs/training/grape_residual_diffusion_v3_512_mps/best.pt \
  --device mps
```

## Newly trained current-glaucoma gate

HYGD 1.1.0 has been downloaded, checksum-verified, and used to train a 512px
shared ConvNeXt-Tiny current-glaucoma research gate. The calibrated held-out
patient AUROC is 0.921. This checkpoint intentionally leaves 2/5/10-year risk
null because it has no longitudinal incidence supervision.

```bash
python scripts/predict_risk_v2.py \
  --image example.jpg \
  --output-dir outputs/current_gate_example \
  --risk-checkpoints \
    outputs/training/hygd_current_gate_convnext_tiny_512_mps/best_calibrated.pt
```

The combined public/controlled-access dataset strategy and results are recorded
in [docs/data_solution.md](docs/data_solution.md). The prepared OHTS application
materials are in [docs/ohts_data_access_request.md](docs/ohts_data_access_request.md).

## RetinaProgress-3Y: gated progression forecasting

The repository now includes a frozen 2.5--3.5-year GRAPE progression cohort,
clinical/fundus/multimodal model ladder, calibrated inference contract,
patient-bootstrap validation, and a deterministic release gate. It also
contains an exploratory Harvard-GDP RNFL thickness-map benchmark using the
publisher's fixed split.

The primary GRAPE models did not pass the release gate. The best
known-glaucoma Harvard-GDP result was exploratory AUROC 0.658 and did not pass
the full discrimination/specificity rule. Diagnostic and conditioned-diffusion
outputs are therefore withheld; the clinician app remains a blinded image
realism pilot. See
[docs/progression_3y_status.md](docs/progression_3y_status.md) for exact results.

## Balanced V5 diffusion and clinician pilot

The newest experiment trains residual diffusion with an exactly balanced
manifest: 648 real longitudinal GRAPE glaucoma pairs plus 648 PAPILA healthy
zero-change stability controls. PAPILA does not contain longitudinal follow-up;
these controls test whether the generator invents change and must not be
described as real normal progression.

Validation-only amplitude calibration was frozen before locked testing. On 94
locked GRAPE glaucoma pairs, global V5 improved MAE from 0.03382 (persistence)
to 0.03313 (paired bootstrap 95% CI for the difference -0.00083 to -0.00057).
On 94 locked healthy controls, generated mean absolute change was 0.00331,
below the prespecified 0.005 stability limit. Mixed-cohort overall MAE does not
beat persistence because an exact copy is optimal for a zero-change control.

The deployed 20-case reader pilot is balanced between glaucoma and healthy
stability controls, balances the hidden reference side, records diagnosis and
progression judgments, and keeps its answer key server-side. Symmetric
side-by-side, blink, and change-highlight views help readers inspect subtle
differences without altering the underlying study photographs. Responses are
saved per case to private Vercel storage and can be exported as CSV from the
protected `/admin` page.

<https://clinician-evaluation.vercel.app>

This release is for blinded formative image evaluation only. It does not
release patient-specific prognosis, glaucoma-onset prediction, or clinical
diagnosis.
