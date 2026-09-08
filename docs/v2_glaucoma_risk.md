# V2 single-fundus glaucoma-risk pipeline

Research prototype only. It is not validated for diagnosis, screening, triage,
treatment, or patient-care decisions.

## What changed

The legacy 128-pixel GRAPE model remains available only as an experiment
baseline. It was trained on glaucoma eyes and cannot estimate healthy-to-
glaucoma incidence.

V2 separates two tasks:

1. An observed-image model consumes a 512×512 full-field view and a 512×512
   optic-disc crop. It estimates image quality, possible current glaucoma,
   annual incident-glaucoma hazards, cumulative 2/5/10-year risk, VCDR, and
   optional disc/cup masks.
2. A separate latent diffusion model can create annual visual scenarios. Its
   pixels never feed the risk model and every exported image is labeled as a
   simulation.

The system refuses to manufacture risk values when a V2 checkpoint is missing,
when the input fails quality checks, or when a supplied output head was not
supervised.

## Incidence cohort contract

`train_risk_v2.py` requires one baseline row per eye:

```text
patient_id,eye_id,image_path,event_observed,event_or_censor_time_years,
current_glaucoma_label,incidence_eligible,vcdr,disc_mask_path,cup_mask_path,
site_id,camera_device
```

The first five fields are required. Optional target fields remain empty; they
are never inferred from filenames. `event_observed=1` means adjudicated incident
glaucoma occurred in `(0, event_or_censor_time_years]`. `event_observed=0` means
the eye was last known event-free at that censor time.

`incidence_eligible` defaults to 1. Cross-sectional current-glaucoma examples
may be included with `incidence_eligible=0`; they supervise the current-disease
head without contributing to survival loss. Both positive and negative
current-glaucoma examples are required before inference exposes incident risk.

Do not create an incidence manifest from the local GRAPE images: all imported
GRAPE eyes are glaucoma-positive and the current local follow-up ends before
five years.

Training example:

```bash
python scripts/train_risk_v2.py \
  --manifest data/incidence/train.csv \
  --output-dir outputs/training/risk_v2_member_1 \
  --backbone convnext_small \
  --image-size 512 \
  --epochs 50 \
  --device cuda
```

Train five members with immutable patient/site splits and different seeds.
Calibrate on a distinct calibration partition, then store the fitted
temperature in each checkpoint. External-site evaluation is required before a
research release. A 10-year result must not be published unless enough held-out
eyes remain observable at 10 years.

Audit a candidate cohort before training:

```bash
python scripts/audit_incidence_cohort.py \
  --manifest data/incidence/all.csv \
  --output outputs/audit_incidence/report.json
```

## Scenario cohort contract

The visual model requires real longitudinal pairs:

```text
patient_id,baseline_image_path,future_image_path,future_year,
risk_model_cumulative_risk
```

`risk_model_cumulative_risk` must be produced by a frozen, out-of-fold risk
model from the baseline image. It must not be derived from the target visit or
its diagnosis label.

```bash
python scripts/train_scenario_v2.py \
  --manifest data/scenarios/train_pairs.csv \
  --output-dir outputs/training/scenario_v2 \
  --image-size 512 \
  --device cuda
```

## Inference

Without a checkpoint, this command still performs the 512-pixel preprocessing
and writes an explicit `risk_checkpoint_required` report:

```bash
python scripts/predict_risk_v2.py \
  --image example.jpg \
  --output-dir outputs/v2_example
```

Validated research checkpoints can be supplied as an ensemble:

```bash
python scripts/predict_risk_v2.py \
  --image example.jpg \
  --output-dir outputs/v2_example \
  --risk-checkpoints member1.pt member2.pt member3.pt member4.pt member5.pt \
  --scenario-checkpoint scenario.pt
```

The output JSON schema is `2.0`. It includes quality/OOD status, current-disease
applicability, annual hazards, 2/5/10-year cumulative risk, ensemble intervals,
and—when separately available—ten annual simulated images and uncertainty maps.

## Release gates

- At least 500 adjudicated incident cases and 2,000 non-converting eyes.
- Patient-level separation and a completely held-out external site.
- Reliable input-quality abstention and subgroup analysis by site, camera,
  demographic variables, and image quality.
- Time-dependent discrimination, Brier score, calibration, and clinical
  decision-curve reporting at 2, 5, and 10 years.
- Blinded review of generated anatomy by eye-care clinicians.
- No diagnostic or prevention claim without prospective validation and the
  applicable regulatory process.
