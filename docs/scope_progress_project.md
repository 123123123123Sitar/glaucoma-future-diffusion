# SCOPE-Progress: frozen project scope

Updated 2026-07-31.

## One defensible research question

Can a quality-gated system use one color fundus photograph from an eye already
diagnosed with glaucoma to:

1. estimate susceptibility to documented progression during the available
   GRAPE follow-up; and
2. generate a blinded, anatomy-preserving visual scenario at a directly
   requested interval within the observed 0–4.83-year range?

This is a research question, not a clinical product claim. The local data do
not support predicting glaucoma onset in healthy eyes, a fixed 4.8-year event
probability, or ten-year progression. Generated images are hypotheses for
human review, not diagnostic evidence.

## The connected system

### SCOPE input gate

The observed photograph is processed at 512×512 as a full-field image and an
optic-disc crop. Quality checks, possible current-glaucoma classification, and
out-of-distribution/uncertainty logic determine whether the system should
abstain. The HYGD current-disease checkpoint has held-out patient AUROC 0.9211,
but its calibration and external-camera validity are not yet release-ready.

### GRAPE progression-susceptibility model

`data/grape/progression_baseline_cohort_v1.csv` contains exactly one earliest
photograph per eye:

- 263 eyes from 144 patients;
- 40 PLR2-positive and 223 PLR2-negative eyes;
- immutable patient-grouped five-fold assignments;
- no patient or fellow-eye leakage between folds.

The primary endpoint is the GRAPE eye-level PLR2 label: progression documented
during that eye's observed study follow-up. GRAPE does not provide an event
date for this label, so the output must not be described as survival risk at a
fixed horizon.

Primary metrics are out-of-fold AUROC and AUPRC. Secondary metrics are Brier
score, expected calibration error, sensitivity, specificity, and balanced
accuracy at a validation-derived threshold. Confidence intervals and subgroup
results are required before manuscript submission.

The first full ImageNet-initialized experiment is a negative result (out-of-fold
AUROC 0.466; AUPRC 0.146) and cannot support clinical usefulness. A
prespecified comparison using retinal-domain initialization from the HYGD gate
also failed (AUROC 0.455; AUPRC 0.148). The progression classifier should be
reported as a failed hypothesis or removed from the paper's primary claims.

### Dual-scale residual diffusion

The global branch learns a bounded 256×256 registered change field and adds it
to the real 512×512 baseline. The V4 detail branch learns the optic-disc region
at twice the effective sampling density. Its feathered contribution contains
only predicted change; it does not paste or regenerate a synthetic baseline.

Each requested horizon is sampled directly from the observed photograph.
Generated year 3 is never reused as the input for year 6. This prevents
recursive error accumulation. The evidence-supported temporal range ends at
4.8339 years; later images, if ever enabled, must be labeled unsupported
extrapolations.

Dual-scale scenarios are generated with:

```bash
python scripts/generate_residual_trajectory_v3.py \
  --image example.jpg \
  --checkpoint outputs/training/grape_residual_diffusion_v3_512_mps/best.pt \
  --detail-checkpoint outputs/training/grape_residual_diffusion_v4_detail/best.pt \
  --output-dir outputs/example_v4 \
  --years 1 2 3 4 4.83 \
  --global-change-scale 0.4877807877 \
  --detail-change-scale 0.2950849204
```

### Balanced blinded clinician evaluation

The current direct-link study contains 20 validation cases, not locked-test
cases: ten GRAPE glaucoma cases with a real observed follow-up and ten PAPILA
healthy stability controls. PAPILA is cross-sectional, so its reference is a
predefined unchanged copy of the baseline, not a later patient visit. This
explicit stability-control design tests whether the generator invents disease
or structural change in a healthy eye; it does not establish longitudinal
normal aging.

For each baseline, the reviewer sees a reference/control and a simulated
follow-up in blinded A/B positions and records:

- which candidate appears observed;
- normal, glaucoma, or uncertain for the baseline and both candidates;
- perceived structural progression in each candidate;
- findings that drove the assessment;
- confidence, elapsed time, and optional comments.

Responses are saved after every case in a central database. The public case
manifest uses opaque candidate names and contains no answer key. This is a
formative usability and realism study; one local optometrist is not sufficient
for an effectiveness claim. A manuscript experiment should preregister the
case-selection rule, include repeated cases for intrarater reliability, and
recruit multiple masked glaucoma specialists.

Clinician link: <https://retina-progress-clinician-pilot.percym788.chatgpt.site>

## Manuscript structure

The novelty is the combination of:

- explicit task/applicability separation;
- patient-grouped progression susceptibility from a single observed image;
- nonrecursive direct-time residual diffusion;
- dual-scale full-field/optic-disc detail preservation; and
- a leakage-resistant clinician realism protocol.

The classifier and generator remain separate. Classifier predictions cannot be
improved or validated by generated pixels, and clinician preference cannot
substitute for patient-level outcome evaluation.

## Current balanced-diffusion evidence

The balanced V5 experiment used 648 real GRAPE glaucoma pairs and 648 PAPILA
healthy zero-change controls. Original patient-grouped GRAPE splits were
preserved, and no patient crossed train, validation, or locked test splits.
Change amplitude was calibrated on validation data only.

On the locked 94-pair glaucoma subset, the global model achieved MAE 0.03313
versus 0.03382 for persistence (paired bootstrap difference -0.00070, 95% CI
-0.00083 to -0.00057; 86.2% pairwise wins). The detail model achieved MAE
0.03267 versus 0.03306 (difference -0.00039, 95% CI -0.00052 to -0.00026).
On 94 locked healthy controls, generated mean absolute change was 0.00331
global and 0.00313 detail, both below the prespecified 0.005 stability limit.
Persistence remains optimal for an unchanged control, so the mixed-cohort
overall MAE does not beat persistence.

An independent public glaucoma classifier produced the same class decision for
each baseline and its generated image (20/20 preservation) in the frozen reader
pack. Its absolute accuracy on these 20 selected cases was only 70%, so this is
a consistency audit, not proof of diagnostic validity or future truth.

The deterministic release gate therefore permits a balanced formative image
pilot only. Patient-specific prognosis, glaucoma-onset prediction, and clinical
use remain disabled.

## Required completion gates

1. Obtain an institutional IRB/non-human-subjects determination before
   recruiting clinicians or collecting publishable responses.
2. Preregister reader-study hypotheses, selection rules, exclusions, and the
   analysis plan; include repeated cases for intrarater reliability.
3. Recruit multiple masked glaucoma specialists and report discrimination,
   realism, diagnostic preservation, agreement, and confidence intervals.
4. Add a genuinely longitudinal healthy/normal-aging cohort; do not relabel
   PAPILA stability controls as real future images.
5. Quantify vessel/rim detail, identity preservation, temporal ordering,
   calibration, and failure cases on independent data.
6. Add external camera/site validation or clearly label the paper as
   development plus formative human evaluation.

Until those gates pass, the correct description is “research prototype.”

## Current evidence checkpoint

The first completed locked evaluation does not support future-image accuracy.
Global V3 had MAE 0.0394 versus 0.0338 for persistence; optic-disc V4 had MAE
0.0412 versus 0.0331. These models can still be studied for perceived realism
and localized anatomy, but they must not be called accurate forecasts unless a
future prespecified model beats persistence on locked patients and passes
clinical review.

Paired bootstrap analysis confirms that this is not sampling ambiguity: the
model-minus-persistence MAE was +0.00561 (95% bootstrap CI +0.00487 to
+0.00634) for V3 and +0.00815 (+0.00674 to +0.00958) for V4. The model beat
persistence on only 5.3% and 8.5% of locked pairs, respectively.

A validation-only change-amplitude calibration materially reduced
over-generation. Calibrated V4 tied persistence overall (MAE 0.033052 versus
0.033058; paired 95% bootstrap CI crossed zero) and showed an exploratory
advantage in the 14 locked pairs longer than three years: paired MAE difference
-0.00119 (95% CI -0.00201 to -0.00038). This was a horizon subgroup analysis,
not the primary endpoint, so it is hypothesis-generating rather than
confirmatory. Calibrated global V3 remained worse than persistence.
