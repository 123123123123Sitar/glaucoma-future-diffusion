# RetinaProgress-3Y frozen protocol

## Primary question

Can a multimodal baseline examination estimate approximately three-year
visual-field progression in an eye with already diagnosed glaucoma, and does a
directly generated year-three fundus scenario add useful information for a
clinician beyond a calibrated numerical score?

This is not a study of glaucoma onset in healthy eyes. It does not claim that a
generated image is the patient's actual future.

## Frozen cohort

The official GRAPE workbook is parsed without discarding visits that lack a
fundus photograph. The primary cohort contains eyes with at least three visual
fields whose complete released follow-up lies between 2.5 and 3.5 years.
Progression is the publisher-provided PLR2 status. Because GRAPE supplies an
eye-level status rather than an event date, the endpoint is described as
approximately-three-year progression, not exact time-to-event risk.

The frozen v1 cohort contains 72 eyes from 38 patients and 16 progressors.
There are 1,115 source visual-field visits. Five outer folds are patient
disjoint and approximately balanced by eye count and progression status.

## Analysis

The prespecified ladder is clinical-only, transferred fundus-only, and
multimodal late fusion. Model and temperature selection occur inside each
outer fold. Only out-of-fold predictions enter aggregate metrics or clinician
cases. Report AUROC, AUPRC, Brier score, sensitivity, specificity, calibration,
and patient-clustered bootstrap intervals.

A research candidate requires AUROC at least 0.70 with its 95% interval above
0.50, sensitivity at least 0.80 at specificity at least 0.50, and Brier score
below the prevalence-only baseline. Every seed and fold is reported.

## Generative model

The residual diffusion model predicts each target time directly from the real
baseline. It may condition on an out-of-fold progression probability, but never
on the true progression label. Synthetic years are never fed back as clinical
observations. The validated interface supports year three only.

Generated images remain disabled unless they are no worse than unchanged-image
persistence on the full locked pair set and improve a prespecified structural
metric without degrading vessel/identity preservation.

## Clinician pilot

The planned single-optometrist crossover compares baseline alone, baseline plus
risk, and baseline plus risk and simulation against held-out PLR2 outcomes.
With one reader this is a feasibility estimate, not multi-reader clinical
validation. Generated-image realism is evaluated separately and is not called
forecast accuracy. An applicable ethics or institutional determination is
required before recruiting or publishing the reader study.

