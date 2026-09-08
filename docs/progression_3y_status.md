# Three-year progression implementation status

## Implemented

- Checksum-verified official GRAPE VF/clinical workbook.
- Complete 1,115-visit extraction, including visits without fundus images.
- Frozen 2.5--3.5-year PLR2 cohort and balanced patient-level outer folds.
- Clinical-only, fundus-only, and multimodal late-fusion model ladder.
- Inner-fold checkpoint selection, temperature scaling, operating thresholds,
  out-of-fold predictions, and patient-clustered bootstrap intervals.
- Versioned `BaselineExam` and `ProgressionForecastReport` contracts.
- Quality/OOD, known-glaucoma, and multimodal-completeness abstention.
- Optional out-of-fold risk conditioning for direct residual diffusion.
- Direct baseline-to-year-three generation; no recursive rollout.

## Release rule

The clinician-facing progression output and simulated image must remain
disabled until the frozen discrimination, calibration, diffusion, and reader
study gates are met. Infrastructure failures may be retried. Locked scientific
results may not be repeatedly tuned until they appear positive.

## Frozen results

The primary GRAPE cohort contains 72 eyes from 38 patients and 16 PLR2
progressors. None of the three primary candidates met the release rule:

| Candidate | AUROC (95% CI) | Sensitivity | Specificity |
| --- | --- | ---: | ---: |
| Clinical | 0.454 (0.274--0.667) | 0.188 | 0.821 |
| Fundus | 0.485 (0.310--0.665) | 0.000 | 1.000 |
| Multimodal | 0.454 (0.272--0.626) | 0.000 | 1.000 |

The official Harvard-GDP OCT archive was downloaded and checksum-verified.
On its known-glaucoma subset, the prespecified VF/demographic model reached
AUROC 0.652 (0.536--0.759). An exploratory RNFL-map plus VF/demographic model
reached AUROC 0.658 (0.535--0.765), sensitivity 0.820, and specificity 0.429.
This OCT analysis is post-hoc because the publisher test split had already
been examined. It is also not a fixed three-year fundus endpoint. It therefore
cannot release the primary output.

The mixed-status Harvard-GDP population achieved AUROC 0.821 on a secondary
pointwise endpoint, but that result mixes glaucoma and non-glaucoma eyes and
does not establish prognosis among patients who already have glaucoma.

The direct diffusion implementation remains available for research training,
but disease-conditioned generation is intentionally suppressed while the
progression gate is closed. The deployed clinician tool remains a blinded
realism pilot and explicitly withholds diagnostic forecasts.
