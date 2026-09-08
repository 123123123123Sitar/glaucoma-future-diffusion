# Current training status

Updated 2026-07-29.

- Global registered residual diffusion V3 completed 3,000 steps. Its best
  validation residual MAE is 0.09644 versus 0.16396 for persistence.
- V4 dual-scale optic-disc training and composition are implemented. The
  initialization smoke run completed and beat persistence on its tiny
  validation subset; this is only an interface check.
- The GRAPE progression cohort is frozen at 263 eyes from 144 patients with 40
  PLR2-positive eyes and patient-grouped five-fold assignments.
- Five-fold progression smoke training completed. Its one-epoch aggregate
  out-of-fold AUROC was 0.504, as expected for an optimization smoke test; it is
  not a scientific result.
- All 29 repository tests pass.
- The full five-fold ImageNet-initialized progression run completed with
  aggregate out-of-fold AUROC 0.466 and AUPRC 0.146. This is a negative result:
  that model does not support a progression-susceptibility claim.
- The full V4 detail run completed, initialized from the global V3
  checkpoint. Its best validation residual MAE was 0.0915 versus 0.1766 for
  persistence.
- On the locked 20-patient test split, neither generator beat persistence:
  global V3 MAE was 0.0394 versus 0.0338; optic-disc V4 MAE was 0.0412 versus
  0.0331. The visual models therefore remain realism experiments rather than
  validated forecasts.
- A validation-only shrinkage calibration improved the comparison. Calibrated
  V4 tied persistence overall (0.033052 versus 0.033058; paired CI includes
  zero) and outperformed persistence in the exploratory >3-year stratum of 14
  pairs (mean paired MAE difference -0.00119; 95% bootstrap CI -0.00201 to
  -0.00038). This subgroup requires confirmation and multiplicity-aware
  analysis.
- A second five-fold progression experiment initialized from the trained HYGD
  retinal encoder completed with AUROC 0.455 and AUPRC 0.148. It is a
  prespecified transfer-learning comparison and also a negative result.
- SCOPE-Fundus remains safely paused after its first resumable epoch checkpoint.

Research prototype only. No current checkpoint is validated for patient care.
