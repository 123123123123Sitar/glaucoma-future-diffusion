# Evaluation protocol

Evaluate only on held-out patients. Keep both eyes from the same patient in the
same split.

Report:

- PSNR, SSIM, LPIPS when available;
- disc/cup Dice and VCDR MAE/correlation when annotations permit;
- progression AUROC, AUPRC, sensitivity, specificity, balanced accuracy, Brier
  score, calibration error, and reliability diagrams;
- results by forecast horizon and number of historical images;
- stable controls, converting eyes, and already-glaucomatous eyes separately
  when labels permit;
- uncertainty percentiles from at least 8 configurable samples.

Research only: not validated for patient care.

For the V2 incidence model, also report time-dependent discrimination and Brier
score at 2, 5, and 10 years, calibration intercept/slope, expected calibration
error, decision curves, censoring counts, and the number still observable at
each horizon. An entire site must remain external to development. Visual
scenario review must include vessel continuity, disc/cup consistency, VCDR
error, temporal ordering, uncertainty, and masked clinician grading.
