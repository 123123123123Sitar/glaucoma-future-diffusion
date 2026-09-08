# Data solution for single-fundus glaucoma prediction

## Implemented immediately

### HYGD 1.1.0 — current-disease gate

- Source:
  <https://www.physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/>
- License: Open Data Commons Attribution.
- Local images: 747 from 288 patients.
- Labels: 548 GON-positive and 199 GON-negative.
- Label basis: comprehensive specialist examination, OCT, visual fields, and
  follow-up—not fundus appearance alone.
- Camera: TOPCON DRI OCT Triton, 45-degree field.
- Role: train the “possible glaucoma already present” gate.
- Not suitable for: time to onset or annual future-image supervision.

The archive and all extracted files were SHA-256 verified. The manifest keeps
all images from a patient in the same split. A shared ImageNet-pretrained
ConvNeXt-Tiny dual-view model was trained at 512×512.

Held-out result:

```text
patients: 58
images: 143
AUROC: 0.9211
temperature: 2.6697
calibrated Brier: 0.2047
operating threshold: 0.7026
sensitivity: 0.9032
specificity: 0.7800
```

Calibration error remains too high for clinical release. HYGD is also
single-camera and geographically limited, so multi-camera external validation
is mandatory.

### GRAPE — diagnosed-eye progression scenarios

- Source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10404253/>
- Role: model progression in eyes already diagnosed with glaucoma.
- Local support: 322 training pairs and 46 held-out pairs.
- Maximum observed pair interval: 4.43 years.
- Not suitable for: healthy-to-glaucoma conversion or validated years 5–10.

The 512px scenario checkpoint preserves observed fine detail, but visual QA
still finds broad texture artifacts at longer horizons. It remains a failed
clinical-plausibility experiment.

## Required controlled-access cohort

### OHTS — incident risk and pre-onset forecasting

OHTS is the direct solution to the missing outcome problem:

- 1,636 initially ocular-hypertensive participants;
- 3,272 eyes with normal optic discs and visual fields at baseline;
- 66,721 fundus photographs;
- annual imaging and approximately 16 years of follow-up;
- reading-center and endpoint-committee adjudication of POAG conversion.

Sources:

- <https://pmc.ncbi.nlm.nih.gov/articles/PMC10586722/>
- <https://brics.nei.nih.gov/dictionary/publicData/dataStructureAction%21lightboxView.ajax?dataStructureName=OHTS_ODRC_FUV&publicArea=true>
- <https://neidatacommons.nei.nih.gov/about>
- dbGaP accession `phs000240.v1.p1`

Access is controlled and may require Login.gov/ID.me, an NEI BRICS account,
IRB approval or determination, a Data User Agreement, and Data Access Committee
or OHTS Coordinating Center permission. The prepared request is in
`docs/ohts_data_access_request.md`.

## Excluded shortcuts

- Generated year-3 images cannot become new observations for year-6
  prediction. Recursive rollout compounds model artifacts.
- Harvard-GDP is OCT RNFL-map and visual-field data, not a fundus-photograph
  dataset; using it as the core model would violate the single-fundus input
  contract.
- Cross-sectional datasets can improve current glaucoma detection and anatomy
  segmentation but cannot create credible onset dates.
- Ten-year risk remains disabled until OHTS or an equivalent cohort supports
  external 10-year calibration.
