# OHTS controlled-access request package

## Why OHTS is required

The Ocular Hypertension Treatment Study is the primary cohort for the proposed
single-fundus incident-glaucoma model. It enrolled participants with normal
optic discs and visual fields at baseline, collected annual photographs, and
adjudicated primary open-angle glaucoma conversion over approximately 16 years.
Published work reports 66,721 photographs from 3,272 eyes of 1,636 subjects.

- Study evidence: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10586722/>
- NEI follow-up imaging schema:
  <https://brics.nei.nih.gov/dictionary/publicData/dataStructureAction%21lightboxView.ajax?dataStructureName=OHTS_ODRC_FUV&publicArea=true>
- NEI Data Commons access overview:
  <https://neidatacommons.nei.nih.gov/about>
- dbGaP accession: `phs000240.v1.p1`

Access requires an NEI BRICS account through Login.gov or ID.me, a Data User
Agreement, and potentially Data Access Committee review. Published users report
also obtaining IRB approval and permission from the OHTS Coordinating Center.

## Requested data fields

- Deidentified participant/random identifier and eye laterality.
- Baseline and follow-up optic-disc/fundus imaging files and file format.
- Photo date expressed relative to randomization.
- OHTS phase and visit type.
- Confirmed/suspected optic-disc progression and confirmation-photo status.
- POAG endpoint status, endpoint type, and event/confirmation date.
- Vertical and horizontal cup-to-disc ratios.
- Rim thinning, rim notch, disc hemorrhage, and photograph clarity grades.
- Treatment assignment and treatment changes, if permitted.
- Site/camera identifiers sufficient for domain-shift evaluation.
- Official exclusion/quality flags and recommended analysis splits, if any.

The model input at deployment remains one color fundus photograph. Non-image
clinical fields are requested only to define reliable outcomes, censoring,
confounding, and evaluation strata.

## Draft research summary

**Title:** Calibrated prediction of incident primary open-angle glaucoma from a
single baseline fundus photograph

**Objective:** Develop and externally validate a research-only model that
estimates cumulative 2-, 5-, and 10-year POAG risk in ocular-hypertension or
glaucoma-suspect eyes. A separate current-disease gate will prevent incident
risk from being shown when glaucoma may already be present.

**Methods:** Use patient-level and site-level separation; discrete-time survival
learning with right/interval censoring; image-quality abstention; calibration;
subgroup analysis; comparison with established OHTS risk factors; and blinded
review of any optional visual scenarios. Generated images will never be used as
classifier inputs or clinical labels.

**Outputs:** Aggregate research metrics, calibration curves, model cards, and
non-identifying model parameters. No participant-level data or images will be
redistributed. No clinical diagnosis, treatment recommendation, or commercial
use is proposed.

**Security:** Store controlled data only in the approved encrypted environment;
restrict access to named study personnel; retain access logs; prohibit copying
to public repositories; and destroy or return data according to the DUA.

## Draft email to the OHTS Coordinating Center

To: `kass@wustl.edu`

Subject: Research data request — longitudinal OHTS optic-disc photographs and
POAG conversion outcomes

Dear OHTS Coordinating Center,

I am requesting information on obtaining controlled research access to
deidentified OHTS baseline/follow-up optic-disc photographs and adjudicated POAG
conversion outcomes. The proposed non-commercial research will develop a
quality-gated, calibrated 2/5/10-year incident-risk model using one baseline
fundus photograph at inference. We will use patient/site-separated evaluation,
survival methods that handle censoring, external validation, and subgroup
analysis. Generated images, if studied, will remain separate from risk
estimation and will not be used for clinical decisions.

Could you please advise the current application route through the OHTS
Coordinating Center or NEI BRICS, required IRB/DUA documents, available imaging
and endpoint tables, and any fees or publication conditions?

Applicant: [name, title, institution]

Principal investigator: [name and institution]

IRB protocol or determination: [identifier/status]

Secure computing environment: [description]

Thank you,

[name and contact information]
