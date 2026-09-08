# RetinaProgress balanced clinician reader protocol

Version: `retina-progress-balanced-v3`  
Status: protocol and software prepared; human-subjects determination pending.

## Study purpose

This formative study tests whether a diffusion generator:

1. preserves glaucoma-versus-normal appearance;
2. avoids inventing structural progression in healthy stability controls; and
3. produces glaucoma follow-ups that are difficult to distinguish from observed
   validation follow-ups.

It does not test patient-specific prognosis, treatment benefit, or diagnostic
deployment.

## Cases

The frozen reader set contains 20 validation-only cases:

- 10 established-glaucoma eyes from GRAPE. The reference candidate is a real,
  registered, color-matched longitudinal follow-up.
- 10 healthy eyes from PAPILA. Because PAPILA is cross-sectional, the reference
  candidate is an unchanged baseline copy used as a pseudo-longitudinal
  stability control.

The two groups are balanced, no locked-test patients are used, and one source
patient contributes at most one case. Candidate A/B positions are balanced
within diagnosis and stored only in the server-side answer key.

## Reader task

For every case, readers receive a baseline and two blinded candidates. They
record:

- which candidate is the reference/control;
- normal, glaucoma, or uncertain classification for baseline and both
  candidates;
- none, possible, or definite structural progression for both candidates;
- visual findings, confidence, and an optional comment.

### Comparison aids

The interface applies the same comparison mode to candidate A and candidate B:

- side-by-side baseline and follow-up photographs;
- a synchronized 700 ms baseline/follow-up blink; and
- a fixed high-sensitivity absolute pixel-change map that uses dark gray for
  little change and amplified yellow/red color for larger change.

These aids do not modify the source study photographs and do not normalize A
and B independently. The fixed map scale makes small differences visible, but
its colors do not represent disease probability. It can highlight compression,
camera, illumination, and registration differences as well as anatomical
differences, so readers are instructed to confirm every judgment in the
original photographs. Both candidates receive identical processing to preserve
blinding.

## Response storage and access

Each completed case is written immediately to a private Vercel Blob store and
is also mirrored to the original D1 study database when that service is
available. The private record includes reviewer code, background, answers,
confidence, elapsed time, server-side truth fields, and timestamps. The public
manifest never contains the answer key.

The study owner can download a CSV from the password-protected results page:

<https://clinician-evaluation.vercel.app/admin>

The admin page is excluded from search indexing, and the export endpoint
returns `401` without the server-configured results passcode. Passcodes and
storage credentials must never be sent to clinicians or committed to source
control.

## Primary endpoints

1. Generated-image detection rate, where 50% is chance.
2. Disease-class preservation: agreement between the generated candidate rating
   and the source diagnosis.
3. Healthy false-progression rate: percentage of generated healthy controls
   rated as possible or definite progression.

## Secondary endpoints

- observed/reference classification agreement;
- confidence-weighted accuracy;
- inter-reader agreement;
- repeated-case intrarater agreement in a later preregistered version;
- subgroup results by reader specialty and experience.

## Analysis

Report reader- and case-clustered bootstrap 95% confidence intervals. Do not
pool multiple ratings as independent observations. Compare detection accuracy
with 50% chance and report disease-class preservation separately for normal and
glaucoma cases. The study is formative until multiple masked glaucoma
specialists complete it.

## Stopping and exclusion rules

- Freeze the manifest, answer key, checkpoint hashes, and analysis code before
  recruitment.
- Exclude incomplete sessions only; do not exclude individual cases after
  viewing responses.
- Report technical failures and missing responses.
- Do not enable a patient-facing forecast regardless of reader-study results.

## Ethics and privacy

Before recruiting clinicians or publishing their responses, obtain an
institutional IRB/exempt/not-human-subjects determination. Use de-identified
public research images only. Do not collect patient information. Reviewer codes
must not contain names or email addresses.

## Required limitations statement

PAPILA healthy cases are cross-sectional stability controls, not observed
longitudinal normal follow-ups. GRAPE contains established glaucoma. Therefore
this pilot evaluates realism, stability, and disease-class preservation—not
healthy-to-glaucoma conversion or patient-specific future anatomy.
