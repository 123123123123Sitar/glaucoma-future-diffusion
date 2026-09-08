# Dataset card

## SIGF

Primary dataset adapter. SIGF is private/request-only through the official
DeepGF project. This repository expects a real local manifest with patient,
eye, visit, image, label, and timing metadata. Irregular intervals are preserved.

## GRAPE

Optional secondary dataset adapter for directly downloadable longitudinal
glaucoma records. GRAPE contains diagnosed glaucoma patients and should not be
treated as a healthy-control dataset.

Missing fields remain null and are never guessed.

Research only: not validated for patient care.
