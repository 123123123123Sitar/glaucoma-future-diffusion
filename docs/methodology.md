# Methodology

This repository models longitudinal color fundus photographs as irregular
sequences. Training examples hide a real future visit and expose only earlier
visits, continuous visit times, a requested horizon, and optional historical
clinical metadata.

The primary evaluation mode is forecasting. Controlled synthesis is separate and
may accept a requested disease class for counterfactual or augmentation studies.
Controlled synthesis must not be reported as genuine future prediction.

Research only: not validated for patient care.

The V3 visual branch uses direct registered residual diffusion. Every real
earlier visit is paired with every later visit from the same eye, while all
eyes from a patient remain in one partition. The future image is registered
and color-normalized to the observed baseline. Diffusion models a bounded
256px change field, which is added to the untouched 512px baseline. Each
horizon is sampled from the real baseline with shared path noise; generated
frames are never recursively treated as observations.

V4 adds a second residual model over an optic-disc-centered crop. Relative to
the global branch, this doubles the effective sampling density for rim and
peripapillary change. At inference, only the detail model's predicted change is
feathered into the global scenario. The real baseline remains the identity
anchor for both branches.
