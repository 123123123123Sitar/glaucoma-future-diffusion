# Model card

Planned model: compact longitudinal latent diffusion conditioned on historical
fundus images, continuous visit times, temporal availability masks, and forecast
horizon.

Outputs include generated sample distribution, VCDR estimate with uncertainty,
progression probability estimate, image-level uncertainty, and quality/OOD
warnings.

Limitations include small converter counts, irregular sampling, potential
camera/treatment confounding, label noise, and lack of clinical validation.

Research only: not validated for patient care.

## V2 model family

V2 is a separate single-image incident-risk design at 512×512 resolution:

- a quality/OOD gate;
- dual full-field and optic-disc encoders;
- current-glaucoma, annual discrete survival, VCDR, and disc/cup heads;
- a checkpoint ensemble for uncertainty;
- an optional identity-conditioned latent diffusion model for visual scenarios.

Generated images are not diagnostic evidence and are prohibited as inputs to
the risk model. Unsupervised heads and missing checkpoints produce null outputs,
not plausible-looking values. Ten-year risk is disabled operationally until an
external cohort supports ten-year calibration.
