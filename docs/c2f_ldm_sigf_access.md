# C2F-LDM and SIGF access status

- The official C2F-LDM code is checked out at `external/C2F-LDM`.
- The code does not include SIGF images or pretrained C2F-LDM checkpoints.
- SIGF is distributed by its owners through a password-protected archive after
  an ethics-reviewed request. No SIGF credentials are present locally.
- The current unattended run therefore resumes the leakage-safe GRAPE/PAPILA
  residual-diffusion experiment. It does not claim to reproduce C2F-LDM.
- If SIGF is later provided legitimately, keep its patient sequences separate
  from GRAPE and create patient-level train, validation, and locked-test splits
  before adapting the C2F-LDM data loader.
