# Draft abstract

## When Visual Plausibility Is Not Prognosis: Leakage-Resistant Evaluation of
Single-Image Glaucoma Progression Classification and Fundus Diffusion

### Background

Generative models can produce visibly changing retinal photographs, but visual
change is not evidence of patient-specific disease forecasting. We evaluated
whether a single baseline color fundus photograph could predict documented
progression in established glaucoma and support anatomy-preserving future-image
simulation.

### Methods

We constructed an eye-level cohort of 263 eyes from 144 GRAPE participants,
including 40 eyes with the PLR2 progression outcome. Patients and fellow eyes
were kept in the same five-fold partition. A dual-view full-field/optic-disc
classifier was evaluated with out-of-fold predictions. Separately, 648 real
earlier-to-later image pairs from 106 participants trained direct-time residual
diffusion models. The global model added a bounded 256-pixel change field to
the observed 512-pixel baseline; a dual-scale model added a higher-density
optic-disc change branch. Generated images were never recursively reused as
inputs. Change amplitude was fitted on validation pairs only. Evaluation used
94 locked pairs from 20 participants and paired bootstrap confidence intervals.

### Results

The ImageNet-initialized progression classifier did not discriminate
progressing eyes (AUROC 0.466; AUPRC 0.146). Retinal-domain initialization from
an external current-glaucoma model did not improve performance (AUROC 0.455;
AUPRC 0.148). Without amplitude calibration, global and optic-disc diffusion
were worse than persistence. After validation-only calibration, the optic-disc
model tied persistence overall (MAE 0.033052 versus 0.033058; paired confidence
interval included zero). In an exploratory subgroup of 14 pairs with intervals
longer than three years, optic-disc diffusion reduced MAE by 0.00119 (95%
bootstrap CI 0.00038–0.00201 in favor of the model). The calibrated global
model remained worse than persistence.

### Conclusion

In this small established-glaucoma cohort, neither single-image classification
nor diffusion supported a general patient-specific forecasting claim. A
dual-scale optic-disc branch showed a hypothesis-generating long-interval
signal, but the overall result demonstrates that plausible retinal change can
coexist with no improvement over persistence. The released protocol, abstention
logic, nonrecursive generator, and blinded clinician interface provide a
reproducible framework for falsifying—rather than assuming—clinical forecasting
claims.

Research prototype only; not validated for diagnosis, prognosis, or patient
care.
