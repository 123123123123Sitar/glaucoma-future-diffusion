# SIGF structural single-image diffusion model

## Research question

Can one baseline fundus photograph support a useful, time-specific simulation of
future glaucoma-related appearance at 1, 3, or 5 years?

The model receives one photograph and a requested future year. It does not receive
the true future glaucoma label, a future photograph, or another visit at inference.

## Photographic findings considered

1. **Optic cup and disc geometry.** The optic cup is the brighter central depression
   inside the optic disc. Vertical cup-to-disc ratio (VCDR) is the cup's vertical
   diameter divided by the disc's vertical diameter. Enlargement can be important,
   but disc size and image quality must also be considered.
2. **Neuroretinal rim thinning and notching.** Glaucoma may produce diffuse rim loss
   or a focal notch. Superior and inferior changes are particularly important.
3. **ISNT pattern.** A typical healthy rim is widest inferiorly, then superiorly,
   nasally, and temporally. Violation can be suspicious, but it is not diagnostic by
   itself and varies with disc anatomy.
4. **Retinal nerve fiber layer defects.** Wedge-shaped or widening dark defects can
   appear around the disc. A red-free, green-channel contrast map helps the model
   see broad RNFL patterns while retaining the original RGB photograph.
5. **Disc hemorrhage.** A small hemorrhage at or near the disc is associated with
   greater progression risk, but it can be temporary and is uncommon. It is visible
   in the RGB crop but is not separately labeled in SIGF.
6. **Peripapillary atrophy.** Tissue and pigment changes surrounding the disc can
   enlarge over time. The crop intentionally includes the peripapillary region.
7. **Vessel displacement.** Rim loss may alter vessel position or produce bending,
   bayoneting, baring, or nasalization. A vessel-contrast channel is supplied to the
   model without editing the photograph.
8. **Pallor and excavation appearance.** Local brightness is supplied as a separate
   conditioning channel. Brightness alone is not treated as glaucoma because camera
   exposure can produce similar changes.

## Model changes

- Uses a true optic-disc-centered crop covering 36% of the retinal field. A prior
  implementation stored this setting but accidentally used the default 50% crop.
- Mirrors left-eye crops during training so nasal and temporal sectors have a
  consistent orientation. This is a reversible orientation normalization, not an
  anatomical alteration.
- Conditions diffusion on three maps: vessel contrast, broad red-free RNFL contrast,
  and robust local pallor.
- Gives extra loss weight to the disc, rim, peripapillary annulus, and superior and
  inferior sectors.
- Uses inverse-frequency weighted sampling for `0->0`, `0->1`, and `1->1` training
  pairs. Validation and locked testing retain their natural frequencies.
- Never supplies the future class label to the model. Balancing changes which real
  pairs are sampled during training; it does not leak the answer.
- Adds a horizon-aware auxiliary head that estimates future glaucoma probability
  from the same single baseline crop. The true future label is a training target,
  never an inference input.
- Adds differentiable reconstruction losses for vessel edges, broad RNFL contrast,
  and pallor so matching those structures matters in addition to raw pixels.

## Current held-out result

With two 15-step trajectories and a change scale fitted only on validation data,
the multitask model was statistically tied with baseline persistence overall. Its
MAE was 0.13% worse overall, with a patient-bootstrap confidence interval spanning
zero. It improved MAE by 0.48% on `0->1` conversions, 0.16% near three years, and
0.70% near five years. The auxiliary future-glaucoma AUC was 0.573 on this test.

These are improvements over the earlier full-field model, which was 13.8% worse
than persistence overall and 18.4% worse on `0->1` conversions. They are not large
enough to claim accurate clinical prediction. Visual review also shows that some
real converting eyes have much larger cup changes than the model generates.

## Evidence required before a positive claim

The redesigned model must beat baseline-image persistence on the locked patient test
set overall and within `0->1` conversions. It must also produce better sampled SSIM
and MAE, plausible 1/3/5-year ordering, and masked expert review results. Automatic
cup outlines are exploratory because the available cross-dataset cup segmenter has
only 0.562 validation Dice. Expert SIGF disc/cup annotations are still needed for a
defensible VCDR endpoint.

## Clinical sources

1. European Glaucoma Society. Terminology and Guidelines for Glaucoma, 4th Edition,
   Part 1. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5583682/>
2. Chauhan et al. Features of optic disc progression in ocular hypertension and
   early glaucoma. <https://pubmed.ncbi.nlm.nih.gov/23719180/>
3. Kim et al. Fundus photography and glaucoma conversion in eyes with large disc
   cupping. <https://pmc.ncbi.nlm.nih.gov/articles/PMC9810728/>
4. Kim et al. Disc hemorrhage findings and glaucoma progression.
   <https://pubmed.ncbi.nlm.nih.gov/32887917/>
5. Lee et al. Progression patterns of localized RNFL defects on red-free photographs.
   <https://pubmed.ncbi.nlm.nih.gov/25120342/>
