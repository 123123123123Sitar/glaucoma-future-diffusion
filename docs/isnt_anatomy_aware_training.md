# ISNT-aware longitudinal training

## Scientific interpretation

The ISNT rule describes the common normal ordering of neuroretinal rim width:
inferior, superior, nasal, then temporal. It is not a deterministic progression
sequence and must not be implemented by enlarging or shrinking four regions in
that order. Published evaluations report high sensitivity but low specificity,
and many normal eyes do not follow the complete rule.

The GRAPE clinical spreadsheet repeats baseline quadrant RNFL values across
visits. Those columns are not longitudinal measurements and must not be used as
RNFL-decline supervision. The v6 experiment mistakenly treated their presence
as longitudinal supervision; because every decline was zero, its ISNT term
reduced to uniform sector weighting and did not teach progression.

V7 instead trains a frozen disc/cup segmenter from PAPILA's two sets of genuine
expert contours. During diffusion training, the segmenter compares the
registered real target with the generated target. This is a model-derived
auxiliary image-structure signal on GRAPE, not OCT ground truth and not a
clinical segmentation claim. Implausible target masks are excluded.

Longitudinal evidence supports emphasizing real sectoral change: progressing
glaucoma eyes lose rim area faster than healthy eyes, with faster change often
in superior-temporal and inferior-temporal sectors. Serial-disc studies also
identify excavation, rim thinning, and inferotemporal change as common signs of
progression.

## Implementation

- `training/anatomy_losses.py` constructs laterality-aware I/S/N/T annular
  attention masks centered on the localized optic disc.
- `disc_cup_segmenter.py` predicts disc and cup probability maps from tight
  optic-disc crops after patient-grouped PAPILA training.
- `disc_cup_losses.py` matches the target probability maps and cup/disc area
  ratio while excluding implausible target masks.
- Repeated GRAPE RNFL values are retained only as baseline clinical metadata;
  their longitudinal loss weight defaults to zero.
- The original diffusion, edge, zero-horizon identity, registration, and healthy
  stability losses remain active.

## Retraining workflow

Rebuild the balanced pair manifest so it includes baseline and future quadrant
RNFL measurements:

```bash
python3 scripts/build_balanced_stability_pair_manifest.py \
  --output data/derived/balanced_stability_pairs_v2_isnt.csv \
  --report data/derived/balanced_stability_pairs_v2_isnt_report.json \
  --conditioning-output data/derived/balanced_stability_conditioning_v2_isnt.csv
```

Then train a new checkpoint; do not overwrite the locked v5 checkpoint:

```bash
python3 scripts/train_residual_diffusion_v3.py \
  --pair-manifest data/derived/balanced_stability_pairs_v2_isnt.csv \
  --conditioning-manifest data/derived/balanced_stability_conditioning_v2_isnt.csv \
  --conditioning-column diagnosis_condition \
  --output-dir outputs/training/balanced_residual_diffusion_v6_isnt \
  --image-size 512 --residual-size 256 --base-channels 32 \
  --max-steps 3000 --anatomy-loss-weight 0.75 \
  --anatomy-sector-emphasis 3.0 --device auto
```

Generated images should not be replaced until the new checkpoint passes the
locked persistence, healthy-stability, anatomy-change, and clinician-review
gates.

## Evidence

- Chan et al., *Translational Vision Science & Technology* (2013),
  https://doi.org/10.1167/tvst.2.5.2
- Poon et al., *American Journal of Ophthalmology* (2017),
  https://doi.org/10.1016/j.ajo.2017.09.018
- Miki et al., *Ophthalmology* (2015),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4808425/
- O'Leary et al., *Journal of Ophthalmology* (2018),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6023854/
