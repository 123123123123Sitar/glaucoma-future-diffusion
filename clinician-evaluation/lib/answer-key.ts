export const STUDY_VERSION = "retina-progress-balanced-v3";

// This module is imported only by the server route. Never expose it through
// the public manifest or a client component.
export type CaseTruth = {
  referenceSide: "A" | "B";
  diagnosis: "normal" | "glaucoma";
  referenceType:
    | "observed_longitudinal_followup"
    | "pseudo_longitudinal_unchanged_stability_control";
};

export const caseTruthById: Record<string, CaseTruth> = {
  "B-C26F00D": { referenceSide: "A", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-8735222": { referenceSide: "B", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-823B897": { referenceSide: "A", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-A11D28A": { referenceSide: "B", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-6723623": { referenceSide: "A", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-A19025F": { referenceSide: "B", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-03ACE63": { referenceSide: "A", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-0BC0579": { referenceSide: "B", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-1B49AC4": { referenceSide: "A", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-F886B47": { referenceSide: "B", diagnosis: "glaucoma", referenceType: "observed_longitudinal_followup" },
  "B-C8B0D54": { referenceSide: "A", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
  "B-BD0E972": { referenceSide: "B", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
  "B-5E3015F": { referenceSide: "A", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
  "B-2E441DA": { referenceSide: "B", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
  "B-7DF1B95": { referenceSide: "A", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
  "B-2A0B403": { referenceSide: "B", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
  "B-F622BF7": { referenceSide: "A", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
  "B-38BD06F": { referenceSide: "B", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
  "B-A598053": { referenceSide: "A", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
  "B-40F15AB": { referenceSide: "B", diagnosis: "normal", referenceType: "pseudo_longitudinal_unchanged_stability_control" },
};
