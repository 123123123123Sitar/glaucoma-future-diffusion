export const FINAL_FOLLOWUP_INDEX = 2;

export type Rating = {
  index: number;
  similarity: number | 'unable' | '';
  observed: string;
  change: string;
  notes: string;
};

export type StoredRating = Rating & { years: number };

export const blankRatings = (): Rating[] => [0, 1, 2].map(index => ({
  index, similarity: '', observed: '', change: '', notes: '',
}));

// Both older three-visit records and new final-only records resume by visit ID.
export function restoreRatings(values?: Rating[] | null): Rating[] {
  return blankRatings().map(blank => values?.find(r => r.index === blank.index) ?? blank);
}

export function finalRating<T extends { index: number }>(values?: T[] | null): T | undefined {
  return values?.find(r => r.index === FINAL_FOLLOWUP_INDEX);
}

export function completeRating(r?: Rating): boolean {
  return !!r && (r.similarity === 'unable' || Number.isInteger(r.similarity) && Number(r.similarity) >= 1 && Number(r.similarity) <= 5)
    && ['none', 'possible', 'definite', 'unable'].includes(r.observed)
    && ['too-little', 'similar', 'too-much', 'unable'].includes(r.change);
}

export function parseFinalSubmission(values: unknown): Rating | null {
  if (!Array.isArray(values) || values.length !== 1) return null;
  const r = values[0];
  if (!r || r.index !== FINAL_FOLLOWUP_INDEX || !completeRating(r)) return null;
  return {
    index: FINAL_FOLLOWUP_INDEX,
    similarity: r.similarity,
    observed: r.observed,
    change: r.change,
    notes: String(r.notes || '').slice(0, 2000),
  };
}

export function mergeFinalRating(existing: StoredRating[], submitted: Rating, years: number[]): StoredRating[] {
  const elapsed = years[FINAL_FOLLOWUP_INDEX];
  if (!Number.isFinite(elapsed)) throw new Error('Final follow-up interval unavailable.');
  return [
    ...existing.filter(r => r.index === 0 || r.index === 1),
    { ...submitted, index: FINAL_FOLLOWUP_INDEX, years: elapsed },
  ].sort((a, b) => a.index - b.index);
}
