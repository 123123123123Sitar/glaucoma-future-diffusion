import assert from 'node:assert/strict';
import test from 'node:test';
import {
  completeRating, finalRating, mergeFinalRating, parseFinalSubmission, restoreRatings,
} from '../lib/sigf-review.ts';

const rating = { index: 2, similarity: 4, observed: 'possible', change: 'similar', notes: 'Final visit' };

test('a complete final rating saves without answers for the earlier visits', () => {
  assert.deepEqual(parseFinalSubmission([rating]), rating);
  assert.equal(parseFinalSubmission([{ ...rating, index: 0 }]), null);
  assert.equal(parseFinalSubmission([{ ...rating, index: 1 }]), null);
  assert.equal(parseFinalSubmission([{ ...rating, similarity: '' }]), null);
  assert.equal(parseFinalSubmission([{ ...rating, observed: '' }]), null);
  assert.equal(parseFinalSubmission([rating, rating]), null);
  assert.equal(parseFinalSubmission([null]), null);
});

test('final-only records resume in the final visit rather than the first slot', () => {
  const restored = restoreRatings([rating]);
  assert.equal(completeRating(restored[0]), false);
  assert.equal(completeRating(restored[1]), false);
  assert.deepEqual(restored[2], rating);
  assert.equal(completeRating(finalRating(restored)), true);
});

test('saving the final visit preserves earlier answers and uses the real final interval', () => {
  const first = { ...rating, index: 0, years: 1.01, notes: 'Earlier answer' };
  const middle = { ...rating, index: 1, years: 3.02, notes: 'Middle answer' };
  const oldFinal = { ...rating, years: 4.944558521560575, similarity: 2 };
  const merged = mergeFinalRating([first, middle, oldFinal], rating, [1.01, 3.02, 4.944558521560575]);
  assert.deepEqual(merged.slice(0, 2), [first, middle]);
  assert.deepEqual(finalRating(merged), { ...rating, years: 4.944558521560575 });
  assert.equal(merged.length, 3);
  assert.equal(mergeFinalRating([], rating, [1, 2, 2.8008213552361396])[0].years, 2.8008213552361396);
});

test('export selection picks exactly the final rating from either record format', () => {
  assert.equal(finalRating([{ ...rating, index: 0 }]), undefined);
  assert.deepEqual(finalRating([rating]), rating);
  assert.deepEqual(finalRating([{ ...rating, index: 0 }, rating]), rating);
});
