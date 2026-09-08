import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("opens directly into the blinded case workflow", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.doesNotMatch(page, /screen.*intro|Begin blinded review|intro-shell/);
  assert.match(page, /CASE \{String\(index \+ 1\)/);
  assert.match(page, /candidateA/);
  assert.match(page, /candidateB/);
  assert.match(page, /Saved to the study database/);
  assert.match(page, /20 image comparisons/);
  assert.match(page, /What is this test\?/);
  assert.match(page, /Research use only/);
  assert.match(page, /diagnosisBaseline/);
  assert.match(page, /Which later image is the comparison image/);
});

test("comparison aids are symmetric and preserve the original study images", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /Side by side/);
  assert.match(page, /Blink/);
  assert.match(page, /Highlight changes/);
  assert.match(page, /Baseline ↔ \{label\}/);
  assert.match(page, /candidate=\{current\.candidateA\}/);
  assert.match(page, /candidate=\{current\.candidateB\}/);
  assert.match(page, /confirm[\s\S]*answer using the original photographs/);
});

test("protected results export is available to the study owner", async () => {
  const [admin, exportRoute, studyRoute] = await Promise.all([
    readFile(new URL("app/admin/page.tsx", root), "utf8"),
    readFile(new URL("app/api/export/route.ts", root), "utf8"),
    readFile(new URL("app/api/study/route.ts", root), "utf8"),
  ]);
  assert.match(admin, /Download clinician responses/);
  assert.match(admin, /Download results CSV/);
  assert.match(admin, /Download earlier database backup/);
  assert.match(exportRoute, /ADMIN_EXPORT_TOKEN/);
  assert.match(exportRoute, /x-admin-key/);
  assert.match(exportRoute, /text\/csv/);
  assert.match(exportRoute, /listAll\("responses\/"\)/);
  assert.match(studyRoute, /put\(`responses\//);
  assert.match(studyRoute, /access: "private"/);
});

test("failed scientific gate remains visible and disables outcome claims", async () => {
  const raw = await readFile(new URL("public/model-status.json", root), "utf8");
  const status = JSON.parse(raw);
  assert.equal(status.progressionOutputReleased, false);
  assert.equal(status.diffusionOutputReleased, true);
  assert.equal(status.status, "balanced_formative_pilot_only");
});

test("public manifest has opaque candidates and no answer key", async () => {
  const raw = await readFile(new URL("public/cases/manifest.json", root), "utf8");
  const manifest = JSON.parse(raw);
  assert.equal(manifest.schemaVersion, "3.0");
  assert.equal(manifest.lockedTestUsed, false);
  assert.equal(manifest.cases.length, 20);
  assert.doesNotMatch(raw, /referenceSide|diagnosis|generatedVs|quality|checkpoint/i);
  for (const item of manifest.cases) {
    assert.match(item.candidateA, /candidate-a\.webp$/);
    assert.match(item.candidateB, /candidate-b\.webp$/);
  }
});

test("server route owns scoring and private central persistence", async () => {
  const [route, worker, key, hosting] = await Promise.all([
    readFile(new URL("app/api/study/route.ts", root), "utf8"),
    readFile(new URL("worker/study-api.ts", root), "utf8"),
    readFile(new URL("lib/answer-key.ts", root), "utf8"),
    readFile(new URL(".openai/hosting.json", root), "utf8"),
  ]);
  assert.match(route, /OAI-Sites-Authorization/);
  assert.match(route, /caseTruthById/);
  assert.match(route, /BLOB|@vercel\/blob|put\(/);
  assert.match(worker, /study_responses_v3/);
  assert.match(worker, /caseTruthById/);
  assert.match(worker, /ON CONFLICT/);
  assert.match(key, /caseTruthById/);
  assert.equal(JSON.parse(hosting).d1, "DB");
});
