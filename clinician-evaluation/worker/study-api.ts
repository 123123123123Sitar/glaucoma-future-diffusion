import { caseTruthById, STUDY_VERSION } from "../lib/answer-key";

const progression = new Set(["none", "possible", "definite"]);
const authenticity = new Set(["A", "B", "unsure"]);
const diagnosis = new Set(["normal", "glaucoma", "uncertain"]);

function response(body: object, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Allow-Methods": "POST,OPTIONS",
    },
  });
}

function csvCell(value: unknown) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function csvResponse(rows: Record<string, unknown>[]) {
  const columns = [
    "session_id", "study_version", "reviewer_code", "specialty",
    "years_experience", "started_at", "completed_at", "feedback", "case_id",
    "case_order", "authenticity_choice", "reference_side", "reference_type",
    "is_correct", "truth_diagnosis", "diagnosis_baseline", "diagnosis_a",
    "diagnosis_b", "progression_a", "progression_b", "features_json",
    "confidence", "notes", "elapsed_seconds", "submitted_at",
  ];
  const csv = [
    columns.join(","),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(",")),
  ].join("\n");
  return new Response(csv, {
    status: 200,
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="retinaprogress-responses.csv"',
      "Cache-Control": "no-store",
    },
  });
}

async function ensureSchema(db: D1Database) {
  await db.batch([
    db.prepare(
      `CREATE TABLE IF NOT EXISTS study_sessions (
        id TEXT PRIMARY KEY,
        study_version TEXT NOT NULL,
        reviewer_code TEXT,
        specialty TEXT NOT NULL,
        years_experience INTEGER,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        feedback TEXT
      )`,
    ),
    db.prepare(
      `CREATE TABLE IF NOT EXISTS study_responses_v3 (
        session_id TEXT NOT NULL,
        case_id TEXT NOT NULL,
        case_order INTEGER NOT NULL,
        authenticity_choice TEXT NOT NULL,
        reference_side TEXT NOT NULL,
        reference_type TEXT NOT NULL,
        is_correct INTEGER,
        truth_diagnosis TEXT NOT NULL,
        diagnosis_baseline TEXT NOT NULL,
        diagnosis_a TEXT NOT NULL,
        diagnosis_b TEXT NOT NULL,
        progression_a TEXT NOT NULL,
        progression_b TEXT NOT NULL,
        features_json TEXT NOT NULL,
        confidence INTEGER NOT NULL,
        notes TEXT NOT NULL,
        elapsed_seconds INTEGER NOT NULL,
        submitted_at TEXT NOT NULL,
        PRIMARY KEY (session_id, case_id)
      )`,
    ),
  ]);
}

export async function handleStudyApi(request: Request, db: D1Database) {
  if (request.method === "OPTIONS") return response({}, 204);
  try {
    await ensureSchema(db);
    if (request.method === "GET") {
      const results = await db
        .prepare(
          `SELECT
             s.id AS session_id, s.study_version, s.reviewer_code, s.specialty,
             s.years_experience, s.started_at, s.completed_at, s.feedback,
             r.case_id, r.case_order, r.authenticity_choice, r.reference_side,
             r.reference_type, r.is_correct, r.truth_diagnosis,
             r.diagnosis_baseline, r.diagnosis_a, r.diagnosis_b,
             r.progression_a, r.progression_b, r.features_json, r.confidence,
             r.notes, r.elapsed_seconds, r.submitted_at
           FROM study_sessions s
           LEFT JOIN study_responses_v3 r ON r.session_id = s.id
           ORDER BY s.started_at, r.case_order`,
        )
        .all<Record<string, unknown>>();
      return csvResponse(results.results ?? []);
    }
    if (request.method !== "POST") return response({ error: "Method not allowed" }, 405);
    const body = await request.json<Record<string, any>>();
    if (body.type === "complete") {
      if (typeof body.sessionId !== "string") {
        return response({ error: "Invalid session" }, 400);
      }
      await db
        .prepare("UPDATE study_sessions SET completed_at = ?, feedback = ? WHERE id = ?")
        .bind(new Date().toISOString(), String(body.feedback ?? "").slice(0, 4000), body.sessionId)
        .run();
      return response({ saved: true });
    }

    const item = body.response;
    const profile = body.profile;
    const truth = caseTruthById[item?.caseId];
    if (
      typeof body.sessionId !== "string" ||
      body.sessionId.length > 100 ||
      !truth ||
      !authenticity.has(item?.authenticity) ||
      !diagnosis.has(item?.diagnosisBaseline) ||
      !diagnosis.has(item?.diagnosisA) ||
      !diagnosis.has(item?.diagnosisB) ||
      !progression.has(item?.progressionA) ||
      !progression.has(item?.progressionB) ||
      !Number.isInteger(item?.confidence) ||
      item.confidence < 0 ||
      item.confidence > 100
    ) {
      return response({ error: "Invalid response" }, 400);
    }

    const now = new Date().toISOString();
    await db.batch([
      db.prepare(
        `INSERT INTO study_sessions
          (id, study_version, reviewer_code, specialty, years_experience, started_at)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
          reviewer_code = excluded.reviewer_code,
          specialty = excluded.specialty,
          years_experience = excluded.years_experience`,
      ).bind(
        body.sessionId,
        STUDY_VERSION,
        String(profile?.reviewerCode ?? "").slice(0, 80) || null,
        String(profile?.specialty ?? "Not specified").slice(0, 80),
        Number.isFinite(Number(profile?.yearsExperience))
          ? Math.max(0, Math.min(80, Number(profile.yearsExperience)))
          : null,
        now,
      ),
      db.prepare(
        `INSERT INTO study_responses_v3
          (session_id, case_id, case_order, authenticity_choice, reference_side,
           reference_type, is_correct, truth_diagnosis, diagnosis_baseline,
           diagnosis_a, diagnosis_b, progression_a, progression_b, features_json, confidence,
           notes, elapsed_seconds, submitted_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(session_id, case_id) DO UPDATE SET
          authenticity_choice = excluded.authenticity_choice,
          is_correct = excluded.is_correct,
          diagnosis_baseline = excluded.diagnosis_baseline,
          diagnosis_a = excluded.diagnosis_a,
          diagnosis_b = excluded.diagnosis_b,
          progression_a = excluded.progression_a,
          progression_b = excluded.progression_b,
          features_json = excluded.features_json,
          confidence = excluded.confidence,
          notes = excluded.notes,
          elapsed_seconds = excluded.elapsed_seconds,
          submitted_at = excluded.submitted_at`,
      ).bind(
        body.sessionId,
        item.caseId,
        Math.max(1, Number(item.caseOrder) || 1),
        item.authenticity,
        truth.referenceSide,
        truth.referenceType,
        item.authenticity === "unsure" ? null : Number(item.authenticity === truth.referenceSide),
        truth.diagnosis,
        item.diagnosisBaseline,
        item.diagnosisA,
        item.diagnosisB,
        item.progressionA,
        item.progressionB,
        JSON.stringify(Array.isArray(item.features) ? item.features.slice(0, 12) : []),
        item.confidence,
        String(item.notes ?? "").slice(0, 4000),
        Math.max(0, Math.min(7200, Number(item.elapsedSeconds) || 0)),
        now,
      ),
    ]);
    return response({ saved: true });
  } catch (error) {
    console.error("study_api_error", error);
    return response({ error: "The study record could not be saved." }, 500);
  }
}
