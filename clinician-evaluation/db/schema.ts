import { integer, primaryKey, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const studySessions = sqliteTable("study_sessions", {
  id: text("id").primaryKey(),
  studyVersion: text("study_version").notNull(),
  reviewerCode: text("reviewer_code"),
  specialty: text("specialty").notNull(),
  yearsExperience: integer("years_experience"),
  startedAt: text("started_at").notNull(),
  completedAt: text("completed_at"),
  feedback: text("feedback"),
});

export const studyResponses = sqliteTable(
  "study_responses_v3",
  {
    sessionId: text("session_id").notNull(),
    caseId: text("case_id").notNull(),
    caseOrder: integer("case_order").notNull(),
    authenticityChoice: text("authenticity_choice").notNull(),
    referenceSide: text("reference_side").notNull(),
    referenceType: text("reference_type").notNull(),
    isCorrect: integer("is_correct"),
    truthDiagnosis: text("truth_diagnosis").notNull(),
    diagnosisBaseline: text("diagnosis_baseline").notNull(),
    diagnosisA: text("diagnosis_a").notNull(),
    diagnosisB: text("diagnosis_b").notNull(),
    progressionA: text("progression_a").notNull(),
    progressionB: text("progression_b").notNull(),
    featuresJson: text("features_json").notNull(),
    confidence: integer("confidence").notNull(),
    notes: text("notes").notNull(),
    elapsedSeconds: integer("elapsed_seconds").notNull(),
    submittedAt: text("submitted_at").notNull(),
  },
  (table) => [primaryKey({ columns: [table.sessionId, table.caseId] })],
);
