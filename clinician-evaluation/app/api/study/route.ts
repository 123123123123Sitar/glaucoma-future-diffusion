import { put } from "@vercel/blob";
import { NextResponse } from "next/server";
import { caseTruthById, STUDY_VERSION } from "../../../lib/answer-key";

export const runtime = "edge";

const progression = new Set(["none", "possible", "definite"]);
const authenticity = new Set(["A", "B", "unsure"]);
const diagnosis = new Set(["normal", "glaucoma", "uncertain"]);
const safeId = /^[A-Za-z0-9-]{1,100}$/;

function cors(response: NextResponse) {
  response.headers.set("Access-Control-Allow-Origin", "*");
  response.headers.set("Access-Control-Allow-Headers", "Content-Type");
  response.headers.set("Access-Control-Allow-Methods", "POST,OPTIONS");
  return response;
}

export async function OPTIONS() {
  return cors(new NextResponse(null, { status: 204 }));
}

async function mirrorToLegacyDatabase(body: unknown) {
  const upstream = process.env.SITES_API_URL;
  if (!upstream) return;
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (process.env.SITES_BYPASS_TOKEN) {
      headers["OAI-Sites-Authorization"] = `Bearer ${process.env.SITES_BYPASS_TOKEN}`;
    }
    await fetch(`${upstream.replace(/\/$/, "")}/api/study`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
  } catch {
    // Vercel Blob is the primary store. A legacy mirror failure must not lose a response.
  }
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Record<string, any>;
    if (!safeId.test(String(body.sessionId ?? ""))) {
      return cors(NextResponse.json({ error: "Invalid study session." }, { status: 400 }));
    }

    const now = new Date().toISOString();
    if (body.type === "complete") {
      const completion = {
        sessionId: body.sessionId,
        completedAt: now,
        feedback: String(body.feedback ?? "").slice(0, 4000),
      };
      await put(`completions/${body.sessionId}.json`, JSON.stringify(completion), {
        access: "private",
        addRandomSuffix: false,
        allowOverwrite: true,
        contentType: "application/json",
      });
      await mirrorToLegacyDatabase(body);
      return cors(NextResponse.json({ saved: true }));
    }

    const item = body.response;
    const profile = body.profile ?? {};
    const truth = caseTruthById[item?.caseId];
    if (
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
      return cors(NextResponse.json({ error: "Please answer every required question." }, { status: 400 }));
    }

    const record = {
      studyVersion: STUDY_VERSION,
      sessionId: body.sessionId,
      reviewerCode: String(profile.reviewerCode ?? "").slice(0, 80),
      specialty: String(profile.specialty ?? "Not specified").slice(0, 80),
      yearsExperience: Number.isFinite(Number(profile.yearsExperience))
        ? Math.max(0, Math.min(80, Number(profile.yearsExperience)))
        : null,
      caseId: item.caseId,
      caseOrder: Math.max(1, Number(item.caseOrder) || 1),
      authenticityChoice: item.authenticity,
      referenceSide: truth.referenceSide,
      referenceType: truth.referenceType,
      isCorrect: item.authenticity === "unsure" ? null : item.authenticity === truth.referenceSide,
      truthDiagnosis: truth.diagnosis,
      diagnosisBaseline: item.diagnosisBaseline,
      diagnosisA: item.diagnosisA,
      diagnosisB: item.diagnosisB,
      progressionA: item.progressionA,
      progressionB: item.progressionB,
      features: Array.isArray(item.features) ? item.features.slice(0, 12) : [],
      confidence: item.confidence,
      notes: String(item.notes ?? "").slice(0, 4000),
      elapsedSeconds: Math.max(0, Math.min(7200, Number(item.elapsedSeconds) || 0)),
      submittedAt: now,
    };
    await put(`responses/${body.sessionId}/${item.caseId}.json`, JSON.stringify(record), {
      access: "private",
      addRandomSuffix: false,
      allowOverwrite: true,
      contentType: "application/json",
    });
    await mirrorToLegacyDatabase(body);
    return cors(NextResponse.json({ saved: true }));
  } catch (error) {
    console.error("study_save_error", error);
    return cors(NextResponse.json({ error: "The study answer could not be saved." }, { status: 500 }));
  }
}
