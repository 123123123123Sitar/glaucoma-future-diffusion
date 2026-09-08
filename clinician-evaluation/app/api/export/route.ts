import { get, list } from "@vercel/blob";
import { NextResponse } from "next/server";

export const runtime = "edge";

const columns = [
  "sessionId", "studyVersion", "reviewerCode", "specialty", "yearsExperience",
  "completedAt", "feedback", "caseId", "caseOrder", "authenticityChoice",
  "referenceSide", "referenceType", "isCorrect", "truthDiagnosis",
  "diagnosisBaseline", "diagnosisA", "diagnosisB", "progressionA",
  "progressionB", "features", "confidence", "notes", "elapsedSeconds",
  "submittedAt",
];

function csvCell(value: unknown) {
  if (value === null || value === undefined) return "";
  const text = Array.isArray(value) ? value.join("; ") : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function listAll(prefix: string) {
  const blobs: Awaited<ReturnType<typeof list>>["blobs"] = [];
  let cursor: string | undefined;
  do {
    const page = await list({ prefix, cursor, limit: 1000 });
    blobs.push(...page.blobs);
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs;
}

async function readJson(pathname: string) {
  const result = await get(pathname, { access: "private", useCache: false });
  if (!result || result.statusCode !== 200) return null;
  return JSON.parse(await new Response(result.stream).text()) as Record<string, unknown>;
}

export async function GET(request: Request) {
  const suppliedKey = request.headers.get("x-admin-key");
  const expectedKey = process.env.ADMIN_EXPORT_TOKEN;
  if (!expectedKey || !suppliedKey || suppliedKey !== expectedKey) {
    return NextResponse.json({ error: "Incorrect results passcode." }, { status: 401 });
  }

  try {
    const [responseBlobs, completionBlobs] = await Promise.all([
      listAll("responses/"),
      listAll("completions/"),
    ]);
    const [records, completions] = await Promise.all([
      Promise.all(responseBlobs.map((blob) => readJson(blob.pathname))),
      Promise.all(completionBlobs.map((blob) => readJson(blob.pathname))),
    ]);
    const completionBySession = new Map(
      completions
        .filter((item): item is Record<string, unknown> => Boolean(item))
        .map((item) => [String(item.sessionId), item]),
    );
    const rows = records
      .filter((item): item is Record<string, unknown> => Boolean(item))
      .sort((a, b) =>
        String(a.submittedAt).localeCompare(String(b.submittedAt)) ||
        Number(a.caseOrder) - Number(b.caseOrder),
      )
      .map((item): Record<string, unknown> => ({
        ...item,
        completedAt: completionBySession.get(String(item.sessionId))?.completedAt ?? "",
        feedback: completionBySession.get(String(item.sessionId))?.feedback ?? "",
      }));
    const csv = [
      columns.join(","),
      ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(",")),
    ].join("\n");
    return new NextResponse(csv, {
      status: 200,
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Disposition": 'attachment; filename="retinaprogress-responses.csv"',
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    console.error("study_export_error", error);
    return NextResponse.json({ error: "The study results could not be downloaded." }, { status: 500 });
  }
}
