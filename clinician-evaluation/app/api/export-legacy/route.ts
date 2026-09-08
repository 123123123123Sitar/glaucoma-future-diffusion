import { NextResponse } from "next/server";

export const runtime = "edge";

export async function GET(request: Request) {
  const suppliedKey = request.headers.get("x-admin-key");
  const expectedKey = process.env.ADMIN_EXPORT_TOKEN;
  if (!expectedKey || !suppliedKey || suppliedKey !== expectedKey) {
    return NextResponse.json({ error: "Incorrect results passcode." }, { status: 401 });
  }

  const upstream = process.env.SITES_API_URL;
  const bypassToken = process.env.SITES_BYPASS_TOKEN;
  if (!upstream || !bypassToken) {
    return NextResponse.json({ error: "The earlier-results archive is not configured." }, { status: 503 });
  }

  try {
    const response = await fetch(`${upstream.replace(/\/$/, "")}/api/study?format=csv`, {
      method: "GET",
      headers: { "OAI-Sites-Authorization": `Bearer ${bypassToken}` },
      cache: "no-store",
    });
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("Content-Type") ?? "text/csv; charset=utf-8",
        "Content-Disposition": 'attachment; filename="retinaprogress-earlier-responses.csv"',
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json({ error: "The earlier results could not be downloaded." }, { status: 502 });
  }
}
