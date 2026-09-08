"use client";

import { useState } from "react";

export default function ResultsPage() {
  const [passcode, setPasscode] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  async function downloadResults(
    endpoint = "/api/export",
    filename = "retinaprogress-responses.csv",
  ) {
    setBusy(true);
    setStatus("Opening the study database…");
    try {
      const response = await fetch(endpoint, {
        headers: { "x-admin-key": passcode },
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: "Download failed." }));
        throw new Error(body.error ?? "Download failed.");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
      setStatus("Results downloaded.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Download failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="admin-shell">
      <section className="admin-card">
        <p className="eyebrow">STUDY ADMIN</p>
        <h1>Download clinician responses</h1>
        <p>
          Each answer is saved to the central study database as soon as the
          clinician selects “Save &amp; next case.” Downloading creates a CSV
          file that opens in Excel or Google Sheets.
        </p>
        <label>
          Results passcode
          <input
            type="password"
            value={passcode}
            onChange={(event) => setPasscode(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && downloadResults()}
            autoComplete="current-password"
          />
        </label>
        <button className="primary" onClick={() => downloadResults()} disabled={!passcode || busy}>
          {busy ? "Preparing file…" : "Download results CSV"}
        </button>
        <button
          className="secondary-download"
          onClick={() =>
            downloadResults(
              "/api/export-legacy",
              "retinaprogress-earlier-responses.csv",
            )
          }
          disabled={!passcode || busy}
        >
          Download earlier database backup
        </button>
        <small className={status.includes("Incorrect") || status.includes("failed") ? "error" : ""}>{status}</small>
        <a href="/">Return to clinician study</a>
      </section>
    </main>
  );
}
