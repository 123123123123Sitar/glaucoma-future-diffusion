#!/usr/bin/env python3
"""Serve same-eye progression sequences and persist reviewer preferences."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.longitudinal_annotations import append_jsonl
from glaucoma_forecast.training.preference_feedback import validate_preference_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--feedback", default="outputs/preferences/progression_preferences.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    report_path = Path(args.report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    html_path = Path(__file__).resolve().parents[1] / "tools" / "progression_preferences" / "index.html"
    eye_id = report_path.parent.name

    def browser_payload():
        baseline = report["input_artifacts"]["full_field"]
        return {
            "eye_id": eye_id,
            "years": report["requested_years"],
            "baseline": f"/file/{quote(str(Path(baseline).resolve()))}",
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "images": [f"/file/{quote(str(Path(path).resolve()))}" for path in candidate["images"]],
                }
                for candidate in report.get("candidate_sequences", [])
            ],
            "disclaimer": report["disclaimer"],
        }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/review.json":
                self._send(200, json.dumps(browser_payload()).encode(), "application/json")
                return
            if path.startswith("/file/"):
                target = Path(path.removeprefix("/file/")).resolve()
                if project_root not in target.parents or not target.is_file():
                    self._send(404, b"not found", "text/plain")
                    return
                self._send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/preference":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                record = json.loads(self.rfile.read(length))
                validate_preference_record(record)
                record["source_report"] = str(report_path)
                append_jsonl(args.feedback, record)
                self._send(200, b'{"saved":true}', "application/json")
            except Exception as error:  # pragma: no cover
                self._send(400, json.dumps({"error": str(error)}).encode(), "application/json")

    if len(report.get("candidate_sequences", [])) < 2:
        raise ValueError("Report must contain candidates; regenerate with --save-candidates")
    print(f"Preference review: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
