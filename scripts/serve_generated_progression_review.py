#!/usr/bin/env python3
"""Serve one generated optic-disc trajectory for a simple visual judgment."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.longitudinal_annotations import append_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--feedback", default="outputs/preferences/generated_progression_reviews.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    report = json.loads(report_path.read_text())
    project_root = Path.cwd().resolve()
    html_path = Path(__file__).resolve().parents[1] / "tools" / "generated_progression_review" / "index.html"
    payload = {
        "baseline": f"/file/{quote(str(Path(report['input_artifacts']['optic_disc']).resolve()))}",
        "images": [f"/file/{quote(str(Path(path).resolve()))}" for path in report["representative_disc_images"]],
        "years": report["requested_years"],
        "checkpoint": report["detail_checkpoint"],
        "checkpoint_step": report["detail_checkpoint_step"],
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
                self._send(200, json.dumps(payload).encode(), "application/json")
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
            if urlparse(self.path).path != "/api/review":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0"))
            record = json.loads(self.rfile.read(length))
            if record.get("judgment") not in {"different", "same", "uncertain"}:
                self._send(400, b'{"error":"invalid judgment"}', "application/json")
                return
            append_jsonl(args.feedback, {**record, "source_report": str(report_path)})
            self._send(200, b'{"saved":true}', "application/json")

    print(f"Generated review: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
