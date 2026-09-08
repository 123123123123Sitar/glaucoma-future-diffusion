#!/usr/bin/env python3
"""Serve the local contour annotation tool and persist completed labels."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.longitudinal_annotations import (
    append_jsonl,
    export_annotation_masks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", default="outputs/annotations/grape_longitudinal_v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    records = {item["annotation_id"]: item for item in batch["records"]}
    html_path = Path(__file__).resolve().parents[1] / "tools" / "longitudinal_annotation" / "index.html"

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
            if path == "/batch.json":
                self._send(200, json.dumps(batch).encode(), "application/json")
                return
            if path.startswith("/assets/"):
                target = (batch_dir / path.removeprefix("/")).resolve()
                if batch_dir not in target.parents or not target.is_file():
                    self._send(404, b"not found", "text/plain")
                    return
                self._send(
                    200,
                    target.read_bytes(),
                    mimetypes.guess_type(target.name)[0] or "application/octet-stream",
                )
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/annotation":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                source = records[str(payload["annotation_id"])]
                paths = export_annotation_masks(payload, batch_dir / "masks")
                saved = {
                    **source,
                    **paths,
                    "progression_label": str(payload["progression_label"]),
                    "quality_acceptable": bool(payload["quality_acceptable"]),
                    "findings": "|".join(payload.get("findings", [])),
                    "annotator_code": str(payload["annotator_code"]),
                    "notes": str(payload.get("notes", "")),
                }
                append_jsonl(batch_dir / "annotations.jsonl", saved)
                current = pd.DataFrame(
                    [json.loads(line) for line in (batch_dir / "annotations.jsonl").read_text().splitlines()]
                )
                current.drop_duplicates("annotation_id", keep="last").to_csv(
                    batch_dir / "annotation_manifest.csv", index=False
                )
                self._send(200, json.dumps({"saved": saved["annotation_id"]}).encode(), "application/json")
            except Exception as error:  # pragma: no cover - browser-facing diagnostics
                self._send(400, json.dumps({"error": str(error)}).encode(), "application/json")

        def log_message(self, format: str, *values) -> None:
            print(format % values)

    print(f"Annotation tool: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
