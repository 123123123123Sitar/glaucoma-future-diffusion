"""Research report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DISCLAIMER = (
    "Research prototype only. Not validated for patient care, diagnosis, triage, "
    "or treatment decisions."
)


def write_json_report(payload: dict[str, Any], output_path: str | Path) -> None:
    data = {"disclaimer": DISCLAIMER, **payload}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_markdown_report(payload: dict[str, Any], output_path: str | Path) -> None:
    lines = ["# Glaucoma forecasting research report", "", f"> {DISCLAIMER}", ""]
    for key, value in payload.items():
        lines.extend([f"## {key}", "", "```json", json.dumps(value, indent=2, default=str), "```", ""])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
