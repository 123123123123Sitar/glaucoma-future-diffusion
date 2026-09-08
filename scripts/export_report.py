#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from glaucoma_forecast.evaluation.reports import write_markdown_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export final research report.")
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output", default="outputs/research_report.md")
    args = parser.parse_args()
    with open(args.audit_json, "r", encoding="utf-8") as handle:
        audit = json.load(handle)
    write_markdown_report({"dataset_summary": audit, "limitations": [
        "small number of converting eyes",
        "irregular intervals",
        "camera changes over time",
        "treatment effects",
        "possible label noise",
        "single-image prediction is limited",
        "not clinically validated",
    ]}, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
