#!/usr/bin/env python3
"""Build a common GRAPE manifest from the official metadata workbook and CFPs.

This parser intentionally uses only Python's standard library for `.xlsx`
inspection so the repository can bootstrap before optional Excel dependencies
are installed.
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import _bootstrap  # noqa: F401
from glaucoma_forecast.data.schema import MANIFEST_COLUMNS, normalize_manifest, validate_manifest

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for si in root.findall("a:si", NS):
        parts = [node.text or "" for node in si.iter(f"{{{NS['a']}}}t")]
        values.append("".join(parts))
    return values


def _cell_ref_to_col(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    value = cell.find("a:v", NS)
    if value is None:
        inline = cell.find("a:is", NS)
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{NS['a']}}}t"))
    text = value.text or ""
    if cell.attrib.get("t") == "s":
        return shared[int(text)]
    return text


def _sheet_rows(zf: ZipFile, sheet_name: str) -> list[list[str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    target = None
    for sheet in workbook.find("a:sheets", NS):
        if sheet.attrib["name"] == sheet_name:
            rid = sheet.attrib[f"{{{REL_NS}}}id"]
            target = relmap[rid]
            break
    if target is None:
        raise ValueError(f"Workbook missing sheet: {sheet_name}")
    sheet_path = "xl/" + target if not target.startswith("/") else target.lstrip("/")
    shared = _shared_strings(zf)
    root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[str]] = []
    for row in root.findall(".//a:sheetData/a:row", NS):
        values: list[str] = []
        for cell in row.findall("a:c", NS):
            col = _cell_ref_to_col(cell.attrib["r"])
            while len(values) <= col:
                values.append("")
            values[col] = _cell_value(cell, shared).strip()
        rows.append(values)
    return rows


def _get(row: list[str], index: int) -> str:
    return row[index] if index < len(row) else ""


def _to_float(value: str) -> float | None:
    if value in {"", "/"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def build_manifest(workbook: Path, image_dir: Path, output: Path) -> tuple[int, int]:
    with ZipFile(workbook) as zf:
        baseline_rows = _sheet_rows(zf, "Baseline")[2:]
        followup_rows = _sheet_rows(zf, "Follow-up")[2:]

    baseline: dict[tuple[str, str], dict[str, str]] = {}
    for row in baseline_rows:
        subject = _get(row, 0)
        laterality = _get(row, 1)
        if not subject or not laterality:
            continue
        baseline[(subject, laterality)] = {
            "age": _get(row, 2),
            "sex": _get(row, 3),
            "baseline_iop": _get(row, 4),
            "cct": _get(row, 5),
            "total_visits": _get(row, 6),
            "progression_plr2": _get(row, 7),
            "progression_plr3": _get(row, 8),
            "progression_md": _get(row, 9),
            "glaucoma_type": _get(row, 10),
            "rnfl_mean": _get(row, 11),
            "rnfl_superior": _get(row, 12),
            "rnfl_nasal": _get(row, 13),
            "rnfl_inferior": _get(row, 14),
            "rnfl_temporal": _get(row, 15),
        }

    rows: list[dict[str, object]] = []
    skipped_missing_cfp = 0
    skipped_missing_file = 0
    for row in followup_rows:
        subject = _get(row, 0)
        laterality = _get(row, 1)
        visit_number = _to_int(_get(row, 2))
        interval_years = _to_float(_get(row, 3))
        cfp = _get(row, 5)
        if not subject or not laterality or visit_number is None:
            continue
        if not cfp or cfp == "/":
            skipped_missing_cfp += 1
            continue
        image_path = image_dir / cfp
        if not image_path.exists():
            skipped_missing_file += 1
            continue
        base = baseline.get((subject, laterality), {})
        manifest_row: dict[str, object] = {
            "patient_id": subject,
            "eye_id": f"{subject}_{laterality}",
            "laterality": laterality,
            "visit_id": f"V{visit_number:03d}",
            "visit_index": visit_number - 1,
            "acquisition_date": None,
            "time_from_baseline_years": interval_years,
            "image_path": str(image_path),
            "glaucoma_label": 1,
            # The workbook gives eye-level progression status in baseline metadata,
            # not visit-specific future labels. Keep the common visit label null.
            "progression_label": None,
            "camera_device": _get(row, 6) or None,
            "image_width": None,
            "image_height": None,
            "quality_score": None,
            "iop": _to_float(_get(row, 4)),
            "rnfl_mean": _to_float(base.get("rnfl_mean", "")),
            "rnfl_superior": _to_float(base.get("rnfl_superior", "")),
            "rnfl_inferior": _to_float(base.get("rnfl_inferior", "")),
            "rnfl_nasal": _to_float(base.get("rnfl_nasal", "")),
            "rnfl_temporal": _to_float(base.get("rnfl_temporal", "")),
            "visual_field_md": None,
            "source_dataset": "GRAPE",
        }
        rows.append(manifest_row)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame = normalize_manifest(__import__("pandas").DataFrame(rows), source_dataset="GRAPE")
    validate_manifest(frame, require_images=True)
    frame.to_csv(output, index=False)

    extras_path = output.with_name("metadata_extra.csv")
    extra_fields = [
        "patient_id",
        "eye_id",
        "age",
        "sex",
        "cct",
        "total_visits",
        "progression_plr2",
        "progression_plr3",
        "progression_md",
        "glaucoma_type",
    ]
    with extras_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=extra_fields)
        writer.writeheader()
        for (subject, laterality), base in sorted(baseline.items(), key=lambda item: (int(item[0][0]), item[0][1])):
            writer.writerow(
                {
                    "patient_id": subject,
                    "eye_id": f"{subject}_{laterality}",
                    **{field: base.get(field, "") for field in extra_fields if field not in {"patient_id", "eye_id"}},
                }
            )

    print(f"Wrote {len(rows)} manifest rows to {output}")
    print(f"Wrote baseline extras to {extras_path}")
    print(f"Skipped follow-up rows with no CFP: {skipped_missing_cfp}")
    print(f"Skipped follow-up rows whose CFP file was absent: {skipped_missing_file}")
    return len(rows), skipped_missing_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GRAPE manifest from VF/clinical workbook and CFP images.")
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--image-dir", default="data/grape/images")
    parser.add_argument("--output", default="data/grape/manifest.csv")
    args = parser.parse_args()
    _, missing = build_manifest(Path(args.workbook), Path(args.image_dir), Path(args.output))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
