"""Deterministic scientific release gates for the progression system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_GATES = {
    "auroc": 0.70,
    "auroc_ci_lower": 0.50,
    "sensitivity": 0.80,
    "specificity": 0.50,
}


def evaluate_progression_candidate(
    report: dict[str, Any],
    gates: dict[str, float] | None = None,
) -> dict[str, Any]:
    required = DEFAULT_GATES if gates is None else gates
    metrics = report.get("aggregate_oof") or report.get("metrics") or {}
    intervals = report.get("patient_bootstrap_95ci") or report.get(
        "bootstrap_95ci", {}
    )
    auroc_interval = intervals.get("auroc", [float("-inf"), float("inf")])
    checks = {
        "auroc": float(metrics.get("auroc", float("-inf")))
        >= required["auroc"],
        "auroc_ci_lower": float(auroc_interval[0])
        > required["auroc_ci_lower"],
        "sensitivity": float(metrics.get("sensitivity", float("-inf")))
        >= required["sensitivity"],
        "specificity": float(metrics.get("specificity", float("-inf")))
        >= required["specificity"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "auroc_95ci": auroc_interval,
    }


def build_release_report(
    candidates: dict[str, dict[str, Any]],
    primary_candidate_names: list[str],
) -> dict[str, Any]:
    evaluated = {
        name: evaluate_progression_candidate(report)
        for name, report in candidates.items()
    }
    primary_passed = any(
        evaluated.get(name, {}).get("passed", False)
        for name in primary_candidate_names
    )
    return {
        "schema_version": "1.0",
        "progression_output_released": primary_passed,
        "diffusion_output_released": False,
        "primary_candidates": primary_candidate_names,
        "candidates": evaluated,
        "clinician_tool_status": (
            "research_candidate" if primary_passed else "withheld_failed_gate"
        ),
        "warning": (
            "No patient-facing or clinical output may be enabled merely because "
            "a secondary population or post-hoc endpoint passes."
        ),
    }


def load_reports(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    return {
        name: json.loads(Path(path).read_text()) for name, path in paths.items()
    }

