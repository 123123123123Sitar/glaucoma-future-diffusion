"""Release criteria for visible, target-grounded optic-disc progression."""

from __future__ import annotations

from typing import Any


DEFAULT_THRESHOLDS = {
    "disc_dice": 0.90,
    "cup_dice": 0.75,
    "target_fraction_min": 0.75,
    "target_fraction_max": 1.25,
    "signal_to_annotation_error": 2.0,
}


def evaluate_anatomy_release(
    report: dict[str, Any], thresholds: dict[str, float] | None = None
) -> dict[str, Any]:
    """Require anatomy change rather than brightness, blur, or scale differences."""

    required = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    observed_change = float(report.get("observed_vcdr_change", 0.0))
    generated_change = float(report.get("generated_vcdr_change_5y", 0.0))
    annotation_error = max(float(report.get("repeat_annotation_error", 0.0)), 1e-6)
    target_fraction = generated_change / observed_change if observed_change > 0 else 0.0
    checks = {
        "segmenter_disc_dice": float(report.get("segmenter_disc_dice", 0.0))
        >= required["disc_dice"],
        "segmenter_cup_dice": float(report.get("segmenter_cup_dice", 0.0))
        >= required["cup_dice"],
        "observed_progressor_signal": observed_change
        >= required["signal_to_annotation_error"] * annotation_error,
        "generated_progression_direction": generated_change > 0,
        "target_relative_magnitude": required["target_fraction_min"]
        <= target_fraction
        <= required["target_fraction_max"],
        "monotonic_trajectory": bool(report.get("monotonic_within_error", False)),
        "outer_disc_preserved": bool(report.get("outer_disc_preserved", False)),
        "vessel_topology_preserved": bool(report.get("vessel_topology_preserved", False)),
        "photometric_artifact_rejected": bool(
            report.get("photometric_artifact_rejected", False)
        ),
        "stable_controls_pass": bool(report.get("stable_controls_pass", False)),
        "human_accepted": bool(report.get("human_accepted", False)),
    }
    return {
        "schema_version": "1.0",
        "passed": all(checks.values()),
        "checks": checks,
        "target_fraction": target_fraction,
        "thresholds": required,
        "interpretation": (
            "High-confidence research simulation gate; not certainty of clinical prognosis."
        ),
    }
