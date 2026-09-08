"""Human preference records for progression candidate selection and tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RATING_FIELDS = (
    "visible_enlargement",
    "temporal_smoothness",
    "identity_preservation",
    "sharpness",
    "artifact_absence",
)


@dataclass(frozen=True)
class PreferencePair:
    eye_id: str
    preferred_candidate: str
    rejected_candidate: str
    reviewer_code: str
    ratings: dict[str, int]


def validate_preference_record(record: dict[str, Any]) -> PreferencePair:
    ranking = [str(value) for value in record.get("ranking", [])]
    if len(ranking) < 2 or len(set(ranking)) != len(ranking):
        raise ValueError("ranking must contain at least two unique candidates")
    reviewer_code = str(record.get("reviewer_code", "")).strip()
    if not reviewer_code:
        raise ValueError("reviewer_code is required")
    ratings = record.get("ratings", {})
    missing = [field for field in RATING_FIELDS if field not in ratings]
    if missing:
        raise ValueError(f"Missing preference ratings: {missing}")
    normalized = {field: int(ratings[field]) for field in RATING_FIELDS}
    if any(value < 1 or value > 5 for value in normalized.values()):
        raise ValueError("Preference ratings must be integers from 1 to 5")
    return PreferencePair(
        eye_id=str(record["eye_id"]),
        preferred_candidate=ranking[0],
        rejected_candidate=ranking[-1],
        reviewer_code=reviewer_code,
        ratings=normalized,
    )


def preference_training_ready(records: list[dict[str, Any]], minimum_pairs: int = 100) -> bool:
    return sum(max(0, len(record.get("ranking", [])) - 1) for record in records) >= minimum_pairs
