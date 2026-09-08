import pytest

from glaucoma_forecast.training.preference_feedback import (
    preference_training_ready,
    validate_preference_record,
)


def test_preference_validation_and_readiness() -> None:
    record = {
        "eye_id": "eye",
        "reviewer_code": "reviewer",
        "ranking": ["a", "b", "c", "d"],
        "ratings": {
            "visible_enlargement": 4,
            "temporal_smoothness": 4,
            "identity_preservation": 5,
            "sharpness": 4,
            "artifact_absence": 5,
        },
    }
    pair = validate_preference_record(record)
    assert pair.preferred_candidate == "a"
    assert not preference_training_ready([record], minimum_pairs=4)
    assert preference_training_ready([record], minimum_pairs=3)


def test_preference_validation_rejects_duplicate_ranks() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_preference_record(
            {
                "eye_id": "eye",
                "reviewer_code": "reviewer",
                "ranking": ["a", "a"],
                "ratings": {},
            }
        )
