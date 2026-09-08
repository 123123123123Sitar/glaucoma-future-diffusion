from glaucoma_forecast.evaluation.anatomy_release_gate import evaluate_anatomy_release


def _passing_report():
    return {
        "segmenter_disc_dice": 0.92,
        "segmenter_cup_dice": 0.78,
        "repeat_annotation_error": 0.03,
        "observed_vcdr_change": 0.12,
        "generated_vcdr_change_5y": 0.11,
        "monotonic_within_error": True,
        "outer_disc_preserved": True,
        "vessel_topology_preserved": True,
        "photometric_artifact_rejected": True,
        "stable_controls_pass": True,
        "human_accepted": True,
    }


def test_anatomy_release_requires_structural_and_human_checks() -> None:
    report = _passing_report()
    assert evaluate_anatomy_release(report)["passed"]
    report["generated_vcdr_change_5y"] = 0.0
    assert not evaluate_anatomy_release(report)["passed"]


def test_anatomy_release_rejects_brightness_only_candidate() -> None:
    report = _passing_report()
    report["photometric_artifact_rejected"] = False
    assert not evaluate_anatomy_release(report)["passed"]
