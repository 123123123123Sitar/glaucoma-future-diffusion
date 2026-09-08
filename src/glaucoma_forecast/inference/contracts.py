"""Stable report contract for the single-image glaucoma-risk pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DISCLAIMER = (
    "Research prototype only. Not validated for diagnosis, screening, triage, "
    "treatment, or patient-care decisions."
)


@dataclass
class RiskForecastReport:
    schema_version: str = "2.0"
    disclaimer: str = DISCLAIMER
    model_status: str = "not_loaded"
    model_version: str | None = None
    input_image: str = ""
    quality_status: str = "unknown"
    quality_warnings: list[str] = field(default_factory=list)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    ood_score: float | None = None
    possible_current_glaucoma_probability: float | None = None
    forecast_applicable: bool = False
    risk_2y: float | None = None
    risk_5y: float | None = None
    risk_10y: float | None = None
    annual_hazard_years_1_to_10: list[float] | None = None
    confidence_intervals: dict[str, list[float]] = field(default_factory=dict)
    vcdr_estimate: float | None = None
    preprocessing: dict[str, Any] = field(default_factory=dict)
    disc_cup_masks: dict[str, str] = field(default_factory=dict)
    annual_scenario_images_512px: list[str] = field(default_factory=list)
    scenario_uncertainty_maps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    training_data_scope: str = (
        "Requires longitudinal non-glaucoma/suspect cohorts with adjudicated "
        "incident-glaucoma outcomes; legacy GRAPE-only weights are unsupported."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineExam:
    """Versioned input contract for known-glaucoma progression research."""

    fundus_image: str
    known_glaucoma: bool
    age: float | None = None
    iop: float | None = None
    cct: float | None = None
    visual_field: list[float | None] | None = None
    rnfl_mean: float | None = None
    rnfl_superior: float | None = None
    rnfl_nasal: float | None = None
    rnfl_inferior: float | None = None
    rnfl_temporal: float | None = None
    sex: str | None = None
    glaucoma_type: str | None = None
    laterality: str | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.fundus_image:
            errors.append("fundus_image_required")
        if self.visual_field is not None and len(self.visual_field) != 61:
            errors.append("visual_field_must_have_61_values")
        if self.laterality and self.laterality.upper() not in {"OD", "OS"}:
            errors.append("laterality_must_be_OD_or_OS")
        return errors

    def to_feature_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "age": self.age,
            "baseline_iop": self.iop,
            "cct": self.cct,
            "rnfl_mean": self.rnfl_mean,
            "rnfl_superior": self.rnfl_superior,
            "rnfl_nasal": self.rnfl_nasal,
            "rnfl_inferior": self.rnfl_inferior,
            "rnfl_temporal": self.rnfl_temporal,
            "sex": self.sex,
            "glaucoma_type": self.glaucoma_type,
            "laterality": self.laterality,
        }
        visual_field = self.visual_field or [None] * 61
        row.update(
            {f"vf_{index:02d}": value for index, value in enumerate(visual_field)}
        )
        return row


@dataclass
class ProgressionForecastReport:
    """Stable output contract for the three-year research system."""

    schema_version: str = "3.0"
    disclaimer: str = DISCLAIMER
    model_status: str = "not_loaded"
    model_version: str | None = None
    endpoint: str = "approximately_three_year_plr2_progression"
    forecast_horizon_years: float = 3.0
    gradability: str = "unknown"
    ood_status: str = "unknown"
    applicability: str = "not_evaluated"
    abstention_reasons: list[str] = field(default_factory=list)
    current_referral_probability: float | None = None
    progression_probability_3y: float | None = None
    ensemble_interval: list[float] | None = None
    decision_threshold: float | None = None
    clinical_observed_fraction: float = 0.0
    simulated_fundus_3y: str | None = None
    change_overlay: str | None = None
    uncertainty_map: str | None = None
    preprocessing: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
