"""Dependency-light horizon metrics for censored glaucoma incidence data."""

from __future__ import annotations

import numpy as np


def status_known_at_horizon(
    event_or_censor_time: np.ndarray,
    event_observed: np.ndarray,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return evaluation mask and binary event-by-horizon labels."""

    time = np.asarray(event_or_censor_time, dtype=float)
    observed = np.asarray(event_observed, dtype=int)
    known_event = (observed == 1) & (time <= horizon)
    known_control = time >= horizon
    known = known_event | known_control
    labels = known_event.astype(int)
    return known, labels


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    n_pos = positives.sum()
    n_neg = negatives.sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def horizon_metrics(
    event_or_censor_time: np.ndarray,
    event_observed: np.ndarray,
    predicted_risk: np.ndarray,
    horizon: float,
) -> dict[str, float | int]:
    known, labels = status_known_at_horizon(
        event_or_censor_time, event_observed, horizon
    )
    risk = np.asarray(predicted_risk, dtype=float)[known]
    labels = labels[known]
    if not known.any():
        return {"horizon": horizon, "n_known": 0, "events": 0, "auc": float("nan"), "brier": float("nan")}
    return {
        "horizon": horizon,
        "n_known": int(known.sum()),
        "events": int(labels.sum()),
        "auc": binary_auc(labels, risk),
        "brier": float(np.mean((risk - labels) ** 2)),
        "mean_predicted_risk": float(risk.mean()),
        "observed_event_fraction": float(labels.mean()),
    }
