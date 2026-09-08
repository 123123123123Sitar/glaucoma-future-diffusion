"""Progression classification metrics without hard dependency on sklearn."""

from __future__ import annotations

import numpy as np

from .survival_metrics import binary_auc


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    recalls = []
    for label in (0, 1):
        mask = y_true == label
        if mask.any():
            recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    positives = int((y_true == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-y_prob, kind="mergesort")
    truth = y_true[order]
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float(precision[truth == 1].sum() / positives)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    prediction = (y_prob >= threshold).astype(int)
    true_positive = int(((prediction == 1) & (y_true == 1)).sum())
    false_positive = int(((prediction == 1) & (y_true == 0)).sum())
    true_negative = int(((prediction == 0) & (y_true == 0)).sum())
    false_negative = int(((prediction == 0) & (y_true == 1)).sum())
    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else float("nan")
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else float("nan")
    )
    return {
        "n": len(y_true),
        "positives": int((y_true == 1).sum()),
        "auroc": binary_auc(y_true, y_prob),
        "auprc": average_precision(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "threshold": float(threshold),
        "balanced_accuracy": balanced_accuracy(y_true, prediction),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }
