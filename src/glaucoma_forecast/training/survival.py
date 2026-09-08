"""Discrete-time survival utilities for 1-to-10-year glaucoma incidence."""

from __future__ import annotations

from glaucoma_forecast.models._torch import require_torch


def cumulative_risk_from_logits(hazard_logits):
    torch = require_torch()
    hazards = torch.sigmoid(hazard_logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return 1.0 - survival


def discrete_time_survival_loss(
    hazard_logits,
    event_or_censor_time_years,
    event_observed,
    sample_weight=None,
):
    """Negative log likelihood with right-censoring in annual intervals.

    An event at t in (k-1, k] contributes survival through k-1 and event
    likelihood in k. A censor time contributes only fully observed intervals.
    """

    torch = require_torch()
    if hazard_logits.ndim != 2:
        raise ValueError("hazard_logits must have shape [batch, years]")
    years = hazard_logits.shape[1]
    time = event_or_censor_time_years.float().clamp(min=0.0, max=float(years))
    observed = event_observed.float()
    bins = torch.arange(1, years + 1, device=hazard_logits.device)[None, :]
    event_bin = torch.ceil(time).clamp(min=1.0).long()
    survival_mask = bins < event_bin[:, None]
    censored_mask = bins <= torch.floor(time)[:, None]
    event_mask = (bins == event_bin[:, None]) & (observed[:, None] > 0.5)
    at_risk_mask = torch.where(observed[:, None] > 0.5, survival_mask, censored_mask)
    log_survival = torch.nn.functional.logsigmoid(-hazard_logits)
    log_event = torch.nn.functional.logsigmoid(hazard_logits)
    log_likelihood = (log_survival * at_risk_mask).sum(dim=1) + (log_event * event_mask).sum(dim=1)
    if sample_weight is None:
        return -log_likelihood.mean()
    weight = sample_weight.float()
    if weight.ndim != 1 or weight.shape[0] != hazard_logits.shape[0]:
        raise ValueError("sample_weight must have shape [batch]")
    return -(log_likelihood * weight).sum() / weight.sum().clamp(min=1.0)
