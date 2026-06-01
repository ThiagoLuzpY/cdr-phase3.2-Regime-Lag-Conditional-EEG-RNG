from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSelectionSummary:
    """Lightweight container for penalized comparison outputs."""
    logL: float
    k_params: int
    n_obs: int
    bic: float


def bic(logL: float, k_params: int, n_obs: int) -> float:
    """Bayesian Information Criterion: BIC = -2 logL + k log n."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    return float(-2.0 * logL + k_params * math.log(n_obs))


def summarize_model(logL: float, k_params: int, n_obs: int) -> ModelSelectionSummary:
    return ModelSelectionSummary(
        logL=float(logL),
        k_params=int(k_params),
        n_obs=int(n_obs),
        bic=bic(logL, k_params, n_obs),
    )