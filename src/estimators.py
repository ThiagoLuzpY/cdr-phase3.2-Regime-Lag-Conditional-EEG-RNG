from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.ising_kernel import IsingKernel


@dataclass(frozen=True)
class EpsilonGridFit:
    """Results from epsilon grid fitting."""
    eps_grid: np.ndarray
    loglik: np.ndarray
    eps_hat_mle: float
    loglik_max: float


@dataclass(frozen=True)
class EpsilonInterval:
    """Uncertainty interval for epsilon."""
    eps_hat: float
    ci_95: Optional[Tuple[float, float]]
    se: Optional[float]
    info: Dict[str, float]


def loglik_trajectory_h1(
    trajectory: np.ndarray,
    J: float,
    h: float,
    epsilon: float,
    kernel: Optional[IsingKernel] = None,
) -> float:
    """Exact trajectory log-likelihood under H1 using P_epsilon.

    Args:
        trajectory: array shape (T+1, 2), entries in {0,1}.
        J: baseline coupling parameter.
        h: baseline temporal alignment parameter.
        epsilon: reweighting parameter (H1).
        kernel: optional IsingKernel instance.

    Returns:
        Total log-likelihood.
    """
    if kernel is None:
        kernel = IsingKernel()

    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(f"trajectory must have shape (T+1,2). got {trajectory.shape}")
    if trajectory.shape[0] < 2:
        raise ValueError("trajectory must contain at least two states.")

    ll = 0.0
    for t in range(trajectory.shape[0] - 1):
        I = (int(trajectory[t, 0]), int(trajectory[t, 1]))
        Ip = (int(trajectory[t + 1, 0]), int(trajectory[t + 1, 1]))
        p = kernel.compute_Peps(Ip, I, J=J, h=h, epsilon=epsilon)
        ll += np.log(max(p, 1e-300))
    return float(ll)


def estimate_epsilon_mle_grid(
    trajectory: np.ndarray,
    J: float,
    h: float,
    eps_grid: Sequence[float],
    kernel: Optional[IsingKernel] = None,
) -> EpsilonGridFit:
    """Estimate epsilon by grid MLE (Phase I canonical, auditable).

    Args:
        trajectory: Markov trajectory (T+1,2).
        J, h: baseline parameters (fixed in Mode A/B).
        eps_grid: candidate epsilon values. Recommend nonnegative grid [0, eps_max].
        kernel: optional kernel instance.

    Returns:
        EpsilonGridFit containing curve and MLE epsilon_hat.
    """
    if kernel is None:
        kernel = IsingKernel()

    eps_grid_arr = np.array(list(eps_grid), dtype=float)
    if eps_grid_arr.ndim != 1 or eps_grid_arr.size < 3:
        raise ValueError("eps_grid must be a 1D sequence with >=3 values.")
    if np.any(np.diff(eps_grid_arr) <= 0):
        raise ValueError("eps_grid must be strictly increasing.")

    loglik = np.zeros_like(eps_grid_arr, dtype=float)
    for i, eps in enumerate(eps_grid_arr):
        loglik[i] = loglik_trajectory_h1(trajectory, J=J, h=h, epsilon=float(eps), kernel=kernel)

    j = int(np.argmax(loglik))
    return EpsilonGridFit(
        eps_grid=eps_grid_arr,
        loglik=loglik,
        eps_hat_mle=float(eps_grid_arr[j]),
        loglik_max=float(loglik[j]),
    )


def _curvature_1d_from_grid(
    eps_grid: np.ndarray,
    loglik: np.ndarray,
    idx_hat: int,
) -> Optional[float]:
    """Estimate second derivative d2/dε2 loglik at eps_hat via local quadratic fit.

    Returns:
        curvature (second derivative) or None if not computable.
        Note: At a maximum, curvature should be negative.
    """
    # Need at least one neighbor on each side
    if idx_hat <= 0 or idx_hat >= len(eps_grid) - 1:
        return None

    e0, e1, e2 = eps_grid[idx_hat - 1], eps_grid[idx_hat], eps_grid[idx_hat + 1]
    l0, l1, l2 = loglik[idx_hat - 1], loglik[idx_hat], loglik[idx_hat + 1]

    # Use symmetric finite-difference second derivative if grid spacing is uniform-ish
    # If not uniform, fallback to local quadratic fit.
    h1 = e1 - e0
    h2 = e2 - e1

    # If near-uniform spacing, use classic formula
    if abs(h1 - h2) <= 1e-12:
        h = h1
        return (l0 - 2.0 * l1 + l2) / (h * h)

    # Quadratic fit: l(e) ≈ a e^2 + b e + c, curvature = 2a
    A = np.array([[e0**2, e0, 1.0], [e1**2, e1, 1.0], [e2**2, e2, 1.0]], dtype=float)
    y = np.array([l0, l1, l2], dtype=float)
    try:
        a, _, _ = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        return None
    return 2.0 * float(a)


def confidence_interval_epsilon_curvature(
    fit: EpsilonGridFit,
    alpha: float = 0.05,
    enforce_nonnegative: bool = True,
) -> EpsilonInterval:
    """Approximate a 1D Wald-type CI for epsilon from loglik curvature at MLE.

    For 1D parameter epsilon (theta fixed), observed information:
        I(eps_hat) ≈ - d²/dε² log L |_{eps_hat}
    so:
        Var(eps_hat) ≈ 1 / I(eps_hat), SE = sqrt(Var)

    If curvature is nonnegative or undefined, CI is returned as None.

    Args:
        fit: EpsilonGridFit from estimate_epsilon_mle_grid.
        alpha: significance level (default 0.05 => 95% CI).
        enforce_nonnegative: if True, clamp CI lower bound at 0.

    Returns:
        EpsilonInterval with ci_95 or None if curvature invalid.
    """
    eps_grid = fit.eps_grid
    loglik = fit.loglik
    idx_hat = int(np.argmax(loglik))

    curvature = _curvature_1d_from_grid(eps_grid, loglik, idx_hat)
    info: Dict[str, float] = {"curvature_d2_logL": float(curvature) if curvature is not None else float("nan")}

    if curvature is None:
        return EpsilonInterval(eps_hat=fit.eps_hat_mle, ci_95=None, se=None, info=info)

    # Observed information is negative curvature at maximum
    I_obs = -curvature
    info["I_obs"] = float(I_obs)

    if not np.isfinite(I_obs) or I_obs <= 0:
        # Not a proper maximum / too flat / numerical issue
        return EpsilonInterval(eps_hat=fit.eps_hat_mle, ci_95=None, se=None, info=info)

    se = float(np.sqrt(1.0 / I_obs))

    # Normal quantile for 95% (avoid scipy.stats dependency)
    # z_{0.975} ≈ 1.95996398454
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else 1.959963984540054

    lo = fit.eps_hat_mle - z * se
    hi = fit.eps_hat_mle + z * se

    if enforce_nonnegative:
        lo = max(0.0, lo)

    return EpsilonInterval(
        eps_hat=fit.eps_hat_mle,
        ci_95=(float(lo), float(hi)),
        se=se,
        info=info,
    )


def halfnormal_logprior(eps: np.ndarray, sigma: float) -> np.ndarray:
    """Half-normal log prior (up to additive constant) for epsilon >= 0."""
    out = np.full_like(eps, -np.inf, dtype=float)
    mask = eps >= 0
    out[mask] = -0.5 * (eps[mask] / float(sigma)) ** 2
    return out


def posterior_over_epsilon_grid(
    fit: EpsilonGridFit,
    prior: str = "halfnormal",
    prior_params: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Compute normalized posterior mass over an epsilon grid (optional Phase I support).

    This is useful for:
    - MAP estimate with explicit prior
    - grid credible intervals
    - sanity checks vs curvature CI

    Args:
        fit: EpsilonGridFit (contains loglik and eps_grid).
        prior: prior name (currently "halfnormal").
        prior_params: e.g. {"sigma": 0.5}.

    Returns:
        dict with posterior weights, MAP, credible interval, etc.
    """
    prior_params = prior_params or {"sigma": 0.5}
    eps = fit.eps_grid

    if prior == "halfnormal":
        logprior = halfnormal_logprior(eps, sigma=float(prior_params["sigma"]))
    else:
        raise ValueError(f"Unsupported prior: {prior}")

    logpost = fit.loglik + logprior

    # Normalize in log-space
    m = float(np.max(logpost))
    w = np.exp(logpost - m)
    w = w / np.sum(w)

    # MAP
    j = int(np.argmax(logpost))
    eps_map = float(eps[j])

    # Equal-tail 95% credible interval
    cdf = np.cumsum(w)
    lo = float(np.interp(0.025, cdf, eps))
    hi = float(np.interp(0.975, cdf, eps))

    return {
        "eps_grid": eps,
        "loglik": fit.loglik,
        "logprior": logprior,
        "logposterior_unnorm": logpost,
        "posterior_weights": w,
        "eps_map": eps_map,
        "credible_interval_95": (lo, hi),
    }