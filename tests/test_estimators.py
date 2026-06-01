import numpy as np

from src.ising_kernel import IsingKernel
from src.estimators import (
    estimate_epsilon_mle_grid,
    confidence_interval_epsilon_curvature,
    posterior_over_epsilon_grid,
)


def test_mle_grid_recovers_epsilon_under_h0_near_zero():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.0, n_steps=1500, seed=42)

    eps_grid = np.linspace(0.0, 0.8, 41)  # 0.02 step
    fit = estimate_epsilon_mle_grid(traj, J=0.5, h=0.1, eps_grid=eps_grid)

    # Under H0, MLE should be near 0 on average; with deterministic seed we check closeness
    assert fit.eps_hat_mle <= 0.06


def test_mle_grid_recovers_epsilon_under_h1_near_true():
    kernel = IsingKernel()
    eps_true = 0.3
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=eps_true, n_steps=2000, seed=100)

    eps_grid = np.linspace(0.0, 0.8, 41)
    fit = estimate_epsilon_mle_grid(traj, J=0.5, h=0.1, eps_grid=eps_grid)

    # grid step is 0.02, so allow a modest tolerance
    assert 0.24 <= fit.eps_hat_mle <= 0.36


def test_curvature_ci_runs_and_is_reasonable_or_none():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=1500, seed=101)

    eps_grid = np.linspace(0.0, 0.8, 41)
    fit = estimate_epsilon_mle_grid(traj, J=0.5, h=0.1, eps_grid=eps_grid)
    ci = confidence_interval_epsilon_curvature(fit)

    # CI may be None in edge cases (flat top / boundary), but should not crash.
    assert ci.eps_hat == fit.eps_hat_mle
    if ci.ci_95 is not None:
        lo, hi = ci.ci_95
        assert lo <= ci.eps_hat <= hi
        assert lo >= 0.0


def test_posterior_grid_returns_map_and_interval():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=1500, seed=102)

    eps_grid = np.linspace(0.0, 0.8, 41)
    fit = estimate_epsilon_mle_grid(traj, J=0.5, h=0.1, eps_grid=eps_grid)

    post = posterior_over_epsilon_grid(fit, prior="halfnormal", prior_params={"sigma": 0.5})
    assert "eps_map" in post
    assert "credible_interval_95" in post
    lo, hi = post["credible_interval_95"]
    assert lo <= post["eps_map"] <= hi