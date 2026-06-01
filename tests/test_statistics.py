import numpy as np

from src.ising_kernel import IsingKernel
from src.statistics import (
    compute_hessian_fisher,
    matrix_rank_symmetric,
    covariance_from_hessian,
    correlation_matrix_from_cov,
)


def test_compute_hessian_fisher_shapes_and_symmetry():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=200, seed=42)

    res = compute_hessian_fisher(traj, J=0.5, h=0.1, epsilon=0.3)

    assert res.hessian.shape == (3, 3)
    assert res.fisher.shape == (3, 3)

    np.testing.assert_allclose(res.hessian, res.hessian.T, atol=1e-10)
    np.testing.assert_allclose(res.fisher, res.fisher.T, atol=1e-10)


def test_rank_and_condition_are_finite_or_reported():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=200, seed=43)

    res = compute_hessian_fisher(traj, J=0.5, h=0.1, epsilon=0.3)

    # rank should be between 0 and 3
    rH = matrix_rank_symmetric(res.hessian, tau_rel=1e-8)
    assert 0 <= rH <= 3

    # condition might be large but must be finite or inf (explicit)
    assert np.isfinite(res.condition_hessian) or np.isinf(res.condition_hessian)


def test_covariance_and_correlation_matrix_are_well_formed():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=200, seed=44)

    res = compute_hessian_fisher(traj, J=0.5, h=0.1, epsilon=0.3)
    cov = covariance_from_hessian(res.hessian, ridge=1e-8)
    corr = correlation_matrix_from_cov(cov)

    assert cov.shape == (3, 3)
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(corr, corr.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(corr), np.ones(3), atol=1e-6)