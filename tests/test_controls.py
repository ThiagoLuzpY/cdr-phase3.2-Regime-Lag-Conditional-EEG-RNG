import numpy as np

from src.controls import (
    time_shuffle_trajectory,
    marginal_preserving_surrogate,
    autocorr_preserving_surrogate_circular_shift,
)
from src.ising_kernel import IsingKernel


def _counts_per_component(traj: np.ndarray):
    c0 = np.bincount(traj[:, 0], minlength=2)
    c1 = np.bincount(traj[:, 1], minlength=2)
    return c0, c1


def test_time_shuffle_preserves_multiset_of_states():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=200, seed=42)

    shuffled = time_shuffle_trajectory(traj, seed=99999)

    # multiset of rows preserved: compare sorted rows
    a = np.sort(traj.view([("a", traj.dtype), ("b", traj.dtype)]), axis=0)
    b = np.sort(shuffled.view([("a", shuffled.dtype), ("b", shuffled.dtype)]), axis=0)
    assert np.array_equal(a, b)


def test_marginal_preserving_surrogate_preserves_component_marginals():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=200, seed=43)

    c0_before, c1_before = _counts_per_component(traj)
    sur = marginal_preserving_surrogate(traj, seed=1234)
    c0_after, c1_after = _counts_per_component(sur)

    assert np.array_equal(c0_before, c0_after)
    assert np.array_equal(c1_before, c1_after)


def test_circular_shift_preserves_component_marginals():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=200, seed=44)

    c0_before, c1_before = _counts_per_component(traj)
    sur = autocorr_preserving_surrogate_circular_shift(traj, seed=999)
    c0_after, c1_after = _counts_per_component(sur)

    assert np.array_equal(c0_before, c0_after)
    assert np.array_equal(c1_before, c1_after)