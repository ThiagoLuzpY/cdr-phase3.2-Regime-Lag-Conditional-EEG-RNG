import numpy as np
import pytest

from src.ising_kernel import IsingKernel


def test_p0_distribution_normalizes_for_all_states():
    kernel = IsingKernel()
    J = 0.5
    h = 0.1

    for state in kernel.states:
        probs = kernel.p0_distribution(state, J=J, h=h)
        assert probs.shape == (4,)
        assert np.all(probs > 0.0)
        np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-12)


def test_peps_distribution_normalizes_for_all_states():
    kernel = IsingKernel()
    J = 0.5
    h = 0.1
    epsilon = 0.3

    for state in kernel.states:
        probs = kernel.peps_distribution(state, J=J, h=h, epsilon=epsilon)
        assert probs.shape == (4,)
        assert np.all(probs > 0.0)
        np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-12)


def test_peps_matches_p0_when_epsilon_zero():
    kernel = IsingKernel()
    J = 0.5
    h = 0.1

    for state in kernel.states:
        p0 = kernel.p0_distribution(state, J=J, h=h)
        peps = kernel.peps_distribution(state, J=J, h=h, epsilon=0.0)
        np.testing.assert_allclose(peps, p0, atol=1e-12)


def test_delta_chi_is_not_identically_zero_for_coupled_kernel():
    kernel = IsingKernel()
    J = 0.5
    h = 0.1

    values = []
    for curr in kernel.states:
        for nxt in kernel.states:
            values.append(kernel.compute_delta_chi(nxt, curr, J=J, h=h))

    values = np.array(values, dtype=float)
    # At least one transition should have a clearly nonzero Δχ
    assert np.any(np.abs(values) > 1e-8), "Δχ unexpectedly collapsed to zero everywhere."


def test_sample_trajectory_shape_and_binary_values():
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=100, seed=42)

    assert traj.shape == (101, 2)
    assert np.issubdtype(traj.dtype, np.integer)
    assert np.all(np.isin(traj, [0, 1]))


def test_sample_trajectory_is_deterministic_given_seed():
    kernel = IsingKernel()
    traj1 = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=50, seed=123)
    traj2 = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=50, seed=123)

    np.testing.assert_array_equal(traj1, traj2)


def test_e6_artifact_bundle_contains_required_keys():
    kernel = IsingKernel()
    bundle = kernel.e6_artifact_bundle(J=0.5, h=0.1, epsilon=0.3, n_steps=20, seed=42)

    required = {
        "trajectory",
        "config_snapshot",
        "rng_seed_manifest",
        "delta_chi_per_step",
        "rho_hat",
    }
    assert required.issubset(bundle.keys())

    traj = bundle["trajectory"]
    assert isinstance(traj, np.ndarray)
    assert traj.shape == (21, 2)

    delta = bundle["delta_chi_per_step"]
    assert isinstance(delta, np.ndarray)
    assert delta.shape == (20,)