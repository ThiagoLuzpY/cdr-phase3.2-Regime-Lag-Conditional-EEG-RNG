import numpy as np

from src.ising_kernel import IsingKernel
from src.validators import (
    gate_G1_H0_recovery,
    gate_G2_H1_recovery,
    gate_G3_controls_collapse,
    gate_G4_identifiability,
)


def test_gate_G1_pass_and_fail():
    r_pass = gate_G1_H0_recovery([0.0, 0.02, 0.03, 0.01])
    assert r_pass.passed

    r_fail = gate_G1_H0_recovery([0.2, 0.2, 0.2, 0.2])
    assert not r_fail.passed


def test_gate_G2_pass_and_fail():
    r_pass = gate_G2_H1_recovery([0.28, 0.30, 0.32])
    assert r_pass.passed

    r_fail = gate_G2_H1_recovery([0.05, 0.10, 0.12])
    assert not r_fail.passed


def test_gate_G3_pass_and_fail():
    r_pass = gate_G3_controls_collapse([0.0, 0.01, 0.03])
    assert r_pass.passed

    r_fail = gate_G3_controls_collapse([0.2, 0.25])
    assert not r_fail.passed


def test_gate_G4_runs_on_short_trajectory():
    # Keep it short to avoid slow tests (finite-diff Fisher/Hessian)
    kernel = IsingKernel()
    traj = kernel.sample_trajectory(J=0.5, h=0.1, epsilon=0.3, n_steps=60, seed=123)

    r = gate_G4_identifiability(traj, J=0.5, h=0.1, epsilon_hat=0.3)
    assert isinstance(r.passed, bool)
    assert "effective_rank_fisher_norm" in r.metrics
    assert "kappa_theta_norm" in r.metrics
    assert "schur_eps" in r.metrics
    assert "r2_eps" in r.metrics