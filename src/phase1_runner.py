from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.controls import run_control_suite
from src.estimators import estimate_epsilon_mle_grid
from src.ising_kernel import IsingKernel
from src.validators import (
    GateResult,
    gate_G1_H0_recovery,
    gate_G2_H1_recovery,
    gate_G3_controls_collapse,
    gate_G4_identifiability,
)


@dataclass(frozen=True)
class Phase1RunConfig:
    J: float = 0.5
    h: float = 0.1
    n_steps: int = 1000

    eps_true_h1: float = 0.3
    eps_grid: Tuple[float, ...] = tuple(np.linspace(0.0, 0.8, 41))

    n_reps: int = 20
    h0_seed_start: int = 42
    h1_seed_start: int = 100
    controls_seed: int = 99999


class Phase1Validator:
    def __init__(self, config: Phase1RunConfig):
        self.cfg = config
        self.kernel = IsingKernel()

    def simulate_many(self, epsilon_true: float, seed_start: int) -> List[np.ndarray]:
        trajs: List[np.ndarray] = []
        for r in range(self.cfg.n_reps):
            seed = seed_start + r
            traj = self.kernel.sample_trajectory(
                J=self.cfg.J,
                h=self.cfg.h,
                epsilon=float(epsilon_true),
                n_steps=self.cfg.n_steps,
                seed=seed,
            )
            trajs.append(traj)
        return trajs

    def estimate_eps_many(self, trajs: Sequence[np.ndarray]) -> List[float]:
        eps_hats: List[float] = []
        for traj in trajs:
            fit = estimate_epsilon_mle_grid(
                traj, J=self.cfg.J, h=self.cfg.h, eps_grid=self.cfg.eps_grid, kernel=self.kernel
            )
            eps_hats.append(fit.eps_hat_mle)
        return eps_hats

    def run_controls_on(self, traj: np.ndarray) -> List[float]:
        controls = run_control_suite(
            trajectory=traj,
            J=self.cfg.J,
            h=self.cfg.h,
            eps_grid=self.cfg.eps_grid,
            controls_seed=self.cfg.controls_seed,
            kernel=self.kernel,
        )
        return [c.eps_hat for c in controls]

    def run_all_gates(self) -> Dict[str, GateResult]:
        # Simulations
        h0_trajs = self.simulate_many(epsilon_true=0.0, seed_start=self.cfg.h0_seed_start)
        h1_trajs = self.simulate_many(epsilon_true=self.cfg.eps_true_h1, seed_start=self.cfg.h1_seed_start)

        # Epsilon estimates
        eps_h0 = self.estimate_eps_many(h0_trajs)
        eps_h1 = self.estimate_eps_many(h1_trajs)

        # Controls: apply to a representative H1 trajectory (strong test)
        control_eps = self.run_controls_on(h1_trajs[0])

        # Gates G1-G3
        g1 = gate_G1_H0_recovery(eps_h0)
        g2 = gate_G2_H1_recovery(eps_h1)
        g3 = gate_G3_controls_collapse(control_eps)

        # Gate G4: identifiability on representative H1 trajectory at eps_hat (use first)
        eps_hat_rep = float(eps_h1[0])
        g4 = gate_G4_identifiability(h1_trajs[0], J=self.cfg.J, h=self.cfg.h, epsilon_hat=eps_hat_rep)

        return {g.name: g for g in [g1, g2, g3, g4]}

    def summary_report(self, gates: Dict[str, GateResult]) -> str:
        lines = []
        lines.append("CDR Phase I — Gate Summary (G1–G4)")
        lines.append("-" * 48)
        for k in sorted(gates.keys()):
            g = gates[k]
            status = "PASS" if g.passed else "FAIL"
            lines.append(f"{g.name}: {status}")
            if g.metrics:
                lines.append(f"  metrics: {g.metrics}")
            if g.thresholds:
                lines.append(f"  thresholds: {g.thresholds}")
            if g.notes:
                lines.append(f"  notes: {g.notes}")
            lines.append("")
        return "\n".join(lines)