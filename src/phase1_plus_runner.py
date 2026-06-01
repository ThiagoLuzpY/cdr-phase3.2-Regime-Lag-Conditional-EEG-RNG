from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.adversarial_kernel import AdversarialIsingKernel
from src.artifacts import plot_histograms, write_json, write_seed_manifest, write_text
from src.controls import run_control_suite
from src.estimators import estimate_epsilon_mle_grid, loglik_trajectory_h1
from src.ising_kernel import IsingKernel
from src.model_selection import summarize_model
from src.validators import (
    GateResult,
    gate_G1_H0_recovery,
    gate_G2_H1_recovery,
    gate_G3_controls_collapse,
    gate_G4_identifiability,
)


@dataclass(frozen=True)
class Phase1PlusConfig:
    # Baseline parameters
    J: float = 0.5
    h: float = 0.1

    # Trajectory
    n_steps: int = 1000

    # H1 truth
    eps_true_h1: float = 0.3

    # Grid for epsilon inference
    eps_grid: Tuple[float, ...] = tuple(np.linspace(0.0, 0.8, 41))

    # Replicates
    n_reps: int = 20
    h0_seed_start: int = 42
    h1_seed_start: int = 100
    controls_seed: int = 99999

    # Phase I+ knobs
    stability_steps_alt: int = 800  # a mild perturbation in n_steps (within envelope-like idea)
    oos_train_frac: float = 0.6

    # Adversarial baseline ladder
    g_grid: Tuple[float, ...] = tuple(np.linspace(0.0, 0.25, 26))  # bounded extra flexibility


class Phase1PlusValidator:
    def __init__(self, cfg: Phase1PlusConfig):
        self.cfg = cfg
        self.kernel = IsingKernel()
        self.adv_kernel = AdversarialIsingKernel()

    # ---------------------------
    # Core simulation + epsilon fits
    # ---------------------------
    def simulate_many(self, epsilon_true: float, seed_start: int, n_steps: int) -> List[np.ndarray]:
        trajs: List[np.ndarray] = []
        for r in range(self.cfg.n_reps):
            seed = seed_start + r
            traj = self.kernel.sample_trajectory(
                J=self.cfg.J, h=self.cfg.h, epsilon=float(epsilon_true), n_steps=int(n_steps), seed=int(seed)
            )
            trajs.append(traj)
        return trajs

    def estimate_eps_many(self, trajs: List[np.ndarray]) -> np.ndarray:
        eps_hat = np.zeros(len(trajs), dtype=float)
        for i, traj in enumerate(trajs):
            fit = estimate_epsilon_mle_grid(traj, J=self.cfg.J, h=self.cfg.h, eps_grid=self.cfg.eps_grid, kernel=self.kernel)
            eps_hat[i] = fit.eps_hat_mle
        return eps_hat

    # ---------------------------
    # Phase I+ Gate G5: stability
    # ---------------------------
    def gate_G5_stability(self, eps_h1_primary: np.ndarray, eps_h1_alt: np.ndarray) -> GateResult:
        # Stability = sign preserved and mean does not shift drastically
        m0 = float(np.mean(eps_h1_primary))
        m1 = float(np.mean(eps_h1_alt))
        sign_ok = (m0 > 0.0) and (m1 > 0.0)
        drift = abs(m1 - m0)
        # conservative thresholds for Phase I+ (tunable later)
        passed = sign_ok and (drift < 0.08)
        return GateResult(
            name="G5_stability",
            passed=passed,
            metrics={"mean_eps_primary": m0, "mean_eps_alt": m1, "abs_drift": float(drift)},
            thresholds={"max_abs_drift": 0.08},
            notes="Stability across a mild pre-registered perturbation (n_steps).",
        )

    # ---------------------------
    # Phase I+ Gate G6: adversarial baseline absorption
    # ---------------------------
    def fit_adversarial_g_mle(self, traj: np.ndarray) -> Tuple[float, float]:
        """Fit g by grid MLE under adversarial null model (epsilon fixed to 0)."""
        best_g = None
        best_ll = None
        for g in self.cfg.g_grid:
            ll = self.adv_kernel.loglik_h0_adv(traj, J=self.cfg.J, h=self.cfg.h, g=float(g))
            if best_ll is None or ll > best_ll:
                best_ll = ll
                best_g = float(g)
        assert best_g is not None and best_ll is not None
        return best_g, float(best_ll)

    def gate_G6_adversarial(self, traj_h1: np.ndarray, eps_hat_rep: float) -> GateResult:
        """
        Compare:
          M1: base CDR model (epsilon free, baseline fixed) -> logL_H1
          M0': adversarial baseline (g free), epsilon=0 -> logL_adv

        Penalized comparison uses BIC with k=1 in both cases (one free parameter),
        so preference reflects fit rather than parameter count inflation.
        """
        T = traj_h1.shape[0] - 1

        # M1 (CDR)
        logL_h1 = loglik_trajectory_h1(traj_h1, J=self.cfg.J, h=self.cfg.h, epsilon=float(eps_hat_rep), kernel=self.kernel)
        m1 = summarize_model(logL_h1, k_params=1, n_obs=T)

        # M0' (adversarial null)
        g_hat, logL_adv = self.fit_adversarial_g_mle(traj_h1)
        m0 = summarize_model(logL_adv, k_params=1, n_obs=T)

        delta_bic = float(m1.bic - m0.bic)  # negative => M1 better

        # Pass if M1 is preferred (delta_bic < 0) AND epsilon does not collapse
        passed = (delta_bic < 0.0) and (eps_hat_rep >= 0.10)

        return GateResult(
            name="G6_adversarial",
            passed=passed,
            metrics={
                "eps_hat_rep": float(eps_hat_rep),
                "g_hat_adv": float(g_hat),
                "logL_M1": float(m1.logL),
                "logL_M0adv": float(m0.logL),
                "bic_M1": float(m1.bic),
                "bic_M0adv": float(m0.bic),
                "delta_bic_M1_minus_M0adv": float(delta_bic),
            },
            thresholds={"requires_delta_bic_negative": 0.0, "eps_min": 0.10},
            notes="Adversarial baseline absorption check (bounded flexibility).",
        )

    # ---------------------------
    # Phase I+ Gate G7: out-of-sample consistency
    # ---------------------------
    def gate_G7_out_of_sample(self, traj_h1: np.ndarray) -> GateResult:
        T = traj_h1.shape[0] - 1
        split = int(self.cfg.oos_train_frac * (T + 1))
        split = max(10, min(split, traj_h1.shape[0] - 10))

        train = traj_h1[:split].copy()
        test = traj_h1[split - 1 :].copy()  # preserve continuity

        fit = estimate_epsilon_mle_grid(train, J=self.cfg.J, h=self.cfg.h, eps_grid=self.cfg.eps_grid, kernel=self.kernel)
        eps_hat = float(fit.eps_hat_mle)

        # Evaluate log-likelihood on TEST under H1(eps_hat) vs H0(eps=0)
        logL_test_h1 = loglik_trajectory_h1(test, J=self.cfg.J, h=self.cfg.h, epsilon=eps_hat, kernel=self.kernel)
        logL_test_h0 = loglik_trajectory_h1(test, J=self.cfg.J, h=self.cfg.h, epsilon=0.0, kernel=self.kernel)

        n_test = test.shape[0] - 1
        m1 = summarize_model(logL_test_h1, k_params=0, n_obs=n_test)  # epsilon fitted on train; test has no extra k
        m0 = summarize_model(logL_test_h0, k_params=0, n_obs=n_test)

        passed = (m1.bic < m0.bic) and (eps_hat >= 0.10)

        return GateResult(
            name="G7_out_of_sample",
            passed=passed,
            metrics={
                "eps_hat_train": eps_hat,
                "logL_test_h1": float(logL_test_h1),
                "logL_test_h0": float(logL_test_h0),
                "bic_test_h1": float(m1.bic),
                "bic_test_h0": float(m0.bic),
                "n_test": float(n_test),
            },
            thresholds={"eps_min": 0.10, "requires_bic_h1_less": 1.0},
            notes="Train epsilon on train segment; require improvement on held-out test segment.",
        )

    # ---------------------------
    # Full Phase I+ run
    # ---------------------------
    def run_phase1_plus(self, out_dir: str = "results/phase1_plus_run") -> Dict[str, object]:
        out = Path(out_dir)

        # H0/H1 simulations (primary)
        h0_trajs = self.simulate_many(0.0, self.cfg.h0_seed_start, self.cfg.n_steps)
        h1_trajs = self.simulate_many(self.cfg.eps_true_h1, self.cfg.h1_seed_start, self.cfg.n_steps)

        eps_h0 = self.estimate_eps_many(h0_trajs)
        eps_h1 = self.estimate_eps_many(h1_trajs)

        # Controls (on representative H1 traj)
        controls = run_control_suite(
            trajectory=h1_trajs[0],
            J=self.cfg.J,
            h=self.cfg.h,
            eps_grid=self.cfg.eps_grid,
            controls_seed=self.cfg.controls_seed,
            kernel=self.kernel,
        )
        eps_controls = np.array([c.eps_hat for c in controls], dtype=float)

        # Core gates (G1–G4)
        g1 = gate_G1_H0_recovery(eps_h0)
        g2 = gate_G2_H1_recovery(eps_h1)
        g3 = gate_G3_controls_collapse(eps_controls)

        # Identifiability on truncated representative trajectory (keep runtime sane)
        traj_diag = h1_trajs[0][:201].copy() if h1_trajs[0].shape[0] > 201 else h1_trajs[0]
        g4 = gate_G4_identifiability(traj_diag, J=self.cfg.J, h=self.cfg.h, epsilon_hat=float(eps_h1[0]))

        # G5 stability: compare with mild perturbation (n_steps_alt)
        h1_alt = self.simulate_many(self.cfg.eps_true_h1, self.cfg.h1_seed_start, self.cfg.stability_steps_alt)
        eps_h1_alt = self.estimate_eps_many(h1_alt)
        g5 = self.gate_G5_stability(eps_h1_primary=eps_h1, eps_h1_alt=eps_h1_alt)

        # G6 adversarial baseline
        g6 = self.gate_G6_adversarial(traj_h1=h1_trajs[0], eps_hat_rep=float(eps_h1[0]))

        # G7 out-of-sample
        g7 = self.gate_G7_out_of_sample(traj_h1=h1_trajs[0])

        gates = {g.name: g for g in [g1, g2, g3, g4, g5, g6, g7]}

        # Write artifacts
        seeds = {
            "h0_seeds": list(range(self.cfg.h0_seed_start, self.cfg.h0_seed_start + self.cfg.n_reps)),
            "h1_seeds": list(range(self.cfg.h1_seed_start, self.cfg.h1_seed_start + self.cfg.n_reps)),
            "controls_seed": self.cfg.controls_seed,
        }
        write_seed_manifest(out / "seed_manifest.json", seeds)

        plot_histograms(out / "plots", eps_h0, eps_h1, eps_controls)

        summary = {
            "config": self.cfg.__dict__,
            "eps_h0": eps_h0,
            "eps_h1": eps_h1,
            "eps_controls": eps_controls,
            "gates": {k: {"passed": v.passed, "metrics": v.metrics, "thresholds": v.thresholds, "notes": v.notes} for k, v in gates.items()},
            "phase1plus_pass": bool(all(g.passed for g in gates.values())),
        }
        write_json(out / "phase1plus_summary.json", summary)

        report_lines = ["CDR Phase I+ — Gates G1–G7", "-" * 60]
        for name in ["G1_H0_recovery", "G2_H1_recovery", "G3_controls_collapse", "G4_identifiability", "G5_stability", "G6_adversarial", "G7_out_of_sample"]:
            g = gates[name]
            report_lines.append(f"{name}: {'PASS' if g.passed else 'FAIL'}")
            report_lines.append(f"  metrics: {g.metrics}")
            report_lines.append(f"  thresholds: {g.thresholds}")
            report_lines.append(f"  notes: {g.notes}")
            report_lines.append("")
        report_lines.append(f"FINAL: {'PASS' if summary['phase1plus_pass'] else 'FAIL'}")
        write_text(out / "gate_report.txt", "\n".join(report_lines))

        return summary