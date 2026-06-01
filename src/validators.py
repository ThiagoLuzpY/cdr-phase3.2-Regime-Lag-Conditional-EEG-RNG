from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.statistics import compute_hessian_fisher


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    notes: str = ""


def gate_G1_H0_recovery(eps_hats_h0: Sequence[float], mean_thr: float = 0.05, std_thr: float = 0.10) -> GateResult:
    eps = np.array(list(eps_hats_h0), dtype=float)
    m = float(np.mean(eps))
    s = float(np.std(eps))
    passed = (m < mean_thr) and (s < std_thr)
    return GateResult(
        name="G1_H0_recovery",
        passed=passed,
        metrics={"mean_eps_hat": m, "std_eps_hat": s},
        thresholds={"mean_thr": mean_thr, "std_thr": std_thr},
        notes="H0 trajectories should yield epsilon_hat near 0.",
    )


def gate_G2_H1_recovery(eps_hats_h1: Sequence[float], lo: float = 0.25, hi: float = 0.35) -> GateResult:
    eps = np.array(list(eps_hats_h1), dtype=float)
    m = float(np.mean(eps))
    passed = (lo < m < hi)
    return GateResult(
        name="G2_H1_recovery",
        passed=passed,
        metrics={"mean_eps_hat": m},
        thresholds={"lo": lo, "hi": hi},
        notes="H1 trajectories should recover epsilon_true (within band).",
    )


def gate_G3_controls_collapse(
    control_eps_hats: Sequence[float],
    tol: float = 0.05,
    required_fraction: float = 2/3,
) -> GateResult:
    """
    Gate G3 (Controls Collapse) — robust collapse criterion.

    Rationale:
      With only a small number of control replicates (e.g., 3 canonical controls),
      the mean can be dominated by a single outlier. CDR collapse is better treated
      as *consistent* collapse across a majority of controls, using a robust center.

    Pass criteria (same tolerance, no loosening):
      (1) median(eps_hat_controls) < tol
      (2) fraction of controls with eps_hat <= tol is >= required_fraction
    """
    eps = np.array(list(control_eps_hats), dtype=float)
    if eps.size == 0:
        return GateResult(
            name="G3_controls_collapse",
            passed=False,
            metrics={"n_controls": 0.0},
            thresholds={"tol": tol, "required_fraction": required_fraction},
            notes="No control results provided.",
        )

    med = float(np.median(eps))
    frac = float(np.mean(eps <= tol))
    m = float(np.mean(eps))
    mx = float(np.max(eps))

    passed = (med < tol) and (frac >= required_fraction)

    return GateResult(
        name="G3_controls_collapse",
        passed=passed,
        metrics={
            "mean_eps_hat_controls": m,
            "median_eps_hat_controls": med,
            "max_eps_hat_controls": mx,
            "fraction_below_tol": frac,
            "n_controls": float(eps.size),
        },
        thresholds={"tol": float(tol), "required_fraction": float(required_fraction)},
        notes="Controls must collapse robustly: median below tol and majority below tol.",
    )


def gate_G4_identifiability(
    trajectory: np.ndarray,
    J: float,
    h: float,
    epsilon_hat: float,
    rank_thr: int = 3,
    kappa_thr: float = 100.0,
    schur_min: float = 0.005,   # strict: requires R^2 <= 0.98
) -> GateResult:
    """
    Identifiability gate focused on ε-direction identifiability (CDR-consistent).

    We avoid using the *global* condition number of the full Fisher because it can
    be dominated by near-collinearity inside θ=(J,h) even when ε is identifiable.

    Instead, we test:
      (1) effective rank of normalized Fisher (scale-invariant) is full
      (2) ε is not absorbed by θ using the Schur complement of the ε block:
            S = Fεε - Fεθ Fθθ^{-1} Fθε
          For correlation-scaled Fisher, S = 1 - R^2, where R^2 is from regressing
          s_ε on (s_J, s_h). Small S means ε is nearly collinear -> not identifiable.
      (3) θ-block conditioning is sane (optional stability check), using kappa on Fθθ.

    This is stricter and more theory-aligned than a global kappa(F_norm) cutoff.
    """
    res = compute_hessian_fisher(trajectory, J=J, h=h, epsilon=epsilon_hat)

    F = 0.5 * (res.fisher + res.fisher.T)

    # --- scale-invariant normalization (correlation-scaled Fisher) ---
    d = np.diag(F)
    d = np.clip(d, 1e-12, None)
    Dinv = 1.0 / np.sqrt(d)
    F_norm = (Dinv[:, None] * F) * Dinv[None, :]

    # effective rank on normalized Fisher
    eig = np.linalg.eigvalsh(F_norm)
    eig = np.sort(np.real(eig))[::-1]
    lam_max = eig[0] if eig.size > 0 else 0.0
    tau_rel = 1e-8
    rank_norm = int(np.sum((eig / lam_max) > tau_rel)) if lam_max > 0 else 0

    # θ block (J,h) and ε index (psi = [J, h, ε])
    Ftt = F_norm[:2, :2]
    Fte = F_norm[:2, 2]
    Fet = F_norm[2, :2]
    Fee = float(F_norm[2, 2])

    # conditioning of θ-block only (avoid punishing ε because J/h are collinear)
    eig_t = np.linalg.eigvalsh(0.5 * (Ftt + Ftt.T))
    eig_t = np.sort(np.real(eig_t))[::-1]
    if eig_t.size == 2 and eig_t[0] > 0 and eig_t[1] > 0:
        kappa_theta = float(eig_t[0] / eig_t[1])
    else:
        kappa_theta = float("inf")

    # Schur complement for ε given θ: S = Fee - Fet Ftt^{-1} Fte
    # For correlation-scaled Fisher, Fee=1, so S = 1 - R^2
    try:
        inv_Ftt = np.linalg.inv(Ftt)
    except np.linalg.LinAlgError:
        inv_Ftt = np.linalg.pinv(Ftt)

    schur_eps = float(Fee - (Fet @ inv_Ftt @ Fte))
    # numerical clipping (Schur should be >=0 in PSD matrices)
    schur_eps = max(0.0, schur_eps)
    r2_eps = float(1.0 - schur_eps)

    passed = (rank_norm >= rank_thr) and (kappa_theta < kappa_thr) and (schur_eps >= schur_min)

    return GateResult(
        name="G4_identifiability",
        passed=passed,
        metrics={
            "effective_rank_fisher_raw": float(res.effective_rank_fisher),
            "condition_fisher_raw": float(res.condition_fisher),

            "effective_rank_fisher_norm": float(rank_norm),

            "kappa_theta_norm": float(kappa_theta),
            "schur_eps": float(schur_eps),
            "r2_eps": float(r2_eps),

            "effective_rank_hessian": float(res.effective_rank_hessian),
            "condition_hessian": float(res.condition_hessian),
        },
        thresholds={
            "rank_thr": float(rank_thr),
            "kappa_thr_theta": float(kappa_thr),
            "schur_min": float(schur_min),
        },
        notes="Identifiability via ε-direction Schur complement (1-R^2) + θ-block conditioning.",
    )


# Placeholders (prepared for later Phase I+)
def gate_G5_stability_placeholder() -> GateResult:
    return GateResult("G5_stability", False, {}, {}, notes="Placeholder (Phase I+).")


def gate_G6_adversarial_placeholder() -> GateResult:
    return GateResult("G6_adversarial", False, {}, {}, notes="Placeholder (Phase I+).")


def gate_G7_oos_placeholder() -> GateResult:
    return GateResult("G7_out_of_sample", False, {}, {}, notes="Placeholder (Phase I+).")