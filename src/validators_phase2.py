from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    notes: str = ""


def gate_F1_injection_recovery(eps_hat: float, eps_true: float, tol_abs: float) -> GateResult:
    passed = abs(eps_hat - eps_true) <= tol_abs
    return GateResult(
        name="F1_injection_recovery",
        passed=passed,
        metrics={"eps_hat": float(eps_hat), "eps_true": float(eps_true), "abs_err": float(abs(eps_hat - eps_true))},
        thresholds={"tol_abs": float(tol_abs)},
        notes="Injected reweighting must be recovered within tolerance.",
    )


def gate_F2_controls_collapse(eps_controls: Sequence[float], tol: float, required_fraction: float) -> GateResult:
    eps = np.array(list(eps_controls), dtype=float)
    if eps.size == 0:
        return GateResult(
            name="F2_controls_collapse",
            passed=False,
            metrics={"n": 0.0},
            thresholds={"tol": float(tol), "required_fraction": float(required_fraction)},
            notes="No control eps provided.",
        )
    med = float(np.median(eps))
    frac = float(np.mean(eps <= tol))
    passed = (med <= tol) and (frac >= required_fraction)
    return GateResult(
        name="F2_controls_collapse",
        passed=passed,
        metrics={
            "median_eps_controls": med,
            "fraction_below_tol": frac,
            "max_eps_controls": float(np.max(eps)),
            "n": float(eps.size),
        },
        thresholds={"tol": float(tol), "required_fraction": float(required_fraction)},
        notes="Controls must collapse robustly (median & majority).",
    )


def gate_F3_holdout_generalization(eps_train: float, eps_test: float, max_delta: float) -> GateResult:
    delta = abs(float(eps_train) - float(eps_test))
    passed = delta <= max_delta
    return GateResult(
        name="F3_holdout_generalization",
        passed=passed,
        metrics={"eps_train": float(eps_train), "eps_test": float(eps_test), "abs_delta": float(delta)},
        thresholds={"max_delta": float(max_delta)},
        notes="Epsilon estimate should generalize to held-out time window.",
    )


def gate_F5_sensitivity(eps_binsA: float, eps_binsB: float, max_delta: float) -> GateResult:
    delta = abs(float(eps_binsA) - float(eps_binsB))
    passed = delta <= max_delta
    return GateResult(
        name="F5_sensitivity",
        passed=passed,
        metrics={"eps_binsA": float(eps_binsA), "eps_binsB": float(eps_binsB), "abs_delta": float(delta)},
        thresholds={"max_delta": float(max_delta)},
        notes="Estimate should be stable under discretization changes.",
    )


def summarize(results: List[GateResult]) -> Dict[str, object]:
    passed_all = all(r.passed for r in results)
    return {
        "passed_all": bool(passed_all),
        "results": [r.__dict__ for r in results],
    }