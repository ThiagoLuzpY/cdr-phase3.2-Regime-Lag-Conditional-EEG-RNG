from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_1_config import StateModelSpec, load_phase3_1_config

from src.phase3_1_loader import load_phase3_1_dataset
from src.phase3_1_features import (
    FeatureLayerResult,
    fit_state_components,
    prepare_phase3_1_features_for_modeling,
)
from src.controls_phase3_1 import (
    generate_proxy_ablation_frames,
    run_phase3_1_controls,
)
from src.phase3_1_metrics import (
    InjectionCurveSummary,
    ModelScore,
    Phase3_1GateResult,
    build_phase3_1_summary_text,
    classify_phase3_1_outcome,
    compact_model_score_dict,
    gate_F6_incremental_joint_lift,
    gate_F7_subject_generalization,
    gate_F8_complexity_penalty,
    gate_F9_proxy_ablation_stability,
    get_final_and_baseline_scores,
    make_model_score,
    save_json,
    save_summary_text,
    summarize_gates,
    summarize_injection_curve,
)


# =========================================================
# Mixed-radix state encoding
# =========================================================

@dataclass(frozen=True)
class MixedStateEncoding:
    """
    Mixed-radix encoder for state models with per-component bin counts.

    Example:
        components bins = (3, 2, 3)
        total states = 18
    """

    bins: Tuple[int, ...]
    multipliers: Tuple[int, ...]
    n_states: int

    @classmethod
    def from_bins(cls, bins: Sequence[int]) -> "MixedStateEncoding":
        bins = tuple(int(b) for b in bins)

        if any(b < 2 for b in bins):
            raise ValueError("all bins must be >= 2")

        multipliers: List[int] = []
        current = 1

        for b in reversed(bins):
            multipliers.insert(0, current)
            current *= b

        return cls(
            bins=bins,
            multipliers=tuple(multipliers),
            n_states=int(current),
        )

    def encode(self, components: np.ndarray) -> np.ndarray:
        comps = np.asarray(components, dtype=int)

        if comps.ndim != 2:
            raise ValueError("components must be a 2D matrix")

        if comps.shape[1] != len(self.bins):
            raise ValueError(
                f"component width mismatch: got {comps.shape[1]}, expected {len(self.bins)}"
            )

        out = np.zeros(comps.shape[0], dtype=int)

        for j, (b, m) in enumerate(zip(self.bins, self.multipliers)):
            values = comps[:, j]

            if np.any(values < 0) or np.any(values >= b):
                raise ValueError(
                    f"component {j} out of range for {b} bins: "
                    f"min={values.min()}, max={values.max()}"
                )

            out += values * m

        return out.astype(int)

    def decode_state(self, state_id: int) -> Tuple[int, ...]:
        x = int(state_id)
        values: List[int] = []

        for b, m in zip(self.bins, self.multipliers):
            value = x // m
            values.append(int(value))
            x = x % m

        return tuple(values)

    def decode_all(self) -> np.ndarray:
        rows = [self.decode_state(i) for i in range(self.n_states)]
        return np.asarray(rows, dtype=int)


# =========================================================
# Empirical kernel and CDR epsilon estimation
# =========================================================

@dataclass
class LocalEmpiricalKernel:
    """
    Lightweight empirical transition kernel used by the Phase III.1 runner.

    P0 is estimated from train transitions with Dirichlet smoothing.
    Delta-chi is computed row-wise from P0 against product-of-component
    marginals over the decoded next-state components.
    """

    probs: np.ndarray
    encoding: MixedStateEncoding
    alpha: float
    delta_chi: np.ndarray

    @classmethod
    def from_transitions(
        cls,
        curr: np.ndarray,
        nxt: np.ndarray,
        encoding: MixedStateEncoding,
        alpha: float,
    ) -> "LocalEmpiricalKernel":
        curr = np.asarray(curr, dtype=int).reshape(-1)
        nxt = np.asarray(nxt, dtype=int).reshape(-1)

        if len(curr) != len(nxt):
            raise ValueError("curr and nxt must have the same length")

        n_states = int(encoding.n_states)
        counts = np.full((n_states, n_states), float(alpha), dtype=float)

        for c, n in zip(curr, nxt):
            if 0 <= c < n_states and 0 <= n < n_states:
                counts[c, n] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        probs = counts / np.maximum(row_sums, 1e-12)

        delta_chi = _compute_delta_chi_matrix(probs, encoding)

        return cls(
            probs=probs,
            encoding=encoding,
            alpha=float(alpha),
            delta_chi=delta_chi,
        )

    @property
    def n_states(self) -> int:
        return int(self.probs.shape[0])

    def row_probs(self, state_id: int, eps: float, min_prob: float) -> np.ndarray:
        state_id = int(state_id)
        base = self.probs[state_id].astype(float)
        delta = self.delta_chi[state_id].astype(float)

        weights = base * np.exp(float(eps) * delta)
        weights = np.maximum(weights, float(min_prob))

        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones_like(weights) / len(weights)

        return weights / total


def _compute_delta_chi_matrix(
    probs: np.ndarray,
    encoding: MixedStateEncoding,
) -> np.ndarray:
    n_states = encoding.n_states
    decoded = encoding.decode_all()

    delta = np.zeros((n_states, n_states), dtype=float)

    for s in range(n_states):
        row = probs[s]
        component_marginals: List[np.ndarray] = []

        for comp_idx, n_bins in enumerate(encoding.bins):
            marg = np.zeros(n_bins, dtype=float)

            for next_state in range(n_states):
                value = decoded[next_state, comp_idx]
                marg[value] += row[next_state]

            marg = np.maximum(marg, 1e-12)
            component_marginals.append(marg)

        for next_state in range(n_states):
            values = decoded[next_state]
            product = 1.0

            for comp_idx, value in enumerate(values):
                product *= component_marginals[comp_idx][value]

            delta[s, next_state] = math.log(
                max(row[next_state], 1e-12) / max(product, 1e-12)
            )

    return delta


def estimate_epsilon_grid(
    curr: np.ndarray,
    nxt: np.ndarray,
    kernel: LocalEmpiricalKernel,
    eps_grid: np.ndarray,
    min_prob: float,
    label: str = "",
    progress_every: int = 10,
    verbose: int = 0,
) -> Tuple[float, float, Dict[str, float]]:
    curr = np.asarray(curr, dtype=int).reshape(-1)
    nxt = np.asarray(nxt, dtype=int).reshape(-1)

    if len(curr) != len(nxt):
        raise ValueError("curr and nxt must have the same length")

    if len(curr) == 0:
        raise RuntimeError(f"No transitions available for epsilon estimation: {label}")

    ll_by_eps: Dict[str, float] = {}

    best_eps = float(eps_grid[0])
    best_ll = -float("inf")

    for i, eps in enumerate(eps_grid):
        eps = float(eps)
        ll = 0.0

        for c, n in zip(curr, nxt):
            probs = kernel.row_probs(int(c), eps=eps, min_prob=min_prob)
            p = max(float(probs[int(n)]), float(min_prob))
            ll += math.log(p)

        ll_by_eps[f"{eps:.4f}"] = float(ll)

        if ll > best_ll:
            best_ll = float(ll)
            best_eps = float(eps)

        if verbose and progress_every and (i % progress_every == 0):
            print(f"[eps-grid:{label}] eps={eps:.4f} ll={ll:.4f}")

    return best_eps, best_ll, ll_by_eps


def simulate_trajectory(
    kernel: LocalEmpiricalKernel,
    eps: float,
    n_steps: int,
    seed: int,
    min_prob: float,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))

    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")

    traj = np.zeros(n_steps + 1, dtype=int)
    traj[0] = int(rng.integers(0, kernel.n_states))

    for t in range(n_steps):
        probs = kernel.row_probs(int(traj[t]), eps=float(eps), min_prob=min_prob)
        traj[t + 1] = int(rng.choice(kernel.n_states, p=probs))

    return traj


# =========================================================
# Split helpers
# =========================================================

def make_chronological_split(n: int, train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    split = int(n * float(train_ratio))

    if split < 2 or split >= n - 1:
        raise RuntimeError(f"Invalid chronological split: split={split}, n={n}")

    return (
        np.arange(0, split, dtype=int),
        np.arange(split, n, dtype=int),
    )


def make_interleaved_split(n: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=int)
    train = idx[idx % 2 == 0]
    test = idx[idx % 2 == 1]

    if len(train) < 2 or len(test) < 2:
        raise RuntimeError("Interleaved split too small")

    return train, test


def make_loso_primary_split(
    df: pd.DataFrame,
    train_ratio: float,
    min_subjects: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if "subject_id" not in df.columns:
        train, test = make_chronological_split(len(df), train_ratio)
        return train, test, "chronological_no_subject_id"

    subjects = sorted([str(x) for x in df["subject_id"].dropna().unique()])

    if len(subjects) < int(min_subjects):
        train, test = make_chronological_split(len(df), train_ratio)
        return train, test, "chronological_insufficient_subjects"

    heldout = subjects[-1]

    train_mask = df["subject_id"].astype(str) != heldout
    test_mask = df["subject_id"].astype(str) == heldout

    train = np.flatnonzero(train_mask.to_numpy())
    test = np.flatnonzero(test_mask.to_numpy())

    if len(train) < 2 or len(test) < 2:
        train, test = make_chronological_split(len(df), train_ratio)
        return train, test, "chronological_loso_too_small"

    return train.astype(int), test.astype(int), f"leave_one_subject_out:{heldout}"


def transition_indices_from_row_split(
    n_rows: int,
    idx_train_rows: np.ndarray,
    idx_test_rows: np.ndarray,
    lag: int,
    df: Optional[pd.DataFrame] = None,
    group_cols: Tuple[str, ...] = ("subject_id", "recording_id"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds transition indices while preventing artificial transitions
    across subject/recording boundaries.

    Without this protection, the runner may incorrectly treat:

        last epoch of SC4001 -> first epoch of SC4002

    as a valid transition. That is not physically or methodologically valid.
    """
    if n_rows < lag + 2:
        raise RuntimeError(f"Too few rows for transitions: n_rows={n_rows}, lag={lag}")

    train_mask = np.zeros(n_rows, dtype=bool)
    test_mask = np.zeros(n_rows, dtype=bool)

    train_mask[np.asarray(idx_train_rows, dtype=int)] = True
    test_mask[np.asarray(idx_test_rows, dtype=int)] = True

    existing_group_cols: Tuple[str, ...] = tuple()

    if df is not None:
        existing_group_cols = tuple(c for c in group_cols if c in df.columns)

    train_t: List[int] = []
    test_t: List[int] = []

    for k in range(n_rows - lag):
        same_segment = True

        if df is not None and existing_group_cols:
            for col in existing_group_cols:
                if str(df.iloc[k][col]) != str(df.iloc[k + lag][col]):
                    same_segment = False
                    break

        if not same_segment:
            continue

        if train_mask[k] and train_mask[k + lag]:
            train_t.append(k)
        elif test_mask[k] and test_mask[k + lag]:
            test_t.append(k)

    train_trans = np.asarray(train_t, dtype=int)
    test_trans = np.asarray(test_t, dtype=int)

    if len(train_trans) < 2 or len(test_trans) < 2:
        raise RuntimeError(
            f"Transition split too small after segment filtering: "
            f"train={len(train_trans)}, test={len(test_trans)}"
        )

    return train_trans, test_trans


# =========================================================
# Gates F1-F5 local wrappers
# =========================================================

def gate_F1_injection_recovery(
    eps_hat: float,
    eps_true: float,
    tol_abs: float,
) -> Phase3_1GateResult:
    err = abs(float(eps_hat) - float(eps_true))
    passed = bool(err <= float(tol_abs))

    return Phase3_1GateResult(
        name="F1_injection_recovery",
        passed=passed,
        metrics={
            "eps_hat": float(eps_hat),
            "eps_true": float(eps_true),
            "abs_err": float(err),
            "tol_abs": float(tol_abs),
        },
        interpretation=(
            "Injected structure was recovered within tolerance."
            if passed
            else "Injected structure was not recovered within tolerance."
        ),
    )


def gate_F2_controls_collapse(
    control_summary: Mapping[str, Any],
) -> Phase3_1GateResult:
    passed = bool(control_summary.get("passed", False))

    return Phase3_1GateResult(
        name="F2_controls_collapse",
        passed=passed,
        metrics=dict(control_summary),
        interpretation=(
            "Controls collapsed below tolerance at the required fraction."
            if passed
            else "Controls did not collapse below tolerance at the required fraction."
        ),
    )


def gate_F3_holdout_generalization(
    eps_train: float,
    eps_test: float,
    max_delta: float,
) -> Phase3_1GateResult:
    delta = abs(float(eps_train) - float(eps_test))
    passed = bool(delta <= float(max_delta))

    return Phase3_1GateResult(
        name="F3_holdout_generalization",
        passed=passed,
        metrics={
            "eps_train": float(eps_train),
            "eps_test": float(eps_test),
            "abs_delta": float(delta),
            "max_delta": float(max_delta),
        },
        interpretation=(
            "Train/test epsilon estimates remain within holdout tolerance."
            if passed
            else "Train/test epsilon estimates diverge beyond holdout tolerance."
        ),
    )


def gate_F5_sensitivity(
    eps_primary: float,
    eps_sensitivity: float,
    max_delta: float,
) -> Phase3_1GateResult:
    delta = abs(float(eps_primary) - float(eps_sensitivity))
    passed = bool(delta <= float(max_delta))

    return Phase3_1GateResult(
        name="F5_sensitivity",
        passed=passed,
        metrics={
            "eps_primary": float(eps_primary),
            "eps_sensitivity": float(eps_sensitivity),
            "abs_delta": float(delta),
            "max_delta": float(max_delta),
        },
        interpretation=(
            "Epsilon remains stable under sensitivity discretization."
            if passed
            else "Epsilon changes beyond tolerance under sensitivity discretization."
        ),
    )


# =========================================================
# Model evaluation
# =========================================================

@dataclass
class EvaluatedModel:
    model: StateModelSpec
    encoding: MixedStateEncoding
    kernel: LocalEmpiricalKernel
    ids: np.ndarray
    curr_all: np.ndarray
    nxt_all: np.ndarray
    curr_train: np.ndarray
    nxt_train: np.ndarray
    curr_test: np.ndarray
    nxt_test: np.ndarray
    eps_train: float
    eps_test: float
    ll_train: float
    ll_test: float
    ll_grid_train: Dict[str, float]
    ll_grid_test: Dict[str, float]
    score: ModelScore
    component_metadata: Dict[str, Any]


def build_transitions(ids: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=int).reshape(-1)

    if len(ids) <= lag:
        raise RuntimeError("Too few ids to build transitions")

    return ids[:-lag], ids[lag:]


def evaluate_state_model(
    df: pd.DataFrame,
    model: StateModelSpec,
    cfg: Any,
    idx_train_rows: np.ndarray,
    idx_test_rows: np.ndarray,
    label: str,
    sensitivity: bool = False,
    verbose: Optional[int] = None,
) -> EvaluatedModel:
    verbose = int(cfg.verbose if verbose is None else verbose)

    comp_result = fit_state_components(
        df=df,
        model=model,
        cfg=cfg,
        train_index=idx_train_rows,
        sensitivity=sensitivity,
    )

    if (
        comp_result.transitions_per_state < float(cfg.min_transitions_per_state)
        and not cfg.skip_models_failing_density
    ):
        raise RuntimeError(
            f"Model {model.name} fails density rule: "
            f"{comp_result.transitions_per_state:.4f} transitions/state"
        )

    encoding = MixedStateEncoding.from_bins(model.bins)
    ids = encoding.encode(comp_result.components_df.to_numpy(dtype=int))

    curr_all, nxt_all = build_transitions(ids, lag=int(cfg.lag))

    idx_train_trans, idx_test_trans = transition_indices_from_row_split(
        n_rows=len(df),
        idx_train_rows=idx_train_rows,
        idx_test_rows=idx_test_rows,
        lag=int(cfg.lag),
        df=df,
        group_cols=("subject_id", "recording_id"),
    )

    curr_train = curr_all[idx_train_trans]
    nxt_train = nxt_all[idx_train_trans]

    curr_test = curr_all[idx_test_trans]
    nxt_test = nxt_all[idx_test_trans]

    kernel = LocalEmpiricalKernel.from_transitions(
        curr=curr_train,
        nxt=nxt_train,
        encoding=encoding,
        alpha=float(cfg.dirichlet_alpha),
    )

    eps_grid = np.asarray(cfg.eps_grid, dtype=float)

    eps_train, ll_train, ll_grid_train = estimate_epsilon_grid(
        curr=curr_train,
        nxt=nxt_train,
        kernel=kernel,
        eps_grid=eps_grid,
        min_prob=float(cfg.min_prob),
        label=f"{label}_train",
        progress_every=int(cfg.progress_every),
        verbose=0,
    )

    eps_test, ll_test, ll_grid_test = estimate_epsilon_grid(
        curr=curr_test,
        nxt=nxt_test,
        kernel=kernel,
        eps_grid=eps_grid,
        min_prob=float(cfg.min_prob),
        label=f"{label}_test",
        progress_every=int(cfg.progress_every),
        verbose=0,
    )

    score = make_model_score(
        model_name=model.name,
        label=model.label,
        n_states=model.n_states,
        n_components=model.n_components,
        n_train=len(curr_train),
        n_test=len(curr_test),
        eps_train=eps_train,
        eps_test=eps_test,
        ll_train=ll_train,
        ll_test=ll_test,
        transitions_per_state=comp_result.transitions_per_state,
        transitions_per_observed_state=comp_result.transitions_per_observed_state,
        complexity_mode="transition_kernel",
    )

    component_metadata = {
        "model": model.name,
        "label": model.label,
        "components": list(model.components),
        "bins": list(model.bins),
        "n_states": int(model.n_states),
        "observed_states": int(comp_result.observed_states),
        "transitions_per_state": float(comp_result.transitions_per_state),
        "transitions_per_observed_state": float(comp_result.transitions_per_observed_state),
        "sensitivity": bool(sensitivity),
        "component_specs": [asdict(s) for s in comp_result.specs],
    }

    if verbose:
        print(
            f"[Phase3.1] {label} | {model.name} | "
            f"eps_train={eps_train:.4f} eps_test={eps_test:.4f} "
            f"states={model.n_states}"
        )

    return EvaluatedModel(
        model=model,
        encoding=encoding,
        kernel=kernel,
        ids=ids,
        curr_all=curr_all,
        nxt_all=nxt_all,
        curr_train=curr_train,
        nxt_train=nxt_train,
        curr_test=curr_test,
        nxt_test=nxt_test,
        eps_train=eps_train,
        eps_test=eps_test,
        ll_train=ll_train,
        ll_test=ll_test,
        ll_grid_train=ll_grid_train,
        ll_grid_test=ll_grid_test,
        score=score,
        component_metadata=component_metadata,
    )


def get_model(cfg: Any, name: str) -> StateModelSpec:
    for model in cfg.state_models:
        if model.name == name:
            return model

    raise KeyError(f"State model not found: {name}")


def active_models(cfg: Any) -> Tuple[StateModelSpec, ...]:
    return tuple(m for m in cfg.state_models if m.active)


def eeg_baseline_model() -> StateModelSpec:
    return StateModelSpec(
        name="B0_eeg_only",
        label="EEG-only baseline: delta_power x eeg_info_bin",
        components=("delta_power", "eeg_info_bin"),
        bins=(3, 3),
        active=True,
    )


def rng_baseline_model() -> StateModelSpec:
    return StateModelSpec(
        name="B1_rng_only",
        label="RNG-only baseline: rng_bit x rng_info_bin",
        components=("rng_bit", "rng_info_bin"),
        bins=(2, 3),
        active=True,
    )


# =========================================================
# Injection curve
# =========================================================

def compute_injection_curve_for_model(
    evaluated: EvaluatedModel,
    cfg: Any,
) -> Tuple[Dict[float, float], InjectionCurveSummary]:
    curve: Dict[float, float] = {}

    for eps_true in cfg.injection_eps_grid:
        sim_traj = simulate_trajectory(
            kernel=evaluated.kernel,
            eps=float(eps_true),
            n_steps=len(evaluated.curr_train),
            seed=int(cfg.random_seed) + int(round(float(eps_true) * 10000)),
            min_prob=float(cfg.min_prob),
        )

        sim_curr, sim_nxt = build_transitions(sim_traj, lag=1)

        eps_hat, _, _ = estimate_epsilon_grid(
            curr=sim_curr,
            nxt=sim_nxt,
            kernel=evaluated.kernel,
            eps_grid=np.asarray(cfg.eps_grid, dtype=float),
            min_prob=float(cfg.min_prob),
            label=f"injection_{eps_true:.4f}",
            progress_every=int(cfg.progress_every),
            verbose=0,
        )

        curve[float(eps_true)] = float(eps_hat)

    summary = summarize_injection_curve(
        curve=curve,
        principal_eps_true=float(cfg.inj_eps_true),
        tol_abs=float(cfg.gate_tol_abs),
    )

    return curve, summary


# =========================================================
# LOSO subject generalization
# =========================================================

def run_loso_subject_generalization(
    raw_df: pd.DataFrame,
    cfg: Any,
    final_model: StateModelSpec,
) -> List[Dict[str, Any]]:
    if "subject_id" not in raw_df.columns:
        return []

    subjects = sorted([str(x) for x in raw_df["subject_id"].dropna().unique()])

    if len(subjects) < int(cfg.min_subjects_for_loso):
        return []

    rows: List[Dict[str, Any]] = []

    for heldout in subjects:
        print(f"[Phase3.1] LOSO heldout subject: {heldout}")

        train_mask = raw_df["subject_id"].astype(str) != heldout
        test_mask = raw_df["subject_id"].astype(str) == heldout

        idx_train = np.flatnonzero(train_mask.to_numpy())
        idx_test = np.flatnonzero(test_mask.to_numpy())

        if len(idx_train) < 10 or len(idx_test) < 10:
            rows.append(
                {
                    "subject_id": heldout,
                    "valid": False,
                    "reason": "too_few_rows",
                    "train_rows": int(len(idx_train)),
                    "test_rows": int(len(idx_test)),
                }
            )
            continue

        try:
            layered = prepare_phase3_1_features_for_modeling(
                df=raw_df,
                cfg=cfg,
                train_index=idx_train,
            ).df

            joint_eval = evaluate_state_model(
                df=layered,
                model=final_model,
                cfg=cfg,
                idx_train_rows=idx_train,
                idx_test_rows=idx_test,
                label=f"loso_{heldout}_joint",
                verbose=0,
            )

            eeg_eval = evaluate_state_model(
                df=layered,
                model=eeg_baseline_model(),
                cfg=cfg,
                idx_train_rows=idx_train,
                idx_test_rows=idx_test,
                label=f"loso_{heldout}_eeg",
                verbose=0,
            )

            rng_eval = evaluate_state_model(
                df=layered,
                model=rng_baseline_model(),
                cfg=cfg,
                idx_train_rows=idx_train,
                idx_test_rows=idx_test,
                label=f"loso_{heldout}_rng",
                verbose=0,
            )

            rows.append(
                {
                    "subject_id": heldout,
                    "valid": True,
                    "train_rows": int(len(idx_train)),
                    "test_rows": int(len(idx_test)),
                    "eps_joint_train": float(joint_eval.eps_train),
                    "eps_joint_test": float(joint_eval.eps_test),
                    "eps_eeg_train": float(eeg_eval.eps_train),
                    "eps_eeg_test": float(eeg_eval.eps_test),
                    "eps_rng_train": float(rng_eval.eps_train),
                    "eps_rng_test": float(rng_eval.eps_test),
                    "joint_lift_over_eeg": float(joint_eval.eps_test - eeg_eval.eps_test),
                    "joint_lift_over_baseline": float(
                        joint_eval.eps_test - max(eeg_eval.eps_test, rng_eval.eps_test)
                    ),
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "subject_id": heldout,
                    "valid": False,
                    "reason": str(exc),
                    "train_rows": int(len(idx_train)),
                    "test_rows": int(len(idx_test)),
                }
            )

    return rows


# =========================================================
# Main runner
# =========================================================

def run_phase3_1() -> None:
    cfg = load_phase3_1_config()

    print("\n============================================================")
    print("CDR Phase III.1 — Advanced EEG + RNG Informational Coupling")
    print("============================================================\n")

    print("[Phase3.1] Project:", cfg.project_name)
    print("[Phase3.1] EEG raw dir:", cfg.eeg_raw_dir)
    print("[Phase3.1] RNG file:", cfg.rng_file)
    print("[Phase3.1] Max subjects:", cfg.max_subjects)
    print("[Phase3.1] Final model:", cfg.final_state_model)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print("\n[Phase3.1] Loading EEG + RNG dataset...")

    dataset = load_phase3_1_dataset(cfg)
    raw_df = dataset.df.copy().reset_index(drop=True)

    sort_cols = [c for c in ["subject_id", "recording_id", "epoch_idx"] if c in raw_df.columns]

    if sort_cols:
        raw_df = raw_df.sort_values(sort_cols).reset_index(drop=True)

    print(
        f"[Phase3.1] Loaded rows={len(raw_df)} | "
        f"subjects={dataset.metadata.get('n_subjects_loaded')}"
    )

    # -----------------------------------------------------
    # Primary split
    # -----------------------------------------------------

    idx_train, idx_test, split_label = make_loso_primary_split(
        df=raw_df,
        train_ratio=float(cfg.train_ratio),
        min_subjects=int(cfg.min_subjects_for_loso),
    )

    print(
        f"[Phase3.1] Primary split: {split_label} | "
        f"train_rows={len(idx_train)} test_rows={len(idx_test)}"
    )

    # -----------------------------------------------------
    # Feature layers I_t, Z_t, Q_t
    # -----------------------------------------------------

    print("\n[Phase3.1] Fitting Phase III.1 feature layers on train only...")

    layer_result: FeatureLayerResult = prepare_phase3_1_features_for_modeling(
        df=raw_df,
        cfg=cfg,
        train_index=idx_train,
    )

    df = layer_result.df.copy().reset_index(drop=True)

    if cfg.save_intermediate_features:
        cfg.interim_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.interim_dir / "phase3_1_layered_features.csv", index=False)

    # -----------------------------------------------------
    # Evaluate active M0-M5 models
    # -----------------------------------------------------

    print("\n[Phase3.1] Evaluating active state models M0-M5...")

    evaluations: Dict[str, EvaluatedModel] = {}
    skipped_models: Dict[str, str] = {}

    for model in active_models(cfg):
        try:
            evaluation = evaluate_state_model(
                df=df,
                model=model,
                cfg=cfg,
                idx_train_rows=idx_train,
                idx_test_rows=idx_test,
                label=model.name,
            )
            evaluations[model.name] = evaluation

        except Exception as exc:
            skipped_models[model.name] = str(exc)
            print(f"[Phase3.1] WARNING: skipped {model.name}: {exc}")

    if cfg.final_state_model not in evaluations:
        raise RuntimeError(
            f"Final model {cfg.final_state_model} was not evaluated. "
            f"Skipped models: {skipped_models}"
        )

    if cfg.baseline_state_model not in evaluations:
        raise RuntimeError(
            f"Baseline model {cfg.baseline_state_model} was not evaluated. "
            f"Skipped models: {skipped_models}"
        )

    final_eval = evaluations[cfg.final_state_model]
    baseline_eval = evaluations[cfg.baseline_state_model]

    model_scores: List[ModelScore] = [ev.score for ev in evaluations.values()]

    # -----------------------------------------------------
    # EEG-only and RNG-only baselines
    # -----------------------------------------------------

    print("\n[Phase3.1] Evaluating EEG-only and RNG-only baselines...")

    eeg_eval = evaluate_state_model(
        df=df,
        model=eeg_baseline_model(),
        cfg=cfg,
        idx_train_rows=idx_train,
        idx_test_rows=idx_test,
        label="eeg_only_baseline",
    )

    rng_eval = evaluate_state_model(
        df=df,
        model=rng_baseline_model(),
        cfg=cfg,
        idx_train_rows=idx_train,
        idx_test_rows=idx_test,
        label="rng_only_baseline",
    )

    # -----------------------------------------------------
    # F1 injection curve
    # -----------------------------------------------------

    print("\n[Phase3.1] Running F1 injection curve...")

    injection_curve, injection_summary = compute_injection_curve_for_model(
        evaluated=final_eval,
        cfg=cfg,
    )

    gate1 = gate_F1_injection_recovery(
        eps_hat=injection_summary.principal_eps_hat,
        eps_true=float(cfg.inj_eps_true),
        tol_abs=float(cfg.gate_tol_abs),
    )

    # -----------------------------------------------------
    # F2 controls
    # -----------------------------------------------------

    print("\n[Phase3.1] Running F2 controls collapse...")

    def estimate_control_eps(control_df: pd.DataFrame, control_name: str) -> float:
        control_eval = evaluate_state_model(
            df=control_df,
            model=final_eval.model,
            cfg=cfg,
            idx_train_rows=idx_train,
            idx_test_rows=idx_test,
            label=f"control_{control_name}",
            verbose=0,
        )
        return float(control_eval.eps_test)

    controls_result = run_phase3_1_controls(
        df=df,
        cfg=cfg,
        estimate_fn=estimate_control_eps,
        seed=int(cfg.random_seed),
    )

    gate2 = gate_F2_controls_collapse(controls_result.summary)

    # -----------------------------------------------------
    # F3 holdout generalization
    # -----------------------------------------------------

    print("\n[Phase3.1] Running F3 holdout generalization...")

    gate3 = gate_F3_holdout_generalization(
        eps_train=final_eval.eps_train,
        eps_test=final_eval.eps_test,
        max_delta=float(cfg.holdout_delta),
    )

    # -----------------------------------------------------
    # F5 sensitivity
    # -----------------------------------------------------

    print("\n[Phase3.1] Running F5 sensitivity...")

    sensitivity_eval = evaluate_state_model(
        df=df,
        model=final_eval.model,
        cfg=cfg,
        idx_train_rows=idx_train,
        idx_test_rows=idx_test,
        label=f"{final_eval.model.name}_sensitivity",
        sensitivity=True,
        verbose=0,
    )

    gate5 = gate_F5_sensitivity(
        eps_primary=final_eval.eps_test,
        eps_sensitivity=sensitivity_eval.eps_test,
        max_delta=float(cfg.sensitivity_delta),
    )

    # -----------------------------------------------------
    # F6 incremental joint lift
    # -----------------------------------------------------

    print("\n[Phase3.1] Running F6 incremental joint lift...")

    gate6 = gate_F6_incremental_joint_lift(
        eps_joint_test=final_eval.eps_test,
        eps_eeg_test=eeg_eval.eps_test,
        eps_rng_test=rng_eval.eps_test,
        joint_lift_min=float(cfg.joint_lift_min),
        strong_joint_eps_min=float(cfg.strong_joint_eps_min),
    )

    # -----------------------------------------------------
    # F7 subject generalization
    # -----------------------------------------------------

    print("\n[Phase3.1] Running F7 subject generalization...")

    subject_results = run_loso_subject_generalization(
        raw_df=raw_df,
        cfg=cfg,
        final_model=final_eval.model,
    )

    gate7 = gate_F7_subject_generalization(
        subject_results=subject_results,
        subject_effect_fraction=float(cfg.subject_effect_fraction),
    )

    # -----------------------------------------------------
    # F8 complexity penalty
    # -----------------------------------------------------

    print("\n[Phase3.1] Running F8 complexity penalty...")

    gate8 = gate_F8_complexity_penalty(
        baseline_score=baseline_eval.score,
        advanced_score=final_eval.score,
        metric=str(cfg.complexity_metric),
        min_test_ll_improvement=float(cfg.min_test_ll_improvement),
    )

    # -----------------------------------------------------
    # F9 proxy ablation
    # -----------------------------------------------------

    print("\n[Phase3.1] Running F9 proxy ablation stability...")

    ablation_frames = generate_proxy_ablation_frames(df, cfg)
    ablation_eps: Dict[str, float] = {}
    ablation_details: Dict[str, Any] = {}

    for ablation in ablation_frames:
        try:
            ablation_eval = evaluate_state_model(
                df=ablation.df,
                model=final_eval.model,
                cfg=cfg,
                idx_train_rows=idx_train,
                idx_test_rows=idx_test,
                label=f"ablation_{ablation.name}",
                verbose=0,
            )
            ablation_eps[ablation.name] = float(ablation_eval.eps_test)
            ablation_details[ablation.name] = {
                "eps_train": float(ablation_eval.eps_train),
                "eps_test": float(ablation_eval.eps_test),
                "metadata": ablation.metadata,
            }
        except Exception as exc:
            ablation_eps[ablation.name] = float(final_eval.eps_test)
            ablation_details[ablation.name] = {
                "error": str(exc),
                "metadata": ablation.metadata,
            }

    gate9 = gate_F9_proxy_ablation_stability(
        full_eps_joint=final_eval.eps_test,
        ablation_eps=ablation_eps,
        ablation_delta_min=float(cfg.ablation_delta_min),
        require_all=True,
    )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    gates = [gate1, gate2, gate3, gate5, gate6, gate7, gate8, gate9]
    gate_summary = summarize_gates(gates)

    final_status = classify_phase3_1_outcome(
        eps_joint_test=final_eval.eps_test,
        eps_eeg_test=eeg_eval.eps_test,
        eps_rng_test=rng_eval.eps_test,
        gates=gates,
        strong_joint_eps_min=float(cfg.strong_joint_eps_min),
        joint_lift_min=float(cfg.joint_lift_min),
    )

    # -----------------------------------------------------
    # Print report
    # -----------------------------------------------------

    print("\n============================================================")
    print("CDR Phase III.1 — Gates")
    print("============================================================")

    for gate in gates:
        print(f"{gate.name}: {'PASS' if gate.passed else 'FAIL'} | metrics={gate.metrics}")

    print("============================================================")
    print(f"FINAL STATUS: {final_status['status']}")
    print(f"INTERPRETATION: {final_status['interpretation']}")
    print("============================================================\n")

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    model_scores_payload = compact_model_score_dict(model_scores)

    results_payload: Dict[str, Any] = {
        "project_name": cfg.project_name,
        "phase_name": cfg.phase_name,
        "split_label": split_label,
        "dataset_metadata": dataset.metadata,
        "feature_layer_artifacts": {
            "has_eeg_info": layer_result.artifacts.eeg_info_spec is not None,
            "has_rng_info": layer_result.artifacts.rng_info_spec is not None,
            "has_q_rng": layer_result.artifacts.q_rng_spec is not None,
            "has_latent": layer_result.artifacts.latent_spec is not None,
        },
        "primary_results": {
            "final_model": cfg.final_state_model,
            "baseline_model": cfg.baseline_state_model,
            "eps_joint_train": float(final_eval.eps_train),
            "eps_joint_test": float(final_eval.eps_test),
            "eps_eeg_train": float(eeg_eval.eps_train),
            "eps_eeg_test": float(eeg_eval.eps_test),
            "eps_rng_train": float(rng_eval.eps_train),
            "eps_rng_test": float(rng_eval.eps_test),
            "joint_lift_over_eeg": float(final_eval.eps_test - eeg_eval.eps_test),
            "joint_lift_over_baseline": float(
                final_eval.eps_test - max(eeg_eval.eps_test, rng_eval.eps_test)
            ),
        },
        "model_scores": model_scores_payload,
        "baseline_scores": {
            "eeg_only": asdict(eeg_eval.score),
            "rng_only": asdict(rng_eval.score),
        },
        "component_metadata": {
            name: ev.component_metadata for name, ev in evaluations.items()
        },
        "skipped_models": skipped_models,
        "injection_curve": asdict(injection_summary),
        "controls": {
            "eps_controls": list(controls_result.eps_controls),
            "control_names": list(controls_result.control_names),
            "control_details": list(controls_result.control_details),
            "summary": controls_result.summary,
        },
        "sensitivity": {
            "eps_primary_test": float(final_eval.eps_test),
            "eps_sensitivity_test": float(sensitivity_eval.eps_test),
            "sensitivity_model_metadata": sensitivity_eval.component_metadata,
        },
        "subject_results": subject_results,
        "ablation": {
            "ablation_eps": ablation_eps,
            "ablation_details": ablation_details,
        },
        "gates": gate_summary,
        "final_status": final_status,
        "config_summary": {
            "max_subjects": int(cfg.max_subjects),
            "rng_use_bits": bool(cfg.rng_use_bits),
            "rng_window_size": int(cfg.rng_window_size),
            "holdout_mode": str(cfg.holdout_mode),
            "inj_eps_true": float(cfg.inj_eps_true),
            "injection_eps_grid": [float(x) for x in cfg.injection_eps_grid],
            "joint_lift_min": float(cfg.joint_lift_min),
            "strong_joint_eps_min": float(cfg.strong_joint_eps_min),
            "state_models": [
                {
                    "name": m.name,
                    "label": m.label,
                    "components": list(m.components),
                    "bins": list(m.bins),
                    "n_states": int(m.n_states),
                    "active": bool(m.active),
                }
                for m in cfg.state_models
            ],
        },
    }

    save_json(cfg.results_dir / "phase3_1_results.json", results_payload)
    save_json(cfg.results_dir / "phase3_1_model_comparison.json", model_scores_payload)
    save_json(cfg.results_dir / "phase3_1_controls.json", results_payload["controls"])
    save_json(cfg.results_dir / "phase3_1_subject_results.json", subject_results)
    save_json(cfg.results_dir / "phase3_1_injection_curve.json", asdict(injection_summary))
    save_json(cfg.results_dir / "phase3_1_ablation.json", results_payload["ablation"])

    summary_text = build_phase3_1_summary_text(
        title="CDR Phase III.1 — Advanced EEG + RNG Informational Coupling",
        model_scores=model_scores,
        gates=gates,
        final_status=final_status,
        injection_summary=injection_summary,
    )

    save_summary_text(cfg.results_dir / "phase3_1_summary.txt", summary_text)

    print(f"[Phase3.1] Results saved to: {cfg.results_dir}")


# =========================================================
# Entrypoint
# =========================================================

if __name__ == "__main__":
    run_phase3_1()