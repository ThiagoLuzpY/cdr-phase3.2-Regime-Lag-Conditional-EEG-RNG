from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Generic helpers
# =========================================================

_EPS = 1e-12


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return default

    return float(np.mean(arr))


def safe_median(values: Sequence[float], default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return default

    return float(np.median(arr))


def safe_std(values: Sequence[float], default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        return default

    return float(np.std(arr, ddof=0))


def as_jsonable(obj: Any) -> Any:
    """
    Converts common numpy / dataclass objects into JSON-safe structures.
    """
    if dataclass_is_instance(obj):
        return asdict(obj)

    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    return obj


def dataclass_is_instance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


def save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(as_jsonable(payload), f, indent=4, ensure_ascii=False)


# =========================================================
# Gate containers
# =========================================================

@dataclass(frozen=True)
class Phase3_1GateResult:
    """
    Generic gate result for Phase III.1.

    This intentionally mirrors the style used in previous CDR phases,
    while allowing richer metadata for advanced gates F6-F9.
    """

    name: str
    passed: bool
    metrics: Dict[str, Any]
    interpretation: str = ""


@dataclass(frozen=True)
class ModelScore:
    """
    Stores model-level scores used for comparison and complexity penalty.
    """

    model: str
    label: str
    n_states: int
    n_components: int
    n_train: int
    n_test: int
    eps_train: float
    eps_test: float
    ll_train: float
    ll_test: float
    n_params: int
    bic_train: float
    bic_test: float
    aic_train: float
    aic_test: float
    transitions_per_state: float
    transitions_per_observed_state: float


@dataclass(frozen=True)
class SubjectEffectSummary:
    """
    Summary of leave-one-subject-out joint lift behavior.
    """

    n_subjects: int
    n_valid_subjects: int
    fraction_positive_lift: float
    median_joint_eps: float
    median_eeg_eps: float
    median_rng_eps: float
    median_lift_over_eeg: float
    median_lift_over_baseline: float
    passed_directional_consistency: bool


@dataclass(frozen=True)
class InjectionCurveSummary:
    """
    Stores injectability behavior across several injected eps values.
    """

    curve: Dict[str, float]
    errors: Dict[str, float]
    recovered_fraction: float
    injectability_class: str
    principal_eps_true: float
    principal_eps_hat: float
    principal_abs_err: float
    principal_passed: bool


# =========================================================
# Complexity / model comparison
# =========================================================

def estimate_markov_parameter_count(
    n_states: int,
    mode: str = "transition_kernel",
) -> int:
    """
    Estimates model complexity for information-criterion comparison.

    The CDR estimator itself estimates epsilon over a fixed grid, but the
    underlying empirical transition representation becomes more flexible as
    the state space grows. This function intentionally penalizes larger state
    spaces.

    Modes:
        - "transition_kernel": n_states * (n_states - 1) + 1
        - "state_space": n_states + 1
        - "epsilon_only": 1
    """
    n_states = int(max(n_states, 1))

    if mode == "transition_kernel":
        return int(n_states * max(n_states - 1, 1) + 1)

    if mode == "state_space":
        return int(n_states + 1)

    if mode == "epsilon_only":
        return 1

    raise ValueError(f"Unsupported parameter-count mode: {mode}")


def bic(log_likelihood: float, n_obs: int, n_params: int) -> float:
    ll = safe_float(log_likelihood, default=-np.inf)
    n = max(int(n_obs), 1)
    k = max(int(n_params), 1)

    if not np.isfinite(ll):
        return float("inf")

    return float(k * math.log(n) - 2.0 * ll)


def aic(log_likelihood: float, n_params: int) -> float:
    ll = safe_float(log_likelihood, default=-np.inf)
    k = max(int(n_params), 1)

    if not np.isfinite(ll):
        return float("inf")

    return float(2.0 * k - 2.0 * ll)


def make_model_score(
    model_name: str,
    label: str,
    n_states: int,
    n_components: int,
    n_train: int,
    n_test: int,
    eps_train: float,
    eps_test: float,
    ll_train: float,
    ll_test: float,
    transitions_per_state: float,
    transitions_per_observed_state: float,
    complexity_mode: str = "transition_kernel",
) -> ModelScore:
    n_params = estimate_markov_parameter_count(
        n_states=n_states,
        mode=complexity_mode,
    )

    return ModelScore(
        model=str(model_name),
        label=str(label),
        n_states=int(n_states),
        n_components=int(n_components),
        n_train=int(n_train),
        n_test=int(n_test),
        eps_train=float(eps_train),
        eps_test=float(eps_test),
        ll_train=float(ll_train),
        ll_test=float(ll_test),
        n_params=int(n_params),
        bic_train=bic(ll_train, n_train, n_params),
        bic_test=bic(ll_test, n_test, n_params),
        aic_train=aic(ll_train, n_params),
        aic_test=aic(ll_test, n_params),
        transitions_per_state=float(transitions_per_state),
        transitions_per_observed_state=float(transitions_per_observed_state),
    )


def model_scores_to_dataframe(scores: Sequence[ModelScore]) -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in scores])


def select_best_model_by_metric(
    scores: Sequence[ModelScore],
    metric: str = "bic_test",
) -> Optional[ModelScore]:
    if not scores:
        return None

    valid = [s for s in scores if np.isfinite(safe_float(getattr(s, metric)))]

    if not valid:
        return None

    if metric in {"bic_train", "bic_test", "aic_train", "aic_test"}:
        return min(valid, key=lambda s: safe_float(getattr(s, metric)))

    return max(valid, key=lambda s: safe_float(getattr(s, metric)))


# =========================================================
# F6 — Incremental Joint Lift
# =========================================================

def gate_F6_incremental_joint_lift(
    eps_joint_test: float,
    eps_eeg_test: float,
    eps_rng_test: float,
    joint_lift_min: float,
    strong_joint_eps_min: float,
) -> Phase3_1GateResult:
    """
    F6 checks whether the joint model adds detectable structure beyond
    the best single-domain baseline.

    Required:
        eps_joint_test - max(eps_eeg_test, eps_rng_test) >= joint_lift_min

    Strong interpretation additionally records whether:
        eps_joint_test >= strong_joint_eps_min
    """
    eps_joint_test = safe_float(eps_joint_test)
    eps_eeg_test = safe_float(eps_eeg_test)
    eps_rng_test = safe_float(eps_rng_test)

    baseline = max(eps_eeg_test, eps_rng_test)
    lift = eps_joint_test - baseline

    passed = bool(lift >= float(joint_lift_min))
    strong = bool(eps_joint_test >= float(strong_joint_eps_min))

    if passed and strong:
        interpretation = (
            "Joint model exceeds single-domain baselines and reaches the "
            "strong joint epsilon threshold."
        )
    elif passed:
        interpretation = (
            "Joint model exceeds single-domain baselines, but does not reach "
            "the strong joint epsilon threshold."
        )
    else:
        interpretation = (
            "Joint model does not exceed the strongest single-domain baseline "
            "by the required margin."
        )

    return Phase3_1GateResult(
        name="F6_incremental_joint_lift",
        passed=passed,
        metrics={
            "eps_joint_test": eps_joint_test,
            "eps_eeg_test": eps_eeg_test,
            "eps_rng_test": eps_rng_test,
            "baseline_max": baseline,
            "joint_lift": lift,
            "joint_lift_min": float(joint_lift_min),
            "strong_joint_eps_min": float(strong_joint_eps_min),
            "strong_joint_threshold_reached": strong,
        },
        interpretation=interpretation,
    )


# =========================================================
# F7 — Subject Generalization
# =========================================================

def summarize_subject_effects(
    subject_results: Sequence[Mapping[str, Any]],
    subject_effect_fraction: float,
) -> SubjectEffectSummary:
    """
    Summarizes leave-one-subject-out joint lift.

    Expected subject_result keys:
        - subject_id
        - eps_joint_test
        - eps_eeg_test
        - eps_rng_test

    Optional:
        - valid
    """
    valid_rows: List[Mapping[str, Any]] = []

    for row in subject_results:
        if not row:
            continue

        if row.get("valid", True) is False:
            continue

        if "eps_joint_test" not in row:
            continue

        valid_rows.append(row)

    n_subjects = len(subject_results)
    n_valid = len(valid_rows)

    if n_valid == 0:
        return SubjectEffectSummary(
            n_subjects=n_subjects,
            n_valid_subjects=0,
            fraction_positive_lift=0.0,
            median_joint_eps=0.0,
            median_eeg_eps=0.0,
            median_rng_eps=0.0,
            median_lift_over_eeg=0.0,
            median_lift_over_baseline=0.0,
            passed_directional_consistency=False,
        )

    joint = np.array([safe_float(r.get("eps_joint_test")) for r in valid_rows])
    eeg = np.array([safe_float(r.get("eps_eeg_test")) for r in valid_rows])
    rng = np.array([safe_float(r.get("eps_rng_test")) for r in valid_rows])

    baseline = np.maximum(eeg, rng)
    lift_over_eeg = joint - eeg
    lift_over_baseline = joint - baseline

    positive = lift_over_baseline > 0.0
    fraction_positive = float(np.mean(positive))

    passed = bool(fraction_positive >= float(subject_effect_fraction))

    return SubjectEffectSummary(
        n_subjects=int(n_subjects),
        n_valid_subjects=int(n_valid),
        fraction_positive_lift=fraction_positive,
        median_joint_eps=float(np.median(joint)),
        median_eeg_eps=float(np.median(eeg)),
        median_rng_eps=float(np.median(rng)),
        median_lift_over_eeg=float(np.median(lift_over_eeg)),
        median_lift_over_baseline=float(np.median(lift_over_baseline)),
        passed_directional_consistency=passed,
    )


def gate_F7_subject_generalization(
    subject_results: Sequence[Mapping[str, Any]],
    subject_effect_fraction: float,
) -> Phase3_1GateResult:
    """
    F7 checks whether the joint direction appears across held-out subjects.
    """
    summary = summarize_subject_effects(
        subject_results=subject_results,
        subject_effect_fraction=subject_effect_fraction,
    )

    passed = bool(summary.passed_directional_consistency)

    if passed:
        interpretation = (
            "The joint effect direction is consistent across the required "
            "fraction of held-out subjects."
        )
    else:
        interpretation = (
            "The joint effect direction is not consistent across the required "
            "fraction of held-out subjects."
        )

    return Phase3_1GateResult(
        name="F7_subject_generalization",
        passed=passed,
        metrics=asdict(summary),
        interpretation=interpretation,
    )


# =========================================================
# F8 — Complexity Penalty
# =========================================================

def gate_F8_complexity_penalty(
    baseline_score: ModelScore,
    advanced_score: ModelScore,
    metric: str = "bic",
    min_test_ll_improvement: float = 0.0,
) -> Phase3_1GateResult:
    """
    F8 checks whether the advanced model is justified after complexity penalty.

    Supported metric:
        - "bic": advanced bic_test must be lower than baseline bic_test
        - "mdl": same operational behavior as BIC
        - "test_ll": advanced test ll must improve by min_test_ll_improvement
    """
    metric = str(metric).lower()

    ll_improvement = safe_float(advanced_score.ll_test) - safe_float(baseline_score.ll_test)

    if metric in {"bic", "mdl"}:
        baseline_value = safe_float(baseline_score.bic_test, default=float("inf"))
        advanced_value = safe_float(advanced_score.bic_test, default=float("inf"))
        passed = bool(advanced_value < baseline_value)
        metric_delta = baseline_value - advanced_value

        interpretation = (
            "Advanced model improves the test-set complexity-penalized score."
            if passed
            else "Advanced model does not improve the test-set complexity-penalized score."
        )

    elif metric == "test_ll":
        baseline_value = safe_float(baseline_score.ll_test, default=-float("inf"))
        advanced_value = safe_float(advanced_score.ll_test, default=-float("inf"))
        passed = bool(ll_improvement >= float(min_test_ll_improvement))
        metric_delta = ll_improvement

        interpretation = (
            "Advanced model improves held-out log-likelihood by the required margin."
            if passed
            else "Advanced model does not improve held-out log-likelihood by the required margin."
        )

    else:
        raise ValueError("metric must be one of: 'bic', 'mdl', 'test_ll'")

    return Phase3_1GateResult(
        name="F8_complexity_penalty",
        passed=passed,
        metrics={
            "metric": metric,
            "baseline_model": baseline_score.model,
            "advanced_model": advanced_score.model,
            "baseline_metric_value": baseline_value,
            "advanced_metric_value": advanced_value,
            "metric_delta_positive_is_good": metric_delta,
            "ll_test_improvement": ll_improvement,
            "min_test_ll_improvement": float(min_test_ll_improvement),
            "baseline_n_states": baseline_score.n_states,
            "advanced_n_states": advanced_score.n_states,
            "baseline_n_params": baseline_score.n_params,
            "advanced_n_params": advanced_score.n_params,
        },
        interpretation=interpretation,
    )


# =========================================================
# F9 — Proxy Ablation Stability
# =========================================================

def summarize_ablation_effects(
    full_eps_joint: float,
    ablation_eps: Mapping[str, float],
    ablation_delta_min: float,
) -> Dict[str, Any]:
    """
    Measures whether removing proxies reduces the joint epsilon.

    Expected ablation names:
        - remove_I_t
        - remove_Z_t
        - remove_Q_t
    """
    full = safe_float(full_eps_joint)

    rows: Dict[str, Dict[str, float | bool]] = {}

    pass_flags: List[bool] = []
    deltas: List[float] = []

    for name, eps in ablation_eps.items():
        eps_value = safe_float(eps)
        delta = full - eps_value
        passed = bool(delta >= float(ablation_delta_min))

        rows[str(name)] = {
            "eps_ablated": eps_value,
            "delta_from_full": delta,
            "passed": passed,
        }

        pass_flags.append(passed)
        deltas.append(delta)

    if not rows:
        return {
            "full_eps_joint": full,
            "ablation_delta_min": float(ablation_delta_min),
            "n_ablations": 0,
            "fraction_ablations_reduced": 0.0,
            "median_delta_from_full": 0.0,
            "all_required_ablations_reduce_signal": False,
            "ablations": {},
        }

    fraction = float(np.mean(pass_flags))
    median_delta = safe_median(deltas)

    return {
        "full_eps_joint": full,
        "ablation_delta_min": float(ablation_delta_min),
        "n_ablations": int(len(rows)),
        "fraction_ablations_reduced": fraction,
        "median_delta_from_full": median_delta,
        "all_required_ablations_reduce_signal": bool(all(pass_flags)),
        "ablations": rows,
    }


def gate_F9_proxy_ablation_stability(
    full_eps_joint: float,
    ablation_eps: Mapping[str, float],
    ablation_delta_min: float,
    require_all: bool = True,
) -> Phase3_1GateResult:
    """
    F9 checks whether I_t/Z_t/Q_t actually contribute to the final signal.

    If require_all=True:
        every ablation must reduce eps by at least ablation_delta_min.

    If require_all=False:
        at least half of ablations must reduce eps by that margin.
    """
    summary = summarize_ablation_effects(
        full_eps_joint=full_eps_joint,
        ablation_eps=ablation_eps,
        ablation_delta_min=ablation_delta_min,
    )

    if require_all:
        passed = bool(summary["all_required_ablations_reduce_signal"])
    else:
        passed = bool(summary["fraction_ablations_reduced"] >= 0.50)

    interpretation = (
        "Proxy ablations reduce the final joint signal, supporting contribution "
        "from the added Phase III.1 layers."
        if passed
        else "Proxy ablations do not reduce the final joint signal enough; the "
        "added layers may not be explaining the detected structure."
    )

    summary["require_all"] = bool(require_all)

    return Phase3_1GateResult(
        name="F9_proxy_ablation_stability",
        passed=passed,
        metrics=summary,
        interpretation=interpretation,
    )


# =========================================================
# Injection curve / injectability
# =========================================================

def summarize_injection_curve(
    curve: Mapping[float, float],
    principal_eps_true: float,
    tol_abs: float,
) -> InjectionCurveSummary:
    """
    Summarizes injection recovery across multiple injected eps values.
    """
    curve_clean: Dict[str, float] = {}
    errors: Dict[str, float] = {}
    pass_flags: List[bool] = []

    for eps_true, eps_hat in curve.items():
        e_true = safe_float(eps_true)
        e_hat = safe_float(eps_hat)

        key = f"{e_true:.4f}"
        err = abs(e_hat - e_true)

        curve_clean[key] = e_hat
        errors[key] = err
        pass_flags.append(err <= float(tol_abs))

    principal_key = f"{safe_float(principal_eps_true):.4f}"

    principal_eps_hat = curve_clean.get(principal_key, 0.0)
    principal_abs_err = errors.get(principal_key, abs(principal_eps_hat - principal_eps_true))
    principal_passed = bool(principal_abs_err <= float(tol_abs))

    recovered_fraction = float(np.mean(pass_flags)) if pass_flags else 0.0

    if recovered_fraction >= 0.80:
        injectability_class = "high"
    elif recovered_fraction >= 0.50:
        injectability_class = "moderate"
    else:
        injectability_class = "low"

    return InjectionCurveSummary(
        curve=curve_clean,
        errors=errors,
        recovered_fraction=recovered_fraction,
        injectability_class=injectability_class,
        principal_eps_true=float(principal_eps_true),
        principal_eps_hat=float(principal_eps_hat),
        principal_abs_err=float(principal_abs_err),
        principal_passed=principal_passed,
    )


# =========================================================
# Final status classification
# =========================================================

def classify_phase3_1_outcome(
    eps_joint_test: float,
    eps_eeg_test: float,
    eps_rng_test: float,
    gates: Sequence[Phase3_1GateResult],
    strong_joint_eps_min: float,
    joint_lift_min: float,
) -> Dict[str, Any]:
    """
    Classifies Phase III.1 outcome into:
        - strong_success
        - moderate_success
        - null_result
        - invalid_result

    Interpretation:
        - invalid_result: some critical gate failed while epsilon appears high
        - strong_success: joint exceeds EEG/RNG baseline and reaches strong threshold
        - moderate_success: joint improves but does not meet strong criteria
        - null_result: no robust joint evidence
    """
    eps_joint = safe_float(eps_joint_test)
    eps_eeg = safe_float(eps_eeg_test)
    eps_rng = safe_float(eps_rng_test)

    baseline = max(eps_eeg, eps_rng)
    lift = eps_joint - baseline

    gate_map = {g.name: g for g in gates}
    failed = [g.name for g in gates if not g.passed]
    passed_all = len(failed) == 0

    critical_gate_names = {
        "F2_controls_collapse",
        "F3_holdout_generalization",
        "F6_incremental_joint_lift",
        "F7_subject_generalization",
        "F8_complexity_penalty",
    }

    failed_critical = [name for name in failed if name in critical_gate_names]

    if eps_joint >= strong_joint_eps_min and failed_critical:
        status = "invalid_result"
        interpretation = (
            "Joint epsilon is elevated, but one or more critical validation "
            "gates failed. The result should be treated as invalid or likely "
            "overfit/artifactual."
        )

    elif (
        passed_all
        and lift >= joint_lift_min
        and eps_joint >= strong_joint_eps_min
    ):
        status = "strong_success"
        interpretation = (
            "Phase III.1 produced strong joint evidence: the joint model exceeds "
            "single-domain baselines, reaches the strong epsilon threshold, and "
            "passes all gates."
        )

    elif lift > 0 and not failed_critical:
        status = "moderate_success"
        interpretation = (
            "Phase III.1 produced a moderate signal: the joint model improves "
            "over baseline directionally, but does not satisfy all strong-success "
            "criteria."
        )

    else:
        status = "null_result"
        interpretation = (
            "Phase III.1 did not reveal robust joint structure beyond the "
            "single-domain baselines under the current design."
        )

    return {
        "status": status,
        "interpretation": interpretation,
        "passed_all_gates": passed_all,
        "failed_gates": failed,
        "failed_critical_gates": failed_critical,
        "eps_joint_test": eps_joint,
        "eps_eeg_test": eps_eeg,
        "eps_rng_test": eps_rng,
        "baseline_max": baseline,
        "joint_lift": lift,
        "strong_joint_eps_min": float(strong_joint_eps_min),
        "joint_lift_min": float(joint_lift_min),
    }


def summarize_gates(gates: Sequence[Phase3_1GateResult]) -> Dict[str, Any]:
    """
    Compact summary for all gates.
    """
    return {
        "passed_all": bool(all(g.passed for g in gates)),
        "n_gates": int(len(gates)),
        "n_passed": int(sum(1 for g in gates if g.passed)),
        "n_failed": int(sum(1 for g in gates if not g.passed)),
        "failed": [g.name for g in gates if not g.passed],
        "gates": {g.name: {"passed": g.passed, "metrics": g.metrics} for g in gates},
    }


# =========================================================
# Text report generation
# =========================================================

def format_gate_line(gate: Phase3_1GateResult) -> str:
    status = "PASS" if gate.passed else "FAIL"
    return f"{gate.name}: {status} | metrics={gate.metrics}"


def build_phase3_1_summary_text(
    title: str,
    model_scores: Sequence[ModelScore],
    gates: Sequence[Phase3_1GateResult],
    final_status: Mapping[str, Any],
    injection_summary: Optional[InjectionCurveSummary] = None,
) -> str:
    """
    Builds a human-readable summary text for results/phase3_1_summary.txt.
    """
    lines: List[str] = []

    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)
    lines.append("")

    lines.append("Model comparison")
    lines.append("-" * 72)

    for score in model_scores:
        lines.append(
            f"{score.model} | eps_train={score.eps_train:.4f} | "
            f"eps_test={score.eps_test:.4f} | states={score.n_states} | "
            f"BIC_test={score.bic_test:.4f} | LL_test={score.ll_test:.4f}"
        )

    lines.append("")

    if injection_summary is not None:
        lines.append("Injection curve")
        lines.append("-" * 72)
        lines.append(f"Injectability class: {injection_summary.injectability_class}")
        lines.append(f"Recovered fraction: {injection_summary.recovered_fraction:.4f}")
        lines.append(f"Principal eps true: {injection_summary.principal_eps_true:.4f}")
        lines.append(f"Principal eps hat: {injection_summary.principal_eps_hat:.4f}")
        lines.append(f"Principal abs err: {injection_summary.principal_abs_err:.4f}")
        lines.append(f"Principal passed: {injection_summary.principal_passed}")
        lines.append("")

    lines.append("Gates")
    lines.append("-" * 72)

    for gate in gates:
        lines.append(format_gate_line(gate))
        if gate.interpretation:
            lines.append(f"  interpretation: {gate.interpretation}")

    lines.append("")
    lines.append("Final status")
    lines.append("-" * 72)
    lines.append(f"status: {final_status.get('status')}")
    lines.append(f"interpretation: {final_status.get('interpretation')}")
    lines.append(f"passed_all_gates: {final_status.get('passed_all_gates')}")
    lines.append(f"failed_gates: {final_status.get('failed_gates')}")
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


def save_summary_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================================================
# Phase III.1 baseline helpers
# =========================================================

def extract_score_by_model(
    scores: Sequence[ModelScore],
    model_name: str,
) -> Optional[ModelScore]:
    for score in scores:
        if score.model == model_name:
            return score

    return None


def get_final_and_baseline_scores(
    scores: Sequence[ModelScore],
    final_model_name: str,
    baseline_model_name: str,
) -> Tuple[Optional[ModelScore], Optional[ModelScore]]:
    final_score = extract_score_by_model(scores, final_model_name)
    baseline_score = extract_score_by_model(scores, baseline_model_name)

    return final_score, baseline_score


def compact_model_score_dict(scores: Sequence[ModelScore]) -> Dict[str, Dict[str, Any]]:
    return {s.model: asdict(s) for s in scores}