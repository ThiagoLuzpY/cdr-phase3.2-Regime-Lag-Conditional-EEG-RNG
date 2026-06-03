from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.phase3_2_config import (
    Phase32Config,
    active_conditional_pairs,
    describe_config,
    load_phase3_2_config,
)

from src.phase3_2_loader import (
    load_or_build_phase3_2_dataset,
)

from src.phase3_2_features import (
    prepare_phase3_2_features_for_modeling,
    save_phase3_2_modeling_features,
    summarize_feature_frame,
)

from src.phase3_2_regimes import (
    build_all_regime_frames,
    build_regime_inventory,
    save_regime_outputs,
    save_regime_summary_text,
)

from src.phase3_2_lagging import (
    build_all_lagged_frames,
    build_lag_inventory,
    save_lagged_outputs,
    save_lag_summary_text,
)

from src.phase3_2_conditional import (
    evaluate_all_conditional_from_lagged_csvs,
    save_conditional_outputs,
    summarize_conditional_pairs,
)

from src.phase3_2_controls import (
    evaluate_all_controls,
    save_control_outputs,
    summarize_control_runs,
)

from src.phase3_2_metrics import (
    run_phase3_2_metrics,
)

from src.phase3_2_diagnostics import (
    run_phase3_2_diagnostics,
)


# =========================================================
# Phase III.2 — Main Runner
# =========================================================
#
# This runner orchestrates the complete Phase III.2 pipeline:
#
#   1. Loader
#   2. Features + Phase III.2D multichannel layer
#   3. Regimes                 — III.2A
#   4. Lagging                 — III.2B
#   5. Conditional CDR         — III.2C + III.2D
#   6. Negative controls       — III.2D multichannel-aware
#   7. Metrics / gates
#   8. Diagnostics             — III.2D multichannel-aware
#   9. Final report
#
# Current phase logic:
#
#   III.2A/B/C — regime, lagged and conditional leakage-safe validation
#   III.2D     — active exploratory multichannel EEG enrichment
#   III.2E     — protocol-only; requires synchronized EEG + QRNG acquisition
#
# Recommended commands:
#
#   Full run:
#       python -m src.phase3_2_runner
#
#   Reuse existing controls:
#       python -m src.phase3_2_runner --skip-controls
#
#   Fast development run:
#       python -m src.phase3_2_runner --fast-controls
#
#   Force loader/features rebuild:
#       python -m src.phase3_2_runner --force-rebuild
#
#   Skip diagnostics:
#       python -m src.phase3_2_runner --skip-diagnostics
#


EXPECTED_CONTROL_MODULE = "Phase_III_2_negative_controls_leakage_safe"
EXPECTED_CONTROL_MODULE_DETAIL = "phase3_2d_multichannel_aware"


# =========================================================
# Helpers
# =========================================================

def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def _step_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _elapsed(start_time: float) -> float:
    return float(time.time() - start_time)


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    return {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default

        return int(value)

    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default

        x = float(value)

        if not np.isfinite(x):
            return default

        return x

    except Exception:
        return default


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype=bool)

    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _is_multichannel_pair_rows(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)

    mask = pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

    for col in ["baseline_model", "augmented_model", "family"]:
        if col not in df.columns:
            continue

        text = df[col].astype(str)

        mask |= text.str.startswith("D")
        mask |= text.str.contains("MCEEG", case=False, na=False)
        mask |= text.str.contains("multichannel", case=False, na=False)

    return mask


# =========================================================
# Output checks
# =========================================================

def controls_outputs_exist(cfg: Phase32Config) -> bool:
    required = [
        Path(cfg.results_dir) / "phase3_2_control_runs.csv",
        Path(cfg.results_dir) / "phase3_2_control_pair_scores.csv",
        Path(cfg.controls_json),
    ]

    return all(path.exists() for path in required)


def controls_outputs_current(cfg: Phase32Config) -> bool:
    if not controls_outputs_exist(cfg):
        return False

    payload = _load_json(Path(cfg.controls_json))

    module_ok = str(payload.get("module", "")) == EXPECTED_CONTROL_MODULE

    if cfg.run_multichannel_analysis and cfg.use_multichannel_eeg:
        detail_ok = str(payload.get("module_detail", "")) == EXPECTED_CONTROL_MODULE_DETAIL
        return bool(module_ok and detail_ok)

    return bool(module_ok)


def summarize_existing_controls(cfg: Phase32Config) -> Dict[str, Any]:
    payload = _load_json(Path(cfg.controls_json))

    if not payload:
        return {
            "available": False,
            "reason": "controls_json_missing",
        }

    summary = payload.get("summary", {})

    return {
        "available": True,
        "module": payload.get("module"),
        "module_detail": payload.get("module_detail"),
        "expected_module": EXPECTED_CONTROL_MODULE,
        "expected_module_detail": EXPECTED_CONTROL_MODULE_DETAIL,
        "is_current": controls_outputs_current(cfg),
        "passed": summary.get("passed"),
        "failed_control_types": summary.get("failed_control_types", []),
        "n_control_types": summary.get("n_control_types"),
        "runs_file": payload.get("runs_file"),
        "pair_scores_file": payload.get("pair_scores_file"),
    }


def conditional_outputs_exist(cfg: Phase32Config) -> bool:
    required = [
        Path(cfg.conditional_results_csv),
        Path(cfg.results_dir) / "phase3_2c_conditional_model_scores.csv",
        Path(cfg.conditional_summary_json),
    ]

    if cfg.run_multichannel_analysis and cfg.use_multichannel_eeg:
        required.append(Path(cfg.multichannel_results_csv))

    return all(path.exists() for path in required)


# =========================================================
# Phase III.2D summary helpers
# =========================================================

def summarize_phase3_2d_artifacts(cfg: Phase32Config) -> Dict[str, Any]:
    inventory = _load_json(Path(cfg.multichannel_inventory_json))
    summary = _load_json(Path(cfg.multichannel_summary_json))
    mc_results = _load_csv_if_exists(Path(cfg.multichannel_results_csv))
    pair_scores = _load_csv_if_exists(Path(cfg.conditional_results_csv))
    control_runs = _load_csv_if_exists(Path(cfg.results_dir) / "phase3_2_control_runs.csv")
    control_pairs = _load_csv_if_exists(Path(cfg.results_dir) / "phase3_2_control_pair_scores.csv")

    mc_pair_rows = pd.DataFrame()
    mc_positive_rows = pd.DataFrame()
    mc_strong_rows = pd.DataFrame()

    if not pair_scores.empty:
        if "valid" in pair_scores.columns:
            valid = pair_scores[_as_bool_series(pair_scores["valid"])].copy()
        else:
            valid = pair_scores.copy()

        if not valid.empty:
            mc_pair_rows = valid[_is_multichannel_pair_rows(valid)].copy()

            if not mc_pair_rows.empty and "conditional_lift" in mc_pair_rows.columns:
                lift = pd.to_numeric(mc_pair_rows["conditional_lift"], errors="coerce").fillna(0.0)
                mc_positive_rows = mc_pair_rows[lift > 0.0].copy()

                eps = pd.to_numeric(mc_pair_rows.get("augmented_eps_test", 0.0), errors="coerce").fillna(0.0)
                frac = pd.to_numeric(mc_pair_rows.get("fraction_positive_lift", 0.0), errors="coerce").fillna(0.0)
                bic = pd.to_numeric(mc_pair_rows.get("bic_improvement", 0.0), errors="coerce").fillna(0.0)

                mc_strong_rows = mc_pair_rows[
                    (lift >= float(cfg.conditional_lift_min))
                    & (eps >= float(cfg.strong_eps_min))
                    & (frac >= float(cfg.subject_effect_fraction))
                    & (bic >= 0.0)
                ].copy()

    mc_control_pair_rows = pd.DataFrame()

    if not control_pairs.empty:
        mc_control_pair_rows = control_pairs[_is_multichannel_pair_rows(control_pairs)].copy()

    source_columns = inventory.get("source_columns", {}) if inventory else {}
    artifact_summary = summary.get("summary", {}) if summary else {}

    return {
        "enabled": bool(cfg.run_multichannel_analysis and cfg.use_multichannel_eeg),
        "protocol_status": "active_exploratory",
        "multichannel_available": bool(source_columns.get("multichannel_available", False)),
        "delta_secondary_source_column": source_columns.get("delta_secondary_source_column"),
        "alpha_secondary_source_column": source_columns.get("alpha_secondary_source_column"),
        "state_n_unique": inventory.get("state_n_unique") if inventory else None,
        "next_state_n_unique": inventory.get("next_state_n_unique") if inventory else None,
        "info_bin_n_unique": inventory.get("info_bin_n_unique") if inventory else None,
        "valid_multichannel_next_state_rows": inventory.get("valid_multichannel_next_state_rows") if inventory else None,
        "valid_multichannel_conditional_rows": inventory.get("valid_multichannel_conditional_rows") if inventory else None,
        "warnings": inventory.get("warnings", []) if inventory else [],
        "artifact_summary_available": bool(summary),
        "artifact_summary_rows": artifact_summary.get("n_rows"),
        "artifact_summary_subjects": artifact_summary.get("subjects"),
        "conditional": {
            "n_multichannel_result_rows": int(len(mc_results)),
            "n_multichannel_pair_rows": int(len(mc_pair_rows)),
            "n_multichannel_positive_lift_rows": int(len(mc_positive_rows)),
            "n_multichannel_strong_candidate_rows": int(len(mc_strong_rows)),
            "max_multichannel_lift": (
                float(pd.to_numeric(mc_pair_rows["conditional_lift"], errors="coerce").fillna(0.0).max())
                if not mc_pair_rows.empty and "conditional_lift" in mc_pair_rows.columns
                else 0.0
            ),
            "max_multichannel_augmented_eps": (
                float(pd.to_numeric(mc_pair_rows["augmented_eps_test"], errors="coerce").fillna(0.0).max())
                if not mc_pair_rows.empty and "augmented_eps_test" in mc_pair_rows.columns
                else 0.0
            ),
            "best_multichannel_bic_improvement": (
                float(pd.to_numeric(mc_pair_rows["bic_improvement"], errors="coerce").fillna(0.0).max())
                if not mc_pair_rows.empty and "bic_improvement" in mc_pair_rows.columns
                else 0.0
            ),
            "best_multichannel_ll_improvement": (
                float(pd.to_numeric(mc_pair_rows["ll_improvement"], errors="coerce").fillna(0.0).max())
                if not mc_pair_rows.empty and "ll_improvement" in mc_pair_rows.columns
                else 0.0
            ),
        },
        "controls": {
            "n_control_run_rows": int(len(control_runs)),
            "n_control_pair_rows": int(len(control_pairs)),
            "n_multichannel_control_pair_rows": int(len(mc_control_pair_rows)),
        },
        "files": {
            "multichannel_features_file": str(cfg.multichannel_features_file),
            "multichannel_inventory_json": str(cfg.multichannel_inventory_json),
            "multichannel_summary_json": str(cfg.multichannel_summary_json),
            "multichannel_results_csv": str(cfg.multichannel_results_csv),
        },
    }


def phase3_2e_protocol_status(cfg: Phase32Config) -> Dict[str, Any]:
    return {
        "status": "protocol_only",
        "enabled_for_empirical_run": bool(cfg.run_synchronized_protocol_analysis),
        "protocol_only": bool(cfg.phase3_2e_protocol_only),
        "requires_real_time_sync": bool(cfg.phase3_2e_requires_real_time_sync),
        "required_timestamp_columns": list(cfg.phase3_2e_required_timestamp_columns),
        "min_sync_quality": float(cfg.phase3_2e_min_sync_quality),
        "max_clock_drift_ms": float(cfg.phase3_2e_max_clock_drift_ms),
        "protocol_doc": str(cfg.phase3_2e_protocol_doc),
        "interpretation": (
            "Phase III.2E is intentionally not executed on Sleep-EDF + ANU QRNG. "
            "It requires synchronously acquired EEG + QRNG timestamps."
        ),
    }


def build_subphase_status(cfg: Phase32Config) -> Dict[str, Any]:
    return {
        "III.2A": {
            "name": "regime-aware sleep-stage validation",
            "status": "active",
        },
        "III.2B": {
            "name": "lagged EEG-RNG alignment",
            "status": "active",
        },
        "III.2C": {
            "name": "leakage-safe conditional CDR",
            "status": "active",
        },
        "III.2D": {
            "name": "multichannel EEG enrichment",
            "status": "active_exploratory" if cfg.run_multichannel_analysis and cfg.use_multichannel_eeg else "disabled",
        },
        "III.2E": phase3_2e_protocol_status(cfg),
    }


# =========================================================
# Summary text
# =========================================================

def build_runner_summary_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("CDR Phase III.2 — Runner Summary")
    lines.append("Leakage-safe conditional estimator aware version")
    lines.append("Phase III.2D multichannel-aware")
    lines.append("=" * 78)
    lines.append("")

    lines.append("Project")
    lines.append("-" * 78)
    lines.append(f"project_name: {report.get('project_name')}")
    lines.append(f"phase: {report.get('phase')}")
    lines.append(f"phase_title: {report.get('phase_title')}")
    lines.append(f"version: {report.get('version')}")
    lines.append(f"runner_version: {report.get('runner_version')}")
    lines.append("")

    lines.append("Subphases")
    lines.append("-" * 78)

    for key, value in report.get("subphases", {}).items():
        lines.append(f"{key}: {value}")

    lines.append("")

    lines.append("Execution")
    lines.append("-" * 78)
    lines.append(f"started_at: {report.get('started_at')}")
    lines.append(f"finished_at: {report.get('finished_at')}")
    lines.append(f"elapsed_seconds: {report.get('elapsed_seconds')}")
    lines.append(f"force_rebuild: {report.get('options', {}).get('force_rebuild')}")
    lines.append(f"skip_controls: {report.get('options', {}).get('skip_controls')}")
    lines.append(f"fast_controls: {report.get('options', {}).get('fast_controls')}")
    lines.append(f"skip_diagnostics: {report.get('options', {}).get('skip_diagnostics')}")
    lines.append(f"n_controls_effective: {report.get('options', {}).get('n_controls_effective')}")
    lines.append("")

    lines.append("Outputs")
    lines.append("-" * 78)

    for key, value in report.get("outputs", {}).items():
        lines.append(f"{key}: {value}")

    lines.append("")

    lines.append("Step summaries")
    lines.append("-" * 78)

    for step_name, payload in report.get("steps", {}).items():
        lines.append(f"{step_name}:")
        lines.append(f"  status: {payload.get('status')}")
        lines.append(f"  elapsed_seconds: {payload.get('elapsed_seconds')}")

        if "summary" in payload:
            lines.append(f"  summary: {payload.get('summary')}")

        if "warning" in payload:
            lines.append(f"  warning: {payload.get('warning')}")

        if "error" in payload:
            lines.append(f"  error: {payload.get('error')}")

        lines.append("")

    phase3_2d = report.get("phase3_2d_summary", {})

    if phase3_2d:
        lines.append("Phase III.2D multichannel summary")
        lines.append("-" * 78)
        lines.append(f"enabled: {phase3_2d.get('enabled')}")
        lines.append(f"multichannel_available: {phase3_2d.get('multichannel_available')}")
        lines.append(f"delta_secondary_source_column: {phase3_2d.get('delta_secondary_source_column')}")
        lines.append(f"alpha_secondary_source_column: {phase3_2d.get('alpha_secondary_source_column')}")
        lines.append(f"state_n_unique: {phase3_2d.get('state_n_unique')}")
        lines.append(f"next_state_n_unique: {phase3_2d.get('next_state_n_unique')}")
        lines.append(f"valid_multichannel_conditional_rows: {phase3_2d.get('valid_multichannel_conditional_rows')}")
        lines.append(f"warnings: {phase3_2d.get('warnings')}")
        lines.append(f"conditional: {phase3_2d.get('conditional')}")
        lines.append(f"controls: {phase3_2d.get('controls')}")
        lines.append("")

    final_status = report.get("final_status", {})

    lines.append("Final status")
    lines.append("-" * 78)
    lines.append(f"status: {final_status.get('status')}")
    lines.append(f"interpretation: {final_status.get('interpretation')}")
    lines.append(f"passed_gates: {final_status.get('passed_gates')}")
    lines.append(f"failed_gates: {final_status.get('failed_gates')}")
    lines.append(f"warning_gates: {final_status.get('warning_gates')}")
    lines.append(f"not_evaluated_gates: {final_status.get('not_evaluated_gates')}")
    lines.append("")

    diagnostics = report.get("diagnostics_final_interpretation", {})

    if diagnostics:
        lines.append("Diagnostics final interpretation")
        lines.append("-" * 78)
        lines.append(f"headline: {diagnostics.get('headline')}")
        lines.append(f"status: {diagnostics.get('status')}")
        lines.append(f"recommendation: {diagnostics.get('recommendation')}")
        lines.append("")

    phase3_2e = report.get("phase3_2e_protocol_status", {})

    if phase3_2e:
        lines.append("Phase III.2E protocol status")
        lines.append("-" * 78)
        lines.append(f"status: {phase3_2e.get('status')}")
        lines.append(f"enabled_for_empirical_run: {phase3_2e.get('enabled_for_empirical_run')}")
        lines.append(f"requires_real_time_sync: {phase3_2e.get('requires_real_time_sync')}")
        lines.append(f"protocol_doc: {phase3_2e.get('protocol_doc')}")
        lines.append(f"interpretation: {phase3_2e.get('interpretation')}")
        lines.append("")

    lines.append("=" * 78)

    return "\n".join(lines)


# =========================================================
# Main pipeline
# =========================================================

def run_phase3_2_pipeline(
    cfg: Optional[Phase32Config] = None,
    force_rebuild: bool = False,
    skip_controls: bool = False,
    fast_controls: bool = False,
    skip_diagnostics: bool = False,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2_config()

    if fast_controls:
        cfg = replace(cfg, n_controls=3)

    np.random.seed(int(cfg.numpy_seed))

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.diagnostics_dir).mkdir(parents=True, exist_ok=True)

    runner_report_json = Path(cfg.results_dir) / "phase3_2_runner_report.json"
    runner_summary_txt = Path(cfg.results_dir) / "phase3_2_runner_summary.txt"

    started_at = _now_str()
    pipeline_start = time.time()

    report: Dict[str, Any] = {
        "project_name": cfg.project_name,
        "phase": cfg.phase_name,
        "phase_title": cfg.phase_title,
        "version": cfg.version,
        "runner_version": "phase3_2_runner_v3_phase3_2d_multichannel_aware",
        "started_at": started_at,
        "finished_at": None,
        "elapsed_seconds": None,
        "subphases": build_subphase_status(cfg),
        "phase3_2e_protocol_status": phase3_2e_protocol_status(cfg),
        "options": {
            "force_rebuild": bool(force_rebuild),
            "skip_controls": bool(skip_controls),
            "fast_controls": bool(fast_controls),
            "skip_diagnostics": bool(skip_diagnostics),
            "n_controls_effective": int(cfg.n_controls),
        },
        "config": describe_config(cfg),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        "steps": {},
        "outputs": {},
        "final_status": {},
        "diagnostics_final_interpretation": {},
        "phase3_2d_summary": {},
    }

    # -----------------------------------------------------
    # Step 1 — Loader
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 1/8 — Loader")

    try:
        base_df, loader_metadata = load_or_build_phase3_2_dataset(
            cfg=cfg,
            force_rebuild=bool(force_rebuild),
        )

        report["steps"]["loader"] = {
            "status": "ok",
            "elapsed_seconds": _elapsed(step_start),
            "summary": {
                "rows": int(len(base_df)),
                "subjects": loader_metadata.get("subjects_loaded"),
                "rng_bit_count": loader_metadata.get("rng_bit_count"),
                "multichannel_loader_available": loader_metadata.get("multichannel_loader_available"),
                "has_secondary_channel_rows": loader_metadata.get("has_secondary_channel_rows"),
                "secondary_channel_subjects": loader_metadata.get("secondary_channel_subjects"),
            },
        }

        report["outputs"]["combined_features_file"] = str(cfg.combined_features_file)
        report["outputs"]["loader_metadata_file"] = str(cfg.loader_metadata_file)

    except Exception as exc:
        report["steps"]["loader"] = {
            "status": "error",
            "elapsed_seconds": _elapsed(step_start),
            "error": str(exc),
        }
        raise

    # -----------------------------------------------------
    # Step 2 — Features + III.2D
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 2/8 — Feature preparation + Phase III.2D multichannel layer")

    try:
        feature_frame = prepare_phase3_2_features_for_modeling(
            df=base_df,
            cfg=cfg,
            train_index=None,
        )

        save_phase3_2_modeling_features(feature_frame, cfg)

        feature_summary = summarize_feature_frame(feature_frame.df)

        report["steps"]["features"] = {
            "status": "ok",
            "elapsed_seconds": _elapsed(step_start),
            "summary": feature_summary,
        }

        report["outputs"]["modeling_features_file"] = str(Path(cfg.interim_dir) / "phase3_2_modeling_features.csv")
        report["outputs"]["feature_specs_file"] = str(Path(cfg.interim_dir) / "phase3_2_feature_specs.json")

        if cfg.run_multichannel_analysis and cfg.use_multichannel_eeg:
            report["outputs"]["phase3_2d_multichannel_features"] = str(cfg.multichannel_features_file)
            report["outputs"]["phase3_2d_multichannel_inventory"] = str(cfg.multichannel_inventory_json)
            report["outputs"]["phase3_2d_multichannel_summary"] = str(cfg.multichannel_summary_json)

    except Exception as exc:
        report["steps"]["features"] = {
            "status": "error",
            "elapsed_seconds": _elapsed(step_start),
            "error": str(exc),
        }
        raise

    # -----------------------------------------------------
    # Step 3 — Regimes
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 3/8 — Regime-aware analysis preparation")

    try:
        regime_frames = build_all_regime_frames(
            df=feature_frame.df,
            cfg=cfg,
            keep_only_valid_subjects=True,
        )

        save_regime_outputs(regime_frames, cfg)
        save_regime_summary_text(regime_frames, cfg)

        regime_inventory = build_regime_inventory(regime_frames)

        report["steps"]["regimes"] = {
            "status": "ok",
            "elapsed_seconds": _elapsed(step_start),
            "summary": {
                "n_regimes": int(len(regime_frames)),
                "valid_regimes": [
                    name for name, frame in regime_frames.items() if frame.valid
                ],
                "inventory_rows": int(len(regime_inventory)),
            },
        }

        report["outputs"]["regime_inventory"] = str(Path(cfg.interim_dir) / "phase3_2_regime_inventory.csv")
        report["outputs"]["regime_summary"] = str(Path(cfg.interim_dir) / "phase3_2_regime_summary.json")

    except Exception as exc:
        report["steps"]["regimes"] = {
            "status": "error",
            "elapsed_seconds": _elapsed(step_start),
            "error": str(exc),
        }
        raise

    # -----------------------------------------------------
    # Step 4 — Lagging
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 4/8 — Lagged EEG-RNG alignment")

    try:
        lagged_frames = build_all_lagged_frames(
            regime_frames=regime_frames,
            cfg=cfg,
            keep_only_valid_subjects=True,
            valid_regimes_only=True,
        )

        save_lagged_outputs(
            lagged_frames=lagged_frames,
            cfg=cfg,
            save_valid_csvs=True,
        )

        save_lag_summary_text(
            lagged_frames=lagged_frames,
            cfg=cfg,
        )

        lag_inventory = build_lag_inventory(lagged_frames)

        report["steps"]["lagging"] = {
            "status": "ok",
            "elapsed_seconds": _elapsed(step_start),
            "summary": {
                "lag_inventory_rows": int(len(lag_inventory)),
                "valid_lagged_frames": (
                    int(lag_inventory["valid"].sum())
                    if not lag_inventory.empty and "valid" in lag_inventory.columns
                    else 0
                ),
            },
        }

        report["outputs"]["lag_inventory"] = str(Path(cfg.interim_dir) / "phase3_2_lag_inventory.csv")
        report["outputs"]["lag_summary"] = str(Path(cfg.interim_dir) / "phase3_2_lag_summary.json")

    except Exception as exc:
        report["steps"]["lagging"] = {
            "status": "error",
            "elapsed_seconds": _elapsed(step_start),
            "error": str(exc),
        }
        raise

    # -----------------------------------------------------
    # Step 5 — Conditional CDR + III.2D
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 5/8 — Conditional CDR + Phase III.2D multichannel pairs")

    try:
        model_scores_df, pair_scores_df = evaluate_all_conditional_from_lagged_csvs(cfg)

        save_conditional_outputs(
            model_scores_df=model_scores_df,
            pair_scores_df=pair_scores_df,
            cfg=cfg,
        )

        conditional_summary = summarize_conditional_pairs(pair_scores_df, cfg)

        report["steps"]["conditional"] = {
            "status": "ok",
            "elapsed_seconds": _elapsed(step_start),
            "summary": conditional_summary,
        }

        report["outputs"]["conditional_model_scores"] = str(Path(cfg.results_dir) / "phase3_2c_conditional_model_scores.csv")
        report["outputs"]["conditional_pair_scores"] = str(cfg.conditional_results_csv)
        report["outputs"]["conditional_summary"] = str(cfg.conditional_summary_json)
        report["outputs"]["phase3_2d_multichannel_results"] = str(cfg.multichannel_results_csv)

    except Exception as exc:
        report["steps"]["conditional"] = {
            "status": "error",
            "elapsed_seconds": _elapsed(step_start),
            "error": str(exc),
        }
        raise

    # -----------------------------------------------------
    # Step 6 — Controls
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 6/8 — Negative controls + Phase III.2D multichannel-aware controls")

    if skip_controls:
        existing_controls = summarize_existing_controls(cfg)

        if controls_outputs_current(cfg):
            report["steps"]["controls"] = {
                "status": "skipped_reusing_current_outputs",
                "elapsed_seconds": _elapsed(step_start),
                "summary": existing_controls,
            }

            print("[Phase3.2 Runner] Controls skipped; current III.2D-aware leakage-safe outputs will be reused.")

        elif controls_outputs_exist(cfg):
            report["steps"]["controls"] = {
                "status": "skipped_reusing_stale_or_unknown_outputs",
                "elapsed_seconds": _elapsed(step_start),
                "summary": existing_controls,
                "warning": (
                    "Existing controls were found, but their module/module_detail does not match "
                    f"{EXPECTED_CONTROL_MODULE}/{EXPECTED_CONTROL_MODULE_DETAIL}. "
                    "Metrics may mark F2 as pending/stale or may not fully reflect III.2D."
                ),
            }

            print("[Phase3.2 Runner] WARNING: controls skipped but existing controls may be stale or not III.2D-aware.")

        else:
            report["steps"]["controls"] = {
                "status": "skipped_missing_existing_outputs",
                "elapsed_seconds": _elapsed(step_start),
                "summary": existing_controls,
                "warning": "Controls skipped but no existing controls output was found.",
            }

            print("[Phase3.2 Runner] WARNING: controls skipped and no existing controls output found.")

    else:
        try:
            runs_df, control_pairs_df = evaluate_all_controls(cfg)

            save_control_outputs(
                runs_df=runs_df,
                pairs_df=control_pairs_df,
                cfg=cfg,
            )

            controls_summary = summarize_control_runs(runs_df, cfg)

            report["steps"]["controls"] = {
                "status": "ok",
                "elapsed_seconds": _elapsed(step_start),
                "summary": controls_summary,
            }

            report["outputs"]["control_runs"] = str(Path(cfg.results_dir) / "phase3_2_control_runs.csv")
            report["outputs"]["control_pair_scores"] = str(Path(cfg.results_dir) / "phase3_2_control_pair_scores.csv")
            report["outputs"]["controls_json"] = str(cfg.controls_json)

        except Exception as exc:
            report["steps"]["controls"] = {
                "status": "error",
                "elapsed_seconds": _elapsed(step_start),
                "error": str(exc),
            }
            raise

    # -----------------------------------------------------
    # Step 7 — Metrics / gates
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 7/8 — Metrics and gates")

    try:
        final_status = run_phase3_2_metrics(cfg)

        report["steps"]["metrics"] = {
            "status": "ok",
            "elapsed_seconds": _elapsed(step_start),
            "summary": asdict(final_status),
        }

        report["final_status"] = asdict(final_status)

        report["outputs"]["gates_json"] = str(cfg.gates_json)
        report["outputs"]["multiple_comparison_guard"] = str(cfg.multiple_comparison_guard_json)
        report["outputs"]["metrics_summary"] = str(Path(cfg.results_dir) / "phase3_2_metrics_summary.txt")
        report["outputs"]["metrics_table"] = str(Path(cfg.results_dir) / "phase3_2_metrics_table.csv")

    except Exception as exc:
        report["steps"]["metrics"] = {
            "status": "error",
            "elapsed_seconds": _elapsed(step_start),
            "error": str(exc),
        }
        raise

    # -----------------------------------------------------
    # Save provisional runner report BEFORE diagnostics
    # -----------------------------------------------------
    report["phase3_2d_summary"] = summarize_phase3_2d_artifacts(cfg)
    report["finished_at"] = _now_str()
    report["elapsed_seconds"] = _elapsed(pipeline_start)
    report["outputs"]["runner_report_json"] = str(runner_report_json)
    report["outputs"]["runner_summary_txt"] = str(runner_summary_txt)

    save_json(runner_report_json, report)

    # -----------------------------------------------------
    # Step 8 — Diagnostics
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 8/8 — Diagnostics + Phase III.2D multichannel-aware report")

    if skip_diagnostics:
        report["steps"]["diagnostics"] = {
            "status": "skipped",
            "elapsed_seconds": _elapsed(step_start),
            "summary": {},
        }

        print("[Phase3.2 Runner] Diagnostics skipped by CLI flag.")

    else:
        try:
            diagnostics_report = run_phase3_2_diagnostics(cfg)
            final_interpretation = _safe_dict(
                diagnostics_report.get("final_interpretation", {})
            )

            phase3_2d_diagnostics = _safe_dict(
                diagnostics_report.get("phase3_2d_multichannel_diagnostics", {})
            )

            report["steps"]["diagnostics"] = {
                "status": "ok",
                "elapsed_seconds": _elapsed(step_start),
                "summary": final_interpretation,
                "phase3_2d_summary": phase3_2d_diagnostics,
            }

            report["diagnostics_final_interpretation"] = final_interpretation
            report["phase3_2d_diagnostics"] = phase3_2d_diagnostics
            report["phase3_2d_summary"] = summarize_phase3_2d_artifacts(cfg)

            diagnostics_dir = Path(cfg.diagnostics_dir)

            report["outputs"]["diagnostics_report"] = str(diagnostics_dir / "phase3_2_diagnostics_report.json")
            report["outputs"]["diagnostics_summary"] = str(diagnostics_dir / "phase3_2_diagnostics_summary.txt")
            report["outputs"]["diagnostics_primary_tests"] = str(diagnostics_dir / "phase3_2_primary_tests.csv")
            report["outputs"]["diagnostics_positive_lift_rows"] = str(diagnostics_dir / "phase3_2_positive_lift_rows.csv")
            report["outputs"]["diagnostics_exploratory_lift_rows"] = str(diagnostics_dir / "phase3_2_exploratory_lift_rows.csv")
            report["outputs"]["diagnostics_strong_candidate_rows"] = str(diagnostics_dir / "phase3_2_strong_candidate_rows.csv")
            report["outputs"]["diagnostics_control_breakdown"] = str(diagnostics_dir / "phase3_2_control_breakdown.csv")

            report["outputs"]["diagnostics_phase3_2c_single_channel_rows"] = str(diagnostics_dir / "phase3_2c_single_channel_rows.csv")
            report["outputs"]["diagnostics_phase3_2c_single_channel_positive_lift_rows"] = str(diagnostics_dir / "phase3_2c_single_channel_positive_lift_rows.csv")

            report["outputs"]["diagnostics_phase3_2d_multichannel_rows"] = str(diagnostics_dir / "phase3_2d_multichannel_rows.csv")
            report["outputs"]["diagnostics_phase3_2d_multichannel_positive_lift_rows"] = str(diagnostics_dir / "phase3_2d_multichannel_positive_lift_rows.csv")
            report["outputs"]["diagnostics_phase3_2d_multichannel_strong_candidate_rows"] = str(diagnostics_dir / "phase3_2d_multichannel_strong_candidate_rows.csv")
            report["outputs"]["diagnostics_phase3_2d_multichannel_best_rows"] = str(diagnostics_dir / "phase3_2d_multichannel_best_rows.csv")
            report["outputs"]["diagnostics_phase3_2d_control_pair_scores"] = str(diagnostics_dir / "phase3_2d_control_pair_scores.csv")

        except Exception as exc:
            report["steps"]["diagnostics"] = {
                "status": "error",
                "elapsed_seconds": _elapsed(step_start),
                "error": str(exc),
            }
            raise

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------
    report["phase3_2d_summary"] = summarize_phase3_2d_artifacts(cfg)
    report["finished_at"] = _now_str()
    report["elapsed_seconds"] = _elapsed(pipeline_start)

    save_json(runner_report_json, report)

    with open(runner_summary_txt, "w", encoding="utf-8") as f:
        f.write(build_runner_summary_text(report))

    save_json(runner_report_json, report)

    print("\n" + "=" * 78)
    print("Phase III.2 runner completed")
    print("Leakage-safe conditional estimator aware version")
    print("Phase III.2D multichannel-aware")
    print("=" * 78)
    print(f"status: {report.get('final_status', {}).get('status')}")
    print(f"interpretation: {report.get('final_status', {}).get('interpretation')}")

    diagnostics_final = report.get("diagnostics_final_interpretation", {})

    if diagnostics_final:
        print(f"diagnostic_headline: {diagnostics_final.get('headline')}")
        print(f"diagnostic_recommendation: {diagnostics_final.get('recommendation')}")

    phase3_2d = report.get("phase3_2d_summary", {})
    phase3_2d_conditional = phase3_2d.get("conditional", {}) if phase3_2d else {}

    if phase3_2d:
        print("")
        print("Phase III.2D summary:")
        print(f"multichannel_available: {phase3_2d.get('multichannel_available')}")
        print(f"state_n_unique: {phase3_2d.get('state_n_unique')}")
        print(f"valid_multichannel_conditional_rows: {phase3_2d.get('valid_multichannel_conditional_rows')}")
        print(f"multichannel positive lift rows: {phase3_2d_conditional.get('n_multichannel_positive_lift_rows')}")
        print(f"multichannel strong candidate rows: {phase3_2d_conditional.get('n_multichannel_strong_candidate_rows')}")
        print(f"max multichannel lift: {phase3_2d_conditional.get('max_multichannel_lift')}")
        print(f"best multichannel BIC improvement: {phase3_2d_conditional.get('best_multichannel_bic_improvement')}")

    print("")
    print(f"runner report: {runner_report_json}")
    print(f"runner summary: {runner_summary_txt}")
    print("=" * 78 + "\n")

    return report


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full CDR Phase III.2 pipeline."
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding loader/features outputs.",
    )

    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip controls and reuse existing III.2D-aware control outputs if available.",
    )

    parser.add_argument(
        "--fast-controls",
        action="store_true",
        help="Run only 3 control replicates for faster development.",
    )

    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip diagnostics generation at the end of the runner.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2_config()

    run_phase3_2_pipeline(
        cfg=cfg,
        force_rebuild=bool(args.force_rebuild),
        skip_controls=bool(args.skip_controls),
        fast_controls=bool(args.fast_controls),
        skip_diagnostics=bool(args.skip_diagnostics),
    )


if __name__ == "__main__":
    main()