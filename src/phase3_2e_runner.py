from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2e_config import (
    Phase32EConfig,
    active_conditional_pairs,
    describe_config,
    load_phase3_2e_config,
)


# =========================================================
# Phase III.2E — Main Runner
# Synchronized EEG–QRNG Protocol Runner
# =========================================================
#
# This runner orchestrates the complete Phase III.2E pipeline:
#
#   1.  Config check
#   2.  Schema validation
#   3.  Synchronization validation
#   4.  Loader
#   5.  Window builder
#   6.  Feature builder
#   7.  Lagged alignment
#   8.  Conditional CDR
#   9.  Negative controls
#   10. Metrics / official gates
#   11. Gate sensitivity diagnostics — F7/F8/F12
#   12. Final diagnostics
#   13. Runner report
#
# Scientific rule:
#
#   Phase III.2E is allowed to run in protocol-only mode when no real
#   synchronized EEG–QRNG CSV exists.
#
#   However, empirical CDR remains blocked until real synchronized EEG + QRNG
#   acquisition timestamps are available and all required gates/controls pass.
#
# Recommended commands:
#
#   Full protocol run:
#       python -m src.phase3_2e_runner
#
#   Skip controls:
#       python -m src.phase3_2e_runner --skip-controls
#
#   Skip diagnostics:
#       python -m src.phase3_2e_runner --skip-diagnostics
#
#   Force rebuild where supported:
#       python -m src.phase3_2e_runner --force-rebuild
#


RUNNER_MODULE = "phase3_2e_runner"
RUNNER_VERSION = "phase3_2e_runner_v1_full_protocol_sync_gate_sensitivity"

PIPELINE_STEPS: List[Tuple[str, str, str]] = [
    ("config", "Config check", "config.phase3_2e_config"),
    ("schema", "Schema validation", "src.phase3_2e_schema"),
    ("sync_validator", "Synchronization validation", "src.phase3_2e_sync_validator"),
    ("loader", "Synchronized loader", "src.phase3_2e_loader"),
    ("windows", "Window builder", "src.phase3_2e_windows"),
    ("features", "Feature builder", "src.phase3_2e_features"),
    ("alignment", "Lagged alignment", "src.phase3_2e_alignment"),
    ("conditional", "Conditional CDR", "src.phase3_2e_conditional"),
    ("controls", "Negative controls", "src.phase3_2e_controls"),
    ("metrics", "Metrics and official gates", "src.phase3_2e_metrics"),
    ("gate_sensitivity", "F7/F8/F12 gate sensitivity", "src.phase3_2e_gate_sensitivity"),
    ("diagnostics", "Final diagnostics", "src.phase3_2e_diagnostics"),
]


# =========================================================
# Generic helpers
# =========================================================

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _elapsed(start_time: float) -> float:
    return float(time.time() - start_time)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


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

    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _step_header(index: int, total: int, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"Step {index}/{total} — {title}")
    print("=" * 78)


def _final_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _file_status(path: Path) -> Dict[str, Any]:
    path = Path(path)

    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
    }


def _payload_status(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "missing"

    if "status" in payload:
        return str(payload.get("status"))

    result = payload.get("result", {})

    if isinstance(result, dict) and "status" in result:
        return str(result.get("status"))

    final = payload.get("final_status", {})

    if isinstance(final, dict) and "status" in final:
        return str(final.get("status"))

    return "unknown"


def _payload_protocol_only(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False

    result = payload.get("result", {})

    if isinstance(result, dict) and bool(result.get("protocol_only", False)):
        return True

    final = payload.get("final_status", {})

    if isinstance(final, dict) and bool(final.get("protocol_only", False)):
        return True

    if str(payload.get("status", "")).lower() == "pending_synchronized_data":
        return True

    return False


def _payload_interpretation(payload: Mapping[str, Any]) -> str:
    if not payload:
        return ""

    result = payload.get("result", {})

    if isinstance(result, dict) and result.get("interpretation"):
        return str(result.get("interpretation"))

    final = payload.get("final_status", {})

    if isinstance(final, dict) and final.get("interpretation"):
        return str(final.get("interpretation"))

    if payload.get("interpretation"):
        return str(payload.get("interpretation"))

    return ""


# =========================================================
# Path registry
# =========================================================

def phase3_2e_paths(cfg: Phase32EConfig) -> Dict[str, Path]:
    results_dir = Path(cfg.results_dir)
    interim_dir = Path(cfg.interim_dir)

    diagnostics_dir = Path(
        getattr(
            cfg,
            "diagnostics_dir",
            results_dir / "diagnostics",
        )
    )

    synchronized_input_csv = Path(
        getattr(
            cfg,
            "synchronized_input_csv",
            Path("data") / "raw" / "phase3_2e" / "synchronized" / "phase3_2e_synchronized_input.csv",
        )
    )

    lagged_frames_dir = Path(
        getattr(
            cfg,
            "lagged_frames_dir",
            interim_dir / "lagged_frames",
        )
    )

    return {
        "synchronized_input_csv": synchronized_input_csv,

        "schema_report": results_dir / "phase3_2e_schema_report.json",
        "schema_summary": results_dir / "phase3_2e_schema_summary.txt",

        "sync_validation_report": results_dir / "phase3_2e_sync_validation_report.json",
        "sync_validation_summary": results_dir / "phase3_2e_sync_validation_summary.txt",

        "loaded_windows": interim_dir / "phase3_2e_loaded_synchronized_windows.csv",
        "loader_metadata": interim_dir / "phase3_2e_loader_metadata.json",
        "loader_report": results_dir / "phase3_2e_loader_report.json",
        "loader_summary": results_dir / "phase3_2e_loader_summary.txt",

        "windows": interim_dir / "phase3_2e_windows.csv",
        "window_inventory": interim_dir / "phase3_2e_window_inventory.csv",
        "windows_report": results_dir / "phase3_2e_windows_report.json",
        "windows_summary": results_dir / "phase3_2e_windows_summary.txt",

        "features": interim_dir / "phase3_2e_features.csv",
        "feature_specs": interim_dir / "phase3_2e_feature_specs.json",
        "features_report": results_dir / "phase3_2e_features_report.json",
        "features_summary": results_dir / "phase3_2e_features_summary.txt",

        "alignment_inventory": interim_dir / "phase3_2e_alignment_inventory.csv",
        "alignment_metadata": interim_dir / "phase3_2e_alignment_metadata.json",
        "alignment_report": results_dir / "phase3_2e_alignment_report.json",
        "alignment_summary": results_dir / "phase3_2e_alignment_summary.txt",
        "lagged_frames_dir": lagged_frames_dir,

        "conditional_model_scores": Path(
            getattr(
                cfg,
                "conditional_model_scores_csv",
                results_dir / "phase3_2e_conditional_model_scores.csv",
            )
        ),
        "conditional_pair_scores": Path(
            getattr(
                cfg,
                "conditional_pair_scores_csv",
                results_dir / "phase3_2e_conditional_pair_scores.csv",
            )
        ),
        "conditional_summary_json": Path(
            getattr(
                cfg,
                "conditional_summary_json",
                results_dir / "phase3_2e_conditional_summary.json",
            )
        ),
        "conditional_summary_txt": Path(
            getattr(
                cfg,
                "conditional_summary_txt",
                results_dir / "phase3_2e_conditional_summary.txt",
            )
        ),
        "conditional_report": results_dir / "phase3_2e_conditional_report.json",

        "control_runs": Path(
            getattr(
                cfg,
                "control_runs_csv",
                results_dir / "phase3_2e_control_runs.csv",
            )
        ),
        "control_pair_scores": Path(
            getattr(
                cfg,
                "control_pair_scores_csv",
                results_dir / "phase3_2e_control_pair_scores.csv",
            )
        ),
        "controls_json": Path(
            getattr(
                cfg,
                "controls_json",
                results_dir / "phase3_2e_controls.json",
            )
        ),
        "controls_summary": Path(
            getattr(
                cfg,
                "controls_summary_txt",
                results_dir / "phase3_2e_controls_summary.txt",
            )
        ),
        "controls_report": results_dir / "phase3_2e_controls_report.json",

        "gates_json": Path(
            getattr(
                cfg,
                "gates_json",
                results_dir / "phase3_2e_gates.json",
            )
        ),
        "multiple_comparison_guard": Path(
            getattr(
                cfg,
                "multiple_comparison_guard_json",
                results_dir / "phase3_2e_multiple_comparison_guard.json",
            )
        ),
        "metrics_table": results_dir / "phase3_2e_metrics_table.csv",
        "metrics_summary": results_dir / "phase3_2e_metrics_summary.txt",
        "metrics_report": results_dir / "phase3_2e_metrics_report.json",

        "gate_sensitivity_report": results_dir / "phase3_2e_gate_sensitivity_report.json",
        "gate_sensitivity_summary": results_dir / "phase3_2e_gate_sensitivity_summary.txt",
        "f7_sensitivity": results_dir / "phase3_2e_f7_subject_sensitivity.csv",
        "f8_sensitivity": results_dir / "phase3_2e_f8_complexity_sensitivity.csv",
        "f12_sensitivity": results_dir / "phase3_2e_f12_epsilon_saturation_sensitivity.csv",

        "diagnostics_dir": diagnostics_dir,
        "diagnostics_report": diagnostics_dir / "phase3_2e_diagnostics_report.json",
        "diagnostics_summary": diagnostics_dir / "phase3_2e_diagnostics_summary.txt",

        "runner_report": results_dir / "phase3_2e_runner_report.json",
        "runner_summary": results_dir / "phase3_2e_runner_summary.txt",
    }


# =========================================================
# Module execution
# =========================================================

def _module_args_for_step(
    step_key: str,
    force_rebuild: bool,
) -> List[str]:
    if not force_rebuild:
        return []

    if step_key == "alignment":
        return [
            "--force-rebuild",
            "--force-rebuild-features",
            "--force-rebuild-windows",
            "--force-rebuild-loader",
        ]

    if step_key == "conditional":
        return [
            "--force-rebuild",
            "--force-rebuild-alignment",
        ]

    if step_key == "controls":
        return [
            "--force-rebuild",
            "--force-rebuild-alignment",
        ]

    if step_key == "metrics":
        return [
            "--force-rebuild-conditional",
            "--force-rebuild-controls",
        ]

    return []


def run_module_as_main(
    module_name: str,
    module_args: Optional[Sequence[str]] = None,
) -> None:
    old_argv = sys.argv[:]

    try:
        sys.argv = [module_name] + list(module_args or [])
        runpy.run_module(module_name, run_name="__main__")

    except SystemExit as exc:
        code = exc.code

        if code not in (0, None):
            raise RuntimeError(
                f"Module {module_name} exited with non-zero status: {code}"
            ) from exc

    finally:
        sys.argv = old_argv


# =========================================================
# Stage payload collector
# =========================================================

def collect_step_payload(step_key: str, cfg: Phase32EConfig, paths: Mapping[str, Path]) -> Dict[str, Any]:
    if step_key == "config":
        return {
            "status": "ok",
            "result": {
                "status": "ok",
                "protocol_only": bool(getattr(cfg, "phase3_2e_protocol_only", True)),
                "interpretation": "Phase III.2E config loaded successfully.",
            },
            "summary": describe_config(cfg),
        }

    payload_path_by_step = {
        "schema": paths["schema_report"],
        "sync_validator": paths["sync_validation_report"],
        "loader": paths["loader_report"],
        "windows": paths["windows_report"],
        "features": paths["features_report"],
        "alignment": paths["alignment_report"],
        "conditional": paths["conditional_report"],
        "controls": paths["controls_json"],
        "metrics": paths["metrics_report"],
        "gate_sensitivity": paths["gate_sensitivity_report"],
        "diagnostics": paths["diagnostics_report"],
    }

    path = payload_path_by_step.get(step_key)

    if not path:
        return {}

    payload = load_json(path)

    if payload:
        return payload

    return {
        "status": "missing_payload",
        "result": {
            "status": "missing_payload",
            "protocol_only": False,
            "interpretation": f"Expected payload not found after step {step_key}.",
        },
        "expected_payload": str(path),
    }


def summarize_step_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "payload_status": _payload_status(payload),
        "protocol_only": _payload_protocol_only(payload),
        "interpretation": _payload_interpretation(payload),
    }


# =========================================================
# Table summaries
# =========================================================

def summarize_tables(paths: Mapping[str, Path]) -> Dict[str, Any]:
    loaded_df = load_csv(paths["loaded_windows"])
    windows_df = load_csv(paths["windows"])
    features_df = load_csv(paths["features"])
    alignment_inventory_df = load_csv(paths["alignment_inventory"])
    model_df = load_csv(paths["conditional_model_scores"])
    pair_df = load_csv(paths["conditional_pair_scores"])
    control_runs_df = load_csv(paths["control_runs"])
    control_pair_df = load_csv(paths["control_pair_scores"])
    metrics_df = load_csv(paths["metrics_table"])
    f7_df = load_csv(paths["f7_sensitivity"])
    f8_df = load_csv(paths["f8_sensitivity"])
    f12_df = load_csv(paths["f12_sensitivity"])

    summary: Dict[str, Any] = {
        "loaded_rows": int(len(loaded_df)),
        "window_rows": int(len(windows_df)),
        "feature_rows": int(len(features_df)),
        "alignment_inventory_rows": int(len(alignment_inventory_df)),
        "conditional_model_rows": int(len(model_df)),
        "conditional_pair_rows": int(len(pair_df)),
        "control_run_rows": int(len(control_runs_df)),
        "control_pair_rows": int(len(control_pair_df)),
        "metrics_table_rows": int(len(metrics_df)),
        "f7_sensitivity_rows": int(len(f7_df)),
        "f8_sensitivity_rows": int(len(f8_df)),
        "f12_sensitivity_rows": int(len(f12_df)),
    }

    if not alignment_inventory_df.empty:
        for col in [
            "n_valid_alignment_rows",
            "n_valid_primary_alignment_rows",
            "n_valid_conditional_rows",
            "n_valid_mc_conditional_rows",
        ]:
            if col in alignment_inventory_df.columns:
                summary[f"alignment_total_{col}"] = int(
                    pd.to_numeric(alignment_inventory_df[col], errors="coerce").fillna(0).sum()
                )

        if "alignment_scope" in alignment_inventory_df.columns:
            summary["alignment_scope_counts"] = (
                alignment_inventory_df["alignment_scope"]
                .fillna("")
                .astype(str)
                .value_counts()
                .to_dict()
            )

        if "lag_direction" in alignment_inventory_df.columns:
            summary["lag_direction_counts"] = (
                alignment_inventory_df["lag_direction"]
                .fillna("")
                .astype(str)
                .value_counts()
                .to_dict()
            )

    if not pair_df.empty and "valid" in pair_df.columns:
        valid_pair_df = pair_df[_as_bool_series(pair_df["valid"])].copy()

        summary["conditional_valid_pair_rows"] = int(len(valid_pair_df))

        if not valid_pair_df.empty and "conditional_lift" in valid_pair_df.columns:
            lift = pd.to_numeric(valid_pair_df["conditional_lift"], errors="coerce").fillna(0.0)
            summary["conditional_positive_lift_rows"] = int((lift > 0.0).sum())
            summary["conditional_max_lift"] = float(lift.max())

        if not valid_pair_df.empty and "primary" in valid_pair_df.columns:
            primary = valid_pair_df[_as_bool_series(valid_pair_df["primary"])].copy()
            summary["conditional_primary_pair_rows"] = int(len(primary))

    return summary


# =========================================================
# Final status extraction
# =========================================================

def extract_final_status(report_payloads: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    diagnostics = report_payloads.get("diagnostics", {})
    metrics = report_payloads.get("metrics", {})
    gate_sensitivity = report_payloads.get("gate_sensitivity", {})

    diagnostics_final = _safe_dict(diagnostics.get("final_interpretation", {}))

    if diagnostics_final:
        return {
            "status": diagnostics_final.get("status", _payload_status(diagnostics)),
            "headline": diagnostics_final.get("headline"),
            "interpretation": diagnostics_final.get("headline"),
            "recommendation": diagnostics_final.get("recommendation"),
            "protocol_only": bool(diagnostics_final.get("protocol_only", False)),
            "can_claim_empirical_cdr": bool(diagnostics_final.get("can_claim_empirical_cdr", False)),
            "source": "diagnostics",
        }

    metrics_final = _safe_dict(metrics.get("final_status", {}))

    if metrics_final:
        return {
            "status": metrics_final.get("status", _payload_status(metrics)),
            "headline": metrics_final.get("status"),
            "interpretation": metrics_final.get("interpretation"),
            "recommendation": metrics_final.get("interpretation"),
            "protocol_only": bool(metrics_final.get("protocol_only", False)),
            "can_claim_empirical_cdr": bool(metrics_final.get("can_claim_empirical_cdr", False)),
            "source": "metrics",
            "passed_gates": metrics_final.get("passed_gates", []),
            "failed_gates": metrics_final.get("failed_gates", []),
            "warning_gates": metrics_final.get("warning_gates", []),
            "not_evaluated_gates": metrics_final.get("not_evaluated_gates", []),
        }

    sensitivity_result = _safe_dict(gate_sensitivity.get("result", {}))

    if sensitivity_result:
        return {
            "status": sensitivity_result.get("status", _payload_status(gate_sensitivity)),
            "headline": sensitivity_result.get("status"),
            "interpretation": sensitivity_result.get("interpretation"),
            "recommendation": sensitivity_result.get("interpretation"),
            "protocol_only": bool(sensitivity_result.get("protocol_only", False)),
            "can_claim_empirical_cdr": bool(sensitivity_result.get("can_claim_empirical_cdr", False)),
            "source": "gate_sensitivity",
        }

    return {
        "status": "unknown",
        "headline": "Runner completed but no final status payload was found.",
        "interpretation": "",
        "recommendation": "",
        "protocol_only": False,
        "can_claim_empirical_cdr": False,
        "source": "fallback",
    }


# =========================================================
# Summary text
# =========================================================

def build_runner_summary_text(report: Mapping[str, Any]) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("CDR Phase III.2E — Runner Summary")
    lines.append("Synchronized EEG–QRNG protocol and gate-sensitivity version")
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

    lines.append("Execution")
    lines.append("-" * 78)
    lines.append(f"started_at: {report.get('started_at')}")
    lines.append(f"finished_at: {report.get('finished_at')}")
    lines.append(f"elapsed_seconds: {report.get('elapsed_seconds')}")
    lines.append(f"force_rebuild: {report.get('options', {}).get('force_rebuild')}")
    lines.append(f"skip_controls: {report.get('options', {}).get('skip_controls')}")
    lines.append(f"skip_diagnostics: {report.get('options', {}).get('skip_diagnostics')}")
    lines.append("")

    lines.append("Scientific guardrail")
    lines.append("-" * 78)
    lines.append(
        "Phase III.2E requires real synchronized EEG + QRNG acquisition timestamps "
        "for empirical interpretation. Protocol-only outputs do not support an "
        "EEG–QRNG coupling claim."
    )
    lines.append("")

    lines.append("Steps")
    lines.append("-" * 78)

    for key, payload in report.get("steps", {}).items():
        lines.append(f"{key}:")
        lines.append(f"  title: {payload.get('title')}")
        lines.append(f"  module: {payload.get('module')}")
        lines.append(f"  status: {payload.get('status')}")
        lines.append(f"  elapsed_seconds: {payload.get('elapsed_seconds')}")
        lines.append(f"  module_args: {payload.get('module_args')}")
        lines.append(f"  payload_status: {payload.get('payload_status')}")
        lines.append(f"  protocol_only: {payload.get('protocol_only')}")

        if payload.get("interpretation"):
            lines.append(f"  interpretation: {payload.get('interpretation')}")

        if payload.get("error"):
            lines.append(f"  error: {payload.get('error')}")

        lines.append("")

    lines.append("Table summary")
    lines.append("-" * 78)

    for key, value in report.get("table_summary", {}).items():
        lines.append(f"{key}: {value}")

    lines.append("")

    lines.append("Output files")
    lines.append("-" * 78)

    for key, value in report.get("outputs", {}).items():
        lines.append(f"{key}: {value}")

    lines.append("")

    final = report.get("final_status", {})

    lines.append("Final status")
    lines.append("-" * 78)
    lines.append(f"status: {final.get('status')}")
    lines.append(f"headline: {final.get('headline')}")
    lines.append(f"protocol_only: {final.get('protocol_only')}")
    lines.append(f"can_claim_empirical_cdr: {final.get('can_claim_empirical_cdr')}")
    lines.append(f"source: {final.get('source')}")
    lines.append(f"interpretation: {final.get('interpretation')}")
    lines.append(f"recommendation: {final.get('recommendation')}")

    if "passed_gates" in final:
        lines.append(f"passed_gates: {final.get('passed_gates')}")
        lines.append(f"failed_gates: {final.get('failed_gates')}")
        lines.append(f"warning_gates: {final.get('warning_gates')}")
        lines.append(f"not_evaluated_gates: {final.get('not_evaluated_gates')}")

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


# =========================================================
# Main pipeline
# =========================================================

def run_phase3_2e_pipeline(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild: bool = False,
    skip_controls: bool = False,
    skip_diagnostics: bool = False,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    np.random.seed(int(getattr(cfg, "numpy_seed", 42)))

    paths = phase3_2e_paths(cfg)

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.interim_dir).mkdir(parents=True, exist_ok=True)
    Path(paths["diagnostics_dir"]).mkdir(parents=True, exist_ok=True)

    runner_report_json = paths["runner_report"]
    runner_summary_txt = paths["runner_summary"]

    started_at = _now_str()
    pipeline_start = time.time()

    report_payloads: Dict[str, Dict[str, Any]] = {}

    report: Dict[str, Any] = {
        "project_name": cfg.project_name,
        "phase": cfg.phase_name,
        "phase_title": cfg.phase_title,
        "version": cfg.version,
        "runner_version": RUNNER_VERSION,
        "module": RUNNER_MODULE,
        "started_at": started_at,
        "finished_at": None,
        "elapsed_seconds": None,
        "options": {
            "force_rebuild": bool(force_rebuild),
            "skip_controls": bool(skip_controls),
            "skip_diagnostics": bool(skip_diagnostics),
        },
        "config": describe_config(cfg),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        "scientific_guardrail": {
            "protocol_only_allowed": bool(getattr(cfg, "allow_protocol_only_without_data", True)),
            "requires_real_synchronized_data_for_claim": bool(
                getattr(cfg, "require_real_synchronized_data_for_claim", True)
            ),
            "synchronized_input_csv": str(paths["synchronized_input_csv"]),
            "interpretation": (
                "Phase III.2E may complete as a protocol implementation without "
                "real synchronized data, but empirical EEG–QRNG claims remain blocked."
            ),
        },
        "steps": {},
        "outputs": {},
        "table_summary": {},
        "payload_status": {},
        "file_status": {},
        "final_status": {},
    }

    total_steps = len(PIPELINE_STEPS)

    for index, (step_key, title, module_name) in enumerate(PIPELINE_STEPS, start=1):
        if skip_controls and step_key == "controls":
            _step_header(index, total_steps, f"{title} — skipped by flag")

            report["steps"][step_key] = {
                "title": title,
                "module": module_name,
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "module_args": [],
                "payload_status": "skipped",
                "protocol_only": False,
                "interpretation": "Controls skipped by CLI flag.",
            }

            continue

        if skip_diagnostics and step_key == "diagnostics":
            _step_header(index, total_steps, f"{title} — skipped by flag")

            report["steps"][step_key] = {
                "title": title,
                "module": module_name,
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "module_args": [],
                "payload_status": "skipped",
                "protocol_only": False,
                "interpretation": "Diagnostics skipped by CLI flag.",
            }

            continue

        step_start = time.time()
        _step_header(index, total_steps, title)

        module_args = _module_args_for_step(
            step_key=step_key,
            force_rebuild=bool(force_rebuild),
        )

        try:
            run_module_as_main(module_name, module_args=module_args)

            payload = collect_step_payload(step_key, cfg, paths)
            report_payloads[step_key] = payload

            payload_summary = summarize_step_payload(payload)

            report["steps"][step_key] = {
                "title": title,
                "module": module_name,
                "status": "ok",
                "elapsed_seconds": _elapsed(step_start),
                "module_args": module_args,
                **payload_summary,
            }

            report["payload_status"][step_key] = payload_summary

        except Exception as exc:
            error_payload = {
                "title": title,
                "module": module_name,
                "status": "error",
                "elapsed_seconds": _elapsed(step_start),
                "module_args": module_args,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

            report["steps"][step_key] = error_payload
            report["finished_at"] = _now_str()
            report["elapsed_seconds"] = _elapsed(pipeline_start)
            report["table_summary"] = summarize_tables(paths)
            report["file_status"] = {
                key: _file_status(path)
                for key, path in paths.items()
                if key not in {"diagnostics_dir", "lagged_frames_dir"}
            }
            report["final_status"] = {
                "status": "runner_error",
                "headline": f"Phase III.2E runner failed at step: {step_key}",
                "interpretation": str(exc),
                "recommendation": "Inspect traceback and rerun after fixing the failing module.",
                "protocol_only": False,
                "can_claim_empirical_cdr": False,
                "source": "runner",
            }

            report["outputs"]["runner_report_json"] = str(runner_report_json)
            report["outputs"]["runner_summary_txt"] = str(runner_summary_txt)

            save_json(runner_report_json, report)

            with open(runner_summary_txt, "w", encoding="utf-8") as f:
                f.write(build_runner_summary_text(report))

            raise

        # Save provisional runner state after each step.
        report["finished_at"] = _now_str()
        report["elapsed_seconds"] = _elapsed(pipeline_start)
        report["table_summary"] = summarize_tables(paths)
        report["outputs"]["runner_report_json"] = str(runner_report_json)
        report["outputs"]["runner_summary_txt"] = str(runner_summary_txt)

        save_json(runner_report_json, report)

    # -----------------------------------------------------
    # Final consolidation
    # -----------------------------------------------------
    for step_key, _, _ in PIPELINE_STEPS:
        if step_key not in report_payloads:
            payload = collect_step_payload(step_key, cfg, paths)
            report_payloads[step_key] = payload

    report["table_summary"] = summarize_tables(paths)

    report["file_status"] = {
        key: _file_status(path)
        for key, path in paths.items()
        if key not in {"diagnostics_dir", "lagged_frames_dir"}
    }

    report["outputs"].update(
        {
            "synchronized_input_csv": str(paths["synchronized_input_csv"]),

            "schema_report": str(paths["schema_report"]),
            "sync_validation_report": str(paths["sync_validation_report"]),
            "loader_report": str(paths["loader_report"]),
            "windows_report": str(paths["windows_report"]),
            "features_report": str(paths["features_report"]),
            "alignment_report": str(paths["alignment_report"]),
            "conditional_report": str(paths["conditional_report"]),
            "controls_json": str(paths["controls_json"]),
            "metrics_report": str(paths["metrics_report"]),
            "gate_sensitivity_report": str(paths["gate_sensitivity_report"]),
            "diagnostics_report": str(paths["diagnostics_report"]),

            "alignment_inventory": str(paths["alignment_inventory"]),
            "conditional_pair_scores": str(paths["conditional_pair_scores"]),
            "conditional_model_scores": str(paths["conditional_model_scores"]),
            "control_runs": str(paths["control_runs"]),
            "control_pair_scores": str(paths["control_pair_scores"]),
            "metrics_table": str(paths["metrics_table"]),

            "f7_sensitivity": str(paths["f7_sensitivity"]),
            "f8_sensitivity": str(paths["f8_sensitivity"]),
            "f12_sensitivity": str(paths["f12_sensitivity"]),

            "runner_report_json": str(runner_report_json),
            "runner_summary_txt": str(runner_summary_txt),
        }
    )

    report["final_status"] = extract_final_status(report_payloads)

    report["finished_at"] = _now_str()
    report["elapsed_seconds"] = _elapsed(pipeline_start)

    save_json(runner_report_json, report)

    with open(runner_summary_txt, "w", encoding="utf-8") as f:
        f.write(build_runner_summary_text(report))

    save_json(runner_report_json, report)

    _final_header("Phase III.2E runner completed")
    print("Synchronized EEG–QRNG protocol and gate-sensitivity version")
    print("=" * 78)

    final = report.get("final_status", {})
    table = report.get("table_summary", {})

    print(f"status: {final.get('status')}")
    print(f"protocol_only: {final.get('protocol_only')}")
    print(f"can_claim_empirical_cdr: {final.get('can_claim_empirical_cdr')}")
    print(f"source: {final.get('source')}")
    print(f"headline: {final.get('headline')}")
    print(f"recommendation: {final.get('recommendation')}")
    print("")
    print("Table summary:")
    print(f"loaded rows: {table.get('loaded_rows')}")
    print(f"window rows: {table.get('window_rows')}")
    print(f"feature rows: {table.get('feature_rows')}")
    print(f"alignment inventory rows: {table.get('alignment_inventory_rows')}")
    print(f"conditional pair rows: {table.get('conditional_pair_rows')}")
    print(f"control run rows: {table.get('control_run_rows')}")
    print(f"metrics table rows: {table.get('metrics_table_rows')}")
    print(f"F7 sensitivity rows: {table.get('f7_sensitivity_rows')}")
    print(f"F8 sensitivity rows: {table.get('f8_sensitivity_rows')}")
    print(f"F12 sensitivity rows: {table.get('f12_sensitivity_rows')}")
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
        description="Run the full CDR Phase III.2E synchronized EEG–QRNG pipeline."
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding stages where supported.",
    )

    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip negative controls and reuse existing outputs if present.",
    )

    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip final diagnostics generation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    run_phase3_2e_pipeline(
        cfg=cfg,
        force_rebuild=bool(args.force_rebuild),
        skip_controls=bool(args.skip_controls),
        skip_diagnostics=bool(args.skip_diagnostics),
    )


if __name__ == "__main__":
    main()