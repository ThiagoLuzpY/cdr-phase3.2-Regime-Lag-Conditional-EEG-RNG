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
# Safe-resume synchronized EEG–QRNG protocol runner
# =========================================================
#
# Pipeline:
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
#
# Safety:
#
#   --safe-resume reuses the existing Conditional and Controls outputs.
#   It never executes those two expensive stages.
#
#   --skip-conditional and --skip-controls are explicit equivalents for
#   reusing each heavy stage independently.
#
#   --force-rebuild can execute expensive stages only together with
#   --confirm-heavy-rebuild. This prevents accidental multi-hour reruns.
#
# Scientific interpretation:
#
#   Operational pipeline completion is independent from gate outcome.
#   A BIC-limited primary lead is a completed scientific result, not a
#   runner failure. Surrogate provenance still blocks an empirical claim.
# =========================================================

RUNNER_MODULE = "phase3_2e_runner"
RUNNER_VERSION = (
    "phase3_2e_runner_v2_safe_resume_"
    "primary_lead_bic_limited_aware"
)

PIPELINE_STEPS: List[Tuple[str, str, str]] = [
    ("config", "Config check", "config.phase3_2e_config"),
    ("schema", "Schema validation", "src.phase3_2e_schema"),
    (
        "sync_validator",
        "Synchronization validation",
        "src.phase3_2e_sync_validator",
    ),
    ("loader", "Synchronized loader", "src.phase3_2e_loader"),
    ("windows", "Window builder", "src.phase3_2e_windows"),
    ("features", "Feature builder", "src.phase3_2e_features"),
    ("alignment", "Lagged alignment", "src.phase3_2e_alignment"),
    ("conditional", "Conditional CDR", "src.phase3_2e_conditional"),
    ("controls", "Negative controls", "src.phase3_2e_controls"),
    ("metrics", "Metrics and official gates", "src.phase3_2e_metrics"),
    (
        "gate_sensitivity",
        "F7/F8/F12 gate sensitivity",
        "src.phase3_2e_gate_sensitivity",
    ),
    ("diagnostics", "Final diagnostics", "src.phase3_2e_diagnostics"),
]

HEAVY_STEPS = {"conditional", "controls"}


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

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
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


def save_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return {}

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        return (
            payload
            if isinstance(payload, dict)
            else {}
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


def load_csv(
    path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            nrows=nrows,
        )

    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame()


def _safe_dict(value: Any) -> Dict[str, Any]:
    return (
        value
        if isinstance(value, dict)
        else {}
    )


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if pd.isna(value):
            return default

        return int(value)

    except Exception:
        return default


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if pd.isna(value):
            return default

        number = float(value)

        return (
            number
            if np.isfinite(number)
            else default
        )

    except Exception:
        return default


def _safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
        "y",
        "sim",
        "t",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
        "nao",
        "não",
        "f",
    }:
        return False

    return default


def _as_bool_series(
    series: pd.Series,
) -> pd.Series:
    if series.empty:
        return pd.Series(
            [],
            index=series.index,
            dtype=bool,
        )

    if pd.api.types.is_bool_dtype(series):
        return (
            series
            .fillna(False)
            .astype(bool)
        )

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
                "sim",
                "t",
            }
        )
    )


def _step_header(
    index: int,
    total: int,
    title: str,
) -> None:
    print("\n" + "=" * 78)
    print(f"Step {index}/{total} — {title}")
    print("=" * 78)


def _final_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _first_existing(
    candidates: Sequence[Path],
) -> Path:
    for path in candidates:
        if Path(path).exists():
            return Path(path)

    return Path(candidates[0])


def _file_status(path: Path) -> Dict[str, Any]:
    path = Path(path)

    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": (
            int(path.stat().st_size)
            if path.exists()
            else 0
        ),
    }


def _payload_status(
    payload: Mapping[str, Any],
) -> str:
    if not payload:
        return "missing"

    if payload.get("status") is not None:
        return str(payload.get("status"))

    result = _safe_dict(
        payload.get("result")
    )

    if result.get("status") is not None:
        return str(result.get("status"))

    final = _safe_dict(
        payload.get("final_status")
    )

    if final.get("status") is not None:
        return str(final.get("status"))

    return "unknown"


def _payload_protocol_only(
    payload: Mapping[str, Any],
) -> bool:
    if not payload:
        return False

    result = _safe_dict(
        payload.get("result")
    )

    final = _safe_dict(
        payload.get("final_status")
    )

    return bool(
        result.get(
            "protocol_only",
            False,
        )
        or final.get(
            "protocol_only",
            False,
        )
        or str(
            payload.get(
                "status",
                "",
            )
        ).lower()
        == "pending_synchronized_data"
    )


def _payload_interpretation(
    payload: Mapping[str, Any],
) -> str:
    if not payload:
        return ""

    result = _safe_dict(
        payload.get("result")
    )

    final = _safe_dict(
        payload.get("final_status")
    )

    for value in (
        result.get("interpretation"),
        final.get("interpretation"),
        payload.get("interpretation"),
    ):
        if value:
            return str(value)

    return ""


# =========================================================
# Path registry
# =========================================================

def phase3_2e_paths(
    cfg: Phase32EConfig,
) -> Dict[str, Path]:
    results_dir = Path(
        cfg.results_dir
    )

    interim_dir = Path(
        cfg.interim_dir
    )

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
            (
                Path("data")
                / "raw"
                / "phase3_2e"
                / "synchronized"
                / "phase3_2e_synchronized_input.csv"
            ),
        )
    )

    lagged_frames_dir = Path(
        getattr(
            cfg,
            "lagged_frames_dir",
            interim_dir / "lagged_frames",
        )
    )

    sync_report = _first_existing(
        [
            results_dir
            / "phase3_2e_sync_report.json",

            results_dir
            / "phase3_2e_sync_validation_report.json",
        ]
    )

    sync_summary = _first_existing(
        [
            results_dir
            / "phase3_2e_sync_summary.txt",

            results_dir
            / "phase3_2e_sync_validation_summary.txt",
        ]
    )

    return {
        "synchronized_input_csv":
            synchronized_input_csv,

        "schema_report":
            results_dir
            / "phase3_2e_schema_report.json",

        "schema_summary":
            results_dir
            / "phase3_2e_schema_summary.txt",

        "sync_validation_report":
            sync_report,

        "sync_validation_summary":
            sync_summary,

        "loaded_windows":
            interim_dir
            / "phase3_2e_loaded_synchronized_windows.csv",

        "loader_metadata":
            interim_dir
            / "phase3_2e_loader_metadata.json",

        "loader_report":
            results_dir
            / "phase3_2e_loader_report.json",

        "loader_summary":
            results_dir
            / "phase3_2e_loader_summary.txt",

        "windows":
            interim_dir
            / "phase3_2e_windows.csv",

        "window_inventory":
            interim_dir
            / "phase3_2e_window_inventory.csv",

        "windows_report":
            results_dir
            / "phase3_2e_windows_report.json",

        "windows_summary":
            results_dir
            / "phase3_2e_windows_summary.txt",

        "features":
            interim_dir
            / "phase3_2e_features.csv",

        "feature_specs":
            interim_dir
            / "phase3_2e_feature_specs.json",

        "features_report":
            results_dir
            / "phase3_2e_features_report.json",

        "features_summary":
            results_dir
            / "phase3_2e_features_summary.txt",

        "alignment_inventory":
            interim_dir
            / "phase3_2e_alignment_inventory.csv",

        "alignment_metadata":
            interim_dir
            / "phase3_2e_alignment_metadata.json",

        "alignment_report":
            results_dir
            / "phase3_2e_alignment_report.json",

        "alignment_summary":
            results_dir
            / "phase3_2e_alignment_summary.txt",

        "lagged_frames_dir":
            lagged_frames_dir,

        "conditional_model_scores":
            Path(
                getattr(
                    cfg,
                    "conditional_model_scores_csv",
                    results_dir
                    / "phase3_2e_conditional_model_scores.csv",
                )
            ),

        "conditional_pair_scores":
            Path(
                getattr(
                    cfg,
                    "conditional_pair_scores_csv",
                    results_dir
                    / "phase3_2e_conditional_pair_scores.csv",
                )
            ),

        "conditional_summary_json":
            Path(
                getattr(
                    cfg,
                    "conditional_summary_json",
                    results_dir
                    / "phase3_2e_conditional_summary.json",
                )
            ),

        "conditional_summary_txt":
            Path(
                getattr(
                    cfg,
                    "conditional_summary_txt",
                    results_dir
                    / "phase3_2e_conditional_summary.txt",
                )
            ),

        "conditional_report":
            results_dir
            / "phase3_2e_conditional_report.json",

        "control_runs":
            Path(
                getattr(
                    cfg,
                    "control_runs_csv",
                    results_dir
                    / "phase3_2e_control_runs.csv",
                )
            ),

        "control_pair_scores":
            Path(
                getattr(
                    cfg,
                    "control_pair_scores_csv",
                    results_dir
                    / "phase3_2e_control_pair_scores.csv",
                )
            ),

        "controls_json":
            Path(
                getattr(
                    cfg,
                    "controls_json",
                    results_dir
                    / "phase3_2e_controls.json",
                )
            ),

        "controls_summary":
            Path(
                getattr(
                    cfg,
                    "controls_summary_txt",
                    results_dir
                    / "phase3_2e_controls_summary.txt",
                )
            ),

        "controls_report":
            results_dir
            / "phase3_2e_controls_report.json",

        "gates_json":
            Path(
                getattr(
                    cfg,
                    "gates_json",
                    results_dir
                    / "phase3_2e_gates.json",
                )
            ),

        "multiple_comparison_guard":
            Path(
                getattr(
                    cfg,
                    "multiple_comparison_guard_json",
                    results_dir
                    / "phase3_2e_multiple_comparison_guard.json",
                )
            ),

        "metrics_table":
            results_dir
            / "phase3_2e_metrics_table.csv",

        "metrics_summary":
            results_dir
            / "phase3_2e_metrics_summary.txt",

        "metrics_report":
            results_dir
            / "phase3_2e_metrics_report.json",

        "gate_sensitivity_report":
            results_dir
            / "phase3_2e_gate_sensitivity_report.json",

        "gate_sensitivity_summary":
            results_dir
            / "phase3_2e_gate_sensitivity_summary.txt",

        "f7_sensitivity":
            results_dir
            / "phase3_2e_f7_subject_sensitivity.csv",

        "f8_sensitivity":
            results_dir
            / "phase3_2e_f8_complexity_sensitivity.csv",

        "f12_sensitivity":
            results_dir
            / "phase3_2e_f12_epsilon_saturation_sensitivity.csv",

        "diagnostics_dir":
            diagnostics_dir,

        "diagnostics_report":
            diagnostics_dir
            / "phase3_2e_diagnostics_report.json",

        "diagnostics_summary":
            diagnostics_dir
            / "phase3_2e_diagnostics_summary.txt",

        "runner_report":
            results_dir
            / "phase3_2e_runner_report.json",

        "runner_summary":
            results_dir
            / "phase3_2e_runner_summary.txt",
    }


# =========================================================
# Existing-artifact validation for safe reuse
# =========================================================

def _validate_nonempty_csv(
    path: Path,
) -> Dict[str, Any]:
    status = _file_status(path)

    if (
        not status["exists"]
        or status["size_bytes"] <= 0
    ):
        return {
            **status,
            "valid": False,
            "rows": 0,
            "reason": "missing_or_empty",
        }

    frame = load_csv(path)

    return {
        **status,
        "valid": not frame.empty,
        "rows": int(len(frame)),
        "reason": (
            "ok"
            if not frame.empty
            else "csv_has_no_rows"
        ),
    }


def _validate_nonempty_json(
    path: Path,
) -> Dict[str, Any]:
    status = _file_status(path)
    payload = load_json(path)

    return {
        **status,
        "valid": bool(payload),
        "status": (
            _payload_status(payload)
            if payload
            else "missing"
        ),
        "reason": (
            "ok"
            if payload
            else "missing_or_invalid_json"
        ),
    }


def validate_heavy_artifacts(
    cfg: Phase32EConfig,
    paths: Optional[Mapping[str, Path]] = None,
) -> Dict[str, Any]:
    paths = dict(
        paths
        or phase3_2e_paths(cfg)
    )

    conditional_files = {
        "conditional_model_scores":
            _validate_nonempty_csv(
                paths[
                    "conditional_model_scores"
                ]
            ),

        "conditional_pair_scores":
            _validate_nonempty_csv(
                paths[
                    "conditional_pair_scores"
                ]
            ),

        "conditional_summary_json":
            _validate_nonempty_json(
                paths[
                    "conditional_summary_json"
                ]
            ),

        "conditional_report":
            _validate_nonempty_json(
                paths[
                    "conditional_report"
                ]
            ),
    }

    control_files = {
        "control_runs":
            _validate_nonempty_csv(
                paths["control_runs"]
            ),

        "control_pair_scores":
            _validate_nonempty_csv(
                paths[
                    "control_pair_scores"
                ]
            ),

        "controls_json":
            _validate_nonempty_json(
                paths["controls_json"]
            ),

        "controls_report":
            _validate_nonempty_json(
                paths["controls_report"]
            ),
    }

    conditional_valid = all(
        item["valid"]
        for item in conditional_files.values()
    )

    controls_valid = all(
        item["valid"]
        for item in control_files.values()
    )

    controls_payload = load_json(
        paths["controls_json"]
    )

    controls_result = _safe_dict(
        controls_payload.get("result")
    )

    controls_summary = _safe_dict(
        controls_payload.get("summary")
    )

    controls_passed = bool(
        controls_summary.get(
            "passed",
            controls_result.get(
                "overall_passed",
                False,
            ),
        )
    )

    return {
        "conditional": {
            "valid": conditional_valid,
            "files": conditional_files,
        },

        "controls": {
            "valid": controls_valid,
            "passed": controls_passed,
            "files": control_files,
        },

        "safe_resume_ready": bool(
            conditional_valid
            and controls_valid
        ),
    }


def _required_artifacts_for_reuse(
    step_key: str,
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    if step_key not in HEAVY_STEPS:
        raise ValueError(
            "Step is not reusable as a heavy "
            f"stage: {step_key}"
        )

    section = _safe_dict(
        validation.get(step_key)
    )

    if not section.get(
        "valid",
        False,
    ):
        invalid = [
            name
            for name, item in _safe_dict(
                section.get("files")
            ).items()
            if not _safe_dict(item).get(
                "valid",
                False,
            )
        ]

        raise RuntimeError(
            f"Cannot reuse {step_key}: required "
            "existing artifacts are missing or "
            f"invalid: {invalid}"
        )

    return section


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

    # Metrics, gate sensitivity and diagnostics
    # must never trigger Conditional or Controls.
    return []


def run_module_as_main(
    module_name: str,
    module_args: Optional[
        Sequence[str]
    ] = None,
) -> None:
    old_argv = sys.argv[:]

    try:
        sys.argv = [
            module_name,
            *list(module_args or []),
        ]

        runpy.run_module(
            module_name,
            run_name="__main__",
        )

    except SystemExit as exc:
        if exc.code not in (
            0,
            None,
        ):
            raise RuntimeError(
                f"Module {module_name} exited "
                "with non-zero status: "
                f"{exc.code}"
            ) from exc

    finally:
        sys.argv = old_argv


# =========================================================
# Stage payload collector
# =========================================================

def collect_step_payload(
    step_key: str,
    cfg: Phase32EConfig,
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    if step_key == "config":
        return {
            "status": "ok",

            "result": {
                "status": "ok",

                "protocol_only": bool(
                    getattr(
                        cfg,
                        "phase3_2e_protocol_only",
                        False,
                    )
                ),

                "interpretation": (
                    "Phase III.2E config "
                    "loaded successfully."
                ),
            },

            "summary":
                describe_config(cfg),
        }

    payload_path_by_step = {
        "schema":
            paths["schema_report"],

        "sync_validator":
            paths["sync_validation_report"],

        "loader":
            paths["loader_report"],

        "windows":
            paths["windows_report"],

        "features":
            paths["features_report"],

        "alignment":
            paths["alignment_report"],

        "conditional":
            paths["conditional_report"],

        "controls":
            paths["controls_json"],

        "metrics":
            paths["metrics_report"],

        "gate_sensitivity":
            paths["gate_sensitivity_report"],

        "diagnostics":
            paths["diagnostics_report"],
    }

    path = payload_path_by_step.get(
        step_key
    )

    if path is None:
        return {}

    payload = load_json(path)

    if payload:
        return payload

    return {
        "status": "missing_payload",

        "result": {
            "status": "missing_payload",

            "protocol_only": False,

            "interpretation": (
                "Expected payload not found "
                f"after step {step_key}."
            ),
        },

        "expected_payload": str(path),
    }


def summarize_step_payload(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "payload_status":
            _payload_status(payload),

        "protocol_only":
            _payload_protocol_only(payload),

        "interpretation":
            _payload_interpretation(payload),
    }


# =========================================================
# Table summaries
# =========================================================

def _primary_definition(
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    windows = tuple(
        int(value)
        for value in getattr(
            cfg,
            "primary_test_window_seconds",
            getattr(
                cfg,
                "primary_window_seconds",
                (),
            ),
        )
    )

    lags = tuple(
        int(value)
        for value in getattr(
            cfg,
            "primary_test_lags_windows",
            getattr(
                cfg,
                "primary_lags_windows",
                (),
            ),
        )
    )

    pairs = tuple(
        (
            str(baseline),
            str(augmented),
        )
        for baseline, augmented in getattr(
            cfg,
            "primary_test_pairs",
            getattr(
                cfg,
                "primary_conditional_pairs",
                (),
            ),
        )
    )

    return {
        "windows": windows,
        "lags": lags,
        "pairs": pairs,
    }


def _effective_primary_mask(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.Series:
    if pair_df.empty:
        return pd.Series(
            [],
            index=pair_df.index,
            dtype=bool,
        )

    required = {
        "window_seconds",
        "lag_windows",
        "baseline_model",
        "augmented_model",
    }

    if not required.issubset(
        pair_df.columns
    ):
        if "primary" in pair_df.columns:
            return _as_bool_series(
                pair_df["primary"]
            )

        return pd.Series(
            False,
            index=pair_df.index,
        )

    definition = _primary_definition(cfg)

    windows = (
        pd.to_numeric(
            pair_df["window_seconds"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )

    lags = (
        pd.to_numeric(
            pair_df["lag_windows"],
            errors="coerce",
        )
        .fillna(999999)
        .astype(int)
    )

    pair_values = list(
        zip(
            pair_df[
                "baseline_model"
            ].astype(str),

            pair_df[
                "augmented_model"
            ].astype(str),
        )
    )

    primary_pair_set = set(
        definition["pairs"]
    )

    pair_mask = pd.Series(
        [
            pair in primary_pair_set
            for pair in pair_values
        ],
        index=pair_df.index,
        dtype=bool,
    )

    return (
        windows.isin(
            definition["windows"]
        )
        & lags.isin(
            definition["lags"]
        )
        & pair_mask
    )


def summarize_tables(
    paths: Mapping[str, Path],
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    loaded_df = load_csv(
        paths["loaded_windows"]
    )

    windows_df = load_csv(
        paths["windows"]
    )

    features_df = load_csv(
        paths["features"]
    )

    alignment_df = load_csv(
        paths["alignment_inventory"]
    )

    model_df = load_csv(
        paths["conditional_model_scores"]
    )

    pair_df = load_csv(
        paths["conditional_pair_scores"]
    )

    control_runs_df = load_csv(
        paths["control_runs"]
    )

    control_pair_df = load_csv(
        paths["control_pair_scores"]
    )

    metrics_df = load_csv(
        paths["metrics_table"]
    )

    f7_df = load_csv(
        paths["f7_sensitivity"]
    )

    f8_df = load_csv(
        paths["f8_sensitivity"]
    )

    f12_df = load_csv(
        paths["f12_sensitivity"]
    )

    summary: Dict[str, Any] = {
        "loaded_rows":
            int(len(loaded_df)),

        "window_rows":
            int(len(windows_df)),

        "feature_rows":
            int(len(features_df)),

        "alignment_inventory_rows":
            int(len(alignment_df)),

        "conditional_model_rows":
            int(len(model_df)),

        "conditional_pair_rows":
            int(len(pair_df)),

        "control_run_rows":
            int(len(control_runs_df)),

        "control_pair_rows":
            int(len(control_pair_df)),

        "metrics_table_rows":
            int(len(metrics_df)),

        "f7_sensitivity_rows":
            int(len(f7_df)),

        "f8_sensitivity_rows":
            int(len(f8_df)),

        "f12_sensitivity_rows":
            int(len(f12_df)),
    }

    if not alignment_df.empty:
        for column in (
            "n_valid_alignment_rows",
            "n_valid_primary_alignment_rows",
            "n_valid_conditional_rows",
            "n_valid_mc_conditional_rows",
        ):
            if column in alignment_df.columns:
                summary[
                    f"alignment_total_{column}"
                ] = int(
                    pd.to_numeric(
                        alignment_df[column],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                )

        for column, key in (
            (
                "alignment_scope",
                "alignment_scope_counts",
            ),
            (
                "lag_direction",
                "lag_direction_counts",
            ),
        ):
            if column in alignment_df.columns:
                summary[key] = (
                    alignment_df[column]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .to_dict()
                )

    if not pair_df.empty:
        valid_pair_df = (
            pair_df[
                _as_bool_series(
                    pair_df["valid"]
                )
            ].copy()
            if "valid" in pair_df.columns
            else pair_df.copy()
        )

        summary[
            "conditional_valid_pair_rows"
        ] = int(
            len(valid_pair_df)
        )

        if not valid_pair_df.empty:
            primary_mask = (
                _effective_primary_mask(
                    valid_pair_df,
                    cfg,
                )
            )

            primary_df = valid_pair_df[
                primary_mask
            ].copy()

            exploratory_df = valid_pair_df[
                ~primary_mask
            ].copy()

            lift = pd.to_numeric(
                valid_pair_df.get(
                    "conditional_lift",
                    pd.Series(
                        0.0,
                        index=valid_pair_df.index,
                    ),
                ),
                errors="coerce",
            ).fillna(0.0)

            summary[
                "conditional_primary_pair_rows"
            ] = int(
                len(primary_df)
            )

            summary[
                "conditional_exploratory_pair_rows"
            ] = int(
                len(exploratory_df)
            )

            summary[
                "conditional_positive_lift_rows"
            ] = int(
                (lift > 0.0).sum()
            )

            primary_lift = pd.to_numeric(
                primary_df.get(
                    "conditional_lift",
                    pd.Series(
                        0.0,
                        index=primary_df.index,
                    ),
                ),
                errors="coerce",
            ).fillna(0.0)

            summary[
                "conditional_primary_positive_lift_rows"
            ] = int(
                (primary_lift > 0.0).sum()
            )

            summary[
                "conditional_max_lift"
            ] = float(
                lift.max()
            )

    return summary


# =========================================================
# Final scientific and operational status
# =========================================================

def extract_final_status(
    report_payloads: Mapping[
        str,
        Dict[str, Any],
    ],
) -> Dict[str, Any]:
    diagnostics = report_payloads.get(
        "diagnostics",
        {},
    )

    metrics = report_payloads.get(
        "metrics",
        {},
    )

    sensitivity = report_payloads.get(
        "gate_sensitivity",
        {},
    )

    diagnostics_final = _safe_dict(
        diagnostics.get(
            "final_interpretation"
        )
    )

    diagnostics_conditional = _safe_dict(
        diagnostics.get(
            "conditional_diagnostics"
        )
    )

    if diagnostics_final:
        scientific_status = (
            diagnostics_final.get(
                "status",
                _payload_status(
                    diagnostics
                ),
            )
        )

        classification = (
            diagnostics_final.get(
                "scientific_classification",
                scientific_status,
            )
        )

        candidate_detected = bool(
            diagnostics_conditional.get(
                "localized_predictive_support",
                False,
            )
            or _safe_int(
                diagnostics_conditional.get(
                    "n_primary_positive_lift_rows",
                    0,
                )
            )
            > 0
        )

        return {
            "status":
                "pipeline_completed",

            "operational_status":
                "pipeline_completed",

            "scientific_status":
                scientific_status,

            "scientific_classification":
                classification,

            "candidate_detected":
                candidate_detected,

            "headline":
                diagnostics_final.get(
                    "headline"
                ),

            "interpretation":
                diagnostics_final.get(
                    "headline"
                ),

            "recommendation":
                diagnostics_final.get(
                    "recommendation"
                ),

            "protocol_only":
                bool(
                    diagnostics_final.get(
                        "protocol_only",
                        False,
                    )
                ),

            "surrogate":
                bool(
                    diagnostics_final.get(
                        "surrogate",
                        False,
                    )
                ),

            "can_claim_empirical_cdr":
                bool(
                    diagnostics_final.get(
                        "can_claim_empirical_cdr",
                        False,
                    )
                ),

            "failed_gates":
                diagnostics_final.get(
                    "failed_gates",
                    [],
                ),

            "warning_gates":
                diagnostics_final.get(
                    "warning_gates",
                    [],
                ),

            "not_evaluated_gates":
                diagnostics_final.get(
                    "not_evaluated_gates",
                    [],
                ),

            "source":
                "diagnostics",
        }

    metrics_final = _safe_dict(
        metrics.get(
            "final_status"
        )
    )

    if metrics_final:
        scientific_status = (
            metrics_final.get(
                "status",
                _payload_status(metrics),
            )
        )

        return {
            "status":
                "pipeline_completed",

            "operational_status":
                "pipeline_completed",

            "scientific_status":
                scientific_status,

            "scientific_classification":
                scientific_status,

            "candidate_detected":
                "lead"
                in str(
                    scientific_status
                ),

            "headline":
                scientific_status,

            "interpretation":
                metrics_final.get(
                    "interpretation"
                ),

            "recommendation":
                metrics_final.get(
                    "interpretation"
                ),

            "protocol_only":
                bool(
                    metrics_final.get(
                        "protocol_only",
                        False,
                    )
                ),

            "surrogate":
                not bool(
                    metrics_final.get(
                        "can_claim_empirical_cdr",
                        False,
                    )
                ),

            "can_claim_empirical_cdr":
                bool(
                    metrics_final.get(
                        "can_claim_empirical_cdr",
                        False,
                    )
                ),

            "passed_gates":
                metrics_final.get(
                    "passed_gates",
                    [],
                ),

            "failed_gates":
                metrics_final.get(
                    "failed_gates",
                    [],
                ),

            "warning_gates":
                metrics_final.get(
                    "warning_gates",
                    [],
                ),

            "not_evaluated_gates":
                metrics_final.get(
                    "not_evaluated_gates",
                    [],
                ),

            "source":
                "metrics",
        }

    sensitivity_result = _safe_dict(
        sensitivity.get("result")
    )

    if sensitivity_result:
        scientific_status = (
            sensitivity_result.get(
                "status",
                _payload_status(
                    sensitivity
                ),
            )
        )

        return {
            "status":
                "pipeline_completed",

            "operational_status":
                "pipeline_completed",

            "scientific_status":
                scientific_status,

            "scientific_classification":
                scientific_status,

            "candidate_detected":
                False,

            "headline":
                scientific_status,

            "interpretation":
                sensitivity_result.get(
                    "interpretation"
                ),

            "recommendation":
                sensitivity_result.get(
                    "interpretation"
                ),

            "protocol_only":
                bool(
                    sensitivity_result.get(
                        "protocol_only",
                        False,
                    )
                ),

            "surrogate":
                True,

            "can_claim_empirical_cdr":
                bool(
                    sensitivity_result.get(
                        "can_claim_empirical_cdr",
                        False,
                    )
                ),

            "source":
                "gate_sensitivity",
        }

    return {
        "status":
            "pipeline_completed_without_final_payload",

        "operational_status":
            "pipeline_completed",

        "scientific_status":
            "unknown",

        "scientific_classification":
            "unknown",

        "candidate_detected":
            False,

        "headline": (
            "Runner completed but no final "
            "scientific payload was found."
        ),

        "interpretation":
            "",

        "recommendation": (
            "Inspect metrics and diagnostics "
            "output files."
        ),

        "protocol_only":
            False,

        "surrogate":
            False,

        "can_claim_empirical_cdr":
            False,

        "source":
            "fallback",
    }


# =========================================================
# Summary text
# =========================================================

def build_runner_summary_text(
    report: Mapping[str, Any],
) -> str:
    lines: List[str] = [
        "=" * 78,

        "CDR Phase III.2E — Runner Summary",

        "Safe-resume synchronized "
        "EEG–QRNG protocol",

        "=" * 78,

        "",

        "Project",

        "-" * 78,

        (
            "project_name: "
            f"{report.get('project_name')}"
        ),

        (
            "phase: "
            f"{report.get('phase')}"
        ),

        (
            "phase_title: "
            f"{report.get('phase_title')}"
        ),

        (
            "version: "
            f"{report.get('version')}"
        ),

        (
            "runner_version: "
            f"{report.get('runner_version')}"
        ),

        "",

        "Execution",

        "-" * 78,

        (
            "started_at: "
            f"{report.get('started_at')}"
        ),

        (
            "finished_at: "
            f"{report.get('finished_at')}"
        ),

        (
            "elapsed_seconds: "
            f"{report.get('elapsed_seconds')}"
        ),

        (
            "force_rebuild: "
            f"{report.get('options', {}).get('force_rebuild')}"
        ),

        (
            "safe_resume: "
            f"{report.get('options', {}).get('safe_resume')}"
        ),

        (
            "reuse_conditional: "
            f"{report.get('options', {}).get('reuse_conditional')}"
        ),

        (
            "reuse_controls: "
            f"{report.get('options', {}).get('reuse_controls')}"
        ),

        (
            "skip_diagnostics: "
            f"{report.get('options', {}).get('skip_diagnostics')}"
        ),

        "",

        "Scientific guardrail",

        "-" * 78,

        (
            "Real synchronized EEG + QRNG acquisition "
            "timestamps are required for an empirical "
            "claim. A surrogate run may validate the "
            "pipeline and identify a primary candidate, "
            "but cannot be presented as empirical "
            "EEG–QRNG evidence."
        ),

        "",

        "Steps",

        "-" * 78,
    ]

    for key, payload in report.get(
        "steps",
        {},
    ).items():
        lines.extend(
            [
                f"{key}:",

                (
                    "  title: "
                    f"{payload.get('title')}"
                ),

                (
                    "  module: "
                    f"{payload.get('module')}"
                ),

                (
                    "  status: "
                    f"{payload.get('status')}"
                ),

                (
                    "  elapsed_seconds: "
                    f"{payload.get('elapsed_seconds')}"
                ),

                (
                    "  module_args: "
                    f"{payload.get('module_args')}"
                ),

                (
                    "  payload_status: "
                    f"{payload.get('payload_status')}"
                ),

                (
                    "  protocol_only: "
                    f"{payload.get('protocol_only')}"
                ),
            ]
        )

        if payload.get("interpretation"):
            lines.append(
                "  interpretation: "
                f"{payload.get('interpretation')}"
            )

        if payload.get("reuse_validation"):
            lines.append(
                "  reuse_validation: "
                f"{payload.get('reuse_validation')}"
            )

        if payload.get("error"):
            lines.append(
                "  error: "
                f"{payload.get('error')}"
            )

        lines.append("")

    lines.extend(
        [
            "Table summary",

            "-" * 78,
        ]
    )

    for key, value in report.get(
        "table_summary",
        {},
    ).items():
        lines.append(
            f"{key}: {value}"
        )

    lines.extend(
        [
            "",

            "Output files",

            "-" * 78,
        ]
    )

    for key, value in report.get(
        "outputs",
        {},
    ).items():
        lines.append(
            f"{key}: {value}"
        )

    final = _safe_dict(
        report.get("final_status")
    )

    lines.extend(
        [
            "",

            "Final status",

            "-" * 78,

            (
                "operational_status: "
                f"{final.get('operational_status')}"
            ),

            (
                "scientific_status: "
                f"{final.get('scientific_status')}"
            ),

            (
                "scientific_classification: "
                f"{final.get('scientific_classification')}"
            ),

            (
                "candidate_detected: "
                f"{final.get('candidate_detected')}"
            ),

            (
                "headline: "
                f"{final.get('headline')}"
            ),

            (
                "protocol_only: "
                f"{final.get('protocol_only')}"
            ),

            (
                "surrogate: "
                f"{final.get('surrogate')}"
            ),

            (
                "can_claim_empirical_cdr: "
                f"{final.get('can_claim_empirical_cdr')}"
            ),

            (
                "source: "
                f"{final.get('source')}"
            ),

            (
                "interpretation: "
                f"{final.get('interpretation')}"
            ),

            (
                "recommendation: "
                f"{final.get('recommendation')}"
            ),

            (
                "failed_gates: "
                f"{final.get('failed_gates', [])}"
            ),

            (
                "warning_gates: "
                f"{final.get('warning_gates', [])}"
            ),

            (
                "not_evaluated_gates: "
                f"{final.get('not_evaluated_gates', [])}"
            ),

            "",

            "=" * 78,
        ]
    )

    return "\n".join(lines)


# =========================================================
# Main pipeline
# =========================================================

def run_phase3_2e_pipeline(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild: bool = False,
    reuse_conditional: bool = False,
    reuse_controls: bool = False,
    skip_diagnostics: bool = False,
    confirm_heavy_rebuild: bool = False,
) -> Dict[str, Any]:
    cfg = (
        cfg
        or load_phase3_2e_config()
    )

    if (
        force_rebuild
        and (
            reuse_conditional
            or reuse_controls
        )
    ):
        raise ValueError(
            "--force-rebuild cannot be combined "
            "with reuse/skip flags."
        )

    if (
        force_rebuild
        and not confirm_heavy_rebuild
    ):
        raise ValueError(
            "Full force rebuild can execute "
            "Conditional and Controls again. "
            "Use --confirm-heavy-rebuild "
            "explicitly to authorize it."
        )

    np.random.seed(
        int(
            getattr(
                cfg,
                "numpy_seed",
                42,
            )
        )
    )

    paths = phase3_2e_paths(cfg)

    Path(
        cfg.results_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        cfg.interim_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        paths["diagnostics_dir"]
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    runner_report_json = (
        paths["runner_report"]
    )

    runner_summary_txt = (
        paths["runner_summary"]
    )

    started_at = _now_str()
    pipeline_start = time.time()

    report_payloads: Dict[
        str,
        Dict[str, Any],
    ] = {}

    heavy_validation = (
        validate_heavy_artifacts(
            cfg,
            paths,
        )
    )

    if reuse_conditional:
        _required_artifacts_for_reuse(
            "conditional",
            heavy_validation,
        )

    if reuse_controls:
        _required_artifacts_for_reuse(
            "controls",
            heavy_validation,
        )

    report: Dict[str, Any] = {
        "project_name":
            cfg.project_name,

        "phase":
            cfg.phase_name,

        "phase_title":
            cfg.phase_title,

        "version":
            cfg.version,

        "runner_version":
            RUNNER_VERSION,

        "module":
            RUNNER_MODULE,

        "started_at":
            started_at,

        "finished_at":
            None,

        "elapsed_seconds":
            None,

        "options": {
            "force_rebuild":
                bool(force_rebuild),

            "safe_resume":
                bool(
                    reuse_conditional
                    and reuse_controls
                ),

            "reuse_conditional":
                bool(reuse_conditional),

            "reuse_controls":
                bool(reuse_controls),

            "skip_diagnostics":
                bool(skip_diagnostics),

            "confirm_heavy_rebuild":
                bool(
                    confirm_heavy_rebuild
                ),
        },

        "heavy_artifact_validation":
            heavy_validation,

        "config":
            describe_config(cfg),

        "active_conditional_pairs":
            list(
                active_conditional_pairs(
                    cfg
                )
            ),

        "scientific_guardrail": {
            "protocol_only_allowed":
                bool(
                    getattr(
                        cfg,
                        "allow_protocol_only_without_data",
                        True,
                    )
                ),

            "requires_real_synchronized_data_for_claim":
                bool(
                    getattr(
                        cfg,
                        "require_real_synchronized_data_for_claim",
                        True,
                    )
                ),

            "synchronized_input_csv":
                str(
                    paths[
                        "synchronized_input_csv"
                    ]
                ),

            "bic_limited_is_operational_success":
                True,

            "interpretation": (
                "A BIC-limited primary lead is a "
                "completed scientific classification, "
                "not a runner error. Surrogate "
                "provenance still blocks an "
                "empirical claim."
            ),
        },

        "steps":
            {},

        "outputs":
            {},

        "table_summary":
            {},

        "payload_status":
            {},

        "file_status":
            {},

        "final_status":
            {},
    }

    total_steps = len(
        PIPELINE_STEPS
    )

    for (
        index,
        (
            step_key,
            title,
            module_name,
        ),
    ) in enumerate(
        PIPELINE_STEPS,
        start=1,
    ):
        reuse_step = bool(
            (
                step_key == "conditional"
                and reuse_conditional
            )
            or (
                step_key == "controls"
                and reuse_controls
            )
        )

        if reuse_step:
            _step_header(
                index,
                total_steps,
                (
                    f"{title} — "
                    "existing outputs reused"
                ),
            )

            section = (
                _required_artifacts_for_reuse(
                    step_key,
                    heavy_validation,
                )
            )

            payload = collect_step_payload(
                step_key,
                cfg,
                paths,
            )

            report_payloads[
                step_key
            ] = payload

            payload_summary = (
                summarize_step_payload(
                    payload
                )
            )

            report["steps"][
                step_key
            ] = {
                "title":
                    title,

                "module":
                    module_name,

                "status":
                    "reused",

                "elapsed_seconds":
                    0.0,

                "module_args":
                    [],

                **payload_summary,

                "reuse_validation":
                    section,

                "interpretation": (
                    f"Existing {step_key} outputs "
                    "were validated and reused; "
                    "the module was not executed."
                ),
            }

            report[
                "payload_status"
            ][step_key] = (
                payload_summary
            )

            print(
                "[Phase3.2E Runner] "
                f"{step_key} not executed; "
                "validated existing outputs "
                "were reused."
            )

            continue

        if (
            skip_diagnostics
            and step_key == "diagnostics"
        ):
            _step_header(
                index,
                total_steps,
                f"{title} — skipped by flag",
            )

            report["steps"][
                step_key
            ] = {
                "title":
                    title,

                "module":
                    module_name,

                "status":
                    "skipped",

                "elapsed_seconds":
                    0.0,

                "module_args":
                    [],

                "payload_status":
                    "skipped",

                "protocol_only":
                    False,

                "interpretation": (
                    "Diagnostics skipped "
                    "by CLI flag."
                ),
            }

            report_payloads[
                step_key
            ] = {}

            continue

        step_start = time.time()

        _step_header(
            index,
            total_steps,
            title,
        )

        module_args = (
            _module_args_for_step(
                step_key,
                force_rebuild=bool(
                    force_rebuild
                ),
            )
        )

        try:
            run_module_as_main(
                module_name,
                module_args=module_args,
            )

            payload = collect_step_payload(
                step_key,
                cfg,
                paths,
            )

            report_payloads[
                step_key
            ] = payload

            payload_summary = (
                summarize_step_payload(
                    payload
                )
            )

            report["steps"][
                step_key
            ] = {
                "title":
                    title,

                "module":
                    module_name,

                "status":
                    "ok",

                "elapsed_seconds":
                    _elapsed(
                        step_start
                    ),

                "module_args":
                    module_args,

                **payload_summary,
            }

            report[
                "payload_status"
            ][step_key] = (
                payload_summary
            )

        except Exception as exc:
            report["steps"][
                step_key
            ] = {
                "title":
                    title,

                "module":
                    module_name,

                "status":
                    "error",

                "elapsed_seconds":
                    _elapsed(
                        step_start
                    ),

                "module_args":
                    module_args,

                "error":
                    str(exc),

                "traceback":
                    traceback.format_exc(),
            }

            report[
                "finished_at"
            ] = _now_str()

            report[
                "elapsed_seconds"
            ] = _elapsed(
                pipeline_start
            )

            report[
                "table_summary"
            ] = summarize_tables(
                paths,
                cfg,
            )

            report[
                "file_status"
            ] = {
                key: _file_status(path)
                for key, path
                in paths.items()
                if key
                not in {
                    "diagnostics_dir",
                    "lagged_frames_dir",
                }
            }

            report[
                "final_status"
            ] = {
                "status":
                    "runner_error",

                "operational_status":
                    "runner_error",

                "scientific_status":
                    "not_completed",

                "scientific_classification":
                    "not_completed",

                "candidate_detected":
                    False,

                "headline": (
                    "Phase III.2E runner "
                    "failed at step: "
                    f"{step_key}"
                ),

                "interpretation":
                    str(exc),

                "recommendation": (
                    "Inspect the traceback and "
                    "rerun after fixing the "
                    "failing module."
                ),

                "protocol_only":
                    False,

                "surrogate":
                    False,

                "can_claim_empirical_cdr":
                    False,

                "source":
                    "runner",
            }

            report[
                "outputs"
            ][
                "runner_report_json"
            ] = str(
                runner_report_json
            )

            report[
                "outputs"
            ][
                "runner_summary_txt"
            ] = str(
                runner_summary_txt
            )

            save_json(
                runner_report_json,
                report,
            )

            runner_summary_txt.write_text(
                build_runner_summary_text(
                    report
                ),
                encoding="utf-8",
            )

            raise

        report[
            "finished_at"
        ] = _now_str()

        report[
            "elapsed_seconds"
        ] = _elapsed(
            pipeline_start
        )

        report[
            "table_summary"
        ] = summarize_tables(
            paths,
            cfg,
        )

        report[
            "outputs"
        ][
            "runner_report_json"
        ] = str(
            runner_report_json
        )

        report[
            "outputs"
        ][
            "runner_summary_txt"
        ] = str(
            runner_summary_txt
        )

        save_json(
            runner_report_json,
            report,
        )

    for (
        step_key,
        _,
        _,
    ) in PIPELINE_STEPS:
        if (
            step_key
            not in report_payloads
        ):
            report_payloads[
                step_key
            ] = collect_step_payload(
                step_key,
                cfg,
                paths,
            )

    report[
        "table_summary"
    ] = summarize_tables(
        paths,
        cfg,
    )

    report[
        "file_status"
    ] = {
        key: _file_status(path)
        for key, path
        in paths.items()
        if key
        not in {
            "diagnostics_dir",
            "lagged_frames_dir",
        }
    }

    report[
        "outputs"
    ].update(
        {
            "synchronized_input_csv":
                str(
                    paths[
                        "synchronized_input_csv"
                    ]
                ),

            "schema_report":
                str(
                    paths["schema_report"]
                ),

            "sync_validation_report":
                str(
                    paths[
                        "sync_validation_report"
                    ]
                ),

            "loader_report":
                str(
                    paths["loader_report"]
                ),

            "windows_report":
                str(
                    paths["windows_report"]
                ),

            "features_report":
                str(
                    paths["features_report"]
                ),

            "alignment_report":
                str(
                    paths["alignment_report"]
                ),

            "conditional_report":
                str(
                    paths["conditional_report"]
                ),

            "controls_json":
                str(
                    paths["controls_json"]
                ),

            "metrics_report":
                str(
                    paths["metrics_report"]
                ),

            "gate_sensitivity_report":
                str(
                    paths[
                        "gate_sensitivity_report"
                    ]
                ),

            "diagnostics_report":
                str(
                    paths[
                        "diagnostics_report"
                    ]
                ),

            "alignment_inventory":
                str(
                    paths[
                        "alignment_inventory"
                    ]
                ),

            "conditional_pair_scores":
                str(
                    paths[
                        "conditional_pair_scores"
                    ]
                ),

            "conditional_model_scores":
                str(
                    paths[
                        "conditional_model_scores"
                    ]
                ),

            "control_runs":
                str(
                    paths["control_runs"]
                ),

            "control_pair_scores":
                str(
                    paths[
                        "control_pair_scores"
                    ]
                ),

            "metrics_table":
                str(
                    paths["metrics_table"]
                ),

            "f7_sensitivity":
                str(
                    paths["f7_sensitivity"]
                ),

            "f8_sensitivity":
                str(
                    paths["f8_sensitivity"]
                ),

            "f12_sensitivity":
                str(
                    paths["f12_sensitivity"]
                ),

            "runner_report_json":
                str(
                    runner_report_json
                ),

            "runner_summary_txt":
                str(
                    runner_summary_txt
                ),
        }
    )

    report[
        "final_status"
    ] = extract_final_status(
        report_payloads
    )

    report[
        "finished_at"
    ] = _now_str()

    report[
        "elapsed_seconds"
    ] = _elapsed(
        pipeline_start
    )

    save_json(
        runner_report_json,
        report,
    )

    runner_summary_txt.write_text(
        build_runner_summary_text(
            report
        ),
        encoding="utf-8",
    )

    _final_header(
        "Phase III.2E runner completed"
    )

    print(
        "Safe-resume synchronized "
        "EEG–QRNG protocol"
    )

    print("=" * 78)

    final = _safe_dict(
        report.get("final_status")
    )

    table = _safe_dict(
        report.get("table_summary")
    )

    print(
        "operational_status: "
        f"{final.get('operational_status')}"
    )

    print(
        "scientific_status: "
        f"{final.get('scientific_status')}"
    )

    print(
        "scientific_classification: "
        f"{final.get('scientific_classification')}"
    )

    print(
        "candidate_detected: "
        f"{final.get('candidate_detected')}"
    )

    print(
        "protocol_only: "
        f"{final.get('protocol_only')}"
    )

    print(
        "surrogate: "
        f"{final.get('surrogate')}"
    )

    print(
        "can_claim_empirical_cdr: "
        f"{final.get('can_claim_empirical_cdr')}"
    )

    print(
        "source: "
        f"{final.get('source')}"
    )

    print(
        "headline: "
        f"{final.get('headline')}"
    )

    print(
        "recommendation: "
        f"{final.get('recommendation')}"
    )

    print("")

    print("Table summary:")

    print(
        "loaded rows: "
        f"{table.get('loaded_rows')}"
    )

    print(
        "window rows: "
        f"{table.get('window_rows')}"
    )

    print(
        "feature rows: "
        f"{table.get('feature_rows')}"
    )

    print(
        "alignment inventory rows: "
        f"{table.get('alignment_inventory_rows')}"
    )

    print(
        "conditional pair rows: "
        f"{table.get('conditional_pair_rows')}"
    )

    print(
        "primary positive rows: "
        f"{table.get('conditional_primary_positive_lift_rows')}"
    )

    print(
        "control run rows: "
        f"{table.get('control_run_rows')}"
    )

    print(
        "control pair rows: "
        f"{table.get('control_pair_rows')}"
    )

    print(
        "metrics table rows: "
        f"{table.get('metrics_table_rows')}"
    )

    print(
        "F7 sensitivity rows: "
        f"{table.get('f7_sensitivity_rows')}"
    )

    print(
        "F8 sensitivity rows: "
        f"{table.get('f8_sensitivity_rows')}"
    )

    print(
        "F12 sensitivity rows: "
        f"{table.get('f12_sensitivity_rows')}"
    )

    print("")

    print(
        f"runner report: "
        f"{runner_report_json}"
    )

    print(
        f"runner summary: "
        f"{runner_summary_txt}"
    )

    print("=" * 78 + "\n")

    return report


# =========================================================
# CLI
# =========================================================

def validate_runner_state(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    cfg = (
        cfg
        or load_phase3_2e_config()
    )

    paths = phase3_2e_paths(cfg)

    heavy = validate_heavy_artifacts(
        cfg,
        paths,
    )

    downstream = {
        "metrics_report":
            _validate_nonempty_json(
                paths["metrics_report"]
            ),

        "gate_sensitivity_report":
            _validate_nonempty_json(
                paths[
                    "gate_sensitivity_report"
                ]
            ),

        "diagnostics_report":
            _validate_nonempty_json(
                paths[
                    "diagnostics_report"
                ]
            ),
    }

    return {
        "runner_version":
            RUNNER_VERSION,

        "safe_resume_ready":
            heavy[
                "safe_resume_ready"
            ],

        "heavy_artifacts":
            heavy,

        "downstream_artifacts":
            downstream,

        "recommended_command": (
            "python -m "
            "src.phase3_2e_runner "
            "--safe-resume"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CDR Phase III.2E "
            "synchronized EEG–QRNG pipeline "
            "with safe reuse of expensive stages."
        )
    )

    parser.add_argument(
        "--safe-resume",
        action="store_true",
        help=(
            "Reuse validated Conditional and "
            "Controls outputs. Neither expensive "
            "module is executed."
        ),
    )

    parser.add_argument(
        "--skip-conditional",
        action="store_true",
        help=(
            "Do not execute Conditional; require "
            "and reuse its existing outputs."
        ),
    )

    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help=(
            "Do not execute Controls; require "
            "and reuse its existing outputs."
        ),
    )

    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help=(
            "Skip final diagnostics generation."
        ),
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Force supported rebuilds. This may "
            "rerun Conditional and Controls and "
            "therefore requires "
            "--confirm-heavy-rebuild."
        ),
    )

    parser.add_argument(
        "--confirm-heavy-rebuild",
        action="store_true",
        help=(
            "Explicit authorization for a force "
            "rebuild that may execute expensive "
            "stages."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate existing heavy and downstream "
            "artifacts without running the pipeline."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    if args.validate_only:
        validation = validate_runner_state(
            cfg
        )

        print(
            json.dumps(
                validation,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
        )

        raise SystemExit(
            0
            if validation[
                "safe_resume_ready"
            ]
            else 1
        )

    reuse_conditional = bool(
        args.safe_resume
        or args.skip_conditional
    )

    reuse_controls = bool(
        args.safe_resume
        or args.skip_controls
    )

    run_phase3_2e_pipeline(
        cfg=cfg,

        force_rebuild=bool(
            args.force_rebuild
        ),

        reuse_conditional=
            reuse_conditional,

        reuse_controls=
            reuse_controls,

        skip_diagnostics=bool(
            args.skip_diagnostics
        ),

        confirm_heavy_rebuild=bool(
            args.confirm_heavy_rebuild
        ),
    )


if __name__ == "__main__":
    main()
