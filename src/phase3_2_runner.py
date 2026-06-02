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
#   2. Features
#   3. Regimes                 — III.2A
#   4. Lagging                 — III.2B
#   5. Conditional CDR         — III.2C
#   6. Negative controls
#   7. Metrics / gates
#   8. Diagnostics
#   9. Final report
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

    return str(payload.get("module", "")) == EXPECTED_CONTROL_MODULE


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
        "expected_module": EXPECTED_CONTROL_MODULE,
        "is_current": str(payload.get("module", "")) == EXPECTED_CONTROL_MODULE,
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

    return all(path.exists() for path in required)


# =========================================================
# Summary text
# =========================================================

def build_runner_summary_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("CDR Phase III.2 — Runner Summary")
    lines.append("Leakage-safe conditional estimator aware version")
    lines.append("=" * 78)
    lines.append("")

    lines.append("Project")
    lines.append("-" * 78)
    lines.append(f"project_name: {report.get('project_name')}")
    lines.append(f"phase: {report.get('phase')}")
    lines.append(f"phase_title: {report.get('phase_title')}")
    lines.append(f"version: {report.get('version')}")
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
        "runner_version": "phase3_2_runner_v2_leakage_safe_diagnostics_aware",
        "started_at": started_at,
        "finished_at": None,
        "elapsed_seconds": None,
        "options": {
            "force_rebuild": bool(force_rebuild),
            "skip_controls": bool(skip_controls),
            "fast_controls": bool(fast_controls),
            "skip_diagnostics": bool(skip_diagnostics),
            "n_controls_effective": int(cfg.n_controls),
        },
        "config": describe_config(cfg),
        "steps": {},
        "outputs": {},
        "final_status": {},
        "diagnostics_final_interpretation": {},
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
    # Step 2 — Features
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 2/8 — Feature preparation")

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
    # Step 5 — Conditional CDR
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 5/8 — Conditional CDR")

    try:
        model_scores_df, pair_scores_df = evaluate_all_conditional_from_lagged_csvs(cfg)

        save_conditional_outputs(
            model_scores_df=model_scores_df,
            pair_scores_df=pair_scores_df,
            cfg=cfg,
        )

        conditional_summary = summarize_conditional_pairs(pair_scores_df)

        report["steps"]["conditional"] = {
            "status": "ok",
            "elapsed_seconds": _elapsed(step_start),
            "summary": conditional_summary,
        }

        report["outputs"]["conditional_model_scores"] = str(Path(cfg.results_dir) / "phase3_2c_conditional_model_scores.csv")
        report["outputs"]["conditional_pair_scores"] = str(cfg.conditional_results_csv)
        report["outputs"]["conditional_summary"] = str(cfg.conditional_summary_json)

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
    _step_header("Step 6/8 — Negative controls")

    if skip_controls:
        existing_controls = summarize_existing_controls(cfg)

        if controls_outputs_current(cfg):
            report["steps"]["controls"] = {
                "status": "skipped_reusing_current_outputs",
                "elapsed_seconds": _elapsed(step_start),
                "summary": existing_controls,
            }

            print("[Phase3.2 Runner] Controls skipped; current leakage-safe outputs will be reused.")

        elif controls_outputs_exist(cfg):
            report["steps"]["controls"] = {
                "status": "skipped_reusing_stale_or_unknown_outputs",
                "elapsed_seconds": _elapsed(step_start),
                "summary": existing_controls,
                "warning": (
                    "Existing controls were found, but their module does not match "
                    f"{EXPECTED_CONTROL_MODULE}. Metrics may mark F2 as pending/stale."
                ),
            }

            print("[Phase3.2 Runner] WARNING: controls skipped but existing controls may be stale.")

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
    report["finished_at"] = _now_str()
    report["elapsed_seconds"] = _elapsed(pipeline_start)
    report["outputs"]["runner_report_json"] = str(runner_report_json)
    report["outputs"]["runner_summary_txt"] = str(runner_summary_txt)

    save_json(runner_report_json, report)

    # -----------------------------------------------------
    # Step 8 — Diagnostics
    # -----------------------------------------------------
    step_start = time.time()
    _step_header("Step 8/8 — Diagnostics")

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

            report["steps"]["diagnostics"] = {
                "status": "ok",
                "elapsed_seconds": _elapsed(step_start),
                "summary": final_interpretation,
            }

            report["diagnostics_final_interpretation"] = final_interpretation

            diagnostics_dir = Path(cfg.diagnostics_dir)

            report["outputs"]["diagnostics_report"] = str(diagnostics_dir / "phase3_2_diagnostics_report.json")
            report["outputs"]["diagnostics_summary"] = str(diagnostics_dir / "phase3_2_diagnostics_summary.txt")
            report["outputs"]["diagnostics_primary_tests"] = str(diagnostics_dir / "phase3_2_primary_tests.csv")
            report["outputs"]["diagnostics_positive_lift_rows"] = str(diagnostics_dir / "phase3_2_positive_lift_rows.csv")
            report["outputs"]["diagnostics_exploratory_lift_rows"] = str(diagnostics_dir / "phase3_2_exploratory_lift_rows.csv")
            report["outputs"]["diagnostics_strong_candidate_rows"] = str(diagnostics_dir / "phase3_2_strong_candidate_rows.csv")
            report["outputs"]["diagnostics_control_breakdown"] = str(diagnostics_dir / "phase3_2_control_breakdown.csv")

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
    report["finished_at"] = _now_str()
    report["elapsed_seconds"] = _elapsed(pipeline_start)

    save_json(runner_report_json, report)

    with open(runner_summary_txt, "w", encoding="utf-8") as f:
        f.write(build_runner_summary_text(report))

    # Save once more with final paths and summary confirmed.
    save_json(runner_report_json, report)

    print("\n" + "=" * 78)
    print("Phase III.2 runner completed")
    print("Leakage-safe conditional estimator aware version")
    print("=" * 78)
    print(f"status: {report.get('final_status', {}).get('status')}")
    print(f"interpretation: {report.get('final_status', {}).get('interpretation')}")

    diagnostics_final = report.get("diagnostics_final_interpretation", {})

    if diagnostics_final:
        print(f"diagnostic_headline: {diagnostics_final.get('headline')}")
        print(f"diagnostic_recommendation: {diagnostics_final.get('recommendation')}")

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
        help="Skip controls and reuse existing control outputs if available.",
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