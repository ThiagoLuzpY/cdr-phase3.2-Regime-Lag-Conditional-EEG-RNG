from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.phase3_2e_config import Phase32EConfig, describe_config, load_phase3_2e_config


# =========================================================
# Phase III.2E — Audit Bundle Builder
# Synchronized EEG–QRNG protocol version
# =========================================================
#
# Purpose:
#
#   Build a reproducibility/audit bundle for CDR Phase III.2E:
#
#       III.2E — synchronized EEG–QRNG protocol and gate-sensitivity validation
#
#   Current status:
#
#       The implementation can complete in protocol-only mode when no real
#       synchronized EEG + QRNG input CSV exists.
#
#       Empirical EEG–QRNG claims remain blocked until a real synchronized
#       dataset is provided and all required controls/gates pass.
#
# The bundle includes:
#
#   - Phase III.2E source code
#   - config
#   - protocol document
#   - runner report and summary
#   - schema/sync/loader/window/feature/alignment reports
#   - conditional outputs
#   - control outputs
#   - metrics/gates outputs
#   - F7/F8/F12 sensitivity outputs
#   - diagnostics outputs
#   - selected small CSV inventories/specs
#   - raw data inventory manifest, without copying raw data files
#   - SHA256 hashes for included files
#
# It does NOT copy raw EDF/RNG/QRNG files into the zip.
#
# It also skips heavy CSV/ZIP/binary outputs by default and records them as
# skipped files in the audit manifest.
#
# Run:
#
#   python -m scripts.make_phase3_2e_audit_bundle
#


# =========================================================
# Constants
# =========================================================

BUNDLE_VERSION = "phase3_2e_audit_bundle_v1_protocol_sync_gate_sensitivity"
BUNDLE_SCOPE = "Phase III.2E"
BUNDLE_SCOPE_LABEL = "phase3_2e"

EXPECTED_PROTOCOL_STATUS = "pending_synchronized_data"

DEFAULT_MAX_BUNDLE_FILE_MB = 10.0
DEFAULT_RAW_HASH_MAX_MB = 250.0

HEAVY_SUFFIXES = {
    ".edf",
    ".bdf",
    ".fif",
    ".set",
    ".mat",
    ".npz",
    ".npy",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".db",
    ".sqlite",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
}


# =========================================================
# Generic helpers
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

    return str(obj)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _now_human() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(root: Path, value: Any) -> Path:
    p = Path(str(value))

    if p.is_absolute():
        return p

    return root / p


def safe_read_json(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def file_record(
    root: Path,
    path: Path,
    category: str,
    compute_hash: bool = True,
) -> Dict[str, Any]:
    path = Path(path)
    stat = path.stat()

    try:
        rel = path.relative_to(root)
    except Exception:
        rel = path

    record: Dict[str, Any] = {
        "category": category,
        "path": str(rel).replace("\\", "/"),
        "absolute_path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "mtime_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "sha256": None,
    }

    if compute_hash:
        record["sha256"] = sha256_file(path)

    return record


def should_skip_bundle_file(
    path: Path,
    max_bundle_file_mb: float,
    include_csv: bool,
) -> Tuple[bool, str]:
    path = Path(path)

    if not path.exists() or not path.is_file():
        return False, ""

    suffix = path.suffix.lower()
    size_bytes = path.stat().st_size
    max_bytes = int(float(max_bundle_file_mb) * 1024 * 1024)

    if suffix == ".csv" and not include_csv:
        return True, "csv_excluded_by_flag"

    if suffix in HEAVY_SUFFIXES:
        return True, f"heavy_suffix_{suffix}"

    if size_bytes > max_bytes:
        return True, f"file_larger_than_{max_bundle_file_mb}_mb"

    return False, ""


def copy_into_staging(
    root: Path,
    staging_root: Path,
    source_path: Path,
    category: str,
    manifest_files: List[Dict[str, Any]],
    missing_files: List[Dict[str, Any]],
    skipped_files: List[Dict[str, Any]],
    compute_hash: bool = True,
    max_bundle_file_mb: float = DEFAULT_MAX_BUNDLE_FILE_MB,
    include_csv: bool = True,
) -> None:
    source_path = Path(source_path)

    if not source_path.exists():
        missing_files.append(
            {
                "category": category,
                "path": str(source_path),
                "reason": "missing",
            }
        )
        return

    if not source_path.is_file():
        missing_files.append(
            {
                "category": category,
                "path": str(source_path),
                "reason": "not_a_file",
            }
        )
        return

    skip, reason = should_skip_bundle_file(
        path=source_path,
        max_bundle_file_mb=float(max_bundle_file_mb),
        include_csv=bool(include_csv),
    )

    if skip:
        stat = source_path.stat()

        skipped_files.append(
            {
                "category": category,
                "path": str(source_path),
                "reason": reason,
                "size_bytes": int(stat.st_size),
                "mtime_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
        return

    try:
        if source_path.is_relative_to(root):
            rel_path = source_path.relative_to(root)
        else:
            rel_path = Path("_external") / source_path.name

    except Exception:
        rel_path = Path("_external") / source_path.name

    target_path = staging_root / rel_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_path, target_path)

    copied_record = file_record(
        root=staging_root,
        path=target_path,
        category=category,
        compute_hash=compute_hash,
    )

    copied_record["source_path"] = str(source_path)

    manifest_files.append(copied_record)


def add_existing_files(
    root: Path,
    staging_root: Path,
    files: List[Tuple[str, Path]],
    manifest_files: List[Dict[str, Any]],
    missing_files: List[Dict[str, Any]],
    skipped_files: List[Dict[str, Any]],
    compute_hash: bool = True,
    max_bundle_file_mb: float = DEFAULT_MAX_BUNDLE_FILE_MB,
    include_csv: bool = True,
) -> None:
    seen: set[str] = set()

    for category, path in files:
        p = Path(path)

        key = str(p.resolve()) if p.exists() else str(p)

        if key in seen:
            continue

        seen.add(key)

        copy_into_staging(
            root=root,
            staging_root=staging_root,
            source_path=p,
            category=category,
            manifest_files=manifest_files,
            missing_files=missing_files,
            skipped_files=skipped_files,
            compute_hash=compute_hash,
            max_bundle_file_mb=float(max_bundle_file_mb),
            include_csv=bool(include_csv),
        )


def add_directory_files(
    root: Path,
    staging_root: Path,
    directory: Path,
    category: str,
    manifest_files: List[Dict[str, Any]],
    missing_files: List[Dict[str, Any]],
    skipped_files: List[Dict[str, Any]],
    patterns: Optional[List[str]] = None,
    compute_hash: bool = True,
    max_bundle_file_mb: float = DEFAULT_MAX_BUNDLE_FILE_MB,
    include_csv: bool = True,
) -> None:
    directory = Path(directory)

    if not directory.exists():
        missing_files.append(
            {
                "category": category,
                "path": str(directory),
                "reason": "directory_missing",
            }
        )
        return

    if not directory.is_dir():
        missing_files.append(
            {
                "category": category,
                "path": str(directory),
                "reason": "not_a_directory",
            }
        )
        return

    if patterns is None:
        patterns = ["*.json", "*.txt", "*.md", "*.csv"]

    seen: set[str] = set()

    for pattern in patterns:
        for path in sorted(directory.rglob(pattern)):
            if not path.is_file():
                continue

            key = str(path.resolve())

            if key in seen:
                continue

            seen.add(key)

            copy_into_staging(
                root=root,
                staging_root=staging_root,
                source_path=path,
                category=category,
                manifest_files=manifest_files,
                missing_files=missing_files,
                skipped_files=skipped_files,
                compute_hash=compute_hash,
                max_bundle_file_mb=float(max_bundle_file_mb),
                include_csv=bool(include_csv),
            )


# =========================================================
# Raw data inventory
# =========================================================

def raw_data_inventory(
    root: Path,
    hash_raw_files: bool = True,
    raw_hash_max_mb: float = DEFAULT_RAW_HASH_MAX_MB,
) -> Dict[str, Any]:
    raw_dir = root / "data" / "raw"

    inventory: Dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "available": raw_dir.exists(),
        "hash_raw_files": bool(hash_raw_files),
        "raw_hash_max_mb": float(raw_hash_max_mb),
        "files": [],
    }

    if not raw_dir.exists():
        inventory["n_files"] = 0
        inventory["total_size_bytes"] = 0
        return inventory

    max_bytes = int(float(raw_hash_max_mb) * 1024 * 1024)

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue

        stat = path.stat()
        rel = path.relative_to(root)

        record: Dict[str, Any] = {
            "path": str(rel).replace("\\", "/"),
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
            "mtime_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            "sha256": None,
            "hash_status": "not_requested",
        }

        if hash_raw_files:
            if stat.st_size <= max_bytes:
                record["sha256"] = sha256_file(path)
                record["hash_status"] = "ok"
            else:
                record["hash_status"] = "skipped_file_too_large"

        inventory["files"].append(record)

    inventory["n_files"] = int(len(inventory["files"]))
    inventory["total_size_bytes"] = int(sum(item["size_bytes"] for item in inventory["files"]))

    return inventory


# =========================================================
# Phase III.2E path registry
# =========================================================

def phase3_2e_paths(root: Path, cfg: Phase32EConfig) -> Dict[str, Path]:
    results_dir = resolve_project_path(root, cfg.results_dir)
    interim_dir = resolve_project_path(root, cfg.interim_dir)

    diagnostics_dir = resolve_project_path(
        root,
        getattr(cfg, "diagnostics_dir", results_dir / "diagnostics"),
    )

    synchronized_input_csv = resolve_project_path(
        root,
        getattr(
            cfg,
            "synchronized_input_csv",
            Path("data") / "raw" / "phase3_2e" / "synchronized" / "phase3_2e_synchronized_input.csv",
        ),
    )

    return {
        # Input/protocol
        "synchronized_input_csv": synchronized_input_csv,

        # Schema/sync
        "schema_report": results_dir / "phase3_2e_schema_report.json",
        "schema_summary": results_dir / "phase3_2e_schema_summary.txt",
        "sync_validation_report": results_dir / "phase3_2e_sync_validation_report.json",
        "sync_validation_summary": results_dir / "phase3_2e_sync_validation_summary.txt",

        # Loader/windows/features/alignment
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

        # Conditional
        "conditional_model_scores": resolve_project_path(
            root,
            getattr(
                cfg,
                "conditional_model_scores_csv",
                results_dir / "phase3_2e_conditional_model_scores.csv",
            ),
        ),
        "conditional_pair_scores": resolve_project_path(
            root,
            getattr(
                cfg,
                "conditional_pair_scores_csv",
                results_dir / "phase3_2e_conditional_pair_scores.csv",
            ),
        ),
        "conditional_summary_json": resolve_project_path(
            root,
            getattr(
                cfg,
                "conditional_summary_json",
                results_dir / "phase3_2e_conditional_summary.json",
            ),
        ),
        "conditional_summary_txt": resolve_project_path(
            root,
            getattr(
                cfg,
                "conditional_summary_txt",
                results_dir / "phase3_2e_conditional_summary.txt",
            ),
        ),
        "conditional_report": results_dir / "phase3_2e_conditional_report.json",

        # Controls
        "control_runs": resolve_project_path(
            root,
            getattr(
                cfg,
                "control_runs_csv",
                results_dir / "phase3_2e_control_runs.csv",
            ),
        ),
        "control_pair_scores": resolve_project_path(
            root,
            getattr(
                cfg,
                "control_pair_scores_csv",
                results_dir / "phase3_2e_control_pair_scores.csv",
            ),
        ),
        "controls_json": resolve_project_path(
            root,
            getattr(
                cfg,
                "controls_json",
                results_dir / "phase3_2e_controls.json",
            ),
        ),
        "controls_summary": resolve_project_path(
            root,
            getattr(
                cfg,
                "controls_summary_txt",
                results_dir / "phase3_2e_controls_summary.txt",
            ),
        ),
        "controls_report": results_dir / "phase3_2e_controls_report.json",

        # Metrics/gates
        "gates_json": resolve_project_path(
            root,
            getattr(
                cfg,
                "gates_json",
                results_dir / "phase3_2e_gates.json",
            ),
        ),
        "multiple_comparison_guard": resolve_project_path(
            root,
            getattr(
                cfg,
                "multiple_comparison_guard_json",
                results_dir / "phase3_2e_multiple_comparison_guard.json",
            ),
        ),
        "metrics_table": results_dir / "phase3_2e_metrics_table.csv",
        "metrics_summary": results_dir / "phase3_2e_metrics_summary.txt",
        "metrics_report": results_dir / "phase3_2e_metrics_report.json",

        # Gate sensitivity
        "gate_sensitivity_report": results_dir / "phase3_2e_gate_sensitivity_report.json",
        "gate_sensitivity_summary": results_dir / "phase3_2e_gate_sensitivity_summary.txt",
        "f7_sensitivity": results_dir / "phase3_2e_f7_subject_sensitivity.csv",
        "f8_sensitivity": results_dir / "phase3_2e_f8_complexity_sensitivity.csv",
        "f12_sensitivity": results_dir / "phase3_2e_f12_epsilon_saturation_sensitivity.csv",

        # Diagnostics
        "diagnostics_dir": diagnostics_dir,
        "diagnostics_report": diagnostics_dir / "phase3_2e_diagnostics_report.json",
        "diagnostics_summary": diagnostics_dir / "phase3_2e_diagnostics_summary.txt",

        # Runner
        "runner_report": results_dir / "phase3_2e_runner_report.json",
        "runner_summary": results_dir / "phase3_2e_runner_summary.txt",
    }


# =========================================================
# Phase III.2E file selection
# =========================================================

def core_source_files(root: Path) -> List[Tuple[str, Path]]:
    return [
        ("project_docs", root / "README.md"),
        ("project_docs", root / "requirements.txt"),
        ("project_docs", root / "pyproject.toml"),
        ("project_docs", root / "setup.cfg"),
        ("project_docs", root / ".gitignore"),

        ("protocol_doc", root / "docs" / "phase3_2e_synchronized_eeg_qrng_protocol.md"),

        ("config", root / "config" / "phase3_2e_config.py"),

        ("source", root / "src" / "phase3_2e_schema.py"),
        ("source", root / "src" / "phase3_2e_sync_validator.py"),
        ("source", root / "src" / "phase3_2e_loader.py"),
        ("source", root / "src" / "phase3_2e_windows.py"),
        ("source", root / "src" / "phase3_2e_features.py"),
        ("source", root / "src" / "phase3_2e_alignment.py"),
        ("source", root / "src" / "phase3_2e_conditional.py"),
        ("source", root / "src" / "phase3_2e_controls.py"),
        ("source", root / "src" / "phase3_2e_metrics.py"),
        ("source", root / "src" / "phase3_2e_gate_sensitivity.py"),
        ("source", root / "src" / "phase3_2e_diagnostics.py"),
        ("source", root / "src" / "phase3_2e_runner.py"),

        ("script", root / "scripts" / "make_phase3_2e_audit_bundle.py"),
    ]


def result_files(root: Path, cfg: Phase32EConfig) -> List[Tuple[str, Path]]:
    paths = phase3_2e_paths(root, cfg)

    candidates: List[Tuple[str, Path]] = [
        # Schema / sync
        ("schema_output", paths["schema_report"]),
        ("schema_output", paths["schema_summary"]),
        ("sync_output", paths["sync_validation_report"]),
        ("sync_output", paths["sync_validation_summary"]),

        # Loader / window / feature / alignment
        ("loader_output", paths["loader_report"]),
        ("loader_output", paths["loader_summary"]),
        ("loader_output", paths["loader_metadata"]),
        ("loader_output", paths["loaded_windows"]),

        ("window_output", paths["windows_report"]),
        ("window_output", paths["windows_summary"]),
        ("window_output", paths["windows"]),
        ("window_output", paths["window_inventory"]),

        ("feature_output", paths["features_report"]),
        ("feature_output", paths["features_summary"]),
        ("feature_output", paths["features"]),
        ("feature_output", paths["feature_specs"]),

        ("alignment_output", paths["alignment_report"]),
        ("alignment_output", paths["alignment_summary"]),
        ("alignment_output", paths["alignment_inventory"]),
        ("alignment_output", paths["alignment_metadata"]),

        # Conditional
        ("conditional_output", paths["conditional_report"]),
        ("conditional_output", paths["conditional_summary_json"]),
        ("conditional_output", paths["conditional_summary_txt"]),
        ("conditional_output", paths["conditional_model_scores"]),
        ("conditional_output", paths["conditional_pair_scores"]),

        # Controls
        ("control_output", paths["controls_json"]),
        ("control_output", paths["controls_summary"]),
        ("control_output", paths["controls_report"]),
        ("control_output", paths["control_runs"]),
        ("control_output", paths["control_pair_scores"]),

        # Metrics/gates
        ("metrics_output", paths["gates_json"]),
        ("metrics_output", paths["multiple_comparison_guard"]),
        ("metrics_output", paths["metrics_table"]),
        ("metrics_output", paths["metrics_summary"]),
        ("metrics_output", paths["metrics_report"]),

        # Gate sensitivity
        ("gate_sensitivity_output", paths["gate_sensitivity_report"]),
        ("gate_sensitivity_output", paths["gate_sensitivity_summary"]),
        ("gate_sensitivity_output", paths["f7_sensitivity"]),
        ("gate_sensitivity_output", paths["f8_sensitivity"]),
        ("gate_sensitivity_output", paths["f12_sensitivity"]),

        # Diagnostics
        ("diagnostics_output", paths["diagnostics_report"]),
        ("diagnostics_output", paths["diagnostics_summary"]),

        # Runner
        ("runner_output", paths["runner_report"]),
        ("runner_output", paths["runner_summary"]),
    ]

    diagnostics_dir = paths["diagnostics_dir"]

    if diagnostics_dir.exists():
        for path in sorted(diagnostics_dir.rglob("*")):
            if path.is_file():
                candidates.append(("diagnostics_output", path))

    return candidates


# =========================================================
# Status extraction
# =========================================================

def extract_phase3_2e_status(root: Path, cfg: Phase32EConfig) -> Dict[str, Any]:
    paths = phase3_2e_paths(root, cfg)

    schema_report = safe_read_json(paths["schema_report"])
    sync_report = safe_read_json(paths["sync_validation_report"])
    loader_report = safe_read_json(paths["loader_report"])
    windows_report = safe_read_json(paths["windows_report"])
    features_report = safe_read_json(paths["features_report"])
    alignment_report = safe_read_json(paths["alignment_report"])
    conditional_report = safe_read_json(paths["conditional_report"])
    controls_report = safe_read_json(paths["controls_json"])
    metrics_report = safe_read_json(paths["metrics_report"])
    gate_sensitivity_report = safe_read_json(paths["gate_sensitivity_report"])
    diagnostics_report = safe_read_json(paths["diagnostics_report"])
    runner_report = safe_read_json(paths["runner_report"])

    final_interpretation = diagnostics_report.get("final_interpretation", {})
    runner_final = runner_report.get("final_status", {})
    metrics_final = metrics_report.get("final_status", {})
    gate_sensitivity_result = gate_sensitivity_report.get("result", {})

    synchronized_input = paths["synchronized_input_csv"]

    return {
        "scope": BUNDLE_SCOPE,
        "expected_protocol_status": EXPECTED_PROTOCOL_STATUS,
        "synchronized_input_csv": str(synchronized_input),
        "synchronized_input_exists": bool(synchronized_input.exists()),

        "schema_status": schema_report.get("status"),
        "sync_validation_status": sync_report.get("status"),
        "loader_status": loader_report.get("status"),
        "windows_status": windows_report.get("status"),
        "features_status": features_report.get("status"),
        "alignment_status": alignment_report.get("status"),
        "conditional_status": conditional_report.get("status"),
        "controls_status": controls_report.get("status"),
        "metrics_status": metrics_report.get("status"),
        "gate_sensitivity_status": gate_sensitivity_report.get("status"),

        "runner_final_status": runner_final,
        "metrics_final_status": metrics_final,
        "gate_sensitivity_result": gate_sensitivity_result,
        "diagnostics_final_interpretation": final_interpretation,

        "can_claim_empirical_cdr": bool(
            runner_final.get(
                "can_claim_empirical_cdr",
                metrics_final.get("can_claim_empirical_cdr", False),
            )
        ),
        "protocol_only": bool(
            runner_final.get(
                "protocol_only",
                metrics_final.get("protocol_only", True),
            )
        ),
        "final_status": runner_final.get(
            "status",
            metrics_final.get("status", EXPECTED_PROTOCOL_STATUS),
        ),
        "headline": runner_final.get(
            "headline",
            final_interpretation.get("headline"),
        ),
        "recommendation": runner_final.get(
            "recommendation",
            final_interpretation.get("recommendation"),
        ),
        "interpretation": (
            "Phase III.2E is complete as a synchronized EEG–QRNG protocol implementation. "
            "Empirical CDR remains blocked until real synchronized EEG + QRNG data are provided."
        ),
    }


def extract_table_summary(root: Path, cfg: Phase32EConfig) -> Dict[str, Any]:
    paths = phase3_2e_paths(root, cfg)
    runner_report = safe_read_json(paths["runner_report"])
    diagnostics_report = safe_read_json(paths["diagnostics_report"])

    runner_table = runner_report.get("table_summary", {})
    diagnostics_tables = diagnostics_report.get("artifact_table_diagnostics", {})

    return {
        "runner_table_summary": runner_table,
        "diagnostics_table_summary": diagnostics_tables,
    }


def extract_status_summary(root: Path, cfg: Phase32EConfig) -> Dict[str, Any]:
    paths = phase3_2e_paths(root, cfg)

    controls_json = safe_read_json(paths["controls_json"])
    metrics_report = safe_read_json(paths["metrics_report"])
    gates_json = safe_read_json(paths["gates_json"])
    conditional_report = safe_read_json(paths["conditional_report"])
    gate_sensitivity_report = safe_read_json(paths["gate_sensitivity_report"])
    diagnostics_report = safe_read_json(paths["diagnostics_report"])
    runner_report = safe_read_json(paths["runner_report"])

    return {
        "bundle_scope": BUNDLE_SCOPE,
        "bundle_scope_label": BUNDLE_SCOPE_LABEL,
        "phase3_2e_status": extract_phase3_2e_status(root, cfg),
        "table_summary": extract_table_summary(root, cfg),
        "runner_final_status": runner_report.get("final_status", {}),
        "diagnostics_final_interpretation": diagnostics_report.get("final_interpretation", {}),
        "metrics_final_status": metrics_report.get("final_status", {}),
        "gates": gates_json.get("gates", []),
        "controls_summary": controls_json.get("summary", {}),
        "conditional_summary": conditional_report.get("summary", {}),
        "gate_sensitivity_summary": {
            "result": gate_sensitivity_report.get("result", {}),
            "official_gate_statuses": gate_sensitivity_report.get("official_gate_statuses", {}),
            "f7_subject_consistency": gate_sensitivity_report.get("f7_subject_consistency", {}),
            "f8_complexity_penalty": gate_sensitivity_report.get("f8_complexity_penalty", {}),
            "f12_epsilon_saturation": gate_sensitivity_report.get("f12_epsilon_saturation", {}),
        },
        "expected_protocol_status": EXPECTED_PROTOCOL_STATUS,
    }


# =========================================================
# Audit summary markdown
# =========================================================

def build_audit_summary_md(
    cfg: Phase32EConfig,
    status_summary: Dict[str, Any],
    manifest: Dict[str, Any],
) -> str:
    phase3_2e = status_summary.get("phase3_2e_status", {})
    runner_final = status_summary.get("runner_final_status", {})
    diagnostics_final = status_summary.get("diagnostics_final_interpretation", {})
    metrics_final = status_summary.get("metrics_final_status", {})
    controls_summary = status_summary.get("controls_summary", {})
    conditional_summary = status_summary.get("conditional_summary", {})
    gate_sensitivity = status_summary.get("gate_sensitivity_summary", {})
    table_summary = status_summary.get("table_summary", {})

    runner_tables = table_summary.get("runner_table_summary", {})

    lines: List[str] = []

    lines.append("# CDR Phase III.2E — Audit Bundle")
    lines.append("")
    lines.append(f"- **Bundle version:** `{BUNDLE_VERSION}`")
    lines.append(f"- **Bundle scope:** `{BUNDLE_SCOPE}`")
    lines.append(f"- **Created at:** `{manifest.get('created_at')}`")
    lines.append(f"- **Project:** `{cfg.project_name}`")
    lines.append(f"- **Phase:** `{cfg.phase_name}`")
    lines.append(f"- **Phase title:** `{cfg.phase_title}`")
    lines.append(f"- **Code/config version:** `{cfg.version}`")
    lines.append("")

    lines.append("## Included subphase")
    lines.append("")
    lines.append("- **III.2E:** synchronized EEG–QRNG protocol, timestamp validation, gate-sensitivity diagnostics and final protocol audit.")
    lines.append("")

    lines.append("## Final status")
    lines.append("")
    lines.append(f"- **Runner status:** `{runner_final.get('status')}`")
    lines.append(f"- **Metrics status:** `{metrics_final.get('status')}`")
    lines.append(f"- **Diagnostics headline:** `{diagnostics_final.get('headline')}`")
    lines.append(f"- **Protocol-only:** `{phase3_2e.get('protocol_only')}`")
    lines.append(f"- **Can claim empirical CDR:** `{phase3_2e.get('can_claim_empirical_cdr')}`")
    lines.append(f"- **Synchronized input exists:** `{phase3_2e.get('synchronized_input_exists')}`")
    lines.append("")
    lines.append("**Interpretation:**")
    lines.append("")
    lines.append(f"> {phase3_2e.get('interpretation')}")
    lines.append("")
    lines.append("**Recommendation:**")
    lines.append("")
    lines.append(f"> {phase3_2e.get('recommendation')}")
    lines.append("")

    lines.append("## Scientific guardrail")
    lines.append("")
    lines.append("- Phase III.2E requires a real synchronized EEG + QRNG dataset for empirical interpretation.")
    lines.append("- Protocol-only execution validates implementation readiness, not physical coupling.")
    lines.append("- Positive EEG–QRNG claims remain blocked until synchronized data exist and controls/gates pass.")
    lines.append("")

    lines.append("## Pipeline status")
    lines.append("")
    lines.append(f"- **Schema:** `{phase3_2e.get('schema_status')}`")
    lines.append(f"- **Synchronization validator:** `{phase3_2e.get('sync_validation_status')}`")
    lines.append(f"- **Loader:** `{phase3_2e.get('loader_status')}`")
    lines.append(f"- **Windows:** `{phase3_2e.get('windows_status')}`")
    lines.append(f"- **Features:** `{phase3_2e.get('features_status')}`")
    lines.append(f"- **Alignment:** `{phase3_2e.get('alignment_status')}`")
    lines.append(f"- **Conditional:** `{phase3_2e.get('conditional_status')}`")
    lines.append(f"- **Controls:** `{phase3_2e.get('controls_status')}`")
    lines.append(f"- **Metrics:** `{phase3_2e.get('metrics_status')}`")
    lines.append(f"- **Gate sensitivity:** `{phase3_2e.get('gate_sensitivity_status')}`")
    lines.append("")

    lines.append("## Runner table summary")
    lines.append("")
    lines.append(f"- **Loaded rows:** `{runner_tables.get('loaded_rows')}`")
    lines.append(f"- **Window rows:** `{runner_tables.get('window_rows')}`")
    lines.append(f"- **Feature rows:** `{runner_tables.get('feature_rows')}`")
    lines.append(f"- **Alignment inventory rows:** `{runner_tables.get('alignment_inventory_rows')}`")
    lines.append(f"- **Conditional pair rows:** `{runner_tables.get('conditional_pair_rows')}`")
    lines.append(f"- **Control run rows:** `{runner_tables.get('control_run_rows')}`")
    lines.append(f"- **Metrics table rows:** `{runner_tables.get('metrics_table_rows')}`")
    lines.append(f"- **F7 sensitivity rows:** `{runner_tables.get('f7_sensitivity_rows')}`")
    lines.append(f"- **F8 sensitivity rows:** `{runner_tables.get('f8_sensitivity_rows')}`")
    lines.append(f"- **F12 sensitivity rows:** `{runner_tables.get('f12_sensitivity_rows')}`")
    lines.append("")

    lines.append("## Conditional results")
    lines.append("")
    lines.append(f"- **Protocol-only:** `{conditional_summary.get('protocol_only')}`")
    lines.append(f"- **Valid rows:** `{conditional_summary.get('n_valid_rows')}`")
    lines.append(f"- **Positive lift rows:** `{conditional_summary.get('n_positive_lift_rows')}`")
    lines.append(f"- **Primary positive lift rows:** `{conditional_summary.get('n_primary_positive_lift_rows')}`")
    lines.append(f"- **Strong candidate rows:** `{conditional_summary.get('n_strong_candidate_rows')}`")
    lines.append(f"- **Primary strong candidate rows:** `{conditional_summary.get('n_primary_strong_candidate_rows')}`")
    lines.append("")

    lines.append("## Controls")
    lines.append("")
    lines.append(f"- **Controls passed:** `{controls_summary.get('passed')}`")
    lines.append(f"- **Required control types:** `{controls_summary.get('required_control_types')}`")
    lines.append(f"- **Failed control types:** `{controls_summary.get('failed_control_types')}`")
    lines.append("")

    lines.append("## Gate sensitivity — F7/F8/F12")
    lines.append("")
    lines.append(f"- **Result:** `{gate_sensitivity.get('result')}`")
    lines.append(f"- **Official gate statuses:** `{gate_sensitivity.get('official_gate_statuses')}`")
    lines.append(f"- **F7 subject consistency:** `{gate_sensitivity.get('f7_subject_consistency')}`")
    lines.append(f"- **F8 complexity penalty:** `{gate_sensitivity.get('f8_complexity_penalty')}`")
    lines.append(f"- **F12 epsilon saturation:** `{gate_sensitivity.get('f12_epsilon_saturation')}`")
    lines.append("")

    lines.append("## Bundle contents")
    lines.append("")
    lines.append(f"- **Included files:** `{manifest.get('n_included_files')}`")
    lines.append(f"- **Missing expected files:** `{manifest.get('n_missing_files')}`")
    lines.append(f"- **Skipped heavy files:** `{manifest.get('n_skipped_files')}`")
    lines.append(f"- **Raw data files inventoried:** `{manifest.get('raw_data_inventory', {}).get('n_files')}`")
    lines.append("")
    lines.append("Raw EEG/RNG/QRNG files are not copied into the bundle. Their metadata and optional hashes are recorded in the manifest.")
    lines.append("")
    lines.append("Large CSV/ZIP/binary outputs are skipped by default to keep the audit package Git-safe.")
    lines.append("")

    return "\n".join(lines)


# =========================================================
# Zip creation
# =========================================================

def make_zip_from_directory(source_dir: Path, zip_path: Path) -> None:
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue

            arcname = path.relative_to(source_dir)
            zf.write(path, arcname=str(arcname).replace("\\", "/"))


# =========================================================
# Main builder
# =========================================================

def build_phase3_2e_audit_bundle(
    cfg: Optional[Phase32EConfig] = None,
    output_dir: Optional[Path] = None,
    keep_staging: bool = False,
    hash_raw_files: bool = True,
    raw_hash_max_mb: float = DEFAULT_RAW_HASH_MAX_MB,
    max_bundle_file_mb: float = DEFAULT_MAX_BUNDLE_FILE_MB,
    include_csv: bool = True,
) -> Dict[str, Any]:
    root = project_root()

    if cfg is None:
        cfg = load_phase3_2e_config()

    if output_dir is None:
        output_dir = resolve_project_path(root, cfg.results_dir) / "audit_bundle"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = _now_stamp()

    bundle_name = f"cdr_phase3_2e_audit_bundle_{stamp}"
    staging_dir = output_dir / f"{bundle_name}_staging"
    zip_path = output_dir / f"{bundle_name}.zip"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    staging_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: List[Dict[str, Any]] = []
    missing_files: List[Dict[str, Any]] = []
    skipped_files: List[Dict[str, Any]] = []

    print("\n" + "=" * 78)
    print("Building CDR Phase III.2E audit bundle")
    print("=" * 78)
    print(f"Project root: {root}")
    print(f"Bundle scope: {BUNDLE_SCOPE}")
    print(f"Staging dir: {staging_dir}")
    print(f"Zip path: {zip_path}")
    print(f"Max bundle file MB: {max_bundle_file_mb}")
    print(f"Include CSV: {include_csv}")

    # -----------------------------------------------------
    # Include source/config/docs
    # -----------------------------------------------------
    source_candidates = core_source_files(root)

    add_existing_files(
        root=root,
        staging_root=staging_dir,
        files=source_candidates,
        manifest_files=manifest_files,
        missing_files=missing_files,
        skipped_files=skipped_files,
        compute_hash=True,
        max_bundle_file_mb=float(max_bundle_file_mb),
        include_csv=bool(include_csv),
    )

    # -----------------------------------------------------
    # Include selected outputs
    # -----------------------------------------------------
    output_candidates = result_files(root, cfg)

    add_existing_files(
        root=root,
        staging_root=staging_dir,
        files=output_candidates,
        manifest_files=manifest_files,
        missing_files=missing_files,
        skipped_files=skipped_files,
        compute_hash=True,
        max_bundle_file_mb=float(max_bundle_file_mb),
        include_csv=bool(include_csv),
    )

    # -----------------------------------------------------
    # Raw data inventory only
    # -----------------------------------------------------
    raw_inventory = raw_data_inventory(
        root=root,
        hash_raw_files=bool(hash_raw_files),
        raw_hash_max_mb=float(raw_hash_max_mb),
    )

    status_summary = extract_status_summary(root, cfg)

    manifest: Dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "bundle_scope": BUNDLE_SCOPE,
        "bundle_scope_label": BUNDLE_SCOPE_LABEL,
        "created_at": _now_human(),
        "created_at_stamp": stamp,
        "project_root": str(root),
        "bundle_name": bundle_name,
        "zip_path": str(zip_path),
        "project": {
            "project_name": cfg.project_name,
            "phase_name": cfg.phase_name,
            "phase_title": cfg.phase_title,
            "version": cfg.version,
        },
        "subphases": {
            "III.2E": "synchronized EEG–QRNG protocol and gate-sensitivity validation",
        },
        "config": describe_config(cfg),
        "status_summary": status_summary,
        "included_files": manifest_files,
        "missing_files": missing_files,
        "skipped_files": skipped_files,
        "n_included_files": int(len(manifest_files)),
        "n_missing_files": int(len(missing_files)),
        "n_skipped_files": int(len(skipped_files)),
        "raw_data_inventory": raw_inventory,
        "git_safety": {
            "intended_for_git_commit": False,
            "do_not_commit_patterns": [
                "results/phase3_2e/audit_bundle/*.zip",
                "results/phase3_2e/audit_bundle/*_staging/",
                "results/phase3_2e/**/*.csv",
                "data/interim/phase3_2e/**/*.csv",
                "data/raw/**",
            ],
            "recommended_commit_scope": [
                "docs/phase3_2e_synchronized_eeg_qrng_protocol.md",
                "config/phase3_2e_config.py",
                "src/phase3_2e_*.py",
                "scripts/make_phase3_2e_audit_bundle.py",
            ],
        },
        "notes": [
            "This is the Phase III.2E audit bundle.",
            "The bundle name includes a timestamp and does not overwrite previous Phase III.2 A/B/C/D bundles.",
            "Raw EEG/RNG/QRNG files are inventoried but not copied into the audit zip.",
            "Large CSV/ZIP/binary outputs are skipped by default and recorded in skipped_files.",
            "Phase III.2E is currently protocol-ready but empirically pending synchronized EEG–QRNG data.",
            "Protocol-only outputs do not support an EEG–QRNG coupling claim.",
            "Empirical interpretation requires a real synchronized dataset and successful controls/gates.",
        ],
    }

    manifest_path = staging_dir / "audit_manifest.json"
    write_json(manifest_path, manifest)

    audit_summary_md = build_audit_summary_md(
        cfg=cfg,
        status_summary=status_summary,
        manifest=manifest,
    )

    summary_path = staging_dir / "AUDIT_SUMMARY.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(audit_summary_md)

    # Add manifest and summary to manifest records.
    manifest_files.append(
        file_record(
            root=staging_dir,
            path=manifest_path,
            category="audit_manifest",
            compute_hash=True,
        )
    )

    manifest_files.append(
        file_record(
            root=staging_dir,
            path=summary_path,
            category="audit_summary",
            compute_hash=True,
        )
    )

    # Rewrite manifest after adding manifest/summary records.
    manifest["included_files"] = manifest_files
    manifest["n_included_files"] = int(len(manifest_files))
    manifest["missing_files"] = missing_files
    manifest["n_missing_files"] = int(len(missing_files))
    manifest["skipped_files"] = skipped_files
    manifest["n_skipped_files"] = int(len(skipped_files))

    write_json(manifest_path, manifest)

    # -----------------------------------------------------
    # Create zip
    # -----------------------------------------------------
    make_zip_from_directory(staging_dir, zip_path)

    zip_sha256 = sha256_file(zip_path)

    final_report = {
        "bundle_version": BUNDLE_VERSION,
        "bundle_scope": BUNDLE_SCOPE,
        "bundle_scope_label": BUNDLE_SCOPE_LABEL,
        "created_at": _now_human(),
        "zip_path": str(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "zip_sha256": zip_sha256,
        "staging_dir": str(staging_dir),
        "staging_kept": bool(keep_staging),
        "n_included_files": int(len(manifest_files)),
        "n_missing_files": int(len(missing_files)),
        "n_skipped_files": int(len(skipped_files)),
        "status_summary": status_summary,
        "git_safety": manifest["git_safety"],
    }

    final_report_path = output_dir / f"{bundle_name}_bundle_report.json"
    write_json(final_report_path, final_report)

    if not keep_staging:
        shutil.rmtree(staging_dir)

    print("\n" + "=" * 78)
    print("Phase III.2E audit bundle completed")
    print("=" * 78)
    print(f"Zip: {zip_path}")
    print(f"SHA256: {zip_sha256}")
    print(f"Bundle report: {final_report_path}")
    print(f"Included files: {len(manifest_files)}")
    print(f"Missing expected files: {len(missing_files)}")
    print(f"Skipped heavy files: {len(skipped_files)}")
    print("=" * 78 + "\n")

    return final_report


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an audit bundle for CDR Phase III.2E."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for the audit bundle.",
    )

    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the unzipped staging directory after creating the zip.",
    )

    parser.add_argument(
        "--no-raw-hash",
        action="store_true",
        help="Do not compute SHA256 hashes for raw data inventory files.",
    )

    parser.add_argument(
        "--raw-hash-max-mb",
        type=float,
        default=DEFAULT_RAW_HASH_MAX_MB,
        help="Maximum raw file size in MB for computing SHA256 in the raw data inventory.",
    )

    parser.add_argument(
        "--max-bundle-file-mb",
        type=float,
        default=DEFAULT_MAX_BUNDLE_FILE_MB,
        help="Maximum individual file size in MB to copy into the audit zip.",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Exclude CSV files from the audit zip and record them as skipped.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    output_dir = Path(args.output_dir) if args.output_dir else None

    build_phase3_2e_audit_bundle(
        cfg=cfg,
        output_dir=output_dir,
        keep_staging=bool(args.keep_staging),
        hash_raw_files=not bool(args.no_raw_hash),
        raw_hash_max_mb=float(args.raw_hash_max_mb),
        max_bundle_file_mb=float(args.max_bundle_file_mb),
        include_csv=not bool(args.no_csv),
    )


if __name__ == "__main__":
    main()