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
# Evidence-guided surrogate replication / safe-resume aware
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
#       The synchronized surrogate execution completed operationally and
#       identified a registered primary localized predictive lead.
#
#       Required negative controls passed. The official F8/BIC gate remained
#       unsatisfied, so the v2 result is preserved as BIC-limited.
#
#       Surrogate provenance blocks any empirical EEG–QRNG claim.
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

BUNDLE_VERSION = (
    "phase3_2e_audit_bundle_v2_"
    "surrogate_primary_lead_bic_limited_safe_resume"
)
BUNDLE_SCOPE = "Phase III.2E"
BUNDLE_SCOPE_LABEL = "phase3_2e"

EXPECTED_OPERATIONAL_STATUS = "pipeline_completed"
EXPECTED_SCIENTIFIC_STATUS = "primary_lead_bic_limited"
EXPECTED_SCIENTIFIC_CLASSIFICATION = (
    "surrogate_primary_localized_predictive_lead_bic_limited"
)

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


def first_existing(candidates: List[Path]) -> Path:
    for path in candidates:
        if Path(path).exists():
            return Path(path)

    return Path(candidates[0])


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "sim", "t"}:
        return True

    if text in {"false", "0", "no", "n", "nao", "não", "f"}:
        return False

    return default


def file_status(path: Path) -> Dict[str, Any]:
    path = Path(path)

    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": (
            int(path.stat().st_size)
            if path.exists() and path.is_file()
            else 0
        ),
    }


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
        getattr(
            cfg,
            "diagnostics_dir",
            results_dir / "diagnostics",
        ),
    )

    synchronized_input_csv = resolve_project_path(
        root,
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
        ),
    )

    sync_report = first_existing(
        [
            results_dir / "phase3_2e_sync_report.json",
            results_dir / "phase3_2e_sync_validation_report.json",
        ]
    )

    sync_summary = first_existing(
        [
            results_dir / "phase3_2e_sync_summary.txt",
            results_dir / "phase3_2e_sync_validation_summary.txt",
        ]
    )

    return {
        # Input/protocol
        "synchronized_input_csv": synchronized_input_csv,

        # Schema/sync
        "schema_report":
            results_dir / "phase3_2e_schema_report.json",
        "schema_summary":
            results_dir / "phase3_2e_schema_summary.txt",
        "sync_validation_report": sync_report,
        "sync_validation_summary": sync_summary,

        # Loader/windows/features/alignment
        "loaded_windows":
            interim_dir / "phase3_2e_loaded_synchronized_windows.csv",
        "loader_metadata":
            interim_dir / "phase3_2e_loader_metadata.json",
        "loader_report":
            results_dir / "phase3_2e_loader_report.json",
        "loader_summary":
            results_dir / "phase3_2e_loader_summary.txt",

        "windows":
            interim_dir / "phase3_2e_windows.csv",
        "window_inventory":
            interim_dir / "phase3_2e_window_inventory.csv",
        "windows_report":
            results_dir / "phase3_2e_windows_report.json",
        "windows_summary":
            results_dir / "phase3_2e_windows_summary.txt",

        "features":
            interim_dir / "phase3_2e_features.csv",
        "feature_specs":
            interim_dir / "phase3_2e_feature_specs.json",
        "features_report":
            results_dir / "phase3_2e_features_report.json",
        "features_summary":
            results_dir / "phase3_2e_features_summary.txt",

        "alignment_inventory":
            interim_dir / "phase3_2e_alignment_inventory.csv",
        "alignment_metadata":
            interim_dir / "phase3_2e_alignment_metadata.json",
        "alignment_report":
            results_dir / "phase3_2e_alignment_report.json",
        "alignment_summary":
            results_dir / "phase3_2e_alignment_summary.txt",

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
        "conditional_report":
            results_dir / "phase3_2e_conditional_report.json",

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
        "controls_report":
            results_dir / "phase3_2e_controls_report.json",

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
        "metrics_table":
            results_dir / "phase3_2e_metrics_table.csv",
        "metrics_summary":
            results_dir / "phase3_2e_metrics_summary.txt",
        "metrics_report":
            results_dir / "phase3_2e_metrics_report.json",

        # Gate sensitivity
        "gate_sensitivity_report":
            results_dir / "phase3_2e_gate_sensitivity_report.json",
        "gate_sensitivity_summary":
            results_dir / "phase3_2e_gate_sensitivity_summary.txt",
        "f7_sensitivity":
            results_dir / "phase3_2e_f7_subject_sensitivity.csv",
        "f8_sensitivity":
            results_dir / "phase3_2e_f8_complexity_sensitivity.csv",
        "f12_sensitivity":
            results_dir
            / "phase3_2e_f12_epsilon_saturation_sensitivity.csv",

        # Diagnostics
        "diagnostics_dir": diagnostics_dir,
        "diagnostics_report":
            diagnostics_dir / "phase3_2e_diagnostics_report.json",
        "diagnostics_summary":
            diagnostics_dir / "phase3_2e_diagnostics_summary.txt",

        # Runner
        "runner_report":
            results_dir / "phase3_2e_runner_report.json",
        "runner_summary":
            results_dir / "phase3_2e_runner_summary.txt",
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
# Audit input validation
# =========================================================

def validate_required_artifacts(
    root: Path,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    paths = phase3_2e_paths(root, cfg)

    required = {
        "schema_report": paths["schema_report"],
        "sync_validation_report": paths["sync_validation_report"],
        "loader_report": paths["loader_report"],
        "windows_report": paths["windows_report"],
        "features_report": paths["features_report"],
        "alignment_report": paths["alignment_report"],
        "conditional_model_scores": paths["conditional_model_scores"],
        "conditional_pair_scores": paths["conditional_pair_scores"],
        "controls_json": paths["controls_json"],
        "control_runs": paths["control_runs"],
        "control_pair_scores": paths["control_pair_scores"],
        "metrics_report": paths["metrics_report"],
        "gates_json": paths["gates_json"],
        "gate_sensitivity_report": paths["gate_sensitivity_report"],
        "diagnostics_report": paths["diagnostics_report"],
        "runner_report": paths["runner_report"],
    }

    statuses = {
        name: file_status(path)
        for name, path in required.items()
    }

    ready = all(
        item["exists"]
        and item["is_file"]
        and item["size_bytes"] > 0
        for item in statuses.values()
    )

    runner_report = safe_read_json(paths["runner_report"])
    runner_final = _safe_dict(runner_report.get("final_status"))
    runner_options = _safe_dict(runner_report.get("options"))

    return {
        "ready": ready,
        "required_files": statuses,
        "runner_consistency": {
            "runner_version": runner_report.get("runner_version"),
            "operational_status": runner_final.get("operational_status"),
            "scientific_status": runner_final.get("scientific_status"),
            "scientific_classification":
                runner_final.get("scientific_classification"),
            "candidate_detected": runner_final.get("candidate_detected"),
            "surrogate": runner_final.get("surrogate"),
            "can_claim_empirical_cdr":
                runner_final.get("can_claim_empirical_cdr"),
            "safe_resume": runner_options.get("safe_resume"),
            "reuse_conditional":
                runner_options.get("reuse_conditional"),
            "reuse_controls": runner_options.get("reuse_controls"),
        },
        "read_only": True,
        "note": (
            "The audit builder validates and copies existing artifacts. "
            "It does not rerun analytical modules."
        ),
    }


def _gate_status_lists(
    gates_json: Dict[str, Any],
) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {
        "PASS": [],
        "FAIL": [],
        "WARN": [],
        "NOT_EVALUATED": [],
    }

    for gate in _safe_list(gates_json.get("gates")):
        if not isinstance(gate, dict):
            continue

        status = str(gate.get("status", "")).upper()
        gate_id = str(gate.get("gate", "")).strip()

        if status in grouped and gate_id:
            grouped[status].append(gate_id)

    return {
        "passed_gates": grouped["PASS"],
        "failed_gates": grouped["FAIL"],
        "warning_gates": grouped["WARN"],
        "not_evaluated_gates": grouped["NOT_EVALUATED"],
    }


# =========================================================
# Status extraction
# =========================================================

def extract_phase3_2e_status(
    root: Path,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
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
    gates_json = safe_read_json(paths["gates_json"])
    sensitivity_report = safe_read_json(
        paths["gate_sensitivity_report"]
    )
    diagnostics_report = safe_read_json(paths["diagnostics_report"])
    runner_report = safe_read_json(paths["runner_report"])

    runner_final = _safe_dict(runner_report.get("final_status"))
    runner_options = _safe_dict(runner_report.get("options"))
    metrics_final = _safe_dict(metrics_report.get("final_status"))
    diagnostics_final = _safe_dict(
        diagnostics_report.get("final_interpretation")
    )
    conditional_diagnostics = _safe_dict(
        diagnostics_report.get("conditional_diagnostics")
    )
    control_diagnostics = _safe_dict(
        diagnostics_report.get("control_diagnostics")
    )
    sensitivity_result = _safe_dict(
        sensitivity_report.get("result")
    )

    gate_lists = _gate_status_lists(gates_json)

    operational_status = runner_final.get(
        "operational_status",
        runner_final.get("status", "unknown"),
    )
    scientific_status = runner_final.get(
        "scientific_status",
        metrics_final.get("status", metrics_report.get("status")),
    )
    scientific_classification = runner_final.get(
        "scientific_classification",
        diagnostics_final.get(
            "scientific_classification",
            scientific_status,
        ),
    )
    candidate_detected = _safe_bool(
        runner_final.get(
            "candidate_detected",
            conditional_diagnostics.get(
                "localized_predictive_support",
                False,
            ),
        )
    )
    surrogate = _safe_bool(
        runner_final.get(
            "surrogate",
            diagnostics_final.get("surrogate", False),
        )
    )
    protocol_only = _safe_bool(
        runner_final.get(
            "protocol_only",
            metrics_final.get("protocol_only", False),
        )
    )
    can_claim = _safe_bool(
        runner_final.get(
            "can_claim_empirical_cdr",
            metrics_final.get("can_claim_empirical_cdr", False),
        )
    )

    headline = runner_final.get(
        "headline",
        diagnostics_final.get("headline"),
    )
    recommendation = runner_final.get(
        "recommendation",
        diagnostics_final.get("recommendation"),
    )

    primary_lead = (
        metrics_report.get("selected_primary_lead")
        or conditional_diagnostics.get("selected_primary_lead")
        or {}
    )

    controls_summary = _safe_dict(controls_report.get("summary"))
    controls_result = _safe_dict(controls_report.get("result"))
    controls_passed = _safe_bool(
        control_diagnostics.get(
            "passed",
            controls_summary.get(
                "passed",
                controls_result.get("overall_passed", False),
            ),
        )
    )

    synchronized_input = paths["synchronized_input_csv"]

    return {
        "scope": BUNDLE_SCOPE,
        "expected_operational_status": EXPECTED_OPERATIONAL_STATUS,
        "expected_scientific_status": EXPECTED_SCIENTIFIC_STATUS,
        "expected_scientific_classification":
            EXPECTED_SCIENTIFIC_CLASSIFICATION,

        "synchronized_input_csv": str(synchronized_input),
        "synchronized_input_exists": synchronized_input.exists(),

        "schema_status": schema_report.get("status"),
        "sync_validation_status": sync_report.get("status"),
        "loader_status": loader_report.get("status"),
        "windows_status": windows_report.get("status"),
        "features_status": features_report.get("status"),
        "alignment_status": alignment_report.get("status"),
        "conditional_cached_status": conditional_report.get("status"),
        "controls_status": controls_report.get("status"),
        "metrics_status": metrics_report.get("status"),
        "gate_sensitivity_status": sensitivity_report.get("status"),

        "operational_status": operational_status,
        "scientific_status": scientific_status,
        "scientific_classification": scientific_classification,
        "candidate_detected": candidate_detected,
        "localized_predictive_support": _safe_bool(
            conditional_diagnostics.get(
                "localized_predictive_support",
                candidate_detected,
            )
        ),
        "surrogate": surrogate,
        "protocol_only": protocol_only,
        "can_claim_empirical_cdr": can_claim,
        "controls_passed": controls_passed,

        "passed_gates": (
            runner_final.get("passed_gates")
            or metrics_final.get("passed_gates")
            or gate_lists["passed_gates"]
        ),
        "failed_gates": (
            runner_final.get("failed_gates")
            or metrics_final.get("failed_gates")
            or gate_lists["failed_gates"]
        ),
        "warning_gates": (
            runner_final.get("warning_gates")
            or metrics_final.get("warning_gates")
            or gate_lists["warning_gates"]
        ),
        "not_evaluated_gates": (
            runner_final.get("not_evaluated_gates")
            or metrics_final.get("not_evaluated_gates")
            or gate_lists["not_evaluated_gates"]
        ),

        "selected_primary_lead": primary_lead,
        "headline": headline,
        "recommendation": recommendation,

        "safe_resume": _safe_bool(
            runner_options.get("safe_resume", False)
        ),
        "reuse_conditional": _safe_bool(
            runner_options.get("reuse_conditional", False)
        ),
        "reuse_controls": _safe_bool(
            runner_options.get("reuse_controls", False)
        ),

        "runner_final_status": runner_final,
        "metrics_final_status": metrics_final,
        "gate_sensitivity_result": sensitivity_result,
        "diagnostics_final_interpretation": diagnostics_final,

        "interpretation": (
            "Phase III.2E completed operationally on synchronized "
            "surrogate data and identified a registered primary "
            "localized predictive lead. Negative controls passed. "
            "The official F8/BIC gate remained unsatisfied, and "
            "surrogate provenance blocks any empirical CDR claim."
            if surrogate and candidate_detected
            else (
                headline
                or "Phase III.2E status was extracted from existing outputs."
            )
        ),
    }

def extract_table_summary(
    root: Path,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    paths = phase3_2e_paths(root, cfg)
    runner_report = safe_read_json(paths["runner_report"])
    diagnostics_report = safe_read_json(paths["diagnostics_report"])

    runner_table = _safe_dict(
        runner_report.get("table_summary")
    )
    diagnostics_tables = _safe_dict(
        diagnostics_report.get(
            "artifact_rows",
            diagnostics_report.get(
                "artifact_table_diagnostics",
                {},
            ),
        )
    )

    return {
        "runner_table_summary": runner_table,
        "diagnostics_table_summary": diagnostics_tables,
    }

def extract_status_summary(
    root: Path,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    paths = phase3_2e_paths(root, cfg)

    controls_json = safe_read_json(paths["controls_json"])
    metrics_report = safe_read_json(paths["metrics_report"])
    gates_json = safe_read_json(paths["gates_json"])
    conditional_report = safe_read_json(paths["conditional_report"])
    sensitivity_report = safe_read_json(
        paths["gate_sensitivity_report"]
    )
    diagnostics_report = safe_read_json(paths["diagnostics_report"])
    runner_report = safe_read_json(paths["runner_report"])
    mc_guard = safe_read_json(paths["multiple_comparison_guard"])

    controls_summary = _safe_dict(controls_json.get("summary"))
    controls_result = _safe_dict(controls_json.get("result"))

    if not controls_summary:
        controls_summary = {
            "passed": controls_result.get("overall_passed"),
            "failed_control_types":
                controls_result.get("failed_control_types", []),
        }

    conditional_summary = _safe_dict(
        conditional_report.get("summary")
    )

    if not conditional_summary:
        conditional_summary = _safe_dict(
            diagnostics_report.get("conditional_diagnostics")
        )

    return {
        "bundle_scope": BUNDLE_SCOPE,
        "bundle_scope_label": BUNDLE_SCOPE_LABEL,
        "phase3_2e_status": extract_phase3_2e_status(root, cfg),
        "table_summary": extract_table_summary(root, cfg),
        "runner_final_status":
            _safe_dict(runner_report.get("final_status")),
        "runner_options":
            _safe_dict(runner_report.get("options")),
        "diagnostics_final_interpretation":
            _safe_dict(
                diagnostics_report.get("final_interpretation")
            ),
        "metrics_final_status":
            _safe_dict(metrics_report.get("final_status")),
        "selected_primary_lead":
            metrics_report.get("selected_primary_lead", {}),
        "gates": _safe_list(gates_json.get("gates")),
        "controls_summary": controls_summary,
        "conditional_summary": conditional_summary,
        "multiple_comparison_guard": mc_guard,
        "gate_sensitivity_summary": {
            "result":
                _safe_dict(sensitivity_report.get("result")),
            "official_gate_statuses":
                _safe_dict(
                    sensitivity_report.get("official_gate_statuses")
                ),
            "f7_subject_consistency":
                _safe_dict(
                    sensitivity_report.get("f7_subject_consistency")
                ),
            "f8_complexity_penalty":
                _safe_dict(
                    sensitivity_report.get("f8_complexity_penalty")
                ),
            "f12_epsilon_saturation":
                _safe_dict(
                    sensitivity_report.get("f12_epsilon_saturation")
                ),
        },
    }

# =========================================================
# Audit summary markdown
# =========================================================

def build_audit_summary_md(
    cfg: Phase32EConfig,
    status_summary: Dict[str, Any],
    manifest: Dict[str, Any],
) -> str:
    phase = _safe_dict(status_summary.get("phase3_2e_status"))
    controls = _safe_dict(status_summary.get("controls_summary"))
    conditional = _safe_dict(status_summary.get("conditional_summary"))
    sensitivity = _safe_dict(
        status_summary.get("gate_sensitivity_summary")
    )
    tables = _safe_dict(status_summary.get("table_summary"))
    runner_tables = _safe_dict(
        tables.get("runner_table_summary")
    )
    lead = _safe_dict(status_summary.get("selected_primary_lead"))
    runner_options = _safe_dict(status_summary.get("runner_options"))

    lines: List[str] = [
        "# CDR Phase III.2E — Audit Bundle",
        "",
        f"- **Bundle version:** `{BUNDLE_VERSION}`",
        f"- **Bundle scope:** `{BUNDLE_SCOPE}`",
        f"- **Created at:** `{manifest.get('created_at')}`",
        f"- **Project:** `{cfg.project_name}`",
        f"- **Phase:** `{cfg.phase_name}`",
        f"- **Phase title:** `{cfg.phase_title}`",
        f"- **Code/config version:** `{cfg.version}`",
        "",
        "## Final status",
        "",
        (
            "- **Operational status:** "
            f"`{phase.get('operational_status')}`"
        ),
        (
            "- **Scientific status:** "
            f"`{phase.get('scientific_status')}`"
        ),
        (
            "- **Scientific classification:** "
            f"`{phase.get('scientific_classification')}`"
        ),
        (
            "- **Candidate detected:** "
            f"`{phase.get('candidate_detected')}`"
        ),
        (
            "- **Localized predictive support:** "
            f"`{phase.get('localized_predictive_support')}`"
        ),
        f"- **Surrogate:** `{phase.get('surrogate')}`",
        f"- **Protocol-only:** `{phase.get('protocol_only')}`",
        (
            "- **Can claim empirical CDR:** "
            f"`{phase.get('can_claim_empirical_cdr')}`"
        ),
        (
            "- **Controls passed:** "
            f"`{phase.get('controls_passed')}`"
        ),
        "",
        "**Headline:**",
        "",
        f"> {phase.get('headline')}",
        "",
        "**Interpretation:**",
        "",
        f"> {phase.get('interpretation')}",
        "",
        "**Recommendation:**",
        "",
        f"> {phase.get('recommendation')}",
        "",
        "## Scientific guardrail",
        "",
        (
            "- The current input is a synchronized surrogate dataset, "
            "not a real simultaneous EEG–QRNG acquisition."
        ),
        (
            "- Candidate detection and gate behavior validate the "
            "pipeline but do not establish empirical coupling."
        ),
        (
            "- The v2 result remains BIC-limited because the official "
            "F8 rule did not favor the augmented model."
        ),
        (
            "- Any future F8 revision must be prospective and defined "
            "before analysis of real synchronized data."
        ),
        "",
        "## Safe-resume execution",
        "",
        (
            "- **Safe resume:** "
            f"`{runner_options.get('safe_resume')}`"
        ),
        (
            "- **Conditional reused:** "
            f"`{runner_options.get('reuse_conditional')}`"
        ),
        (
            "- **Controls reused:** "
            f"`{runner_options.get('reuse_controls')}`"
        ),
        (
            "- The audit builder is read-only and does not rerun "
            "conditional, controls, metrics, sensitivity or diagnostics."
        ),
        "",
        "## Pipeline status",
        "",
        f"- **Schema:** `{phase.get('schema_status')}`",
        (
            "- **Synchronization validator:** "
            f"`{phase.get('sync_validation_status')}`"
        ),
        f"- **Loader:** `{phase.get('loader_status')}`",
        f"- **Windows:** `{phase.get('windows_status')}`",
        f"- **Features:** `{phase.get('features_status')}`",
        f"- **Alignment:** `{phase.get('alignment_status')}`",
        (
            "- **Conditional cached label:** "
            f"`{phase.get('conditional_cached_status')}`"
        ),
        f"- **Controls:** `{phase.get('controls_status')}`",
        f"- **Metrics:** `{phase.get('metrics_status')}`",
        (
            "- **Gate sensitivity:** "
            f"`{phase.get('gate_sensitivity_status')}`"
        ),
        "",
        (
            "> The cached conditional report may retain older "
            "exploratory wording. The authoritative classification "
            "comes from metrics, sensitivity, diagnostics and runner."
        ),
        "",
        "## Official gates",
        "",
        f"- **Passed:** `{phase.get('passed_gates')}`",
        f"- **Failed:** `{phase.get('failed_gates')}`",
        f"- **Warnings:** `{phase.get('warning_gates')}`",
        (
            "- **Not evaluated:** "
            f"`{phase.get('not_evaluated_gates')}`"
        ),
        "",
        "## Selected primary lead",
        "",
        f"- **Window seconds:** `{lead.get('window_seconds')}`",
        f"- **Lag windows:** `{lead.get('lag_windows')}`",
        f"- **Lag seconds:** `{lead.get('lag_seconds')}`",
        f"- **Baseline model:** `{lead.get('baseline_model')}`",
        f"- **Augmented model:** `{lead.get('augmented_model')}`",
        f"- **Conditional lift:** `{lead.get('conditional_lift')}`",
        f"- **Held-out LL improvement:** `{lead.get('ll_improvement')}`",
        f"- **BIC improvement:** `{lead.get('bic_improvement')}`",
        f"- **AIC improvement:** `{lead.get('aic_improvement')}`",
        (
            "- **Positive subjects:** "
            f"`{lead.get('n_subjects_positive_lift')}/"
            f"{lead.get('n_subjects_evaluated')}`"
        ),
        "",
        "## Runner table summary",
        "",
        f"- **Loaded rows:** `{runner_tables.get('loaded_rows')}`",
        f"- **Window rows:** `{runner_tables.get('window_rows')}`",
        f"- **Feature rows:** `{runner_tables.get('feature_rows')}`",
        (
            "- **Alignment inventory rows:** "
            f"`{runner_tables.get('alignment_inventory_rows')}`"
        ),
        (
            "- **Conditional model rows:** "
            f"`{runner_tables.get('conditional_model_rows')}`"
        ),
        (
            "- **Conditional pair rows:** "
            f"`{runner_tables.get('conditional_pair_rows')}`"
        ),
        (
            "- **Primary positive rows:** "
            f"`{runner_tables.get('conditional_primary_positive_lift_rows')}`"
        ),
        (
            "- **Control run rows:** "
            f"`{runner_tables.get('control_run_rows')}`"
        ),
        (
            "- **Control pair rows:** "
            f"`{runner_tables.get('control_pair_rows')}`"
        ),
        (
            "- **Metrics table rows:** "
            f"`{runner_tables.get('metrics_table_rows')}`"
        ),
        (
            "- **F7 sensitivity rows:** "
            f"`{runner_tables.get('f7_sensitivity_rows')}`"
        ),
        (
            "- **F8 sensitivity rows:** "
            f"`{runner_tables.get('f8_sensitivity_rows')}`"
        ),
        (
            "- **F12 sensitivity rows:** "
            f"`{runner_tables.get('f12_sensitivity_rows')}`"
        ),
        "",
        "## Conditional and controls",
        "",
        (
            "- **Conditional valid rows:** "
            f"`{conditional.get('n_valid_rows')}`"
        ),
        (
            "- **Conditional positive rows:** "
            f"`{conditional.get('n_positive_lift_rows')}`"
        ),
        (
            "- **Primary positive rows:** "
            f"`{conditional.get('n_primary_positive_lift_rows')}`"
        ),
        (
            "- **Controls passed:** "
            f"`{controls.get('passed')}`"
        ),
        (
            "- **Required control types:** "
            f"`{controls.get('required_control_types')}`"
        ),
        (
            "- **Failed control types:** "
            f"`{controls.get('failed_control_types')}`"
        ),
        "",
        "## Gate sensitivity — F7/F8/F12",
        "",
        (
            "- **Official gate statuses:** "
            f"`{sensitivity.get('official_gate_statuses')}`"
        ),
        (
            "- **F7 subject consistency:** "
            f"`{sensitivity.get('f7_subject_consistency')}`"
        ),
        (
            "- **F8 complexity penalty:** "
            f"`{sensitivity.get('f8_complexity_penalty')}`"
        ),
        (
            "- **F12 epsilon saturation:** "
            f"`{sensitivity.get('f12_epsilon_saturation')}`"
        ),
        "",
        "## Bundle contents",
        "",
        (
            "- **Included files:** "
            f"`{manifest.get('n_included_files')}`"
        ),
        (
            "- **Missing expected files:** "
            f"`{manifest.get('n_missing_files')}`"
        ),
        (
            "- **Skipped heavy files:** "
            f"`{manifest.get('n_skipped_files')}`"
        ),
        (
            "- **Raw files inventoried:** "
            f"`{_safe_dict(manifest.get('raw_data_inventory')).get('n_files')}`"
        ),
        "",
        (
            "Raw EEG/RNG/QRNG files are inventoried but not copied "
            "into the bundle. Large files are skipped according to "
            "the configured size and suffix rules."
        ),
        "",
    ]

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
    cfg = cfg or load_phase3_2e_config()

    validation = validate_required_artifacts(root, cfg)

    if not validation["ready"]:
        invalid = [
            name
            for name, status
            in validation["required_files"].items()
            if not (
                status["exists"]
                and status["is_file"]
                and status["size_bytes"] > 0
            )
        ]
        raise RuntimeError(
            "Audit bundle cannot be built because required artifacts "
            f"are missing or empty: {invalid}"
        )

    if output_dir is None:
        output_dir = (
            resolve_project_path(root, cfg.results_dir)
            / "audit_bundle"
        )

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
    print("Analytical stages executed: none")

    add_existing_files(
        root=root,
        staging_root=staging_dir,
        files=core_source_files(root),
        manifest_files=manifest_files,
        missing_files=missing_files,
        skipped_files=skipped_files,
        compute_hash=True,
        max_bundle_file_mb=float(max_bundle_file_mb),
        include_csv=bool(include_csv),
    )

    add_existing_files(
        root=root,
        staging_root=staging_dir,
        files=result_files(root, cfg),
        manifest_files=manifest_files,
        missing_files=missing_files,
        skipped_files=skipped_files,
        compute_hash=True,
        max_bundle_file_mb=float(max_bundle_file_mb),
        include_csv=bool(include_csv),
    )

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
        "execution_mode": {
            "audit_builder_read_only": True,
            "conditional_executed": False,
            "controls_executed": False,
            "metrics_executed": False,
            "gate_sensitivity_executed": False,
            "diagnostics_executed": False,
        },
        "config": describe_config(cfg),
        "validation": validation,
        "status_summary": status_summary,
        "included_files": manifest_files,
        "missing_files": missing_files,
        "skipped_files": skipped_files,
        "n_included_files": len(manifest_files) + 2,
        "n_missing_files": len(missing_files),
        "n_skipped_files": len(skipped_files),
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
            "This is the Phase III.2E v2 audit bundle.",
            (
                "The synchronized input is surrogate and cannot support "
                "an empirical EEG–QRNG claim."
            ),
            (
                "A registered primary localized predictive lead was found "
                "and survived the required negative controls."
            ),
            (
                "The official F8/BIC gate remained unsatisfied; the current "
                "result is preserved as BIC-limited."
            ),
            (
                "Conditional and controls were reused by the safe-resume "
                "runner and were not recomputed by this builder."
            ),
            (
                "The cached conditional status may retain older exploratory "
                "wording; the authoritative classification comes from "
                "metrics, sensitivity, diagnostics and runner."
            ),
            (
                "Raw EEG/RNG/QRNG files are inventoried but not copied "
                "into the ZIP."
            ),
            (
                "Large CSV, archive and binary outputs are skipped according "
                "to bundle rules and recorded in skipped_files."
            ),
        ],
    }

    summary_path = staging_dir / "AUDIT_SUMMARY.md"
    summary_path.write_text(
        build_audit_summary_md(
            cfg=cfg,
            status_summary=status_summary,
            manifest=manifest,
        ),
        encoding="utf-8",
    )

    manifest_files.append(
        file_record(
            root=staging_dir,
            path=summary_path,
            category="audit_summary",
            compute_hash=True,
        )
    )

    manifest_path = staging_dir / "audit_manifest.json"

    manifest_files.append(
        {
            "category": "audit_manifest",
            "path": "audit_manifest.json",
            "absolute_path": str(manifest_path),
            "size_bytes": None,
            "mtime": None,
            "mtime_human": None,
            "sha256": None,
            "hash_status": "self_referential_not_hashed",
        }
    )

    manifest["included_files"] = manifest_files
    manifest["n_included_files"] = len(manifest_files)
    manifest["missing_files"] = missing_files
    manifest["n_missing_files"] = len(missing_files)
    manifest["skipped_files"] = skipped_files
    manifest["n_skipped_files"] = len(skipped_files)

    write_json(manifest_path, manifest)

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
        "n_included_files": len(manifest_files),
        "n_missing_files": len(missing_files),
        "n_skipped_files": len(skipped_files),
        "status_summary": status_summary,
        "validation": validation,
        "git_safety": manifest["git_safety"],
    }

    final_report_path = (
        output_dir / f"{bundle_name}_bundle_report.json"
    )
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
    print("Analytical stages executed: none")
    print("=" * 78 + "\n")

    return final_report

# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only audit bundle for CDR Phase III.2E."
        )
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
        help=(
            "Keep the unzipped staging directory after creating the ZIP."
        ),
    )

    parser.add_argument(
        "--no-raw-hash",
        action="store_true",
        help=(
            "Do not compute SHA256 hashes for raw data inventory files."
        ),
    )

    parser.add_argument(
        "--raw-hash-max-mb",
        type=float,
        default=DEFAULT_RAW_HASH_MAX_MB,
        help=(
            "Maximum raw file size in MB for computing SHA256."
        ),
    )

    parser.add_argument(
        "--max-bundle-file-mb",
        type=float,
        default=DEFAULT_MAX_BUNDLE_FILE_MB,
        help=(
            "Maximum individual file size in MB to copy into the ZIP."
        ),
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help=(
            "Exclude CSV files from the ZIP and record them as skipped."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate required artifacts without creating a bundle."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_phase3_2e_config()
    root = project_root()

    if args.validate_only:
        validation = validate_required_artifacts(root, cfg)

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
            if validation["ready"]
            else 1
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else None
    )

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
