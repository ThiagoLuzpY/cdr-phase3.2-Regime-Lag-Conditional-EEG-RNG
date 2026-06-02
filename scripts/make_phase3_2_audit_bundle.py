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

from config.phase3_2_config import Phase32Config, describe_config, load_phase3_2_config


# =========================================================
# Phase III.2 — Audit Bundle Builder
# =========================================================
#
# Purpose:
#
#   Build a reproducibility/audit bundle for CDR Phase III.2.
#
# The bundle includes:
#
#   - core Phase III.2 source code
#   - config
#   - README / requirements if available
#   - runner report and summary
#   - conditional model outputs
#   - control outputs
#   - metrics/gates outputs
#   - diagnostics outputs
#   - selected interim inventories/specs
#   - raw data inventory manifest, without copying raw EDF/RNG files
#   - SHA256 hashes for included files
#
# It does NOT copy raw EDF files into the zip by default.
# Instead, it records raw file names, sizes, mtimes and optional hashes.
#
# Run:
#
#   python -m scripts.make_phase3_2_audit_bundle
#


# =========================================================
# Constants
# =========================================================

BUNDLE_VERSION = "phase3_2_audit_bundle_v1"
EXPECTED_FINAL_STATUS = "exploratory_lead_with_saturation_warning"


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

    rel = path.relative_to(root) if path.is_relative_to(root) else path

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


def copy_into_staging(
    root: Path,
    staging_root: Path,
    source_path: Path,
    category: str,
    manifest_files: List[Dict[str, Any]],
    missing_files: List[Dict[str, Any]],
    compute_hash: bool = True,
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
    compute_hash: bool = True,
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
            compute_hash=compute_hash,
        )


def add_directory_files(
    root: Path,
    staging_root: Path,
    directory: Path,
    category: str,
    manifest_files: List[Dict[str, Any]],
    missing_files: List[Dict[str, Any]],
    patterns: Optional[List[str]] = None,
    compute_hash: bool = True,
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
        patterns = ["*"]

    seen: set[str] = set()

    for pattern in patterns:
        for path in directory.rglob(pattern):
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
                compute_hash=compute_hash,
            )


# =========================================================
# Raw data inventory
# =========================================================

def raw_data_inventory(
    root: Path,
    hash_raw_files: bool = True,
    raw_hash_max_mb: float = 250.0,
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
# Phase III.2 file selection
# =========================================================

def core_source_files(root: Path) -> List[Tuple[str, Path]]:
    candidates = [
        ("project_docs", root / "README.md"),
        ("project_docs", root / "requirements.txt"),
        ("project_docs", root / "pyproject.toml"),
        ("project_docs", root / "setup.cfg"),
        ("project_docs", root / ".gitignore"),

        ("config", root / "config" / "phase3_2_config.py"),

        ("source", root / "src" / "phase3_2_loader.py"),
        ("source", root / "src" / "phase3_2_features.py"),
        ("source", root / "src" / "phase3_2_regimes.py"),
        ("source", root / "src" / "phase3_2_lagging.py"),
        ("source", root / "src" / "phase3_2_conditional.py"),
        ("source", root / "src" / "phase3_2_controls.py"),
        ("source", root / "src" / "phase3_2_metrics.py"),
        ("source", root / "src" / "phase3_2_diagnostics.py"),
        ("source", root / "src" / "phase3_2_runner.py"),

        ("script", root / "scripts" / "make_phase3_2_audit_bundle.py"),
    ]

    return candidates


def result_files(root: Path, cfg: Phase32Config) -> List[Tuple[str, Path]]:
    results_dir = resolve_project_path(root, cfg.results_dir)
    interim_dir = resolve_project_path(root, cfg.interim_dir)
    diagnostics_dir = resolve_project_path(root, cfg.diagnostics_dir)

    candidates = [
        # Runner
        ("runner_output", results_dir / "phase3_2_runner_report.json"),
        ("runner_output", results_dir / "phase3_2_runner_summary.txt"),

        # Conditional
        ("conditional_output", results_dir / "phase3_2c_conditional_model_scores.csv"),
        ("conditional_output", resolve_project_path(root, cfg.conditional_results_csv)),
        ("conditional_output", resolve_project_path(root, cfg.conditional_summary_json)),
        ("conditional_output", results_dir / "phase3_2c_conditional_summary.txt"),

        # Controls
        ("control_output", results_dir / "phase3_2_control_runs.csv"),
        ("control_output", results_dir / "phase3_2_control_pair_scores.csv"),
        ("control_output", resolve_project_path(root, cfg.controls_json)),
        ("control_output", results_dir / "phase3_2_controls_summary.txt"),

        # Metrics / gates
        ("metrics_output", resolve_project_path(root, cfg.gates_json)),
        ("metrics_output", resolve_project_path(root, cfg.multiple_comparison_guard_json)),
        ("metrics_output", results_dir / "phase3_2_metrics_table.csv"),
        ("metrics_output", results_dir / "phase3_2_metrics_summary.txt"),

        # Loader / features / regime / lag inventories
        ("interim_output", resolve_project_path(root, cfg.loader_metadata_file)),
        ("interim_output", resolve_project_path(root, cfg.combined_features_file)),
        ("interim_output", interim_dir / "phase3_2_modeling_features.csv"),
        ("interim_output", interim_dir / "phase3_2_feature_specs.json"),
        ("interim_output", interim_dir / "phase3_2_regime_inventory.csv"),
        ("interim_output", interim_dir / "phase3_2_regime_summary.json"),
        ("interim_output", interim_dir / "phase3_2_lag_inventory.csv"),
        ("interim_output", interim_dir / "phase3_2_lag_summary.json"),
    ]

    # Add all diagnostics outputs.
    if diagnostics_dir.exists():
        for path in sorted(diagnostics_dir.rglob("*")):
            if path.is_file():
                candidates.append(("diagnostics_output", path))

    return candidates


# =========================================================
# Status extraction
# =========================================================

def extract_status_summary(root: Path, cfg: Phase32Config) -> Dict[str, Any]:
    results_dir = resolve_project_path(root, cfg.results_dir)
    diagnostics_dir = resolve_project_path(root, cfg.diagnostics_dir)

    runner_report = safe_read_json(results_dir / "phase3_2_runner_report.json")
    gates_json = safe_read_json(resolve_project_path(root, cfg.gates_json))
    controls_json = safe_read_json(resolve_project_path(root, cfg.controls_json))
    diagnostics_json = safe_read_json(diagnostics_dir / "phase3_2_diagnostics_report.json")
    conditional_summary = safe_read_json(resolve_project_path(root, cfg.conditional_summary_json))

    return {
        "runner_final_status": runner_report.get("final_status", {}),
        "diagnostics_final_interpretation": diagnostics_json.get("final_interpretation", {}),
        "gates_final_status": gates_json.get("final_status", {}),
        "controls_summary": controls_json.get("summary", {}),
        "conditional_summary": conditional_summary.get("summary", {}),
        "expected_final_status": EXPECTED_FINAL_STATUS,
    }


# =========================================================
# Audit summary markdown
# =========================================================

def build_audit_summary_md(
    cfg: Phase32Config,
    status_summary: Dict[str, Any],
    manifest: Dict[str, Any],
) -> str:
    final_status = status_summary.get("gates_final_status", {})
    diagnostics_final = status_summary.get("diagnostics_final_interpretation", {})
    controls_summary = status_summary.get("controls_summary", {})
    conditional_summary = status_summary.get("conditional_summary", {})

    lines: List[str] = []

    lines.append("# CDR Phase III.2 — Audit Bundle")
    lines.append("")
    lines.append(f"- **Bundle version:** `{BUNDLE_VERSION}`")
    lines.append(f"- **Created at:** `{manifest.get('created_at')}`")
    lines.append(f"- **Project:** `{cfg.project_name}`")
    lines.append(f"- **Phase:** `{cfg.phase_name}`")
    lines.append(f"- **Phase title:** `{cfg.phase_title}`")
    lines.append(f"- **Code/config version:** `{cfg.version}`")
    lines.append("")

    lines.append("## Final status")
    lines.append("")
    lines.append(f"- **Metrics status:** `{final_status.get('status')}`")
    lines.append(f"- **Diagnostics headline:** `{diagnostics_final.get('headline')}`")
    lines.append(f"- **Diagnostics status:** `{diagnostics_final.get('status')}`")
    lines.append("")
    lines.append("**Interpretation:**")
    lines.append("")
    lines.append(f"> {final_status.get('interpretation')}")
    lines.append("")
    lines.append("**Diagnostic recommendation:**")
    lines.append("")
    lines.append(f"> {diagnostics_final.get('recommendation')}")
    lines.append("")

    lines.append("## Gate summary")
    lines.append("")
    lines.append(f"- **Passed gates:** `{final_status.get('passed_gates')}`")
    lines.append(f"- **Failed gates:** `{final_status.get('failed_gates')}`")
    lines.append(f"- **Warning gates:** `{final_status.get('warning_gates')}`")
    lines.append(f"- **Not evaluated gates:** `{final_status.get('not_evaluated_gates')}`")
    lines.append("")

    lines.append("## Controls")
    lines.append("")
    lines.append(f"- **Controls passed:** `{controls_summary.get('passed')}`")
    lines.append(f"- **Failed control types:** `{controls_summary.get('failed_control_types')}`")
    lines.append(f"- **Number of control types:** `{controls_summary.get('n_control_types')}`")
    lines.append("")

    lines.append("## Conditional results")
    lines.append("")
    lines.append(f"- **Valid pair rows:** `{conditional_summary.get('n_valid_rows')}`")
    lines.append(f"- **Positive lift rows:** `{conditional_summary.get('n_positive_lift_rows')}`")
    lines.append(f"- **Strong candidate rows:** `{conditional_summary.get('n_strong_candidate_rows')}`")
    lines.append(f"- **Best by lift:** `{conditional_summary.get('best_by_conditional_lift')}`")
    lines.append("")

    lines.append("## Bundle contents")
    lines.append("")
    lines.append(f"- **Included files:** `{manifest.get('n_included_files')}`")
    lines.append(f"- **Missing expected files:** `{manifest.get('n_missing_files')}`")
    lines.append(f"- **Raw data files inventoried:** `{manifest.get('raw_data_inventory', {}).get('n_files')}`")
    lines.append("")
    lines.append("Raw EDF/RNG files are not copied into the bundle by default. Their metadata and hashes are recorded in the manifest when available.")
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

def build_phase3_2_audit_bundle(
    cfg: Optional[Phase32Config] = None,
    output_dir: Optional[Path] = None,
    keep_staging: bool = False,
    hash_raw_files: bool = True,
    raw_hash_max_mb: float = 250.0,
) -> Dict[str, Any]:
    root = project_root()

    if cfg is None:
        cfg = load_phase3_2_config()

    if output_dir is None:
        output_dir = resolve_project_path(root, cfg.results_dir) / "audit_bundle"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = _now_stamp()

    bundle_name = f"cdr_phase3_2_audit_bundle_{stamp}"
    staging_dir = output_dir / f"{bundle_name}_staging"
    zip_path = output_dir / f"{bundle_name}.zip"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    staging_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: List[Dict[str, Any]] = []
    missing_files: List[Dict[str, Any]] = []

    print("\n" + "=" * 78)
    print("Building CDR Phase III.2 audit bundle")
    print("=" * 78)
    print(f"Project root: {root}")
    print(f"Staging dir: {staging_dir}")
    print(f"Zip path: {zip_path}")

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
        compute_hash=True,
    )

    # -----------------------------------------------------
    # Include outputs
    # -----------------------------------------------------
    output_candidates = result_files(root, cfg)

    add_existing_files(
        root=root,
        staging_root=staging_dir,
        files=output_candidates,
        manifest_files=manifest_files,
        missing_files=missing_files,
        compute_hash=True,
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
        "config": describe_config(cfg),
        "status_summary": status_summary,
        "included_files": manifest_files,
        "missing_files": missing_files,
        "n_included_files": int(len(manifest_files)),
        "n_missing_files": int(len(missing_files)),
        "raw_data_inventory": raw_inventory,
        "notes": [
            "Raw EDF/RNG files are inventoried but not copied into the audit zip.",
            "Controls are expected to be leakage-safe Phase III.2 controls.",
            "Final status is exploratory only unless F7/F8/F12 constraints are resolved in future runs.",
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
    write_json(manifest_path, manifest)

    # -----------------------------------------------------
    # Create zip
    # -----------------------------------------------------
    make_zip_from_directory(staging_dir, zip_path)

    zip_sha256 = sha256_file(zip_path)

    final_report = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": _now_human(),
        "zip_path": str(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "zip_sha256": zip_sha256,
        "staging_dir": str(staging_dir),
        "staging_kept": bool(keep_staging),
        "n_included_files": int(len(manifest_files)),
        "n_missing_files": int(len(missing_files)),
        "status_summary": status_summary,
    }

    final_report_path = output_dir / f"{bundle_name}_bundle_report.json"
    write_json(final_report_path, final_report)

    if not keep_staging:
        shutil.rmtree(staging_dir)

    print("\n" + "=" * 78)
    print("Phase III.2 audit bundle completed")
    print("=" * 78)
    print(f"Zip: {zip_path}")
    print(f"SHA256: {zip_sha256}")
    print(f"Bundle report: {final_report_path}")
    print(f"Included files: {len(manifest_files)}")
    print(f"Missing expected files: {len(missing_files)}")
    print("=" * 78 + "\n")

    return final_report


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an audit bundle for CDR Phase III.2."
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
        default=250.0,
        help="Maximum raw file size in MB for computing SHA256 in the raw data inventory.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2_config()

    output_dir = Path(args.output_dir) if args.output_dir else None

    build_phase3_2_audit_bundle(
        cfg=cfg,
        output_dir=output_dir,
        keep_staging=bool(args.keep_staging),
        hash_raw_files=not bool(args.no_raw_hash),
        raw_hash_max_mb=float(args.raw_hash_max_mb),
    )


if __name__ == "__main__":
    main()