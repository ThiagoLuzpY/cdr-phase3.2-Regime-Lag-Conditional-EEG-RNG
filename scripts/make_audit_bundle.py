from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile


EXCLUDE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
    "__pycache__",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_files(run_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in run_dir.rglob("*"):
        if p.is_dir():
            # skip cache-like dirs
            if p.name in EXCLUDE_NAMES:
                continue
            continue
        if p.name in EXCLUDE_NAMES:
            continue
        # skip pyc
        if p.suffix.lower() in {".pyc"}:
            continue
        files.append(p)
    # Stable deterministic ordering
    files.sort(key=lambda x: str(x.as_posix()))
    return files


def try_git_commit(root_dir: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            commit = r.stdout.strip()
            return commit if commit else None
        return None
    except Exception:
        return None


def try_pip_freeze() -> str | None:
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
        return None
    except Exception:
        return None


def make_manifest(run_dir: Path, project_root: Path, include_pip_freeze: bool) -> Dict:
    files = iter_files(run_dir)

    entries = []
    total_bytes = 0

    for f in files:
        rel = f.relative_to(run_dir).as_posix()
        size = f.stat().st_size
        total_bytes += size
        digest = sha256_file(f)
        entries.append(
            {
                "path": rel,
                "size_bytes": int(size),
                "sha256": digest,
            }
        )

    now_utc = datetime.now(timezone.utc).isoformat()

    manifest: Dict = {
        "schema": "CDR_AUDIT_BUNDLE_MANIFEST_v1",
        "created_utc": now_utc,
        "run_dir": run_dir.as_posix(),
        "file_count": len(entries),
        "total_bytes": int(total_bytes),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "git_commit": try_git_commit(project_root),
        "files": entries,
    }

    if include_pip_freeze:
        manifest["pip_freeze"] = try_pip_freeze()

    return manifest


def write_manifest(run_dir: Path, manifest: Dict) -> Path:
    out_path = run_dir / "artifacts_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def add_file_to_zip_deterministic(zf: zipfile.ZipFile, file_path: Path, arcname: str) -> None:
    """
    Deterministic zip entry:
    - fixed timestamp (1980-01-01 00:00:00)
    - stable ordering handled by caller
    """
    data = file_path.read_bytes()
    info = zipfile.ZipInfo(arcname)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    # Preserve UNIX permissions in a consistent way (optional)
    info.external_attr = (0o644 & 0xFFFF) << 16
    zf.writestr(info, data)


def build_zip(run_dir: Path, zip_path: Path, deterministic: bool = True) -> None:
    files = iter_files(run_dir)
    # Always include manifest if it exists (we will create it before calling this)
    files.sort(key=lambda x: str(x.as_posix()))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files:
            arcname = f.relative_to(run_dir).as_posix()
            if deterministic:
                add_file_to_zip_deterministic(zf, f, arcname)
            else:
                zf.write(f, arcname)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create CDR audit bundle (manifest + zip) for a run directory.")
    ap.add_argument("--run_dir", required=True, help="Run directory (e.g., results/phase1_plus_full)")
    ap.add_argument("--project_root", default=".", help="Project root for git commit detection (default: .)")
    ap.add_argument("--include_pip_freeze", action="store_true", help="Include pip freeze in manifest (optional).")
    ap.add_argument("--deterministic_zip", action="store_true", help="Create deterministic zip (recommended).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    project_root = Path(args.project_root).resolve()

    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"ERROR: run_dir does not exist or is not a directory: {run_dir}")

    # 1) Create manifest (includes hashes)
    manifest = make_manifest(run_dir=run_dir, project_root=project_root, include_pip_freeze=args.include_pip_freeze)
    manifest_path = write_manifest(run_dir, manifest)

    # 2) Build zip bundle including manifest
    zip_path = run_dir / "run_bundle.zip"
    build_zip(run_dir, zip_path, deterministic=args.deterministic_zip)

    print("OK")
    print("Manifest:", manifest_path.as_posix())
    print("Bundle:", zip_path.as_posix())
    print("Files:", manifest["file_count"])
    print("Total bytes:", manifest["total_bytes"])


if __name__ == "__main__":
    main()