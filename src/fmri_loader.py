from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker


@dataclass(frozen=True)
class FMRIPaths:
    dataset_root: Path
    subject: str
    task: str
    bold_nii: Path
    bold_json: Optional[Path]


@dataclass(frozen=True)
class AtlasInfo:
    name: str
    maps_path: object
    labels: List[str]
    background_label: Optional[str] = None


@dataclass(frozen=True)
class FMRIResult:
    paths: FMRIPaths
    atlas: AtlasInfo
    tr: float
    n_scans: int
    n_rois: int
    dataframe: pd.DataFrame


def _normalize_subject(subject: str) -> str:
    subject = subject.strip()
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"
    return subject


def _normalize_task(task: str) -> str:
    return task.strip().lower()


def resolve_subject_task_paths(
    dataset_root: str | Path,
    subject: str = "sub-01",
    task: str = "effort",
) -> FMRIPaths:
    """
    Resolve the BOLD NIfTI and its sidecar JSON for one subject/task.

    Expected structure:
    data/raw/fmri/ds002938/sub-01/func/sub-01_task-effort_bold.nii.gz
    """
    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    subject = _normalize_subject(subject)
    task = _normalize_task(task)

    func_dir = root / subject / "func"
    if not func_dir.exists():
        raise FileNotFoundError(f"func directory not found: {func_dir}")

    nii_candidates = sorted(func_dir.glob(f"{subject}_task-{task}_bold.nii.gz"))
    if not nii_candidates:
        raise FileNotFoundError(
            f"No BOLD file found for subject='{subject}', task='{task}' in {func_dir}"
        )

    bold_nii = nii_candidates[0]
    bold_json = bold_nii.with_suffix("").with_suffix(".json")
    if not bold_json.exists():
        bold_json = None

    return FMRIPaths(
        dataset_root=root,
        subject=subject,
        task=task,
        bold_nii=bold_nii,
        bold_json=bold_json,
    )


def infer_tr_from_sidecar(bold_json: Optional[Path], default_tr: float = 2.0) -> float:
    """
    Read RepetitionTime from the BIDS sidecar JSON if available.
    Falls back to default_tr.
    """
    if bold_json is None:
        return float(default_tr)

    try:
        with bold_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        tr = meta.get("RepetitionTime", default_tr)
        return float(tr)
    except Exception:
        return float(default_tr)


def fetch_atlas_info(
    atlas_name: str = "harvard_oxford",
    data_dir: Optional[str | Path] = None,
) -> AtlasInfo:
    """
    Fetch/load a nilearn atlas and return the map path + ROI labels.

    Supported:
    - 'harvard_oxford'
    - 'aal'
    """
    atlas_name = atlas_name.strip().lower()

    if atlas_name == "harvard_oxford":
        atlas = datasets.fetch_atlas_harvard_oxford(
            "cort-maxprob-thr25-2mm",
            data_dir=None if data_dir is None else str(data_dir),
        )
        labels = list(atlas.labels)
        background = labels[0] if labels else None
        return AtlasInfo(
            name="harvard_oxford_cort_maxprob_thr25_2mm",
            maps_path=atlas.maps,
            labels=labels,
            background_label=background,
        )

    if atlas_name == "aal":
        atlas = datasets.fetch_atlas_aal(
            data_dir=None if data_dir is None else str(data_dir),
        )
        labels = list(atlas.labels)
        return AtlasInfo(
            name="aal",
            maps_path=atlas.maps,
            labels=labels,
            background_label=None,
        )

    raise ValueError(
        f"Unsupported atlas_name='{atlas_name}'. Use 'harvard_oxford' or 'aal'."
    )


def _clean_labels_for_dataframe(labels: List[str], n_columns: int) -> List[str]:
    """
    Make labels align with the extracted matrix columns.
    Removes background label if present and falls back to generic ROI names if needed.
    """
    cleaned = list(labels)

    if cleaned and cleaned[0].strip().lower() in {"background", "bg"}:
        cleaned = cleaned[1:]

    if len(cleaned) != n_columns:
        cleaned = [f"ROI_{i:03d}" for i in range(1, n_columns + 1)]

    return cleaned


def extract_roi_timeseries(
    bold_nii: str | Path,
    atlas: AtlasInfo,
    tr: float,
    standardize: bool = True,
    detrend: bool = True,
    smoothing_fwhm: Optional[float] = None,
    low_pass: Optional[float] = None,
    high_pass: Optional[float] = None,
    memory_level: int = 1,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Extract ROI time series from one 4D BOLD NIfTI using a label atlas.
    Returns a DataFrame indexed by scan number with time_seconds column.
    """
    bold_nii = Path(bold_nii).resolve()
    if not bold_nii.exists():
        raise FileNotFoundError(f"BOLD NIfTI not found: {bold_nii}")

    masker = NiftiLabelsMasker(
        labels_img=atlas.maps_path,
        standardize=standardize,
        detrend=detrend,
        smoothing_fwhm=smoothing_fwhm,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=float(tr),
        memory="nilearn_cache",
        memory_level=memory_level,
        verbose=verbose,
    )

    ts = masker.fit_transform(str(bold_nii))
    if ts.ndim != 2:
        raise ValueError(f"Unexpected time-series shape: {ts.shape}")

    n_scans, n_rois = ts.shape
    colnames = _clean_labels_for_dataframe(atlas.labels, n_rois)

    df = pd.DataFrame(ts, columns=colnames)
    df.index.name = "scan"
    df.insert(0, "time_seconds", np.arange(n_scans, dtype=float) * float(tr))

    return df


def load_subject_timeseries(
    dataset_root: str | Path,
    subject: str = "sub-01",
    task: str = "effort",
    atlas_name: str = "harvard_oxford",
    atlas_data_dir: Optional[str | Path] = None,
    default_tr: float = 2.0,
    standardize: bool = True,
    detrend: bool = True,
    smoothing_fwhm: Optional[float] = None,
    low_pass: Optional[float] = None,
    high_pass: Optional[float] = None,
    verbose: int = 0,
) -> FMRIResult:
    """
    Full convenience loader:
    - resolves dataset paths
    - reads TR
    - fetches atlas
    - extracts ROI time series
    """
    paths = resolve_subject_task_paths(
        dataset_root=dataset_root,
        subject=subject,
        task=task,
    )

    tr = infer_tr_from_sidecar(paths.bold_json, default_tr=default_tr)

    atlas = fetch_atlas_info(
        atlas_name=atlas_name,
        data_dir=atlas_data_dir,
    )

    df = extract_roi_timeseries(
        bold_nii=paths.bold_nii,
        atlas=atlas,
        tr=tr,
        standardize=standardize,
        detrend=detrend,
        smoothing_fwhm=smoothing_fwhm,
        low_pass=low_pass,
        high_pass=high_pass,
        verbose=verbose,
    )

    # ===== DOMAIN-SPECIFIC DIMENSIONALITY REDUCTION =====
    # Reduce ROI count to computationally feasible state space
    # (analogous to OPSD selecting 4 key grid variables)

    roi_cols = [c for c in df.columns if c != "time_seconds"]
    n_rois_total = len(roi_cols)
    target_n_rois = 5  # 3^6 = 729 states (computationally feasible)

    if n_rois_total > target_n_rois:
        # Select evenly spaced ROIs across brain
        step = n_rois_total // target_n_rois
        selected_idx = [i * step for i in range(target_n_rois)]
        selected_rois = [roi_cols[i] for i in selected_idx]

        # Keep time_seconds + selected ROIs
        df = df[["time_seconds"] + selected_rois]

        if verbose > 0:
            print(f"[fMRI-Domain] Reduced {n_rois_total} → {target_n_rois} ROIs")
            print(f"[fMRI-Domain] State space: 3^{target_n_rois} = {3 ** target_n_rois} states")
            print(f"[fMRI-Domain] Selected: {selected_rois}")
    # ===== END DOMAIN REDUCTION =====

    return FMRIResult(
        paths=paths,
        atlas=atlas,
        tr=tr,
        n_scans=len(df),
        n_rois=df.shape[1] - 1,  # excluding time_seconds
        dataframe=df,
    )


def quick_report_fmri(result: FMRIResult) -> dict:
    """
    Small metadata report for logging/artifacts.
    """
    return {
        "dataset_root": str(result.paths.dataset_root),
        "subject": result.paths.subject,
        "task": result.paths.task,
        "bold_nii": str(result.paths.bold_nii),
        "bold_json": None if result.paths.bold_json is None else str(result.paths.bold_json),
        "atlas_name": result.atlas.name,
        "tr": float(result.tr),
        "n_scans": int(result.n_scans),
        "n_rois": int(result.n_rois),
        "roi_columns": list(result.dataframe.columns[1:]),
    }