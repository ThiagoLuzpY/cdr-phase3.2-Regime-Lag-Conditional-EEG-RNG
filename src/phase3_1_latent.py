from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Optional sklearn support
# =========================================================

try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except Exception:
    KMeans = None
    SKLEARN_AVAILABLE = False


# =========================================================
# Data containers
# =========================================================

@dataclass(frozen=True)
class StandardizerArtifacts:
    """
    Train-fitted standardization parameters.

    Important:
        These parameters must be fitted only on the training partition
        and then applied to validation/test partitions.
    """

    features: Tuple[str, ...]
    means: Dict[str, float]
    stds: Dict[str, float]


@dataclass(frozen=True)
class LatentModelArtifacts:
    """
    Train-fitted latent-state model artifacts.

    This stores enough information to assign Z_t to unseen rows
    without refitting on holdout/test data.
    """

    method: str
    features: Tuple[str, ...]
    k: int
    centers: np.ndarray
    standardizer: StandardizerArtifacts
    random_state: int
    fitted_rows: int


@dataclass(frozen=True)
class LatentAssignmentResult:
    """
    Output of latent-state assignment.
    """

    df: pd.DataFrame
    artifacts: LatentModelArtifacts
    label_column: str
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class LatentDiagnostics:
    """
    Compact diagnostic report for a latent assignment.
    """

    label_column: str
    k: int
    n_rows: int
    counts: Dict[str, int]
    proportions: Dict[str, float]
    entropy: float
    normalized_entropy: float
    min_cluster_fraction: float
    max_cluster_fraction: float
    degenerate: bool


# =========================================================
# Numeric helpers
# =========================================================

_EPS = 1e-12


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return 0.0

    return float(np.mean(arr))


def _safe_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        return 1.0

    std = float(np.std(arr, ddof=0))
    if std <= _EPS:
        return 1.0

    return std


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str], fill_value: float = 0.0) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = fill_value

    return out


def _replace_nonfinite(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(fill_value)
    return out


def _normalize_index(index: Optional[Sequence[int]], n_rows: int) -> np.ndarray:
    if index is None:
        return np.arange(n_rows, dtype=int)

    idx = np.asarray(index, dtype=int).reshape(-1)

    if len(idx) == 0:
        raise ValueError("index cannot be empty")

    if np.any(idx < 0) or np.any(idx >= n_rows):
        raise ValueError("index contains out-of-range positions")

    return idx


def _entropy_from_counts(counts: np.ndarray) -> Tuple[float, float]:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()

    if total <= 0:
        return 0.0, 0.0

    probs = counts / total
    probs = probs[probs > 0]

    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(max(len(counts), 2)))

    if max_entropy <= 0:
        return entropy, 0.0

    return entropy, float(entropy / max_entropy)


# =========================================================
# Standardization
# =========================================================

def fit_standardizer(
    df: pd.DataFrame,
    features: Sequence[str],
    train_index: Optional[Sequence[int]] = None,
) -> StandardizerArtifacts:
    """
    Fits feature-wise mean/std on train rows only.
    """
    features = tuple(str(f) for f in features)

    if not features:
        raise ValueError("features cannot be empty")

    work = _ensure_columns(df, features, fill_value=0.0)
    work = _replace_nonfinite(work)

    idx_train = _normalize_index(train_index, len(work))
    train = work.iloc[idx_train]

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    for col in features:
        values = train[col].astype(float).to_numpy()
        means[col] = _safe_mean(values)
        stds[col] = _safe_std(values)

    return StandardizerArtifacts(
        features=features,
        means=means,
        stds=stds,
    )


def apply_standardizer(
    df: pd.DataFrame,
    artifacts: StandardizerArtifacts,
) -> np.ndarray:
    """
    Applies train-fitted standardization to a dataframe.
    """
    work = _ensure_columns(df, artifacts.features, fill_value=0.0)
    work = _replace_nonfinite(work)

    columns: List[np.ndarray] = []

    for col in artifacts.features:
        values = work[col].astype(float).to_numpy()
        mean = artifacts.means.get(col, 0.0)
        std = artifacts.stds.get(col, 1.0)

        if std <= _EPS:
            std = 1.0

        columns.append((values - mean) / std)

    if not columns:
        return np.zeros((len(df), 1), dtype=float)

    return np.vstack(columns).T.astype(float)


# =========================================================
# KMeans fitting / fallback
# =========================================================

def _simple_kmeans(
    x: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Deterministic fallback k-means implementation.

    Used only if scikit-learn is unavailable.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 2:
        raise ValueError("x must be 2D")

    if len(x) < k:
        raise ValueError(f"not enough samples for k={k}: n={len(x)}")

    rng = np.random.default_rng(seed)
    initial_idx = rng.choice(len(x), size=k, replace=False)
    centers = x[initial_idx].copy()

    for _ in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()

        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.any(mask):
                new_centers[cluster_id] = x[mask].mean(axis=0)

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return centers.astype(float)


def _fit_kmeans_centers(
    x_train: np.ndarray,
    k: int,
    random_state: int,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)

    if len(x_train) < k:
        raise RuntimeError(
            f"Cannot fit latent k-means with k={k}; only {len(x_train)} train rows"
        )

    if SKLEARN_AVAILABLE:
        model = KMeans(
            n_clusters=int(k),
            random_state=int(random_state),
            n_init=10,
        )
        model.fit(x_train)
        return np.asarray(model.cluster_centers_, dtype=float)

    warnings.warn(
        "scikit-learn not available. Using simple deterministic k-means fallback.",
        RuntimeWarning,
    )
    return _simple_kmeans(
        x=x_train,
        k=int(k),
        seed=int(random_state),
        max_iter=100,
    )


def _assign_nearest_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    centers = np.asarray(centers, dtype=float)

    if x.ndim != 2:
        raise ValueError("x must be 2D")

    if centers.ndim != 2:
        raise ValueError("centers must be 2D")

    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)

    return labels.astype(int)


# =========================================================
# Latent Z_t fitting / applying
# =========================================================

def fit_latent_model(
    df: pd.DataFrame,
    cfg: Any,
    train_index: Optional[Sequence[int]] = None,
    features: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> LatentModelArtifacts:
    """
    Fits the latent Z_t model on train rows only.

    Parameters
    ----------
    df:
        Feature dataframe.
    cfg:
        Phase3_1Config.
    train_index:
        Row positions used for fitting.
    features:
        Optional override for latent features.
    k:
        Optional override for number of clusters.
    """
    latent_method = str(getattr(cfg, "latent_method", "kmeans"))

    if latent_method != "kmeans":
        raise ValueError("Only latent_method='kmeans' is currently supported")

    latent_features = tuple(features or getattr(cfg, "latent_features"))
    latent_k = int(k or getattr(cfg, "latent_k", 3))
    random_state = int(getattr(cfg, "latent_random_state", 42))

    idx_train = _normalize_index(train_index, len(df))

    standardizer = fit_standardizer(
        df=df,
        features=latent_features,
        train_index=idx_train,
    )

    x_all = apply_standardizer(df, standardizer)
    x_train = x_all[idx_train]

    centers = _fit_kmeans_centers(
        x_train=x_train,
        k=latent_k,
        random_state=random_state,
    )

    return LatentModelArtifacts(
        method=latent_method,
        features=latent_features,
        k=latent_k,
        centers=centers,
        standardizer=standardizer,
        random_state=random_state,
        fitted_rows=int(len(idx_train)),
    )


def apply_latent_model(
    df: pd.DataFrame,
    artifacts: LatentModelArtifacts,
    label_column: str = "latent_state",
) -> pd.DataFrame:
    """
    Applies a train-fitted latent model to any dataframe.
    """
    out = df.copy().reset_index(drop=True)
    out = _replace_nonfinite(out)

    x = apply_standardizer(out, artifacts.standardizer)
    labels = _assign_nearest_centers(x, artifacts.centers)

    out[label_column] = labels.astype(int)

    return out


def fit_transform_latent_model(
    df: pd.DataFrame,
    cfg: Any,
    train_index: Optional[Sequence[int]] = None,
    label_column: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> LatentAssignmentResult:
    """
    Fits latent model on train rows and applies to the full dataframe.
    """
    label_column = label_column or str(getattr(cfg, "latent_component_name", "latent_state"))

    artifacts = fit_latent_model(
        df=df,
        cfg=cfg,
        train_index=train_index,
        features=features,
        k=k,
    )

    out = apply_latent_model(
        df=df,
        artifacts=artifacts,
        label_column=label_column,
    )

    diagnostics = latent_diagnostics(
        labels=out[label_column].to_numpy(),
        k=artifacts.k,
        label_column=label_column,
    )

    return LatentAssignmentResult(
        df=out,
        artifacts=artifacts,
        label_column=label_column,
        diagnostics=asdict(diagnostics),
    )


# =========================================================
# Diagnostics
# =========================================================

def latent_diagnostics(
    labels: Sequence[int],
    k: int,
    label_column: str = "latent_state",
    degenerate_min_fraction: float = 0.02,
) -> LatentDiagnostics:
    """
    Computes distribution diagnostics for latent labels.
    """
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    k = int(k)

    if len(labels_arr) == 0:
        counts = np.zeros(k, dtype=int)
    else:
        counts = np.bincount(labels_arr, minlength=k)[:k]

    total = int(counts.sum())

    if total <= 0:
        proportions = np.zeros(k, dtype=float)
    else:
        proportions = counts.astype(float) / float(total)

    entropy, normalized_entropy = _entropy_from_counts(counts)

    if len(proportions) == 0:
        min_fraction = 0.0
        max_fraction = 0.0
    else:
        min_fraction = float(np.min(proportions))
        max_fraction = float(np.max(proportions))

    degenerate = bool(min_fraction < float(degenerate_min_fraction))

    return LatentDiagnostics(
        label_column=str(label_column),
        k=k,
        n_rows=total,
        counts={str(i): int(counts[i]) for i in range(k)},
        proportions={str(i): float(proportions[i]) for i in range(k)},
        entropy=float(entropy),
        normalized_entropy=float(normalized_entropy),
        min_cluster_fraction=min_fraction,
        max_cluster_fraction=max_fraction,
        degenerate=degenerate,
    )


def subject_latent_diagnostics(
    df: pd.DataFrame,
    label_column: str = "latent_state",
    subject_column: str = "subject_id",
    k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Computes latent distribution diagnostics per subject.
    """
    if label_column not in df.columns:
        raise KeyError(f"Missing label column: {label_column}")

    if subject_column not in df.columns:
        raise KeyError(f"Missing subject column: {subject_column}")

    if k is None:
        k = int(df[label_column].max()) + 1 if len(df) else 0

    rows: List[Dict[str, Any]] = []

    for subject_id, group in df.groupby(subject_column):
        diag = latent_diagnostics(
            labels=group[label_column].to_numpy(),
            k=int(k),
            label_column=label_column,
        )

        row = asdict(diag)
        row["subject_id"] = str(subject_id)
        rows.append(row)

    return pd.DataFrame(rows)


def sleep_stage_latent_diagnostics(
    df: pd.DataFrame,
    label_column: str = "latent_state",
    stage_column: str = "sleep_stage",
    k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Computes latent distribution diagnostics per sleep stage.
    """
    if label_column not in df.columns:
        raise KeyError(f"Missing label column: {label_column}")

    if stage_column not in df.columns:
        raise KeyError(f"Missing stage column: {stage_column}")

    if k is None:
        k = int(df[label_column].max()) + 1 if len(df) else 0

    rows: List[Dict[str, Any]] = []

    for stage, group in df.groupby(stage_column):
        diag = latent_diagnostics(
            labels=group[label_column].to_numpy(),
            k=int(k),
            label_column=label_column,
        )

        row = asdict(diag)
        row["sleep_stage"] = str(stage)
        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# Latent controls / ablations
# =========================================================

def shuffle_latent_state(
    df: pd.DataFrame,
    label_column: str = "latent_state",
    seed: int = 42,
    within_subject: bool = True,
    subject_column: str = "subject_id",
) -> pd.DataFrame:
    """
    Shuffles latent labels while preserving other variables.

    If within_subject=True, label distribution is preserved per subject.
    """
    if label_column not in df.columns:
        raise KeyError(f"Missing label column: {label_column}")

    out = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if within_subject and subject_column in out.columns:
        for _, idx in out.groupby(subject_column).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            values = out.loc[idx_arr, label_column].to_numpy()
            rng.shuffle(values)
            out.loc[idx_arr, label_column] = values
    else:
        values = out[label_column].to_numpy()
        rng.shuffle(values)
        out[label_column] = values

    return out


def remove_latent_state(
    df: pd.DataFrame,
    label_column: str = "latent_state",
    replacement: int = 0,
) -> pd.DataFrame:
    """
    Ablates Z_t by replacing latent state with a constant.
    """
    out = df.copy().reset_index(drop=True)
    out[label_column] = int(replacement)
    return out


def remap_latent_state_by_frequency(
    df: pd.DataFrame,
    label_column: str = "latent_state",
) -> pd.DataFrame:
    """
    Reorders latent labels by descending frequency.

    This is useful for stable reporting across runs while preserving structure.
    """
    if label_column not in df.columns:
        raise KeyError(f"Missing label column: {label_column}")

    out = df.copy().reset_index(drop=True)

    counts = out[label_column].value_counts().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(counts.index.tolist())}

    out[label_column] = out[label_column].map(mapping).astype(int)

    return out


# =========================================================
# Leave-one-subject latent fitting helpers
# =========================================================

def make_train_index_excluding_subject(
    df: pd.DataFrame,
    heldout_subject: str,
    subject_column: str = "subject_id",
) -> np.ndarray:
    """
    Returns row positions excluding one subject.
    """
    if subject_column not in df.columns:
        raise KeyError(f"Missing subject column: {subject_column}")

    mask = df[subject_column].astype(str) != str(heldout_subject)
    return np.flatnonzero(mask.to_numpy())


def make_test_index_for_subject(
    df: pd.DataFrame,
    heldout_subject: str,
    subject_column: str = "subject_id",
) -> np.ndarray:
    """
    Returns row positions for one held-out subject.
    """
    if subject_column not in df.columns:
        raise KeyError(f"Missing subject column: {subject_column}")

    mask = df[subject_column].astype(str) == str(heldout_subject)
    return np.flatnonzero(mask.to_numpy())


def fit_latent_loso_artifacts(
    df: pd.DataFrame,
    cfg: Any,
    heldout_subject: str,
    subject_column: str = "subject_id",
) -> LatentModelArtifacts:
    """
    Fits latent model using all subjects except heldout_subject.
    """
    train_index = make_train_index_excluding_subject(
        df=df,
        heldout_subject=heldout_subject,
        subject_column=subject_column,
    )

    if len(train_index) == 0:
        raise RuntimeError(
            f"No training rows remain after excluding subject={heldout_subject}"
        )

    return fit_latent_model(
        df=df,
        cfg=cfg,
        train_index=train_index,
    )


def assign_latent_loso(
    df: pd.DataFrame,
    cfg: Any,
    heldout_subject: str,
    label_column: Optional[str] = None,
    subject_column: str = "subject_id",
) -> LatentAssignmentResult:
    """
    Fits latent model excluding heldout_subject and applies to full dataframe.
    """
    label_column = label_column or str(getattr(cfg, "latent_component_name", "latent_state"))

    artifacts = fit_latent_loso_artifacts(
        df=df,
        cfg=cfg,
        heldout_subject=heldout_subject,
        subject_column=subject_column,
    )

    out = apply_latent_model(
        df=df,
        artifacts=artifacts,
        label_column=label_column,
    )

    diagnostics = latent_diagnostics(
        labels=out[label_column].to_numpy(),
        k=artifacts.k,
        label_column=label_column,
    )

    return LatentAssignmentResult(
        df=out,
        artifacts=artifacts,
        label_column=label_column,
        diagnostics=asdict(diagnostics),
    )


# =========================================================
# Serialization
# =========================================================

def latent_artifacts_to_dict(artifacts: LatentModelArtifacts) -> Dict[str, Any]:
    return {
        "method": artifacts.method,
        "features": list(artifacts.features),
        "k": int(artifacts.k),
        "centers": np.asarray(artifacts.centers, dtype=float).tolist(),
        "standardizer": {
            "features": list(artifacts.standardizer.features),
            "means": dict(artifacts.standardizer.means),
            "stds": dict(artifacts.standardizer.stds),
        },
        "random_state": int(artifacts.random_state),
        "fitted_rows": int(artifacts.fitted_rows),
    }


def save_latent_artifacts(
    path: Path,
    artifacts: LatentModelArtifacts,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = latent_artifacts_to_dict(artifacts)

    if diagnostics is not None:
        payload["diagnostics"] = dict(diagnostics)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def load_latent_artifacts(path: Path) -> LatentModelArtifacts:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    standardizer_payload = payload["standardizer"]

    standardizer = StandardizerArtifacts(
        features=tuple(standardizer_payload["features"]),
        means={str(k): float(v) for k, v in standardizer_payload["means"].items()},
        stds={str(k): float(v) for k, v in standardizer_payload["stds"].items()},
    )

    return LatentModelArtifacts(
        method=str(payload["method"]),
        features=tuple(payload["features"]),
        k=int(payload["k"]),
        centers=np.asarray(payload["centers"], dtype=float),
        standardizer=standardizer,
        random_state=int(payload["random_state"]),
        fitted_rows=int(payload["fitted_rows"]),
    )


# =========================================================
# Convenience high-level function
# =========================================================

def add_latent_state_layer(
    df: pd.DataFrame,
    cfg: Any,
    train_index: Optional[Sequence[int]] = None,
    label_column: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> LatentAssignmentResult:
    """
    Main convenience entrypoint for Phase III.1.

    Fits Z_t on train only, applies it to the full dataframe,
    optionally saves artifacts, and returns the transformed dataframe.
    """
    result = fit_transform_latent_model(
        df=df,
        cfg=cfg,
        train_index=train_index,
        label_column=label_column,
    )

    if save_path is not None:
        save_latent_artifacts(
            path=save_path,
            artifacts=result.artifacts,
            diagnostics=result.diagnostics,
        )

    return result