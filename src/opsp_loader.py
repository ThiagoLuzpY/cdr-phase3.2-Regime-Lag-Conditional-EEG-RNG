from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class OPSDSelection:
    country: str
    columns: Dict[str, str]  # role -> real column name


def _read_header(csv_path: Path) -> List[str]:
    df0 = pd.read_csv(csv_path, nrows=0)
    return list(df0.columns)


def _pick_first_matching(cols: List[str], country: str, patterns: List[str]) -> Optional[str]:
    prefix = f"{country}_"
    candidates = [c for c in cols if c.startswith(prefix)]
    for pat in patterns:
        for c in candidates:
            if pat in c:
                return c
    return None


def resolve_columns(
    csv_path: Path,
    country: str,
    patterns: Dict[str, List[str]],
    explicit: Optional[Dict[str, str]] = None,
) -> OPSDSelection:
    cols = _read_header(csv_path)

    # detect index col name (singleindex geralmente é "utc_timestamp")
    if "utc_timestamp" not in cols:
        # aceita também "timestamp" ou variações
        pass

    if explicit:
        # valida
        missing = [k for k, v in explicit.items() if v not in cols]
        if missing:
            raise ValueError(f"Explicit columns not found in CSV header: {missing}")
        return OPSDSelection(country=country, columns=explicit)

    resolved: Dict[str, str] = {}
    for role, pats in patterns.items():
        hit = _pick_first_matching(cols, country, pats)
        if hit is not None:
            resolved[role] = hit

    # regra mínima: load/wind/solar devem existir; price é opcional
    required = ["load", "wind", "solar"]
    missing_req = [r for r in required if r not in resolved]
    if missing_req:
        # mensagem bem explícita para você decidir
        prefix = f"{country}_"
        country_cols = [c for c in cols if c.startswith(prefix)]
        raise ValueError(
            "Could not auto-resolve required roles: "
            f"{missing_req}\n"
            f"Country '{country}' has {len(country_cols)} columns. "
            "Inspect and set explicit_columns in config if needed."
        )

    return OPSDSelection(country=country, columns=resolved)


def load_timeseries(
    csv_path: Path,
    selection: OPSDSelection,
    start: str,
    end: str,
) -> pd.DataFrame:
    # Sempre trazemos utc_timestamp + colunas selecionadas
    usecols = ["utc_timestamp"] + list(selection.columns.values())

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        parse_dates=["utc_timestamp"],
    ).set_index("utc_timestamp")

    df = df.sort_index()
    df = df.loc[start:end].copy()

    # renomeia para roles (load/wind/solar/price) para pipeline ficar estável
    inv = {v: k for k, v in selection.columns.items()}
    df = df.rename(columns=inv)

    return df


def quick_report(df: pd.DataFrame) -> Dict[str, float]:
    rep = {}
    for c in df.columns:
        rep[f"missing_frac_{c}"] = float(df[c].isna().mean())
    rep["n_rows"] = float(len(df))
    return rep