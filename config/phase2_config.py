from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def default_patterns() -> Dict[str, List[str]]:
    """Padrões genéricos para resolução automática de colunas."""
    return {
        "load": [
            "_load_actual_",
            "_load_",
            "_load_forecast_",
        ],
        "wind": [
            "_wind_generation_actual",
            "_wind_onshore_generation_actual",
            "_wind_offshore_generation_actual",
            "_wind_",
        ],
        "solar": [
            "_solar_generation_actual",
            "_solar_",
            "_pv_",
        ],
        "price": [
            "_price_day_ahead_",
            "_price_",
        ],
    }


@dataclass(frozen=True)
class DataConfig:
    csv_path: str = "data/raw/opsp/time_series_60min_singleindex.csv"
    country: str = "DE_LU"
    start: str = "2019-01-01"
    end: str = "2019-12-31"
    explicit_columns: Optional[Dict[str, str]] = None
    patterns: Dict[str, List[str]] = field(default_factory=default_patterns)


@dataclass(frozen=True)
class DiscretizeConfig:
    n_bins: int = 3
    quantiles: Tuple[float, float] = (0.33, 0.66)
    missing_policy: str = "drop"


@dataclass(frozen=True)
class KernelConfig:
    dirichlet_alpha: float = 1e-2
    min_prob: float = 1e-12
    eps_grid: Tuple[float, ...] = tuple([i * 0.01 for i in range(0, 81)])


@dataclass(frozen=True)
class GatesPhase2:
    inj_eps_true: float = 0.30
    inj_tol_abs: float = 0.05

    # limiar para colapso dos controles
    controls_tol: float = 0.05

    # fração mínima de controles que devem colapsar
    controls_required_fraction: float = 2 / 3

    holdout_max_delta: float = 0.10

    sensitivity_bins4: int = 4
    sensitivity_max_delta: float = 0.12


@dataclass(frozen=True)
class Phase2Config:

    data: DataConfig = DataConfig()
    disc: DiscretizeConfig = DiscretizeConfig()
    kernel: KernelConfig = KernelConfig()
    gates: GatesPhase2 = GatesPhase2()

    # controles usados para F2 (colapso legítimo)
    collapse_controls: Tuple[str, ...] = (
        "weekly_blocks",
        "seasonal_strata",
        "month_hour_weektype",
    )

    # controles adversariais / stress tests
    stress_controls: Tuple[str, ...] = (
        "rows_shuffle",
        "columns_shuffle",
    )

    # seeds reprodutíveis
    control_seeds: Tuple[int, ...] = (999, 1001, 1002, 1003, 1004)

    results_dir: str = "results/phase2_opsp"


# instancia "pré-registrada"
CFG = Phase2Config()