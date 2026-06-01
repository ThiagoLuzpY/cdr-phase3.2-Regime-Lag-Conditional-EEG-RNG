"""
ecology_loader.py — V3 (max data retention)
"""

import os
import pandas as pd
import numpy as np


def load_lynx_hare_dataset(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Lynx-Hare dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path, sep=';')

    # ---------------------------
    # Normalize columns
    # ---------------------------
    df.columns = [
        str(c).strip()
        .lower()
        .replace(' ', '')
        .replace('(monthly)', '')
        .replace('(', '')
        .replace(')', '')
        .replace('_', '')
        .replace(';', '')
        for c in df.columns
    ]

    print("Colunas detectadas:", list(df.columns))

    hare_col = next((c for c in df.columns if 'hare' in c), None)
    lynx_col = next((c for c in df.columns if 'lynx' in c), None)
    year_col = next((c for c in df.columns if 'year' in c), None)

    if hare_col is None or lynx_col is None:
        raise ValueError("Missing hare/lynx columns")

    # ---------------------------
    # Numeric conversion
    # ---------------------------
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df[hare_col] = pd.to_numeric(df[hare_col], errors='coerce')
    df[lynx_col] = pd.to_numeric(df[lynx_col], errors='coerce')

    df = df.sort_values(by=year_col).reset_index(drop=True)

    # ---------------------------
    # FILL STRATEGY (CRUCIAL)
    # ---------------------------
    # NÃO inventa tendência nova, apenas mantém continuidade

    df[hare_col] = df[hare_col].ffill().bfill()
    df[lynx_col] = df[lynx_col].ffill().bfill()

    # ---------------------------
    # Log + returns
    # ---------------------------
    eps = 1e-6

    df["hare_log"] = np.log(df[hare_col] + eps)
    df["lynx_log"] = np.log(df[lynx_col] + eps)

    df["hare_log_return"] = df["hare_log"].diff()
    df["lynx_log_return"] = df["lynx_log"].diff()

    # Remove apenas primeira linha (diff)
    df_features = df[[
        "hare_log_return",
        "lynx_log_return"
    ]].iloc[1:].copy()

    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # última limpeza leve
    df_features = df_features.dropna().reset_index(drop=True)

    print(f"Total linhas finais: {len(df_features)}")

    return {
        "years": df[year_col].iloc[1:].values,
        "features_df": df_features
    }


def build_predator_prey_matrix(data):

    df = data["features_df"]

    X = df.to_numpy(dtype=float)

    print(f"Matriz final shape: {X.shape}")

    return X