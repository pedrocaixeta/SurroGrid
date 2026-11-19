"""Step 3 of Inference: Feature preprocessing (log1p, seasonal encodings, VMD, scaling).

Replicates `Preprocessor._universal_X_trafo` and `Preprocessor._fit_or_transform_X`
from `0.Legacy_Folders/GridForecast/3_transformer/transformer_model.py` without
requiring PyTorch so it runs inside the lightweight `inference` conda environment.

Outputs:
- `df_vmd` (8760 x 49, unscaled): 13 scalars (with log1p on n_lines, tot_R_grid) +
  6 seasonal sin/cos terms + 30 VMD columns (6 signals x (4 modes + 1 residual)).
- `df_scaled` (8760 x 49, scaled): exact 49-column input tensor (as DataFrame)
  scaled with the frozen training `scaler_X` ready for the Transformer forward pass.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from vmdpy import VMD

import pipeline_config as cfg


def _vmd_single_signal(args):
    """Run VMD decomposition on one 1D hourly signal and return (col_name, DataFrame)."""
    col_name, signal, vmd_settings = args
    K = int(vmd_settings["K"])
    signal = np.asarray(signal, dtype=float)

    if np.all(signal == 0):
        modes_df = pd.DataFrame(
            np.zeros((len(signal), K + 1), dtype=np.float64),
            columns=[f"{col_name}_{i + 1}" for i in range(K)] + [f"{col_name}_res"],
        )
    else:
        u, _, _ = VMD(
            signal,
            vmd_settings["alpha"],
            vmd_settings["tau"],
            K,
            vmd_settings["DC"],
            vmd_settings["init"],
            vmd_settings["tol"],
        )
        modes_df = pd.DataFrame(u.T, columns=[f"{col_name}_{i + 1}" for i in range(K)])
        modes_df[f"{col_name}_res"] = signal - modes_df.sum(axis=1)

    return col_name, modes_df


def compute_vmd_features(df_raw, vmd_cols=None, vmd_settings=None, n_cpu=1):
    """Compute VMD modes (1..K + residual) for each column in `vmd_cols`."""
    vmd_cols = list(cfg.VMD_COLS if vmd_cols is None else vmd_cols)
    vmd_settings = dict(cfg.VMD_SETTINGS if vmd_settings is None else vmd_settings)

    tasks = [(col, df_raw[col].to_numpy(dtype=float), vmd_settings) for col in vmd_cols]

    if int(n_cpu) > 1 and len(tasks) > 1:
        max_workers = min(int(n_cpu), len(tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = dict(pool.map(_vmd_single_signal, tasks))
    else:
        results = dict(_vmd_single_signal(t) for t in tasks)

    # Preserve exact deterministic column ordering matching Preprocessor._universal_X_trafo
    ordered_dfs = [results[col] for col in vmd_cols]
    df_vmd_only = pd.concat(ordered_dfs, axis=1)
    df_vmd_only.index = df_raw.index
    return df_vmd_only


def engineer_features(df_raw, n_cpu=1):
    """Apply log1p, sinusoidal seasonal encodings, and VMD decomposition to `df_raw`.

    Matches `Preprocessor._universal_X_trafo` in `transformer_model.py` exactly.
    Returns an unscaled (8760 x 49) DataFrame.
    """
    df_X = df_raw[cfg.X_BASE_COLS].copy()

    # 1. Log1p transform on skewed grid topology proxies (deterministic, in-place)
    for col in cfg.LOG1P_COLS:
        if col in df_X.columns:
            df_X[col] = np.log1p(df_X[col].astype(np.float32))

    # 2. Sinusoidal seasonal encodings for year (8760h), week (168h), and day (24h)
    hours = df_X.index.to_numpy(dtype=np.float64)
    seasonal_cols = []
    for freq in cfg.TS_PERIODS:
        seasonal_cols.append(
            pd.Series(np.sin(2 * np.pi / freq * hours), index=df_X.index, name=f"sin_{freq}")
        )
        seasonal_cols.append(
            pd.Series(np.cos(2 * np.pi / freq * hours), index=df_X.index, name=f"cos_{freq}")
        )
    if seasonal_cols:
        df_X = pd.concat([df_X, pd.concat(seasonal_cols, axis=1)], axis=1)

    # 3. VMD decomposition replacing the 6 raw hourly VMD columns
    df_vmd_modes = compute_vmd_features(df_X, cfg.VMD_COLS, cfg.VMD_SETTINGS, n_cpu=n_cpu)
    df_engineered = pd.concat([df_X.drop(columns=cfg.VMD_COLS), df_vmd_modes], axis=1)

    return df_engineered.astype(np.float32)


def load_scalers(scaler_path=None):
    """Load frozen training StandardScaler parameters from JSON."""
    path = Path(scaler_path if scaler_path is not None else cfg.SCALER_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler file not found at '{path}'. "
            "Run `python3 export_scalers.py` once to export training scalers from ts_train.h5."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scale_features(df_vmd, scaler_data):
    """Standardize `df_vmd` using the frozen training `scaler_X` parameters.

    Reproduces `sklearn.preprocessing.StandardScaler.transform` identically
    (leaving seasonal sin/cos columns unscaled and enforcing exact column order).
    """
    scale_cols = scaler_data["scale_feature_names"]
    feature_names = scaler_data["feature_names"]

    missing = [c for c in scale_cols if c not in df_vmd.columns]
    if missing:
        raise ValueError(f"Missing engineered feature(s) required by scaler_X: {missing}")

    mean = np.asarray(scaler_data["scaler_X_mean"], dtype=np.float64)
    scale = np.asarray(scaler_data["scaler_X_scale"], dtype=np.float64)
    # Match sklearn StandardScaler handling of zero-variance features (scale == 0 -> 1.0)
    safe_scale = np.where(scale == 0.0, 1.0, scale)

    df_scaled = df_vmd.copy()
    vals = df_scaled[scale_cols].to_numpy(dtype=np.float64)
    df_scaled[scale_cols] = ((vals - mean) / safe_scale).astype(np.float32)

    # Enforce exact column ordering expected by the Transformer checkpoint
    df_scaled = df_scaled.reindex(columns=feature_names).astype(np.float32)

    if df_scaled.isna().any().any():
        nan_cols = df_scaled.columns[df_scaled.isna().any()].tolist()
        raise ValueError(f"NaN values detected in scaled features: {nan_cols}")

    return df_scaled


def preprocess_features(df_raw, scaler_path=None, n_cpu=1, skip_scaling=False):
    """Run full preprocessing: feature engineering + VMD + StandardScaler.

    Returns `(df_vmd, df_scaled)`. If `skip_scaling=True`, `df_scaled` is `None`.
    """
    df_vmd = engineer_features(df_raw, n_cpu=n_cpu)
    if skip_scaling:
        return df_vmd, None

    scaler_data = load_scalers(scaler_path)
    df_scaled = scale_features(df_vmd, scaler_data)
    return df_vmd, df_scaled
