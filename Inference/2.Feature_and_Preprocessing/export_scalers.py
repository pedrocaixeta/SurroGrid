#!/usr/bin/env python3
"""One-time utility to export frozen training scalers (`scaler_X`, `scaler_y`).

Saved `.pt` checkpoints from `train_combined_data.py` only store
`{'config', 'state_dict'}` and do not persist the fitted `StandardScaler`
objects. This script reproduces the exact train/val split and
`Preprocessor.fit_transform` from `transformer_model.py` on `ts_train.h5` and
saves the resulting scaler parameters to a small portable JSON file.

Memory-safe design:
- Drops unused columns and frees raw DataFrames immediately.
- Uses `StandardScaler.partial_fit()` grid-by-grid so peak RAM stays low even
  across ~1,000 training grids.

Usage:
  # Fast (~5 s on LRZ using existing VMD4 cache):
  python3 export_scalers.py \
      --ts_train_path /dss/.../GridForecast/0_preprocessing/Data/ts_train.h5 \
      --vmd_cache_path /dss/.../GridForecast/3_transformer/data/VMD4_PVprodexpected_T_copavg_demandnetactivepre_heatspace_heatwater.h5

  # Without VMD4 cache (computes VMD4 in parallel across --n_cpu workers):
  python3 export_scalers.py \
      --ts_train_path ../../0.Legacy_Folders/GridForecast/0_preprocessing/Data/ts_train.h5 \
      --n_cpu 4
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from vmdpy import VMD

import pipeline_config as cfg


def _vmd_one_grid(args):
    """Compute VMD4 modes + residual for all VMD_COLS of a single training grid."""
    grid_id, values_dict, vmd_settings = args
    K = int(vmd_settings["K"])
    cols = []
    arrays = []
    for col, signal in values_dict.items():
        signal = np.asarray(signal, dtype=float)
        if np.all(signal == 0):
            modes = np.zeros((len(signal), K + 1), dtype=np.float64)
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
            u_t = u.T
            res = (signal - u_t.sum(axis=1))[:, None]
            modes = np.concatenate([u_t, res], axis=1)
        cols.extend([f"{col}_{i + 1}" for i in range(K)] + [f"{col}_res"])
        arrays.append(modes)
    return grid_id, np.concatenate(arrays, axis=1).astype(np.float64)


def _parse_ts_periods(raw):
    """Parse TS_PERIODS from list[int] or comma-separated string (e.g., '8760,168,24,12')."""
    if raw is None:
        return list(cfg.TS_PERIODS)
    if isinstance(raw, str):
        s = raw.strip()
        return [int(p.strip()) for p in s.replace(" ", ",").split(",") if p.strip()] if s else []
    return [int(p) for p in raw]


def _build_feature_layout(ts_periods, vmd_k_modes):
    """Return exact (feature_names, scale_feature_names, non_vmd_cols, period_names, vmd_cols_out)."""
    period_names = []
    for freq in ts_periods:
        period_names.extend([f"sin_{freq}", f"cos_{freq}"])

    if int(vmd_k_modes) >= 1 and cfg.VMD_COLS:
        non_vmd_base = [c for c in cfg.X_BASE_COLS if c not in cfg.VMD_COLS]
        vmd_cols_out = [
            f"{base}_{k + 1}" if k < int(vmd_k_modes) else f"{base}_res"
            for base in cfg.VMD_COLS
            for k in range(int(vmd_k_modes) + 1)
        ]
    else:
        non_vmd_base = list(cfg.X_BASE_COLS)
        vmd_cols_out = []

    feature_names = non_vmd_base + period_names + vmd_cols_out
    scale_feature_names = non_vmd_base + vmd_cols_out
    return feature_names, scale_feature_names, non_vmd_base, period_names, vmd_cols_out


def export_scalers(
    ts_train_path,
    output_path=None,
    vmd_cache_path=None,
    n_cpu=2,
    max_grids=None,
    model_name="baseline_alpha_peak_HPOfromOct17th",
    vmd_k_modes=None,
    ts_periods=None,
):
    """Fit scaler_X and scaler_y identically to `train_combined_data.py` and save JSON."""
    t0 = time.perf_counter()
    output_path = Path(output_path if output_path is not None else cfg.SCALER_PATH)
    vmd_k_modes = int(cfg.VMD_K_MODES if vmd_k_modes is None else vmd_k_modes)
    ts_periods = _parse_ts_periods(ts_periods)
    vmd_settings = dict(cfg.VMD_SETTINGS)
    vmd_settings["K"] = vmd_k_modes

    feature_names, scale_feature_names, non_vmd_base, period_names, vmd_cols_out = (
        _build_feature_layout(ts_periods=ts_periods, vmd_k_modes=vmd_k_modes)
    )

    print(f"Loading training targets from: {ts_train_path}")
    # Load y first (only 4 cols, very small) to determine exact grid IDs & split
    df_y_full = pd.read_hdf(ts_train_path, key="y")
    unique_grids = df_y_full.index.get_level_values(0).unique().tolist()

    # 1. Reproduce exact train/val split from TransformerTrainer._auto_load_data
    rng = random.Random(cfg.TRAIN_RANDOM_STATE)
    rng.shuffle(unique_grids)
    n_val = max(1, int(len(unique_grids) * cfg.TRAIN_VALIDATION_RATIO))
    train_grids_all = set(unique_grids[n_val:])

    if max_grids is not None:
        train_grids_all = set(sorted(train_grids_all)[: int(max_grids)])
        print(f"WARNING: Dry-run mode with max_grids={len(train_grids_all)}!")

    # Fit scaler_y on S = sqrt(P^2 + Q^2) and immediately free df_y
    train_mask_y = df_y_full.index.get_level_values(0).isin(train_grids_all)
    p_col, q_col = cfg.TARGET_COLS
    P = df_y_full.loc[train_mask_y, p_col].astype("float32").to_numpy().reshape(-1, 1)
    Q = df_y_full.loc[train_mask_y, q_col].astype("float32").to_numpy().reshape(-1, 1)
    S = np.sqrt(P ** 2 + Q ** 2).astype("float32")
    scaler_y = StandardScaler()
    scaler_y.fit(S)
    n_rows_expected = int(len(S))
    del df_y_full, P, Q, S, train_mask_y
    gc.collect()
    print(
        f"Fitted scaler_y on {len(train_grids_all)} training grids ({n_rows_expected:,} rows): "
        f"mean={scaler_y.mean_[0]:.6f}, scale={scaler_y.scale_[0]:.6f}"
    )

    # 2. Load X and immediately subset to training grids + 19 base columns, then free full table
    print(f"Loading training features from: {ts_train_path}")
    df_X_full = pd.read_hdf(ts_train_path, key="X")
    train_mask_X = df_X_full.index.get_level_values(0).isin(train_grids_all)
    X_train = df_X_full.loc[train_mask_X, cfg.X_BASE_COLS].astype("float32").copy()
    del df_X_full, train_mask_X
    gc.collect()

    # Apply log1p in-place on LOG1P_COLS
    for col in cfg.LOG1P_COLS:
        if col in X_train.columns:
            X_train[col] = np.log1p(X_train[col].astype("float32"))

    # 3. Fit scaler_X using streaming partial_fit (exact same result as batch fit, minimal RAM)
    scaler_X = StandardScaler()

    if vmd_k_modes == 0:
        # No VMD decomposition: fit directly on all base features
        chunk_rows = 100 * cfg.N_HOURS
        for start in range(0, len(X_train), chunk_rows):
            end = min(start + chunk_rows, len(X_train))
            scaler_X.partial_fit(X_train[non_vmd_base].iloc[start:end].to_numpy(dtype=np.float64))
    elif vmd_cache_path is not None and os.path.exists(vmd_cache_path):
        print(f"Loading cached VMD{vmd_k_modes} training data from: {vmd_cache_path} (key='data_train')")
        grid_vmd = pd.read_hdf(vmd_cache_path, key="data_train")
        if len(grid_vmd) != len(X_train):
            raise ValueError(
                f"Cached VMD{vmd_k_modes} row count ({len(grid_vmd)}) does not match X_train ({len(X_train)})."
            )
        missing_vmd = [c for c in vmd_cols_out if c not in grid_vmd.columns]
        if missing_vmd:
            raise ValueError(
                f"Cached VMD file '{vmd_cache_path}' is missing expected columns for K={vmd_k_modes}: {missing_vmd[:5]}"
            )
        grid_vmd = grid_vmd.reindex(columns=vmd_cols_out)
        # Fit chunk-by-chunk (100 grids at a time) to keep peak RAM low
        chunk_rows = 100 * cfg.N_HOURS
        for start in range(0, len(X_train), chunk_rows):
            end = min(start + chunk_rows, len(X_train))
            non_vmd_chunk = X_train[non_vmd_base].iloc[start:end].to_numpy(dtype=np.float64)
            vmd_chunk = grid_vmd.iloc[start:end].to_numpy(dtype=np.float64)
            scaler_X.partial_fit(np.concatenate([non_vmd_chunk, vmd_chunk], axis=1))
        del grid_vmd
        gc.collect()
    else:
        grid_ids = X_train.index.get_level_values(0).unique().tolist()
        print(
            f"Computing VMD (K={vmd_k_modes}) across {len(grid_ids)} grids "
            f"with {n_cpu} CPU(s) using streaming partial_fit..."
        )
        # Pre-extract compact numpy dicts per grid, then free X_train before spawning workers
        non_vmd_by_grid = {}
        tasks = []
        for gid in grid_ids:
            g_df = X_train.xs(gid, level=0)
            non_vmd_by_grid[gid] = g_df[non_vmd_base].to_numpy(dtype=np.float64)
            vmd_dict = {col: g_df[col].to_numpy(dtype=np.float64) for col in cfg.VMD_COLS}
            tasks.append((gid, vmd_dict, vmd_settings))
        del X_train
        gc.collect()

        if int(n_cpu) > 1 and len(tasks) > 1:
            # Use 'spawn' context so workers do not duplicate parent memory pages
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=int(n_cpu), mp_context=ctx) as pool:
                for idx, (gid, vmd_arr) in enumerate(pool.map(_vmd_one_grid, tasks), start=1):
                    scaler_X.partial_fit(np.concatenate([non_vmd_by_grid.pop(gid), vmd_arr], axis=1))
                    if idx % 50 == 0 or idx == len(tasks):
                        print(f"  VMD progress: {idx}/{len(tasks)} grids ({time.perf_counter() - t0:.1f}s)")
        else:
            for idx, task in enumerate(tasks, start=1):
                gid, vmd_arr = _vmd_one_grid(task)
                scaler_X.partial_fit(np.concatenate([non_vmd_by_grid.pop(gid), vmd_arr], axis=1))
                if idx % 50 == 0 or idx == len(tasks):
                    print(f"  VMD progress: {idx}/{len(tasks)} grids ({time.perf_counter() - t0:.1f}s)")

    payload = {
        "model_name": model_name,
        "source_ts_train": str(ts_train_path),
        "train_n_grids": len(train_grids_all),
        "train_n_rows": n_rows_expected,
        "validation_ratio": cfg.TRAIN_VALIDATION_RATIO,
        "random_state": cfg.TRAIN_RANDOM_STATE,
        "X_BASE_COLS": cfg.X_BASE_COLS,
        "LOG1P_COLS": cfg.LOG1P_COLS,
        "TS_PERIODS": ts_periods,
        "VMD_COLS": cfg.VMD_COLS,
        "VMD_K_MODES": vmd_k_modes,
        "VMD_SETTINGS": vmd_settings,
        "TARGET_COLS": cfg.TARGET_COLS,
        "feature_names": feature_names,
        "scale_feature_names": scale_feature_names,
        "scaler_X_mean": [float(v) for v in scaler_X.mean_],
        "scaler_X_scale": [float(v) for v in scaler_X.scale_],
        "scaler_y_mean": float(scaler_y.mean_[0]),
        "scaler_y_scale": float(scaler_y.scale_[0]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Saved scalers ({len(feature_names)} total features, {len(scale_feature_names)} scaled) "
        f"to '{output_path}' in {time.perf_counter() - t0:.1f}s."
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export frozen training scalers for Transformer inference.")
    parser.add_argument(
        "--ts_train_path",
        default=str(
            Path(__file__).resolve().parents[2]
            / "0.Legacy_Folders"
            / "GridForecast"
            / "0_preprocessing"
            / "Data"
            / "ts_train.h5"
        ),
        help="Path to ts_train.h5 used during training",
    )
    parser.add_argument(
        "--vmd_cache_path",
        default=None,
        help="Optional path to precomputed VMD{K}_PVprodexpected_T_copavg_demandnetactivepre_heatspace_heatwater.h5",
    )
    parser.add_argument(
        "--output_path",
        default=cfg.SCALER_PATH,
        help="Output JSON path for frozen scaler parameters",
    )
    parser.add_argument(
        "--model_name",
        default="baseline_alpha_peak_HPOfromOct17th",
        help="Model name recorded inside the exported scaler JSON",
    )
    parser.add_argument(
        "--vmd_k",
        type=int,
        default=cfg.VMD_K_MODES,
        help="Number of VMD modes K (e.g., 3 for baseline_mae_maex_HPOfromOct15th, 4 for alpha_peak)",
    )
    parser.add_argument(
        "--ts_periods",
        default=",".join(str(p) for p in cfg.TS_PERIODS),
        help="Comma-separated seasonal periods in hours (e.g., '8760,168,24,12')",
    )
    parser.add_argument("--n_cpu", type=int, default=2, help="Number of CPUs if VMD must be computed")
    parser.add_argument(
        "--max_grids",
        type=int,
        default=None,
        help="Optional cap on number of training grids (strictly for quick testing)",
    )
    args = parser.parse_args()

    export_scalers(
        ts_train_path=args.ts_train_path,
        output_path=args.output_path,
        vmd_cache_path=args.vmd_cache_path,
        n_cpu=args.n_cpu,
        max_grids=args.max_grids,
        model_name=args.model_name,
        vmd_k_modes=args.vmd_k,
        ts_periods=args.ts_periods,
    )
