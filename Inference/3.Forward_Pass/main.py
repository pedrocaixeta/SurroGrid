#!/usr/bin/env python3
"""Global orchestrator for Inference/3.Forward_Pass (Step 4).

For a single `*_DmdAllocated_Features.h5` file from `2.Feature_and_Preprocessing`:
1. Copies the input `.h5` file from `DATA_GRID_DIR` into `STORAGE_DIR`.
2. Loads `/surrogate_in/features_scaled` (e.g., 8760 x 45) and validates exact feature order.
3. Loads the trained Transformer checkpoint (`baseline_mae_maex_HPOfromOct15th.pt`)
   and its matching frozen training scalers (`scalers_*.json`).
4. Runs the circularly-padded forward pass and inverse-scales `(P, Q)` to MW / MVAr.
5. Extracts the transformer's nominal capacity (`sn_mva`) from `/raw_data/net`
   to compute hourly loading (%) and a grid-level overload summary.
6. Saves `/surrogate_out/demand_import` and `/surrogate_out/summary`, records
   `/performance` metadata, and renames `<stem>_Features.h5` ->
   `<stem>_Features_Predicted.h5` ONLY after all steps finish successfully.

Paths and model settings are defined in `pipeline_config.py`.
"""

import argparse
import io
import json
import os
import platform
import socket
import time

# Disable HDF5 file locking on network / HPC filesystems
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTORCH_NVML_DISABLE", "1")

import h5py
import numpy as np
import pandas as pd
import torch

import pipeline_config as cfg
from save_grid import SaveFile, find_grid_file_by_id
from resource_report import resource_report
from transformer_model import load_scalers, load_surrogate_model, run_forward_pass


def _extract_trafo_sn_mva(h5_path: str) -> float:
    """Read total in-service transformer nominal capacity (MVA) from `/raw_data/net` JSON."""
    try:
        with h5py.File(str(h5_path), "r") as f:
            if "raw_data/net" not in f:
                return float("nan")
            raw_net = f["raw_data/net"][()]
            if isinstance(raw_net, bytes):
                raw_net = raw_net.decode("utf-8")
            net_data = json.loads(raw_net)

        trafo_info = net_data.get("_object", {}).get("trafo")
        if trafo_info is None:
            return float("nan")

        df_trafo = pd.read_json(
            io.StringIO(trafo_info["_object"]), orient=trafo_info["orient"]
        )
        if df_trafo.empty or "sn_mva" not in df_trafo.columns:
            return float("nan")
        if "in_service" in df_trafo.columns:
            df_trafo = df_trafo[df_trafo["in_service"] != False]
        return float(df_trafo["sn_mva"].sum())
    except Exception as e:
        print(f"Warning: Could not read transformer sn_mva from '{h5_path}': {e}")
        return float("nan")


def _record_performance(output_path, step_name, t_start, ot_start, t_start_str):
    """Append wall/CPU/RSS/SLURM metrics to `/performance` inside `output_path`."""
    try:
        t_end = time.perf_counter()
        ot_end = os.times()
        t_end_str = time.strftime("%Y-%m-%d %H:%M:%S")
        wall_sec = max(0.0, t_end - t_start)

        user_sec = max(0.0, ot_end[0] - ot_start[0]) + max(0.0, ot_end[2] - ot_start[2])
        system_sec = max(0.0, ot_end[1] - ot_start[1]) + max(0.0, ot_end[3] - ot_start[3])
        cpu_sec = user_sec + system_sec

        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_rss = int(ru) * 1024 if platform.system() == "Linux" else int(ru)
        except Exception:
            peak_rss = 0

        cpu_model = "Unknown"
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            cpu_model = platform.processor() or "Unknown"

        new_entry = {
            "step": step_name,
            "timestamp_start": t_start_str,
            "timestamp_end": t_end_str,
            "wall_seconds": wall_sec,
            "cpu_seconds": cpu_sec,
            "peak_rss_bytes": peak_rss,
            "cpu_model": cpu_model,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "N/A"),
            "slurm_cpus_allocated": os.environ.get(
                "SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_ON_NODE", "1")
            ),
            "slurm_mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU", "N/A"),
            "slurm_cluster": os.environ.get("SLURM_CLUSTER_NAME", "N/A"),
        }

        df_new = pd.DataFrame([new_entry])
        try:
            df_existing = pd.read_hdf(output_path, key="performance")
            df_existing = df_existing[df_existing["step"] != step_name]
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_all = df_new

        with pd.HDFStore(output_path, mode="a") as store:
            store.put("performance", df_all, format="fixed")
    except Exception as e:
        print(f"Warning: Failed to write performance metadata: {e}")


def main():
    t_start = time.perf_counter()
    ot_start = os.times()
    t_start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Inference Step 4: Run Transformer surrogate forward pass."
    )
    parser.add_argument("inputfile_id", help="Grid ID prefix (e.g., 0197 or 51431)")
    parser.add_argument(
        "--n_cpu", type=int, default=4, help="Number of CPU threads for PyTorch inference"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on ('auto' uses cuda if available, else cpu)",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Optional override for pipeline_config.DATA_GRID_DIR",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional override for pipeline_config.STORAGE_DIR",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional override for pipeline_config.MODEL_PATH",
    )
    parser.add_argument(
        "--scaler_path",
        default=None,
        help="Optional override for pipeline_config.SCALER_PATH",
    )
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir is not None else cfg.DATA_GRID_DIR
    output_dir = args.output_dir if args.output_dir is not None else cfg.STORAGE_DIR
    model_path = args.model_path if args.model_path is not None else cfg.MODEL_PATH
    scaler_path = args.scaler_path if args.scaler_path is not None else cfg.SCALER_PATH

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cpu":
        torch.set_num_threads(max(1, int(args.n_cpu)))

    with resource_report(include_children=True, name="Surrogate Forward Pass Pipeline"):
        input_filename = find_grid_file_by_id(input_dir, args.inputfile_id)
        sf = SaveFile(input_filename, input_dir=input_dir, output_dir=output_dir)

        print(
            f"Running Transformer forward pass on {input_filename} "
            f"(ID {args.inputfile_id}) on device='{device}' ({args.n_cpu} thread(s))..."
        )

        # 1. Copy input *_Features.h5 into STORAGE_DIR so upstream files stay untouched
        sf.copy_save_file()

        # 2. Load scaled 8760 x 49 features, scalers, and trained Transformer checkpoint
        df_scaled = pd.read_hdf(sf.output_path, key=cfg.H5_KEY_FEATURES_SCALED)
        scaler_data = load_scalers(scaler_path)
        model, model_cfg = load_surrogate_model(model_path, device=device)

        # 3. Execute forward pass and inverse-scale outputs to MW / MVAr / MVA
        with resource_report(include_children=True, name="Step 4: Transformer Forward Pass"):
            df_pred = run_forward_pass(model, model_cfg, df_scaled, scaler_data, device=device)

        # 4. Compare predicted apparent power S_MVA against transformer nominal capacity sn_mva
        sn_mva = _extract_trafo_sn_mva(sf.output_path)
        if np.isfinite(sn_mva) and sn_mva > 0:
            df_pred["loading_pct"] = (df_pred["s_mva"] / sn_mva * 100.0).astype(np.float32)
            max_loading_pct = float(df_pred["loading_pct"].max())
            hours_overloaded = int((df_pred["loading_pct"] > 100.0).sum())
            needs_replacement = bool(max_loading_pct > 100.0)
        else:
            df_pred["loading_pct"] = np.nan
            max_loading_pct = float("nan")
            hours_overloaded = -1
            needs_replacement = False

        df_summary = pd.DataFrame(
            [
                {
                    "model_name": scaler_data.get("model_name", os.path.basename(model_path)),
                    "trafo_sn_mva": sn_mva,
                    "peak_p_import_mw": float(df_pred["p_mw"].max()),
                    "peak_p_feedin_mw": float(df_pred["p_mw"].min()),
                    "peak_s_mva": float(df_pred["s_mva"].max()),
                    "max_loading_pct": max_loading_pct,
                    "hours_overloaded_100pct": hours_overloaded,
                    "needs_replacement": needs_replacement,
                }
            ]
        )

        # 5. Save hourly predictions and summary back into HDF5
        sf.save_df(df_pred, cfg.H5_KEY_PREDICTION)
        sf.save_df(df_summary, cfg.H5_KEY_SUMMARY)
        print(
            f"Saved {cfg.H5_KEY_PREDICTION} (shape={df_pred.shape}) | "
            f"peak_P_import={df_pred['p_mw'].max():.4f} MW, "
            f"peak_P_feedin={df_pred['p_mw'].min():.4f} MW, "
            f"peak_S={df_pred['s_mva'].max():.4f} MVA vs sn_mva={sn_mva:.3f} MVA "
            f"(max loading={max_loading_pct:.1f}%)"
        )

        # 6. Log performance metrics and rename file with '_Predicted' suffix
        _record_performance(sf.output_path, "forward_pass", t_start, ot_start, t_start_str)
        final_path = SaveFile.append_suffix(sf.output_path, cfg.OUTPUT_SUFFIX)
        print(f"All steps finished successfully! Saved: {final_path}")


if __name__ == "__main__":
    main()
