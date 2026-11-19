#!/usr/bin/env python3
"""Global orchestrator for Inference/3.feature_generation (Steps 2 & 3).

For a single `*_DmdAllocated.h5` file produced by `Inference/1.Cheap_Ground_Truth`:
1. Copies the input `.h5` file from `DATA_GRID_DIR` into `STORAGE_DIR`.
2. Extracts the 19 raw base features (`extract_features.py`) and writes
   `/surrogate_in/features_raw` (8760 x 19).
3. Runs surrogate feature preprocessing (`preprocess_features.py`):
   - Applies `log1p`, sinusoidal seasonal encodings, and VMD (`K=4`) -> writes
     `/surrogate_in/features_vmd` (8760 x 49, unscaled).
   - Applies frozen training `StandardScaler` (`scaler_X`) -> writes
     `/surrogate_in/features_scaled` (8760 x 49, ready for Step 4 forward pass).
4. Appends performance metadata to `/performance`.
5. Renames `<stem>_DmdAllocated.h5` -> `<stem>_DmdAllocated_Features.h5` ONLY
   after all steps finish successfully.

Paths and feature settings are defined in `pipeline_config.py`.
"""

import argparse
import os
import platform
import socket
import time

# Disable HDF5 file locking on network / HPC filesystems
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import pandas as pd

import pipeline_config as cfg
from save_grid import SaveFile, find_grid_file_by_id
from resource_report import resource_report
from extract_features import extract_raw_features
from preprocess_features import preprocess_features


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
        description="Inference Steps 2 & 3: Extract base features, run VMD, and scale for Transformer."
    )
    parser.add_argument("inputfile_id", help="Grid ID prefix (e.g., 0197 or 51431)")
    parser.add_argument(
        "--n_cpu", type=int, default=1, help="Number of CPUs for parallel VMD decomposition"
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
        "--scaler_path",
        default=None,
        help="Optional override for pipeline_config.SCALER_PATH",
    )
    parser.add_argument(
        "--skip_scaling",
        action="store_true",
        help="Debug option: skip StandardScaler step (only saves features_raw and features_vmd)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir is not None else cfg.DATA_GRID_DIR
    output_dir = args.output_dir if args.output_dir is not None else cfg.STORAGE_DIR

    with resource_report(include_children=True, name="Feature Generation & Preprocessing Pipeline"):
        input_filename = find_grid_file_by_id(input_dir, args.inputfile_id)
        sf = SaveFile(input_filename, input_dir=input_dir, output_dir=output_dir)

        print(
            f"Running feature generation on {input_filename} "
            f"(ID {args.inputfile_id}) with {args.n_cpu} CPU(s)..."
        )

        # 1. Copy input .h5 into STORAGE_DIR so upstream files stay untouched
        sf.copy_save_file()

        # 2. Extract 19 raw base features
        with resource_report(include_children=True, name="Step 2: Raw Feature Extraction"):
            df_raw = extract_raw_features(sf.output_path)
            sf.save_df(df_raw, cfg.H5_KEY_FEATURES_RAW)
            print(f"Saved {cfg.H5_KEY_FEATURES_RAW}: shape={df_raw.shape}")

        # 3. Feature engineering (log1p, seasonal sin/cos, VMD) + StandardScaler
        with resource_report(include_children=True, name="Step 3: VMD & Scaling"):
            df_vmd, df_scaled = preprocess_features(
                df_raw,
                scaler_path=args.scaler_path,
                n_cpu=args.n_cpu,
                skip_scaling=args.skip_scaling,
            )
            sf.save_df(df_vmd, cfg.H5_KEY_FEATURES_VMD)
            print(f"Saved {cfg.H5_KEY_FEATURES_VMD}: shape={df_vmd.shape}")

            if df_scaled is not None:
                sf.save_df(df_scaled, cfg.H5_KEY_FEATURES_SCALED)
                print(f"Saved {cfg.H5_KEY_FEATURES_SCALED}: shape={df_scaled.shape}")

        # 4. Log performance metrics and rename file with completion suffix
        _record_performance(sf.output_path, "feature_generation", t_start, ot_start, t_start_str)
        final_path = SaveFile.append_suffix(sf.output_path, cfg.OUTPUT_SUFFIX)
        print(f"All steps finished successfully! Saved: {final_path}")


if __name__ == "__main__":
    main()
