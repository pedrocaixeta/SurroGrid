#!/usr/bin/env python3
"""Unified ("direct") end-to-end Inference orchestrator for a single LV grid.

Runs all three verified inference stages in one job via isolated subprocesses,
using a temporary node-local staging directory (`$TMPDIR` / `tempfile`), and
writes a single final `<raw_stem>_direc_Inferred.h5` file into `STORAGE_DIR`
only after all stages succeed:

1. `1.Cheap_Ground_Truth/main.py`
   - Demand allocation (without mobility), MV building disconnection, and
     pre-expansion pandapower load flow (`OMP/MKL/OPENBLAS=1`).
2. `2.Feature_and_Preprocessing/main.py`
   - 19 raw baseline features -> log1p + seasonal sin/cos + VMD (K=4) ->
     frozen `scaler_X` (`8760 x 49`, `OMP/MKL/OPENBLAS=1`).
3. `3.Forward_Pass/main.py`
   - Transformer surrogate forward pass (`baseline_alpha_peak_HPOfromOct17th.pt`)
     + inverse `scaler_y` -> hourly `p_mw`, `q_mvar`, `s_mva`, `loading_pct`,
     and `/surrogate_out/summary` (`OMP/MKL=n_cpu`).
4. Repack & atomic completion
   - Copies the configured HDF5 keys (`keep_mode="medium"` by default) into a
     clean `.h5` file so dropped intermediates actually reclaim disk space.
   - Appends a consolidated `direct_inference` row to `/performance`.
   - Atomically renames to `<raw_stem>_direc_Inferred.h5`.

Paths and retention settings are configured in `pipeline_config.py` and can
also be overridden via CLI flags.
"""

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Disable HDF5 file locking on network / HPC filesystems
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import pandas as pd
import tables  # noqa: F401  # Registers PyTables' blosc filter for h5py.copy

import pipeline_config as cfg


# --------------------------------------------------------------------------- #
# ----------------------------- Helper Utilities ---------------------------- #
# --------------------------------------------------------------------------- #
def find_input_grid(directory, file_id, start_from_step=1):
    """Locate the input `.h5` file matching `file_id` in `directory`.

    Supports unpadded ('197'), 4-digit ('0197'), and 5-digit ('51431') IDs.
    When `start_from_step == 1`, ignores already-processed output files.
    When `start_from_step == 2`, prefers `*_DmdAllocated.h5`.
    """
    if start_from_step == 1:
        exclude_suffixes = (
            "_DmdAllocated.h5",
            "_Features.h5",
            "_Predicted.h5",
            f"{cfg.OUTPUT_SUFFIX}.h5",
            "_pwrflw.h5",
        )
    else:
        exclude_suffixes = (
            "_Features.h5",
            "_Predicted.h5",
            f"{cfg.OUTPUT_SUFFIX}.h5",
            "_pwrflw.h5",
        )

    all_entries = sorted(os.listdir(directory))
    h5_files = [
        fname
        for fname in all_entries
        if fname.endswith(".h5") and not fname.endswith(exclude_suffixes)
    ]

    try:
        val = int(file_id)
        valid_prefixes = (str(val), f"{val:04d}", f"{val:05d}")
    except ValueError:
        valid_prefixes = (str(file_id), str(file_id).zfill(4), str(file_id).zfill(5))

    matched_files = [fname for fname in h5_files if fname.split("_", 1)[0] in valid_prefixes]
    if not matched_files:
        raise IndexError(
            f"No input grid file found in '{directory}' matching ID '{file_id}' "
            f"(start_from_step={start_from_step})."
        )

    if start_from_step == 2:
        dmd_matches = [f for f in matched_files if f.endswith("_DmdAllocated.h5")]
        if dmd_matches:
            return dmd_matches[0]

    return matched_files[0]


def _derive_raw_stem(input_filename):
    """Strip `.h5` and any known stage suffix to recover the clean raw grid stem."""
    stem = os.path.splitext(input_filename)[0]
    for suffix in ("_Predicted", "_Features", "_DmdAllocated", cfg.OUTPUT_SUFFIX):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _build_stage_env(n_cpu, blas_single_thread=True):
    """Construct a clean environment dict for a subprocess stage."""
    env = os.environ.copy()
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    if blas_single_thread:
        # Prevent nested BLAS/OpenMP threads inside multiprocessing worker pools
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[var] = "1"
    else:
        # Match PyTorch CPU intra-op threads to allocated CPUs
        threads = str(max(1, int(n_cpu)))
        env["OMP_NUM_THREADS"] = threads
        env["MKL_NUM_THREADS"] = threads
        env["OPENBLAS_NUM_THREADS"] = threads
        env["PYTORCH_NVML_DISABLE"] = "1"

    return env


def _run_stage(stage_label, cmd, cwd, env):
    """Execute one inference stage subprocess and raise on non-zero exit."""
    print(f"\n{'=' * 72}\n[{stage_label}] Starting...\n  cwd: {cwd}\n  cmd: {' '.join(cmd)}\n{'=' * 72}", flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
    elapsed = time.perf_counter() - t0
    print(f"[{stage_label}] Finished in {elapsed:.2f} s.", flush=True)


def _find_single_h5(directory, required_suffix=None):
    """Return the path to the single expected `.h5` file inside a staging directory."""
    matches = sorted(
        fname
        for fname in os.listdir(directory)
        if fname.endswith(".h5") and (required_suffix is None or fname.endswith(required_suffix))
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected 1 staged HDF5 file (suffix={required_suffix}) in '{directory}', "
            f"found {len(matches)}: {matches}"
        )
    return os.path.join(directory, matches[0])


def _pack_final_hdf5(src_h5_path, dst_h5_path, keep_mode):
    """Write the final HDF5 file according to `keep_mode` ('slim', 'medium', 'full').

    Copies selected HDF5 nodes directly into a fresh file via `h5py` so that
    PyTables/blosc datasets and raw HDF5 datasets (e.g. `raw_data/net`) are
    preserved losslessly while omitted groups reclaim full disk space.
    """
    keep_keys = cfg.HDF5_KEEP_KEYS[keep_mode]
    if keep_keys is None:
        shutil.copy2(src_h5_path, dst_h5_path)
        return

    with h5py.File(src_h5_path, "r") as fin, h5py.File(dst_h5_path, "w") as fout:
        for key in keep_keys:
            if key not in fin:
                print(f"Warning: Expected HDF5 key '{key}' not found in '{src_h5_path}', skipping.")
                continue
            parent = os.path.dirname(key)
            name = os.path.basename(key)
            target = fout.require_group(parent) if parent else fout
            fin.copy(fin[key], target, name=name)


def _record_performance(output_path, step_name, t_start, ot_start, t_start_str):
    """Append consolidated wall/CPU/RSS/SLURM metrics to `/performance`."""
    try:
        t_end = time.perf_counter()
        ot_end = os.times()
        t_end_str = time.strftime("%Y-%m-%d %H:%M:%S")
        wall_sec = max(0.0, t_end - t_start)

        # Include both parent and child subprocess CPU time
        user_sec = max(0.0, ot_end[0] - ot_start[0]) + max(0.0, ot_end[2] - ot_start[2])
        system_sec = max(0.0, ot_end[1] - ot_start[1]) + max(0.0, ot_end[3] - ot_start[3])
        cpu_sec = user_sec + system_sec

        try:
            import resource
            ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ru_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            ru_max = max(ru_self, ru_child)
            peak_rss = int(ru_max) * 1024 if platform.system() == "Linux" else int(ru_max)
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


# --------------------------------------------------------------------------- #
# ------------------------ Core Single-Grid Runner -------------------------- #
# --------------------------------------------------------------------------- #
def run_direct_inference(
    inputfile_id,
    n_cpu=1,
    device="auto",
    input_dir=None,
    output_dir=None,
    model_path=None,
    scaler_path=None,
    keep_mode=None,
    tmp_dir=None,
    torch_python=None,
    start_from_step=1,
):
    """Execute Steps 1 -> 2 -> 3 for a single grid and return the final `.h5` path.

    Designed as a self-contained callable so future multi-grid DSO batching
    can invoke `run_direct_inference(...)` directly in a loop.
    """
    t_start = time.perf_counter()
    ot_start = os.times()
    t_start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    input_dir = str(Path(input_dir if input_dir is not None else cfg.DATA_GRID_DIR).resolve())
    output_dir = str(Path(output_dir if output_dir is not None else cfg.STORAGE_DIR).resolve())
    model_path = str(Path(model_path if model_path is not None else cfg.MODEL_PATH).resolve())
    scaler_path = str(Path(scaler_path if scaler_path is not None else cfg.SCALER_PATH).resolve())
    keep_mode = keep_mode if keep_mode is not None else cfg.DEFAULT_KEEP_MODE
    step3_python = torch_python if torch_python is not None else sys.executable

    if keep_mode not in cfg.HDF5_KEEP_KEYS:
        raise ValueError(f"Unsupported keep_mode '{keep_mode}'. Choose from {list(cfg.HDF5_KEEP_KEYS)}.")

    # Verify input file upfront before allocating temp staging directories
    input_filename = find_input_grid(input_dir, inputfile_id, start_from_step=start_from_step)
    raw_stem = _derive_raw_stem(input_filename)
    os.makedirs(output_dir, exist_ok=True)

    final_path = os.path.join(output_dir, f"{raw_stem}{cfg.OUTPUT_SUFFIX}.h5")
    inprogress_path = os.path.join(output_dir, f".{raw_stem}{cfg.OUTPUT_SUFFIX}.inprogress.h5")

    print(
        f"Direct Inference Pipeline | ID={inputfile_id} | input='{input_filename}' | "
        f"n_cpu={n_cpu} | device={device} | keep_mode={keep_mode} | start_from_step={start_from_step}",
        flush=True,
    )

    env_blas1 = _build_stage_env(n_cpu=n_cpu, blas_single_thread=True)
    env_torch = _build_stage_env(n_cpu=n_cpu, blas_single_thread=False)

    # All intermediate outputs live in node-local temp ($TMPDIR) and are auto-cleaned on exit
    with tempfile.TemporaryDirectory(prefix=f"surrogrid_inf_{inputfile_id}_", dir=tmp_dir) as work_dir:
        step1_dir = os.path.join(work_dir, "step1")
        step2_dir = os.path.join(work_dir, "step2")
        step3_dir = os.path.join(work_dir, "step3")
        os.makedirs(step1_dir, exist_ok=True)
        os.makedirs(step2_dir, exist_ok=True)
        os.makedirs(step3_dir, exist_ok=True)

        # ---- Stage 1: Cheap Ground Truth (Demand Allocation + MV Disconnect + Pre-PF) ---- #
        if start_from_step == 1:
            cmd_step1 = [
                sys.executable,
                cfg.STEP1_SCRIPT,
                str(inputfile_id),
                "--n_cpu",
                str(n_cpu),
                "--input_dir",
                input_dir,
                "--output_dir",
                step1_dir,
            ]
            _run_stage("Stage 1/3: Cheap Ground Truth", cmd_step1, cfg.STEP1_DIR, env_blas1)
            step1_out = _find_single_h5(step1_dir, required_suffix="_DmdAllocated.h5")
            step2_input_dir = step1_dir
        else:
            step1_out = None
            step2_input_dir = input_dir

        # ---- Stage 2: Feature Extraction & Preprocessing (VMD + Scaler) ---- #
        cmd_step2 = [
            sys.executable,
            cfg.STEP2_SCRIPT,
            str(inputfile_id),
            "--n_cpu",
            str(n_cpu),
            "--input_dir",
            step2_input_dir,
            "--output_dir",
            step2_dir,
            "--scaler_path",
            scaler_path,
        ]
        _run_stage("Stage 2/3: Feature Extraction & Preprocessing", cmd_step2, cfg.STEP2_DIR, env_blas1)
        step2_out = _find_single_h5(step2_dir, required_suffix="_Features.h5")

        # Free Stage 1 staging file immediately to minimize peak temp disk usage
        if step1_out is not None and os.path.exists(step1_out):
            os.remove(step1_out)

        # ---- Stage 3: Surrogate Forward Pass (Transformer + Inverse Scaler) ---- #
        cmd_step3 = [
            step3_python,
            cfg.STEP3_SCRIPT,
            str(inputfile_id),
            "--n_cpu",
            str(n_cpu),
            "--device",
            str(device),
            "--input_dir",
            step2_dir,
            "--output_dir",
            step3_dir,
            "--model_path",
            model_path,
            "--scaler_path",
            scaler_path,
        ]
        _run_stage("Stage 3/3: Surrogate Forward Pass", cmd_step3, cfg.STEP3_DIR, env_torch)
        step3_out = _find_single_h5(step3_dir, required_suffix="_Predicted.h5")

        # Free Stage 2 staging file immediately
        if os.path.exists(step2_out):
            os.remove(step2_out)

        # ---- Stage 4: Pack Final HDF5, Log Performance & Atomic Rename ---- #
        print(f"\n[Finalizing] Packing '{keep_mode}' HDF5 -> {final_path}", flush=True)
        try:
            if os.path.exists(inprogress_path):
                os.remove(inprogress_path)

            _pack_final_hdf5(step3_out, inprogress_path, keep_mode=keep_mode)
            _record_performance(inprogress_path, "direct_inference", t_start, ot_start, t_start_str)
            os.replace(inprogress_path, final_path)
        except Exception:
            if os.path.exists(inprogress_path):
                os.remove(inprogress_path)
            raise

    final_mb = os.path.getsize(final_path) / 1e6
    total_wall = time.perf_counter() - t_start
    print(
        f"\nAll 3 stages finished successfully in {total_wall:.2f} s! "
        f"Saved ({final_mb:.2f} MB, keep_mode='{keep_mode}'): {final_path}",
        flush=True,
    )
    return final_path


# --------------------------------------------------------------------------- #
# ---------------------------------- CLI ------------------------------------ #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Unified direct inference runner (Steps 1 -> 2 -> 3) for a single LV grid."
    )
    parser.add_argument("inputfile_id", help="Grid ID prefix (e.g., 0197 or 51433)")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=1,
        help="Number of CPUs for multiprocessing (Steps 1-2) and PyTorch threads (Step 3)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for Step 3 forward pass ('auto' uses cuda if available, else cpu)",
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
    parser.add_argument(
        "--keep_mode",
        default=cfg.DEFAULT_KEEP_MODE,
        choices=list(cfg.HDF5_KEEP_KEYS.keys()),
        help=(
            "HDF5 keys to retain in final output: "
            "'slim' (~5-6 MB), 'medium' (~40 MB, default, keeps /urbs_in/*), 'full' (~120 MB)."
        ),
    )
    parser.add_argument(
        "--tmp_dir",
        default=None,
        help="Optional parent directory for staging temp files (defaults to $TMPDIR / system temp)",
    )
    parser.add_argument(
        "--torch_python",
        default=None,
        help=(
            "Optional Python executable for Step 3 only (defaults to current interpreter; "
            "useful in WSL when 'inference' omits PyTorch and 'training' provides it)."
        ),
    )
    parser.add_argument(
        "--start_from_step",
        type=int,
        default=1,
        choices=[1, 2],
        help="Start from Step 1 (raw .h5, default) or Step 2 (existing *_DmdAllocated.h5).",
    )
    args = parser.parse_args()

    run_direct_inference(
        inputfile_id=args.inputfile_id,
        n_cpu=args.n_cpu,
        device=args.device,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        keep_mode=args.keep_mode,
        tmp_dir=args.tmp_dir,
        torch_python=args.torch_python,
        start_from_step=args.start_from_step,
    )


if __name__ == "__main__":
    main()
