#!/usr/bin/env python3
"""Global orchestrator for Inference Step 1 (Demand Allocation & Pre-Powerflow).

Runs the full per-grid preparation workflow inside the universal `inference`
conda environment for a single raw grid `.h5` file from `1.Grid_Importing`:

1. Demand allocation (`gridalloc`):
   - Samples roofs and simulates 1 kW solar profiles (`urbs_in/supim`, `urbs_in/process`)
   - Generates per-building electrical, space heat, and water heat demands (`urbs_in/demand`)
   - Computes per-building air-source heat pump COP (`urbs_in/eff_factor`)
   - Samples vehicle ownership statistics (`n_cars_tot` in `raw_data/buildings`, no `emobpy`)
   - Copies the raw `.h5` into `STORAGE_DIR` and saves all per-building tables
2. MV building disconnection (`disconnect_MV_buildings`):
   - Marks loads with >100 kW (non-residential) or >250 kW (residential/mixed)
     peak electricity demand as `in_service = False` in `raw_data/net`
   - Zeros their demands in `urbs_in/demand`
3. Pre-expansion power flow (`pre_powerflow`):
   - Builds per-bus active + reactive pre-expansion loads (`PF_ELC = 0.959`)
   - Runs 8760-hour pandapower BFSW load flow and saves `/pwrflw/input/demand_pre`
     and `/pwrflw/output/pre/{demand_import, vm, line_loads}`
4. Completion & rename:
   - Appends performance metadata to `/performance`
   - Renames `<stem>.h5` -> `<stem>_DmdAllocated.h5` ONLY after all steps succeed

Paths are configured in `pipeline_config.py` (`DATA_GRID_DIR`, `STORAGE_DIR`,
`OUTPUT_SUFFIX`) and can also be overridden via CLI flags.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import errno
import gc
import os
import platform
import random
import socket
import sys
import time
import traceback
from pathlib import Path

# Disable HDF5 file locking on network / HPC filesystems
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Configure import paths so gridalloc (and unmodified districtgenerator),
# disconnect_MV_buildings, and pre_powerflow can all be imported cleanly.
BASE_DIR = Path(__file__).resolve().parent
GRIDALLOC_DIR = BASE_DIR / "gridalloc"
for path_str in (str(BASE_DIR), str(GRIDALLOC_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np
import pandas as pd

from config import config
from gridalloc.main import run_demand_allocation
from src.classes.save_grid import SaveFile
from src.classes.resource_report import resource_report
from disconnect_MV_buildings.disconnect_MV_buildings import process_hdf5_file as disconnect_mv_buildings
from pre_powerflow.run_pwrflw import run_pre_powerflow_on_file


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


def run_single_grid(
    inputfile_id,
    n_cpu=1,
    input_dir=None,
    output_dir=None,
    grid_filename=None,
    verbose=True,
):
    """Run the complete Step 1 pipeline (demand allocation, MV disconnect, pre-PF) for one grid."""
    t_start = time.perf_counter()
    ot_start = os.times()
    t_start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    if verbose:
        with resource_report(include_children=True, name="Global Demand Allocation Pipeline"):
            with resource_report(include_children=True, name="Step 1: Demand Allocation"):
                working_path = run_demand_allocation(
                    inputfile_id,
                    n_cpu=n_cpu,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    grid_filename=grid_filename,
                )

            with resource_report(include_children=True, name="Step 2: MV Building Disconnection"):
                mv_stats = disconnect_mv_buildings(Path(working_path))
                if mv_stats is None:
                    raise RuntimeError(f"MV building disconnection failed for '{working_path}'.")
                print(
                    f"MV disconnection summary: "
                    f"{mv_stats['matching_found']}/{mv_stats['total_buildings']} building(s) disconnected "
                    f"({mv_stats['remaining_percent']:.2f}% remaining active)."
                )

            with resource_report(include_children=True, name="Step 3: Pre-Expansion Powerflow"):
                run_pre_powerflow_on_file(working_path, n_cpu=n_cpu)

            _record_performance(working_path, "demand_allocation", t_start, ot_start, t_start_str)
            final_path = SaveFile.append_suffix(working_path, config.OUTPUT_SUFFIX)
            print(f"All steps finished successfully! Saved: {final_path}")
    else:
        working_path = run_demand_allocation(
            inputfile_id,
            n_cpu=n_cpu,
            input_dir=input_dir,
            output_dir=output_dir,
            grid_filename=grid_filename,
        )
        mv_stats = disconnect_mv_buildings(Path(working_path))
        if mv_stats is None:
            raise RuntimeError(f"MV building disconnection failed for '{working_path}'.")
        run_pre_powerflow_on_file(working_path, n_cpu=n_cpu)
        _record_performance(working_path, "demand_allocation", t_start, ot_start, t_start_str)
        final_path = SaveFile.append_suffix(working_path, config.OUTPUT_SUFFIX)

    return final_path, time.perf_counter() - t_start


def _is_storage_full_error(exc, tb_str=""):
    """Return True if `exc` (or its chain / traceback) indicates disk quota or space exhaustion."""
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, OSError) and getattr(cur, "errno", None) in (errno.EDQUOT, errno.ENOSPC):
            return True
        cur = cur.__cause__ or cur.__context__
    tb_lower = tb_str.lower()
    return (
        "disk quota exceeded" in tb_lower
        or "no space left on device" in tb_lower
        or "[errno 122]" in tb_lower
        or "[errno 28]" in tb_lower
    )


def _batch_worker(task):
    """Top-level worker function for grid-level multiprocessing in batch mode."""
    grid_id, raw_stem, input_dir, output_dir, overwrite = task

    # Re-seed RNGs from OS entropy so forked workers never share identical stochastic sequences
    random.seed()
    np.random.seed()

    expected_out = os.path.join(output_dir, f"{raw_stem}{config.OUTPUT_SUFFIX}.h5")
    if (not overwrite) and os.path.isfile(expected_out):
        return ("skipped", grid_id, raw_stem, 0.0, "")

    try:
        _, elapsed = run_single_grid(
            inputfile_id=grid_id,
            n_cpu=1,
            input_dir=input_dir,
            output_dir=output_dir,
            grid_filename=f"{raw_stem}.h5",
            verbose=False,
        )
        return ("ok", grid_id, raw_stem, elapsed, "")
    except Exception as exc:
        # Remove any incomplete working copy so re-runs start cleanly (never touch *_DmdAllocated.h5)
        partial_path = os.path.join(output_dir, f"{raw_stem}.h5")
        if os.path.isfile(partial_path):
            try:
                os.remove(partial_path)
            except OSError:
                pass
        tb_str = traceback.format_exc()
        status = "fatal" if _is_storage_full_error(exc, tb_str) else "failed"
        return (status, grid_id, raw_stem, 0.0, tb_str)
    finally:
        # Reclaim pandas/pandapower objects promptly across long multi-hundred-grid worker runs
        gc.collect()


def run_batch_manifest(manifest_path, start_line, end_line, n_workers, input_dir, output_dir, overwrite=False):
    """Process a slice of a tab-separated manifest (`GRID_ID<TAB>RAW_STEM`) using a worker pool."""
    input_dir = input_dir if input_dir is not None else config.DATA_GRID_DIR
    output_dir = output_dir if output_dir is not None else config.STORAGE_DIR
    os.makedirs(output_dir, exist_ok=True)

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            if start_line is not None and line_no < start_line:
                continue
            if end_line is not None and line_no > end_line:
                break
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            grid_id = parts[0]
            raw_stem = parts[1] if len(parts) > 1 else parts[0]
            entries.append((grid_id, raw_stem, input_dir, output_dir, overwrite))

    total = len(entries)
    if total == 0:
        print(f"No manifest entries in slice [{start_line}, {end_line}] of '{manifest_path}'.")
        return 0

    n_workers = max(1, min(int(n_workers), total))
    t0 = time.perf_counter()
    print(
        f"Batch Step 1: processing {total} grid(s) with {n_workers} parallel worker(s) "
        f"(lines {start_line}..{end_line})",
        flush=True,
    )

    succeeded = 0
    skipped = 0
    failed_ids = []
    fatal_abort = False
    cancelled = 0

    if n_workers > 1:
        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            futures = [executor.submit(_batch_worker, task) for task in entries]
            for done_idx, fut in enumerate(as_completed(futures), start=1):
                status, grid_id, raw_stem, elapsed, err_tb = fut.result()
                if status == "ok":
                    succeeded += 1
                    print(f"  [{done_idx}/{total}] OK      {raw_stem} ({elapsed:.1f}s)", flush=True)
                elif status == "skipped":
                    skipped += 1
                    print(f"  [{done_idx}/{total}] SKIPPED {raw_stem}", flush=True)
                elif status == "fatal":
                    failed_ids.append(grid_id)
                    fatal_abort = True
                    print(f"  [{done_idx}/{total}] FATAL   {raw_stem} (ID {grid_id})", file=sys.stderr)
                    print(err_tb, file=sys.stderr, flush=True)
                    cancelled = sum(1 for f in futures if f.cancel())
                    print(
                        f"FATAL: Storage limit reached (EDQUOT/ENOSPC). Aborting batch early "
                        f"(cancelled {cancelled} queued grid(s); waiting for active workers to clean up).",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                else:
                    failed_ids.append(grid_id)
                    print(f"  [{done_idx}/{total}] FAILED  {raw_stem} (ID {grid_id})", file=sys.stderr)
                    print(err_tb, file=sys.stderr, flush=True)
        finally:
            executor.shutdown(wait=True, cancel_futures=fatal_abort)
    else:
        for done_idx, task in enumerate(entries, start=1):
            status, grid_id, raw_stem, elapsed, err_tb = _batch_worker(task)
            if status == "ok":
                succeeded += 1
                print(f"  [{done_idx}/{total}] OK      {raw_stem} ({elapsed:.1f}s)", flush=True)
            elif status == "skipped":
                skipped += 1
                print(f"  [{done_idx}/{total}] SKIPPED {raw_stem}", flush=True)
            elif status == "fatal":
                failed_ids.append(grid_id)
                fatal_abort = True
                cancelled = total - done_idx
                print(f"  [{done_idx}/{total}] FATAL   {raw_stem} (ID {grid_id})", file=sys.stderr)
                print(err_tb, file=sys.stderr, flush=True)
                print(
                    f"FATAL: Storage limit reached (EDQUOT/ENOSPC). Aborting batch early "
                    f"(skipped {cancelled} remaining grid(s)).",
                    file=sys.stderr,
                    flush=True,
                )
                break
            else:
                failed_ids.append(grid_id)
                print(f"  [{done_idx}/{total}] FAILED  {raw_stem} (ID {grid_id})", file=sys.stderr)
                print(err_tb, file=sys.stderr, flush=True)

    wall = time.perf_counter() - t0
    cancel_note = f", cancelled={cancelled}" if fatal_abort else ""
    print(
        f"Batch summary: succeeded={succeeded}, skipped={skipped}, failed={len(failed_ids)}{cancel_note} "
        f"| wall={wall:.1f}s ({wall / max(1, succeeded):.2f}s/grid)",
        flush=True,
    )
    if fatal_abort:
        print(f"Aborted on Grid IDs: {' '.join(failed_ids)}", file=sys.stderr, flush=True)
        return 2
    if failed_ids:
        print(f"Failed Grid IDs: {' '.join(failed_ids)}", file=sys.stderr, flush=True)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Inference Step 1 orchestrator: demand allocation, MV disconnection, and pre-expansion power flow."
    )
    parser.add_argument(
        "inputfile_id",
        nargs="?",
        default=None,
        help="Single grid ID prefix (e.g., 51431 or 0298). Omit when using --manifest.",
    )
    parser.add_argument(
        "--n_cpu", type=int, default=1, help="Number of CPUs available for parallel execution"
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Optional override for config.DATA_GRID_DIR (raw input grids)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional override for config.STORAGE_DIR (output directory)",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional exact raw .h5 filename (skips scanning input_dir by ID)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Batch mode: path to tab-separated manifest file (`GRID_ID<TAB>RAW_STEM`)",
    )
    parser.add_argument(
        "--start_line", type=int, default=None, help="Batch mode: 1-indexed inclusive start line"
    )
    parser.add_argument(
        "--end_line", type=int, default=None, help="Batch mode: 1-indexed inclusive end line"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Batch mode: re-run grids even if *_DmdAllocated.h5 already exists",
    )
    args = parser.parse_args()

    if args.manifest:
        exit_code = run_batch_manifest(
            manifest_path=args.manifest,
            start_line=args.start_line,
            end_line=args.end_line,
            n_workers=args.n_cpu,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        sys.exit(exit_code)

    if not args.inputfile_id:
        parser.error("Provide either `inputfile_id` or `--manifest`.")

    run_single_grid(
        inputfile_id=args.inputfile_id,
        n_cpu=args.n_cpu,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        grid_filename=args.filename,
        verbose=True,
    )


if __name__ == "__main__":
    main()
