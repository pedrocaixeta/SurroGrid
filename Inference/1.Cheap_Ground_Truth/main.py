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
import os
import platform
import socket
import sys
import time
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


def main():
    t_start = time.perf_counter()
    ot_start = os.times()
    t_start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Inference Step 1 orchestrator: demand allocation, MV disconnection, and pre-expansion power flow."
    )
    parser.add_argument("inputfile_id", help="Grid ID prefix (e.g., 51431 or 0298)")
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
    args = parser.parse_args()

    with resource_report(include_children=True, name="Global Demand Allocation Pipeline"):
        # Step 1: Allocate demands (solar, electricity, heat, HP COP, car counts)
        with resource_report(include_children=True, name="Step 1: Demand Allocation"):
            working_path = run_demand_allocation(
                args.inputfile_id,
                n_cpu=args.n_cpu,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
            )

        # Step 2: Disconnect MV-scale buildings (>100 kW non-res, >250 kW res/mixed)
        with resource_report(include_children=True, name="Step 2: MV Building Disconnection"):
            mv_stats = disconnect_mv_buildings(Path(working_path))
            if mv_stats is None:
                raise RuntimeError(f"MV building disconnection failed for '{working_path}'.")
            print(
                f"MV disconnection summary: "
                f"{mv_stats['matching_found']}/{mv_stats['total_buildings']} building(s) disconnected "
                f"({mv_stats['remaining_percent']:.2f}% remaining active)."
            )

        # Step 3: Compute pre-expansion power flow (demand_net_active_pre for surrogate)
        with resource_report(include_children=True, name="Step 3: Pre-Expansion Powerflow"):
            run_pre_powerflow_on_file(working_path, n_cpu=args.n_cpu)

        # Step 4: Log performance metadata and append completion suffix
        _record_performance(working_path, "demand_allocation", t_start, ot_start, t_start_str)
        final_path = SaveFile.append_suffix(working_path, config.OUTPUT_SUFFIX)
        print(f"All steps finished successfully! Saved: {final_path}")


if __name__ == "__main__":
    main()
