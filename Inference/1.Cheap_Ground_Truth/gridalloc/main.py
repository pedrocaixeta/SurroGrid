#!/usr/bin/env python3
"""Demand allocation step (baseline without-mobility) for a single grid HDF5 file.

Can be invoked via `run_demand_allocation(...)` from the top-level orchestrator
`Inference/2.demand_allocation/main.py` or executed directly as a standalone script.
"""

import argparse
import os
import time

import src.classes.grid as grd
from src.classes.save_grid import find_grid_file_by_id
from src.classes.resource_report import resource_report
from config import config


def run_demand_allocation(inputfile_id, n_cpu=1, input_dir=None, output_dir=None, grid_filename=None):
    """Run solar, electricity, space/water heat, HP COP, and car-count sampling.

    Writes the copied working .h5 file into `output_dir` (default `config.STORAGE_DIR`)
    without appending the final `_DmdAllocated` suffix yet, and returns its path.
    Passing `grid_filename` directly skips scanning `input_dir` with `os.listdir`
    (much faster on network filesystems containing 85k+ grids).
    """
    input_dir = input_dir if input_dir is not None else config.DATA_GRID_DIR
    output_dir = output_dir if output_dir is not None else config.STORAGE_DIR

    if grid_filename is not None:
        inputfile = grid_filename
    else:
        inputfile = find_grid_file_by_id(input_dir, inputfile_id)

    settings = {
        "grid_filename": inputfile,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "weather_data_exists": True,
        "parallel": (int(n_cpu) > 1),
        "n_cpu": int(n_cpu),
    }

    print(f"Running demand allocation on {inputfile} (ID {inputfile_id}) with {settings['n_cpu']} CPU(s)!")

    # Setup grid and generate demands
    # Order of operations is important: Weather -> Solar -> Electricity -> Heat -> Car Statistics
    GRD = grd.Grid(settings)
    GRD.retrieve_weather()

    with resource_report(include_children=True, name="Solar Generation"):
        GRD.generate_solar()
    with resource_report(include_children=True, name="Electricity Generation"):
        GRD.generate_electricity()
    with resource_report(include_children=True, name="Heat + HP COP Generation"):
        GRD.generate_heat()
    with resource_report(include_children=True, name="Car Ownership Sampling"):
        GRD.generate_mobility()

    # Build per-building urbs_in tables needed by downstream feature extraction
    GRD.create_weather_urbs()
    GRD.create_supim()
    GRD.create_demand()
    GRD.create_tve()
    GRD.create_processes()

    # Copy raw .h5 to STORAGE_DIR and save per-building tables
    GRD.save_grid_data()

    return GRD.SF.output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low voltage grid baseline demand allocation.")
    parser.add_argument("inputfile_id", help="Input grid ID prefix (e.g., 51431 or 0298)")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of CPUs for parallel generation")
    parser.add_argument("--input_dir", default=None, help="Optional override for config.DATA_GRID_DIR")
    parser.add_argument("--output_dir", default=None, help="Optional override for config.STORAGE_DIR")
    args = parser.parse_args()

    with resource_report(include_children=True, name="Demand Allocation"):
        out_path = run_demand_allocation(
            args.inputfile_id,
            n_cpu=args.n_cpu,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
        print(f"Saved working demand-allocated file to: {out_path}")
