#!/usr/bin/env python3
"""Single-grid space-heating simulation for p_max tuning.

Processes one grid `.h5` file (weather -> electrical internal gains -> TABULA space heat)
using the specified `(p_max_res, p_max_ghd)` refurbishment parameters, writes the resulting
`.h5` file into `output_dir`, and returns per-grid residential and GHD floor-area and
space-heat sums for fast aggregation.
"""

import argparse
import numpy as np
import pandas as pd

import src.classes.grid as grd
from src.classes.save_grid import find_grid_file_by_id
from config import config


def run_single_grid_heat(
    grid_filename,
    input_dir=None,
    output_dir=None,
    p_max_res=None,
    p_max_ghd=None,
    n_cpu=1,
    seed=None,
):
    """Simulate space-heating demand for one grid file and save the result to `output_dir`.

    Returns a dictionary containing the output path and summed floor areas / space-heating
    demands for purely residential and purely GHD buildings in this grid.
    """
    input_dir = input_dir if input_dir is not None else config.DATA_GRID_DIR
    output_dir = output_dir if output_dir is not None else config.STORAGE_DIR
    p_max_res = p_max_res if p_max_res is not None else config.RETROFIT_P_MAX_RES
    p_max_ghd = p_max_ghd if p_max_ghd is not None else config.RETROFIT_P_MAX_GHD

    if seed is not None:
        np.random.seed(int(seed))

    settings = {
        "grid_filename": grid_filename,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "weather_data_exists": True,
        "parallel": (int(n_cpu) > 1),
        "n_cpu": int(n_cpu),
    }

    GRD = grd.Grid(settings)
    GRD.retrieve_weather()
    GRD.generate_electricity()
    GRD.generate_heat(p_max_res=p_max_res, p_max_ghd=p_max_ghd)
    GRD.create_demand()
    output_path = GRD.save_grid_data()

    # Aggregate purely residential and purely GHD space heat and floor areas for this grid
    dfb = GRD.df_buildings
    res_area = pd.to_numeric(dfb.get("residential_floor_area", 0.0), errors="coerce").fillna(0.0)
    nonres_area = pd.to_numeric(dfb.get("nonresidential_floor_area", 0.0), errors="coerce").fillna(0.0)
    sh_kwh = pd.to_numeric(dfb.get("space_heat_kwh", 0.0), errors="coerce").fillna(0.0)
    retrofit = pd.to_numeric(dfb.get("retrofit", 0), errors="coerce").fillna(0).astype(int)

    pure_res_mask = (nonres_area == 0.0) & (res_area > 0.0)
    pure_ghd_mask = (res_area == 0.0) & (nonres_area > 0.0)

    return {
        "grid_filename": grid_filename,
        "output_path": output_path,
        "n_pure_res": int(pure_res_mask.sum()),
        "n_pure_res_renov": int((retrofit[pure_res_mask] > 0).sum()),
        "res_floor_area_m2": float(res_area[pure_res_mask].sum()),
        "res_space_heat_kwh": float(sh_kwh[pure_res_mask].sum()),
        "n_pure_ghd": int(pure_ghd_mask.sum()),
        "n_pure_ghd_renov": int((retrofit[pure_ghd_mask] > 0).sum()),
        "ghd_floor_area_m2": float(nonres_area[pure_ghd_mask].sum()),
        "ghd_space_heat_kwh": float(sh_kwh[pure_ghd_mask].sum()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-grid space-heat simulation for p_max tuning.")
    parser.add_argument("inputfile_id", help="Input grid ID prefix or filename")
    parser.add_argument("--n_cpu", type=int, default=1, help="CPUs for intra-grid parallelism")
    parser.add_argument("--input_dir", default=None, help="Override config.DATA_GRID_DIR")
    parser.add_argument("--output_dir", default=None, help="Override config.STORAGE_DIR")
    parser.add_argument("--p_max_res", type=float, default=None, help="Residential p_max")
    parser.add_argument("--p_max_ghd", type=float, default=None, help="GHD p_max")
    args = parser.parse_args()

    in_dir = args.input_dir if args.input_dir is not None else config.DATA_GRID_DIR
    fname = (
        args.inputfile_id
        if args.inputfile_id.endswith(".h5")
        else find_grid_file_by_id(in_dir, args.inputfile_id)
    )
    stats = run_single_grid_heat(
        fname,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        p_max_res=args.p_max_res,
        p_max_ghd=args.p_max_ghd,
        n_cpu=args.n_cpu,
    )
    print(f"Completed {fname} -> {stats['output_path']}")
