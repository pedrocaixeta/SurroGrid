"""Run pre-expansion time-series power flow on a demand-allocated HDF5 grid file.

Reads `/urbs_in/demand` and `/raw_data/net` (already updated by
`disconnect_MV_buildings`) from the working `.h5` file, runs the 8760-hour
pandapower BFSW load flow, and writes:
- `/pwrflw/input/demand_pre`
- `/pwrflw/output/pre/demand_import`
- `/pwrflw/output/pre/vm`
- `/pwrflw/output/pre/line_loads`
"""

import argparse
import warnings
import h5py
import pandas as pd
import pandapower as pp
from pandas.errors import PerformanceWarning

from pre_powerflow.src import demands as dmnds
from pre_powerflow.src import powerflow as pwrflw
from pre_powerflow.src.resource_report import resource_report


def _save_df(h5_path, df, key):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        with pd.HDFStore(str(h5_path), mode="a", complib="blosc", complevel=9) as store:
            store.put(key, df)


def run_pre_powerflow_on_file(file_path, n_cpu=1):
    """Run pre-expansion power flow in-place on the working HDF5 file `file_path`."""
    print(f"Running pre-expansion power flow on {file_path} with {n_cpu} CPU(s)...")

    # 1. Build and save pre-expansion active + reactive bus demand (kW / kVAr)
    df_raw_demand = pd.read_hdf(str(file_path), key="/urbs_in/demand")
    df_pre_demand = dmnds.obtain_pre_demand(df_raw_demand)
    _save_df(file_path, df_pre_demand, "/pwrflw/input/demand_pre")

    # 2. Load pandapower network (disconnected MV loads already have in_service=False)
    with h5py.File(str(file_path), "r") as f:
        json_data = f["raw_data/net"][()]
        if isinstance(json_data, bytes):
            json_data = json_data.decode("utf-8")
        grid = pp.from_json_string(json_data)

    grid = pwrflw.prepare_grid(grid)

    # 3. Execute pre-expansion power flow
    with resource_report(name="Pre-Expansion Powerflow Run", include_children=True):
        ext_import_pre, vm_pre, line_loads_pre = pwrflw.pf(
            grid, df_pre_demand, parallel=(int(n_cpu) > 1), n_cpu=int(n_cpu)
        )
        _save_df(file_path, ext_import_pre, "/pwrflw/output/pre/demand_import")
        _save_df(file_path, vm_pre, "/pwrflw/output/pre/vm")
        _save_df(file_path, line_loads_pre, "/pwrflw/output/pre/line_loads")

    return ext_import_pre


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pre-expansion power flow on a working .h5 file.")
    parser.add_argument("file_path", help="Path to the working .h5 file")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of CPUs for parallel power flow")
    args = parser.parse_args()

    run_pre_powerflow_on_file(args.file_path, n_cpu=args.n_cpu)
    print("Pre-expansion power flow done!")
