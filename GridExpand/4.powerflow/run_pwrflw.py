"""Run time-series power-flow for a single scenario file.

This is the entrypoint for GridExpand step 4 (powerflow).

It selects one input `.h5` file from `Input/` based on the provided `inputfile_id`:
the script matches the prefix before the first underscore, e.g. `0_... .h5`.

The selected file is copied to `Output/` and augmented with:

- `/pwrflw/input/*` demand tables (pre/post expansion)
- `/pwrflw/output/pre/*` and `/pwrflw/output/post/*` power-flow results

See `README.md` in this folder for required HDF5 keys and expected outputs.
"""

import src.save_grid as svgrd
import src.demands as dmnds
import src.powerflow as pwrflw

import argparse
import os
from src.resource_report import resource_report
from config import config

if __name__ == "__main__":
    import time
    t_start = time.perf_counter()
    ot_start = os.times()
    t_start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    ##### Read args + Obtain relevant input_files #####:
    parser = argparse.ArgumentParser(description="Low voltage grid DER allocation.")
    parser.add_argument("inputfile_id", help="Input file name (no path)")
    parser.add_argument("--n_cpu", default=1, help="Number of CPUs available for parallel generation")
    args = parser.parse_args()

    # list all .h5 files in your directory
    all_entries = os.listdir(config.DATA_DIR)
    h5_files = [fname for fname in all_entries if fname.endswith(".h5")]
    # find file with correct id prefix
    input_id_str = str(args.inputfile_id)
    matched_files = [fname for fname in h5_files if fname.split('_', 1)[0] == input_id_str]
    input_file = matched_files[0]


    ##### Input Settings + Setup #####
    settings = {
        "file": input_file,
        "parallel": True,
        "n_cpu": int(args.n_cpu)
    }
    print(f"Running input file {settings['file']} (ID {args.inputfile_id}) with {settings['n_cpu']} CPUs!")

    # Save file handler
    SF = svgrd.SaveFile(settings["file"])


    ##### Obtaining Power Demands #####
    # Read-out and preprocess demand before and after DER expansion
    df_pre_demand, df_post_demand = dmnds.obtain_demand(SF)

    # Save to be retrieved later by ML model
    SF.save_df(df_pre_demand, "/pwrflw/input/demand_pre")
    SF.save_df(df_post_demand, "/pwrflw/input/demand_post")


    ##### Powerflow #####
    # Readout grid from file
    grid = SF.get_input_grid()
    # Remove any load restrictions and replace transformer with switch
    grid = pwrflw.prepare_grid(grid)

    # Run powerflow pre DER expansion
    with resource_report(name="Pre-Expansion Powerflow Run", include_children=True):
        ext_import_pre, vm_pre, line_loads_pre = pwrflw.pf(grid, df_pre_demand, settings["parallel"], settings["n_cpu"])
        # Save results
        SF.save_df(ext_import_pre, "/pwrflw/output/pre/demand_import")
        SF.save_df(vm_pre, "/pwrflw/output/pre/vm")
        SF.save_df(line_loads_pre, "/pwrflw/output/pre/line_loads")

    # Run powerflow post DER expansion
    with resource_report(name="Post-Expansion Powerflow Run", include_children=True):
        ext_import_post, vm_post, line_loads_post = pwrflw.pf(grid, df_post_demand, settings["parallel"], settings["n_cpu"])
        ##### Save results #####
        SF.save_df(ext_import_post, "/pwrflw/output/post/demand_import")
        SF.save_df(vm_post, "/pwrflw/output/post/vm")
        SF.save_df(line_loads_post, "/pwrflw/output/post/line_loads")

    # Rename output file to append '_pwrflw' suffix
    file_name, file_extension = os.path.splitext(SF.output_path)
    new_output_path = f"{file_name}_pwrflw{file_extension}"
    if os.path.exists(new_output_path):
        os.remove(new_output_path)
    os.rename(SF.output_path, new_output_path)
    print(f"Renamed output file to: {os.path.basename(new_output_path)}")

    # Collect performance metrics (optional, should not block execution)
    try:
        t_end = time.perf_counter()
        ot_end = os.times()
        t_end_str = time.strftime("%Y-%m-%d %H:%M:%S")
        wall_sec = max(0.0, t_end - t_start)
        
        user_sec = max(0.0, ot_end[0] - ot_start[0])
        system_sec = max(0.0, ot_end[1] - ot_start[1])
        user_sec += max(0.0, ot_end[2] - ot_start[2])
        system_sec += max(0.0, ot_end[3] - ot_start[3])
        cpu_sec = user_sec + system_sec
        
        try:
            import resource
            import platform
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_rss = int(ru) * 1024 if platform.system() == "Linux" else int(ru)
        except Exception:
            peak_rss = 0
            
        cpu_model = "Unknown"
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            import platform
            cpu_model = platform.processor() or "Unknown"
            
        import socket
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")
        slurm_partition = os.environ.get("SLURM_JOB_PARTITION", "N/A")
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_ON_NODE", "1"))
        slurm_mem = os.environ.get("SLURM_MEM_PER_CPU", "N/A")
        slurm_cluster = os.environ.get("SLURM_CLUSTER_NAME", "N/A")
        hostname = socket.gethostname()
        
        new_entry = {
            "step": "powerflow",
            "timestamp_start": t_start_str,
            "timestamp_end": t_end_str,
            "wall_seconds": wall_sec,
            "cpu_seconds": cpu_sec,
            "peak_rss_bytes": peak_rss,
            "cpu_model": cpu_model,
            "hostname": hostname,
            "slurm_job_id": slurm_job_id,
            "slurm_partition": slurm_partition,
            "slurm_cpus_allocated": slurm_cpus,
            "slurm_mem_per_cpu": slurm_mem,
            "slurm_cluster": slurm_cluster
        }
        
        import pandas as pd
        df_new = pd.DataFrame([new_entry])
        
        try:
            df_existing = pd.read_hdf(new_output_path, key="performance")
            df_existing = df_existing[df_existing["step"] != "powerflow"]
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
            
        with pd.HDFStore(new_output_path, mode="a") as store:
            store.put("performance", df_all, format="fixed")
    except Exception as e:
        print(f"Warning: Failed to write performance metadata: {e}")

    print("Done!")
    os._exit(0)