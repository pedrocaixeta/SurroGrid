# -*- coding: utf-8 -*-
import urbs
import os
import shutil
import time
import argparse
from urbs.resource_report import resource_report

# Note - this urbs version is deviating in the following ways from urbs-lvds (04 Feb 2025):
# :: removed grid optimization, 14a/bui-react, uhp, coordination, curtailment
# :: keeping tsam, flexibility, power_price, different shares of electrification, vartariff
# :: removed microgrid inputs
# :: removed LP file and excel generation
# :: removed CO2 limit/environmental commodities
# :: removed inputs from Global Excel sheet
# :: removed multiple input scenarios
# :: removed several support_timeframes
# :: removed reactive power support


if __name__ == '__main__':
    t_start = time.perf_counter()
    ot_start = os.times()
    t_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    with resource_report(include_children=True, name="Urbs Script") as rr_main:
        ### Read args:
        parser = argparse.ArgumentParser(description="Low voltage grid DER allocation.")
        parser.add_argument("inputfile_id", help="Input file name (no path)")
        parser.add_argument("--n_cpu", default=1, help="Number of CPUs available for parallel generation")
        args = parser.parse_args()

        ### Obtain relevant input_files
        # list all .h5 files in your directory
        input_dir = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/2nd_RUN/2.Demand_Allocated/"
        all_entries = os.listdir(input_dir)
        h5_files = [fname for fname in all_entries if fname.endswith(".h5")]
        # find file with correct id prefix
        input_id_str = str(args.inputfile_id)
        matched_files = [fname for fname in h5_files if fname.split('_', 1)[0] == input_id_str]
        input_file = matched_files[0]


        ### Give global run settings
        global_settings = {
            "input_file": input_file,
            # "input_file": 'N2775500E4431500_86154_1_-6.h5',    # input file name in dir "Input" 
            # "input_file": 'N2827500E4503500_93426_5_41.h5',    # input file name in dir "Input"
            "tsam": False,                           # apply time series aggregation ("True", "False")
            "noTypicalPeriods": 6,                  # tsam: number of aggregated typical periods (int, max 52) 
            "hoursPerPeriod": 168,                  # tsam: length of typical period (int)

            # Electrification
            "PV_electr": 100,       # 100           # % of building nodes adopting PV (0-100)
            "HP_electr": 100,       # 100           # % of building nodes adopting HP (0-100)
            "EV_electr": 100,       # 100           # % of building nodes adopting EV (0-100)

            # CPUs
            "n_cpu": int(args.n_cpu)
        }

        print("Following global settings are applied:")
        for key, value in global_settings.items():
            print(f"{key:<16} {value:>1}")
        print("\n")


        ### Input and result handling
        # Extract input path
        input_file = global_settings['input_file']
        input_path = os.path.join(input_dir, input_file)

        # Create result directory (format: datetime-inputfile-resultname), copy input and runfile into it 
        result_dir = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/2nd_RUN/3.Post_Urbs/"
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, input_file) 
        shutil.copyfile(input_path, result_path)


        ### Run defined scenario through pyomo model setup and solver
        urbs.run_lvds_opt(input_path,      # path to input files
                        result_path,     # path to store results
                        result_dir,
                        global_settings) # all input settings  

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
                "step": "urbs",
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
            
            # Calculate renamed path
            scenario_name = urbs.read_scenario_name(global_settings)
            file_name, file_extension = os.path.splitext(result_path)
            new_result_path = f"{file_name}_{scenario_name}{file_extension}"
            
            try:
                df_existing = pd.read_hdf(new_result_path, key="performance")
                df_existing = df_existing[df_existing["step"] != "urbs"]
                df_all = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df_all = df_new
                
            with pd.HDFStore(new_result_path, mode="a") as store:
                store.put("performance", df_all, format="fixed")
        except Exception as e:
            print(f"Warning: Failed to write performance metadata: {e}")