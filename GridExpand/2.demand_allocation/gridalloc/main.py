#!/usr/bin/env python3
import argparse
import os
import time

import src.classes.grid as grd
from src.classes.resource_report import resource_report
from config import config


if __name__ == '__main__':
    t_start = time.perf_counter()
    ot_start = os.times()
    t_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    with resource_report(include_children=True, name="Main Script") as rr_main:
        ####### Input arguments: #######
        parser = argparse.ArgumentParser(description="Low voltage grid DER allocation.")
        parser.add_argument("inputfile_id", help="Input file name (no path)")
        parser.add_argument("--n_cpu", default=1, help="Number of CPUs available for parallel generation")
        args = parser.parse_args()

        #### Obtain relevant input file ####
        # list all .h5 files in your directory
        all_entries = os.listdir(config.DATA_GRID_DIR)
        h5_files = [fname for fname in all_entries if fname.endswith(".h5")]
        # find file with correct id prefix
        input_id_str = str(args.inputfile_id)
        matched_files = [fname for fname in h5_files if fname.split('_', 1)[0] == input_id_str]
        inputfile = matched_files[0]

        ####### Run Settings: #######
        settings = {
            # "grid_filename": "N2775500E4431500_86154_1_-6.h5"
            "grid_filename": inputfile,         # Name of input file
            "weather_data_exists": True,        # Is weather data already included in input grid file's raw data? (recommended, as on HPC cluster no outside API access)
            "parallel": (int(args.n_cpu) > 1),  # Parallelized run?
            "n_cpu": int(args.n_cpu)            # cpus if parallel 
        }                    

        print(f"Running input file {inputfile} (ID {args.inputfile_id}) with {settings['n_cpu']} CPUs!")
        #----------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------#

        ### Setup grid which stores all relevant data for assigning demands
        GRD = grd.Grid(settings)
        # GRD.df_buildings = GRD.df_buildings.iloc[0:5].reset_index(drop=True)

        ### Data and Demand Generation
        # Order of operations is important: Weather -> Solar -> Electricity -> Heat -> Mobility
        GRD.retrieve_weather()          # Weather

        with resource_report(include_children=True, name="Solar Generation") as rr:
            GRD.generate_solar()        # Solar data
        with resource_report(include_children=True, name="Electricity Generation") as rr:
            GRD.generate_electricity()  # Electricity
        with resource_report(include_children=True, name="Heat Generation") as rr:
            GRD.generate_heat()         # Heat
        with resource_report(include_children=True, name="Mobility Generation") as rr:
            GRD.generate_mobility()     # Mobility

        ### Conversion of generated data to urbs outputs
        GRD.create_weather_urbs()       # Weather
        GRD.create_supim()              # SupIm
        GRD.create_demand()             # Demands
        GRD.create_tve()                # Eff Factor
        GRD.create_bsp()                # Buy-Sell-Price
        GRD.create_processes()          # Process
        GRD.create_commodities()        # Commoditites
        GRD.create_process_commodity()  # Process Commodity
        GRD.create_storages()           # Storage

        ### Saving Grid Data to .h5
        df = GRD.save_grid_data()

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
                "step": "demand_allocation",
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
            
            output_path = GRD.SF.output_path
            try:
                df_existing = pd.read_hdf(output_path, key="performance")
                df_existing = df_existing[df_existing["step"] != "demand_allocation"]
                df_all = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df_all = df_new
                
            with pd.HDFStore(output_path, mode="a") as store:
                store.put("performance", df_all, format="fixed")
        except Exception as e:
            print(f"Warning: Failed to write performance metadata: {e}")