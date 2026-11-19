"""
This script extracts peak power/demand data from Pylovo grid .h5 files,
pre-URBS inputs, and post-URBS optimization results, and generates
separate aggregated peak demand reports for Pedro and Elias.
"""

import io
import json
import multiprocessing
import os
from pathlib import Path
import h5py
import pandas as pd

# Disable HDF5 file locking to avoid conflicts when reading files over network/WSL mounts
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================

INPUT_FOLDERS = [
    "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/result3.urbs__input4/",
    "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/EliasH/PostUrbs/"
]
OUTPUT_FOLDER = "/dss/dsshome1/05/go49cer2/SurroGrid/GridExpand/4.powerflow/Output/"

# Hardcoded lists of overloaded (affected) PLZs and Grid Indices.
# These replace the dynamic reading of 'buildings_overloadedindexes_PostUrbs.xlsx'.
AFFECTED_PLZ = {
    "63739", "63741", "63743", "63840", "63843", "80539", "80995", 
    "81829", "81925", "81927", "82024", "82481", "83080", "83088", 
    "83104", "83308", "83435", "83512", "84085", "84359", "84555", 
    "85053", "85054", "85084", "85399", "86163", "86165", "86399", 
    "86672", "91074", "91094", "92224", "92280", "92533", "93158", 
    "93449", "94315", "94362", "95213", "95643", "96049", "96050", 
    "96264", "97422", "97737"
}

AFFECTED_INDEXES = {
    21, 74, 119, 146, 181, 199, 214, 219, 244, 246, 247, 303, 326, 340, 
    407, 414, 417, 421, 423, 434, 440, 441, 476, 510, 566, 567, 579, 
    587, 590, 633, 637, 641, 673, 674, 705, 706, 714, 715, 726, 731, 
    743, 745, 748, 780, 784, 798, 819, 820, 821, 881, 890, 908, 929, 
    953, 972, 973, 977, 980, 992, 1036, 1048, 1052, 1063, 1069, 1071, 
    1085, 1114, 1120, 1145, 1170, 1173, 1177, 1194, 1214, 1232, 1239, 
    1260, 1311, 1322, 1339, 1350, 1359, 1426, 1461
}

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def extract_plz_from_filename(filename):
    """
    Extracts the PLZ (postal code) from the grid filename.
    Example: '1036_N3042500E4198500_63840_1_3_...' -> '63840'
    Returns PLZ as a string.
    """
    try:
        parts = filename.split('_')
        if len(parts) >= 3:
            return parts[2]
    except Exception:
        pass
    return None


def extract_index_from_filename(filename):
    """
    Extracts the grid index from the grid filename.
    Example: '1036_N3042500E4198500_63840_1_3_...' -> 1036
    Returns grid index as an integer.
    """
    try:
        parts = filename.split('_')
        if len(parts) >= 1:
            return int(parts[0])
    except Exception:
        pass
    return None


def process_single_h5(args):
    """
    Processes a single H5 file to extract:
    1. Pylovo maximum load peak (from raw_data/net JSON).
    2. Pre-URBS maximum electricity demand peak.
    3. Post-URBS maximum import peak (net load).
    """
    file_idx, total_files, h5_file, datasource_idx = args
    print(f"[{file_idx+1}/{total_files}] Processing {h5_file.name}...")
    
    grid_idx = extract_index_from_filename(h5_file.name)
    plz = extract_plz_from_filename(h5_file.name)
    
    if grid_idx is None:
        return None
        
    num_buildings = None
    num_lines = None
    pylovo_grid_peak = None
    pre_urbs_grid_peak = None
    post_urbs_grid_peak = None
    post_urbs_ts_1 = None
    post_urbs_peak_2 = None
    post_urbs_ts_2 = None
    post_urbs_peak_3 = None
    post_urbs_ts_3 = None
    
    try:
        with h5py.File(str(h5_file), 'r') as file:
            # 0. Count Buildings
            try:
                if 'raw_data/buildings' in file:
                    df_buildings = pd.read_hdf(str(h5_file), key='/raw_data/buildings')
                    num_buildings = len(df_buildings)
            except Exception:
                pass

            # 1. Pylovo Peak Load and Line Count (extracting from pandapower net json format)
            try:
                if 'raw_data/net' in file:
                    net_json = file['raw_data/net'][()]
                    data = json.loads(net_json)
                    if '_object' in data:
                        if 'load' in data['_object']:
                            load_data = data['_object']['load']
                            df_load = pd.read_json(io.StringIO(load_data['_object']), orient=load_data['orient'])
                            if 'p_mw' in df_load.columns:
                                pylovo_grid_peak = df_load['p_mw'].sum() * 1000.0  # Convert MW to kW
                        if 'line' in data['_object']:
                            line_data = data['_object']['line']
                            df_line = pd.read_json(io.StringIO(line_data['_object']), orient=line_data['orient'])
                            num_lines = len(df_line)
            except Exception:
                pass

            # 2. Pre-URBS Peak Demand
            try:
                df_demand = pd.read_hdf(str(h5_file), key='/urbs_in/demand')
                elec_cols = [col for col in df_demand.columns if col[1] == 'electricity'] #retrieves the columns were the second element of the tuple is 'electricity'
                if elec_cols:
                    df_elec = df_demand[elec_cols]
                    total_ts = df_elec.sum(axis=1) #Summing with axis=1 means summing across the columns for each individual row
                    pre_urbs_grid_peak = total_ts.max()
            except Exception:
                pass

            # 3. Post-URBS Peak Demand (net grid import minus feed-in)
            try:
                df_tau = pd.read_hdf(str(h5_file), key='/urbs_out/MILP/tau_pro')
                mask = df_tau.index.get_level_values('pro').isin(['import', 'feed_in'])
                df_filtered = df_tau[mask]
                if not df_filtered.empty:
                    df_unstacked = df_filtered.unstack(level='pro')
                    import_series = df_unstacked['import'].fillna(0) if 'import' in df_unstacked.columns else pd.Series(0.0, index=df_unstacked.index)
                    feed_in_series = df_unstacked['feed_in'].fillna(0) if 'feed_in' in df_unstacked.columns else pd.Series(0.0, index=df_unstacked.index)
                    net_load = import_series - feed_in_series
                    total_ts = net_load.groupby(level='t').sum()
                    if not total_ts.empty:
                        sorted_ts = total_ts.reindex(total_ts.abs().sort_values(ascending=False).index)
                        if len(sorted_ts) >= 1:
                            post_urbs_grid_peak = sorted_ts.iloc[0]
                            post_urbs_ts_1 = sorted_ts.index[0]
                        if len(sorted_ts) >= 2:
                            post_urbs_peak_2 = sorted_ts.iloc[1]
                            post_urbs_ts_2 = sorted_ts.index[1]
                        if len(sorted_ts) >= 3:
                            post_urbs_peak_3 = sorted_ts.iloc[2]
                            post_urbs_ts_3 = sorted_ts.index[2]
            except Exception:
                pass
            
    except Exception as e:
        print(f"Error reading {h5_file.name}: {e}")
        return None
        
    return {
        'Grid ID': grid_idx,
        'PLZ': plz,
        'Number of buildings': num_buildings,
        'Number of lines': num_lines,
        'Max aggregated demand from Pylovo': pylovo_grid_peak,
        'Max aggregated demand from Demand Allocation (Pre Urbs)': pre_urbs_grid_peak,
        'Max aggregated demand from Post Urbs': post_urbs_grid_peak,
        'Post-Urbs Peak Demand TS': post_urbs_ts_1,
        '2nd highest Post-Urbs demand': post_urbs_peak_2,
        '2nd highest Post-Urbs demand TS': post_urbs_ts_2,
        '3rd highest Post-Urbs demand': post_urbs_peak_3,
        '3rd highest Post-Urbs demand TS': post_urbs_ts_3,
        'datasource_idx': datasource_idx
    }

# =====================================================================
# MAIN REPORT GENERATION
# =====================================================================

def generate_reports():
    # 1. Find and process .h5 files from INPUT_FOLDERS
    h5_files = []
    for idx, folder in enumerate(INPUT_FOLDERS):
        search_path = Path(folder)
        if not search_path.exists():
            print(f"WARNING: The folder '{folder}' does not exist! Skipping...")
            continue
        for f in search_path.glob("*.h5"):
            h5_files.append((f, idx))
    
    total_files = len(h5_files)
    print(f"Found {total_files} .h5 files to check.\n")
    
    if total_files == 0:
        print("No H5 files found to process. Exiting.")
        return
        
    # Run the processing in parallel
    valid_files_args = [(i, total_files, f, idx) for i, (f, idx) in enumerate(h5_files)]
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"Starting parallel execution using {num_cpus} worker processes...")
    
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(process_single_h5, valid_files_args)
        
    results = [r for r in results if r is not None]
    if not results:
        print("No valid data extracted.")
        return
        
    # Combine results
    df_all = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define common rename map for the metrics columns matching the user's images
    rename_metrics = {
        'Number of buildings': '# buildings',
        'Number of lines': '# Lines',
        'Max aggregated demand from Pylovo': 'Max dmd Pylovo',
        'Max aggregated demand from Demand Allocation (Pre Urbs)': 'Max allocated dmd',
        'Max aggregated demand from Post Urbs': '1st Highest PostUrbs dmd',
        'Post-Urbs Peak Demand TS': 'TS to 1st Highest',
        '2nd highest Post-Urbs demand': '2nd Highest PostUrbs dmd',
        '2nd highest Post-Urbs demand TS': 'TS to 2nd Highest',
        '3rd highest Post-Urbs demand': '3rd Highest PostUrbs dmd',
        '3rd highest Post-Urbs demand TS': 'TS to 3rd Highest'
    }
    
    # 2. Create Pedro's Table (Pedro's path, datasource_idx == 0)
    df_pedro_all = df_all[df_all['datasource_idx'] == 0] if not df_all.empty else pd.DataFrame()
    if not df_pedro_all.empty:
        df_pedro_out = df_pedro_all.drop_duplicates(subset=['Grid ID'], keep='last').copy()
        df_pedro_out['Affected?'] = df_pedro_out['Grid ID'].apply(
            lambda x: 'Yes' if x in AFFECTED_INDEXES else 'No'
        )
        df_pedro_out = df_pedro_out.rename(columns=rename_metrics)
        
        cols_pedro = [
            'Grid ID', 'Affected?', '# buildings', '# Lines', 'Max dmd Pylovo', 
            'Max allocated dmd', '1st Highest PostUrbs dmd', 'TS to 1st Highest', 
            '2nd Highest PostUrbs dmd', 'TS to 2nd Highest', '3rd Highest PostUrbs dmd', 
            'TS to 3rd Highest'
        ]
        df_pedro_out = df_pedro_out[cols_pedro].sort_values(by='Grid ID')
        
        out_path = output_dir / "pedro_aggregated_peaks.xlsx"
        df_pedro_out.to_excel(out_path, index=False)
        print(f"Saved Pedro's table to {out_path} with {len(df_pedro_out)} rows.")
    else:
        print("No files found in Pedro's datasource. Pedro's table not created.")
        
    # 3. Create Elias's Table (Elias's path, datasource_idx == 1)
    df_elias_all = df_all[df_all['datasource_idx'] == 1] if not df_all.empty else pd.DataFrame()
    if not df_elias_all.empty:
        df_elias_out = df_elias_all.drop_duplicates(subset=['Grid ID'], keep='last').copy()
        df_elias_out['PLZ affected?'] = df_elias_out['PLZ'].apply(
            lambda x: 'yes' if str(x) in AFFECTED_PLZ else 'no'
        )
        df_elias_out = df_elias_out.rename(columns=rename_metrics)
        
        cols_elias = [
            'Grid ID', 'PLZ', 'PLZ affected?', '# buildings', '# Lines', 'Max dmd Pylovo', 
            'Max allocated dmd', '1st Highest PostUrbs dmd', 'TS to 1st Highest', 
            '2nd Highest PostUrbs dmd', 'TS to 2nd Highest', '3rd Highest PostUrbs dmd', 
            'TS to 3rd Highest'
        ]
        df_elias_out = df_elias_out[cols_elias].sort_values(by=['PLZ', 'Grid ID'])
        
        out_path = output_dir / "elias_aggregated_peaks.xlsx"
        df_elias_out.to_excel(out_path, index=False)
        print(f"Saved Elias's table to {out_path} with {len(df_elias_out)} rows.")
    else:
        print("No files found in Elias's datasource. Elias's table not created.")


if __name__ == "__main__":
    generate_reports()
