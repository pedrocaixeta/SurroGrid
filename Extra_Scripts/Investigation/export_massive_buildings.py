import pandas as pd
import os
from pathlib import Path
import multiprocessing

# Disable HDF5 file locking to avoid errors when accessing files over WSL mounts or network shares
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# =====================================================================
# 1. SETTINGS
# =====================================================================

# Put the path to the folder containing your old .h5 files here.
# (Since you are running this inside your WSL Ubuntu terminal, use Linux paths!)
INPUT_H5_FOLDER = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/result3.urbs__input4/"
OUTPUT_EXCEL_FILE = "/dss/dsshome1/05/go49cer2/SurroGrid/GridExpand/4.powerflow/Output/massive_buildings_test.xlsx"

# This is the list of all the PLZs to search for (Case 1 and 2)!
PLZS_TO_SEARCH = [
    "63739", "63741", "63743", "63840", "63843", "80539", "80995", 
    "81829", "81925", "81927", "82024", "82481", "83080", "83088", 
    "83104", "83308", "83435", "83512", "84085", "84359", "84555", 
    "85053", "85054", "85084", "85399", "86163", "86165", "86399", 
    "86672", "91074", "91094", "92224", "92280", "92533", "93158", 
    "93449", "94315", "94362", "95213", "95643", "96049", "96050", 
    "96264", "97422", "97737"
]

# This is the list of all the grid indexes to search for (Case 3 and 4)!
INDEXES_TO_SEARCH = [
    21, 74, 119, 146, 181, 199, 214, 219, 244, 246, 247, 303, 326, 340, 
    407, 414, 417, 421, 423, 434, 440, 441, 476, 510, 566, 567, 579, 
    587, 590, 633, 637, 641, 673, 674, 705, 706, 714, 715, 726, 731, 
    743, 745, 748, 780, 784, 798, 819, 820, 821, 881, 890, 908, 929, 
    953, 972, 973, 977, 980, 992, 1036, 1048, 1052, 1063, 1069, 1071, 
    1085, 1114, 1120, 1145, 1170, 1173, 1177, 1194, 1214, 1232, 1239, 
    1260, 1311, 1322, 1339, 1350, 1359, 1426, 1461
]

# =====================================================================
# 2. SCRIPT LOGIC
# =====================================================================

def extract_plz_from_filename(filename):
    """
    Extracts the PLZ (postal code) from the Pylovo grid filename.
    Example filename: '1036_N3042500E4198500_63840_1_3_...'
    The third element when splitting by underscores is the PLZ ('63840').
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
    Extracts the grid index from the Pylovo grid filename.
    Example filename: '1036_N3042500E4198500_63840_1_3_...'
    The first element when splitting by underscores is the index ('1036').
    """
    try:
        parts = filename.split('_')
        if len(parts) >= 1:
            return int(parts[0])
    except Exception:
        pass
    return None

def process_file_helper(args):
    """
    Helper function to process a single H5 file in parallel.
    Args:
        args: A tuple containing (file_idx, total_files, grid_idx, plz, h5_file_path)
    Returns:
        A pandas DataFrame of processed building stats, or None.
    """
    file_idx, total_files, grid_idx, plz, h5_file = args
    print(f"[{file_idx+1}/{total_files}] Processing Grid {grid_idx} (PLZ {plz}) in {h5_file.name}...")
    
    try:
        # Read the raw buildings data from the .h5 file
        df_buildings = pd.read_hdf(str(h5_file), key='/raw_data/buildings')
        
        # Make sure we have 'area' and 'floors' columns to do our math
        if 'area' in df_buildings.columns and 'floors' in df_buildings.columns:
            df_buildings['total area'] = df_buildings['area'] * df_buildings['floors']
        else:
            return None
            
        massive = df_buildings.copy()

        # If we found buildings in this grid, prepare them
        if not massive.empty:
            massive['Grid Index'] = grid_idx
            massive['PLZ'] = plz
            massive['OSM ID'] = massive['vertice_id'] if 'vertice_id' in massive.columns else ''
            massive['Bus ID'] = massive['bus'] if 'bus' in massive.columns else ''
            
            # Make sure the 'use' column exists (even if empty) for a clean output
            if 'use' not in massive.columns:
                massive['use'] = ''
                
            # Make sure these columns exist, even if empty, so the output looks clean
            if 'height' not in massive.columns: massive['height'] = ''
            if 'type' not in massive.columns: massive['type'] = ''
            
            # Attempt to read Pylovo peak load from pandapower net json
            pylovo_peaks = {}
            try:
                import json
                import io
                import h5py
                with h5py.File(str(h5_file), 'r') as file:
                    if 'raw_data/net' in file:
                        net_json = file['raw_data/net'][()]
                        data = json.loads(net_json)
                        if '_object' in data and 'load' in data['_object']:
                            load_data = data['_object']['load']
                            df_load = pd.read_json(io.StringIO(load_data['_object']), orient=load_data['orient'])
                            if 'bus' in df_load.columns and 'p_mw' in df_load.columns:
                                # Convert MW to kW
                                pylovo_peaks = (df_load.set_index('bus')['p_mw'] * 1000.0).to_dict()
            except Exception:
                pass

            # Attempt to read demand profiles (before URBS) and MILP results (after URBS)
            try:
                df_demand = pd.read_hdf(str(h5_file), key='/urbs_in/demand')
            except Exception:
                df_demand = None
                
            try:
                df_tau = pd.read_hdf(str(h5_file), key='/urbs_out/MILP/tau_pro')
            except Exception:
                df_tau = None
            
            # Dictionary storage for fast lookup
            pre_peaks = {}
            pre_totals = {}
            post_peaks = {}
            post_totals = {}

            # 1. Vectorized Pre-URBS calculations
            if df_demand is not None:
                # Filter columns where commodity is 'electricity'
                elec_cols = [col for col in df_demand.columns if col[1] == 'electricity']
                if elec_cols:
                    df_elec = df_demand[elec_cols].copy()
                    df_elec.columns = df_elec.columns.get_level_values(0)  # Rename to bus ID
                    pre_peaks = df_elec.max().to_dict()
                    pre_totals = df_elec.sum().to_dict()

            # 2. Vectorized Post-URBS calculations
            if df_tau is not None:
                try:
                    # Extract only import/feed_in rows
                    mask = df_tau.index.get_level_values('pro').isin(['import', 'feed_in'])
                    df_filtered = df_tau[mask]
                    
                    # Unstack 'pro' level to columns
                    df_unstacked = df_filtered.unstack(level='pro')
                    
                    # Make sure columns exist, otherwise fill with 0
                    import_series = df_unstacked['import'] if 'import' in df_unstacked.columns else pd.Series(0.0, index=df_unstacked.index)
                    feed_in_series = df_unstacked['feed_in'] if 'feed_in' in df_unstacked.columns else pd.Series(0.0, index=df_unstacked.index)
                    
                    net_load = import_series - feed_in_series
                    
                    # Group by 'sit' (bus ID) and aggregate
                    post_peaks = net_load.groupby(level='sit').max().to_dict()
                    post_totals = net_load.groupby(level='sit').sum().to_dict()
                except Exception:
                    pass

            # Populate lists using dictionary lookup
            pylovo_peak_list = []
            peak_pre_list = []
            total_pre_list = []
            peak_post_list = []
            total_post_list = []

            for idx, row in massive.iterrows():
                bus_id = row.get('bus')
                pylovo_peak_list.append(pylovo_peaks.get(bus_id, 0.0))
                peak_pre_list.append(pre_peaks.get(bus_id, 0.0))
                total_pre_list.append(pre_totals.get(bus_id, 0.0))
                peak_post_list.append(post_peaks.get(bus_id, 0.0))
                total_post_list.append(post_totals.get(bus_id, 0.0))

            # Add the calculated stats as new columns in the DataFrame
            massive['Pylovo Peak Load (kW)'] = pylovo_peak_list
            massive['Peak Elec Load Pre-URBS (kW)'] = peak_pre_list
            massive['Total Elec Energy Pre-URBS (kWh)'] = total_pre_list
            massive['Peak Load Post-URBS (kW)'] = peak_post_list
            massive['Total Energy Post-URBS (kWh)'] = total_post_list

            cols_to_keep = [
                'Grid Index', 'PLZ', 'OSM ID', 'Bus ID', 'use', 'type', 'floors', 'height', 'area', 'total area',
                'Pylovo Peak Load (kW)',
                'Peak Elec Load Pre-URBS (kW)', 'Total Elec Energy Pre-URBS (kWh)',
                'Peak Load Post-URBS (kW)', 'Total Energy Post-URBS (kWh)'
            ]
            
            # Make sure all columns in cols_to_keep are defined
            for col in cols_to_keep:
                if col not in massive.columns:
                    massive[col] = ''
            
            return massive[cols_to_keep]

    except KeyError:
        # Happens if file lacks '/raw_data/buildings'
        pass
    except Exception as e:
        print(f"Error reading {h5_file.name}: {e}")
        
    return None

def find_and_export_massive_buildings(input_folder, output_file, filter_type):
    search_path = Path(input_folder)
    print(f"\n=======================================================")
    print(f"Processing Filter Type: '{filter_type}'")
    print(f"Input Folder:  {search_path.absolute()}")
    print(f"Output File:   {output_file}")
    print(f"=======================================================")
    
    # Check if the directory actually exists
    if not search_path.exists():
        print(f"ERROR: The folder '{input_folder}' does not exist!")
        return

    # Find all .h5 files in the folder (but NOT in sub-folders, to avoid backups)
    h5_files = list(search_path.glob("*.h5"))
    total_files = len(h5_files)
    print(f"Found {total_files} .h5 files to check.\n")
    
    # Prepare arguments for files matching our filter list
    valid_files_args = []
    for file_idx, h5_file in enumerate(h5_files):
        grid_idx = extract_index_from_filename(h5_file.name)
        plz = extract_plz_from_filename(h5_file.name)
        
        is_valid = False
        if filter_type == "plz":
            if plz in PLZS_TO_SEARCH:
                is_valid = True
        elif filter_type == "index":
            if grid_idx in INDEXES_TO_SEARCH:
                is_valid = True
        elif filter_type == "not_index":
            if grid_idx is not None and grid_idx not in INDEXES_TO_SEARCH:
                is_valid = True
                
        if is_valid:
            valid_files_args.append((file_idx, total_files, grid_idx, plz, h5_file))
            
    print(f"Of the found files, {len(valid_files_args)} match the filter '{filter_type}' and will be processed.")
    
    if not valid_files_args:
        print("No matching files found. Skipping this case.")
        return
        
    # Get the number of CPU cores to use.
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    print(f"Starting parallel execution using {num_cpus} worker processes...")
    
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(process_file_helper, valid_files_args)
        
    # Filter out None and empty dataframes
    all_massive_buildings = [df for df in results if df is not None and not df.empty]
            
    # Once we checked all files, combine the results and save the Excel file
    if all_massive_buildings:
        # Combine all the small tables into one big table
        final_df = pd.concat(all_massive_buildings, ignore_index=True)
        
        # Sort it so it looks nice (Groups by PLZ, then puts the biggest buildings first)
        final_df = final_df.sort_values(by=['PLZ', 'total area'], ascending=[True, False])
        
        # Make sure the output directory exists
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to Excel
        final_df.to_excel(str(out_path), index=False)
        print(f"\nSuccess! Found {len(final_df)} buildings.")
        print(f"They have been saved to: {out_path.absolute()}")
    else:
        print(f"\nNo buildings were found matching the filter.")

# This tells Python to run the function when you start the script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export building stats from H5 grid files.")
    parser.add_argument("--input-folder", default=INPUT_H5_FOLDER, help="Path to folder containing H5 files.")
    parser.add_argument("--output-file", default=OUTPUT_EXCEL_FILE, help="Path to output Excel file.")
    parser.add_argument("--filter-type", default="plz", choices=["plz", "index", "not_index"], help="Filtering type.")
    
    args = parser.parse_args()
    find_and_export_massive_buildings(args.input_folder, args.output_file, args.filter_type)
