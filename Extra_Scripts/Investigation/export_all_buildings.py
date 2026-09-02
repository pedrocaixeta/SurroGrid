import pandas as pd
import os
from pathlib import Path
import multiprocessing
import json
import io
import h5py

# Disable HDF5 file locking to avoid errors when accessing files over WSL mounts or network shares
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# =====================================================================
# SCRIPT LOGIC
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
            if 'use' not in massive.columns: massive['use'] = ''
            if 'height' not in massive.columns: massive['height'] = ''
            if 'type' not in massive.columns: massive['type'] = ''
            
            # Attempt to read Pylovo peak load and bus status from pandapower net json
            pylovo_peaks = {}
            bus_in_service = {}
            try:
                with h5py.File(str(h5_file), 'r') as file:
                    if 'raw_data/net' in file:
                        net_json = file['raw_data/net'][()]
                        data = json.loads(net_json)
                        if '_object' in data:
                            if 'load' in data['_object']:
                                load_data = data['_object']['load']
                                df_load = pd.read_json(io.StringIO(load_data['_object']), orient=load_data['orient'])
                                if 'bus' in df_load.columns:
                                    if 'p_mw' in df_load.columns:
                                        # Convert MW to kW
                                        pylovo_peaks = (df_load.set_index('bus')['p_mw'] * 1000.0).to_dict()
                                    if 'in_service' in df_load.columns:
                                        bus_in_service.update(df_load.set_index('bus')['in_service'].to_dict())
                            
                            if 'bus' in data['_object']:
                                bus_data = data['_object']['bus']
                                df_bus = pd.read_json(io.StringIO(bus_data['_object']), orient=bus_data['orient'])
                                if 'in_service' in df_bus.columns:
                                    # If the load is out of service (verified in the codeblock above) OR the bus is out of service, we consider it disconnected.
                                    bus_status = df_bus['in_service'].to_dict()
                                    for b, status in bus_status.items():
                                        if not status or b not in bus_in_service:
                                            bus_in_service[b] = status
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
            
            # 1. Vectorized Pre-URBS calculations
            commodities = ['electricity', 'space_heat', 'water_heat', 'mobility0']
            pre_peaks = {c: {} for c in commodities}
            pre_totals = {c: {} for c in commodities}
            
            if df_demand is not None:
                for c in commodities:
                    c_cols = [col for col in df_demand.columns if col[1] == c]
                    if c_cols:
                        df_c = df_demand[c_cols].copy()
                        df_c.columns = df_c.columns.get_level_values(0)  # Rename to bus ID
                        pre_peaks[c] = df_c.max().to_dict()
                        pre_totals[c] = df_c.sum().to_dict()

            # 2. Vectorized Post-URBS calculations
            post_import_peaks = {}
            post_import_totals = {}
            post_export_peaks = {}
            post_export_totals = {}
            
            if df_tau is not None:
                try:
                    # Extract only import/feed_in rows
                    mask = df_tau.index.get_level_values('pro').isin(['import', 'feed_in'])
                    df_filtered = df_tau[mask]
                    
                    # Unstack 'pro' level to columns
                    df_unstacked = df_filtered.unstack(level='pro')
                    
                    if 'import' in df_unstacked.columns:
                        import_series = df_unstacked['import']
                        post_import_peaks = import_series.groupby(level='sit').max().to_dict()
                        post_import_totals = import_series.groupby(level='sit').sum().to_dict()
                        
                    if 'feed_in' in df_unstacked.columns:
                        feed_in_series = df_unstacked['feed_in']
                        post_export_peaks = feed_in_series.groupby(level='sit').max().to_dict()
                        post_export_totals = feed_in_series.groupby(level='sit').sum().to_dict()
                except Exception:
                    pass

            # Populate lists using dictionary lookup
            res_pylovo = []
            res_in_service = []
            res_pre = {f'Peak {c}': [] for c in commodities}
            res_pre.update({f'Total {c}': [] for c in commodities})
            
            res_post_imp_peak = []
            res_post_imp_total = []
            res_post_exp_peak = []
            res_post_exp_total = []

            for idx, row in massive.iterrows():
                bus_id = row.get('bus')
                res_pylovo.append(pylovo_peaks.get(bus_id, 0.0))
                
                if pd.notna(bus_id) and bus_id in bus_in_service:
                    is_connected = bus_in_service[bus_id]
                    res_in_service.append('Yes' if is_connected else 'No')
                else:
                    res_in_service.append('No')
                
                for c in commodities:
                    res_pre[f'Peak {c}'].append(pre_peaks[c].get(bus_id, 0.0))
                    res_pre[f'Total {c}'].append(pre_totals[c].get(bus_id, 0.0))
                    
                res_post_imp_peak.append(post_import_peaks.get(bus_id, 0.0))
                res_post_imp_total.append(post_import_totals.get(bus_id, 0.0))
                res_post_exp_peak.append(post_export_peaks.get(bus_id, 0.0))
                res_post_exp_total.append(post_export_totals.get(bus_id, 0.0))

            # Add the calculated stats as new columns in the DataFrame
            massive['Pylovo Peak Load (kW)'] = res_pylovo
            massive['Connected to Net'] = res_in_service
            
            massive['Peak Elec Load Pre-URBS (kW)'] = res_pre['Peak electricity']
            massive['Total Elec Energy Pre-URBS (kWh)'] = res_pre['Total electricity']
            massive['Peak Space Heat Load Pre-URBS (kW)'] = res_pre['Peak space_heat']
            massive['Total Space Heat Energy Pre-URBS (kWh)'] = res_pre['Total space_heat']
            massive['Peak Water Heat Load Pre-URBS (kW)'] = res_pre['Peak water_heat']
            massive['Total Water Heat Energy Pre-URBS (kWh)'] = res_pre['Total water_heat']
            massive['Peak Mobility0 Load Pre-URBS (kW)'] = res_pre['Peak mobility0']
            massive['Total Mobility0 Energy Pre-URBS (kWh)'] = res_pre['Total mobility0']
            
            massive['Peak Energy Import Post-URBS (kW)'] = res_post_imp_peak
            massive['Total Energy Import Post-URBS (kWh)'] = res_post_imp_total
            massive['Peak Energy Export Post-URBS (kW)'] = res_post_exp_peak
            massive['Total Energy Export Post-URBS (kWh)'] = res_post_exp_total

            # ---------------------------------------------------------
            # DECIDE WHICH COLUMNS TO OUTPUT
            # The goal is to include ALL columns that were originally in 
            # the /raw_data/buildings table, plus our new calculated ones.
            # ---------------------------------------------------------
            
            # Here is the list of the new columns we've created above
            new_calculated_columns = [
                'Grid Index', 'PLZ', 'OSM ID', 'Bus ID', 'Connected to Net', 'total area',
                'Pylovo Peak Load (kW)',
                'Peak Elec Load Pre-URBS (kW)', 'Total Elec Energy Pre-URBS (kWh)',
                'Peak Space Heat Load Pre-URBS (kW)', 'Total Space Heat Energy Pre-URBS (kWh)',
                'Peak Water Heat Load Pre-URBS (kW)', 'Total Water Heat Energy Pre-URBS (kWh)',
                'Peak Mobility0 Load Pre-URBS (kW)', 'Total Mobility0 Energy Pre-URBS (kWh)',
                'Peak Energy Import Post-URBS (kW)', 'Total Energy Import Post-URBS (kWh)',
                'Peak Energy Export Post-URBS (kW)', 'Total Energy Export Post-URBS (kWh)'
            ]
            
            # Get the list of all original columns from the H5 file's building data
            original_columns = df_buildings.columns.tolist()
            
            # We will create a combined list, maintaining a nice order without duplicates
            final_columns = []
            
            # First, add all the original columns
            for col in original_columns:
                if col not in final_columns:
                    final_columns.append(col)
                    
            # Next, add our new calculated columns
            for col in new_calculated_columns:
                if col not in final_columns:
                    final_columns.append(col)
            
            # Finally, just in case any column is missing for some reason, fill it with empty strings
            for col in final_columns:
                if col not in massive.columns:
                    massive[col] = ''
            
            # Return our final prepared dataframe
            return massive[final_columns]

    except KeyError:
        # Happens if file lacks '/raw_data/buildings'
        pass
    except Exception as e:
        print(f"Error reading {h5_file.name}: {e}")
        
    return None

def find_and_export_all_buildings(input_folder, output_file):
    search_path = Path(input_folder)
    print(f"\n=======================================================")
    print(f"Processing ALL grids in Folder (No filter)")
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
    
    # Prepare arguments for files
    valid_files_args = []
    for file_idx, h5_file in enumerate(h5_files):
        grid_idx = extract_index_from_filename(h5_file.name)
        plz = extract_plz_from_filename(h5_file.name)
        valid_files_args.append((file_idx, total_files, grid_idx, plz, h5_file))
            
    if not valid_files_args:
        print("No files found.")
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
        print(f"\nNo buildings were found.")

# This tells Python to run the function when you start the script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export building stats from all H5 grid files.")
    parser.add_argument("--input-folder", required=True, help="Path to folder containing H5 files.")
    parser.add_argument("--output-file", required=True, help="Path to output Excel file.")
    
    args = parser.parse_args()
    find_and_export_all_buildings(args.input_folder, args.output_file)
