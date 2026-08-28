# =====================================================================
# Based on: GridExpand/4.powerflow/my_scripts/disconnect_MV_buildings.py
#
# This script scans HDF5 files in the demand allocation folder and 
# disconnects buildings with electric peak demand loads above certain 
# thresholds to protect the grid.
# 
# Disconnection Thresholds:
#   - Public and Commercial buildings: > 100 kW peak electricity demand
#   - Residential buildings: > 250 kW peak electricity demand
#
# Log output prints:
#   - Grid indexes of H5 files processed.
#   - Number of matching buildings found & disconnected per file.
#   - The percentage of original buildings that remain active.
# =====================================================================

import os
import io
import json
from pathlib import Path
import h5py
import pandas as pd

# Disable HDF5 file locking to avoid errors on network mounts or HPC filesystems
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Folder containing intermediate allocated-demand grid H5 files
INPUT_FOLDER = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/2nd_RUN/2.Demand_Allocated/"

# Peak demand thresholds in kilowatts (kW)
COMMERCIAL_PUBLIC_THRESHOLD_KW = 100.0
RESIDENTIAL_THRESHOLD_KW = 250.0


def extract_index_from_filename(filename):
    """
    Extracts the grid index integer from the prefix of a grid filename.
    Example: '534_N3237500E4299500_...' -> 534
    """
    try:
        parts = filename.split('_')
        return int(parts[0])
    except (ValueError, IndexError):
        return None


def process_hdf5_file(file_path):
    """
    Reads the HDF5 grid file, evaluates each building against electricity 
    peak demand thresholds, sets high-demand building loads as out of service
    in the pandapower network data, and saves the updated network back to H5.
    
    Returns:
        A dictionary containing processing statistics, or None if failed.
    """
    # -----------------------------------------------------------------
    # Step 1: Read buildings and demand tables from the HDF5 file
    # -----------------------------------------------------------------
    try:
        # Load buildings table (contains use type and associated bus ID)
        df_buildings = pd.read_hdf(str(file_path), key='/raw_data/buildings')
        # Load demand profile time series (columns correspond to (bus_id, commodity))
        df_demand = pd.read_hdf(str(file_path), key='/urbs_in/demand')
    except Exception as e:
        print(f"  [Error] Failed to read HDF5 keys from {file_path.name}: {e}")
        return None

    # Total number of buildings in the grid
    total_buildings = len(df_buildings)
    if total_buildings == 0:
        return {
            "total_buildings": 0,
            "matching_found": 0,
            "newly_disconnected": 0,
            "remaining_percent": 100.0
        }

    # -----------------------------------------------------------------
    # Step 2: Compute peak electricity demand for each bus/building
    # -----------------------------------------------------------------
    # MultiIndex columns in df_demand: level 0 is bus ID, level 1 is commodity
    # Filter columns to only keep those representing 'electricity'
    elec_cols = [col for col in df_demand.columns if col[1] == 'electricity']
    
    # Store peak demand (maximum value over time) in a dictionary mapped by bus ID
    bus_peak_demands = {}
    for col in elec_cols:
        bus_id_str = str(col[0])
        peak_value = df_demand[col].max()
        bus_peak_demands[bus_id_str] = peak_value

    # -----------------------------------------------------------------
    # Step 3: Check each building against the peak load thresholds
    # -----------------------------------------------------------------
    buses_to_disconnect = set()
    matching_buildings_count = 0

    for idx, row in df_buildings.iterrows():
        bldg_use = str(row.get('use', '')).strip().lower()
        bus_id = row.get('bus')
        
        # Skip if building is not attached to any bus
        if pd.isna(bus_id):
            continue
            
        bus_id_str = str(bus_id)
        
        # Get peak electricity demand for this bus (default to 0.0 if not found)
        peak_demand = bus_peak_demands.get(bus_id_str, 0.0)
        
        should_disconnect = False
        
        # Determine if building exceeds the allowed peak demand threshold
        if bldg_use in ['public', 'commercial']:
            if peak_demand > COMMERCIAL_PUBLIC_THRESHOLD_KW:
                should_disconnect = True
        elif bldg_use == 'residential':
            if peak_demand > RESIDENTIAL_THRESHOLD_KW:
                should_disconnect = True
                
        if should_disconnect:
            matching_buildings_count += 1
            # Add to the set of buses to turn off in the network
            buses_to_disconnect.add(bus_id_str)

    # If no buildings need to be disconnected, we can skip updating the network
    if not buses_to_disconnect:
        return {
            "total_buildings": total_buildings,
            "matching_found": 0,
            "newly_disconnected": 0,
            "remaining_percent": 100.0
        }

    # -----------------------------------------------------------------
    # Step 4: Disconnect buses in the pandapower network data
    # -----------------------------------------------------------------
    newly_disconnected_count = 0
    try:
        # Open HDF5 file in read-write mode
        with h5py.File(str(file_path), 'r+') as f:
            if 'raw_data/net' not in f:
                print(f"  [Warning] 'raw_data/net' is missing in {file_path.name}.")
                return None
                
            # Read network JSON string
            net_json = f['raw_data/net'][()]
            net_data = json.loads(net_json)
            
            # Load the 'load' dataframe which is serialized as JSON
            if '_object' in net_data and 'load' in net_data['_object']:
                load_table_info = net_data['_object']['load']
                df_load = pd.read_json(
                    io.StringIO(load_table_info['_object']), 
                    orient=load_table_info['orient']
                )
                
                # Make sure the 'in_service' column exists
                if 'in_service' not in df_load.columns:
                    df_load['in_service'] = True
                
                # Identify which loads are on the buses we want to disconnect
                load_mask = df_load['bus'].astype(str).isin(buses_to_disconnect)
                
                # Count how many loads are newly turned off (were in_service = True)
                newly_disconnected_count = df_load[load_mask & (df_load['in_service'] != False)].shape[0]
                
                # Turn off the selected loads
                df_load.loc[load_mask, 'in_service'] = False
                
                # Save the load table DataFrame back to the network structure
                load_table_info['_object'] = df_load.to_json(orient=load_table_info['orient'])
                
                # Re-serialize the entire network to JSON and write back to HDF5
                updated_net_json = json.dumps(net_data)
                del f['raw_data/net']
                f.create_dataset('raw_data/net', data=updated_net_json)
            else:
                print(f"  [Warning] Load table is missing in network json of {file_path.name}.")
                return None
                
    except Exception as e:
        print(f"  [Error] Failed to write changes to network in {file_path.name}: {e}")
        return None

    # Calculate remaining active buildings
    remaining_count = total_buildings - matching_buildings_count
    remaining_percent = (remaining_count / total_buildings) * 100.0

    return {
        "total_buildings": total_buildings,
        "matching_found": matching_buildings_count,
        "newly_disconnected": newly_disconnected_count,
        "remaining_percent": remaining_percent
    }


def main():
    input_dir = Path(INPUT_FOLDER)
    if not input_dir.exists():
        print(f"Error: The input directory '{INPUT_FOLDER}' does not exist.")
        return

    # Find and sort all .h5 files in the folder
    h5_files = sorted(list(input_dir.glob("*.h5")))
    print(f"Found {len(h5_files)} .h5 files in {INPUT_FOLDER}")

    print("\nStarting processing...")
    print("=" * 80)
    print(f"{'Grid Index':<12} | {'Total Bldgs':<12} | {'Found & Disc':<14} | {'Remaining %':<12}")
    print("=" * 80)

    processed_indices = []

    for file_path in h5_files:
        grid_idx = extract_index_from_filename(file_path.name)
        if grid_idx is None:
            # Skip if filename doesn't start with a valid grid index
            continue

        processed_indices.append(grid_idx)
        stats = process_hdf5_file(file_path)
        
        if stats is not None:
            print(f"{grid_idx:<12} | {stats['total_buildings']:<12} | {stats['matching_found']:<14} | {stats['remaining_percent']:>10.2f}%")
        else:
            print(f"{grid_idx:<12} | {'Failed':<12} | {'-':<14} | {'-':<12}")

    print("=" * 80)
    print(f"Successfully processed {len(processed_indices)} grid files.")
    print(f"Processed grid indexes list: {processed_indices}")


if __name__ == "__main__":
    main()
