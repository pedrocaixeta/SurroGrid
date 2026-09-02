import os
import io
import json
from pathlib import Path
import h5py
import pandas as pd

# Disable HDF5 file locking to avoid conflicts when reading files over network/WSL mounts
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# =====================================================================
# CONFIGURATION
# =====================================================================

# The folder where the .h5 files are located
INPUT_FOLDER = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/result3.urbs__input4/"

# The indexes of the grids that need to be processed
AFFECTED_INDEXES = {
    21, 74, 146, 181, 199, 219, 244, 247, 303, 407, 
    421, 423, 476, 587, 641, 673, 715, 731, 743, 798, 
    819, 821, 953, 972, 980, 992, 1048, 1052, 1063, 
    1173, 1260, 1311, 1339, 1426, 1461
}

# The threshold for disconnecting a building (in kW)
DEMAND_THRESHOLD_KW = 100.0

# The maximum percentage of buildings that can be disconnected (10% = 0.10)
MAX_DISCONNECT_RATIO = 0.10

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def extract_index_from_filename(filename):
    """
    Extracts the grid index from the beginning of the filename.
    Example: '534_N3237500E4299500_...' -> 534
    """
    try:
        return int(filename.split('_')[0])
    except Exception:
        return None

def process_file(file_path):
    """
    Processes a single .h5 file:
    1. Finds public and commercial buildings.
    2. Checks if their max electrical demand > 100 kW.
    3. Respects the rule to not drop more than 10% of total buildings.
    4. Modifies the grid network to disconnect them (in_service = False).
    
    Returns: (disconnected_count, stop_reason)
    """
    print(f"--- Processing {file_path.name} ---")
    
    # ---------------------------------------------------------
    # Step 1: Read the buildings and demand data
    # ---------------------------------------------------------
    try:
        df_buildings = pd.read_hdf(str(file_path), key='/raw_data/buildings')
        df_demand = pd.read_hdf(str(file_path), key='/urbs_in/demand')
        
        # Read the current network to find already disconnected loads
        already_disconnected_buses = set()
        with h5py.File(str(file_path), 'r') as f:
            if 'raw_data/net' in f:
                net_json = f['raw_data/net'][()]
                data = json.loads(net_json)
                if '_object' in data and 'load' in data['_object']:
                    load_data = data['_object']['load']
                    df_load_temp = pd.read_json(io.StringIO(load_data['_object']), orient=load_data['orient'])
                    already_disconnected_buses = set(df_load_temp[df_load_temp['in_service'] == False]['bus'])
                    
    except Exception as e:
        print(f"  [Error] Could not read data from {file_path.name}: {e}")
        return (0, 0, "Error reading data")

    # Total number of buildings in this grid
    total_buildings = len(df_buildings)
    max_allowed_drop = int(total_buildings * MAX_DISCONNECT_RATIO)
    
    # If the grid is so small that 10% is 0 buildings, we can't disconnect any
    if max_allowed_drop == 0:
        print("  [Info] Grid is too small to drop any buildings (10% rule). Skipping.")
        return (0, "Grid too small (10% rule)")

    # Check if we have the 'use' column to identify building types
    if 'use' not in df_buildings.columns:
        print("  [Warning] Could not find the 'use' column in buildings data. Skipping.")
        return (0, "Missing 'use' column")
    
    # Filter for public and commercial buildings
    mask_pub_com = df_buildings['use'].str.lower().isin(['public', 'commercial'])
    pub_com_buildings = df_buildings[mask_pub_com]
    
    if pub_com_buildings.empty:
        print("  [Info] No public or commercial buildings found. Skipping.")
        return (0, 0, "No public/commercial buildings")

    # ---------------------------------------------------------
    # Step 2: Find buildings with max demand > 100 kW
    # ---------------------------------------------------------
    
    # Extract only the electricity demand columns
    elec_cols = [col for col in df_demand.columns if col[1] == 'electricity']
    if not elec_cols:
        print("  [Warning] No electricity demand columns found. Skipping.")
        return (0, "No electricity demand columns")
    
    df_elec = df_demand[elec_cols]
    
    # Find the maximum demand over all timestamps for each building
    # df_elec.max(axis=0) returns a Series where the index is the column tuple, e.g. (building_id, 'electricity')
    max_demands = df_elec.max(axis=0)
    
    buses_to_disconnect = []
    building_demands = []
    already_disconnected_count = 0
    
    # Iterate over the public/commercial buildings and check their max demand
    for bldg_id, row in pub_com_buildings.iterrows():
        bus_id = row.get('bus')
        if pd.isna(bus_id):
            continue
            
        # The site ID in URBS demand columns corresponds to the 'bus' ID
        bus_col_str = (str(bus_id), 'electricity')
        bus_col_int = (int(bus_id) if str(bus_id).isdigit() or isinstance(bus_id, (int, float)) else bus_id, 'electricity')
        
        # Check which format was used in the columns
        if bus_col_str in max_demands.index:
            demand = max_demands[bus_col_str]
        elif bus_col_int in max_demands.index:
            demand = max_demands[bus_col_int]
        else:
            continue
            
        # If the demand exceeds our threshold, mark the bus for disconnection
        if demand > DEMAND_THRESHOLD_KW:
            if bus_id in already_disconnected_buses:
                already_disconnected_count += 1
                continue
                
            # Avoid adding the same bus multiple times if multiple buildings share a bus
            if bus_id not in buses_to_disconnect:
                buses_to_disconnect.append(bus_id)
                building_demands.append(demand)
            
    if not buses_to_disconnect:
        print(f"  [Info] No new public/commercial buildings exceed {DEMAND_THRESHOLD_KW} kW. Skipping.")
        return (0, already_disconnected_count, "No new buildings > 100 kW threshold")
        
    # ---------------------------------------------------------
    # Step 3: Enforce the 10% maximum limit
    # ---------------------------------------------------------
    
    stop_reason = "All eligible buildings disconnected"
    if len(buses_to_disconnect) > max_allowed_drop:
        print(f"  [Info] Found {len(buses_to_disconnect)} building buses to drop, but max allowed buildings is {max_allowed_drop}.")
        
        # We zip the demands and IDs, sort them descending by demand, and keep the top `max_allowed_drop`
        sorted_pairs = sorted(zip(building_demands, buses_to_disconnect), reverse=True)
        
        # Extract just the bus IDs from the sorted list
        buses_to_disconnect = [bus_id for demand, bus_id in sorted_pairs[:max_allowed_drop]]
        stop_reason = "10% limit reached"
    
    print(f"  [Action] Disconnecting {len(buses_to_disconnect)} loads (at buses): {buses_to_disconnect}")

    # ---------------------------------------------------------
    # Step 4: Disconnect buildings in the pandapower network
    # ---------------------------------------------------------
    
    # We open the HDF5 file in read/write mode ('r+') to modify the network
    try:
        with h5py.File(str(file_path), 'r+') as f:
            if 'raw_data/net' not in f:
                print("  [Warning] No network data ('raw_data/net') found in the file.")
                return (0, already_disconnected_count, "Missing network data")
                
            # Read the JSON string representing the network
            net_json = f['raw_data/net'][()]
            data = json.loads(net_json)
            
            # Navigate to the 'load' dataframe inside the pandapower JSON
            if '_object' in data and 'load' in data['_object']:
                load_data = data['_object']['load']
                
                # Convert the JSON representation of the loads back into a pandas DataFrame
                df_load = pd.read_json(io.StringIO(load_data['_object']), orient=load_data['orient'])
                
                # Identify the loads that correspond to the buildings we want to disconnect.
                # In Pylovo, the load is placed on the building's bus. 
                # So we match the 'bus' column in df_load with our buses_to_disconnect.
                disconnect_strs = [str(b) for b in buses_to_disconnect]
                mask = df_load['bus'].astype(str).isin(disconnect_strs)
                
                # Set them as out of service
                df_load.loc[mask, 'in_service'] = False
                
                # Serialize the DataFrame back to JSON
                load_data['_object'] = df_load.to_json(orient=load_data['orient'])
                
                # Convert the entire network dict back to a JSON string
                new_net_json = json.dumps(data)
                
                # Delete the old dataset and create a new one with the updated string
                del f['raw_data/net']
                f.create_dataset('raw_data/net', data=new_net_json)
                
                print("  [Success] Network modified successfully.")
                return (len(buses_to_disconnect), already_disconnected_count, stop_reason)
            else:
                print("  [Warning] Network JSON format is missing the 'load' objects.")
                return (0, already_disconnected_count, "Missing load objects in network")
                
    except Exception as e:
        print(f"  [Error] Failed to update network in {file_path.name}: {e}")
        return (0, already_disconnected_count, "Error updating network")

# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    folder_path = Path(INPUT_FOLDER)
    
    if not folder_path.exists():
        print(f"Error: The folder '{INPUT_FOLDER}' does not exist.")
        return
        
    # Find all .h5 files in the directory
    all_h5_files = list(folder_path.glob("*.h5"))
    print(f"Found {len(all_h5_files)} .h5 files in the directory.")
    
    processed_count = 0
    summary_results = []
    
    for file_path in all_h5_files:
        grid_idx = extract_index_from_filename(file_path.name)
        
        # Only process files whose index is in the AFFECTED_INDEXES set
        if grid_idx in AFFECTED_INDEXES:
            stats = process_file(file_path)
            if stats:
                disc_count, already_disc_count, reason = stats
                summary_results.append({
                    "Grid ID": grid_idx,
                    "Newly Disconnected": disc_count,
                    "Already Disconnected": already_disc_count,
                    "Stop Reason": reason
                })
            processed_count += 1
            
    print(f"\nFinished! Processed {processed_count} affected grids.")
    
    print("\n" + "="*95)
    print("                                SUMMARY OVERVIEW")
    print("="*95)
    # Sort the results by Grid ID for a cleaner output
    for res in sorted(summary_results, key=lambda x: x["Grid ID"]):
        print(f"Grid ID: {res['Grid ID']:<5} | Newly Disconnected: {res['Newly Disconnected']:<3} | Already Disconnected: {res['Already Disconnected']:<3} | Reason: {res['Stop Reason']}")
    print("="*95)

if __name__ == "__main__":
    main()
