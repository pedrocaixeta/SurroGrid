import sys
import glob
import os
import shutil
import h5py
import pandapower as pp
import networkx as nx

def merge_short_lines(net, min_length_km=0.015):
    """
    Finds lines shorter than min_length_km and removes them by merging 
    the two buses they connect.
    
    To safely merge Bus B into Bus A:
    1. Reconnect all lines, loads, and other elements from Bus B to Bus A.
    2. Deactivate Bus B.
    3. Delete the short line.
    """
    df_lines = net.line.copy()
    
    # 1. Find all short lines
    short_lines = df_lines[df_lines["length_km"] < min_length_km] #list of lines with length smaller than min_length_km
    
    for line_idx, _ in short_lines.iterrows():
        # Check if line was already removed in a previous iteration
        if line_idx not in net.line.index:
            continue
            
        # Dynamically fetch current bus ids from the active net.line
        bus_a = net.line.at[line_idx, "from_bus"]
        bus_b = net.line.at[line_idx, "to_bus"]
        
        # If the line has become a self-loop due to previous merges, just drop it
        if bus_a == bus_b:
            net.line = net.line.drop(index=line_idx)
            continue
        
        # Decide which bus to keep (Bus A) and which to remove (Bus B)
        # We prefer to keep the bus that has more lines connected to it (junctions).
        lines_in_a = len(net.line[(net.line["from_bus"] == bus_a) | (net.line["to_bus"] == bus_a)])
        lines_in_b = len(net.line[(net.line["from_bus"] == bus_b) | (net.line["to_bus"] == bus_b)])

        # If Bus B has more connections, swap them so Bus A is always the one we keep
        if lines_in_b > lines_in_a:
            bus_a, bus_b = bus_b, bus_a
            
            
        # --- REDIRECT EVERYTHING FROM BUS B TO BUS A ---
        
        # A. Redirect Lines
        net.line.loc[net.line["from_bus"] == bus_b, "from_bus"] = bus_a #it finds any line starting at bus_b and redirects it to start at bus_a.
        net.line.loc[net.line["to_bus"] == bus_b, "to_bus"] = bus_a # it finds any line ending at bus_b and redirects it to end at bus_a.
        
        # B. Redirect Loads (Consumers)
        net.load.loc[net.load["bus"] == bus_b, "bus"] = bus_a
        
        # C. Redirect Static Generators (PV panels)
        if "sgen" in net and not net.sgen.empty:
            net.sgen.loc[net.sgen["bus"] == bus_b, "bus"] = bus_a
            
        # D. Redirect Switches
        if "switch" in net and not net.switch.empty:
            net.switch.loc[net.switch["bus"] == bus_b, "bus"] = bus_a
            net.switch.loc[(net.switch["element"] == bus_b) & (net.switch["et"] == "b"), "element"] = bus_a
            
        # E. Redirect Transformers (2-winding)
        if "trafo" in net and not net.trafo.empty:
            net.trafo.loc[net.trafo["hv_bus"] == bus_b, "hv_bus"] = bus_a
            net.trafo.loc[net.trafo["lv_bus"] == bus_b, "lv_bus"] = bus_a

        # F. Redirect Transformers (3-winding)
        if "trafo3w" in net and not net.trafo3w.empty:
            net.trafo3w.loc[net.trafo3w["hv_bus"] == bus_b, "hv_bus"] = bus_a
            net.trafo3w.loc[net.trafo3w["mv_bus"] == bus_b, "mv_bus"] = bus_a
            net.trafo3w.loc[net.trafo3w["lv_bus"] == bus_b, "lv_bus"] = bus_a

        # G. Redirect External Grid Slack Connections
        if "ext_grid" in net and not net.ext_grid.empty:
            net.ext_grid.loc[net.ext_grid["bus"] == bus_b, "bus"] = bus_a

        # H. Redirect Storages
        if "storage" in net and not net.storage.empty:
            net.storage.loc[net.storage["bus"] == bus_b, "bus"] = bus_a

        # I. Redirect Generators
        if "gen" in net and not net.gen.empty:
            net.gen.loc[net.gen["bus"] == bus_b, "bus"] = bus_a

        # J. Redirect Shunts
        if "shunt" in net and not net.shunt.empty:
            net.shunt.loc[net.shunt["bus"] == bus_b, "bus"] = bus_a

        # K. Redirect Ward Equivalents
        if "ward" in net and not net.ward.empty:
            net.ward.loc[net.ward["bus"] == bus_b, "bus"] = bus_a

        # L. Redirect Decoupled Ward Equivalents
        if "dward" in net and not net.dward.empty:
            net.dward.loc[net.dward["bus"] == bus_b, "bus"] = bus_a
            
        # --- CLEANUP ---
        
        # 1. Remove the short line
        net.line = net.line.drop(index=line_idx)
        
        # 2. Deactivate Bus B
        net.bus.loc[bus_b, "in_service"] = False

    return net

def verify_grid_health(net):
    """
    Verifies if the grid is consistent and healthy for power flow calculations:
    1. The grid must contain at least one active external grid connection (slack node).
    2. Active elements (loads, sgens, lines, switches, trafos) must not connect to inactive/deactivated buses.
    3. No active elements should be disconnected from the slack node(s).
    """
    # 1. Slack Node Check
    active_slacks = net.ext_grid[net.ext_grid.in_service == True]
    if len(active_slacks) == 0:
        print("Health Check Failed: No active external grid connection (slack node) found.")
        return False
        
    slack_buses = set(active_slacks.bus.values)
    
    # 2. Check for active elements connected to inactive buses
    inactive_buses = set(net.bus[net.bus.in_service == False].index)
    
    # Loads
    if "load" in net and not net.load.empty:
        invalid_loads = net.load[(net.load.in_service == True) & (net.load.bus.isin(inactive_buses))]
        if not invalid_loads.empty:
            print(f"Health Check Failed: Active loads connect to inactive buses: {invalid_loads.index.tolist()}")
            return False
            
    # Static Generators (sgen)
    if "sgen" in net and not net.sgen.empty:
        invalid_sgens = net.sgen[(net.sgen.in_service == True) & (net.sgen.bus.isin(inactive_buses))]
        if not invalid_sgens.empty:
            print(f"Health Check Failed: Active sgens connect to inactive buses: {invalid_sgens.index.tolist()}")
            return False

    # Lines
    if "line" in net and not net.line.empty:
        invalid_lines = net.line[(net.line.in_service == True) & ((net.line.from_bus.isin(inactive_buses)) | (net.line.to_bus.isin(inactive_buses)))]
        if not invalid_lines.empty:
            print(f"Health Check Failed: Active lines connect to inactive buses: {invalid_lines.index.tolist()}")
            return False

    # Switches
    if "switch" in net and not net.switch.empty:
        invalid_sw_bus = net.switch[(net.switch.closed == True) & (net.switch.bus.isin(inactive_buses))]
        invalid_sw_elem = net.switch[(net.switch.closed == True) & (net.switch.et == "b") & (net.switch.element.isin(inactive_buses))]
        if not invalid_sw_bus.empty or not invalid_sw_elem.empty:
            print("Health Check Failed: Active switches connect to inactive buses.")
            return False

    # Transformers (trafo)
    if "trafo" in net and not net.trafo.empty:
        invalid_trafos = net.trafo[(net.trafo.in_service == True) & ((net.trafo.hv_bus.isin(inactive_buses)) | (net.trafo.lv_bus.isin(inactive_buses)))]
        if not invalid_trafos.empty:
            print(f"Health Check Failed: Active transformers connect to inactive buses: {invalid_trafos.index.tolist()}")
            return False

    # 3. Connectivity check: verify all active buses are connected to a slack node
    try:
        # Create a networkx graph representing the grid topology
        # respect_switches=True checks connection through closed switches
        g = pp.topology.create_nxgraph(net, respect_switches=True)
        
        # Collect all reachable buses from any active slack bus
        reachable = set()
        for slack in slack_buses:
            if g.has_node(slack):
                reachable.update(nx.descendants(g, slack))
                reachable.add(slack)
                
        # Find all active buses
        active_buses = set(net.bus[net.bus.in_service == True].index)
        
        # Check if any active bus is disconnected from all slacks
        disconnected = active_buses - reachable
        if len(disconnected) > 0:
            print(f"Health Check Failed: {len(disconnected)} active buses are disconnected from the external grid: {list(disconnected)}")
            return False
            
    except Exception as e:
        print(f"Health Check Failed: Error during connectivity analysis: {e}")
        return False
        
    print("Health Check Passed: Grid is consistent and fully connected to the external grid.")
    return True

def patch_grid_file(input_filepath, output_filepath, min_length_km=0.015):
    # 1. Read the network from the HDF5 file
    with h5py.File(input_filepath, 'r') as f:
        json_data = f['raw_data/net'][()].decode('utf-8')
    
    net = pp.from_json_string(json_data)
    
    # Check how many short lines exist before merging
    short_lines_count = len(net.line[net.line["length_km"] < min_length_km])
    
    if short_lines_count == 0:
        print("No short lines identified.")
        return False
        
    # 2. Apply the merge algorithm
    net = merge_short_lines(net, min_length_km=min_length_km)
    
    # Check how many short lines exist after merging
    short_lines_after = len(net.line[net.line["length_km"] < min_length_km])
    
    if short_lines_after != 0:
        print(f"{short_lines_after} of the identified short lines could not be cut off the grids.")
        return False

    # Verify if the patched grid is healthy and consistent
    if not verify_grid_health(net):
        print("Grid health verification failed. Grid will not be saved and job will not be submitted.")
        return False
        
    # 3. Serialize back to JSON and save to the new HDF5 file
    patched_json_data = pp.to_json(net).encode('utf-8')
    
    # Copy original file to output file, then modify output file
    shutil.copy2(input_filepath, output_filepath)
    
    with h5py.File(output_filepath, 'a') as f:
        del f['raw_data/net']
        f.create_dataset('raw_data/net', data=patched_json_data)
        
    print(f"SUCCESS: all {short_lines_count} short lines were removed and grid passed health checks")
    return True

if __name__ == "__main__":
    input_folder = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/2nd_RUN/3.Post_Urbs/"
    output_folder = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/2nd_RUN/_3.PostUrbs_noshortlines/"
    
    os.makedirs(output_folder, exist_ok=True)

    # Find all .h5 files in the input folder
    search_pattern = os.path.join(input_folder, "*.h5")
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"Error: No grid files found matching pattern '{search_pattern}'")
        sys.exit(1)
        
    print(f"Found {len(matching_files)} .h5 files to process.")
    
    success_count = 0
    for target_file in matching_files:
        filename = os.path.basename(target_file)
        output_file = os.path.join(output_folder, filename)
        
        print(f"\nProcessing {filename}...")
        patched = patch_grid_file(target_file, output_file, min_length_km=0.015)
        
        if patched:
            print(f"Successfully processed and saved to {output_file}")
            success_count += 1
        else:
            print(f"File {filename} was not patched or failed health check.")
            
    print(f"\nProcessing complete. Successfully patched {success_count} out of {len(matching_files)} files.")
