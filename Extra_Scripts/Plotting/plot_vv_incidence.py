"""
Calculate Voltage Violation Incidence for Multiple Grid Scenarios

This script scans a directory for GridExpand / SurroGrid HDF5 (.h5) 
scenario files. For each grid, it calculates the frequency of voltage 
violations (both overvoltage and undervoltage) strictly for building buses.

The frequency is calculated as:
(Total Violations) / (Number of Buildings * Total Timesteps)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Disable HDF5 file locking to prevent crashes on shared/cluster filesystems
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# --- CONFIGURATION ---
PATH_TO_GRIDS = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/EliasH/PostPowerflow/"
PATH_TO_PLOT = "/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/Extra_Scripts/Plotting/output/My_figures"
PLOT_TITLE = "Incidence of Voltage Violations by Grid Size - Elias"
PLOT_FILENAME = "incidence_of_voltage_violations_by_grid_size_Elias.png"
PLOT_COLORMAP = "Greens" # Choose the colormap (e.g. 'Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis')


def get_building_bus_ids(filepath):
    """
    Opens the HDF5 file and identifies the columns (bus IDs) that correspond to buildings.
    """
    buildings_key = '/raw_data/buildings'
    
    with pd.HDFStore(filepath, mode='r') as store:
        if buildings_key not in store:
            print(f"Error: Buildings key not found in the HDF5 file {filepath}.")
            return []
        
        buildings_df = store[buildings_key]
        # Extract unique bus IDs where buildings are connected
        building_bus_ids = buildings_df['bus'].unique().tolist()
        
    return building_bus_ids


def count_voltage_violations(filepath, building_bus_ids):
    """
    Loads the voltage data for the specific building buses and counts violations.
    Uses pandas and numpy instead of manual loops because it's significantly faster.
    """
    vm_key = '/pwrflw/output/post/vm'
    
    with pd.HDFStore(filepath, mode='r') as store:
        if vm_key not in store:
            return None
        vm_all = store[vm_key]
        
    # Filter the voltage table to only include columns for building buses
    building_columns = [col for col in vm_all.columns if int(col) in building_bus_ids]
    if not building_columns:
        return None
    vm_buildings = vm_all[building_columns]
    
    # Extract values as a numpy array for very fast mathematical operations
    voltage_values = vm_buildings.values
    
    # Total observations = Timestamps * Number of Building Columns
    n_timesteps = len(vm_buildings.index)
    if n_timesteps != 8760:
        print(f"Error: Number of timesteps is not 8760 in the HDF5 file {filepath}.")
        return None
        
    total_observations = n_timesteps * len(building_columns)
    
    # Count undervoltage and overvoltage entries
    undervoltage_count = int((voltage_values < 0.9).sum())
    overvoltage_count = int((voltage_values > 1.1).sum())

    v_min_observed = float(np.nanmin(voltage_values)) 
    v_max_observed = float(np.nanmax(voltage_values)) 
    
    return {
        'undervoltage_count': undervoltage_count,
        'overvoltage_count': overvoltage_count,
        'total_observations': total_observations,
        'v_min_observed': v_min_observed,
        'v_max_observed': v_max_observed,
        'building_columns': building_columns,
    }


def print_voltage_statistics(grid_index, n_buildings, undervoltage_count, overvoltage_count, stats):
    """
    Prints the calculated statistics formatted nicely for a specific grid.
    """
    total_observations = stats['total_observations']
    building_columns = stats['building_columns']
    v_min_observed = stats['v_min_observed']
    v_max_observed = stats['v_max_observed']
    
    total_violations_count = undervoltage_count + overvoltage_count
    
    # Calculate frequencies (Frequency = count / (number_of_buildings * 8760))
    overvoltage_frequency = overvoltage_count / total_observations
    undervoltage_frequency = undervoltage_count / total_observations
    total_violation_frequency = total_violations_count / total_observations

    print(f"\n--- BUILDING BUS VOLTAGE VIOLATION STATISTICS: {grid_index} ---")
    print(f"Number of buildings:              {n_buildings:,}")
    print(f"Building bus IDs:                 {building_columns}")
    print(f"Total observations (Time x Bld):  {total_observations:,}")
    print(f"Observed voltage range:           [{v_min_observed:.4f}, {v_max_observed:.4f}] p.u.")
    print("-" * 60)
    print(f"Undervoltage Violations (< 0.9 p.u.):")
    print(f"  Count:                          {undervoltage_count:,}")
    print(f"  Frequency (Rate):               {undervoltage_frequency:.6f} ({undervoltage_frequency * 100:.4f}%)")
    print("-" * 60)
    print(f"Overvoltage Violations (> 1.1 p.u.):")
    print(f"  Count:                          {overvoltage_count:,}")
    print(f"  Frequency (Rate):               {overvoltage_frequency:.6f} ({overvoltage_frequency * 100:.4f}%)")
    print("-" * 60)
    print(f"Total Voltage Violations (outside [0.9, 1.1] p.u.):")
    print(f"  Count:                          {total_violations_count:,}")
    print(f"  Frequency (Rate):               {total_violation_frequency:.6f} ({total_violation_frequency * 100:.4f}%)")
    print("=" * 70)


def plot_voltage_violations_scatter(number_of_buildings, undervoltage, overvoltage, v_min_obs, v_max_obs, output_dir, title, filename, colormap):
    """
    Generates a scatter plot of Voltage Violation Frequency vs Number of Grid Buildings.
    Undervoltage frequency is plotted on the positive Y-axis.
    Overvoltage frequency is plotted on the negative Y-axis.
    Color intensity represents the magnitude of the highest violation.
    """
    building_counts = []
    violation_freqs = []
    violation_magnitudes = []
    
    for grid_idx in number_of_buildings.keys():
        n_bld = number_of_buildings[grid_idx]
        total_obs = n_bld * 8760
        
        # Calculate frequencies as percentages
        under_freq = (undervoltage[grid_idx] / total_obs) * 100.0
        over_freq = (overvoltage[grid_idx] / total_obs) * 100.0
        
        # Calculate maximum violation magnitudes
        under_mag = max(0.0, 0.9 - v_min_obs[grid_idx]) if not np.isnan(v_min_obs[grid_idx]) else 0.0
        over_mag = max(0.0, v_max_obs[grid_idx] - 1.1) if not np.isnan(v_max_obs[grid_idx]) else 0.0
        
        # Add undervoltage point (positive Y)
        building_counts.append(n_bld)
        violation_freqs.append(under_freq)
        violation_magnitudes.append(under_mag)
        
        # Add overvoltage point (negative Y)
        building_counts.append(n_bld)
        violation_freqs.append(-over_freq)
        violation_magnitudes.append(over_mag)
        
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    
    # Ensure there's a valid maximum for the colormap
    vmax = max(violation_magnitudes) if violation_magnitudes and max(violation_magnitudes) > 0 else 0.1
    
    # Plot using a colormap based on violation magnitude
    scatter = ax.scatter(
        building_counts, 
        violation_freqs, 
        c=violation_magnitudes, 
        cmap=colormap, 
        alpha=0.8, 
        edgecolor='darkgray', # Slight edge so 0 magnitude (white/light) is still visible
        linewidth=0.5,
        s=40,
        vmin=0.0,
        vmax=vmax
    )
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Number of Grid Buildings', fontsize=12)
    ax.set_ylabel('Violation Frequency [%]\n(>0: Undervoltage, <0: Overvoltage)', fontsize=12)
    ax.set_xlim(left=0)
    
    ax.grid(True, which='both', linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a horizontal line at Y=0 to clearly separate under and over voltages
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)
    
    # Add a colorbar to explain the color coding
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Max Violation Magnitude [p.u.]', fontsize=10)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nSuccessfully saved scatter plot to: {out_path}")


def main():
    
    if not os.path.exists(PATH_TO_GRIDS): # Verify the path exists so the script doesn't fail silently
        print(f"Warning: The directory '{PATH_TO_GRIDS}' does not exist.") 
        return
        
    print(f"Scanning grids in: {PATH_TO_GRIDS}")
    
    # Dictionaries to store the metrics for each grid, keyed by the filename.
    number_of_buildings = {}
    overvoltage = {}
    undervoltage = {}
    v_min_obs = {}
    v_max_obs = {}
    
    # Iterate through all files in the directory
    for filename in os.listdir(PATH_TO_GRIDS):
        
        # 1. Skip files that don't match the expected name format
        if not filename.endswith("_pwrflw.h5"):
            continue    
        
        filepath = os.path.join(PATH_TO_GRIDS, filename)
        grid_index = filename.split('_')[0]
        
        # 2. Identify the columns that correspond to building buses
        building_bus_ids = get_building_bus_ids(filepath)
        if not building_bus_ids:
            print(f"Skipping {grid_index}: No building buses found.")
            continue
            
        # 3. Retrieve the number of buildings in this grid
        number_of_buildings[grid_index] = len(building_bus_ids)
        
        # 4. Loop through entries and count violations
        voltage_stats = count_voltage_violations(filepath, building_bus_ids)
        if voltage_stats is None:
            print(f"Skipping {grid_index}: Missing voltage data.")
            continue
            
        # Save to our dictionaries using the grid_index as the key
        undervoltage[grid_index] = voltage_stats['undervoltage_count']
        overvoltage[grid_index] = voltage_stats['overvoltage_count']
        v_min_obs[grid_index] = voltage_stats['v_min_observed']
        v_max_obs[grid_index] = voltage_stats['v_max_observed']
        
        # 5. Print statistics about the voltage violation for this grid
        """print_voltage_statistics(
            grid_index, 
            number_of_buildings[grid_index], 
            undervoltage[grid_index], 
            overvoltage[grid_index], 
            voltage_stats
        )"""

    # 6. Generate the scatter plot
    print("\nGenerating scatter plot...")
    plot_voltage_violations_scatter(number_of_buildings, undervoltage, overvoltage, v_min_obs, v_max_obs, PATH_TO_PLOT, PLOT_TITLE, PLOT_FILENAME, PLOT_COLORMAP)


if __name__ == "__main__":
    main()
