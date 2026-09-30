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
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend to avoid Wayland/Qt display warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Disable HDF5 file locking to prevent crashes on shared/cluster filesystems
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# --- CONFIGURATION ---
PATH_TO_GRIDS = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/4th_RUN/4.Power_Flown/"
PATH_TO_PLOT = "/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/Extra_Scripts/Plotting/output/My_figures"
PLOT_TITLE = "Incidence of Voltage Violations by Grid Size - 4th Run all grids"
PLOT_FILENAME = "incidence_of_voltage_violations_by_grid_size_AllGrids_4thRun.png"
PLOT_COLORMAP_UNDER = "YlOrBr" # Colormap for undervoltage (e.g. 'Blues_r', 'Purples_r'. 'YlOrBr')
PLOT_COLORMAP_OVER = "Purples"     # Colormap for overvoltage (e.g. 'Reds', 'Oranges')


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
    #print(f"Building bus IDs:                 {building_columns}")
    #print(f"Total observations (Time x Bld):  {total_observations:,}")
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


def plot_voltage_violations_scatter(number_of_buildings, undervoltage, overvoltage, v_min_obs, v_max_obs, output_dir, title, filename, cmap_under, cmap_over):
    """
    Generates a scatter plot of Voltage Violation Frequency vs Number of Grid Buildings.
    Undervoltage frequency is plotted on the positive Y-axis.
    Overvoltage frequency is plotted on the negative Y-axis.
    Color intensity represents the magnitude of the highest violation.
    """
    building_counts = []
    violation_freqs = []
    extreme_voltages = []
    
    for grid_idx in number_of_buildings.keys():
        n_bld = number_of_buildings[grid_idx]
        total_obs = n_bld * 8760
        
        # Calculate frequencies as percentages
        under_freq = (undervoltage[grid_idx] / total_obs) * 100.0
        over_freq = (overvoltage[grid_idx] / total_obs) * 100.0
        
        # Use actual voltage extremes directly (defaulting to healthy 1.0 if missing)
        v_min_val = v_min_obs[grid_idx] if not np.isnan(v_min_obs[grid_idx]) else 1.0
        v_max_val = v_max_obs[grid_idx] if not np.isnan(v_max_obs[grid_idx]) else 1.0
        
        # Add overvoltage point (positive Y)
        building_counts.append(n_bld)
        violation_freqs.append(over_freq)
        extreme_voltages.append(v_max_val)
        
        # Add undervoltage point (negative Y)
        building_counts.append(n_bld)
        violation_freqs.append(-under_freq)
        extreme_voltages.append(v_min_val)
        
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    
    # Determine bounds
    vmin_val = min(extreme_voltages) if extreme_voltages else 0.7
    vmax_val = max(extreme_voltages) if extreme_voltages else 1.2
    if vmin_val >= 0.9: vmin_val = 0.8
    if vmax_val <= 1.1: vmax_val = 1.2
    
    # Create a custom colormap that is completely transparent between 0.9 and 1.1
    c_under = plt.get_cmap(cmap_under)
    c_over = plt.get_cmap(cmap_over)
    p1 = (0.9 - vmin_val) / (vmax_val - vmin_val)
    p2 = (1.1 - vmin_val) / (vmax_val - vmin_val)
    
    positions = []
    colors_list = []
    def with_alpha(color, alpha=0.8):
        return (color[0], color[1], color[2], alpha)

    for i in np.linspace(0, 1, 50):
        positions.append(i * p1)
        colors_list.append(with_alpha(c_under(i)))
        
    positions.append(p1 + 1e-6)
    colors_list.append((1, 1, 1, 0)) # Fully transparent white
    positions.append(p2 - 1e-6)
    colors_list.append((1, 1, 1, 0))
    
    for i in np.linspace(0, 1, 50):
        positions.append(p2 + i * (1 - p2))
        colors_list.append(with_alpha(c_over(i)))
        
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_vv', list(zip(positions, colors_list)))
    
    # Ensure colored points have a solid edge, while healthy points have a faint edge
    edge_colors = ['darkgray' if (v < 0.9 or v > 1.1) else (0.7, 0.7, 0.7, 0.5) for v in extreme_voltages]
    
    # Plot using the custom colormap with transparent deadband
    scatter = ax.scatter(
        building_counts, 
        violation_freqs, 
        c=extreme_voltages, 
        cmap=custom_cmap, 
        edgecolors=edge_colors,
        linewidth=0.5,
        s=40,
        vmin=vmin_val,
        vmax=vmax_val
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Number of Grid Buildings', fontsize=12)
    ax.set_ylabel('Violation Incidence [%]', fontsize=12)
    ax.set_xlim(left=0)
    
    ax.grid(True, which='both', linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a horizontal line at Y=0 to clearly separate under and over voltages
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)
    
    # Add a colorbar to explain the color coding
    cbar = plt.colorbar(scatter, ax=ax)
    
    # Move the label to the left side (between the plot and the colorbar)
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_label('Highest Voltage Violations Observed [p.u.]', fontsize=10, labelpad=5)
    
    # Ensure the absolute minimum and maximum observed voltages are explicitly ticked
    standard_ticks = np.arange(np.ceil(vmin_val * 10) / 10, np.floor(vmax_val * 10) / 10 + 0.05, 0.1)
    all_ticks = sorted(list(set(np.round(standard_ticks, 2)) | {round(vmin_val, 3), 1.0, round(vmax_val, 3)}))
    cbar.set_ticks(all_ticks)
    
    # Calculate violation percentages
    total_grids = len(number_of_buildings)
    grids_with_under = sum(1 for idx in number_of_buildings.keys() if undervoltage[idx] > 0)
    grids_with_over = sum(1 for idx in number_of_buildings.keys() if overvoltage[idx] > 0)
    pct_under = (grids_with_under / total_grids) * 100.0 if total_grids > 0 else 0
    pct_over = (grids_with_over / total_grids) * 100.0 if total_grids > 0 else 0
    
    caption = (
        "The bubbles in the positive pane represent the incidence of overvoltage violations (when the voltage magnitude rises above 1.1 p.u.) "
        "and those in the negative pane represent the incidence of undervoltage violations (when the voltage magnitude drops below 0.9 p.u.). "
        "The 'Violation Incidence' is calculated as the count of voltage violations for all building buses "
        "divided by (quantity of buildings in the grid × total quantity of time stamps). "
        f"{pct_under:.1f}% of the grids violate < 0.9 p.u. and {pct_over:.1f}% violate > 1.1 p.u."
    )
    
    # Add the text below the x-axis
    fig.text(0.05, -0.05, caption, ha='left', va='top', fontsize=9, color='#555555', wrap=True)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    # Use bbox_inches='tight' so the text below the plot isn't cut off when saved
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
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
    plot_voltage_violations_scatter(number_of_buildings, undervoltage, overvoltage, v_min_obs, v_max_obs, PATH_TO_PLOT, PLOT_TITLE, PLOT_FILENAME, PLOT_COLORMAP_UNDER, PLOT_COLORMAP_OVER)


if __name__ == "__main__":
    main()
