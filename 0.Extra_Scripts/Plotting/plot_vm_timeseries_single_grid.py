"""
Plot Average Voltage Magnitude Timeseries

This script opens a specified GridExpand / SurroGrid HDF5 (.h5) file,
extracts the voltage magnitudes for all building buses, calculates the 
hourly average and percentile statistics across these buses, and plots 
the resulting timeseries over a year (8760 timesteps).

The plot includes:
- A solid horizontal line at 1.0 p.u. (ideal voltage)
- Dashed horizontal lines at 0.9 p.u. and 1.1 p.u. (voltage limits)
- X-axis formatted to show months
- 68% and 96% Percentile bands showing spatial distribution
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend to avoid Wayland/Qt display warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Disable HDF5 file locking to prevent crashes on shared/cluster filesystems
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# --- CONFIGURATION ---
PATH_TO_GRID = "Extra_Scripts/Plotting/output/0117_N2842500E4300500_82194_6_1_PV100_HP100_EV100_VarTar0_CapPr0_pwrflw.h5"
PATH_TO_PLOT = "Extra_Scripts/Plotting"
PLOT_TITLE = "Average voltage magnitude in building buses for each time stamp"
PLOT_FILENAME = "average_voltage_magnitude_timeseries.png"
PLOT_COLOR = "#1f77b4"  # Default matplotlib blue color


def get_building_bus_ids(filepath):
    """
    Opens the HDF5 file and identifies the columns (bus IDs) that correspond to buildings.
    
    Args:
        filepath (str): Path to the HDF5 file.
        
    Returns:
        list: A list of bus IDs (integers) that have buildings connected.
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


def compute_building_vm_statistics(filepath, building_bus_ids):
    """
    Loads the voltage magnitude data ('/pwrflw/output/post/vm') for the specific building buses and 
    computes the average voltage magnitude and percentiles for each timestep.
    
    Args:
        filepath (str): Path to the HDF5 file.
        building_bus_ids (list): List of bus IDs representing buildings.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the 'mean', 'p16', 'p84', 'p02', and 'p98' 
                      voltage statistics per timestep.
    """
    # Try post-expansion results first, then fallback to pre-expansion
    vm_key = '/pwrflw/output/post/vm'
    
    with pd.HDFStore(filepath, mode='r') as store:
        if vm_key not in store:
            print(f"Error: Voltage data not found in {filepath}")
            return None
            
        vm_all = store[vm_key]
        
    # Filter the voltage table to only include columns for building buses
    building_columns = [col for col in vm_all.columns if int(col) in building_bus_ids]
    
    if not building_columns:
        print("Error: No matching building columns found in the voltage data.")
        return None
        
    vm_buildings = vm_all[building_columns]
    
    # Calculate statistics across all building buses for each timestep
    vm_stats = pd.DataFrame(index=vm_buildings.index)
    vm_stats['mean'] = vm_buildings.mean(axis=1)
    vm_stats['p16'] = vm_buildings.quantile(0.16, axis=1)
    vm_stats['p84'] = vm_buildings.quantile(0.84, axis=1)
    vm_stats['p02'] = vm_buildings.quantile(0.02, axis=1)
    vm_stats['p98'] = vm_buildings.quantile(0.98, axis=1)
    
    return vm_stats


def plot_voltage_magnitude_timeseries(vm_stats, output_dir, title, filename, color):
    """
    Generates a time series plot and a Load Duration Curve (LDC) of the average voltage magnitude with percentile bands.
    
    Args:
        vm_stats (pd.DataFrame): The voltage statistics data per timestep.
        output_dir (str): Directory to save the plot.
        title (str): Title of the plot.
        filename (str): Name of the output image file.
        color (str): Color of the plotted line and bands.
    """
    import matplotlib.ticker as mtick
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 8760 timesteps corresponds to 1 year of hourly data
    n_timesteps = len(vm_stats)
    
    # Create a datetime index for the x-axis starting from Jan 1st (using a non-leap year like 2023)
    time_index = pd.date_range(start="2023-01-01 00:00", periods=n_timesteps, freq="h")
    vm_stats.index = time_index
    
    # Create the 1x2 plot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==========================================
    # LEFT PANEL: Time Series Plot
    # ==========================================
    
    # Aggregate 24 hours for each point to be plotted
    aggregated_vm = vm_stats.resample('24h').mean()
    
    # Plot 96% Percentile Band (2nd to 98th percentile)
    ax1.fill_between(
        aggregated_vm.index, 
        aggregated_vm['p02'], 
        aggregated_vm['p98'], 
        color=color, alpha=0.15, label="96% Percentile Band", linewidth=0
    )
    
    # Plot 68% Percentile Band (16th to 84th percentile)
    ax1.fill_between(
        aggregated_vm.index, 
        aggregated_vm['p16'], 
        aggregated_vm['p84'], 
        color=color, alpha=0.35, label="68% Percentile Band", linewidth=0
    )
    
    # Plot the aggregated expected average line
    ax1.plot(
        aggregated_vm.index, 
        aggregated_vm['mean'], 
        color=color, linewidth=2, label="Expected Timeseries (24-Hour Agg.)"
    )
    
    # Plot horizontal reference lines
    ax1.axhline(1.0, color='black', linestyle='-', linewidth=1.5, zorder=3)
    ax1.axhline(1.1, color='gray', linestyle='--', linewidth=1.2, zorder=3)
    ax1.axhline(0.9, color='gray', linestyle='--', linewidth=1.2, zorder=3)
    
    # Format the x-axis to show month abbreviations (Jan, Feb, Mar, etc.)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Set labels and title
    ax1.set_title(title, fontsize=14, pad=15)
    ax1.set_ylabel('Average Voltage Magnitude [p.u.]', fontsize=12)
    
    # Configure the y-axis limits and ticks
    ymin = min(0.85, vm_stats['p02'].min() - 0.05)
    ymax = max(1.15, vm_stats['p98'].max() + 0.05)
    ax1.set_ylim(ymin, ymax)
    ax1.set_yticks([0.9, 1.0, 1.1])
    
    # Enhance grid visibility
    ax1.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
    ax1.grid(True, which='minor', color='gray', linestyle=':', alpha=0.1)
    
    # Add a legend
    ax1.legend(loc='upper right')

    # ==========================================
    # RIGHT PANEL: Load Duration Curve (LDC)
    # ==========================================
    
    # For a duration curve, we sort each statistical series independently in descending order
    ldc_mean = np.sort(vm_stats['mean'])[::-1]
    ldc_p16 = np.sort(vm_stats['p16'])[::-1]
    ldc_p84 = np.sort(vm_stats['p84'])[::-1]
    ldc_p02 = np.sort(vm_stats['p02'])[::-1]
    ldc_p98 = np.sort(vm_stats['p98'])[::-1]
    
    # Create x-axis percentage array (0 to 100%)
    x_pct = np.linspace(0, 100, n_timesteps)
    
    ax2.fill_between(
        x_pct, 
        ldc_p02, 
        ldc_p98, 
        color=color, alpha=0.15, label="96% Percentile Band (≈2σ)", linewidth=0
    )
    ax2.fill_between(
        x_pct, 
        ldc_p16, 
        ldc_p84, 
        color=color, alpha=0.35, label="68% Percentile Band (≈1σ)", linewidth=0
    )
    ax2.plot(
        x_pct, 
        ldc_mean, 
        color=color, linewidth=2, label="Expected LDC (Hourly)"
    )
    
    # Horizontal reference lines
    ax2.axhline(1.0, color='black', linestyle='-', linewidth=1.5, zorder=3)
    ax2.axhline(1.1, color='gray', linestyle='--', linewidth=1.2, zorder=3)
    ax2.axhline(0.9, color='gray', linestyle='--', linewidth=1.2, zorder=3)
    
    # Configure axes
    ax2.set_title(title + " (LDC)", fontsize=14, pad=15)
    ax2.set_ylabel('Average Voltage Magnitude [p.u.]', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylim(ymin, ymax)
    ax2.set_yticks([0.9, 1.0, 1.1])
    ax2.set_xlim(0, 100)
    
    # Format x-axis as percentage
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
    
    ax2.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
    ax2.grid(True, which='minor', color='gray', linestyle=':', alpha=0.1)
    ax2.legend(loc='upper right')

    # ==========================================
    # CAPTION & SAVING
    # ==========================================
    
    # Calculate timestamps with violations and their percentages using the original hourly mean
    under_count = (vm_stats['mean'] < 0.9).sum()
    over_count = (vm_stats['mean'] > 1.1).sum()
    under_pct = (under_count / n_timesteps) * 100
    over_pct = (over_count / n_timesteps) * 100
    
    # Add caption explaining the two averages (grid-wide spatial average vs 24h temporal average)
    caption = (
        f"At each hourly timestep, the average of the voltage across all building grids was computed.\n"
        f"This grid-wide hourly average fell below 0.9 p.u. for {under_count} timestamps ({under_pct:.2f}%) and exceeded 1.1 p.u. for {over_count} timestamps ({over_pct:.2f}%).\n"
        f"Note: For visual clarity, the Time Series plot (left) additionally aggregates these hourly averages into 24-hour periods (temporal average), "
        f"while the Load Duration Curve (right) displays the raw sorted hourly values."
    )
    
    # Ensure layout fits well before adding the figure text
    plt.tight_layout()
    
    # Add the text below the plots. The bbox_inches='tight' in savefig will ensure it is kept.
    fig.text(0.05, -0.05, caption, ha='left', va='top', fontsize=11, color='#333333')
    
    # Save the plot
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved time series plot to: {out_path}")


def main():
    # 1. Check if the grid file exists
    if not os.path.exists(PATH_TO_GRID):
        print(f"Warning: The file '{PATH_TO_GRID}' does not exist.") 
        return
        
    # 2. Identify which columns correspond to building buses
    print("Extracting building buses...")
    building_bus_ids = get_building_bus_ids(PATH_TO_GRID)
    if not building_bus_ids:
        print("Stopping execution: No building buses found.")
        return
        
    print(f"Found {len(building_bus_ids)} building buses.")
    
    # 3. Calculate the voltage magnitude statistics for each timestep
    print("Computing voltage magnitude statistics across building buses for each timestep...")
    vm_stats = compute_building_vm_statistics(PATH_TO_GRID, building_bus_ids)
    if vm_stats is None:
        print("Stopping execution: Could not compute voltage magnitude statistics.")
        return
        
    # 4. Generate the plot
    print("Generating plot...")
    plot_voltage_magnitude_timeseries(
        vm_stats=vm_stats,
        output_dir=PATH_TO_PLOT,
        title=PLOT_TITLE,
        filename=PLOT_FILENAME,
        color=PLOT_COLOR
    )

if __name__ == "__main__":
    main()
