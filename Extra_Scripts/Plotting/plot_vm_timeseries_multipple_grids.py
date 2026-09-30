"""
Plot Average Voltage Magnitude Timeseries for Multiple Grids (Binned by quantity of buildings)

This script scans a directory of GridExpand / SurroGrid HDF5 (.h5) files,
extracts the voltage magnitudes for all building buses, dynamically groups the grids into 
5 bins based on the number of buildings they contain, and plots 
the resulting timeseries and Load Duration Curves (LDC) over a year.

The plot includes:
- A solid horizontal line at 1.0 p.u. (ideal voltage)
- Dashed horizontal lines at 0.9 p.u. and 1.1 p.u. (voltage limits)
- 68% and 96% Percentile bands showing spatial distribution within the bin
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
PATH_TO_GRIDS = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/4th_RUN/4.Power_Flown/"
PATH_TO_PLOT = "/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/Extra_Scripts/Plotting/output/My_figures"
PLOT_TITLE = "Voltage Magnitude Time series, grouped by grid size - 4th Run All Grids"
PLOT_FILENAME = "vm_timeseries_4th_run_all_grids.png"
PLOT_COLOR = "purple" 


def extract_building_vm_for_grid(filepath):
    """
    Helper function to load the voltage magnitude data for building buses of a single grid.
    Returns the dataframe of building buses voltage magnitudes.
    """
    buildings_key = '/raw_data/buildings'
    
    with pd.HDFStore(filepath, mode='r') as store:
        if buildings_key not in store:
            print(f"Skipping {filepath}: Buildings key not found.")
            return None
        buildings_df = store[buildings_key]
        building_bus_ids = buildings_df['bus'].tolist()
        
        vm_key = '/pwrflw/output/post/vm'
        if vm_key not in store:
            print(f"Skipping {filepath}: Voltage data not found.")
            return None
                
        vm_all = store[vm_key]
        
    building_columns = [col for col in vm_all.columns if int(col) in building_bus_ids]
    if not building_columns:
        print(f"Skipping {filepath}: No matching building columns found.")
        return None
        
    # To ensure column names don't conflict across grids, prefix with grid name
    grid_index = os.path.basename(filepath).split('_')[0]
    vm_buildings = vm_all[building_columns].copy()
    vm_buildings.columns = [f"{grid_index}_{col}" for col in building_columns]
    
    return vm_buildings


def compute_binned_grid_vm_statistics(grids_dir, num_bins=5):
    """
    Scans the directory for all _pwrflw.h5 grids, extracts building voltages, 
    bins the grids by the number of buildings, and computes multi-grid spatial 
    statistics for each bin for each timestep.
    """
    grid_dfs = []
    
    print(f"Scanning grids in: {grids_dir}")
    for filename in os.listdir(grids_dir):
        if not filename.endswith("_pwrflw.h5"):
            continue
            
        filepath = os.path.join(grids_dir, filename)
        vm_df = extract_building_vm_for_grid(filepath)
        if vm_df is not None:
            num_buildings = len(vm_df.columns)
            grid_dfs.append((filename, vm_df, num_buildings))
            print(f"Added {filename} ({num_buildings} building buses).")
            
    if not grid_dfs:
        print("Error: No valid grids found to process.")
        return None
        
    # Determine dynamic min and max for binning
    min_b = min([x[2] for x in grid_dfs])
    max_b = max([x[2] for x in grid_dfs])
    print(f"Grid sizes range from {min_b} to {max_b} buildings.")
    
    # Create bin edges (num_bins + 1 edges)
    bin_edges = np.linspace(min_b, max_b, num_bins + 1)
    
    binned_stats = []
    
    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i+1]
        
        # Inclusive on low, exclusive on high (except last bin which is inclusive)
        if i == num_bins - 1:
            bin_grids = [x for x in grid_dfs if low <= x[2] <= high]
            bin_label = f"{int(low)} to {int(high)} bldngs"
        else:
            bin_grids = [x for x in grid_dfs if low <= x[2] < high]
            bin_label = f"{int(low)} to {int(high)-1} bldngs"
            
        print(f"\nBin {i+1} ({bin_label}): {len(bin_grids)} grids found.")
        
        if not bin_grids:
            print(f"Warning: Bin {i+1} is empty.")
            binned_stats.append((bin_label, None, 0))
            continue
            
        # Concatenate dataframes for this bin horizontally
        dfs_to_concat = [x[1] for x in bin_grids]
        combined_vm = pd.concat(dfs_to_concat, axis=1)
        
        print(f"Computing statistics for {len(combined_vm.columns)} total building buses in this bin...")
        vm_stats = pd.DataFrame(index=combined_vm.index)
        vm_stats['mean'] = combined_vm.mean(axis=1)
        vm_stats['p16'] = combined_vm.quantile(0.16, axis=1)
        vm_stats['p84'] = combined_vm.quantile(0.84, axis=1)
        vm_stats['p02'] = combined_vm.quantile(0.02, axis=1)
        vm_stats['p98'] = combined_vm.quantile(0.98, axis=1)
        
        binned_stats.append((bin_label, vm_stats, len(bin_grids)))
        
    return binned_stats


def plot_binned_voltage_magnitude_timeseries(binned_stats, output_dir, title, filename, color):
    """
    Generates a 5x2 grid of time series plots and LDCs for the binned voltage data.
    """
    import matplotlib.ticker as mtick
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_bins = len(binned_stats)
    
    # Pre-calculate limits to determine proportional height ratios
    row_limits = []
    for bin_label, vm_stats, num_grids in binned_stats:
        if vm_stats is None:
            row_limits.append((0.89, 1.11))
        else:
            row_limits.append((min(0.89, vm_stats['p02'].min() - 0.01), max(1.11, vm_stats['p98'].max() + 0.01)))
            
    height_ratios = [ymax - ymin for ymin, ymax in row_limits]
    total_range = sum(height_ratios)
    fig_height = max(8.0, total_range * 15.0) # 15 inches per 1.0 p.u. range
    
    fig, axes = plt.subplots(num_bins, 2, figsize=(16, fig_height), gridspec_kw={'width_ratios': [2, 1], 'height_ratios': height_ratios})
    fig.suptitle(title, fontsize=18, y=0.985)
    
    # Add shared vertical axis label for bins on the far left, shifted left to increase space
    fig.supylabel("Average Voltage Magnitude [p.u.]", fontsize=16, x=0.01, fontweight='bold')
    
    # Set up consistent x-axis data structures
    time_index = pd.date_range(start="2023-01-01 00:00", periods=8760, freq="h")
    x_pct = np.linspace(0, 100, 8760)
            
    for row, (bin_label, vm_stats, num_grids) in enumerate(binned_stats):
        ax1 = axes[row, 0]
        ax2 = axes[row, 1]
        
        if num_grids > 0:
            ylabel_text = f"{num_grids} grids with\n{bin_label}"
        else:
            ylabel_text = bin_label
        
        if vm_stats is None:
            ax1.text(0.5, 0.5, "No grids in this bin", ha='center', va='center', fontsize=12)
            ax2.text(0.5, 0.5, "No grids in this bin", ha='center', va='center', fontsize=12)
            ax1.set_ylabel(ylabel_text, fontsize=12, fontweight='bold', labelpad=15)
            ax1.set_yticks([])
            ax1.set_xticks([])
            ax2.set_yticks([])
            ax2.set_xticks([])
            continue
            
        vm_stats.index = time_index
        n_timesteps = len(vm_stats)
        
        # Calculate timestamps with violations using the hourly mean
        under_count = (vm_stats['mean'] < 0.9).sum()
        over_count = (vm_stats['mean'] > 1.1).sum()
        under_pct = (under_count / n_timesteps) * 100
        over_pct = (over_count / n_timesteps) * 100
        
        violation_text = f"Violations: <0.9 p.u. ({under_pct:.1f}%) | >1.1 p.u. ({over_pct:.1f}%)"
        
        row_ymin, row_ymax = row_limits[row]
        
        yticks = [0.9, 1.0, 1.1]
        if row_ymin < 0.85:
            curr = 0.8
            while curr > row_ymin:
                yticks.append(round(curr, 1))
                curr -= 0.1
        if row_ymax > 1.15:
            curr = 1.2
            while curr < row_ymax:
                yticks.append(round(curr, 1))
                curr += 0.1
        yticks = sorted(list(set(yticks)))
        
        # ==========================================
        # LEFT PANEL: Time Series Plot
        # ==========================================
        aggregated_vm = vm_stats.resample('24h').mean()
        
        ax1.fill_between(aggregated_vm.index, aggregated_vm['p02'], aggregated_vm['p98'], color=color, alpha=0.15, linewidth=0)
        ax1.fill_between(aggregated_vm.index, aggregated_vm['p16'], aggregated_vm['p84'], color=color, alpha=0.35, linewidth=0)
        ax1.plot(aggregated_vm.index, aggregated_vm['mean'], color=color, linewidth=2)
        
        ax1.axhline(1.0, color='black', linestyle='-', linewidth=1.5, zorder=3)
        ax1.axhline(1.1, color='gray', linestyle='--', linewidth=1.2, zorder=3)
        ax1.axhline(0.9, color='gray', linestyle='--', linewidth=1.2, zorder=3)
        
        ax1.set_xlim(aggregated_vm.index.min(), aggregated_vm.index.max())
        ax1.set_ylim(row_ymin, row_ymax)
        ax1.set_yticks(yticks)
        
        ax1.set_ylabel(ylabel_text, fontsize=12, fontweight='bold', labelpad=15)
        
        # Add violation text inside the plot
        ax1.text(0.01, 0.04, violation_text, transform=ax1.transAxes, 
                 ha='left', va='bottom', fontsize=11, 
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.4'))
        
        ax1.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
        ax1.grid(True, which='minor', color='gray', linestyle=':', alpha=0.1)
        
        # Only show bottom X-axis labels for timeseries
        if row == num_bins - 1:
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        else:
            ax1.set_xticklabels([])
            
        # Add legend only to the first row to reduce clutter
        if row == 0:
            ax1.plot([], [], color=color, linewidth=2, label="Expected Timeseries (24h Agg.)")
            ax1.fill_between([], [], [], color=color, alpha=0.35, label="68% Band")
            ax1.fill_between([], [], [], color=color, alpha=0.15, label="96% Band")
            ax1.legend(loc='upper right', fontsize=10)

        # ==========================================
        # RIGHT PANEL: Load Duration Curve (LDC)
        # ==========================================
        ldc_mean = np.sort(vm_stats['mean'])[::-1]
        ldc_p16 = np.sort(vm_stats['p16'])[::-1]
        ldc_p84 = np.sort(vm_stats['p84'])[::-1]
        ldc_p02 = np.sort(vm_stats['p02'])[::-1]
        ldc_p98 = np.sort(vm_stats['p98'])[::-1]
        
        ax2.fill_between(x_pct, ldc_p02, ldc_p98, color=color, alpha=0.15, linewidth=0)
        ax2.fill_between(x_pct, ldc_p16, ldc_p84, color=color, alpha=0.35, linewidth=0)
        ax2.plot(x_pct, ldc_mean, color=color, linewidth=2)
        
        ax2.axhline(1.0, color='black', linestyle='-', linewidth=1.5, zorder=3)
        ax2.axhline(1.1, color='gray', linestyle='--', linewidth=1.2, zorder=3)
        ax2.axhline(0.9, color='gray', linestyle='--', linewidth=1.2, zorder=3)
        
        ax2.set_xlim(0, 100)
        ax2.set_ylim(row_ymin, row_ymax)
        ax2.set_yticks(yticks)
        ax2.tick_params(labelleft=False)  # Remove y-axis numbers
        
        ax2.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
        ax2.grid(True, which='minor', color='gray', linestyle=':', alpha=0.1)
        
        # Only show bottom X-axis labels for LDC
        if row == num_bins - 1:
            ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
        else:
            ax2.set_xticklabels([])
            
        # Add legend only to the first row to reduce clutter
        if row == 0:
            ax2.plot([], [], color=color, linewidth=2, label="Expected LDC (Hourly)")
            ax2.fill_between([], [], [], color=color, alpha=0.35, label="68% Band")
            ax2.fill_between([], [], [], color=color, alpha=0.15, label="96% Band")
            ax2.legend(loc='upper right', fontsize=10)

    # ==========================================
    # CAPTION & SAVING
    # ==========================================
    caption = (
        "Note: The dataset of grids has been divided into 5 bins based on the number of building buses.\n"
        "At each hourly timestep, a spatial average of the voltage across the building buses within each bin was computed.\n"
        "For visual clarity, the Time Series plots (left) aggregate these hourly averages into 24-hour periods,\n"
        "while the Load Duration Curves (right) display the raw sorted hourly values."
    )
    
    # Remove tight_layout completely to gain absolute manual control over spacing
    plt.subplots_adjust(top=0.96, bottom=0.08, left=0.12, right=0.98, hspace=0.1, wspace=0.05)
    
    # Anchor caption nicely using the last (bottom-left) plot's bounding box
    fig.text(0.1, 0.01, caption, ha='left', va='bottom', fontsize=12, color='#333333')
    
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccessfully saved 5x2 time series plot to: {out_path}")


def main():
    if not os.path.exists(PATH_TO_GRIDS):
        print(f"Warning: The directory '{PATH_TO_GRIDS}' does not exist.") 
        return
        
    binned_stats = compute_binned_grid_vm_statistics(PATH_TO_GRIDS, num_bins=5)
    if binned_stats is None:
        print("Stopping execution.")
        return
        
    print("\nGenerating plot...")
    plot_binned_voltage_magnitude_timeseries(
        binned_stats=binned_stats,
        output_dir=PATH_TO_PLOT,
        title=PLOT_TITLE,
        filename=PLOT_FILENAME,
        color=PLOT_COLOR
    )

if __name__ == "__main__":
    main()
