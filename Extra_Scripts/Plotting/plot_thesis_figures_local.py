"""Plotting Thesis Figures from MA_Elias.

This Python script is designed to run in a headless HPC environment (compute nodes)
to generate Figure 4.2 b) and Figure 4.1 from Elias's Master Thesis.
It saves the plots directly to the GridForecast/0_preprocessing directory as PNG image files.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib
# CRITICAL: Use the 'Agg' backend. This allows matplotlib to generate and save
# plots in the background without needing a graphical interface (display).
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Disable HDF5 file locking. This is necessary because HDF5 files on cluster network
# mounts (like LRZ's DSS storage) can lock up if multiple processes attempt to read them.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


def get_pwrflw_files(pwrflw_dir):
    """
    Scans the given directory for HDF5 output files ending with 'pwrflw.h5'.
    If none are found in the primary directory, it falls back to a local folder.
    """
    # Look for files matching the pattern: <pwrflw_dir>/*.h5
    all_files = glob.glob(os.path.join(pwrflw_dir, '*.h5'))
    h5_files = sorted([f for f in all_files if f.endswith(expected_ending)])
    
    # If no files are found, try local fallback directory
    if not h5_files:
        print(f"No files found in primary directory ending with {expected_ending}. Checking local fallback...")
        fallback_files = glob.glob(os.path.join('../../local_destination', '*.h5'))
        h5_files = sorted([f for f in fallback_files if f.endswith(expected_ending)])
        
    return h5_files


def plot_figure_4_2_a(pwrflw_dir, out_dir, preurbs=False):
    """
    Generates Figure 4.2 a): Scatter plot of relative peak load change vs grid building number.
    """
    h5_files = get_pwrflw_files(pwrflw_dir)
    if not h5_files:
        print("Error: No HDF5 power flow files ending in pwrflw.h5 found!")
        return
        
    case_label = "Pre-Urbs" if preurbs else "Post-Urbs"
    file_suffix = "_pre" if preurbs else ""
    input_key = 'pwrflw/input/demand_pre' if preurbs else 'pwrflw/input/demand_post'
    output_key = 'pwrflw/output/pre/demand_import' if preurbs else 'pwrflw/output/post/demand_import'
    color = 'darkorange' if preurbs else 'steelblue'
        
    print(f"Scanning {len(h5_files)} HDF5 files for peak load data ({case_label})...")
    
    building_counts = []
    rel_changes = []
    processed_count = 0
    
    for filepath in h5_files:
        try:
            with pd.HDFStore(filepath, 'r') as store:
                if input_key in store and output_key in store and '/raw_data/buildings' in store:
                    # Get the correct number of buildings from the raw_data table
                    num_buildings = len(store['/raw_data/buildings'])
                    
                    df_in = store[input_key]
                    df_out = store[output_key]
                    
                    p_out = df_out['p_mw'].values * 1000.0
                    
                    if isinstance(df_in.columns, pd.MultiIndex):
                        if 'electricity' in df_in.columns.get_level_values(1):
                            p_in = df_in.xs('electricity', level=1, axis=1).sum(axis=1).values
                        else:
                            p_in = df_in.sum(axis=1).values
                    else:
                        p_cols = [c for c in df_in.columns if 'reactive' not in str(c).lower()]
                        p_in = df_in[p_cols].sum(axis=1).values
                        
                    peak_in_overall = np.max(np.abs(p_in))
                    # To avoid exploding ratios due to near-zero loads and baseline transformer losses,
                    # we only consider timestamps where the load is at least 10% of the peak load.
                    valid_mask = np.abs(p_in) > (0.10 * peak_in_overall)

                    import_mask = (p_in > 0) & valid_mask
                    feedin_mask = (p_in < 0) & valid_mask

                    # Compute relative change per timestamp: (p_out/p_in - 1) * 100
                    # For import (p_in > 0), losses make p_out > p_in -> Positive percentage
                    # For feed-in (p_in < 0), losses make abs(p_out) < abs(p_in) -> Negative percentage
                    
                    if np.any(import_mask):
                        rel_import = (p_out[import_mask] / p_in[import_mask] - 1.0) * 100.0
                        max_import_rel = np.max(rel_import)
                        
                        if max_import_rel > 400:
                            print(f"\n[OUTLIER DETECTED AND EXCLUDED (IMPORT)]")
                            print(f"File: {os.path.basename(filepath)}")
                            print(f"Buildings: {num_buildings} | Relative Change: {max_import_rel:.2f}%\n")
                        else:
                            building_counts.append(num_buildings)
                            rel_changes.append(max_import_rel)
                            
                    if np.any(feedin_mask):
                        rel_feedin = (p_out[feedin_mask] / p_in[feedin_mask] - 1.0) * 100.0
                        # The largest deviation for feed-in creates the most negative relative change
                        max_feedin_rel = np.min(rel_feedin)
                        
                        if max_feedin_rel < -400:
                            print(f"\n[OUTLIER DETECTED AND EXCLUDED (FEED-IN)]")
                            print(f"File: {os.path.basename(filepath)}")
                            print(f"Buildings: {num_buildings} | Relative Change: {max_feedin_rel:.2f}%\n")
                        else:
                            building_counts.append(num_buildings)
                            rel_changes.append(max_feedin_rel)
                            
                    processed_count += 1
                        
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")
            
    if not building_counts:
        print("No peak load data could be loaded. Skipping Figure 4.2 a).")
        return
        
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    ax.scatter(building_counts, rel_changes, color=color, alpha=0.5, edgecolor='none', s=40)
    
    ax.set_xlabel('Number of Grid Buildings', fontsize=12)
    ax.set_ylabel('Rel. Peak Load Change [%]', fontsize=12)
    ax.set_xlim(left=0)
    
    ax.grid(True, which='both', linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add 'a)' textbox
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="darkgray", lw=1, alpha=0.8)
    ax.text(0.02, 0.95, "a)", transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', bbox=bbox_props)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'figure_4_2_a{file_suffix}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Successfully saved Figure 4.2 a) ({case_label}) to: {out_path}")


def plot_figure_4_2_b(pwrflw_dir, out_dir, preurbs=False):
    """
    Generates Figure 4.2 b): Histogram of Grid-Wide Maximum Upper and Lower Voltage Deviations.
    
    This function reads the voltage magnitudes from the power flow output files (`_pwrflw.h5`)
    under the key `/pwrflw/output/post/vm` (or `/pwrflw/output/pre/vm` if preurbs=True) and
    plots the histogram of grid-wide maximum and minimum voltage magnitudes over a year.
    """
    h5_files = get_pwrflw_files(pwrflw_dir)
    if not h5_files:
        print("Error: No HDF5 power flow files ending in pwrflw.h5 found!")
        return
        
    vm_key = '/pwrflw/output/pre/vm' if preurbs else '/pwrflw/output/post/vm'
    case_label = "Pre-Urbs" if preurbs else "Post-Urbs"
    file_suffix = "_pre" if preurbs else ""
    
    print(f"Scanning {len(h5_files)} HDF5 files for voltage data ({case_label})...")
    
    max_voltages = []
    min_voltages = []
    processed_count = 0
    
    # Loop through each scenario file and extract min/max voltages
    for filepath in h5_files:
        try:
            # pd.HDFStore allows us to open and read specific tables within the HDF5 file
            with pd.HDFStore(filepath, 'r') as store:
                # Check if the voltage dataset key exists in the file
                if vm_key in store:
                    vm = store[vm_key]
                    # Append the maximum and minimum voltage magnitudes to our lists
                    max_voltages.append(vm.values.max())
                    min_voltages.append(vm.values.min())
                    processed_count += 1
                    print(f"Successfully processed: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")
            
    print(f"Finished processing! Total files successfully processed: {processed_count}")
    
    if not max_voltages:
        print(f"No voltage data could be loaded for {case_label}. Skipping Figure 4.2 b).")
        return
        
    # Convert lists to numpy arrays for mathematical calculations
    max_voltages = np.array(max_voltages)
    min_voltages = np.array(min_voltages)
    
    total_grids = len(max_voltages)
    grids_under_09 = np.sum(min_voltages < 0.9)
    grids_over_11 = np.sum(max_voltages > 1.1)
    
    # Calculate percentage of grids violating voltage bounds (0.9 to 1.1 p.u.)
    pct_under_09 = (grids_under_09 / total_grids) * 100.0
    pct_over_11 = (grids_over_11 / total_grids) * 100.0
    
    # Print stats to the output logs
    print(f"--- Stats for Figure 4.2 b) ({case_label}) ---")
    print(f"Total Grids: {total_grids}")
    print(f"Min voltage across all grids: {min_voltages.min():.3f} p.u.")
    print(f"Max voltage across all grids: {max_voltages.max():.3f} p.u.")
    print(f"Grids with voltage < 0.9 p.u.: {grids_under_09} ({pct_under_09:.2f}%)")
    print(f"Grids with voltage > 1.1 p.u.: {grids_over_11} ({pct_over_11:.2f}%)")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    bins = np.arange(0.64, 1.261, 0.02)
    
    # Set colors
    if preurbs:
        c1 = 'darkorange'
        c2 = 'plum'
    else:
        c1 = 'steelblue'
        c2 = 'mediumaquamarine'

    # Plot histograms for both upper and lower deviations
    ax.hist(max_voltages, bins=bins, color=c1, edgecolor='white', alpha=0.9, label=f'Max. Upper Voltage Deviation ({case_label})')
    ax.hist(min_voltages, bins=bins, color=c2, edgecolor='white', alpha=0.9, label=f'Max. Lower Voltage Deviation ({case_label})')
    
    # Set y-axis to logarithmic scale (since most grids are well within limits, violations are rare)
    ax.set_yscale('log')
    ax.set_xlim(0.65, 1.25)
    ax.set_ylim(0.8, 1000)
    
    ax.set_xlabel('Voltage Magnitude (p.u.)', fontsize=12)
    ax.set_ylabel('LV Grid Count', fontsize=12)
    
    ax.grid(True, which='both', linestyle='-', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Draw vertical dashed lines at the 0.9 and 1.1 p.u. limits
    ax.axvline(0.9, color='black', linestyle='--', linewidth=1.2)
    ax.axvline(1.1, color='black', linestyle='--', linewidth=1.2)
    
    # Add textbox labels showing percentages
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="darkgray", lw=1, alpha=0.8)
    ax.text(0.81, 30, f"< 0.9 p.u.: {pct_under_09:.1f}%", ha="center", va="center", size=10, bbox=bbox_props)
    ax.text(1.17, 30, f"> 1.1 p.u.: {pct_over_11:.1f}%", ha="center", va="center", size=10, bbox=bbox_props)
    
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    # Save the plot
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'figure_4_2_b{file_suffix}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Successfully saved Figure 4.2 b) ({case_label}) to: {out_path}")



def plot_figure_4_2_b_filtered(pwrflw_dir, out_dir, preurbs=False):
    """
    Generates Figure 4.2 b): Histogram of Grid-Wide Maximum Upper and Lower Voltage Deviations.
    This filtered version drops the buses connected to final buildings before taking max/min voltages.
    """
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    h5_files = get_pwrflw_files(pwrflw_dir)
    if not h5_files:
        print("Error: No HDF5 power flow files ending in pwrflw.h5 found!")
        return
        
    vm_key = '/pwrflw/output/pre/vm' if preurbs else '/pwrflw/output/post/vm'
    case_label = "Pre-Urbs" if preurbs else "Post-Urbs"
    file_suffix = "_pre" if preurbs else ""
    
    print(f"Scanning {len(h5_files)} HDF5 files for voltage data ({case_label}) (Filtered)...")
    
    max_voltages = []
    min_voltages = []
    processed_count = 0
    
    for filepath in h5_files:
        try:
            with pd.HDFStore(filepath, 'r') as store:
                if vm_key in store and '/raw_data/buildings' in store:
                    vm = store[vm_key]
                    buildings = store['/raw_data/buildings']
                    
                    # Extract building bus IDs
                    building_buses = buildings['bus'].values
                    
                    # Drop building bus columns
                    vm_filtered = vm.drop(columns=[col for col in vm.columns if int(col) in building_buses], errors='ignore')
                    # loops through the columns in vm.column (each column is a bus id) and saves those that aslo exist in 
                    # building_buses into the newly defined "columns" variable.These columns are dropeed from vm.

                    if not vm_filtered.empty:
                        max_voltages.append(vm_filtered.values.max())
                        min_voltages.append(vm_filtered.values.min())
                        processed_count += 1
                        print(f"Successfully processed: {os.path.basename(filepath)}")
                    else:
                        print(f"Warning: No buses left after filtering in {filepath}")
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")
            
    print(f"Finished processing! Total files successfully processed: {processed_count}")
    
    if not max_voltages:
        print(f"No voltage data could be loaded for {case_label}. Skipping Figure 4.2 b) Filtered.")
        return
        
    max_voltages = np.array(max_voltages)
    min_voltages = np.array(min_voltages)
    
    total_grids = len(max_voltages)
    grids_under_09 = np.sum(min_voltages < 0.9)
    grids_over_11 = np.sum(max_voltages > 1.1)
    
    pct_under_09 = (grids_under_09 / total_grids) * 100.0
    pct_over_11 = (grids_over_11 / total_grids) * 100.0
    
    print(f"--- Stats for Figure 4.2 b) Filtered ({case_label}) ---")
    print(f"Total Grids: {total_grids}")
    print(f"Min voltage across non-building buses: {min_voltages.min():.3f} p.u.")
    print(f"Max voltage across non-building buses: {max_voltages.max():.3f} p.u.")
    print(f"Grids with voltage < 0.9 p.u.: {grids_under_09} ({pct_under_09:.2f}%)")
    print(f"Grids with voltage > 1.1 p.u.: {grids_over_11} ({pct_over_11:.2f}%)")
    
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    bins = np.arange(0.64, 1.261, 0.02)
    
    if preurbs:
        c1 = 'darkorange'
        c2 = 'plum'
    else:
        c1 = 'steelblue'
        c2 = 'mediumaquamarine'

    ax.hist(max_voltages, bins=bins, color=c1, edgecolor='white', alpha=0.9, label=f'Max. Upper Volts Dev. Filtered ({case_label})')
    ax.hist(min_voltages, bins=bins, color=c2, edgecolor='white', alpha=0.9, label=f'Max. Lower Volts Dev. Filtered ({case_label})')
    
    ax.set_yscale('log')
    ax.set_xlim(0.65, 1.25)
    ax.set_ylim(0.8, 1000)
    
    ax.set_xlabel('Voltage Magnitude (p.u.)', fontsize=12)
    ax.set_ylabel('LV Grid Count', fontsize=12)
    
    ax.grid(True, which='both', linestyle='-', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.axvline(0.9, color='black', linestyle='--', linewidth=1.2)
    ax.axvline(1.1, color='black', linestyle='--', linewidth=1.2)
    
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="darkgray", lw=1, alpha=0.8)
    ax.text(0.81, 30, f"< 0.9 p.u.: {pct_under_09:.1f}%", ha="center", va="center", size=10, bbox=bbox_props)
    ax.text(1.17, 30, f"> 1.1 p.u.: {pct_over_11:.1f}%", ha="center", va="center", size=10, bbox=bbox_props)
    
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'figure_4_2_b_filtered{file_suffix}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Successfully saved Figure 4.2 b) Filtered ({case_label}) to: {out_path}")


def plot_figure_4_1(pwrflw_dir, out_dir, preurbs=False):
    """
    Generates Figure 4.1: Net Transformer Active, Reactive, and Apparent Load distributions.
    
    This function plots the normalized active, reactive, and apparent power time series at
    daily aggregation (left) and the corresponding hourly Load Duration Curves (right)
    across all grids.
    """
    h5_files = get_pwrflw_files(pwrflw_dir)
    if not h5_files:
        print("Error: No HDF5 power flow files ending in pwrflw.h5 found!")
        return
        
    case_label = "Pre-Urbs" if preurbs else "Post-Urbs"
    file_suffix = "_pre" if preurbs else ""
    import_key = 'pwrflw/output/pre/demand_import' if preurbs else 'pwrflw/output/post/demand_import'
        
    print(f"Loading {len(h5_files)} grids from raw forecasting datasets ({case_label})...")
    
    p_series_list = []
    q_series_list = []
    processed_count = 0
    
    # Loop through each scenario file and load active/reactive power time-series
    for i, filepath in enumerate(h5_files):
        try:
            with pd.HDFStore(filepath, 'r') as store:
                if import_key in store:
                    df_import = store[import_key]
                    
                    # Convert MW/MVAr to kW/kVAr
                    p_kw = df_import['p_mw'].values * 1000.0
                    q_kvar = df_import['q_mvar'].values * 1000.0
                    
                    # Create MultiIndex for this batch to keep them structured by grid index
                    hours_len = len(p_kw)
                    index = pd.MultiIndex.from_product([[i], np.arange(hours_len)], names=['batch', 'hour'])
                    
                    p_series_list.append(pd.Series(p_kw, index=index))
                    q_series_list.append(pd.Series(q_kvar, index=index))
                    processed_count += 1
                    print(f"Successfully loaded: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")
            
    print(f"Finished loading! Total files successfully loaded: {processed_count}")
    
    if not p_series_list:
        print("No active/reactive power data could be loaded. Skipping Figure 4.1.")
        return
        
    # Combine all series into single DataFrames
    P = pd.concat(p_series_list)
    Q = pd.concat(q_series_list)
    
    # Apparent power formula: S = sqrt(P^2 + Q^2)
    S = np.sqrt(P**2 + Q**2)
    
    grids = np.arange(processed_count)
    
    # Annual mean apparent power per grid for timeseries normalization
    mean_S_per_grid = S.groupby(level='batch').mean()
    
    # Average maximum apparent power over all grids for Load Duration Curve (LDC) normalization
    max_S_per_grid = S.groupby(level='batch').max()
    norm_ldc_constant = max_S_per_grid.mean()
    print(f"Apparent power LDC normalization constant (mean of grid max S): {norm_ldc_constant:.2f} kVA")
    
    # Setup daily aggregation timeline
    daily_time = pd.date_range(start='2024-01-01', periods=365, freq='D')
    
    ts_data = {'P': [], 'Q': [], 'S': []}
    ldc_data = {'P': [], 'Q': [], 'S': []}
    
    # Compute daily timeseries and sorted hourly LDCs per grid
    for g in grids:
        Pg = P.loc[g]
        Qg = Q.loc[g]
        Sg = S.loc[g]
        mean_Sg = mean_S_per_grid.loc[g]
        
        # 1. Normalize for daily timeseries (divided by mean apparent power of this grid)
        if mean_Sg == 0:
            Pg_daily = np.zeros(365)
            Qg_daily = np.zeros(365)
            Sg_daily = np.zeros(365)
        else:
            Pg_daily = (Pg / mean_Sg).groupby(Pg.index // 24).mean().values[:365]
            Qg_daily = (Qg / mean_Sg).groupby(Qg.index // 24).mean().values[:365]
            Sg_daily = (Sg / mean_Sg).groupby(Sg.index // 24).mean().values[:365]
        
        ts_data['P'].append(Pg_daily)
        ts_data['Q'].append(Qg_daily)
        ts_data['S'].append(Sg_daily)
        
        # 2. Normalize LDC (divided by the average maximum apparent power over all grids)
        # Sort in descending order to form a Load Duration Curve
        ldc_data['P'].append(np.sort(Pg.values)[::-1] / norm_ldc_constant)
        ldc_data['Q'].append(np.sort(Qg.values)[::-1] / norm_ldc_constant)
        ldc_data['S'].append(np.sort(Sg.values)[::-1] / norm_ldc_constant)
        
    for k in ts_data:
        ts_data[k] = np.array(ts_data[k])
        ldc_data[k] = np.array(ldc_data[k])
        
    # Plotting Figure 4.1
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    
    if preurbs:
        row_labels = [
            ('Active Power P Import', 'P', 'Norm. Active Power Import $P_i(t) / \\langle |S_i| \\rangle$', 'Norm. Active Power Import $P_i / \\langle \\max |S| \\rangle$', 'darkorange'),
            ('Reactive Power Q Import', 'Q', 'Norm. React. Power Import $Q_i(t) / \\langle |S_i| \\rangle$', 'Norm. React. Power Import $Q_i / \\langle \\max |S| \\rangle$', 'mediumvioletred'),
            ('Apparent Power |S| Load', 'S', 'Norm. Apparent Power Load $|S_i(t)| / \\langle |S_i| \\rangle$', 'Norm. Apparent Power Load $S_i / \\langle \\max |S| \\rangle$', 'teal')
        ]
    else:
        row_labels = [
            ('Active Power P Import', 'P', 'Norm. Active Power Import $P_i(t) / \\langle |S_i| \\rangle$', 'Norm. Active Power Import $P_i / \\langle \\max |S| \\rangle$', 'red'),
            ('Reactive Power Q Import', 'Q', 'Norm. React. Power Import $Q_i(t) / \\langle |S_i| \\rangle$', 'Norm. React. Power Import $Q_i / \\langle \\max |S| \\rangle$', 'purple'),
            ('Apparent Power |S| Load', 'S', 'Norm. Apparent Power Load $|S_i(t)| / \\langle |S_i| \\rangle$', 'Norm. Apparent Power Load $S_i / \\langle \\max |S| \\rangle$', 'steelblue')
        ]
    
    for r_idx, (title, key, y_lbl_ts, y_lbl_ldc, color) in enumerate(row_labels):
        # Timeseries plot (Column 0)
        ax_ts = axes[r_idx, 0]
        d_ts = ts_data[key]
        
        # Plot mean timeline
        ax_ts.plot(daily_time, np.nanmean(d_ts, axis=0), color=color, linewidth=1.5, label='Expected Timeseries (24 h Agg.)')
        # Add percentile bands to show variation range
        ax_ts.fill_between(daily_time, np.nanpercentile(d_ts, 16, axis=0), np.nanpercentile(d_ts, 84, axis=0), color=color, alpha=0.3, label='68% Percentile Band')
        ax_ts.fill_between(daily_time, np.nanpercentile(d_ts, 2, axis=0), np.nanpercentile(d_ts, 98, axis=0), color=color, alpha=0.1, label='96% Percentile Band')
        
        ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_ts.set_xlim(daily_time[0], daily_time[-1])
        ax_ts.set_ylabel(y_lbl_ts)
        ax_ts.set_title(f"Net Transformer {title}")
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(loc='upper right')
        
        # Load Duration Curve plot (Column 1)
        ax_ldc = axes[r_idx, 1]
        d_ldc = ldc_data[key]
        x_pct = np.linspace(0, 100, d_ldc.shape[1])
        
        # Plot mean LDC curve
        ax_ldc.plot(x_pct, np.nanmean(d_ldc, axis=0), color=color, linewidth=1.5, label='Expected LDC (Hourly)')
        # Add percentile bands
        ax_ldc.fill_between(x_pct, np.nanpercentile(d_ldc, 16, axis=0), np.nanpercentile(d_ldc, 84, axis=0), color=color, alpha=0.3, label='68% Percentile Band ($\\approx 1\\sigma$)')
        ax_ldc.fill_between(x_pct, np.nanpercentile(d_ldc, 2, axis=0), np.nanpercentile(d_ldc, 98, axis=0), color=color, alpha=0.1, label='96% Percentile Band ($\\approx 2\\sigma$)')
        
        ax_ldc.set_xlim(0, 100)
        ax_ldc.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}%"))
        ax_ldc.set_ylabel(y_lbl_ldc)
        ax_ldc.set_title(f"Net Transformer {title}")
        ax_ldc.grid(True, alpha=0.3)
        ax_ldc.legend(loc='upper right')
        
    # Save the plot
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'figure_4_1{file_suffix}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Successfully saved Figure 4.1 ({case_label}) to: {out_path}")


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    out_dir_custom = os.path.join(script_dir, "figures", "3rd batch")
    pwrflw_dir_custom = "/home/pedro/Linux-AntigravityProjects/SurroGrid-linux/hpc_mount"

    print("=========================================")
    print("Thesis Figure Plotting Tool (Local Mount Edition)")
    print("=========================================")
    print(f"Target figures   : 4.2a (Pre and Post)")
    print(f"Input directory  : {pwrflw_dir_custom}")
    print(f"Output directory : {out_dir_custom}")
    print("=========================================\n")

    plot_figure_4_2_a(pwrflw_dir_custom, out_dir_custom, preurbs=False)
    plot_figure_4_2_a(pwrflw_dir_custom, out_dir_custom, preurbs=True)

    print("\nJob complete!")
