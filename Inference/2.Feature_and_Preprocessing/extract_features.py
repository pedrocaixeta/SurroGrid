"""Step 2 of Inference: Extract the 19 raw base features from a *_DmdAllocated.h5 file.

Directly replicates the feature-extraction logic in
`0.Legacy_Folders/GridForecast/0_preprocessing/0.extract_ML_data.ipynb` for the
baseline (without-mobility) Transformer surrogate model.

Output DataFrame:
- Shape: (8760, 19), dtype float32
- Index: `hour` (0 .. 8759)
- Columns: exact order of `pipeline_config.X_BASE_COLS`
"""

import h5py
import numpy as np
import pandas as pd
import pandapower as pp

import pipeline_config as cfg


# --------------------------------------------------------------------------- #
# ----------- Helper Functions (from 0.extract_ML_data.ipynb) --------------- #
# --------------------------------------------------------------------------- #
def create_PV_prod_up_to(supim, processes, fraction_of_total=cfg.PV_FRACTION_OF_TOTAL):
    """Compute expected hourly PV generation (kW) by greedily selecting best roofs
    up to `fraction_of_total` of maximum annual PV energy potential."""
    pv_100_peak = (
        processes[processes["Process"].str.startswith("Rooftop PV")]
        .set_index(["Site", "Process"])["cap-up"]
        .to_frame()
        .T
    )

    cap_map = {}
    for site, cap_proc in pv_100_peak.columns:
        solar_proc = cap_proc.replace("Rooftop PV", "solar")
        cap_map[(site, solar_proc)] = pv_100_peak[(site, cap_proc)].iloc[0]

    cap_series = pd.Series(
        [cap_map.get((site, proc), pd.NA) for site, proc in supim.columns],
        index=supim.columns,
    )
    df_scaled = supim * cap_series

    max_energy = float(df_scaled.sum().sum())
    if max_energy <= 0.0 or fraction_of_total <= 0.0:
        return pd.Series(0.0, index=supim.index, dtype=float)

    target_energy = max_energy * fraction_of_total
    supim_sums = supim.sum(axis=0)
    energy_sums = df_scaled.sum(axis=0)
    order = supim_sums.sort_values(ascending=False).index

    selected_ts = []
    total_energy = 0.0
    for col in order:
        col_energy = float(energy_sums[col])
        if col_energy <= 0.0:
            continue
        if total_energy + col_energy <= target_energy:
            selected_ts.append(df_scaled[col])
            total_energy += col_energy
        else:
            remaining = target_energy - total_energy
            frac = remaining / col_energy
            selected_ts.append(df_scaled[col] * frac)
            total_energy = target_energy
            break

    if not selected_ts:
        return pd.Series(0.0, index=supim.index, dtype=float)

    return pd.concat(selected_ts, axis=1).sum(axis=1)


def get_heat_and_cop(eff_factor, demand):
    """Compute hourly grid-wide space heat (kW), water heat (kW), and heat-demand-weighted COP."""
    mask_sh = demand.columns.get_level_values(1).str.startswith("space_heat")
    space_heat = demand.loc[:, mask_sh].droplevel(1, axis=1)
    space_heat_tot = space_heat.sum(axis=1)

    mask_wh = demand.columns.get_level_values(1).str.startswith("water_heat")
    water_heat = demand.loc[:, mask_wh].droplevel(1, axis=1)
    water_heat_tot = water_heat.sum(axis=1)

    mask_cop = eff_factor.columns.get_level_values(1).str.startswith("heatpump_air")
    hp_cop = eff_factor.loc[:, mask_cop].droplevel(1, axis=1)

    total_heat_cols = water_heat + space_heat
    denom = water_heat_tot + space_heat_tot
    weighted_sum = hp_cop.multiply(total_heat_cols).sum(axis=1)

    weighted_cop = pd.Series(np.nan, index=denom.index, dtype=float)
    nonzero = denom != 0
    weighted_cop.loc[nonzero] = weighted_sum.loc[nonzero] / denom.loc[nonzero]

    # Fallback: simple mean COP across buildings when total heat demand is zero
    row_mean_cop = hp_cop.mean(axis=1)
    weighted_cop.loc[~nonzero] = row_mean_cop.loc[~nonzero]

    return space_heat_tot, water_heat_tot, weighted_cop


def get_number_lines(grid):
    """Count feeder lines emanating from the transformer LV bus."""
    trafo_buses = grid.trafo[["hv_bus", "lv_bus"]].values[0]
    ext_grid_bus = int(grid.ext_grid.loc[0, "bus"])
    lv_bus = [bus for bus in trafo_buses if bus != ext_grid_bus][0]
    hh_strands = grid.line[grid.line["from_bus"] == lv_bus].index.values
    return int(len(hh_strands))


def get_total_resistance_grid(grid):
    """Compute total grid line resistance (Ohm) as sum(length_km * r_ohm_per_km)."""
    return float((grid.line["length_km"] * grid.line["r_ohm_per_km"]).sum())


# --------------------------------------------------------------------------- #
# ------------------------ Main Feature Extractor --------------------------- #
# --------------------------------------------------------------------------- #
def extract_raw_features(h5_path, pv_fraction_of_total=cfg.PV_FRACTION_OF_TOTAL):
    """Extract all 19 baseline raw features from a single *_DmdAllocated.h5 file.

    Returns a DataFrame of shape (8760, 19) indexed by `hour` (0..8759) with
    dtype `float32` and columns ordered according to `pipeline_config.X_BASE_COLS`.
    """
    with pd.HDFStore(str(h5_path), mode="r") as store:
        demand_net_active_pre = store["pwrflw/output/pre/demand_import"]["p_mw"].to_numpy(dtype=np.float32)
        temp_air = store["raw_data/weather"]["temp_air"].to_numpy(dtype=np.float32)

        supim = store["urbs_in/supim"]
        processes = store["urbs_in/process"]
        demand = store["urbs_in/demand"]
        eff_factor = store["urbs_in/eff_factor"]
        buildings = store["raw_data/buildings"]
        region = store["raw_data/region"]

    # Hourly aggregated features (Cheap Ground Truth)
    pv_prod_expected = create_PV_prod_up_to(
        supim.copy(), processes.copy(), fraction_of_total=pv_fraction_of_total
    ).to_numpy(dtype=np.float32)
    heat_space, heat_water, cop_avg = get_heat_and_cop(eff_factor.copy(), demand.copy())

    n_hours = len(demand_net_active_pre)
    if n_hours != cfg.N_HOURS:
        raise ValueError(f"Expected {cfg.N_HOURS} hourly timesteps in '{h5_path}', got {n_hours}.")

    # Building disaggregation & mobility proxy scalars
    pure_res = buildings["nonresidential_floor_area"] == 0
    pure_nonres = buildings["residential_floor_area"] == 0
    mixed = (buildings["residential_floor_area"] != 0) & (buildings["nonresidential_floor_area"] != 0)

    n_occ = int(round(sum(x for occ in buildings["occ_list"] for x in occ), 0))
    n_flats = int(buildings["houses_per_building"].sum())
    n_cars = int(buildings["n_cars_tot"].sum())
    regiostar7 = float(region["regio7"])

    # Power flow proxy scalars from pandapower network
    with h5py.File(str(h5_path), "r") as f:
        net_json = f["raw_data/net"][()]
        if isinstance(net_json, bytes):
            net_json = net_json.decode("utf-8")
        net = pp.from_json_string(net_json)

    n_lines = float(get_number_lines(net))
    tot_r_grid = float(get_total_resistance_grid(net))

    # Assemble 8760 x 19 DataFrame (scalars broadcast across all hours, matching ts_train.h5)
    feature_data = {
        "demand_net_active_pre": demand_net_active_pre,
        "heat_water": heat_water.to_numpy(dtype=np.float32),
        "heat_space": heat_space.to_numpy(dtype=np.float32),
        "cop_avg": cop_avg.to_numpy(dtype=np.float32),
        "PV_prod_expected": pv_prod_expected,
        "regiostar7": np.full(n_hours, regiostar7, dtype=np.float32),
        "n_cars": np.full(n_hours, n_cars, dtype=np.float32),
        "n_lines": np.full(n_hours, n_lines, dtype=np.float32),
        "tot_R_grid": np.full(n_hours, tot_r_grid, dtype=np.float32),
        "n_pure_res_buildings": np.full(n_hours, len(buildings[pure_res]), dtype=np.float32),
        "n_pure_nonres_buildings": np.full(n_hours, len(buildings[pure_nonres]), dtype=np.float32),
        "n_mixed_buildings": np.full(n_hours, len(buildings[mixed]), dtype=np.float32),
        "res_bldng_area_base_sum": np.full(n_hours, buildings[pure_res]["area"].sum(), dtype=np.float32),
        "nonres_bldng_area_base_sum": np.full(n_hours, buildings[pure_nonres]["area"].sum(), dtype=np.float32),
        "mixed_bldng_area_base_sum": np.full(n_hours, buildings[mixed]["area"].sum(), dtype=np.float32),
        "bldng_area_floors_sum": np.full(n_hours, (buildings["area"] * buildings["floors"]).sum(), dtype=np.float32),
        "n_flats": np.full(n_hours, n_flats, dtype=np.float32),
        "n_occ": np.full(n_hours, n_occ, dtype=np.float32),
        "T": temp_air,
    }

    df_raw = pd.DataFrame(feature_data, index=pd.Index(np.arange(n_hours, dtype=int), name="hour"))
    df_raw = df_raw[cfg.X_BASE_COLS].astype(np.float32)

    if df_raw.isna().any().any():
        nan_cols = df_raw.columns[df_raw.isna().any()].tolist()
        raise ValueError(f"NaN values detected in extracted raw features for '{h5_path}': {nan_cols}")

    return df_raw
