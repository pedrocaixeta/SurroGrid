from config import config
import re
import pandas as pd
import numpy as np
from src.external.districtgenerator.classes import *
import warnings


##############################################################
############ Refurbishment (Gebäudesanierung) Model ##########
##############################################################
def _extract_construction_year(constructi_val):
    """Extract the integer construction year from `constructi` using the same regex as TABULA."""
    if pd.isna(constructi_val):
        return 1970  # Fallback aligned with datahandler.py default
    match = re.search(r"(\d+)(?!.*\d)", str(constructi_val))
    return int(match.group(1)) if match else 1970


def calc_retrofit_probability(
    construction_year,
    p_max,
    target_year=None,
    weibull_l=None,
    weibull_k=None,
):
    """Compute building refurbishment probability via a scaled Weibull CDF.

    P_renov(construction_year) = p_max * (1 - exp(-((target_year - construction_year) / L)^k))
    """
    if target_year is None:
        target_year = config.RETROFIT_TARGET_YEAR
    if weibull_l is None:
        weibull_l = config.RETROFIT_WEIBULL_L
    if weibull_k is None:
        weibull_k = config.RETROFIT_WEIBULL_K

    year = _extract_construction_year(construction_year)
    age = max(0.0, float(target_year - year))

    # Scaled Weibull CDF: approaches 0 for new buildings and saturates at p_max for old stock
    weibull_cdf = 1.0 - np.exp(-((age / float(weibull_l)) ** float(weibull_k)))
    return float(np.clip(p_max * weibull_cdf, 0.0, 1.0))


def sample_building_retrofit(
    df_buildings,
    p_max_res=None,
    p_max_ghd=None,
    prob_level_1=None,
):
    """Sample a TABULA retrofit level (0, 1, or 2) for each building in `df_buildings`.

    Decision logic per building:
    1. Select sector `p_max`: if `residential_floor_area >= nonresidential_floor_area`,
       use `p_max_res`; otherwise use `p_max_ghd` (falls back to building `type` if both
       floor-area columns are zero/missing).
    2. Compute age-dependent refurbishment probability `P_renov(construction_year)` via scaled Weibull CDF.
    3. Draw whether the building is refurbished; if yes, sample TABULA level 1
       (`tabula_retrofit`, conventional GEG) with probability `prob_level_1` and level 2
       (`tabula_adv_retrofit`, ambitious BEG) with probability `1 - prob_level_1`.
    """
    if p_max_res is None:
        p_max_res = config.RETROFIT_P_MAX_RES
    if p_max_ghd is None:
        p_max_ghd = config.RETROFIT_P_MAX_GHD
    if prob_level_1 is None:
        prob_level_1 = config.RETROFIT_PROB_LEVEL_1

    prob_level_2 = 1.0 - float(prob_level_1)
    retrofit_levels = []

    for _, row in df_buildings.iterrows():
        # Determine dominant use (residential vs. GHD) for mixed and single-use buildings
        res_area = pd.to_numeric(row.get("residential_floor_area", 0), errors="coerce")
        nonres_area = pd.to_numeric(row.get("nonresidential_floor_area", 0), errors="coerce")
        res_area = 0.0 if pd.isna(res_area) else float(res_area)
        nonres_area = 0.0 if pd.isna(nonres_area) else float(nonres_area)

        if (res_area > 0.0) or (nonres_area > 0.0):
            use_res_prob = res_area >= nonres_area
        else:
            use_res_prob = row.get("type", "") in config.BUILDINGS_SHORT

        p_max = p_max_res if use_res_prob else p_max_ghd
        p_renov = calc_retrofit_probability(row.get("constructi", np.nan), p_max=p_max)

        # Sample refurbishment status and, if refurbished, sample TABULA retrofit depth (1 or 2)
        if np.random.random() < p_renov:
            level = int(np.random.choice([1, 2], p=[prob_level_1, prob_level_2]))
        else:
            level = 0
        retrofit_levels.append(level)

    df_buildings["retrofit"] = np.array(retrofit_levels, dtype=int)
    return df_buildings


##############################################################
############## Generation, Publicly Callable #################
##############################################################
def sample_statistics(df_buildings, p_max_res=None, p_max_ghd=None):
    """Sample missing construction years for non-residential buildings and assign TABULA retrofit levels."""
    df_buildings.loc[df_buildings["constructi"].isna(), ["constructi"]] = np.random.choice(
        config.AGE_GHD_DISTRIBUTION["age"],
        size=len(df_buildings[df_buildings["constructi"].isna()]["constructi"]),
        p=config.AGE_GHD_DISTRIBUTION["prob"],
    )

    # Sample TABULA refurbishment level (0: unrenovated, 1: standard retrofit, 2: advanced retrofit)
    df_buildings = sample_building_retrofit(
        df_buildings,
        p_max_res=p_max_res,
        p_max_ghd=p_max_ghd,
    )
    return df_buildings


def generate_heat_demands(df_buildings, df_elec_demand, df_elec_demand_res, df_elec_demand_ghd, weather_data, zip):
    """Simulate hourly space-heating demand (kW) for all buildings in the grid.

    Domestic hot water (DHW) generation is intentionally skipped in this tuning module
    because `p_max` is calibrated purely on space-heating demand (Nutzwärmeenergie per m²).
    """
    scenario_rows = []
    new_id = 0
    id_to_bus = {}
    split_elec_dict = {}

    for _, row in df_buildings.iterrows():
        bus = row["bus"]
        bus_col = (bus, "electricity")
        b_type = row.get("type", "")

        nonres_area = pd.to_numeric(row.get("nonresidential_floor_area", 0), errors="coerce")
        res_area = pd.to_numeric(row.get("residential_floor_area", 0), errors="coerce")
        if pd.isna(nonres_area):
            nonres_area = 0
        if pd.isna(res_area):
            res_area = 0

        is_mixed = (res_area > 0) and (nonres_area > 0)

        if is_mixed:
            # --- Mixed Building: Split into residential and non-residential components ---
            # Both components inherit the same building-level `retrofit` value.
            row_res = row.copy()
            row_res["type"] = b_type if b_type in ["SFH", "MFH", "TH", "AB"] else "AB"
            row_res["area"] = res_area
            row_res["NWG"] = 0  # 0 = Residential space-heating profile
            row_res["new_id"] = new_id
            id_to_bus[new_id] = bus
            if bus_col in df_elec_demand_res.columns:
                split_elec_dict[new_id] = df_elec_demand_res[bus_col]
            scenario_rows.append(row_res)
            new_id += 1

            row_nonres = row.copy()
            row_nonres["type"] = "Commercial"
            row_nonres["area"] = nonres_area
            row_nonres["NWG"] = 1  # 1 = Non-Residential space-heating profile
            row_nonres["new_id"] = new_id
            id_to_bus[new_id] = bus
            if bus_col in df_elec_demand_ghd.columns:
                split_elec_dict[new_id] = df_elec_demand_ghd[bus_col]
            scenario_rows.append(row_nonres)
            new_id += 1

        else:
            # --- Standard single-use building ---
            row_reg = row.copy()
            row_reg["area"] = row.get("area") * row.get("floors")

            if b_type in ["SFH", "MFH", "TH", "AB"]:
                row_reg["NWG"] = 0  # Residential
                if bus_col in df_elec_demand_res.columns:
                    split_elec_dict[new_id] = df_elec_demand_res[bus_col]
                elif bus_col in df_elec_demand.columns:
                    split_elec_dict[new_id] = df_elec_demand[bus_col]
            else:
                row_reg["NWG"] = 1  # Non-Residential
                if bus_col in df_elec_demand_ghd.columns:
                    split_elec_dict[new_id] = df_elec_demand_ghd[bus_col]
                elif bus_col in df_elec_demand.columns:
                    split_elec_dict[new_id] = df_elec_demand[bus_col]

            row_reg["new_id"] = new_id
            id_to_bus[new_id] = bus
            scenario_rows.append(row_reg)
            new_id += 1

    scenario = pd.DataFrame(scenario_rows)
    scenario.reset_index(drop=True, inplace=True)
    df_elec_demand_split = pd.DataFrame(split_elec_dict)

    if "retrofit" not in scenario.columns:
        scenario["retrofit"] = 0

    scenario = scenario[
        ["new_id", "bus", "type", "constructi", "area", "floors", "houses_per_building", "occ_list", "NWG", "retrofit"]
    ]
    scenario.rename(
        inplace=True,
        columns={
            "new_id": "id",
            "type": "building",
            "houses_per_building": "nb_flat",
            "occ_list": "nb_occ",
            "constructi": "year",
        },
    )
    scenario["year"] = scenario["year"].str.extract(r"(\d+)(?!.*\d)").astype(int)
    scenario["nb_occ"] = scenario["nb_occ"].apply(lambda x: [int(round(y, 0)) for y in x])
    scenario["retrofit"] = scenario["retrofit"].fillna(0).astype(int)

    # Extract location data and run TEASER + 5R1C thermal simulation
    site_data = pd.read_csv(f"{config.DISTGEN_DATA_PATH}/site_data.txt", delimiter="\t", dtype={"Zip": str})
    heat_data = Datahandler(scenario, scenario_name="example", zip_code=str(zip))
    heat_data.generateEnvironment(weather_data, site_data)
    heat_data.initializeBuildings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        heat_data.generateBuildings()
        df_space_heat, _, _ = heat_data.generateDemands(df_elec_demand_split)

    # Map demands back to bus and sum parts from the same building
    df_space_heat.columns = [id_to_bus[int(float(col))] for col in df_space_heat.columns]
    df_space_heat = df_space_heat.T.groupby(level=0).sum().T
    df_space_heat.columns = pd.MultiIndex.from_product([df_space_heat.columns, ["space_heat"]])

    return df_space_heat
