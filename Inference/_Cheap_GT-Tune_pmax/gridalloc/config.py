from pathlib import Path
import sys
import pandas as pd

# Absolute paths so statistical datasets and top-level pipeline_config resolve from any working dir
GRIDALLOC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = GRIDALLOC_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline_config import (
    BASELINE_Q0_GHD,
    BASELINE_Q0_RES,
    DATA_GRID_DIR,
    NET_AREA_FACTOR,
    OUTPUT_SUFFIX,
    STORAGE_DIR,
    TARGET_Q_GHD,
    TARGET_Q_RES,
)


class Config:
    #--------------------------------------------------------------#
    #----------------- Paths/Dataset Readout ----------------------#
    #--------------------------------------------------------------#
    DATA_GRID_DIR = DATA_GRID_DIR
    STORAGE_DIR = STORAGE_DIR
    OUTPUT_SUFFIX = OUTPUT_SUFFIX
    DATA_STAT_DIR = str(GRIDALLOC_DIR / "data" / "statistics")

    #--------------------------------------------------------------#
    #-------------- Weather Data API Connections ------------------#
    #--------------------------------------------------------------#
    PVGIS_URL = "https://re.jrc.ec.europa.eu/api/tmy"
    REF_YEAR = 2009
    TIME_ZONE = 1
    OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
    OPENMETEO_TIME_ZONE = "UTC+01:00"

    #--------------------------------------------------------------#
    #---------------- Electrical Demand Assignment ----------------#
    #--------------------------------------------------------------#
    # Needed because internal heat gains from occupants and appliances/lighting
    # enter the 5R1C space-heating balance in districtgenerator
    ELEC_BY_HHSIZE_CDFS_NOHEAT = pd.read_csv(f"{DATA_STAT_DIR}/inhabited_buildings/elec_by_hhsize_cdfs_noheat.csv", header=[0, 1])
    HH_SIZE_DISTRIBUTION = pd.read_csv(f"{DATA_STAT_DIR}/inhabited_buildings/hh_size_distribution.csv", header=[0], skiprows=1)
    ELEC_LPS_PATH = f"{DATA_STAT_DIR}/inhabited_buildings/elec_lps.h5"

    ELEC_GHD_PATH = f"{DATA_STAT_DIR}/uninhabited_buildings/elec_ghd_per_m2.csv"
    DHW_GHD_PATH = f"{DATA_STAT_DIR}/uninhabited_buildings/dhw_ghd_per_m2.csv"
    TYPE_GHD_DISTRIBUTION = pd.read_csv(f"{DATA_STAT_DIR}/uninhabited_buildings/nonresbuilding_usetype_distribution.csv", header=[0], skiprows=1)
    AGE_GHD_DISTRIBUTION = pd.read_csv(f"{DATA_STAT_DIR}/uninhabited_buildings/nonresbuilding_age_distribution.csv", header=[0], skiprows=1)

    #--------------------------------------------------------------#
    #----------------- Heat Generator Constants -------------------#
    #--------------------------------------------------------------#
    DISTGEN_DATA_PATH = f"{DATA_STAT_DIR}/general/"
    ALBEDO = 0.2                        # Ground reflectance used for solar heat gains on building surfaces

    # Time
    HOLIDAYS = [1, 6, 100, 121, 141, 152, 162, 276, 305, 358, 359, 360, 365]
    INITIAL_DAY = [4]
    DATA_LENGTH = 31536000

    # Building design
    T_SET_MIN = 20.0
    T_SET_MIN_NIGHT = 18.0
    T_SET_MAX = 23.0
    T_SET_MAX_NIGHT = 28.0
    VENTILATION_RATE = 0.5
    BUILDINGS_SHORT = ["SFH", "MFH", "TH", "AB"]
    BUILDINGS_LONG = ["single_family_house", "multi_family_house", "terraced_house", "apartment_block"]
    RETROFIT_SHORT = [0, 1, 2]
    RETROFIT_LONG = ["tabula_standard", "tabula_retrofit", "tabula_adv_retrofit"]
    DHWLOAD = [4662.1, 4662.1, 4662.1, 3999.8]
    MEAN_DRAWOFF_VOL = [40, 40, 40, 40]

    # Physics
    RHO_AIR = 1.2
    C_P_AIR = 1000.0
    RHO_WATER = 1000.0
    C_P_WATER = 4.18

    #--------------------------------------------------------------#
    #---- Building Refurbishment (Gebäudesanierung) & Tuning ------#
    #--------------------------------------------------------------#
    # P_renov(construction_year) = p_max * (1 - exp(-((RETROFIT_TARGET_YEAR - construction_year) / L)^k))
    RETROFIT_TARGET_YEAR = 2045
    RETROFIT_WEIBULL_L = 45.0
    RETROFIT_WEIBULL_K = 3.5
    RETROFIT_P_MAX_RES = 0.40
    RETROFIT_P_MAX_GHD = 0.40
    RETROFIT_PROB_LEVEL_1 = 0.85
    RETROFIT_PROB_LEVEL_2 = 0.15

    # Calibration targets and unrenovated baselines [kWh_th/(m²·a)]
    TARGET_Q_RES = TARGET_Q_RES
    TARGET_Q_GHD = TARGET_Q_GHD
    BASELINE_Q0_RES = BASELINE_Q0_RES
    BASELINE_Q0_GHD = BASELINE_Q0_GHD
    NET_AREA_FACTOR = NET_AREA_FACTOR


config = Config()
