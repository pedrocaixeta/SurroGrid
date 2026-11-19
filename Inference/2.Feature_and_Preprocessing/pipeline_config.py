"""Top-level configuration for Inference/3.feature_generation.

Edit DATA_GRID_DIR, STORAGE_DIR, OUTPUT_SUFFIX, and SCALER_PATH here to control
where demand-allocated grids are read from and where feature-enriched HDF5 files
are written.

Feature lists and VMD/preprocessing hyperparameters mirror the exact `baseline`
configuration from `train_combined_data.py` used to train
`baseline_alpha_peak_HPOfromOct17th.pt`.
"""

from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# ----------------------------- Data Paths ---------------------------------- #
# --------------------------------------------------------------------------- #
# Input folder containing *_DmdAllocated.h5 files from Inference/1.Cheap_Ground_Truth
DATA_GRID_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/2.Dmd_Allocated/"

# Output folder where completed *_DmdAllocated_Features.h5 files are saved
STORAGE_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/3.Features_Generated/"

# Suffix appended to the filename once extraction + VMD + scaling finish successfully
OUTPUT_SUFFIX = "_Features"

# Frozen training scalers exported once from ts_train.h5 via export_scalers.py
SCALER_PATH = str(PIPELINE_DIR / "scalers_baseline_alpha_peak_HPOfromOct17th.json")

# --------------------------------------------------------------------------- #
# ----------------------- Feature Extraction Settings ----------------------- #
# --------------------------------------------------------------------------- #
# Ground-truth PV adoption fraction used in create_PV_prod_up_to (no NAP adjustment yet)
PV_FRACTION_OF_TOTAL = 0.6766

# Expected number of hourly timesteps per grid
N_HOURS = 8760

# --------------------------------------------------------------------------- #
# ------------- Surrogate Preprocessing Settings (baseline model) ----------- #
# --------------------------------------------------------------------------- #
# Exact 19 base features in the order expected by baseline_alpha_peak_HPOfromOct17th.pt
X_BASE_COLS = [
    # Cheap Ground Truth (5 hourly series)
    "demand_net_active_pre",
    "heat_water",
    "heat_space",
    "cop_avg",
    "PV_prod_expected",
    # Mobility Proxies (2 scalars; T appended at end to match train_combined_data.py)
    "regiostar7",
    "n_cars",
    # Power Flow Proxies (2 scalars)
    "n_lines",
    "tot_R_grid",
    # Disaggregation — building counts (3 scalars)
    "n_pure_res_buildings",
    "n_pure_nonres_buildings",
    "n_mixed_buildings",
    # Disaggregation — base footprint areas (3 scalars)
    "res_bldng_area_base_sum",
    "nonres_bldng_area_base_sum",
    "mixed_bldng_area_base_sum",
    # Disaggregation — other building characteristics (3 scalars)
    "bldng_area_floors_sum",
    "n_flats",
    "n_occ",
    # Ambient temperature (1 hourly series — appended last in baseline approach)
    "T",
]

# Columns transformed with log1p before scaling
LOG1P_COLS = ["n_lines", "tot_R_grid"]

# Periods (hours) for sinusoidal seasonal encodings (sin_<p>, cos_<p>)
TS_PERIODS = [8760, 168, 24]

# Hourly columns decomposed via Variational Mode Decomposition (VMD)
VMD_COLS = [
    "demand_net_active_pre",
    "heat_water",
    "heat_space",
    "cop_avg",
    "PV_prod_expected",
    "T",
]
VMD_K_MODES = 4
VMD_SETTINGS = {
    "alpha": 2000,      # moderate bandwidth constraint
    "tau": 0.0,         # noise-tolerance (0 for noiseless)
    "K": VMD_K_MODES,   # number of modes
    "DC": 1,            # include DC part
    "init": 1,          # initialize omegas uniformly
    "tol": 1e-5,        # convergence tolerance
}

# Target columns used when fitting scaler_y in export_scalers.py
TARGET_COLS = ["demand_net_active_post", "demand_net_reactive_post"]

# Train/val split parameters from train_combined_data.py (for exact scaler reproduction)
TRAIN_VALIDATION_RATIO = 0.2
TRAIN_RANDOM_STATE = 42

# HDF5 output keys written under /surrogate_in/
H5_KEY_FEATURES_RAW = "/surrogate_in/features_raw"
H5_KEY_FEATURES_VMD = "/surrogate_in/features_vmd"
H5_KEY_FEATURES_SCALED = "/surrogate_in/features_scaled"
