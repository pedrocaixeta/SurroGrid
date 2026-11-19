"""Top-level configuration for Inference/3.Forward_Pass.

Edit DATA_GRID_DIR, STORAGE_DIR, MODEL_PATH, and SCALER_PATH here to control
where feature-preprocessed grids are read from, which trained surrogate model
is loaded, and where predicted HDF5 files are saved.
"""

from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# ----------------------------- Data Paths ---------------------------------- #
# --------------------------------------------------------------------------- #
# Input folder containing *_DmdAllocated_Features.h5 files from 2.Feature_and_Preprocessing
DATA_GRID_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/3.Features_Generated/"

# Output folder where completed *_DmdAllocated_Features_Predicted.h5 files are saved
STORAGE_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/4.Forward_Past/"

# Suffix appended to the filename once the forward pass succeeds
OUTPUT_SUFFIX = "_Predicted"

# --------------------------------------------------------------------------- #
# -------------------------- Model & Scaler Paths --------------------------- #
# --------------------------------------------------------------------------- #
# Trained Transformer surrogate checkpoint and its matching frozen training scalers
MODEL_PATH = str(PIPELINE_DIR / "baseline_alpha_peak_HPOfromOct17th.pt")
SCALER_PATH = str(PIPELINE_DIR / "scalers_baseline_alpha_peak_HPOfromOct17th.json")

# --------------------------------------------------------------------------- #
# ------------------------------ HDF5 Keys ---------------------------------- #
# --------------------------------------------------------------------------- #
# Input key written by 2.Feature_and_Preprocessing
H5_KEY_FEATURES_SCALED = "/surrogate_in/features_scaled"

# Output keys written by 3.Forward_Pass
H5_KEY_PREDICTION = "/surrogate_out/demand_import"   # Hourly p_mw, q_mvar, s_mva, loading_pct
H5_KEY_SUMMARY = "/surrogate_out/summary"            # Grid-level peak load vs trafo sn_mva
