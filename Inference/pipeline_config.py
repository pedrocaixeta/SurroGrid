"""Top-level configuration for the unified (direct) Inference pipeline.

Edit DATA_GRID_DIR, STORAGE_DIR, OUTPUT_SUFFIX, or DEFAULT_KEEP_MODE here to
control where raw grids are read from, where final `*_direc_Inferred.h5` files
are saved, and which intermediate HDF5 groups are retained.
"""

from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# ----------------------------- Data Paths ---------------------------------- #
# --------------------------------------------------------------------------- #
# Input folder containing raw grid .h5 files from Grid_Importing
DATA_GRID_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/0.Raw_grids/"

# Output folder where completed *_direc_Inferred.h5 files are saved
STORAGE_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Directly_Inferred/"

# Suffix appended to the raw grid stem once all 3 stages succeed
OUTPUT_SUFFIX = "_direc_Inferred"

# --------------------------------------------------------------------------- #
# ------------------------ Sub-Stage Entrypoints ---------------------------- #
# --------------------------------------------------------------------------- #
STEP1_DIR = PIPELINE_DIR / "1.Cheap_Ground_Truth"
STEP2_DIR = PIPELINE_DIR / "2.Feature_and_Preprocessing"
STEP3_DIR = PIPELINE_DIR / "3.Forward_Pass"

STEP1_SCRIPT = str(STEP1_DIR / "main.py")
STEP2_SCRIPT = str(STEP2_DIR / "main.py")
STEP3_SCRIPT = str(STEP3_DIR / "main.py")

# Default surrogate checkpoint and matching frozen training scalers
MODEL_PATH = str(STEP3_DIR / "baseline_alpha_peak_HPOfromOct17th.pt")
SCALER_PATH = str(STEP3_DIR / "scalers_baseline_alpha_peak_HPOfromOct17th.json")

# --------------------------------------------------------------------------- #
# -------------------------- HDF5 Retention Modes --------------------------- #
# --------------------------------------------------------------------------- #
# Controls which HDF5 groups/datasets are packed into the final *_direc_Inferred.h5.
#   - "slim"   (~5-6 MB/grid): raw_data, pre-expansion trafo load, raw features,
#              surrogate predictions/summary, performance.
#   - "medium" (~40 MB/grid):  slim + urbs_in/* (allows re-running Step 2 PV/HP
#              tweaks later without re-running the ~8-min Step 1).
#   - "full"   (~120 MB/grid): keep every HDF5 key identical to step-by-step output.
DEFAULT_KEEP_MODE = "medium"

HDF5_KEEP_KEYS = {
    "slim": [
        "raw_data",
        "pwrflw/output/pre/demand_import",
        "surrogate_in/features_raw",
        "surrogate_out",
        "performance",
    ],
    "medium": [
        "raw_data",
        "urbs_in",
        "pwrflw/output/pre/demand_import",
        "surrogate_in/features_raw",
        "surrogate_out",
        "performance",
    ],
    "full": None,  # Copy entire HDF5 file verbatim
}
