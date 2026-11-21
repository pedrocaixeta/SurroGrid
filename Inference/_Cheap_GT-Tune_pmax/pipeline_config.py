"""Top-level configuration for Inference/1.Cheap_GT-Tune_pmax.

Configures input/output directories and reference targets for tuning the
Weibull building refurbishment probability parameters (p_max_res, p_max_ghd)
against the NAP / Genehmigung Szenariorahmen 2025 (Scenario B 2045) benchmarks.
"""

# Input folder containing raw ground-truth grid .h5 files
DATA_GRID_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/1.Grid_Sampled/"

# Output folder where each iteration saves its processed grid .h5 files
# (cleared at the start of every iteration so only the latest batch remains)
STORAGE_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/NAP_determine_Gebäudesanierungsrate/"

# Suffix appended to each processed grid filename
OUTPUT_SUFFIX = "_DmdAllocated"

# Progressive doubling batch sizes (final pass automatically runs all available grids)
TUNING_BATCH_SIZES = [50, 100, 200, 400, 800]

# Fixed random seed for shuffling the grid file list so small batches are representative
TUNING_SHUFFLE_SEED = 42

# Reference specific space-heating targets from Genehm_SR_2025Strom.pdf (Scenario B 2045)
# Table 2 (p. 26) for private households (HH) and Table 5 (p. 29) for commercial (GHD) [kWh_th/(m²·a)]
TARGET_Q_RES = 103.8
TARGET_Q_GHD = 90.6

# Unrenovated (p_max = 0) baseline specific space-heating demands across all 1,463 GT grids [kWh_th/(m²·a)]
BASELINE_Q0_RES = 141.03
BASELINE_Q0_GHD = 117.64

# Net heated floor area factor (net_leased_area = NET_AREA_FACTOR * gross_floor_area)
NET_AREA_FACTOR = 0.75
