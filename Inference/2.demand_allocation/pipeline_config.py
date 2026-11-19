"""Top-level path configuration for Inference/2.demand_allocation.

Edit DATA_GRID_DIR and STORAGE_DIR here to change where raw input grids are
read from and where processed *_DmdAllocated.h5 files are saved.
"""

# Input folder containing raw grid .h5 files from 1.Grid_Importing
DATA_GRID_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/1.Raw_grids/"

# Output folder where completed *_DmdAllocated.h5 files are saved
STORAGE_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/2.Dmd_Allocated/"

# Suffix appended to the output filename once all steps succeed
OUTPUT_SUFFIX = "_DmdAllocated"
