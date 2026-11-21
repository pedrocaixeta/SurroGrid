"""Top-level path configuration for Inference/1.Cheap_Ground_Truth.

Edit DATA_GRID_DIR and STORAGE_DIR here to change where raw input grids are
read from and where processed *_DmdAllocated.h5 files are saved.
"""

# Input folder containing raw grid .h5 files from Grid_Importing
DATA_GRID_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/1.Grid_Sampled/"

# Output folder where completed *_DmdAllocated.h5 files are saved
STORAGE_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/NAP_determine_Gebäudesanierungsrate/"

# Suffix appended to the output filename once all steps succeed
OUTPUT_SUFFIX = "_DmdAllocated"
