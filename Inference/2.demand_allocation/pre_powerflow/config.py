from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline_config import STORAGE_DIR


class Config():
    ### Data directories (centrally defined in ../pipeline_config.py)
    DATA_DIR = STORAGE_DIR
    STORAGE_DIR = STORAGE_DIR

    ### Power factor for pre-expansion building electricity demand
    PF_ELC = 0.959      # Electricity, source: https://www.researchgate.net/publication/285577915_Representative_electrical_load_profiles_of_residential_buildings_in_Germany_with_a_temporal_resolution_of_one_second

config = Config()