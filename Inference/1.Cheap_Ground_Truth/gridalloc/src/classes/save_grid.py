from config import config

import os
import pandas as pd
import shutil

import warnings
from pandas.errors import PerformanceWarning


def find_grid_file_by_id(directory, file_id, exclude_suffixes=("_DmdAllocated.h5", "_pwrflw.h5")):
    """Find a raw .h5 grid file in `directory` whose prefix before '_' matches `file_id`.

    Supports unpadded (e.g. '298'), 4-digit zero-padded ('0298'), and 5-digit
    ('51431') ID formats while ignoring already-processed output files.
    """
    all_entries = sorted(os.listdir(directory))
    h5_files = [
        fname
        for fname in all_entries
        if fname.endswith(".h5") and not fname.endswith(exclude_suffixes)
    ]

    try:
        val = int(file_id)
        valid_prefixes = (str(val), f"{val:04d}", f"{val:05d}")
    except ValueError:
        valid_prefixes = (str(file_id), str(file_id).zfill(4), str(file_id).zfill(5))

    matched_files = [fname for fname in h5_files if fname.split("_", 1)[0] in valid_prefixes]

    if not matched_files:
        raise IndexError(f"No grid file found in '{directory}' matching ID {file_id}. Found 0 matches.")

    return matched_files[0]


class SaveFile:
    def __init__(self, filename, input_dir=None, output_dir=None):
        self.filename = filename
        self.input_dir = input_dir if input_dir is not None else config.DATA_GRID_DIR
        self.output_dir = output_dir if output_dir is not None else config.STORAGE_DIR
        self.input_path = self._get_readpath()
        self.output_path = self._generate_savepath()

        if os.path.abspath(self.input_path) == os.path.abspath(self.output_path):
            raise ValueError(
                f"Input and output paths are identical ('{self.input_path}'). "
                "Use a separate STORAGE_DIR so raw input files are never overwritten."
            )

        # HDF5 keys from which to extract raw input data
        self.building_dir = "raw_data/buildings"
        self.region_dir = "raw_data/region"
        self.weather_dir = "raw_data/weather"

    def _get_readpath(self):
        return os.path.join(self.input_dir, self.filename)

    def _generate_savepath(self):
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, self.filename)

    def get_input_data(self):
        df_buildings = pd.read_hdf(self.input_path, key=self.building_dir)
        df_region = pd.read_hdf(self.input_path, key=self.region_dir)
        try:
            df_weather = pd.read_hdf(self.input_path, key=self.weather_dir)
        except Exception:
            df_weather = None
        return df_buildings, df_region, df_weather

    def copy_save_file(self):
        shutil.copy2(self.input_path, self.output_path)

    def save_df(self, df, dir):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PerformanceWarning)
            with pd.HDFStore(self.output_path, mode="a", complib="blosc", complevel=9) as store:
                store.put(dir, df)

    @staticmethod
    def append_suffix(file_path, suffix=None):
        """Append `suffix` before '.h5' once all processing steps finish successfully."""
        suffix = suffix if suffix is not None else config.OUTPUT_SUFFIX
        stem, ext = os.path.splitext(file_path)
        if stem.endswith(suffix):
            return file_path
        new_path = f"{stem}{suffix}{ext}"
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(file_path, new_path)
        return new_path
