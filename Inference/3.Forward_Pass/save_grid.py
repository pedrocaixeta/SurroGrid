"""HDF5 I/O helper for Inference/3.feature_generation.

Locates `*_DmdAllocated.h5` input files by grid ID, copies them to `STORAGE_DIR`
so inputs are never modified in-place, writes `/surrogate_in/*` feature tables
with blosc compression, and appends `_Features` only after all steps succeed.
"""

import os
import shutil
import warnings
import pandas as pd
from pandas.errors import PerformanceWarning

import pipeline_config as cfg


def find_grid_file_by_id(directory, file_id, exclude_suffixes=("_Features.h5", "_Predicted.h5")):
    """Find a demand-allocated .h5 grid file in `directory` matching `file_id`.

    Supports unpadded ('197'), 4-digit ('0197'), and 5-digit ('51431') prefixes.
    Prefers files ending with '_DmdAllocated.h5' when multiple files share an ID.
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
        raise IndexError(
            f"No grid file found in '{directory}' matching ID {file_id}. Found 0 matches."
        )

    # Prefer *_DmdAllocated.h5 if both raw/gt and demand-allocated files exist in the same folder
    dmd_alloc_matches = [f for f in matched_files if f.endswith("_DmdAllocated.h5")]
    if dmd_alloc_matches:
        return dmd_alloc_matches[0]

    return matched_files[0]


class SaveFile:
    """Manages copying an input HDF5 file to STORAGE_DIR and saving feature tables."""

    def __init__(self, filename, input_dir=None, output_dir=None):
        self.filename = filename
        self.input_dir = input_dir if input_dir is not None else cfg.DATA_GRID_DIR
        self.output_dir = output_dir if output_dir is not None else cfg.STORAGE_DIR

        self.input_path = os.path.join(self.input_dir, self.filename)
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, self.filename)

        if os.path.abspath(self.input_path) == os.path.abspath(self.output_path):
            raise ValueError(
                f"Input and output paths are identical ('{self.input_path}'). "
                "Use a separate STORAGE_DIR so upstream files are never overwritten."
            )

    def copy_save_file(self):
        """Copy the input .h5 file to the working path in STORAGE_DIR."""
        shutil.copy2(self.input_path, self.output_path)

    def save_df(self, df, key):
        """Write a DataFrame to the working .h5 file using blosc level-9 compression."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PerformanceWarning)
            with pd.HDFStore(self.output_path, mode="a", complib="blosc", complevel=9) as store:
                store.put(key, df)

    @staticmethod
    def append_suffix(file_path, suffix=None):
        """Append `suffix` before '.h5' once all feature-generation steps succeed."""
        suffix = suffix if suffix is not None else cfg.OUTPUT_SUFFIX
        stem, ext = os.path.splitext(file_path)
        if stem.endswith(suffix):
            return file_path
        new_path = f"{stem}{suffix}{ext}"
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(file_path, new_path)
        return new_path
