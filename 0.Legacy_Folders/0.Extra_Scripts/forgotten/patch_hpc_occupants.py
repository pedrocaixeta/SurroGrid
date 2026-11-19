import os
import glob
import warnings
import pandas as pd

# Optional import of tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed in the HPC environment
    def tqdm(iterable, desc=""):
        total = len(iterable)
        print(f"{desc}...")
        for i, item in enumerate(iterable):
            yield item
            # Print progress every 10% or at the end
            if (i + 1) % max(1, total // 10) == 0 or i == total - 1:
                print(f"  Progress: {i + 1}/{total} ({(i + 1)/total * 100:.1f}%)")

HPC_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/result1.gridsamping__input2/"
KEY = "raw_data/buildings"

def patch_files():
    # Find all .h5 files in the target directory
    search_pattern = os.path.join(HPC_DIR, "*.h5")
    h5_files = glob.glob(search_pattern)
    
    if not h5_files:
        print(f"No .h5 files found in {HPC_DIR}")
        return
        
    print(f"Found {len(h5_files)} .h5 files to process in {HPC_DIR}")
    
    # Suppress performance warnings from HDF5
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    
    success_count = 0
    error_count = 0
    unchanged_count = 0
    
    for file_path in tqdm(h5_files, desc="Patching files"):
        try:
            # Read the buildings DataFrame
            with pd.HDFStore(file_path, mode="r") as store:
                if KEY not in store:
                    print(f"\nSkipping {os.path.basename(file_path)}: Key '{KEY}' not found.")
                    unchanged_count += 1
                    continue
                df = store.get(KEY)
            
            # Check if expected columns exist
            if "occupants" not in df.columns or "use" not in df.columns:
                print(f"\nSkipping {os.path.basename(file_path)}: 'occupants' or 'use' column missing.")
                unchanged_count += 1
                continue
            
            # Identify non-residential buildings
            non_res_mask = df["use"] != "Residential"
            
            # Check if there are any non-residential occupants that are not yet null
            to_modify = df.loc[non_res_mask & df["occupants"].notna()]
            if len(to_modify) == 0:
                unchanged_count += 1
                continue
                
            # Set occupants to None (becomes NaN in standard pandas float64 column)
            df.loc[non_res_mask, "occupants"] = None
            
            # Save back to H5 overwriting the raw_data/buildings key
            with pd.HDFStore(file_path, mode="a", complib='blosc', complevel=9) as store:
                store.put(KEY, df)
                
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {os.path.basename(file_path)}: {e}")
            error_count += 1
            
    print(f"\nPatching complete!")
    print(f"Successfully patched: {success_count} files.")
    print(f"Skipped/Unchanged: {unchanged_count} files.")
    if error_count > 0:
        print(f"Errors encountered: {error_count} files.")

if __name__ == "__main__":
    patch_files()
