import pandas as pd
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

h5_file = "Input/534_N3237500E4299500_92533_3_5_PV100_HP100_EV100_VarTar0_CapPr0.h5"

print(f"Reading {h5_file}")
df = pd.read_hdf(h5_file, key='/raw_data/buildings')
print("Total rows in /raw_data/buildings:", len(df))

# Check for duplicates
exact_dup = df.duplicated().sum()
print("Exact duplicates:", exact_dup)

if 'vertice_id' in df.columns:
    unique_ids = df['vertice_id'].nunique()
    print("Unique vertice_id (OSM ID):", unique_ids)
    print("Duplicates based on vertice_id:", len(df) - unique_ids)
else:
    print("No vertice_id column found.")
    print("Columns:", df.columns.tolist())
