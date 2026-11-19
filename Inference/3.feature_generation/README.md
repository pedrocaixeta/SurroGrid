# Inference — Step 3: Feature Generation & Preprocessing

Extracts the 19 baseline surrogate input features from a `*_DmdAllocated.h5` file and performs all pre-forward-pass preprocessing (`log1p`, sinusoidal seasonal encodings, VMD `K=4`, and `StandardScaler`) inside the lightweight `inference` conda environment.

## Workflow (`main.py`)

For a given grid ID:
1. Copies `<stem>_DmdAllocated.h5` from `DATA_GRID_DIR` to `STORAGE_DIR`.
2. **Feature Extraction (`extract_features.py`)**: computes the 19 raw base features (6 hourly series + 13 grid scalars) and saves `/surrogate_in/features_raw` (`8760 x 19`).
3. **Feature Preprocessing (`preprocess_features.py`)**:
   - Applies `log1p` (`n_lines`, `tot_R_grid`), seasonal `sin`/`cos` (`8760`, `168`, `24`), and VMD (`K=4` + residual on 6 hourly series) -> saves `/surrogate_in/features_vmd` (`8760 x 49`, unscaled).
   - Standardizes using the frozen training `scaler_X` (`scalers_baseline_alpha_peak_HPOfromOct17th.json`) -> saves `/surrogate_in/features_scaled` (`8760 x 49`, ready for Step 4 forward pass).
4. Logs `/performance` and appends `_Features` (`<stem>_DmdAllocated_Features.h5`) only upon full success.

## One-Time Scaler Export

Because `.pt` checkpoints only store `{'config', 'state_dict'}`, export the exact training `scaler_X` and `scaler_y` once from `ts_train.h5`:
```bash
# On LRZ (fast, ~5 s using cached VMD4 file):
python3 export_scalers.py \
  --ts_train_path /dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/0_preprocessing/Data/ts_train.h5 \
  --vmd_cache_path /dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/3_transformer/data/VMD4_PVprodexpected_T_copavg_demandnetactivepre_heatspace_heatwater.h5
```

## Configuration & Usage

Paths and settings live in `pipeline_config.py`.

```bash
# On LRZ Linux Cluster
sbatch run_cluster_serialstd.sh 0197

# Local test
conda activate inference
python3 main.py 0197 --n_cpu 4 \
  --input_dir ../1.Cheap_Ground_Truth/gridalloc/results \
  --output_dir /tmp/feat_test
```
