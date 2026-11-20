# Inference — Step 2: Feature Extraction & Preprocessing

Extracts the 19 baseline surrogate input features from a `*_DmdAllocated.h5` file (produced by Step 1) and applies all preprocessing transformations (`log1p`, sinusoidal seasonal encodings, VMD `K=4`, and frozen training `StandardScaler`) needed by `3.Forward_Pass/`.

```text
Step_by_Step/1.Cheap_Ground_Truthed/<grid>_DmdAllocated.h5
  -->  [Step 2: Feature Extraction & Preprocessing]
  -->  Step_by_Step/2.Featured_and_Preprocessed/<grid>_DmdAllocated_Features.h5
```

## What It Does (`main.py`)

For a given grid ID, `main.py` copies `<grid>_DmdAllocated.h5` into `STORAGE_DIR` and runs:

1. **Feature Extraction (`extract_features.py`)**
   - Computes the 19 raw base features (6 hourly series: `demand_net_active_pre` [MW], `heat_water` [kW], `heat_space` [kW], `cop_avg`, `PV_prod_expected` [kW], `T`; plus 13 grid-level scalars) -> saves `/surrogate_in/features_raw` (`8760 x 19`).
2. **Feature Preprocessing (`preprocess_features.py`)**
   - Applies `log1p` to `n_lines` and `tot_R_grid`, adds 6 seasonal `sin`/`cos` encodings (periods `8760`, `168`, `24` h), and decomposes the 6 hourly series via Variational Mode Decomposition (`K=4` modes + residual = 30 columns) -> saves `/surrogate_in/features_vmd` (`8760 x 49`, unscaled).
   - Standardizes the 43 non-seasonal features using the frozen training `scaler_X` from `../3.Forward_Pass/scalers_baseline_alpha_peak_HPOfromOct17th.json` -> saves `/surrogate_in/features_scaled` (`8760 x 49`, ready for Step 3).
3. **Completion & Logging**
   - Logs `/performance` metrics and appends `_Features` (`<grid>_DmdAllocated_Features.h5`) **only upon full success**.

## Configuration

Paths, feature lists, and VMD settings live in [pipeline_config.py](pipeline_config.py):

- **`DATA_GRID_DIR`**: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Step_by_Step/1.Cheap_Ground_Truthed/`
- **`STORAGE_DIR`**: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Step_by_Step/2.Featured_and_Preprocessed/`
- **`OUTPUT_SUFFIX`**: `_Features`
- **`SCALER_PATH`**: `../3.Forward_Pass/scalers_baseline_alpha_peak_HPOfromOct17th.json`

## Environment & SLURM Resources

- **Conda environment**: `inference`.
- **SLURM script**: [run_cluster_serialstd.sh](run_cluster_serialstd.sh) — `serial_std`, 4 CPUs, 4 GB RAM, 5 min walltime (~3–5 s runtime per grid).

## Usage

### On LRZ Linux Cluster (SLURM)
```bash
sbatch run_cluster_serialstd.sh 0197
```

### Local Test Run
```bash
conda activate inference
python3 main.py 0197 --n_cpu 4 \
  --input_dir ../1.Cheap_Ground_Truth/gridalloc/results \
  --output_dir /tmp/inference_step2_test
```

## One-Time Scaler Export (`export_scalers.py`)

Because `.pt` checkpoints only store `{'config', 'state_dict'}`, `export_scalers.py` fits and exports the exact training `scaler_X` and `scaler_y` once from `ts_train.h5` (already exported into `../3.Forward_Pass/scalers_baseline_alpha_peak_HPOfromOct17th.json`). Only re-run if switching to a different trained model:

```bash
python3 export_scalers.py \
  --ts_train_path /dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/0_preprocessing/Data/ts_train.h5 \
  --vmd_cache_path /dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/3_transformer/data/VMD4_PVprodexpected_T_copavg_demandnetactivepre_heatspace_heatwater.h5
```
