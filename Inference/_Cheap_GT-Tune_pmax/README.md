# Inference — `1.Cheap_GT-Tune_pmax` (Gebäudesanierung `p_max` Tuner)

Tunes the maximum refurbishment probability parameters (`RETROFIT_P_MAX_RES` and `RETROFIT_P_MAX_GHD`) of the scaled Weibull building renovation model:

```text
P_renov(construction_year) = p_max * (1 - exp(-((2045 - construction_year) / L)^k))
```

against the specific space-heating targets (`Nutzwärmeenergie` per m² net heated area) from `Genehm_SR_2025Strom.pdf` (Scenario B 2045):
- **Residential (Table 2, p. 26):** `103.8 kWh_th / (m²·a)` (unrenovated GT baseline: `141.03 kWh_th / (m²·a)`)
- **Non-residential / GHD (Table 5, p. 29):** `90.6 kWh_th / (m²·a)` (unrenovated GT baseline: `117.64 kWh_th / (m²·a)`)

## Algorithm (`main.py`)

1. Lists all ground-truth grids from `DATA_GRID_DIR` (`/dss/.../3rd_batch/1.Grid_Sampled/`) and shuffles them deterministically (`seed = 42`).
2. Iterates through progressive batch sizes: `50 -> 100 -> 200 -> 400 -> 800 -> all grids`.
3. At the start of each iteration, clears all `.h5` files in `STORAGE_DIR` (`/dss/.../3rd_batch/NAP_determine_Gebäudesanierungsrate/`) so only the latest iteration's `.h5` files remain.
4. Simulates space-heating demand in parallel across `n_cpu` worker processes (skipping rooftop PV, OpenDHW hot water, HP COP, mobility, MV disconnection, and pre-powerflow while retaining electrical internal heat gains).
5. Computes the area-weighted specific space-heating demand (`sum(space_heat) / sum(0.75 * floor_area)`) separately for purely residential and purely GHD buildings, updates `p_max_res` and `p_max_ghd`, and logs results to `STORAGE_DIR/tuning_history.csv`.

## Usage

### On LRZ Cluster (SLURM)
```bash
sbatch run_cluster_serialstd.sh
```

### Local Test Run
```bash
conda activate inference
python3 main.py --n_cpu 2 \
  --input_dir ../../0.Legacy_Folders/GridExpand/2.demand_allocation/gridalloc/data/grids \
  --output_dir /tmp/test_tune_pmax \
  --batch_sizes 2,4 --max_grids 4
```
