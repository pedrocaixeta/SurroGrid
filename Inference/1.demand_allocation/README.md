# Inference — Step 1: Demand Allocation & Pre-Expansion Powerflow

Generates all per-building demand profiles and pre-expansion powerflow outputs needed by the **baseline (without-mobility)** Transformer surrogate model from raw grid `.h5` files produced by `1.Grid_Importing`.

## Pipeline Overview

1. **Step 1a (`main_demand_allocation.py`, conda env: `grid_alloc`)**
   - Uses unmodified demand-generation logic from `GridExpand/2.demand_allocation/gridalloc`:
     - Solar roof sampling + 1 kW AC capacity factors (`urbs_in/supim`, `urbs_in/process`)
     - Per-building electrical demand (`urbs_in/demand` -> `electricity`)
     - Per-building space heat, domestic hot water (`urbs_in/demand` -> `space_heat`, `water_heat`), and air-source heat pump COP (`urbs_in/eff_factor` -> `heatpump_air`)
     - Statistical car ownership (`n_cars_tot` in `raw_data/buildings`, skipping `emobpy`)
   - Copies the raw `.h5` file into `STORAGE_DIR` and saves all per-building tables.
   - Applies MV-level building disconnection (`>100 kW` pure non-residential, `>250 kW` residential/mixed): sets `in_service = False` in `raw_data/net` and zeros disconnected buses in `urbs_in/demand`.

2. **Step 1b (`main_pre_powerflow.py`, conda env: `pwrflw-hpc`)**
   - Builds per-bus active/reactive pre-expansion demand (`PF_ELC = 0.959`) and saves `/pwrflw/input/demand_pre`.
   - Runs 8760-hour pandapower BFSW load flow (identical to `GridExpand/4.powerflow`) and saves:
     - `/pwrflw/output/pre/demand_import` (`p_mw`, `q_mvar`)
     - `/pwrflw/output/pre/vm`
     - `/pwrflw/output/pre/line_loads`
   - Renames the completed file to `<original_stem>_DmdAllocated.h5` only after all steps succeed.

## Configuration

Edit the top of `config.py` to change paths:
- `DATA_GRID_DIR`: input directory containing raw `.h5` files
- `STORAGE_DIR`: output directory for `*_DmdAllocated.h5` files
- `OUTPUT_SUFFIX`: default `_DmdAllocated`

## Usage

### On LRZ Linux Cluster (SLURM)
```bash
# Single grid ID
sbatch run_cluster_serialstd.sh 51431

# Range of grid IDs (inclusive)
./start_batch_jobs_serialstd.sh 51431 51436
```

### Local Test Run
```bash
conda run -n grid_alloc python main_demand_allocation.py 51433 --n_cpu 4 \
  --input_dir ../../1.Grid_Importing/gridreadout/results \
  --output_dir /tmp/inference_test

conda run -n pwrflw-hpc python main_pre_powerflow.py 51433 --n_cpu 4 \
  --output_dir /tmp/inference_test
```
