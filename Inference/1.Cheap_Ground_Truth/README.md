# Inference — Step 1: Cheap Ground Truth (Demand Allocation & Pre-Expansion Powerflow)

Generates the per-building demand profiles and pre-expansion power-flow results required by the **baseline (without-mobility)** Transformer surrogate model, starting from raw `.h5` grid files exported by `Grid_Importing/`.

```text
0.Raw_grids/<grid>.h5  -->  [Step 1: Cheap Ground Truth]  -->  Step_by_Step/1.Cheap_Ground_Truthed/<grid>_DmdAllocated.h5
```

## What It Does (`main.py`)

For a given grid ID, `main.py` copies the raw `.h5` file into `STORAGE_DIR` and runs three stages:

1. **Demand Allocation (`gridalloc/`)**
   - Computes solar PV capacity factors (`/urbs_in/supim`, `/urbs_in/process`), per-building electricity, space heat, and domestic hot water demands (`/urbs_in/demand`), air-source heat pump COP (`/urbs_in/eff_factor`), and statistical car counts (`n_cars_tot` in `/raw_data/buildings`, skipping slow `emobpy` mobility profiles).
2. **MV Building Exclusion (`disconnect_MV_buildings/`)**
   - Flags oversized buildings (`>100 kW` peak electricity for pure non-residential; `>250 kW` for residential/mixed) as `in_service = False` in `/raw_data/net` and zeros their demands in `/urbs_in/demand`.
3. **Pre-Expansion Load Flow (`pre_powerflow/`)**
   - Builds hourly active/reactive bus loads (`PF_ELC = 0.959`), runs an 8,760-hour pandapower BFSW power flow, and writes `/pwrflw/input/demand_pre` and `/pwrflw/output/pre/{demand_import, vm, line_loads}`.
4. **Completion & Logging**
   - Appends timing/memory metrics to `/performance` and renames the output file with `_DmdAllocated` **only after all stages succeed**.

## Configuration

Paths and suffixes are configured in [pipeline_config.py](pipeline_config.py):

- **`DATA_GRID_DIR`**: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/0.Raw_grids/`
- **`STORAGE_DIR`**: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Step_by_Step/1.Cheap_Ground_Truthed/`
- **`OUTPUT_SUFFIX`**: `_DmdAllocated`

## Environment & SLURM Resources

- **Conda environment**: `inference` (created from `Inference/environment_inference.yml`: `conda env create -f ../environment_inference.yml`).
- **SLURM script**: [run_cluster_serialstd.sh](run_cluster_serialstd.sh) — `serial_std`, 4 CPUs, 8 GB RAM, 11 min walltime (`OMP_NUM_THREADS=1` to avoid BLAS oversubscription across worker processes).

## Usage

### On LRZ Linux Cluster (SLURM)
```bash
sbatch run_cluster_serialstd.sh 0197
```

### Local Test Run
```bash
conda activate inference
python3 main.py 51433 --n_cpu 4 \
  --input_dir ../../Grid_Importing/gridreadout/results \
  --output_dir /tmp/inference_step1_test
```
