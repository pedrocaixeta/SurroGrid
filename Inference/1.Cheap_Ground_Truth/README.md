# Inference — Step 2: Demand Allocation & Pre-Expansion Powerflow

Generates per-building demand tables and pre-expansion power-flow results needed by the **baseline (without-mobility)** Transformer surrogate model from raw `.h5` files produced by `1.Grid_Importing`.

## Workflow (`main.py`, conda env: `inference`)

For a given grid ID, `main.py` orchestrates:
1. **`gridalloc/`**: Solar PV capacity factors (`urbs_in/supim`, `urbs_in/process`), per-building electricity, space heat, and water heat demands (`urbs_in/demand`), air-source heat pump COP (`urbs_in/eff_factor`), and statistical car ownership (`n_cars_tot` in `raw_data/buildings`, skipping `emobpy`).
2. **`disconnect_MV_buildings/`**: Marks buildings exceeding peak electricity thresholds (`>100 kW` pure non-residential, `>250 kW` residential/mixed) as `in_service = False` in `raw_data/net` and zeros their demands in `urbs_in/demand`.
3. **`pre_powerflow/`**: Computes pre-expansion active/reactive bus loads (`PF_ELC = 0.959`), runs 8760-hour pandapower BFSW load flow, and saves `/pwrflw/input/demand_pre` and `/pwrflw/output/pre/{demand_import, vm, line_loads}`.
4. **Completion**: Records `/performance` metrics and appends `_DmdAllocated` to the output filename only after all steps succeed.

## Configuration

Paths are defined in [pipeline_config.py](pipeline_config.py):
- `DATA_GRID_DIR`: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/1.Raw_grids/`
- `STORAGE_DIR`: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/2.Dmd_Allocated/`
- `OUTPUT_SUFFIX`: `_DmdAllocated`

## Environment

Create the universal `inference` conda environment from `Inference/environment_inference.yml`:
```bash
conda env create -f ../environment_inference.yml
```

## Usage

### On LRZ Linux Cluster (SLURM)
```bash
sbatch run_cluster_serialstd.sh 51431
```

### Local Test Run
```bash
conda activate inference
python3 main.py 51433 --n_cpu 4 \
  --input_dir ../../1.Grid_Importing/gridreadout/results \
  --output_dir /tmp/inference_test
```
