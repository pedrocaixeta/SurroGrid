# Inference — Unified Direct Pipeline

End-to-end single-grid inference runner that executes all three verified SurroGrid inference stages in a single SLURM job, using node-local temp staging (`$TMPDIR`) and writing a single `<grid>_direc_Inferred.h5` file into `STORAGE_DIR` only after all stages succeed.

```text
0.Raw_grids/<grid>.h5
  -->  [Stage 1: 1.Cheap_Ground_Truth]        (OMP/MKL/OPENBLAS=1)
  -->  [Stage 2: 2.Feature_and_Preprocessing] (OMP/MKL/OPENBLAS=1)
  -->  [Stage 3: 3.Forward_Pass]              (OMP/MKL=n_cpu, CPU PyTorch)
  -->  Directly_Inferred/<grid>_direc_Inferred.h5
```

The standalone step-by-step folders (`1.Cheap_Ground_Truth/`, `2.Feature_and_Preprocessing/`, `3.Forward_Pass/`) remain completely intact and are invoked as isolated subprocesses — zero duplicated code, zero `sys.modules` collisions, and clean per-stage BLAS/OpenMP threading.

## Output HDF5 Keys (`--keep_mode`)

By default (`keep_mode="medium"`, ~40 MB vs ~120 MB full), `main.py` repacks the final HDF5 file cleanly into `STORAGE_DIR` (atomically renamed to `*_direc_Inferred.h5` after success):

| Mode | Approx. Size | Retained Keys |
| --- | ---: | --- |
| `slim` | ~5-6 MB | `raw_data/*`, `pwrflw/output/pre/demand_import`, `surrogate_in/features_raw`, `surrogate_out/*`, `performance` |
| `medium` **(default)** | ~40 MB | `slim` + `urbs_in/*` (enables re-running Step 2 PV/HP feature tweaks without re-running Step 1) |
| `full` | ~120 MB | All HDF5 keys identical to the step-by-step output |

The `/performance` table preserves all three per-stage rows (`demand_allocation`, `feature_generation`, `forward_pass`) plus a consolidated `direct_inference` row.

## Configuration

Edit [pipeline_config.py](pipeline_config.py) to change default paths or retention mode:

- **`DATA_GRID_DIR`**: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/0.Raw_grids/`
- **`STORAGE_DIR`**: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Directly_Inferred/`
- **`OUTPUT_SUFFIX`**: `_direc_Inferred`
- **`DEFAULT_KEEP_MODE`**: `medium`

## Environment & SLURM Resources

- **Conda environment (LRZ)**: `inference` (includes all Stage 1-2 packages plus CPU-only PyTorch via `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
- **SLURM script**: [run_cluster_serialstd.sh](run_cluster_serialstd.sh) — `serial_std`, 4 CPUs, 8 GB RAM, 20 min walltime (~8 min for Step 1 on a 125-building grid + ~6 s for Steps 2-3).

## Usage

### On LRZ Linux Cluster (SLURM)
```bash
sbatch run_cluster_serialstd.sh 0197
```

### Local Test Run (WSL)
Locally, the `inference` environment omits PyTorch while `training` provides it, so pass `--torch_python`:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pedro/miniconda3/envs/inference/bin/python main.py 51433 \
  --n_cpu 4 \
  --input_dir ../Grid_Importing/gridreadout/results \
  --output_dir /tmp/inference_direct_test \
  --torch_python /home/pedro/miniconda3/envs/training/bin/python
```

To re-run only Stages 2-3 on an existing `*_DmdAllocated.h5` file (e.g., for parity checks or feature tweaks):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pedro/miniconda3/envs/inference/bin/python main.py 0197 \
  --n_cpu 4 \
  --start_from_step 2 \
  --input_dir 1.Cheap_Ground_Truth/gridalloc/results \
  --output_dir /tmp/inference_direct_test \
  --torch_python /home/pedro/miniconda3/envs/training/bin/python
```
