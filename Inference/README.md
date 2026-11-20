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

## Surrogate Model Choice: Why `mae_maex` (NMAE) Instead of `alpha_peak`?

Even though the end goal is transformer sizing based on annual **peak apparent load `|S|`** (`max(|S|) > sn_mva`), we intentionally use **`baseline_mae_maex_HPOfromOct15th.pt`** (trained with normalized MAE / `mae_maex` loss; 45 features: VMD `K=3`, seasonal periods `[8760, 168, 24, 12]`) rather than `baseline_alpha_peak_HPOfromOct17th.pt` (trained with peak-weighted `alpha_peak` loss; 49 features: VMD `K=4`, `[8760, 168, 24]`).

On the held-out test set (`train_combined-5862890.out` vs `train_combined-5863019.out`), the `mae_maex` model unexpectedly outperformed `alpha_peak` on **peak load prediction itself** as well as whole-year accuracy:

| Held-out Test Metric (1h) | `baseline_mae_maex` (NMAE) | `baseline_alpha_peak` |
| --- | ---: | ---: |
| `|S|` Peak Load MdAPE | **4.15%** | 5.76% |
| `|S|` Peak Load Mean MAPE | **6.67%** | 7.47% |
| `|S|` Peak Load R^2 | **0.9839** | 0.9690 |
| `|S|` Peak Load 95th percentile (`Median + (p95 - Median)`) | **16.86%** | 17.65% |
| `P` Peak Import MdAPE / R^2 | **4.31%** / **0.9830** | 5.10% / 0.9688 |
| `Q` Peak Import MdAPE / R^2 | **4.34%** / **0.9787** | 6.58% / 0.9666 |
| `|S|` Median NMAE (full year) | **9.10%** | 9.62% |
| Phase Angle MAE | **4.62 deg** | 5.70 deg |

In addition, visual inspection of `S_07_peak_demand_grid_batched.png` and `P_07_peak_demand_grid_batched.png` confirmed symmetric scatter along the 1:1 diagonal across the entire range (`0` to `1.75 MVA`) with no systematic under-prediction of peaks. `2.Feature_and_Preprocessing` automatically reads `VMD_K_MODES` and `TS_PERIODS` from the paired `scalers_*.json`, so switching models only requires updating `MODEL_PATH` and `SCALER_PATH`.

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
- **Single-grid SLURM script**: [infer_single_grid.sh](infer_single_grid.sh) — `serial_long` (`--qos=cm4_serial_long`), 4 CPUs, 6 GB RAM, 20 min walltime (~8 min for Step 1 on a 125-building grid + ~6 s for Steps 2-3).
- **DSO batch SLURM array script**: [infer_all_grids_from_dso.sh](infer_all_grids_from_dso.sh) — self-submitting SLURM job array (`serial_long` + `--qos=cm4_serial_long` by default; override with `--partition serial_std --qos ""`) that scans `0.Raw_grids/` for all grids belonging to a DSO short code (`AllgN`, `ByWrk`, `LEW`, `MFrak`, `Nrgie`, `RegB`, `SWA`, `SWM`, `SWU`, `nVNB`), automatically skips grids that already have `*_direc_Inferred.h5` in `Directly_Inferred/`, and submits a throttled job array (`%50` concurrency by default).

## Usage

### On LRZ Linux Cluster (SLURM)

Single grid:
```bash
sbatch infer_single_grid.sh 0197
```

All grids of a DSO (run directly on `cm4login1`, **not** wrapped in `sbatch`):
```bash
# Preview matching / already-done / pending counts without submitting
./infer_all_grids_from_dso.sh SWU --dry-run

# Smoke-test first 5 pending grids
./infer_all_grids_from_dso.sh SWU --limit 5

# Submit all pending grids for the hardcoded default DSO (or pass a DSO short code)
./infer_all_grids_from_dso.sh Nrgie

# For very large DSOs (e.g., ByWrk with 38k grids), chunk multiple grids per array task
./infer_all_grids_from_dso.sh ByWrk --grids-per-task 5 --max-concurrent 50
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
