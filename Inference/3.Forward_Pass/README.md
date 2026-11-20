# Inference — Step 3: Surrogate Forward Pass

Runs the trained baseline Transformer surrogate (`baseline_alpha_peak_HPOfromOct17th.pt`) on the scaled `8760 x 49` features produced by Step 2 and outputs the predicted post-expansion hourly transformer load (`p_mw`, `q_mvar`, `s_mva`, and `loading_pct` vs transformer nominal rating `sn_mva`).

```text
Step_by_Step/2.Featured_and_Preprocessed/<grid>_DmdAllocated_Features.h5
  -->  [Step 3: Surrogate Forward Pass]
  -->  Step_by_Step/3.Forward_Past/<grid>_DmdAllocated_Features_Predicted.h5
```

## What It Does (`main.py`)

For a given grid ID, `main.py` copies `<grid>_DmdAllocated_Features.h5` into `STORAGE_DIR` and runs:

1. **Load Inputs & Checkpoint (`transformer_model.py`)**
   - Reads `/surrogate_in/features_scaled` (`8760 x 49`) and validates exact column order against `scalers_baseline_alpha_peak_HPOfromOct17th.json`.
   - Loads `baseline_alpha_peak_HPOfromOct17th.pt` on CPU (automatically stripping `torch.compile`'s `_orig_mod.` state-dict prefix with `strict=True`).
2. **Forward Pass & Inverse Scaling**
   - Builds circularly-padded daily windows (`365 x 216 x 49`, `core_len=24`, `pad_hours=96`), runs inference (~1–2 s on 4 CPU threads), and inverse-scales `(P, Q)` back to MW / MVAr using the frozen training `scaler_y`.
3. **Transformer Loading Assessment & Output**
   - Reads total in-service transformer nominal capacity `sn_mva` from `/raw_data/net`, computes `s_mva = sqrt(p_mw^2 + q_mvar^2)` and `loading_pct = (s_mva / sn_mva) * 100`.
   - Saves two tables:
     - `/surrogate_out/demand_import` — 8,760 hourly rows (`p_mw`, `q_mvar`, `s_mva`, `loading_pct`).
     - `/surrogate_out/summary` — 1-row grid summary (`model_name`, `trafo_sn_mva`, `peak_p_import_mw`, `peak_p_feedin_mw`, `peak_s_mva`, `max_loading_pct`, `hours_overloaded_100pct`, `needs_replacement`).
4. **Completion & Logging**
   - Logs `/performance` metrics and appends `_Predicted` (`<grid>_DmdAllocated_Features_Predicted.h5`) **only upon full success**.

## Configuration

Paths and checkpoint settings live in [pipeline_config.py](pipeline_config.py):

- **`DATA_GRID_DIR`**: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Step_by_Step/2.Featured_and_Preprocessed/`
- **`STORAGE_DIR`**: `/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Step_by_Step/3.Forward_Past/`
- **`OUTPUT_SUFFIX`**: `_Predicted`
- **`MODEL_PATH`**: `./baseline_alpha_peak_HPOfromOct17th.pt`
- **`SCALER_PATH`**: `./scalers_baseline_alpha_peak_HPOfromOct17th.json`

## Environment & SLURM Resources

- **Conda environment**: `inference` with CPU-only PyTorch installed:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
  *(Locally in WSL, if `inference` intentionally omits PyTorch, run this step using the `training` conda environment instead.)*
- **SLURM script**: [run_cluster_serialstd.sh](run_cluster_serialstd.sh) — `serial_std`, 4 CPUs, 4 GB RAM, 5 min walltime (~2 s runtime per grid; no GPU node needed).
- **Note on PyTorch warning**: `UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True` is completely harmless and expected for Pre-LN Transformers.

## Usage

### On LRZ Linux Cluster (SLURM)
```bash
sbatch run_cluster_serialstd.sh 0197
```

### Local Test Run
```bash
conda activate training   # or 'inference' if CPU-only torch is installed
python3 main.py 0197 --n_cpu 4 \
  --input_dir /tmp/inference_step2_test \
  --output_dir /tmp/inference_step3_test
```
