# Inference — Step 3: Surrogate Forward Pass

Runs the trained Transformer surrogate (`baseline_alpha_peak_HPOfromOct17th.pt`) on the scaled 49-feature input produced by `2.Feature_and_Preprocessing` and outputs the predicted hourly transformer load (`p_mw`, `q_mvar`, `s_mva`, and `loading_pct` vs transformer `sn_mva`).

## Workflow (`main.py`)

For a given grid ID:
1. Copies `<stem>_DmdAllocated_Features.h5` from `DATA_GRID_DIR` into `STORAGE_DIR`.
2. Loads `/surrogate_in/features_scaled` (`8760 x 49`), `baseline_alpha_peak_HPOfromOct17th.pt` (automatically stripping `torch.compile`'s `_orig_mod.` prefix with `strict=True`), and `scalers_baseline_alpha_peak_HPOfromOct17th.json`.
3. Assembles circularly-padded windows (`365 x 216 x 49`), runs the Transformer forward pass (~1-2 s on 4 CPU threads), and inverse-scales `(P, Q)` to MW / MVAr using `scaler_y`.
4. Computes hourly `s_mva = sqrt(p_mw^2 + q_mvar^2)` and transformer `loading_pct` against `sn_mva` from `/raw_data/net`.
5. Saves `/surrogate_out/demand_import` and `/surrogate_out/summary`, logs `/performance`, and appends `_Predicted` (`<stem>_DmdAllocated_Features_Predicted.h5`).

## Configuration & Usage

Paths and checkpoint settings live in `pipeline_config.py`.

```bash
# On LRZ Linux Cluster (CPU, ~2 s per grid)
sbatch run_cluster_serialstd.sh 0197

# Local test
conda activate training
python3 main.py 0197 --n_cpu 4 \
  --input_dir /tmp/feat_test \
  --output_dir /tmp/pred_test
```
