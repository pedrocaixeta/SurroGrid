"""
This script trains the Transformer-based time series forecasting model on the combined train + validation data.
At the end, it performs an evaluation of this model considering the test data
"""

import subprocess, sys

# Ensure required dependencies are installed in the container environment before importing project modules
for pkg in ["tables", "vmdpy"]:
    try:
        __import__(pkg)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
        except Exception as e:
            print(f"Warning: could not install {pkg}: {e}")

import torch, math, os, json
from transformer_model import TransformerTrainer, TransformerConfig

# Strategy to make the job run faster in the HPC: Enable TensorFloat-32 (TF32) for dramatic speedup on Ampere (A100) and Hopper (H100) Tensor Cores
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')


##################### SETTINGS #####################
directory_with_data = '/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/0_preprocessing/Data/'
loss_function = 'mae_maex' # 'mae_maex' | 'alpha_peak'
transformer_approach = 'baseline' # 'baseline' | 'mobility' | 'small_grids'
model_name = str(transformer_approach) + '_' + str(loss_function) + '_HPOfromOct15th'
###################################################

common_features_for_all_approaches = [# Grouped according to Table 3.2 in Elias' Thesis
    'demand_net_active_pre', 'heat_water', 'heat_space', 'cop_avg', 'PV_prod_expected', #Cheap Ground Truth
    'regiostar7', 'n_cars', #'T' #Mobility proxies
    'n_lines', 'tot_R_grid', #Power Flow Proxies
    'n_pure_res_buildings', 'n_pure_nonres_buildings', 'n_mixed_buildings', #Disaggregation - Number of each building type                                     
    'res_bldng_area_base_sum', 'nonres_bldng_area_base_sum', 'mixed_bldng_area_base_sum', #Disaggregation - Sum of floor area of each building type
    'bldng_area_floors_sum', 'n_flats', 'n_occ'] #Disaggregation - Other building characteristics
    # 'mobility', 'mob_avail', 'mob_last_avail'] # Mobility Ground Truth

common_vmd_col = ["demand_net_active_pre", "heat_water", "heat_space", "cop_avg", "PV_prod_expected"]
      

if transformer_approach == 'baseline':
    Path_to_training_data = directory_with_data + 'ts_train.h5'
    Path_to_TEST_data = directory_with_data + 'ts_test.h5'
    features = common_features_for_all_approaches + ['T']
    vmd_cols = common_vmd_col + ["T"]

elif transformer_approach == 'mobility':
    Path_to_training_data = directory_with_data + 'ts_train.h5'
    Path_to_TEST_data = directory_with_data + 'ts_test.h5'
    features = common_features_for_all_approaches + ['mobility', 'mob_avail', 'mob_last_avail']
    vmd_cols = common_vmd_col + ['mobility', 'mob_avail', 'mob_last_avail']

elif transformer_approach == 'small_grids':
    Path_to_training_data = directory_with_data + 'ts_train_large_grids.h5'
    Path_to_TEST_data = directory_with_data + 'ts_test_large_grids.h5'
    features = common_features_for_all_approaches + ['mobility', 'mob_avail', 'mob_last_avail']
    vmd_cols = common_vmd_col + ['mobility', 'mob_avail', 'mob_last_avail']




manual_cfg = {
    # Preprocessing settings
    '_data': {
        'hdf_data_path': Path_to_training_data,
        'key_X': "X",
        'key_y': "y",
        'train_grids': "all",
        'validation_ratio': 0.2, # validation split ratio applied to grids in ts_train.h5
        'random_state': 42,
        'X_BASE_COLS': features,
        'TARGET_COLS':        ["demand_net_active_post", "demand_net_reactive_post"],
        'ZERO_BASE_FEATURES': [],
        'LOG1P_COLS':         ["n_lines", "tot_R_grid"],
        'TS_PERIODS':         [8760, 168, 24, 12],
        'VMD_COLS':           vmd_cols,
        'VMD_K_MODES':         3,
        'VMD_APPROACH':        "read",  # None | read | write
    },
    # Model / architecture
    'input_mlp_layers': 2,
    'input_mlp_hidden': 128,
    'input_dropout': 0.0518586,
    'output_mlp_layers': 1,
    'output_mlp_hidden': 512,
    'output_dropout': 0.114268,
    'd_model': 256,
    'num_heads': 8,
    'ff_dim': 1024,
    'attn_dropout':  0.0594138,
    'ffn_dropout':  0.0531122,
    'num_transformer_layers': 1,
    'activation': 'gelu',
    # Aggregation settings
    'core_len': 24,
    'pad_hours': 96,
    'aggregation_mode': "conv",
    "conv_padding": 6,
    'agg_hours': 1,
    # Training
    'learning_rate': 0.000236578,
    'weight_decay':  0.00830902,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'scheduler_epochs': 45,
    'epochs': 43,
    'batch_size': 128,
    'grad_clip': 0,
    'loss_type': loss_function,  #'mae_maex', 'alpha_peak'
    'multi_target_lambda': 1,
    'patience': 3,
    # reporting and Others
    'seed': 42,
    "num_workers": 14,
    'compile_model': True,
    'bottom_rung_report': 5,
    'full_metrics_every': 5,
}

test_data_cfg = {
    'hdf_data_path': Path_to_TEST_data,
    'key_X': 'X',
    'key_y': 'y',
}

# 1. Initialize trainer
transformer_trainer = TransformerTrainer(manual_cfg) #creates an instance of the class TransformerTrainer
print('Train batches:', len(transformer_trainer.train_loader), ', Val batches:', len(transformer_trainer.val_loader))
print('Input dims ->', transformer_trainer.cfg.in_features, ', Output dims ->', transformer_trainer.cfg.out_features)

# 2. Train on combined train + validation data
#returns a dictionary with the information of epochs and the final loss
transformer_trainer.train_on_trainval(epochs=int(round( manual_cfg['epochs']*(1.0 - manual_cfg['_data']['validation_ratio'] )))) #Assumming that manual_cfg['epochs'] is the optimal number of epochs outputted from the HPO (without validation data)

# 3. Save model
models_dir = "/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/3_transformer/models"
model_dir = os.path.join(models_dir, model_name)
os.makedirs(model_dir, exist_ok=True)

ckpt_path = os.path.join(model_dir, f"{model_name}.pt")
transformer_trainer.save(ckpt_path, use_best=True)
print(f"Model saved to: {ckpt_path}")

# 4. Evaluate metrics with best weights on test set and save plots
test_results = transformer_trainer.evaluate_on_test_with_plots(test_data_cfg, save_dir=model_dir)
print("Test results:", test_results)

# 5. Evaluate MdAPE metrics and percentile spreads for Table 4.1 (Elias's Thesis)
print("\n" + "="*50)
print("Evaluating MdAPE metrics on test set (Table 4.1 format)...")
mdape_results = transformer_trainer.evaluate_on_test_mdape(test_data_cfg)

table41_rows = {
    "P": [
        ("Agg. Feed-In MdAPE [%]", "Agg. Feed-In MdAPE", "Agg. Feed-In MdAPE +(p95-MdAPE)", "Agg. Feed-In MdAPE -(MdAPE-p5)", 100.0),
        ("Agg. Import MdAPE [%]", "Agg. Import MdAPE", "Agg. Import MdAPE +(p95-MdAPE)", "Agg. Import MdAPE -(MdAPE-p5)", 100.0),
        ("Median NMAE [%]", "MA(E/MAEx) Md (1h)", "MA(E/MAEx) Md +(p95-Md) (1h)", "MA(E/MAEx) Md -(Md-p5) (1h)", 100.0),
        ("Peak Feed-In MdAPE [%]", "Peak Feed-In MdAPE (1h)", "Peak Feed-In MdAPE +(p95-MdAPE) (1h)", "Peak Feed-In MdAPE -(MdAPE-p5) (1h)", 100.0),
        ("Peak Feed-In Median Δt [d]", "Peak Feed-In Median Δt (1h)", "Peak Feed-In Median Δt +(p95-Md) (1h)", "Peak Feed-In Median Δt -(Md-p5) (1h)", 1.0 / 24.0),
        ("Peak Import MdAPE [%]", "Peak Import MdAPE (1h)", "Peak Import MdAPE +(p95-MdAPE) (1h)", "Peak Import MdAPE -(MdAPE-p5) (1h)", 100.0),
        ("Peak Import Median Δt [d]", "Peak Import Median Δt (1h)", "Peak Import Median Δt +(p95-Md) (1h)", "Peak Import Median Δt -(Md-p5) (1h)", 1.0 / 24.0),
    ],
    "Q": [
        ("Agg. Feed-In MdAPE [%]", "Agg. Feed-In MdAPE", "Agg. Feed-In MdAPE +(p95-MdAPE)", "Agg. Feed-In MdAPE -(MdAPE-p5)", 100.0),
        ("Median NMAE [%]", "MA(E/MAEx) Md (1h)", "MA(E/MAEx) Md +(p95-Md) (1h)", "MA(E/MAEx) Md -(Md-p5) (1h)", 100.0),
        ("Peak Import MdAPE [%]", "Peak Import MdAPE (1h)", "Peak Import MdAPE +(p95-MdAPE) (1h)", "Peak Import MdAPE -(MdAPE-p5) (1h)", 100.0),
    ],
    "S": [
        ("Agg. Load MdAPE [%]", "Agg. Import MdAPE", "Agg. Import MdAPE +(p95-MdAPE)", "Agg. Import MdAPE -(MdAPE-p5)", 100.0),
        ("Median NMAE [%]", "MA(E/MAEx) Md (1h)", "MA(E/MAEx) Md +(p95-Md) (1h)", "MA(E/MAEx) Md -(Md-p5) (1h)", 100.0),
        ("Peak Load MdAPE [%]", "Peak Import MdAPE (1h)", "Peak Import MdAPE +(p95-MdAPE) (1h)", "Peak Import MdAPE -(MdAPE-p5) (1h)", 100.0),
        ("Peak Load Median Δt [d]", "Peak Import Median Δt (1h)", "Peak Import Median Δt +(p95-Md) (1h)", "Peak Import Median Δt -(Md-p5) (1h)", 1.0 / 24.0),
    ],
}

def format_table41_value(metrics, key, scale):
    value = metrics.get(key)
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value) * scale:.4f}".replace(".", ",")


table41_report = ["Table 4.1 results (tab-separated; decimal comma)"]
for target_name, metric_rows in table41_rows.items():
    display_name = "|S|" if target_name == "S" else target_name
    table41_report.extend([
        "",
        display_name,
        "Metric\tMedian\tp95 (+/-)\tp5 (+/-)",
    ])
    metrics = mdape_results.get(target_name, {})
    for label, median_key, p95_key, p5_key, scale in metric_rows:
        values = [format_table41_value(metrics, key, scale) for key in (median_key, p95_key, p5_key)]
        table41_report.append("\t".join([label, *values]))

table41_text = "\n".join(table41_report)
print("\n" + table41_text)
table41_path = os.path.join(model_dir, "table41_metrics.tsv")
with open(table41_path, "w", encoding="utf-8") as report_file:
    report_file.write(table41_text + "\n")
print(f"Table 4.1 results saved to: {table41_path}")