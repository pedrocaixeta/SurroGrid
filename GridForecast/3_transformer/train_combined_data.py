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



manual_cfg = {
    # Preprocessing settings
    '_data': {
        'hdf_data_path': '/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/0_preprocessing/Data/ts_train.h5',
        'key_X': "X",
        'key_y': "y",
        'train_grids': "all",
        'test_ratio': 0.2,
        'random_state': 42,
        'X_BASE_COLS':        ['T',
                               'demand_net_active_pre',
                               'heat_water', 
                               'heat_space', 
                               'cop_avg',
                               'PV_prod_expected',
                               'res_bldng_area_base_sum', 'nonres_bldng_area_base_sum', 'mixed_bldng_area_base_sum', 'bldng_area_floors_sum',
                               'n_cars', 'n_pure_res_buildings', 'n_pure_nonres_buildings', 'n_mixed_buildings', 'n_flats', 'n_occ',
                               'n_lines', 'tot_R_grid', 'regiostar7',
                              ],
        'TARGET_COLS':        ["demand_net_active_post", "demand_net_reactive_post"],
        'ZERO_BASE_FEATURES': [],
        'LOG1P_COLS':         ["n_lines", "tot_R_grid"],
        'TS_PERIODS':         [8760, 168, 24, 12],
        'VMD_COLS':           ["demand_net_active_pre", "heat_water", "heat_space", "cop_avg", "PV_prod_expected", "T"],
        'VMD_K_MODES':         2,
        'VMD_APPROACH':        "read",  # None | read | write
    },
    # Model / architecture
    'input_mlp_layers': 1,
    'input_mlp_hidden': 512,
    'input_dropout': 0.07186070916348294,
    'output_mlp_layers': 3,
    'output_mlp_hidden': 256,
    'output_dropout': 0.2153543986687135,
    'd_model': 256,
    'num_heads': 8,
    'ff_dim': 1024,
    'attn_dropout': 0.09134270625090402,
    'ffn_dropout': 0.1250813755097966,
    'num_transformer_layers': 1,
    'activation': 'gelu',
    # Aggregation settings
    'core_len': 120,
    'pad_hours': 96,
    'aggregation_mode': "conv",
    "conv_padding": 6,
    'agg_hours': 1,
    # Training
    'learning_rate': 0.0017054877358583258,
    'weight_decay': 0.06661963941968659,
    'optimizer': 'adamw',
    'scheduler': 'plateau',
    'scheduler_epochs': 38,
    'epochs': 38,
    'batch_size': 128,
    'grad_clip': 0,
    'loss_type': 'mae_maex',  # try: 'mae_maex', 'alpha_peak'
    'multi_target_lambda': 1,
    'patience': 3,
    # reporting and Others
    'seed': 42,
    "num_workers": 14,
    'compile_model': True,
    'bottom_rung_report': None,
    'full_metrics_every': 5,
}

test_data_cfg = {
    'hdf_data_path': '/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/0_preprocessing/Data/ts_test.h5',
    'key_X': 'X',
    'key_y': 'y',
}

# 1. Initialize trainer
transformer_trainer = TransformerTrainer(manual_cfg) #creates an instance of the class TransformerTrainer
print('Train batches:', len(transformer_trainer.train_loader), ', Val batches:', len(transformer_trainer.val_loader))
print('Input dims ->', transformer_trainer.cfg.in_features, ', Output dims ->', transformer_trainer.cfg.out_features)

# 2. Train on combined train + validation data
transformer_trainer.train_on_trainval(epochs=38) #returns a dictionary with the information of epochs and the final loss

# 3. Save model
model_name = "baseline_mae_maex_oldHPO"
models_dir = "/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/3_transformer/models"
model_dir = os.path.join(models_dir, model_name)
os.makedirs(model_dir, exist_ok=True)

ckpt_path = os.path.join(model_dir, f"{model_name}.pt")
transformer_trainer.save(ckpt_path, use_best=True)
print(f"Model saved to: {ckpt_path}")

# 4. Evaluate metrics with best weights on test set and save plots
test_results = transformer_trainer.evaluate_on_test_with_plots(test_data_cfg, save_dir=model_dir)
print("Test results:", test_results)