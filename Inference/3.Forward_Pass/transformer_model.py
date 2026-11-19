"""Inference-only Transformer surrogate architecture and forward-pass helpers.

Trimmed from `0.Legacy_Folders/GridForecast/3_transformer/transformer_model.py`
to keep ONLY the model architecture (`TransformerConfig`,
`SinusoidalPositionalEncoding`, `PerTimestepMLP`, `TransformerForecastModel`),
strict checkpoint loading (stripping `torch.compile`'s `_orig_mod.` prefix),
circularly-padded window assembly, and output inverse-scaling.
"""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


# ============================================================
# Config
# ============================================================
@dataclass
class TransformerConfig:
    """Architecture and windowing hyperparameters loaded from `.pt` checkpoint."""
    core_len: int = 24
    pad_hours: int = 96
    agg_hours: int = 1
    in_features: int = 49
    out_features: int = 2

    input_mlp_layers: int = 3
    input_mlp_hidden: int = 128
    input_dropout: float = 0.0

    d_model: int = 256
    num_heads: int = 8
    ff_dim: int = 1024
    num_transformer_layers: int = 1
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0

    output_mlp_layers: int = 3
    output_mlp_hidden: int = 512
    output_dropout: float = 0.0

    activation: str = "gelu"
    aggregation_mode: str = "conv"  # 'sum' | 'conv'
    conv_padding: int = 6

    @classmethod
    def from_dict(cls, cfg_dict: Dict) -> "TransformerConfig":
        """Build a TransformerConfig ignoring training-only keys in the checkpoint."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        return cls(**filtered)


def get_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = name.lower()
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
    }[name]


# ============================================================
# Model Components (exact match with training architecture)
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PerTimestepMLP(nn.Module):
    """Shared-weight MLP applied independently at each timestep."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        activation: str,
        final_activation: bool = False,
    ):
        super().__init__()
        modules: List[nn.Module] = []
        act = get_activation(activation)
        if layers <= 0:
            modules.append(nn.Linear(in_dim, out_dim))
            if final_activation:
                modules.append(act)
        else:
            modules.append(nn.Linear(in_dim, hidden_dim))
            modules.append(act)
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            for _ in range(layers - 1):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(act)
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
            if hidden_dim != out_dim:
                modules.append(nn.Linear(hidden_dim, out_dim))
                if final_activation:
                    modules.append(act)
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, f = x.shape
        y = self.net(x.reshape(b * s, f))
        return y.reshape(b, s, -1)


class TransformerForecastModel(nn.Module):
    """Transformer-based forecasting model operating on (W, T, F) windows."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.aggregation_mode not in ("sum", "conv"):
            raise ValueError("aggregation_mode must be 'sum' or 'conv'")
        if cfg.conv_padding < 0:
            raise ValueError("conv_padding must be >= 0")
        if cfg.conv_padding > cfg.pad_hours:
            cfg.conv_padding = int(cfg.pad_hours)

        self.input_mlp = PerTimestepMLP(
            cfg.in_features,
            cfg.d_model,
            cfg.input_mlp_hidden,
            cfg.input_mlp_layers,
            cfg.input_dropout,
            cfg.activation,
            final_activation=True,
        )
        self.positional = SinusoidalPositionalEncoding(
            cfg.d_model, max_len=cfg.core_len + 2 * cfg.pad_hours
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=max(cfg.attn_dropout, cfg.ffn_dropout),
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_transformer_layers
        )

        if cfg.aggregation_mode == "conv" and (cfg.agg_hours > 1 or cfg.conv_padding > 0):
            kernel = cfg.agg_hours + 2 * cfg.conv_padding
            self.conv_agg = nn.Conv1d(
                in_channels=cfg.d_model,
                out_channels=cfg.d_model,
                kernel_size=kernel,
                stride=cfg.agg_hours,
                groups=cfg.d_model,
                bias=True,
                padding=0,
            )
        else:
            self.conv_agg = None

        self.output_mlp = PerTimestepMLP(
            cfg.d_model,
            cfg.out_features,
            cfg.output_mlp_hidden,
            cfg.output_mlp_layers,
            cfg.output_dropout,
            cfg.activation,
            final_activation=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_mlp(x)
        h = self.positional(h)
        h = self.transformer(h)
        if self.cfg.aggregation_mode == "conv":
            pad = int(self.cfg.pad_hours)
            conv_pad = int(min(self.cfg.conv_padding, self.cfg.pad_hours))
            agg = int(self.cfg.agg_hours)
            start = pad - conv_pad if conv_pad > 0 else pad
            end = pad + self.cfg.core_len + conv_pad if conv_pad > 0 else pad + self.cfg.core_len
            h_slice = h[:, start:end]
            if (agg > 1) or (conv_pad > 0):
                h = self.conv_agg(h_slice.transpose(1, 2)).transpose(1, 2)
            else:
                h = h_slice
        return self.output_mlp(h)


# ============================================================
# Checkpoint Loading & Inference Execution
# ============================================================
def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip `_orig_mod.` prefix added when a checkpoint was saved from a torch.compile model."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def load_surrogate_model(ckpt_path: str, device: str = "cpu") -> Tuple[TransformerForecastModel, TransformerConfig]:
    """Load trained `.pt` checkpoint into eval-mode TransformerForecastModel with strict key check."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" not in payload or "state_dict" not in payload:
        raise KeyError(f"Checkpoint '{ckpt_path}' must contain 'config' and 'state_dict' keys.")

    cfg = TransformerConfig.from_dict(payload["config"])
    model = TransformerForecastModel(cfg)
    clean_state = _strip_compile_prefix(payload["state_dict"])
    # Strict=True guarantees every trained tensor loads with zero silent mismatches
    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def load_scalers(scaler_path: str) -> Dict:
    """Load frozen training scaler parameters from JSON."""
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler JSON file not found at '{path}'.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_forward_pass(
    model: TransformerForecastModel,
    cfg: TransformerConfig,
    df_scaled: pd.DataFrame,
    scaler_data: Dict,
    device: str = "cpu",
    window_batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """Run circularly-padded windowing, forward pass, and output inverse-scaling.

    Args:
        model: Loaded `TransformerForecastModel` in eval mode.
        cfg: Corresponding `TransformerConfig`.
        df_scaled: Scaled input DataFrame of shape (8760, 49) from `/surrogate_in/features_scaled`.
        scaler_data: Loaded JSON dict containing `feature_names`, `scaler_y_mean`, `scaler_y_scale`.
        device: 'cpu' or 'cuda'.
        window_batch_size: Optional mini-batch size across windows (defaults to all 365 windows).

    Returns:
        DataFrame indexed by `hour` (0..8759) with columns `['p_mw', 'q_mvar', 's_mva']`.
    """
    # 1. Verify feature columns and exact order match the trained checkpoint
    expected_features = scaler_data["feature_names"]
    if list(df_scaled.columns) != expected_features:
        missing = [c for c in expected_features if c not in df_scaled.columns]
        if missing:
            raise ValueError(f"Missing scaled feature(s) in /surrogate_in/features_scaled: {missing}")
        df_scaled = df_scaled.reindex(columns=expected_features)

    X_arr = df_scaled.to_numpy(dtype=np.float32, copy=False)
    n_hours, n_feats = X_arr.shape
    if n_feats != cfg.in_features:
        raise ValueError(
            f"Feature count mismatch: input has {n_feats} features, checkpoint expects {cfg.in_features}."
        )

    core_len = int(cfg.core_len)
    pad = int(cfg.pad_hours)
    agg = int(cfg.agg_hours)
    if n_hours % agg != 0:
        raise ValueError(f"Series length ({n_hours}) must be divisible by agg_hours ({agg}).")

    # 2. Assemble circularly-padded sliding windows (identical to TransformerTrainer.predict)
    starts_hours = np.arange(0, max(n_hours - core_len + 1, 0), core_len, dtype=int)
    if len(starts_hours) == 0:
        starts_hours = np.array([0], dtype=int)
    base_hours = np.arange(-pad, core_len + pad, dtype=int)
    idx_X = (starts_hours[:, None] + base_hours[None, :]) % n_hours  # (W, core_len + 2*pad)

    X_windows = torch.from_numpy(X_arr[idx_X])
    n_windows = X_windows.shape[0]
    bs = n_windows if window_batch_size is None else max(1, int(window_batch_size))

    # 3. Forward pass
    pred_parts = []
    with torch.no_grad():
        for start in range(0, n_windows, bs):
            xb = X_windows[start : start + bs].to(device)
            pred = model(xb)
            if cfg.aggregation_mode == "sum":
                p = pred[:, pad:-pad] if pad > 0 else pred
                if agg > 1:
                    Bm, Tm, Fm = p.shape
                    p = p.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
            else:
                p = pred
            pred_parts.append(p.detach().cpu().numpy())

    preds_scaled = np.concatenate(pred_parts, axis=0).reshape(-1, cfg.out_features)
    n_out = n_hours // agg if agg > 1 else n_hours
    preds_scaled = preds_scaled[:n_out]

    # 4. Inverse-transform scaled (P, Q) using frozen training S_MVA mean & scale
    # Exact match with Preprocessor.inverse_transform_y (dual-target case)
    mean_S = float(scaler_data["scaler_y_mean"])
    scale_S = float(scaler_data["scaler_y_scale"])
    preds_orig = (preds_scaled.astype(np.float64) * scale_S + mean_S).astype(np.float32)

    p_mw = preds_orig[:, 0]
    q_mvar = preds_orig[:, 1] if cfg.out_features >= 2 else np.zeros_like(p_mw)
    s_mva = np.sqrt(p_mw ** 2 + q_mvar ** 2).astype(np.float32)

    df_pred = pd.DataFrame(
        {
            "p_mw": p_mw,
            "q_mvar": q_mvar,
            "s_mva": s_mva,
        },
        index=pd.Index(np.arange(n_out, dtype=int), name="hour"),
    )

    if df_pred.isna().any().any():
        raise ValueError("NaN values detected in surrogate predictions!")

    return df_pred
