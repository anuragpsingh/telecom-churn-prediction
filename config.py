"""
config.py — Central configuration for the Mobile Churn Prediction Framework.

Domain: European mobile subscribers (prepaid & postpaid).
Network KPIs: SINR, RSRP, Call Success Rate, Bearer Establishment Rate.
Billing: top-ups, billing rate, overage charges, late payments.
"""

from __future__ import annotations
import os
import platform
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# OpenMP thread guard — MUST run before any xgboost import.
# On macOS 26 beta (arm64), libxgboost.dylib segfaults when OpenMP
# auto-detects thread count.  OMP_NUM_THREADS=1 at the shell level is
# the most reliable fix:  OMP_NUM_THREADS=1 python pipeline.py
# ---------------------------------------------------------------------------
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Device detection — MPS (Apple M-series) → CUDA → CPU
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


DEVICE = _detect_device()


# ---------------------------------------------------------------------------
# Feature schema — single source of truth
# ---------------------------------------------------------------------------

NUMERICAL_FEATURES: List[str] = [
    # ---- Demographics -------------------------------------------------------
    "age",

    # ---- Account / Tenure ---------------------------------------------------
    "tenure_months",
    "monthly_plan_cost_eur",        # plan base price  (5 – 20 EUR)
    "total_spend_6mo_eur",          # actual spend including extras

    # ---- Network Quality (LTE / 5G KPIs) ------------------------------------
    # SINR: Signal-to-Interference-plus-Noise Ratio (dB)
    #   > 20 dB  = excellent  |  10–20 = good  |  0–10 = fair  |  < 0 = poor
    "avg_sinr_db",
    # RSRP: Reference Signal Received Power (dBm)
    #   > -80 dBm = excellent  |  -80 to -90 = good  |  -90 to -100 = fair  |  < -100 = poor
    "avg_rsrp_dbm",
    # Percentage of voice calls successfully connected
    "call_success_rate_pct",
    # Percentage of data bearer sessions successfully established
    "bearer_establishment_rate_pct",
    # Percentage of active calls that drop unexpectedly
    "avg_call_drop_rate_pct",
    # Average download speed in Mbps
    "avg_data_speed_mbps",

    # ---- Usage --------------------------------------------------------------
    "avg_monthly_data_gb",
    "avg_daily_data_usage_mb",      # daily data consumption in MB
    "avg_daily_call_minutes",
    "avg_monthly_sms",
    "roaming_sessions_6mo",
    "intl_call_minutes_monthly",

    # ---- Billing ------------------------------------------------------------
    "num_topups_6mo",               # prepaid recharges in last 6 months
    "avg_topup_amount_eur",         # average top-up value
    "num_late_payments_6mo",        # postpaid late payment events
    "data_overage_charges_eur",     # extra charges for exceeding data bundle

    # ---- Customer Service ---------------------------------------------------
    "num_complaints_6mo",
    "num_support_calls_6mo",
    "days_since_last_complaint",
]

CATEGORICAL_FEATURES: List[str] = [
    "plan_type",            # prepaid | postpaid
    "plan_tier",            # basic | standard | premium | unlimited
    "contract_duration",    # monthly | 12_month | 24_month | prepaid
    "payment_method",       # direct_debit | credit_card | cash | mobile_wallet
    "network_generation",   # 3G | 4G | 5G
    "device_type",          # smartphone | feature_phone | tablet | mifi
    "gender",               # Male | Female
    "is_roaming_enabled",   # yes | no
    "has_handset_subsidy",  # yes | no  (phone subsidised by operator)
]

TARGET_COLUMN = "churned"

# Cardinalities used by the Tab-Transformer embedding layers
CATEGORICAL_CARDINALITIES: Dict[str, int] = {
    "plan_type":           2,
    "plan_tier":           4,
    "contract_duration":   4,
    "payment_method":      4,
    "network_generation":  3,
    "device_type":         4,
    "gender":              2,
    "is_roaming_enabled":  2,
    "has_handset_subsidy": 2,
}

# Plan tier → monthly base price range (EUR)
PLAN_PRICE_RANGES: Dict[str, tuple] = {
    "basic":     (5.0,  9.0),
    "standard":  (9.0,  14.0),
    "premium":   (14.0, 18.0),
    "unlimited": (17.0, 20.0),
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    n_samples:    int   = 60_000
    churn_rate:   float = 0.28    # ~28% mobile churn (higher than fixed-line)
    random_seed:  int   = 42
    test_size:    float = 0.15
    val_size:     float = 0.10    # fraction of remaining after test split


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingConfig:
    numerical_scaler:    str   = "standard"   # "standard" | "robust" | "minmax"
    handle_outliers:     bool  = True
    outlier_method:      str   = "iqr"
    outlier_cap_factor:  float = 3.0


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
@dataclass
class XGBoostConfig:
    n_estimators:          int   = 1000
    learning_rate:         float = 0.05
    max_depth:             int   = 6
    min_child_weight:      int   = 3
    subsample:             float = 0.8
    colsample_bytree:      float = 0.8
    gamma:                 float = 0.1
    reg_alpha:             float = 0.1
    reg_lambda:            float = 1.0
    scale_pos_weight:      float = 2.57   # ≈ (1 - 0.28) / 0.28
    tree_method:           str   = "hist"
    device:                str   = "cpu"  # XGBoost has no Metal/MPS backend
    n_jobs:                int   = int(os.environ.get("OMP_NUM_THREADS", "1"))
    eval_metric:           str   = "aucpr"
    early_stopping_rounds: int   = 50
    verbose_eval:          int   = 100
    model_path:            str   = "checkpoints/xgboost_mobile_churn.json"


# ---------------------------------------------------------------------------
# Tab-Transformer (PyTorch, MPS GPU)
# ---------------------------------------------------------------------------
@dataclass
class TransformerConfig:
    embedding_dim:       int        = 32
    numerical_proj_dim:  int        = 32
    d_model:             int        = 64
    num_heads:           int        = 4
    num_encoder_layers:  int        = 3
    dim_feedforward:     int        = 256
    dropout:             float      = 0.15
    mlp_hidden_dims:     List[int]  = field(default_factory=lambda: [128, 64])
    epochs:              int        = 60
    batch_size:          int        = 512
    learning_rate:       float      = 3e-4
    weight_decay:        float      = 1e-4
    patience:            int        = 10
    lr_scheduler:        str        = "cosine"
    device:              str        = DEVICE
    use_mixed_precision: bool       = (DEVICE == "cuda")
    model_path:          str        = "checkpoints/transformer_mobile_churn.pt"
    best_model_path:     str        = "checkpoints/transformer_mobile_churn_best.pt"


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
@dataclass
class EnsembleConfig:
    xgb_weight:           float = 0.50
    transformer_weight:   float = 0.50
    calibrate:            bool  = True
    calibration_method:   str   = "isotonic"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    primary_metric:   str   = "roc_auc"
    threshold:        float = 0.5
    bootstrap_rounds: int   = 1000
    results_dir:      str   = "results/"
    save_plots:       bool  = True
    compute_shap:     bool  = True


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class ChurnConfig:
    data:           DataConfig          = field(default_factory=DataConfig)
    preprocessing:  PreprocessingConfig = field(default_factory=PreprocessingConfig)
    xgboost:        XGBoostConfig       = field(default_factory=XGBoostConfig)
    transformer:    TransformerConfig   = field(default_factory=TransformerConfig)
    ensemble:       EnsembleConfig      = field(default_factory=EnsembleConfig)
    evaluation:     EvaluationConfig    = field(default_factory=EvaluationConfig)

    numerical_features:        List[str]       = field(default_factory=lambda: NUMERICAL_FEATURES)
    categorical_features:      List[str]       = field(default_factory=lambda: CATEGORICAL_FEATURES)
    categorical_cardinalities: Dict[str, int]  = field(default_factory=lambda: CATEGORICAL_CARDINALITIES)
    target_column:             str             = TARGET_COLUMN

    def __post_init__(self):
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results",     exist_ok=True)

    def all_features(self) -> List[str]:
        return self.numerical_features + self.categorical_features

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "Mobile Churn Prediction — Configuration",
            "=" * 62,
            f"  Platform        : {platform.platform()}",
            f"  Device          : {DEVICE.upper()}",
            f"  Samples         : {self.data.n_samples:,}",
            f"  Churn rate      : {self.data.churn_rate:.1%}",
            f"  Numerical feats : {len(self.numerical_features)}",
            f"  Categorical feats: {len(self.categorical_features)}",
            f"  XGB device      : {self.xgboost.device}",
            f"  Transformer dev : {self.transformer.device}",
            "=" * 62,
        ]
        return "\n".join(lines)


cfg = ChurnConfig()
