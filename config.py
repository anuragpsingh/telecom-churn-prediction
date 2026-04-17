"""
config.py — Central configuration for the Telecom Churn Prediction Framework.

Covers data schema, preprocessing, model hyperparameters, training settings,
and device selection (Apple M-series MPS / CUDA / CPU).
"""

from __future__ import annotations
import os
import platform
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# OpenMP thread guard — MUST run before any xgboost import.
#
# On macOS 26 beta (arm64), libxgboost.dylib segfaults when OpenMP
# auto-detects thread count at startup.  Setting OMP_NUM_THREADS=1
# via the shell before launching Python is the most reliable fix:
#
#   OMP_NUM_THREADS=1 python pipeline.py
#
# The fallback below covers interactive / notebook usage where the env
# var is not set at the shell level.
# ---------------------------------------------------------------------------
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Device detection — auto-selects MPS (Apple M-series), CUDA, or CPU
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    """Return the best available compute device string."""
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"          # Apple M1 / M2 / M3 / M4 / M5 Pro
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


DEVICE = _detect_device()


# ---------------------------------------------------------------------------
# Feature schema — single source of truth for all downstream code
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES: List[str] = [
    # Demographics
    "age",
    # Account
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_monthly_charges",
    # Usage
    "avg_daily_call_minutes",
    "avg_monthly_data_gb",
    "avg_monthly_sms",
    "roaming_usage_min",
    # Network quality
    "avg_call_drop_rate",
    "avg_data_speed_mbps",
    "network_outage_hours_6mo",
    # Customer service
    "num_complaints_6mo",
    "num_support_calls_6mo",
    "days_since_last_complaint",
    # Engagement
    "app_logins_monthly",
    "feature_adoption_score",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_type",
    "payment_method",
    "paperless_billing",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

TARGET_COLUMN = "churned"

# Cardinality of each categorical feature (used by transformer embeddings)
CATEGORICAL_CARDINALITIES: Dict[str, int] = {
    "gender":             2,
    "senior_citizen":     2,
    "has_partner":        2,
    "has_dependents":     2,
    "contract_type":      3,   # month-to-month, one-year, two-year
    "payment_method":     4,   # electronic check, mailed check, bank transfer, credit card
    "paperless_billing":  2,
    "phone_service":      2,
    "multiple_lines":     3,   # yes, no, no phone service
    "internet_service":   3,   # DSL, Fiber optic, No
    "online_security":    3,
    "online_backup":      3,
    "device_protection":  3,
    "tech_support":       3,
    "streaming_tv":       3,
    "streaming_movies":   3,
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    n_samples: int          = 60_000
    churn_rate: float       = 0.265   # realistic telecom churn ~26%
    random_seed: int        = 42
    test_size: float        = 0.15
    val_size: float         = 0.10    # fraction of *remaining* after test split


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingConfig:
    numerical_scaler: str   = "standard"   # "standard" | "robust" | "minmax"
    handle_outliers: bool   = True
    outlier_method: str     = "iqr"        # "iqr" | "zscore"
    outlier_cap_factor: float = 3.0


# ---------------------------------------------------------------------------
# XGBoost model
# ---------------------------------------------------------------------------
@dataclass
class XGBoostConfig:
    # Core
    n_estimators: int       = 1000
    learning_rate: float    = 0.05
    max_depth: int          = 6
    min_child_weight: int   = 3
    subsample: float        = 0.8
    colsample_bytree: float = 0.8
    gamma: float            = 0.1
    reg_alpha: float        = 0.1
    reg_lambda: float       = 1.0
    scale_pos_weight: float = 2.77  # ≈ (1 - churn_rate) / churn_rate
    # Device: XGBoost does not yet support Apple Metal (MPS).
    # Use hist + all CPU threads for M-series; switch to "cuda" for NVIDIA.
    tree_method: str        = "hist"
    device: str             = "cpu"
    n_jobs: int             = int(os.environ.get("OMP_NUM_THREADS", "1"))
    # Training
    eval_metric: str        = "aucpr"
    early_stopping_rounds: int = 50
    verbose_eval: int       = 100
    # Paths
    model_path: str         = "checkpoints/xgboost_churn.json"


# ---------------------------------------------------------------------------
# Transformer model (Tab-Transformer, PyTorch)
# ---------------------------------------------------------------------------
@dataclass
class TransformerConfig:
    # Embedding
    embedding_dim: int      = 32    # dim per categorical embedding
    numerical_proj_dim: int = 32    # project each numerical scalar to this dim
    # Transformer encoder
    d_model: int            = 64    # must equal embedding_dim if sharing projection
    num_heads: int          = 4
    num_encoder_layers: int = 3
    dim_feedforward: int    = 256
    dropout: float          = 0.15
    # MLP classifier head
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    # Training
    epochs: int             = 60
    batch_size: int         = 512   # larger batch for GPU throughput
    learning_rate: float    = 3e-4
    weight_decay: float     = 1e-4
    patience: int           = 10    # early stopping patience (val AUROC)
    lr_scheduler: str       = "cosine"   # "cosine" | "step" | "none"
    # Device (MPS / CUDA / CPU)
    device: str             = DEVICE
    use_mixed_precision: bool = (DEVICE == "cuda")  # MPS AMP not yet stable
    # Paths
    model_path: str         = "checkpoints/transformer_churn.pt"
    best_model_path: str    = "checkpoints/transformer_churn_best.pt"


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
@dataclass
class EnsembleConfig:
    xgb_weight: float       = 0.50
    transformer_weight: float = 0.50
    # Calibration
    calibrate: bool         = True
    calibration_method: str = "isotonic"  # "sigmoid" | "isotonic"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    primary_metric: str     = "roc_auc"
    threshold: float        = 0.5       # decision boundary (tunable)
    bootstrap_rounds: int   = 1000
    results_dir: str        = "results/"
    save_plots: bool        = True
    compute_shap: bool      = True


# ---------------------------------------------------------------------------
# Master config — compose all sub-configs into one object
# ---------------------------------------------------------------------------
@dataclass
class ChurnConfig:
    data:           DataConfig          = field(default_factory=DataConfig)
    preprocessing:  PreprocessingConfig = field(default_factory=PreprocessingConfig)
    xgboost:        XGBoostConfig       = field(default_factory=XGBoostConfig)
    transformer:    TransformerConfig   = field(default_factory=TransformerConfig)
    ensemble:       EnsembleConfig      = field(default_factory=EnsembleConfig)
    evaluation:     EvaluationConfig    = field(default_factory=EvaluationConfig)

    # Feature schema shortcuts
    numerical_features:     List[str]       = field(default_factory=lambda: NUMERICAL_FEATURES)
    categorical_features:   List[str]       = field(default_factory=lambda: CATEGORICAL_FEATURES)
    categorical_cardinalities: Dict[str, int] = field(
        default_factory=lambda: CATEGORICAL_CARDINALITIES
    )
    target_column:          str             = TARGET_COLUMN

    def __post_init__(self):
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def all_features(self) -> List[str]:
        return self.numerical_features + self.categorical_features

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Telecom Churn Prediction — Configuration",
            "=" * 60,
            f"  Platform   : {platform.platform()}",
            f"  Device     : {DEVICE.upper()}",
            f"  Samples    : {self.data.n_samples:,}",
            f"  Churn rate : {self.data.churn_rate:.1%}",
            f"  Num features    : {len(self.numerical_features)}",
            f"  Cat features    : {len(self.categorical_features)}",
            f"  XGB device      : {self.xgboost.device}",
            f"  Transformer dev : {self.transformer.device}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Convenience singleton
cfg = ChurnConfig()
