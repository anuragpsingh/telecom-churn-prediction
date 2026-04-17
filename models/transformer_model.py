"""
models/transformer_model.py — Tab-Transformer for tabular churn prediction.

Architecture:
  Categorical features  →  Learnable embeddings (dim = embedding_dim)
  Numerical features    →  Per-feature linear projection to embedding_dim
  All embeddings        →  Stacked → Transformer Encoder (multi-head attention)
  CLS token             →  MLP classifier head → churn probability

GPU support:
  - Apple M-series (M5 Pro): device = "mps"  (Metal Performance Shaders)
  - NVIDIA GPU:               device = "cuda"
  - CPU fallback:             device = "cpu"
"""

from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import ChurnConfig, cfg, DEVICE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def make_dataloader(
    X: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool = True,
    device: str = "cpu",
) -> DataLoader:
    """Build a DataLoader from numpy arrays. Tensors are kept on CPU; the
    training loop moves them to the target device batch-by-batch to avoid
    OOM on large datasets."""
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_t = torch.tensor(y, dtype=torch.long)
        ds  = TensorDataset(X_t, y_t)
    else:
        ds  = TensorDataset(X_t)

    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        pin_memory  = (device in ("cuda",)),   # pin_memory not supported for MPS
        num_workers = 0,                        # 0 is safest for MPS/Metal
        drop_last   = False,
    )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class TabTransformer(nn.Module):
    """
    Tab-Transformer: each feature is projected to the same embedding dim,
    then processed jointly by a Transformer encoder. A learnable CLS token
    aggregates global context for the classifier head.
    """

    def __init__(
        self,
        num_numerical:     int,
        categorical_cardinalities: Dict[str, int],
        d_model:           int   = 64,
        num_heads:         int   = 4,
        num_encoder_layers:int   = 3,
        dim_feedforward:   int   = 256,
        dropout:           float = 0.15,
        mlp_hidden_dims:   List[int] = None,
    ):
        super().__init__()
        mlp_hidden_dims = mlp_hidden_dims or [128, 64]

        self.d_model         = d_model
        self.num_numerical   = num_numerical
        self.cat_names       = list(categorical_cardinalities.keys())

        # ---- Numerical projections (one linear layer per feature) ---------
        # We project each scalar to d_model using a shared weight matrix
        # (equivalent to a per-feature linear with no bias sharing).
        self.num_proj = nn.Linear(num_numerical, num_numerical * d_model)
        # After reshape: [batch, num_numerical, d_model]

        # ---- Categorical embeddings ---------------------------------------
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 2, d_model)  # +2 for unknown & padding
            for cardinality in categorical_cardinalities.values()
        ])

        # ---- CLS token ----------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- Transformer encoder ------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = num_heads,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,   # [batch, seq, d_model]
            norm_first      = True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers      = num_encoder_layers,
            enable_nested_tensor = False,
        )

        # ---- MLP classifier head ------------------------------------------
        in_dim = d_model
        layers: List[nn.Module] = []
        for h in mlp_hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,             # [batch, num_numerical + num_categorical]
        num_idx: List[int],
        cat_idx: List[int],
    ) -> torch.Tensor:               # [batch] logits
        batch = x.shape[0]

        # -- Numerical tokens -----------------------------------------------
        x_num = x[:, num_idx]                        # [B, N_num]
        num_emb = self.num_proj(x_num)               # [B, N_num * d_model]
        num_emb = num_emb.view(batch, self.num_numerical, self.d_model)

        # -- Categorical tokens ---------------------------------------------
        cat_embs = []
        for i, idx in enumerate(cat_idx):
            codes = x[:, idx].long().clamp(min=0)    # unknown = 0
            cat_embs.append(self.cat_embeddings[i](codes))  # [B, d_model]
        cat_emb = torch.stack(cat_embs, dim=1)       # [B, N_cat, d_model]

        # -- Concatenate numerical + categorical + CLS ----------------------
        cls  = self.cls_token.expand(batch, -1, -1)  # [B, 1, d_model]
        seq  = torch.cat([cls, num_emb, cat_emb], dim=1)   # [B, 1+N, d_model]

        # -- Transformer encoder -------------------------------------------
        encoded = self.transformer(seq)              # [B, 1+N, d_model]
        cls_out = encoded[:, 0, :]                   # [B, d_model]  CLS repr

        # -- Classifier head -----------------------------------------------
        logits = self.mlp_head(cls_out).squeeze(-1)  # [B]
        return logits

    def predict_proba_tensor(
        self, x: torch.Tensor, num_idx: List[int], cat_idx: List[int]
    ) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x, num_idx, cat_idx))


# ---------------------------------------------------------------------------
# High-level model wrapper
# ---------------------------------------------------------------------------

class TabTransformerChurnModel:
    """
    Wraps TabTransformer with fit() / predict_proba() similar to sklearn API.

    Handles:
      - Device placement (MPS / CUDA / CPU)
      - Mixed-precision training (CUDA only; MPS AMP not yet stable)
      - LR scheduling, early stopping on val AUROC
      - Checkpointing
    """

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg    = config
        self.tcfg   = config.transformer
        self.device = torch.device(self.tcfg.device)
        self.net_: Optional[TabTransformer] = None
        self.num_idx_: List[int] = []
        self.cat_idx_: List[int] = []
        self.best_auroc_: float = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_net(self) -> TabTransformer:
        tc = self.tcfg
        return TabTransformer(
            num_numerical            = len(self.cfg.numerical_features),
            categorical_cardinalities= self.cfg.categorical_cardinalities,
            d_model                  = tc.d_model,
            num_heads                = tc.num_heads,
            num_encoder_layers       = tc.num_encoder_layers,
            dim_feedforward          = tc.dim_feedforward,
            dropout                  = tc.dropout,
            mlp_hidden_dims          = tc.mlp_hidden_dims,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        num_idx: List[int],
        cat_idx: List[int],
    ) -> "TabTransformerChurnModel":
        from sklearn.metrics import roc_auc_score

        tc = self.tcfg
        self.num_idx_ = num_idx
        self.cat_idx_ = cat_idx

        self.net_ = self._build_net()

        n_params = sum(p.numel() for p in self.net_.parameters() if p.requires_grad)
        logger.info(
            "TabTransformer | device=%s | params=%s | train=%d val=%d",
            self.device, f"{n_params:,}", len(y_train), len(y_val),
        )

        # Class weight for imbalance
        pos_weight = torch.tensor(
            [(1 - self.cfg.data.churn_rate) / self.cfg.data.churn_rate],
            dtype=torch.float32,
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.net_.parameters(),
            lr           = tc.learning_rate,
            weight_decay = tc.weight_decay,
        )

        # LR scheduler
        if tc.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=tc.epochs, eta_min=tc.learning_rate * 0.01
            )
        elif tc.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
        else:
            scheduler = None

        # Mixed precision (CUDA only)
        use_amp = tc.use_mixed_precision and (str(self.device) == "cuda")
        scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

        train_loader = make_dataloader(
            X_train, y_train, tc.batch_size, shuffle=True,  device=tc.device
        )
        val_loader   = make_dataloader(
            X_val,   y_val,   tc.batch_size, shuffle=False, device=tc.device
        )

        patience_counter = 0
        self.best_auroc_ = 0.0

        for epoch in range(1, tc.epochs + 1):
            # ---- Train ---------------------------------------------------
            self.net_.train()
            total_loss = 0.0
            for batch in train_loader:
                xb, yb = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type = "cuda" if str(self.device) == "cuda" else "cpu",
                    enabled     = use_amp,
                ):
                    logits = self.net_(xb, num_idx, cat_idx)
                    loss   = criterion(logits, yb.float())

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(yb)

            avg_loss = total_loss / len(y_train)

            # ---- Validate ------------------------------------------------
            self.net_.eval()
            all_proba: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    xb = batch[0].to(self.device)
                    p  = torch.sigmoid(
                        self.net_(xb, num_idx, cat_idx)
                    ).cpu().numpy()
                    all_proba.extend(p.tolist())

            val_auroc = roc_auc_score(y_val, all_proba)

            if scheduler is not None:
                scheduler.step()

            # ---- Early stopping & checkpointing --------------------------
            if val_auroc > self.best_auroc_:
                self.best_auroc_ = val_auroc
                patience_counter  = 0
                self.save(self.tcfg.best_model_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d | loss=%.4f | val_AUROC=%.4f | best=%.4f | LR=%.2e",
                    epoch, tc.epochs, avg_loss, val_auroc, self.best_auroc_,
                    optimizer.param_groups[0]["lr"],
                )

            if patience_counter >= tc.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Load best checkpoint
        self.load(self.tcfg.best_model_path)
        logger.info("Loaded best model | val_AUROC=%.4f", self.best_auroc_)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.net_ is not None, "Model not trained yet."
        loader = make_dataloader(
            X, None, self.tcfg.batch_size, shuffle=False, device=self.tcfg.device
        )
        self.net_.eval()
        all_proba = []
        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(self.device)
                p  = torch.sigmoid(
                    self.net_(xb, self.num_idx_, self.cat_idx_)
                ).cpu().numpy()
                all_proba.extend(p.tolist())
        return np.array(all_proba, dtype=np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.tcfg.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":  self.net_.state_dict(),
            "num_idx":     self.num_idx_,
            "cat_idx":     self.cat_idx_,
            "best_auroc":  self.best_auroc_,
        }, path)

    def load(self, path: Optional[str] = None) -> "TabTransformerChurnModel":
        path = path or self.tcfg.best_model_path
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if self.net_ is None:
            self.net_ = self._build_net()
        self.net_.load_state_dict(ckpt["state_dict"])
        self.num_idx_    = ckpt["num_idx"]
        self.cat_idx_    = ckpt["cat_idx"]
        self.best_auroc_ = ckpt.get("best_auroc", 0.0)
        self.net_.eval()
        return self
