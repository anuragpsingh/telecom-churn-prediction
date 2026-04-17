"""
models/xgboost_model.py — XGBoost churn classifier.

NOTE on Apple M-series GPU:
  XGBoost does not yet support Apple Metal (MPS).  The `hist` tree method
  with `nthread=-1` (all logical cores) gives the best throughput on M-series
  chips and is close in speed to CUDA for tabular workloads of this scale.
  When running on a machine with an NVIDIA GPU, set device="cuda" in
  XGBoostConfig and xgboost will use the GPU automatically.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from config import ChurnConfig, cfg

logger = logging.getLogger(__name__)


class XGBoostChurnModel:
    """Production-grade XGBoost wrapper for telecom churn prediction."""

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg    = config
        self.xcfg   = config.xgboost
        self.model_: Optional[xgb.XGBClassifier] = None
        self.feature_names_: Optional[List[str]] = None
        self.best_iteration_: int = 0
        self.evals_result_: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Build estimator
    # ------------------------------------------------------------------

    def _build_estimator(self) -> xgb.XGBClassifier:
        xc = self.xcfg
        import os
        nthread = int(os.environ.get("OMP_NUM_THREADS", str(xc.n_jobs)))
        return xgb.XGBClassifier(
            n_estimators          = xc.n_estimators,
            learning_rate         = xc.learning_rate,
            max_depth             = xc.max_depth,
            min_child_weight      = xc.min_child_weight,
            subsample             = xc.subsample,
            colsample_bytree      = xc.colsample_bytree,
            gamma                 = xc.gamma,
            reg_alpha             = xc.reg_alpha,
            reg_lambda            = xc.reg_lambda,
            scale_pos_weight      = xc.scale_pos_weight,
            tree_method           = xc.tree_method,
            device                = xc.device,
            nthread               = nthread,
            eval_metric           = xc.eval_metric,
            early_stopping_rounds = xc.early_stopping_rounds,
            random_state          = self.cfg.data.random_seed,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "XGBoostChurnModel":
        self.feature_names_ = feature_names
        self.model_         = self._build_estimator()

        logger.info(
            "Training XGBoost | train=%d val=%d | device=%s",
            len(y_train), len(y_val), self.xcfg.device,
        )

        self.model_.fit(
            X_train, y_train,
            eval_set         = [(X_train, y_train), (X_val, y_val)],
            verbose          = self.xcfg.verbose_eval,
        )

        self.best_iteration_ = self.model_.best_iteration
        self.evals_result_   = self.model_.evals_result()
        logger.info("Best iteration: %d", self.best_iteration_)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-1 probability for each sample."""
        assert self.model_ is not None, "Model not trained yet."
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_importance(
        self, importance_type: str = "gain"
    ) -> Dict[str, float]:
        assert self.model_ is not None
        scores = self.model_.get_booster().get_score(
            importance_type=importance_type
        )
        names  = self.feature_names_ or list(scores.keys())
        # Re-index by position if feature names are available
        fi = {}
        for k, v in scores.items():
            # XGBoost keys are "f0", "f1" … when no feature names set
            try:
                idx = int(k[1:])
                fi[names[idx]] = v
            except (ValueError, IndexError):
                fi[k] = v
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    def top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        fi = self.feature_importance()
        return list(fi.items())[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.xcfg.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(path)
        logger.info("XGBoost model saved → %s", path)

    def load(self, path: Optional[str] = None) -> "XGBoostChurnModel":
        path = path or self.xcfg.model_path
        self.model_ = self._build_estimator()
        self.model_.load_model(path)
        logger.info("XGBoost model loaded from %s", path)
        return self

    # ------------------------------------------------------------------
    # Cross-validation helper
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "roc_auc",
    ) -> np.ndarray:
        """Runs stratified k-fold CV on a no-early-stopping estimator."""
        est = xgb.XGBClassifier(
            n_estimators     = max(1, self.best_iteration_ or 300),
            learning_rate    = self.xcfg.learning_rate,
            max_depth        = self.xcfg.max_depth,
            subsample        = self.xcfg.subsample,
            colsample_bytree = self.xcfg.colsample_bytree,
            scale_pos_weight = self.xcfg.scale_pos_weight,
            tree_method      = self.xcfg.tree_method,
            device           = self.xcfg.device,
            n_jobs           = self.xcfg.n_jobs,
            random_state     = self.cfg.data.random_seed,
        )
        scores = cross_val_score(est, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        logger.info("CV %s (k=%d): %.4f ± %.4f", scoring, cv, scores.mean(), scores.std())
        return scores
