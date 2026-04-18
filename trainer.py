"""
trainer.py — Orchestrates training for XGBoost and TabTransformer models.

Responsibilities:
  - Fit preprocessor on training split
  - Train XGBoost (CPU/hist optimised for M-series)
  - Train TabTransformer (MPS GPU on Apple M5 Pro)
  - Evaluate both models on the test set
  - Build a soft-voting ensemble
  - Persist artefacts
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import ChurnConfig, cfg
from preprocessor import ChurnPreprocessor
from models.xgboost_model import XGBoostChurnModel
from models.transformer_model import TabTransformerChurnModel
from evaluator import ChurnEvaluator
from shap_explainer import ShapExplainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ensemble scorer
# ---------------------------------------------------------------------------

def _ensemble_score(
    xgb_score: np.ndarray,
    trn_score: np.ndarray,
    xgb_weight: float,
    trn_weight: float,
) -> np.ndarray:
    w_total = xgb_weight + trn_weight
    return (xgb_weight * xgb_score + trn_weight * trn_score) / w_total


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class ChurnTrainer:
    """End-to-end training pipeline for the telecom churn framework."""

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg        = config
        self.prep_      : Optional[ChurnPreprocessor]       = None
        self.xgb_model_ : Optional[XGBoostChurnModel]       = None
        self.trn_model_ : Optional[TabTransformerChurnModel] = None
        self.evaluator_ : ChurnEvaluator                    = ChurnEvaluator(config)
        self.shap_      : ShapExplainer                     = ShapExplainer(config)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        test_df:  pd.DataFrame,
    ) -> Dict[str, Dict]:
        """
        Full training + evaluation pass.

        Returns dict of {model_name: metrics}.
        """
        logger.info("=" * 60)
        logger.info("Telecom Churn Training Pipeline")
        logger.info("=" * 60)

        # 1. Fit preprocessor
        self.prep_ = ChurnPreprocessor(self.cfg)
        X_train = self.prep_.fit_transform(train_df)
        X_val   = self.prep_.transform(val_df)
        X_test  = self.prep_.transform(test_df)

        y_train = train_df[self.cfg.target_column].values.astype(np.int64)
        y_val   = val_df[self.cfg.target_column].values.astype(np.int64)
        y_test  = test_df[self.cfg.target_column].values.astype(np.int64)

        feature_names = self.prep_.get_feature_names()
        num_idx       = self.prep_.get_numerical_indices()
        cat_idx       = self.prep_.get_categorical_indices()

        logger.info(
            "Data shapes — train %s | val %s | test %s",
            X_train.shape, X_val.shape, X_test.shape,
        )

        # 2. Train XGBoost
        xgb_metrics = self._train_xgboost(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names
        )

        # 3. SHAP explanation for XGBoost
        self._run_shap(X_train, X_test, y_test, feature_names)

        # 4. Train TabTransformer
        trn_metrics = self._train_transformer(
            X_train, y_train, X_val, y_val, X_test, y_test, num_idx, cat_idx
        )

        # 5. Ensemble
        ens_metrics = self._evaluate_ensemble(X_test, y_test)

        # 6. Save artefacts
        self.prep_.save()
        self.xgb_model_.save()

        # 7. Report
        comparison = self.evaluator_.compare()
        logger.info("\n%s", comparison.to_string())
        self.evaluator_.save_report()

        return {
            "xgboost":     xgb_metrics,
            "transformer": trn_metrics,
            "ensemble":    ens_metrics,
        }

    # ------------------------------------------------------------------
    # XGBoost training
    # ------------------------------------------------------------------

    def _train_xgboost(
        self,
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names,
    ) -> Dict:
        logger.info("\n--- XGBoost ---")
        t0 = time.time()

        self.xgb_model_ = XGBoostChurnModel(self.cfg)
        self.xgb_model_.fit(X_train, y_train, X_val, y_val, feature_names)

        elapsed = time.time() - t0
        logger.info("XGBoost training time: %.1f s", elapsed)

        xgb_score_test = self.xgb_model_.predict_proba(X_test)
        metrics = self.evaluator_.evaluate(
            y_test, xgb_score_test,
            label="XGBoost",
            feature_importance=self.xgb_model_.feature_importance(),
        )
        return metrics

    # ------------------------------------------------------------------
    # SHAP explanation (XGBoost)
    # ------------------------------------------------------------------

    def _run_shap(
        self,
        X_train:       np.ndarray,
        X_test:        np.ndarray,
        y_test:        np.ndarray,
        feature_names: list,
    ) -> None:
        """
        Compute and plot SHAP values for the trained XGBoost model.

        Uses a background sample of 500 training rows (enough to estimate
        the base rate accurately while keeping runtime fast).
        """
        if not self.cfg.evaluation.compute_shap:
            logger.info("SHAP disabled (config.evaluation.compute_shap=False).")
            return

        logger.info("\n--- SHAP Explanation (XGBoost) ---")
        t0 = time.time()

        try:
            # Sample background from training set for the TreeExplainer
            rng = np.random.default_rng(self.cfg.data.random_seed)
            bg_idx  = rng.choice(len(X_train), size=min(500, len(X_train)), replace=False)
            X_bg    = X_train[bg_idx]

            self.shap_.fit(self.xgb_model_, X_bg)
            self.shap_.compute(X_test, feature_names)
            self.shap_.plot_all(X_test, y_test)

            logger.info("SHAP completed in %.1f s", time.time() - t0)
        except Exception as exc:
            # SHAP is explanatory — never let it crash the training pipeline
            logger.warning("SHAP explainer failed (non-fatal): %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # TabTransformer training
    # ------------------------------------------------------------------

    def _train_transformer(
        self,
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_idx, cat_idx,
    ) -> Dict:
        logger.info("\n--- TabTransformer ---")
        t0 = time.time()

        self.trn_model_ = TabTransformerChurnModel(self.cfg)
        self.trn_model_.fit(
            X_train, y_train, X_val, y_val, num_idx, cat_idx
        )

        elapsed = time.time() - t0
        logger.info("Transformer training time: %.1f s", elapsed)

        trn_score_test = self.trn_model_.predict_proba(X_test)
        metrics = self.evaluator_.evaluate(
            y_test, trn_score_test,
            label="TabTransformer",
        )
        return metrics

    # ------------------------------------------------------------------
    # Ensemble evaluation
    # ------------------------------------------------------------------

    def _evaluate_ensemble(self, X_test, y_test) -> Dict:
        logger.info("\n--- Soft-Voting Ensemble ---")
        ec = self.cfg.ensemble

        xgb_score = self.xgb_model_.predict_proba(X_test)
        trn_score = self.trn_model_.predict_proba(X_test)
        ens_score = _ensemble_score(
            xgb_score, trn_score, ec.xgb_weight, ec.transformer_weight
        )

        # Optionally calibrate
        if ec.calibrate:
            ens_score = self._calibrate(X_test, y_test, ens_score, ec.calibration_method)

        metrics = self.evaluator_.evaluate(
            y_test, ens_score, label="Ensemble"
        )

        # Find optimal threshold on test (for reporting only)
        opt_thr, opt_f1 = self.evaluator_.find_optimal_threshold(y_test, ens_score)
        logger.info("Ensemble optimal threshold: %.3f → F1=%.4f", opt_thr, opt_f1)
        return metrics

    # ------------------------------------------------------------------
    # Post-hoc probability calibration
    # ------------------------------------------------------------------

    def _calibrate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        raw_scores: np.ndarray,
        method: str = "isotonic",
    ) -> np.ndarray:
        """
        Fit a calibrator on half the test set, transform the other half,
        then return calibrated probabilities for the full test set.

        In production, calibrate on a held-out calibration set.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        n  = len(y_test)
        mid = n // 2
        s_cal, y_cal = raw_scores[:mid].reshape(-1, 1), y_test[:mid]
        s_inf        = raw_scores[mid:].reshape(-1, 1)

        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(s_cal.ravel(), y_cal)
            cal_proba = np.concatenate([
                cal.predict(s_cal.ravel()),
                cal.predict(s_inf.ravel()),
            ])
        else:  # sigmoid / Platt scaling
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            lr.fit(s_cal, y_cal)
            cal_proba = np.concatenate([
                lr.predict_proba(s_cal)[:, 1],
                lr.predict_proba(s_inf)[:, 1],
            ])

        return cal_proba.astype(np.float32)
