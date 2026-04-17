"""
inference.py — Production-grade inference engine for the churn framework.

Supports:
  - Single customer prediction
  - Batch prediction (pandas DataFrame or numpy array)
  - Ensemble scoring (XGBoost + TabTransformer)
  - Confidence labels ("high_risk" / "medium_risk" / "low_risk")
  - Lazy model loading (models initialised once on first call)

Usage example:
    predictor = ChurnPredictor.from_checkpoints()
    result = predictor.predict_single(customer_dict)
    batch_df = predictor.predict_batch(df)
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import ChurnConfig, cfg
from preprocessor import ChurnPreprocessor
from models.xgboost_model import XGBoostChurnModel
from models.transformer_model import TabTransformerChurnModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class ChurnPrediction:
    customer_id:       Optional[str]
    churn_probability: float
    churn_label:       int               # 0 or 1
    risk_segment:      str               # "high_risk" | "medium_risk" | "low_risk"
    xgb_score:         float
    transformer_score: float
    latency_ms:        float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id":       self.customer_id,
            "churn_probability": round(self.churn_probability, 4),
            "churn_label":       self.churn_label,
            "risk_segment":      self.risk_segment,
            "xgb_score":         round(self.xgb_score, 4),
            "transformer_score": round(self.transformer_score, 4),
            "latency_ms":        round(self.latency_ms, 2),
        }


def _risk_segment(prob: float) -> str:
    if prob >= 0.65:
        return "high_risk"
    if prob >= 0.35:
        return "medium_risk"
    return "low_risk"


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class ChurnPredictor:
    """Lazy-loaded inference engine that wraps preprocessor + ensemble."""

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg           = config
        self._prep:  Optional[ChurnPreprocessor]       = None
        self._xgb:   Optional[XGBoostChurnModel]       = None
        self._trn:   Optional[TabTransformerChurnModel] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        config:           ChurnConfig = cfg,
        preprocessor_path: str        = "checkpoints/preprocessor.pkl",
        xgb_path:          str        = "checkpoints/xgboost_churn.json",
        transformer_path:  str        = "checkpoints/transformer_churn_best.pt",
    ) -> "ChurnPredictor":
        predictor = cls(config)
        predictor._load_models(preprocessor_path, xgb_path, transformer_path)
        return predictor

    def _load_models(
        self,
        preprocessor_path: str,
        xgb_path: str,
        transformer_path: str,
    ) -> None:
        if self._loaded:
            return
        logger.info("Loading churn models …")

        # Preprocessor
        self._prep = ChurnPreprocessor.load(preprocessor_path)

        # XGBoost
        self._xgb = XGBoostChurnModel(self.cfg)
        self._xgb.load(xgb_path)

        # Transformer — build net first so we can set indices
        self._trn = TabTransformerChurnModel(self.cfg)
        self._trn.num_idx_ = self._prep.get_numerical_indices()
        self._trn.cat_idx_ = self._prep.get_categorical_indices()
        self._trn.load(transformer_path)

        self._loaded = True
        logger.info("All models loaded and ready.")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (xgb_scores, transformer_scores) for a preprocessed array."""
        xgb_s = self._xgb.predict_proba(X)
        trn_s = self._trn.predict_proba(X)
        return xgb_s, trn_s

    def _ensemble_score(
        self, xgb_s: np.ndarray, trn_s: np.ndarray
    ) -> np.ndarray:
        ec  = self.cfg.ensemble
        wt  = ec.xgb_weight + ec.transformer_weight
        return (ec.xgb_weight * xgb_s + ec.transformer_weight * trn_s) / wt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(
        self,
        customer: Dict[str, Any],
        customer_id: Optional[str] = None,
        threshold: float = 0.5,
    ) -> ChurnPrediction:
        """
        Predict churn for a single customer dictionary.

        Parameters
        ----------
        customer    : dict with all feature columns (may include extra keys)
        customer_id : optional identifier for tracking
        threshold   : decision boundary (default 0.5)

        Returns
        -------
        ChurnPrediction dataclass
        """
        t0 = time.perf_counter()

        df  = pd.DataFrame([customer])
        X   = self._prep.transform(df)

        xgb_s, trn_s = self._score(X)
        ens_s  = self._ensemble_score(xgb_s, trn_s)

        prob   = float(ens_s[0])
        latency= (time.perf_counter() - t0) * 1000

        return ChurnPrediction(
            customer_id       = customer_id,
            churn_probability = prob,
            churn_label       = int(prob >= threshold),
            risk_segment      = _risk_segment(prob),
            xgb_score         = float(xgb_s[0]),
            transformer_score = float(trn_s[0]),
            latency_ms        = latency,
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Predict churn for a DataFrame of customers.

        Returns a DataFrame with prediction columns appended.
        """
        t0 = time.perf_counter()

        X = self._prep.transform(df)
        xgb_s, trn_s = self._score(X)
        ens_s = self._ensemble_score(xgb_s, trn_s)

        results = df.copy()
        if id_column and id_column in df.columns:
            results = results[[id_column]]
        else:
            results = pd.DataFrame(index=df.index)

        results["churn_probability"]  = ens_s.round(4)
        results["churn_label"]        = (ens_s >= threshold).astype(int)
        results["risk_segment"]       = [_risk_segment(p) for p in ens_s]
        results["xgb_score"]          = xgb_s.round(4)
        results["transformer_score"]  = trn_s.round(4)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Batch inference: %d records | %.1f ms total | %.3f ms/record",
            len(df), elapsed, elapsed / max(len(df), 1),
        )
        return results

    def predict_batch_from_file(
        self,
        path: str,
        output_path: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a CSV, run batch inference, optionally save results."""
        df  = pd.read_csv(path)
        out = self.predict_batch(df, id_column=id_column)
        if output_path:
            out.to_csv(output_path, index=False)
            logger.info("Results saved → %s", output_path)
        return out

    # ------------------------------------------------------------------
    # Warm-up (pre-compile MPS/CUDA kernels)
    # ------------------------------------------------------------------

    def warmup(self, n: int = 4) -> None:
        """Run a dummy batch to trigger GPU kernel compilation."""
        logger.info("Warming up inference engine (n=%d) …", n)
        dummy = {f: 0 for f in self.cfg.numerical_features}
        dummy.update({f: "No" for f in self.cfg.categorical_features})
        for _ in range(n):
            self.predict_single(dummy)
        logger.info("Warmup complete.")


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    predictor = ChurnPredictor.from_checkpoints()
    predictor.warmup()

    # Test single prediction
    sample_customer = {
        "age": 35,
        "tenure_months": 3,
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "avg_monthly_charges": 95.0,
        "avg_daily_call_minutes": 45.0,
        "avg_monthly_data_gb": 20.0,
        "avg_monthly_sms": 100,
        "roaming_usage_min": 5.0,
        "avg_call_drop_rate": 0.08,
        "avg_data_speed_mbps": 200.0,
        "network_outage_hours_6mo": 5.0,
        "num_complaints_6mo": 3,
        "num_support_calls_6mo": 4,
        "days_since_last_complaint": 15.0,
        "app_logins_monthly": 2,
        "feature_adoption_score": 1.0,
        "gender": "Male",
        "senior_citizen": 0,
        "has_partner": 0,
        "has_dependents": 0,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check",
        "paperless_billing": 1,
        "phone_service": 1,
        "multiple_lines": "No",
        "internet_service": "Fiber_optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
    }

    result = predictor.predict_single(sample_customer, customer_id="CUST-0001")
    print("\nPrediction:", result.to_dict())
