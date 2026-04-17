"""
inference.py — Production inference engine for mobile churn prediction.

Supports:
  - Single subscriber prediction (real-time, ~1 ms)
  - Batch prediction from DataFrame or CSV
  - Ensemble scoring (XGBoost + TabTransformer)
  - Risk segments: high_risk / medium_risk / low_risk
  - Lazy model loading (once on first call)

Usage:
    predictor = ChurnPredictor.from_checkpoints()
    result    = predictor.predict_single(subscriber_dict)
    batch_df  = predictor.predict_batch(df)
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
    subscriber_id:     Optional[str]
    churn_probability: float
    churn_label:       int           # 0 = retained, 1 = churn
    risk_segment:      str           # high_risk | medium_risk | low_risk
    xgb_score:         float
    transformer_score: float
    latency_ms:        float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscriber_id":     self.subscriber_id,
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
    """Lazy-loaded inference engine wrapping preprocessor + ensemble."""

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg     = config
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
        config:            ChurnConfig = cfg,
        preprocessor_path: str = "checkpoints/preprocessor.pkl",
        xgb_path:          str = "checkpoints/xgboost_mobile_churn.json",
        transformer_path:  str = "checkpoints/transformer_mobile_churn_best.pt",
    ) -> "ChurnPredictor":
        predictor = cls(config)
        predictor._load_models(preprocessor_path, xgb_path, transformer_path)
        return predictor

    def _load_models(self, prep_path, xgb_path, trn_path) -> None:
        if self._loaded:
            return
        logger.info("Loading mobile churn models …")
        self._prep = ChurnPreprocessor.load(prep_path)
        self._xgb  = XGBoostChurnModel(self.cfg)
        self._xgb.load(xgb_path)
        self._trn  = TabTransformerChurnModel(self.cfg)
        self._trn.num_idx_ = self._prep.get_numerical_indices()
        self._trn.cat_idx_ = self._prep.get_categorical_indices()
        self._trn.load(trn_path)
        self._loaded = True
        logger.info("Models loaded.")

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score(self, X: np.ndarray):
        return self._xgb.predict_proba(X), self._trn.predict_proba(X)

    def _ensemble(self, xgb_s, trn_s) -> np.ndarray:
        ec = self.cfg.ensemble
        return (ec.xgb_weight * xgb_s + ec.transformer_weight * trn_s) / (
            ec.xgb_weight + ec.transformer_weight
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(
        self,
        subscriber: Dict[str, Any],
        subscriber_id: Optional[str] = None,
        threshold: float = 0.5,
    ) -> ChurnPrediction:
        t0  = time.perf_counter()
        X   = self._prep.transform(pd.DataFrame([subscriber]))
        xs, ts = self._score(X)
        prob = float(self._ensemble(xs, ts)[0])
        ms   = (time.perf_counter() - t0) * 1000
        return ChurnPrediction(
            subscriber_id     = subscriber_id,
            churn_probability = prob,
            churn_label       = int(prob >= threshold),
            risk_segment      = _risk_segment(prob),
            xgb_score         = float(xs[0]),
            transformer_score = float(ts[0]),
            latency_ms        = ms,
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        t0 = time.perf_counter()
        X  = self._prep.transform(df)
        xs, ts = self._score(X)
        ens = self._ensemble(xs, ts)

        out = pd.DataFrame(index=df.index)
        if id_column and id_column in df.columns:
            out[id_column] = df[id_column]
        out["churn_probability"]  = ens.round(4)
        out["churn_label"]        = (ens >= threshold).astype(int)
        out["risk_segment"]       = [_risk_segment(p) for p in ens]
        out["xgb_score"]          = xs.round(4)
        out["transformer_score"]  = ts.round(4)

        ms = (time.perf_counter() - t0) * 1000
        logger.info("Batch: %d records | %.1f ms | %.3f ms/record",
                    len(df), ms, ms / max(len(df), 1))
        return out

    def predict_batch_from_file(
        self, path: str, output_path: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        df  = pd.read_csv(path)
        out = self.predict_batch(df, id_column=id_column)
        if output_path:
            out.to_csv(output_path, index=False)
            logger.info("Results → %s", output_path)
        return out

    def warmup(self, n: int = 4) -> None:
        logger.info("Warming up inference engine (n=%d) …", n)
        dummy = _sample_high_risk_subscriber()
        for _ in range(n):
            self.predict_single(dummy)
        logger.info("Warmup complete.")


# ---------------------------------------------------------------------------
# Sample subscriber helpers
# ---------------------------------------------------------------------------

def _sample_high_risk_subscriber() -> Dict[str, Any]:
    """
    A prepaid subscriber with poor network quality, low top-ups,
    multiple complaints, and high call drop rate — classic high-risk profile.
    """
    return {
        # Demographics
        "age": 24,
        # Account
        "tenure_months": 3.0,
        "monthly_plan_cost_eur": 8.5,
        "total_spend_6mo_eur": 22.0,
        # Network quality KPIs
        "avg_sinr_db": -2.5,            # poor SINR
        "avg_rsrp_dbm": -107.0,         # weak signal
        "call_success_rate_pct": 93.5,  # below threshold
        "bearer_establishment_rate_pct": 94.1,
        "avg_call_drop_rate_pct": 5.2,  # high drop rate
        "avg_data_speed_mbps": 2.1,
        # Usage
        "avg_monthly_data_gb": 1.2,
        "avg_daily_data_usage_mb": 41.0,
        "avg_daily_call_minutes": 18.0,
        "avg_monthly_sms": 45,
        "roaming_sessions_6mo": 0,
        "intl_call_minutes_monthly": 0.0,
        # Billing
        "num_topups_6mo": 1,            # barely topped up → high lapse risk
        "avg_topup_amount_eur": 7.0,
        "num_late_payments_6mo": 0,
        "data_overage_charges_eur": 0.0,
        # Customer service
        "num_complaints_6mo": 4,
        "num_support_calls_6mo": 5,
        "days_since_last_complaint": 8.0,
        # Categorical
        "plan_type": "prepaid",
        "plan_tier": "basic",
        "contract_duration": "prepaid",
        "payment_method": "cash",
        "network_generation": "4G",
        "device_type": "smartphone",
        "gender": "Male",
        "is_roaming_enabled": "no",
        "has_handset_subsidy": "no",
    }


def _sample_low_risk_subscriber() -> Dict[str, Any]:
    """
    A long-tenure postpaid subscriber on a 24-month contract with
    good network quality and subsidised handset — classic low-risk profile.
    """
    return {
        "age": 38,
        "tenure_months": 54.0,
        "monthly_plan_cost_eur": 17.5,
        "total_spend_6mo_eur": 107.0,
        "avg_sinr_db": 18.5,
        "avg_rsrp_dbm": -78.0,
        "call_success_rate_pct": 99.2,
        "bearer_establishment_rate_pct": 99.5,
        "avg_call_drop_rate_pct": 0.4,
        "avg_data_speed_mbps": 95.0,
        "avg_monthly_data_gb": 22.0,
        "avg_daily_data_usage_mb": 760.0,
        "avg_daily_call_minutes": 35.0,
        "avg_monthly_sms": 20,
        "roaming_sessions_6mo": 4,
        "intl_call_minutes_monthly": 12.0,
        "num_topups_6mo": 0,
        "avg_topup_amount_eur": 0.0,
        "num_late_payments_6mo": 0,
        "data_overage_charges_eur": 0.0,
        "num_complaints_6mo": 0,
        "num_support_calls_6mo": 1,
        "days_since_last_complaint": 180.0,
        "plan_type": "postpaid",
        "plan_tier": "unlimited",
        "contract_duration": "24_month",
        "payment_method": "direct_debit",
        "network_generation": "5G",
        "device_type": "smartphone",
        "gender": "Female",
        "is_roaming_enabled": "yes",
        "has_handset_subsidy": "yes",
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    predictor = ChurnPredictor.from_checkpoints()
    predictor.warmup()

    for label, sub in [
        ("High-risk prepaid", _sample_high_risk_subscriber()),
        ("Low-risk postpaid",  _sample_low_risk_subscriber()),
    ]:
        r = predictor.predict_single(sub, subscriber_id=label)
        print(f"\n[{label}]")
        for k, v in r.to_dict().items():
            print(f"  {k:25s}: {v}")
