"""
preprocessor.py — Sklearn ColumnTransformer pipeline for churn features.

Responsibilities:
  - Numerical: outlier capping → StandardScaler (or Robust/MinMax)
  - Categorical: OrdinalEncoder (integer codes for XGBoost tree splits)
  - Fit on train, transform train/val/test consistently
  - Persist / reload fitted pipeline
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
)

from config import ChurnConfig, cfg

logger = logging.getLogger(__name__)

_SCALERS = {
    "standard": StandardScaler,
    "robust":   RobustScaler,
    "minmax":   MinMaxScaler,
}


# ---------------------------------------------------------------------------
# Custom outlier capper (IQR or Z-score)
# ---------------------------------------------------------------------------

class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap numerical values at [lower_fence, upper_fence] derived from training data."""

    def __init__(self, method: str = "iqr", factor: float = 3.0):
        self.method = method
        self.factor = factor
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "OutlierCapper":
        X = np.asarray(X, dtype=float)
        if self.method == "iqr":
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = q3 - q1
            self.lower_ = q1 - self.factor * iqr
            self.upper_ = q3 + self.factor * iqr
        else:  # zscore
            mu  = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
            self.lower_ = mu - self.factor * std
            self.upper_ = mu + self.factor * std
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            X[:, j] = np.clip(X[:, j], self.lower_[j], self.upper_[j])
        return X


# ---------------------------------------------------------------------------
# Main preprocessor class
# ---------------------------------------------------------------------------

class ChurnPreprocessor:
    """
    Wraps a ColumnTransformer that handles numerical and categorical features.

    After fit():
      - transform() returns a 2-D numpy array suitable for XGBoost / PyTorch.
      - feature_names_out lists the column names in the transformed order.
    """

    PIPELINE_PATH = "checkpoints/preprocessor.pkl"

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg            = config
        self.pipeline_: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Build sklearn pipeline
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> ColumnTransformer:
        pcfg = self.cfg.preprocessing

        ScalerCls = _SCALERS[pcfg.numerical_scaler]

        if pcfg.handle_outliers:
            num_pipe = Pipeline([
                ("capper", OutlierCapper(method=pcfg.outlier_method,
                                         factor=pcfg.outlier_cap_factor)),
                ("scaler", ScalerCls()),
            ])
        else:
            num_pipe = Pipeline([("scaler", ScalerCls())])

        # OrdinalEncoder: use –1 for unknown categories (safe for inference)
        cat_pipe = Pipeline([
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            )),
        ])

        ct = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.cfg.numerical_features),
                ("cat", cat_pipe, self.cfg.categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return ct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ChurnPreprocessor":
        """Fit on training DataFrame (must contain feature columns)."""
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(df[self.cfg.all_features()])
        self.feature_names_out_ = (
            self.cfg.numerical_features + self.cfg.categorical_features
        )
        logger.info(
            "Preprocessor fitted | num=%d  cat=%d  total=%d features",
            len(self.cfg.numerical_features),
            len(self.cfg.categorical_features),
            len(self.feature_names_out_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame to a float32 numpy array."""
        assert self.pipeline_ is not None, "Call fit() before transform()."
        return self.pipeline_.transform(df[self.cfg.all_features()]).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        assert self.feature_names_out_ is not None, "Not fitted yet."
        return list(self.feature_names_out_)

    def get_numerical_indices(self) -> List[int]:
        names = self.get_feature_names()
        return [names.index(f) for f in self.cfg.numerical_features]

    def get_categorical_indices(self) -> List[int]:
        names = self.get_feature_names()
        return [names.index(f) for f in self.cfg.categorical_features]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.PIPELINE_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Preprocessor saved → %s", path)

    @staticmethod
    def load(path: str = PIPELINE_PATH) -> "ChurnPreprocessor":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("Preprocessor loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Convenience: split a combined DataFrame into (X, y) arrays
    # ------------------------------------------------------------------

    def extract_Xy(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X_transformed, y) for a labelled DataFrame."""
        X = self.transform(df)
        y = df[self.cfg.target_column].values.astype(np.int64)
        return X, y


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from data_generator import TelecomDataGenerator

    gen   = TelecomDataGenerator()
    df    = gen.generate()
    train, val, test = gen.split(df)

    prep  = ChurnPreprocessor()
    X_tr  = prep.fit_transform(train)
    X_va  = prep.transform(val)
    X_te  = prep.transform(test)

    print("Train shape:", X_tr.shape)
    print("Val   shape:", X_va.shape)
    print("Test  shape:", X_te.shape)
    print("Feature names:", prep.get_feature_names())
    prep.save()
