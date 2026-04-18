"""
shap_explainer.py — SHAP (SHapley Additive exPlanations) for the XGBoost churn model.

What SHAP does
--------------
SHAP assigns each feature a "contribution score" for every single prediction.
Unlike regular feature importance (which is global), SHAP tells you:

  "For THIS subscriber, avg_sinr_db pushed their churn probability UP by +0.18,
   while their 24-month contract pulled it DOWN by -0.31."

This makes the model interpretable for both data scientists and business users.

Plots generated
---------------
1. shap_summary_beeswarm.png  — Global: distribution of SHAP values per feature
2. shap_bar_global.png        — Global: mean |SHAP| ranked bar chart
3. shap_waterfall_highrisk.png — Local: breakdown for the highest-risk subscriber
4. shap_waterfall_lowrisk.png  — Local: breakdown for the lowest-risk subscriber
5. shap_dependence_<feat>.png  — Dependence plots for top-3 features
6. shap_heatmap.png            — Heatmap across a random sample of subscribers

Saved data
----------
results/shap_values.npy       — Full SHAP value matrix [n_test, n_features]
results/shap_summary.csv      — Mean |SHAP| per feature (sorted)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt

from config import ChurnConfig, cfg

logger = logging.getLogger(__name__)


class ShapExplainer:
    """
    Wraps shap.TreeExplainer for an XGBoost churn model.

    TreeExplainer is:
      - Exact (not an approximation) for tree-based models
      - Very fast: O(TLD) where T=trees, L=leaves, D=depth
      - Consistent: satisfies all Shapley axioms (efficiency, symmetry, dummy)
    """

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg         = config
        self.results_dir = Path(config.evaluation.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.explainer_  = None
        self.shap_values_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, xgb_model, X_background: np.ndarray) -> "ShapExplainer":
        """
        Build the TreeExplainer from a trained XGBoostChurnModel.

        Parameters
        ----------
        xgb_model    : fitted XGBoostChurnModel instance
        X_background : representative background dataset (training set or a
                       sample of it) used to compute the expected value (base rate)
        """
        import shap
        assert xgb_model.model_ is not None, "XGBoost model must be trained first."

        logger.info("Fitting SHAP TreeExplainer …")
        # TreeExplainer takes the underlying booster directly
        self.explainer_ = shap.TreeExplainer(
            xgb_model.model_,
            data              = X_background,
            feature_perturbation = "interventional",  # causal interpretation
            model_output      = "probability",        # SHAP in probability space
        )
        logger.info(
            "SHAP base value (expected churn probability): %.4f",
            float(np.mean(self.explainer_.expected_value))
            if hasattr(self.explainer_.expected_value, "__len__")
            else float(self.explainer_.expected_value),
        )
        return self

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(
        self,
        X_test:        np.ndarray,
        feature_names: List[str],
        max_samples:   int = 2000,
    ) -> np.ndarray:
        """
        Compute SHAP values for the test set.

        Parameters
        ----------
        X_test        : preprocessed test array  [n_test, n_features]
        feature_names : list of feature names (same order as X_test columns)
        max_samples   : cap for expensive computations (full matrix still saved)

        Returns
        -------
        shap_values : np.ndarray of shape [n_test, n_features]
                      positive = pushed churn probability UP
                      negative = pushed churn probability DOWN
        """
        import shap
        assert self.explainer_ is not None, "Call fit() before compute()."

        self.feature_names_ = feature_names
        n = len(X_test)

        logger.info("Computing SHAP values for %d test subscribers …", n)
        raw = self.explainer_.shap_values(X_test, check_additivity=False)

        # TreeExplainer returns list [class0, class1] for binary classification
        # We want class-1 (churn) SHAP values
        if isinstance(raw, list):
            sv = raw[1]
        else:
            sv = raw

        self.shap_values_ = sv.astype(np.float32)

        # Persist raw values and summary CSV
        np.save(str(self.results_dir / "shap_values.npy"), self.shap_values_)
        self._save_summary_csv()

        logger.info("SHAP values computed and saved.")
        return self.shap_values_

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _shap_df(self, X: np.ndarray) -> pd.DataFrame:
        """Wrap X as a DataFrame with feature names for shap plots."""
        return pd.DataFrame(X, columns=self.feature_names_)

    def plot_all(
        self,
        X_test:    np.ndarray,
        y_test:    np.ndarray,
        max_display: int = 20,
        heatmap_samples: int = 200,
    ) -> None:
        """
        Generate all SHAP plots and save to results/.

        Parameters
        ----------
        X_test          : preprocessed test array
        y_test          : true binary labels
        max_display     : max features shown in summary / bar plots
        heatmap_samples : number of subscribers in heatmap (keep small)
        """
        assert self.shap_values_ is not None, "Call compute() before plot_all()."
        import shap

        sv   = self.shap_values_
        feat = self._shap_df(X_test)

        logger.info("Generating SHAP plots …")

        self._plot_summary_beeswarm(sv, feat, max_display)
        self._plot_bar_global(sv, max_display)
        self._plot_waterfall_highrisk(sv, feat, X_test)
        self._plot_waterfall_lowrisk(sv, feat, X_test)
        self._plot_top3_dependence(sv, feat, X_test)
        self._plot_heatmap(sv, feat, heatmap_samples)

        logger.info("All SHAP plots saved to %s", self.results_dir)

    # ------------------------------------------------------------------
    # Individual plot methods
    # ------------------------------------------------------------------

    def _plot_summary_beeswarm(self, sv, feat_df, max_display):
        """
        Beeswarm plot: every dot is one subscriber.
        X-axis = SHAP value (impact on churn probability).
        Colour  = actual feature value (red = high, blue = low).

        This is the most information-rich SHAP plot — read it as:
          "High SINR (blue on left) reduces churn probability.
           Low SINR (red on left) increases it."
        """
        import shap
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv, feat_df,
            max_display    = max_display,
            show           = False,
            plot_type      = "dot",
            color_bar_label= "Feature value\n(red=high, blue=low)",
        )
        plt.title("SHAP Summary — Feature Impact on Churn Probability", fontsize=13, pad=12)
        plt.tight_layout()
        out = self.results_dir / "shap_summary_beeswarm.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Beeswarm plot saved → %s", out)

    def _plot_bar_global(self, sv, max_display):
        """
        Bar plot: mean |SHAP value| per feature = global importance ranking.
        Longer bar → feature has larger average impact across all subscribers.
        """
        import shap

        mean_abs = np.abs(sv).mean(axis=0)
        indices  = np.argsort(mean_abs)[::-1][:max_display]
        names    = [self.feature_names_[i] for i in indices]
        values   = mean_abs[indices]

        fig, ax = plt.subplots(figsize=(9, 7))
        bars = ax.barh(names[::-1], values[::-1],
                       color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.85, len(names))))
        ax.set_xlabel("Mean |SHAP value|  (average impact on churn probability)", fontsize=11)
        ax.set_title("Global Feature Importance — XGBoost SHAP", fontsize=13)

        # Annotate bars with values
        for bar, val in zip(bars, values[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        plt.tight_layout()
        out = self.results_dir / "shap_bar_global.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Bar plot saved → %s", out)

    def _plot_waterfall_highrisk(self, sv, feat_df, X_test):
        """
        Waterfall for the single highest-risk subscriber.
        Shows which features drove this specific prediction up or down
        from the base rate (average churn probability).

        Read it as:
          base_value (0.28)
          + avg_sinr_db contribution  (+0.18)
          + num_complaints contribution (+0.14)
          - has_handset_subsidy contribution (-0.05)
          ...
          = final probability (0.87)
        """
        import shap
        idx_max = int(np.argmax(sv.sum(axis=1)))   # highest total SHAP
        self._waterfall_single(sv, feat_df, idx_max,
                               "shap_waterfall_highrisk.png",
                               "Waterfall — Highest-Risk Subscriber")

    def _plot_waterfall_lowrisk(self, sv, feat_df, X_test):
        """Waterfall for the single lowest-risk subscriber."""
        idx_min = int(np.argmin(sv.sum(axis=1)))
        self._waterfall_single(sv, feat_df, idx_min,
                               "shap_waterfall_lowrisk.png",
                               "Waterfall — Lowest-Risk Subscriber")

    def _waterfall_single(self, sv, feat_df, idx, filename, title):
        import shap
        exp = shap.Explanation(
            values       = sv[idx],
            base_values  = float(self.explainer_.expected_value[1])
                           if hasattr(self.explainer_.expected_value, "__len__")
                           else float(self.explainer_.expected_value),
            data         = feat_df.iloc[idx].values,
            feature_names= self.feature_names_,
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(exp, max_display=15, show=False)
        plt.title(title, fontsize=12, pad=8)
        plt.tight_layout()
        out = self.results_dir / filename
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Waterfall saved → %s", out)

    def _plot_top3_dependence(self, sv, feat_df, X_test):
        """
        Dependence plots for the top-3 most impactful features.
        X-axis = raw feature value.
        Y-axis = SHAP value for that feature.
        Colour = automatically chosen most interacting feature.

        Shows non-linear relationships, e.g.:
          "SINR below 5 dB dramatically increases churn;
           above 15 dB the benefit flattens out."
        """
        import shap
        mean_abs = np.abs(sv).mean(axis=0)
        top3     = np.argsort(mean_abs)[::-1][:3]

        for rank, feat_idx in enumerate(top3, 1):
            feat_name = self.feature_names_[feat_idx]
            fig, ax   = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(
                feat_idx,
                sv,
                feat_df,
                ax          = ax,
                show        = False,
                dot_size    = 8,
                alpha       = 0.5,
            )
            ax.set_title(
                f"SHAP Dependence — {feat_name}  (rank #{rank} globally)",
                fontsize=12,
            )
            ax.set_ylabel("SHAP value\n(impact on churn probability)", fontsize=10)
            plt.tight_layout()
            safe_name = feat_name.replace("/", "_")
            out = self.results_dir / f"shap_dependence_{safe_name}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Dependence plot saved → %s", out)

    def _plot_heatmap(self, sv, feat_df, n_samples: int):
        """
        Heatmap across n_samples subscribers × all features.
        Rows = subscribers (sorted by total SHAP = risk score).
        Columns = features (sorted by mean |SHAP|).
        Colour = SHAP value (red = pushing toward churn, blue = away).

        Good for spotting clusters of subscribers with similar risk drivers.
        """
        import shap

        # Take a random sample for readability
        rng = np.random.default_rng(42)
        idx = rng.choice(len(sv), size=min(n_samples, len(sv)), replace=False)
        sv_sub   = sv[idx]
        feat_sub = feat_df.iloc[idx]

        exp = shap.Explanation(
            values        = sv_sub,
            base_values   = np.zeros(len(sv_sub)),   # heatmap doesn't need base_values
            data          = feat_sub.values,
            feature_names = self.feature_names_,
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        shap.plots.heatmap(exp, max_display=20, show=False, instance_order=exp.sum(1))
        plt.title(
            f"SHAP Heatmap — {n_samples} subscribers (sorted by total risk score)",
            fontsize=12, pad=10,
        )
        plt.tight_layout()
        out = self.results_dir / "shap_heatmap.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Heatmap saved → %s", out)

    # ------------------------------------------------------------------
    # CSV summary
    # ------------------------------------------------------------------

    def _save_summary_csv(self) -> None:
        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        mean_pos = self.shap_values_.mean(axis=0)       # direction of average effect
        df = pd.DataFrame({
            "feature":        self.feature_names_,
            "mean_abs_shap":  mean_abs.round(6),
            "mean_shap":      mean_pos.round(6),        # positive = on avg pushes toward churn
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        out = self.results_dir / "shap_summary.csv"
        df.to_csv(out, index=False)
        logger.info("SHAP summary CSV saved → %s", out)

        # Log top-10 to console
        logger.info("\nTop-10 features by mean |SHAP|:\n%s",
                    df.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # Convenience: print a single subscriber's explanation
    # ------------------------------------------------------------------

    def explain_subscriber(
        self,
        X_row: np.ndarray,
        subscriber_id: str = "unknown",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return a ranked DataFrame of SHAP contributions for one subscriber.

        Example output:
            rank  feature                    shap_value   direction
               1  num_complaints_6mo          +0.2341     ↑ churn
               2  avg_sinr_db                 +0.1823     ↑ churn
               3  has_handset_subsidy         -0.1540     ↓ churn
               ...
        """
        assert self.explainer_ is not None, "Call fit() first."
        raw  = self.explainer_.shap_values(X_row.reshape(1, -1), check_additivity=False)
        sv   = raw[1][0] if isinstance(raw, list) else raw[0]
        base = (float(self.explainer_.expected_value[1])
                if hasattr(self.explainer_.expected_value, "__len__")
                else float(self.explainer_.expected_value))

        df = pd.DataFrame({
            "feature":    self.feature_names_,
            "shap_value": sv,
            "feature_value": X_row,
        })
        df["abs_shap"]  = df["shap_value"].abs()
        df["direction"] = df["shap_value"].apply(
            lambda v: "↑ churn" if v > 0 else ("↓ churn" if v < 0 else "—")
        )
        df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
        df["rank"] = df.index + 1

        final_prob = base + sv.sum()
        logger.info(
            "Subscriber %s | base=%.4f | SHAP sum=%.4f | final_prob≈%.4f",
            subscriber_id, base, sv.sum(), final_prob,
        )
        return df[["rank", "feature", "feature_value", "shap_value", "direction"]]
