"""
evaluator.py — Evaluation metrics and reporting for churn models.

Computes:
  - AUROC, AUPRC, Brier score, F1, Precision, Recall, Accuracy
  - Bootstrap confidence intervals
  - Confusion matrix
  - Calibration curve
  - Optional SHAP values (XGBoost only in this version)
  - Saves all plots to results/
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ChurnConfig, cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def _brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, y_score))


def _classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc":     _auroc(y_true, y_score),
        "auprc":     _auprc(y_true, y_score),
        "brier":     _brier(y_true, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (point_estimate, lower_ci, upper_ci)."""
    rng  = np.random.default_rng(seed)
    idx  = np.arange(len(y_true))
    vals = [
        metric_fn(y_true[s], y_score[s])
        for s in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n))
    ]
    alpha  = (1 - ci) / 2
    return (
        metric_fn(y_true, y_score),
        float(np.percentile(vals, 100 * alpha)),
        float(np.percentile(vals, 100 * (1 - alpha))),
    )


# ---------------------------------------------------------------------------
# Plotting helpers (deferred import so matplotlib is optional)
# ---------------------------------------------------------------------------

def _plot_roc(y_true, y_score, label: str, out_path: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC Curve — {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved → %s", out_path)


def _plot_pr(y_true, y_score, label: str, out_path: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, name=label, ax=ax)
    ax.set_title(f"Precision–Recall — {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("PR curve saved → %s", out_path)


def _plot_confusion(tn, fp, fn, tp, label: str, out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val:,}", ha="center", va="center",
                color="white" if val > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(f"Confusion Matrix — {label}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_calibration(y_true, y_score, label: str, out_path: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(fi: Dict[str, float], label: str, out_path: str, top_n: int = 20) -> None:
    import matplotlib.pyplot as plt
    items = sorted(fi.items(), key=lambda x: x[1])[-top_n:]
    names, scores = zip(*items)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, scores)
    ax.set_title(f"Feature Importance (gain) — {label}")
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class ChurnEvaluator:
    """Evaluate one or more churn models and save reports."""

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg     = config
        self.ecfg    = config.evaluation
        self.results_: Dict[str, Dict[str, Any]] = {}

        Path(self.ecfg.results_dir).mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true:  np.ndarray,
        y_score: np.ndarray,
        label:   str = "model",
        feature_importance: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation pass.

        Parameters
        ----------
        y_true  : ground-truth binary labels
        y_score : predicted probabilities for class 1
        label   : display name for the model
        feature_importance : dict of {feature_name: score} (optional)
        threshold : decision boundary (default from config)
        """
        thr     = threshold or self.ecfg.threshold
        y_pred  = (y_score >= thr).astype(int)

        metrics = _classification_report(y_true, y_pred, y_score)

        # Bootstrap CI for AUROC
        auroc_pt, auroc_lo, auroc_hi = bootstrap_metric(
            y_true, y_score, _auroc,
            n=self.ecfg.bootstrap_rounds,
        )
        metrics["auroc_ci_low"]  = auroc_lo
        metrics["auroc_ci_high"] = auroc_hi

        logger.info(
            "[%s] AUROC=%.4f (95%% CI [%.4f, %.4f]) | AUPRC=%.4f | F1=%.4f",
            label, auroc_pt, auroc_lo, auroc_hi, metrics["auprc"], metrics["f1"],
        )

        if self.ecfg.save_plots:
            rdir = self.ecfg.results_dir
            safe = label.replace(" ", "_").lower()
            try:
                _plot_roc(y_true, y_score, label,
                          f"{rdir}/{safe}_roc.png")
                _plot_pr(y_true, y_score, label,
                         f"{rdir}/{safe}_pr.png")
                _plot_confusion(
                    metrics["tn"], metrics["fp"],
                    metrics["fn"], metrics["tp"],
                    label, f"{rdir}/{safe}_confusion.png",
                )
                _plot_calibration(y_true, y_score, label,
                                  f"{rdir}/{safe}_calibration.png")
                if feature_importance:
                    _plot_feature_importance(
                        feature_importance, label,
                        f"{rdir}/{safe}_feature_importance.png",
                    )
            except Exception as exc:
                logger.warning("Plot generation failed: %s", exc)

        self.results_[label] = metrics
        return metrics

    def compare(self) -> pd.DataFrame:
        """Return a DataFrame comparing all evaluated models."""
        if not self.results_:
            return pd.DataFrame()
        cols = ["auroc", "auprc", "f1", "precision", "recall", "accuracy", "brier"]
        rows = {
            lbl: {c: m.get(c, float("nan")) for c in cols}
            for lbl, m in self.results_.items()
        }
        return pd.DataFrame(rows).T.sort_values("auroc", ascending=False)

    def save_report(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.ecfg.results_dir, "evaluation_report.json")
        with open(path, "w") as fh:
            json.dump(self.results_, fh, indent=2, default=float)
        logger.info("Evaluation report saved → %s", path)

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Sweep thresholds and return (best_threshold, best_metric_value).
        Useful for tuning the decision boundary post-training.
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        metric_fn_map = {
            "f1":        lambda yt, yp: f1_score(yt, yp, zero_division=0),
            "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
            "recall":    lambda yt, yp: recall_score(yt, yp, zero_division=0),
        }
        fn = metric_fn_map.get(metric, metric_fn_map["f1"])

        thresholds  = np.linspace(0.05, 0.95, 181)
        best_thr, best_val = 0.5, 0.0
        for thr in thresholds:
            val = fn(y_true, (y_score >= thr).astype(int))
            if val > best_val:
                best_val, best_thr = val, thr

        logger.info("Optimal threshold (%s): %.3f → %.4f", metric, best_thr, best_val)
        return float(best_thr), float(best_val)
