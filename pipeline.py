"""
pipeline.py — End-to-end orchestration for the Telecom Churn framework.

Stages:
  1. Data generation (or load from CSV)
  2. Train/val/test split
  3. Preprocessing (fit on train)
  4. XGBoost training
  5. TabTransformer training (MPS GPU on Apple M5 Pro)
  6. Ensemble evaluation
  7. Reports & artefact persistence

Run:
    python pipeline.py                   # generate data and train
    python pipeline.py --data data/telecom_churn.csv  # use existing CSV
    python pipeline.py --skip-transformer             # XGBoost only (fast)
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure local imports resolve
sys.path.insert(0, str(Path(__file__).parent))

from config import ChurnConfig, cfg
from data_generator import MobileDataGenerator as TelecomDataGenerator
from trainer import ChurnTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    config:           ChurnConfig = cfg,
    data_path:        str         = None,
    skip_transformer: bool        = False,
    skip_xgboost:     bool        = False,
) -> dict:
    """
    Execute the full churn prediction pipeline.

    Parameters
    ----------
    config           : ChurnConfig instance
    data_path        : if provided, load CSV instead of generating data
    skip_transformer : skip TabTransformer training
    skip_xgboost     : skip XGBoost training

    Returns
    -------
    dict of {model_name: metrics}
    """
    t_total = time.time()

    logger.info(config.summary())

    # ----------------------------------------------------------------
    # 1. Data
    # ----------------------------------------------------------------
    gen = TelecomDataGenerator(config)

    if data_path and Path(data_path).exists():
        logger.info("Loading data from %s", data_path)
        df = TelecomDataGenerator.load(data_path)
    else:
        logger.info("Generating synthetic mobile subscriber data …")
        df = gen.generate()
        gen.save(df, "data/mobile_churn.csv")

    train_df, val_df, test_df = gen.split(df)

    logger.info(
        "Dataset summary:\n"
        "  Total    : %d  |  churn=%.2f%%\n"
        "  Train    : %d  |  churn=%.2f%%\n"
        "  Val      : %d  |  churn=%.2f%%\n"
        "  Test     : %d  |  churn=%.2f%%",
        len(df),       df[config.target_column].mean() * 100,
        len(train_df), train_df[config.target_column].mean() * 100,
        len(val_df),   val_df[config.target_column].mean() * 100,
        len(test_df),  test_df[config.target_column].mean() * 100,
    )

    # ----------------------------------------------------------------
    # 2. Train
    # ----------------------------------------------------------------
    trainer = ChurnTrainer(config)

    # Optionally patch config to skip models
    if skip_transformer:
        logger.warning("Skipping TabTransformer (--skip-transformer flag).")
        config.ensemble.transformer_weight = 0.0
        config.ensemble.xgb_weight         = 1.0
        config.ensemble.calibrate          = False

    if skip_xgboost:
        logger.warning("Skipping XGBoost (--skip-xgboost flag).")
        config.ensemble.xgb_weight         = 0.0
        config.ensemble.transformer_weight = 1.0
        config.ensemble.calibrate          = False

    results = trainer.run(train_df, val_df, test_df)

    # ----------------------------------------------------------------
    # 3. Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - t_total
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed in %.1f s", elapsed)
    logger.info("=" * 60)

    print("\n📊  Final Results")
    print("-" * 50)
    comparison = trainer.evaluator_.compare()
    print(comparison.to_string())

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Telecom Churn Prediction Pipeline"
    )
    p.add_argument("--data",              type=str,  default=None,
                   help="Path to existing CSV (skips generation)")
    p.add_argument("--n-samples",         type=int,  default=None,
                   help="Number of synthetic samples to generate")
    p.add_argument("--skip-transformer",  action="store_true",
                   help="Skip TabTransformer training")
    p.add_argument("--skip-xgboost",      action="store_true",
                   help="Skip XGBoost training")
    p.add_argument("--epochs",            type=int,  default=None,
                   help="Override transformer epochs")
    p.add_argument("--batch-size",        type=int,  default=None,
                   help="Override transformer batch size")
    p.add_argument("--seed",              type=int,  default=None,
                   help="Override random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Apply CLI overrides to config
    if args.n_samples is not None:
        cfg.data.n_samples = args.n_samples
    if args.epochs is not None:
        cfg.transformer.epochs = args.epochs
    if args.batch_size is not None:
        cfg.transformer.batch_size = args.batch_size
    if args.seed is not None:
        cfg.data.random_seed = args.seed

    run_pipeline(
        config           = cfg,
        data_path        = args.data,
        skip_transformer = args.skip_transformer,
        skip_xgboost     = args.skip_xgboost,
    )
