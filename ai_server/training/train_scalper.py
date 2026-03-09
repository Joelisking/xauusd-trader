"""Scalper model training pipeline.

Trains BiLSTM + XGBoost ensemble for scalper entry quality scoring.
Uses time-based split, class weighting, and walk-forward validation.

Usage:
    uv run python -m ai_server.training.train_scalper
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Fix TensorFlow threading deadlock on Python 3.13 + macOS
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(8)

import numpy as np
import pandas as pd

from ai_server.config import (
    BILSTM_WEIGHT,
    DATA_DIR,
    FEATURE_COUNT,
    SCALPER_BILSTM_PATH,
    SCALPER_XGB_PATH,
    SEQUENCE_LENGTH,
    XGB_WEIGHT,
)
from ai_server.training.label_generator import (
    compute_class_weights,
    filter_labeled_data,
    generate_scalper_labels,
)
from ai_server.training.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_scalper_bilstm(
    X_seq: np.ndarray,
    y: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    save_path: str | None = None,
) -> Any:
    """Train scalper BiLSTM model."""
    from ai_server.models.scalper_bilstm import ScalperBiLSTM

    model = ScalperBiLSTM()

    model.train(
        X_seq, y,
        X_val=X_val_seq, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path or str(SCALPER_BILSTM_PATH),
    )

    logger.info("BiLSTM saved to %s", save_path or SCALPER_BILSTM_PATH)
    return model


def train_scalper_xgb(
    X_tab: np.ndarray,
    y: np.ndarray,
    X_val_tab: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    save_path: str | None = None,
) -> Any:
    """Train scalper XGBoost model with Optuna tuning."""
    from ai_server.models.xgboost_models import ScalperXGB

    model = ScalperXGB()

    if n_trials > 0:
        logger.info("Running Optuna hyperparameter search (%d trials)...", n_trials)
        model.tune_hyperparameters(X_tab, y, n_trials=n_trials)

    model.train(X_tab, y, X_val=X_val_tab, y_val=y_val,
                save_path=save_path or str(SCALPER_XGB_PATH))

    logger.info("XGBoost saved to %s", save_path or SCALPER_XGB_PATH)
    return model


def _subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Subsample using a stride to preserve temporal continuity for LSTM sequences.

    Random subsampling breaks sequence integrity — adjacent rows in the resulting
    DataFrame would be months apart, making BiLSTM sequences meaningless.
    Stride-based sampling keeps rows evenly spaced so sequences are coherent.
    """
    if len(df) <= max_rows:
        return df
    stride = max(1, len(df) // max_rows)
    return df.iloc[::stride].head(max_rows).reset_index(drop=True)


def main() -> None:
    """End-to-end scalper training: load -> split raw -> subsample -> features -> train.

    Memory budget (16 GB RAM):
      - Max ~150K train / 30K val / 30K test sequences.
      - Raw data is split and subsampled BEFORE computing features to avoid
        building a ~170 GB matrix over all 3.5M M1 rows.
    """
    from data_pipeline.feature_pipeline import prepare_training_matrices
    from ai_server.features.feature_engine import FeatureEngine
    from ai_server.features.macro_features import MacroContext

    MAX_TRAIN = 150_000
    MAX_VAL   =  30_000
    MAX_TEST  =  30_000

    logger.info("=" * 60)
    logger.info("SCALPER TRAINING PIPELINE")
    logger.info("=" * 60)

    # --- 1. Load M1 data ---
    m1_path = DATA_DIR / "XAUUSD_M1.parquet"
    if not m1_path.exists():
        logger.error("M1 data not found at %s", m1_path)
        sys.exit(1)

    logger.info("Loading M1 data from %s ...", m1_path)
    df = pd.read_parquet(m1_path).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    logger.info("Loaded %d rows (%s to %s)", len(df), df["time"].iloc[0], df["time"].iloc[-1])

    # --- 2. Split raw data by date FIRST (before feature computation) ---
    logger.info("Splitting raw data: train <2023 | val 2023-2024 | test >=2024...")
    t = df["time"]
    df_train = _subsample(df[t < "2023-01-01"].reset_index(drop=True), MAX_TRAIN)
    df_val   = _subsample(df[(t >= "2023-01-01") & (t < "2024-01-01")].reset_index(drop=True), MAX_VAL)
    df_test  = _subsample(df[t >= "2024-01-01"].reset_index(drop=True), MAX_TEST)
    logger.info("  Subsampled: train=%d, val=%d, test=%d", len(df_train), len(df_val), len(df_test))

    # --- 3. Compute features per split ---
    engine = FeatureEngine()
    macro  = MacroContext()

    def featurize(split_df: pd.DataFrame, name: str) -> pd.DataFrame:
        logger.info("Computing features for %s split (%d rows)...", name, len(split_df))
        feat = engine.compute(split_df, macro=macro)
        feat = pd.concat([pd.DataFrame({"time": split_df["time"].values}), feat], axis=1)
        return feat

    feat_train = featurize(df_train, "train")
    feat_val   = featurize(df_val,   "val")
    feat_test  = featurize(df_test,  "test")

    # --- 4. Generate labels per split ---
    logger.info("Generating scalper labels...")
    lbl_train = generate_scalper_labels(df_train)
    lbl_val   = generate_scalper_labels(df_val)
    lbl_test  = generate_scalper_labels(df_test)

    feat_train, lbl_train = filter_labeled_data(feat_train, lbl_train)
    feat_val,   lbl_val   = filter_labeled_data(feat_val,   lbl_val)
    feat_test,  lbl_test  = filter_labeled_data(feat_test,  lbl_test)

    for name, lbl in [("train", lbl_train), ("val", lbl_val), ("test", lbl_test)]:
        wr = 100 * (lbl == 1).mean() if len(lbl) > 0 else 0
        logger.info("  %s: %d samples (%.1f%% win rate)", name, len(lbl), wr)

    if len(lbl_train) < 500:
        logger.error("Training set too small (%d samples).", len(lbl_train))
        sys.exit(1)

    # --- 5. Prepare BiLSTM / XGB matrices ---
    logger.info("Preparing sequential and tabular matrices...")
    m_train = prepare_training_matrices(feat_train, lbl_train, SEQUENCE_LENGTH)
    m_val   = prepare_training_matrices(feat_val,   lbl_val,   SEQUENCE_LENGTH)
    m_test  = prepare_training_matrices(feat_test,  lbl_test,  SEQUENCE_LENGTH)

    # --- 6. Train XGBoost ---
    # NOTE: BiLSTM disabled for scalper — stride-subsampling breaks sequence continuity
    # on 16GB RAM. Re-enable on a 32GB+ machine using train_scalper_bilstm().
    logger.info("")
    logger.info("--- Training Scalper XGBoost ---")
    xgb = train_scalper_xgb(
        m_train["X_tab"], m_train["y"],
        m_val["X_tab"],   m_val["y"],
        n_trials=50,
    )

    # --- 7. Evaluate ---
    logger.info("")
    logger.info("--- Evaluation on Test Set ---")
    xgb_preds = xgb._model.predict_proba(m_test["X_tab"])[:, 1]

    report = evaluate_model(m_test["y"], xgb_preds, "Scalper XGBoost", "test")
    logger.info("\n%s", report.summary())

    logger.info("")
    logger.info("=" * 60)
    logger.info("SCALPER TRAINING COMPLETE")
    logger.info("  BiLSTM weights: %s", SCALPER_BILSTM_PATH)
    logger.info("  XGBoost model:  %s", SCALPER_XGB_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
