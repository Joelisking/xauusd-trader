"""Swing model training pipeline.

Trains BiLSTM + XGBoost ensemble for swing entry and trend scoring.
Same architecture as scalper but trained on H1 data with different labels.

Usage:
    uv run python -m ai_server.training.train_swing
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

from ai_server.config import (
    BILSTM_WEIGHT,
    DATA_DIR,
    SEQUENCE_LENGTH,
    SWING_BILSTM_PATH,
    SWING_XGB_PATH,
    XGB_WEIGHT,
)
from ai_server.training.label_generator import (
    compute_class_weights,
    filter_labeled_data,
    generate_swing_labels,
)
from ai_server.training.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_swing_bilstm(
    X_seq: np.ndarray,
    y: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    save_path: str | None = None,
) -> Any:
    """Train swing BiLSTM model."""
    from ai_server.models.swing_bilstm import SwingBiLSTM

    model = SwingBiLSTM()

    model.train(
        X_seq, y,
        X_val=X_val_seq, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path or str(SWING_BILSTM_PATH),
    )

    logger.info("BiLSTM saved to %s", save_path or SWING_BILSTM_PATH)
    return model


def train_swing_xgb(
    X_tab: np.ndarray,
    y: np.ndarray,
    X_val_tab: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    save_path: str | None = None,
) -> Any:
    """Train swing XGBoost model with Optuna tuning."""
    from ai_server.models.xgboost_models import SwingXGB

    model = SwingXGB()

    if n_trials > 0:
        logger.info("Running Optuna hyperparameter search (%d trials)...", n_trials)
        model.tune_hyperparameters(X_tab, y, n_trials=n_trials)

    model.train(X_tab, y, X_val=X_val_tab, y_val=y_val,
                save_path=save_path or str(SWING_XGB_PATH))

    logger.info("XGBoost saved to %s", save_path or SWING_XGB_PATH)
    return model


def main() -> None:
    """End-to-end swing training: load H1 data -> features -> labels -> train.

    H1 has 58K rows over 10 years — fits fully in memory, no subsampling needed.
    BiLSTM works properly here (contiguous sequences, no stride gaps).
    Ensemble = BiLSTM * 0.55 + XGBoost * 0.45 (both used for swing).
    """
    import pandas as pd
    from data_pipeline.feature_pipeline import prepare_training_matrices
    from ai_server.features.feature_engine import FeatureEngine
    from ai_server.features.macro_features import MacroContext

    logger.info("=" * 60)
    logger.info("SWING TRAINING PIPELINE")
    logger.info("=" * 60)

    # --- 1. Load H1 data ---
    h1_path = DATA_DIR / "XAUUSD_H1.parquet"
    if not h1_path.exists():
        logger.error("H1 data not found at %s", h1_path)
        sys.exit(1)

    logger.info("Loading H1 data from %s ...", h1_path)
    df = pd.read_parquet(h1_path).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    logger.info("Loaded %d rows (%s to %s)", len(df), df["time"].iloc[0], df["time"].iloc[-1])

    # --- 2. Split raw data by date first ---
    logger.info("Splitting: train <2023 | val 2023-2024 | test >=2024...")
    t = df["time"]
    df_train = df[t < "2023-01-01"].reset_index(drop=True)
    df_val   = df[(t >= "2023-01-01") & (t < "2024-01-01")].reset_index(drop=True)
    df_test  = df[t >= "2024-01-01"].reset_index(drop=True)
    logger.info("  Rows: train=%d, val=%d, test=%d", len(df_train), len(df_val), len(df_test))

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

    # --- 4. Generate labels ---
    logger.info("Generating swing labels (96-bar horizon)...")
    lbl_train = generate_swing_labels(df_train)
    lbl_val   = generate_swing_labels(df_val)
    lbl_test  = generate_swing_labels(df_test)

    feat_train, lbl_train = filter_labeled_data(feat_train, lbl_train)
    feat_val,   lbl_val   = filter_labeled_data(feat_val,   lbl_val)
    feat_test,  lbl_test  = filter_labeled_data(feat_test,  lbl_test)

    for name, lbl in [("train", lbl_train), ("val", lbl_val), ("test", lbl_test)]:
        wr = 100 * (lbl == 1).mean() if len(lbl) > 0 else 0
        logger.info("  %s: %d samples (%.1f%% win rate)", name, len(lbl), wr)

    if len(lbl_train) < 200:
        logger.error("Training set too small (%d samples).", len(lbl_train))
        sys.exit(1)

    # --- 5. Prepare matrices ---
    logger.info("Preparing sequential and tabular matrices...")
    m_train = prepare_training_matrices(feat_train, lbl_train, SEQUENCE_LENGTH)
    m_val   = prepare_training_matrices(feat_val,   lbl_val,   SEQUENCE_LENGTH)
    m_test  = prepare_training_matrices(feat_test,  lbl_test,  SEQUENCE_LENGTH)

    # --- 6. Train BiLSTM ---
    logger.info("")
    logger.info("--- Training Swing BiLSTM ---")
    bilstm = train_swing_bilstm(
        m_train["X_seq"], m_train["y"],
        m_val["X_seq"],   m_val["y"],
        epochs=100,
        batch_size=32,
    )

    # --- 7. Train XGBoost ---
    logger.info("")
    logger.info("--- Training Swing XGBoost ---")
    xgb = train_swing_xgb(
        m_train["X_tab"], m_train["y"],
        m_val["X_tab"],   m_val["y"],
        n_trials=50,
    )

    # --- 8. Evaluate ---
    # NOTE: SwingXGB disabled — AUC ~0.50 (random, best_iteration=0) due to regime shift.
    # Swing uses BiLSTM-only scoring for now. Re-enable XGB after sufficient live data
    # accumulates (6+ months of retraining on current regime data).
    logger.info("")
    logger.info("--- Evaluation on Test Set ---")
    bilstm_preds = bilstm._model.predict(m_test["X_seq"], batch_size=256, verbose=0).squeeze()

    report = evaluate_model(m_test["y"], bilstm_preds, "Swing BiLSTM", "test")
    logger.info("\n%s", report.summary())

    logger.info("")
    logger.info("=" * 60)
    logger.info("SWING TRAINING COMPLETE")
    logger.info("  BiLSTM weights: %s", SWING_BILSTM_PATH)
    logger.info("  XGBoost model:  %s", SWING_XGB_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
