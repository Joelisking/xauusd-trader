"""Regime classifier training pipeline.

Trains a model to classify market regime: Trending / Ranging / Crisis.
Labels derived from ADX and VIX thresholds.

Usage:
    uv run python -m ai_server.training.train_regime
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

from ai_server.config import DATA_DIR, REGIME_CLF_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_regime_labels(
    adx: np.ndarray,
    vix: np.ndarray,
) -> np.ndarray:
    """Generate regime labels from ADX and VIX values.

    Classes:
        0 = Trending (ADX > 25)
        1 = Ranging (ADX < 20)
        2 = Crisis (VIX > 25)

    Crisis takes precedence over ADX-based classification.
    """
    labels = np.ones(len(adx), dtype=np.int32)  # Default: ranging

    labels[adx > 25] = 0  # Trending
    labels[adx < 20] = 1  # Ranging
    labels[vix > 25] = 2  # Crisis (overrides)

    return labels


def extract_regime_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Extract regime-relevant features from OHLCV data.

    Computes ATR, ADX, Bollinger width, volume ratio, and spread ratio
    directly from the raw data (no macro dependency).
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["tick_volume"].values.astype(float) if "tick_volume" in df.columns else np.ones(len(df))

    n = len(df)

    # ATR(14)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(14).mean().values

    # ATR ratio (short/long volatility)
    atr_5 = pd.Series(tr).rolling(5).mean().values
    atr_20 = pd.Series(tr).rolling(20).mean().values
    atr_ratio = np.where(atr_20 > 0, atr_5 / atr_20, 1.0)

    # ADX(14) — simplified
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(-low, prepend=-low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    smoothed_atr = pd.Series(tr).ewm(span=14).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=14).mean().values / np.maximum(smoothed_atr, 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=14).mean().values / np.maximum(smoothed_atr, 1e-10)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(span=14).mean().values

    # Bollinger width
    sma_20 = pd.Series(close).rolling(20).mean().values
    std_20 = pd.Series(close).rolling(20).std().values
    bb_width = np.where(sma_20 > 0, (2 * std_20) / sma_20, 0.0)

    # Volume ratio
    vol_sma = pd.Series(volume).rolling(20).mean().values
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 1.0)

    # Spread ratio (if available)
    if "spread" in df.columns:
        spread = df["spread"].values.astype(float)
        spread_sma = pd.Series(spread).rolling(20).mean().values
        spread_ratio = np.where(spread_sma > 0, spread / spread_sma, 1.0)
    else:
        spread_ratio = np.ones(n)

    # Range expansion: (high - low) / close
    range_pct = (high - low) / np.maximum(close, 1.0)

    # Close rate of change
    close_roc_5 = np.zeros(n)
    close_roc_5[5:] = (close[5:] - close[:-5]) / np.maximum(close[:-5], 1.0)

    feature_names = [
        "atr_14", "atr_ratio", "adx", "plus_di", "minus_di",
        "bb_width", "vol_ratio", "spread_ratio", "range_pct", "close_roc_5",
    ]

    X = np.column_stack([
        atr_14, atr_ratio, adx, plus_di, minus_di,
        bb_width, vol_ratio, spread_ratio, range_pct, close_roc_5,
    ])

    return X, feature_names


def main() -> None:
    """End-to-end regime classifier training."""
    from data_pipeline.feature_pipeline import load_parquet

    logger.info("=" * 60)
    logger.info("REGIME CLASSIFIER TRAINING")
    logger.info("=" * 60)

    # --- 1. Load H1 data (regime is evaluated on H1/H4 timeframes) ---
    h1_path = DATA_DIR / "XAUUSD_H1.parquet"
    if not h1_path.exists():
        logger.error("H1 data not found at %s", h1_path)
        sys.exit(1)

    logger.info("Loading H1 data...")
    df = load_parquet("H1")
    logger.info("Loaded %d rows", len(df))

    # --- 2. Extract regime features ---
    logger.info("Computing regime features...")
    X, feature_names = extract_regime_features(df)
    logger.info("Features: %s", feature_names)

    # --- 3. Generate labels from ADX + VIX ---
    # ADX is feature index 2; VIX we approximate from volatility
    # (real VIX comes from macro data, but for labeling we use ADX-based regime)
    adx = X[:, 2]  # ADX column
    # Approximate crisis from extreme ATR ratio (no VIX in price data)
    atr_ratio = X[:, 1]
    # Use high ATR ratio as crisis proxy when VIX unavailable
    pseudo_vix = np.where(atr_ratio > 2.0, 30.0, 15.0)

    labels = generate_regime_labels(adx, pseudo_vix)

    # Drop NaN rows (from rolling calculations)
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    labels = labels[valid_mask]

    # Check class distribution
    for cls, name in [(0, "trending"), (1, "ranging"), (2, "crisis")]:
        count = (labels == cls).sum()
        logger.info("  Class %d (%s): %d samples (%.1f%%)",
                     cls, name, count, 100 * count / len(labels))

    # --- 4. Time-based split (70/15/15 by time) ---
    n_valid = len(labels)
    train_end = int(n_valid * 0.70)
    val_end = int(n_valid * 0.85)

    X_train, y_train = X[:train_end], labels[:train_end]
    X_val, y_val = X[train_end:val_end], labels[train_end:val_end]
    X_test, y_test = X[val_end:], labels[val_end:]

    logger.info("Split: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(y_test))

    if len(y_train) < 100:
        logger.error("Training set too small.")
        sys.exit(1)

    # --- 5. Train ---
    logger.info("")
    logger.info("--- Training Regime Classifier ---")
    from ai_server.models.regime_classifier import RegimeClassifier

    model = RegimeClassifier()
    model.build_model(feature_count=X_train.shape[1])
    model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=50,
        batch_size=64,
        save_path=str(REGIME_CLF_PATH),
    )

    # --- 6. Evaluate ---
    logger.info("")
    logger.info("--- Evaluation on Test Set ---")
    from sklearn.metrics import classification_report
    import tensorflow as tf

    test_preds = model._model.predict(X_test.astype(np.float32), verbose=0)
    test_pred_classes = np.argmax(test_preds, axis=1)
    report = classification_report(
        y_test, test_pred_classes,
        target_names=["trending", "ranging", "crisis"],
    )
    logger.info("\n%s", report)

    # Save
    model.save(str(REGIME_CLF_PATH))

    logger.info("")
    logger.info("=" * 60)
    logger.info("REGIME TRAINING COMPLETE")
    logger.info("  Model saved: %s", REGIME_CLF_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
