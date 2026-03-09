"""NFP direction model training pipeline.

Simple XGBoost model for post-NFP gold direction prediction.
Input: previous NFP surprise, gold price level, DXY level, preceding 5-day gold trend.

Since NFP events are rare (~12/year), this model trains on synthetic/historical
NFP-like scenarios derived from price action around high-impact economic releases.

Usage:
    uv run python -m ai_server.training.train_nfp
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_server.config import DATA_DIR, NFP_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


NFP_FEATURE_NAMES = [
    "nfp_surprise",      # Actual - Forecast (thousands)
    "gold_price_level",  # Current gold price
    "dxy_level",         # Current DXY level (approximated)
    "gold_5d_trend",     # 5-day gold return %
]


def prepare_nfp_data(
    nfp_events: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare NFP training data from historical event records.

    Each event dict should have:
        - nfp_surprise: float (actual - forecast, thousands)
        - gold_price: float
        - dxy_level: float
        - gold_5d_return: float (%)
        - gold_direction_after: int (0=down, 1=up)

    Returns:
        (X, y) arrays
    """
    X = []
    y = []

    for event in nfp_events:
        X.append([
            event.get("nfp_surprise", 0),
            event.get("gold_price", 3000),
            event.get("dxy_level", 104),
            event.get("gold_5d_return", 0),
        ])
        y.append(event.get("gold_direction_after", 0))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def generate_nfp_proxy_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate NFP-like training data from high-volatility events in price history.

    NFP releases are monthly (12/year), so over 10 years we have ~120 real events.
    To augment, we identify all "NFP-like" moments: large sudden moves in gold
    (ATR(1) > 3x ATR(20) on H1) that mimic the kind of volatility NFP produces.

    Features:
        - nfp_surprise: approximated from the size/direction of the move
        - gold_price_level: close price at event
        - dxy_level: approximated as inverse correlation proxy
        - gold_5d_trend: 5-day return before event

    Label: 1 if gold closes higher 4 hours after event, 0 if lower.
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(df)

    # Compute ATR
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]
    atr_1 = tr  # single-bar true range
    atr_20 = pd.Series(tr).rolling(20).mean().values

    # 5-day return (on H1 = 120 bars)
    lookback = 120
    forward = 4  # 4 H1 bars = 4 hours
    gold_5d_ret = np.zeros(n)
    gold_5d_ret[lookback:] = (close[lookback:] - close[:-lookback]) / close[:-lookback] * 100

    X_list = []
    y_list = []

    for i in range(lookback, n - forward):
        if atr_20[i] <= 0 or np.isnan(atr_20[i]):
            continue

        ratio = atr_1[i] / atr_20[i]
        if ratio < 2.5:
            continue  # Not volatile enough to be NFP-like

        # Direction and size of the move = NFP surprise proxy
        move = close[i] - close[i - 1]
        surprise_proxy = move / atr_20[i]  # Normalized surprise

        # Gold price level
        price_level = close[i]

        # DXY proxy: inverse of gold trend (simplified)
        dxy_proxy = 104.0 - gold_5d_ret[i] * 0.5  # Rough inverse

        # 5-day trend
        trend_5d = gold_5d_ret[i]

        # Label: does gold go up in next 4 hours?
        future_close = close[i + forward]
        label = 1.0 if future_close > close[i] else 0.0

        X_list.append([surprise_proxy, price_level, dxy_proxy, trend_5d])
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def main() -> None:
    """End-to-end NFP model training."""
    from data_pipeline.feature_pipeline import load_parquet

    logger.info("=" * 60)
    logger.info("NFP DIRECTION MODEL TRAINING")
    logger.info("=" * 60)

    # --- 1. Load H1 data ---
    h1_path = DATA_DIR / "XAUUSD_H1.parquet"
    if not h1_path.exists():
        logger.error("H1 data not found at %s", h1_path)
        sys.exit(1)

    logger.info("Loading H1 data...")
    df = load_parquet("H1")
    logger.info("Loaded %d rows", len(df))

    # --- 2. Generate NFP-proxy training data ---
    logger.info("Generating NFP-proxy training data from high-volatility events...")
    X, y = generate_nfp_proxy_data(df)
    logger.info("Found %d NFP-like events (%.0f%% up)", len(y), 100 * y.mean() if len(y) > 0 else 0)

    if len(y) < 50:
        logger.error("Too few events (%d). Need at least 50.", len(y))
        sys.exit(1)

    # --- 3. Time-based split (80/10/10) ---
    n = len(y)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info("Split: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(y_test))

    # --- 4. Train ---
    logger.info("")
    logger.info("--- Training NFP Direction Model ---")
    from ai_server.models.nfp_model import NFPDirectionModel

    model = NFPDirectionModel()
    model.train(X_train, y_train, X_val=X_val, y_val=y_val,
                save_path=str(NFP_MODEL_PATH))

    # --- 5. Evaluate ---
    logger.info("")
    logger.info("--- Evaluation on Test Set ---")

    correct = 0
    for i in range(len(X_test)):
        direction, confidence = model.predict_direction(X_test[i])
        pred_label = 1.0 if direction == "up" else 0.0
        if pred_label == y_test[i]:
            correct += 1

    accuracy = 100 * correct / max(len(X_test), 1)
    logger.info("Test accuracy: %.1f%% (%d/%d)", accuracy, correct, len(X_test))

    # Save
    model.save(str(NFP_MODEL_PATH))

    logger.info("")
    logger.info("=" * 60)
    logger.info("NFP TRAINING COMPLETE")
    logger.info("  Model saved: %s", NFP_MODEL_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
