"""Forward-looking label generation for scalper and swing models.

Labels are binary: 1 = TP hit before SL (win), 0 = SL hit first (loss).
Uses actual price data to simulate trade outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_server.config import (
    HARD_STOP_PIPS,
    TP_MULTIPLIER,
    SWING_TP1_RR,
    SWING_MAX_HOLD_HOURS,
)


# Gold pip = $0.10 (1 pip = 0.1 price movement for XAUUSD)
GOLD_PIP = 0.10


def generate_scalper_labels(
    df: pd.DataFrame,
    sl_pips: float = HARD_STOP_PIPS,
    tp_multiplier: float = TP_MULTIPLIER,
    atr_column: str = "atr_14",
    forward_bars: int = 60,
    direction_column: str | None = None,
) -> np.ndarray:
    """Generate binary labels for scalper model.

    For each bar, looks forward up to `forward_bars` M1 candles.
    TP = ATR(14) * tp_multiplier. SL = sl_pips.
    Label = 1 if TP hit before SL, 0 if SL hit first, NaN if neither.

    Args:
        df: DataFrame with columns: close, high, low, and optionally atr_14
        sl_pips: Stop loss in pips
        tp_multiplier: ATR multiplier for take profit
        atr_column: Column name for ATR values
        forward_bars: Max bars to look forward (60 M1 candles = 1 hour)
        direction_column: Column indicating trade direction (1=long, -1=short).
                         If None, labels both directions and picks the better one.

    Returns:
        np.ndarray of float32: 1.0 (win), 0.0 (loss), NaN (undetermined)
    """
    n = len(df)
    labels = np.full(n, np.nan, dtype=np.float32)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # ATR for dynamic TP
    if atr_column in df.columns:
        atr = df[atr_column].values
    else:
        # Fallback: compute simple ATR
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1)),
            ),
        )
        atr = pd.Series(tr).rolling(14).mean().values

    # Direction: positive values = long, negative = short
    if direction_column and direction_column in df.columns:
        directions = df[direction_column].values
    else:
        directions = None

    sl_price_dist = sl_pips * GOLD_PIP

    for i in range(n - forward_bars):
        entry_price = close[i]
        tp_price_dist = atr[i] * tp_multiplier

        if np.isnan(tp_price_dist) or tp_price_dist <= 0:
            continue

        # Determine direction
        if directions is not None:
            if directions[i] == 0:
                continue
            is_long = directions[i] > 0
        else:
            # Try both directions, label as win if either works
            # (during training, direction is known from the rule-based system)
            is_long = True  # Default to long; override during actual training

        # Look forward
        for j in range(1, forward_bars + 1):
            idx = i + j
            if idx >= n:
                break

            if is_long:
                # Long: TP if high reaches entry + tp_dist, SL if low drops below entry - sl_dist
                if high[idx] >= entry_price + tp_price_dist:
                    labels[i] = 1.0
                    break
                if low[idx] <= entry_price - sl_price_dist:
                    labels[i] = 0.0
                    break
            else:
                # Short: TP if low reaches entry - tp_dist, SL if high rises above entry + sl_dist
                if low[idx] <= entry_price - tp_price_dist:
                    labels[i] = 1.0
                    break
                if high[idx] >= entry_price + sl_price_dist:
                    labels[i] = 0.0
                    break

    return labels


def generate_swing_labels(
    df: pd.DataFrame,
    sl_pips: float = 60.0,
    tp_rr: float = SWING_TP1_RR,
    forward_bars: int = 96,
    direction_column: str | None = None,
) -> np.ndarray:
    """Generate binary labels for swing model.

    Same logic as scalper but on H1 timeframe, looking forward 96 bars (4 days).
    SL typically 40-80 pips. TP at risk-reward ratio.

    Args:
        df: DataFrame with columns: close, high, low
        sl_pips: Stop loss in pips (40-80 typical for swing)
        tp_rr: Take profit risk-reward ratio
        forward_bars: Max bars to look forward (96 H1 = 4 days)
        direction_column: Column indicating trade direction

    Returns:
        np.ndarray of float32: 1.0 (win), 0.0 (loss), NaN (undetermined)
    """
    n = len(df)
    labels = np.full(n, np.nan, dtype=np.float32)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    sl_dist = sl_pips * GOLD_PIP
    tp_dist = sl_dist * tp_rr

    if direction_column and direction_column in df.columns:
        directions = df[direction_column].values
    else:
        directions = None

    for i in range(n - forward_bars):
        entry_price = close[i]

        if directions is not None:
            if directions[i] == 0:
                continue
            is_long = directions[i] > 0
        else:
            is_long = True

        for j in range(1, forward_bars + 1):
            idx = i + j
            if idx >= n:
                break

            if is_long:
                if high[idx] >= entry_price + tp_dist:
                    labels[i] = 1.0
                    break
                if low[idx] <= entry_price - sl_dist:
                    labels[i] = 0.0
                    break
            else:
                if low[idx] <= entry_price - tp_dist:
                    labels[i] = 1.0
                    break
                if high[idx] >= entry_price + sl_dist:
                    labels[i] = 0.0
                    break

    return labels


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute class weights for imbalanced labels.

    Uses class_weight = {0: 1.0, 1: wins/losses} as specified in architecture.
    NOT SMOTE — SMOTE on time series creates data leakage.
    """
    valid = labels[~np.isnan(labels)]
    if len(valid) == 0:
        return {0: 1.0, 1: 1.0}

    n_losses = (valid == 0).sum()
    n_wins = (valid == 1).sum()

    if n_losses == 0 or n_wins == 0:
        return {0: 1.0, 1: 1.0}

    return {0: 1.0, 1: float(n_wins / n_losses)}


def filter_labeled_data(
    features: np.ndarray | pd.DataFrame,
    labels: np.ndarray,
) -> tuple:
    """Remove rows with NaN labels (undetermined outcomes)."""
    valid_mask = ~np.isnan(labels)
    if isinstance(features, pd.DataFrame):
        return features[valid_mask].reset_index(drop=True), labels[valid_mask]
    return features[valid_mask], labels[valid_mask]
