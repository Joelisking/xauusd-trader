"""Derived & Temporal features — 32 features.

Temporal (hour, day), round-number distances, spread ratios,
session identifiers, drawdown, and other derived metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_server.config import (
    SESSION_ASIAN_START,
    SESSION_ASIAN_END,
    SESSION_LONDON_OPEN_START,
    SESSION_LONDON_OPEN_END,
    SESSION_LONDON_START,
    SESSION_LONDON_END,
    SESSION_OVERLAP_START,
    SESSION_OVERLAP_END,
    SESSION_NY_START,
    SESSION_NY_END,
)


# Gold round numbers (key psychological levels)
ROUND_LEVELS = list(range(2000, 4100, 100))  # $2000, $2100, ..., $4000


def _get_session_id(hour: int) -> int:
    """Map UTC hour to session number (0-5)."""
    if SESSION_ASIAN_START <= hour < SESSION_ASIAN_END:
        return 0  # Asian
    elif SESSION_LONDON_OPEN_START <= hour < SESSION_LONDON_OPEN_END:
        return 1  # London Open
    elif SESSION_LONDON_START <= hour < SESSION_LONDON_END:
        return 2  # London
    elif SESSION_OVERLAP_START <= hour < SESSION_OVERLAP_END:
        return 3  # London-NY Overlap
    elif SESSION_NY_START <= hour < SESSION_NY_END:
        return 4  # NY
    else:
        return 5  # NY Close / Asian


def compute_derived_features(
    df: pd.DataFrame,
    account_drawdown: float = 0.0,
    days_to_news: float = 30.0,
    session_number: int = 1,
) -> pd.DataFrame:
    """Compute all 32 derived & temporal features.

    Input: DataFrame with columns: time (or DatetimeIndex), open, high, low, close,
           tick_volume (or volume), spread (optional).
    Output: DataFrame with 32 feature columns.
    """
    features = pd.DataFrame(index=df.index)

    # Ensure we have a time column or DatetimeIndex
    if "time" in df.columns:
        times = pd.to_datetime(df["time"])
    elif isinstance(df.index, pd.DatetimeIndex):
        times = pd.Series(df.index)
    else:
        # Fallback: create synthetic timestamps
        times = pd.Series(pd.date_range("2025-01-01", periods=len(df), freq="min"))

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Use .dt accessor for Series
    hour = times.dt.hour
    dow = times.dt.dayofweek

    # --- Temporal (6) ---
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24).values
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24).values
    features["day_sin"] = np.sin(2 * np.pi * dow / 5).values
    features["day_cos"] = np.cos(2 * np.pi * dow / 5).values
    features["hour_of_day"] = hour.values.astype(np.float32)
    features["day_of_week"] = dow.values.astype(np.float32)

    # --- Round number distance (3) ---
    price_arr = close.values
    nearest_round = np.array([
        min(ROUND_LEVELS, key=lambda lv: abs(lv - p)) for p in price_arr
    ], dtype=np.float32)
    features["dist_to_round_100"] = (price_arr - nearest_round) / price_arr
    features["dist_to_round_abs"] = np.abs(price_arr - nearest_round)

    # Distance from $50 sub-levels too
    round_50 = list(range(2050, 4050, 100))
    nearest_50 = np.array([
        min(round_50, key=lambda lv: abs(lv - p)) for p in price_arr
    ], dtype=np.float32)
    features["dist_to_round_50"] = np.abs(price_arr - nearest_50)

    # --- Spread features (3) ---
    if "spread" in df.columns:
        spread = df["spread"].astype(float)
    else:
        spread = pd.Series(np.ones(len(df)), index=df.index)
    spread_avg = spread.rolling(20).mean().replace(0, np.nan)
    features["spread_ratio"] = spread / spread_avg
    features["spread_raw"] = spread
    features["spread_zscore"] = (spread - spread_avg) / spread.rolling(20).std().replace(0, np.nan)

    # --- Volume features (3) ---
    vol = df.get("tick_volume", df.get("volume", pd.Series(np.ones(len(df)), index=df.index)))
    vol_avg = vol.rolling(20).mean().replace(0, np.nan)
    features["volume_ratio_20"] = vol / vol_avg
    features["volume_spike"] = (vol > 1.5 * vol_avg).astype(np.float32)
    features["volume_trend"] = vol.rolling(10).mean() / vol.rolling(50).mean().replace(0, np.nan)

    # --- Session features (4) ---
    session_ids = np.array([_get_session_id(h) for h in hour], dtype=np.float32)
    features["session_id"] = session_ids
    features["is_overlap"] = (session_ids == 3).astype(np.float32)
    features["is_london"] = ((session_ids >= 1) & (session_ids <= 3)).astype(np.float32)
    features["session_number"] = float(session_number)

    # --- Drawdown and risk (3) ---
    features["account_drawdown"] = float(account_drawdown)
    running_max = close.cummax()
    features["price_drawdown"] = (close - running_max) / running_max.replace(0, np.nan)
    features["drawdown_recovery"] = (close - close.cummin()) / (running_max - close.cummin()).replace(0, np.nan)

    # --- News proximity (2) ---
    features["days_to_news"] = float(days_to_news)
    features["news_proximity_score"] = max(0, 1.0 - days_to_news / 7.0)

    # --- Intraday position (4) ---
    daily_open = close.iloc[0] if len(close) > 0 else 0
    features["intraday_return"] = (close - daily_open) / max(daily_open, 1)
    features["intraday_range_pct"] = (high.rolling(60).max() - low.rolling(60).min()) / close
    features["close_vs_daily_high"] = (close - high.rolling(240).max()) / close
    features["close_vs_daily_low"] = (close - low.rolling(240).min()) / close

    # --- Momentum features (4) ---
    features["momentum_3bar"] = close.diff(3) / close.shift(3).replace(0, np.nan)
    features["momentum_10bar"] = close.diff(10) / close.shift(10).replace(0, np.nan)
    features["acceleration"] = close.diff().diff()
    features["trend_consistency"] = (
        (close.diff() > 0).rolling(10).mean() - 0.5
    ) * 2  # -1 to 1

    # Fill NaN from warmup period
    features = features.fillna(0)
    return features
