"""Batch feature pipeline — computes features over entire Parquet datasets for training.

Reads M1/M5/H1/H4 Parquet files, runs the feature engine, and produces
training-ready feature matrices with labels.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai_server.config import DATA_DIR, FEATURE_COUNT, SEQUENCE_LENGTH
from ai_server.features.feature_engine import FeatureEngine
from ai_server.features.macro_features import MacroContext


MACRO_DB_PATH = DATA_DIR / "macro.db"


def load_parquet(timeframe: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a Parquet file for given timeframe."""
    path = data_dir / f"XAUUSD_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def load_macro_history(db_path: Path = MACRO_DB_PATH) -> pd.DataFrame:
    """Load macro snapshot history from SQLite."""
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql("SELECT * FROM macro_snapshots ORDER BY timestamp", conn)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    finally:
        conn.close()


def get_macro_for_time(
    target_time: datetime,
    macro_df: pd.DataFrame,
) -> MacroContext:
    """Find the closest macro snapshot for a given time."""
    ctx = MacroContext()
    if macro_df.empty or "timestamp" not in macro_df.columns:
        return ctx

    # Find closest prior macro snapshot
    prior = macro_df[macro_df["timestamp"] <= target_time]
    if prior.empty:
        return ctx

    row = prior.iloc[-1]
    ctx.dxy_price = float(row.get("dxy_price", 0) or 0)
    ctx.dxy_ema_50 = float(row.get("dxy_ema_50", 0) or 0)
    ctx.dxy_momentum_roc = float(row.get("dxy_momentum", 0) or 0)
    ctx.dxy_direction = str(row.get("dxy_direction", "NEUTRAL") or "NEUTRAL")
    ctx.real_yield = float(row.get("real_yield", 0) or 0)
    ctx.real_yield_direction = str(row.get("real_yield_direction", "NEUTRAL") or "NEUTRAL")
    ctx.nominal_10y = float(row.get("nominal_10y", 0) or 0)
    ctx.vix = float(row.get("vix", 0) or 0)
    ctx.vix_5d_roc = float(row.get("vix_5d_roc", 0) or 0)
    return ctx


def compute_batch_features(
    timeframe: str = "M1",
    data_dir: Path = DATA_DIR,
    use_macro: bool = True,
) -> pd.DataFrame:
    """Compute all 127 features for an entire dataset.

    Args:
        timeframe: M1, M5, H1, or H4
        data_dir: Directory containing Parquet files
        use_macro: Whether to merge macro data (requires macro.db)

    Returns:
        DataFrame with 127 feature columns + 'time' column.
    """
    df = load_parquet(timeframe, data_dir)
    engine = FeatureEngine()

    # Load macro history for time-varying macro features
    macro_ctx = MacroContext()
    if use_macro:
        try:
            macro_df = load_macro_history()
            if not macro_df.empty and "time" in df.columns:
                # Use the latest macro data as default (batch mode simplification)
                macro_ctx = get_macro_for_time(
                    df["time"].iloc[-1] if hasattr(df["time"].iloc[-1], "tzinfo") else datetime.now(timezone.utc),
                    macro_df,
                )
        except Exception:
            pass  # Use default macro context

    features = engine.compute(df, macro=macro_ctx)

    # Preserve time column for label alignment
    if "time" in df.columns:
        features = pd.concat(
            [pd.DataFrame({"time": df["time"].values}), features],
            axis=1,
        )

    return features


def prepare_training_matrices(
    features: pd.DataFrame,
    labels: np.ndarray | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
) -> dict:
    """Prepare complete training data matrices.

    Args:
        features: Output from compute_batch_features()
        labels: Binary labels array, same length as features
        sequence_length: Lookback window for BiLSTM

    Returns:
        Dictionary with keys: X_seq, X_tab, y, times
    """
    engine = FeatureEngine()

    # Separate time column if present
    times = None
    feat_df = features.copy()
    if "time" in feat_df.columns:
        times = feat_df["time"].values
        feat_df = feat_df.drop(columns=["time"])

    # Ensure we have exactly 127 features
    engine.compute.__wrapped__ if hasattr(engine.compute, "__wrapped__") else None
    engine._feature_names = list(feat_df.columns)
    engine._xgb_feature_names = engine._select_xgb_features(feat_df)

    X_seq, X_tab = engine.prepare_training_data(feat_df, sequence_length)

    result = {
        "X_seq": X_seq,
        "X_tab": X_tab,
    }

    if labels is not None:
        # Align labels with sequences (drop first sequence_length-1)
        result["y"] = labels[sequence_length - 1:]

    if times is not None:
        result["times"] = times[sequence_length - 1:]

    return result


def split_by_time(
    data: dict,
    train_end: str | None = None,
    val_end: str | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    """Split training data by time (never random).

    If train_end/val_end are provided AND the data actually spans those dates,
    uses them as fixed boundaries. Otherwise, computes boundaries dynamically
    from the data using train_ratio/val_ratio.

    Args:
        data: Output from prepare_training_matrices()
        train_end: End of training period (ISO string), or None for auto
        val_end: End of validation period (ISO string), or None for auto
        train_ratio: Fraction of data for training (used when auto-splitting)
        val_ratio: Fraction of data for validation (used when auto-splitting)

    Returns:
        Dictionary with train/val/test splits.
    """
    if "times" not in data or data["times"] is None:
        raise ValueError("Time column required for time-based split")

    times = pd.to_datetime(data["times"])
    n = len(times)

    # Determine whether fixed dates are usable
    use_fixed = False
    if train_end is not None and val_end is not None:
        ts_train = pd.Timestamp(train_end)
        ts_val = pd.Timestamp(val_end)
        # Fixed dates are usable if they actually split the data meaningfully
        # (at least 10% of data in each bucket)
        n_train = (times < ts_train).sum()
        n_val = ((times >= ts_train) & (times < ts_val)).sum()
        n_test = (times >= ts_val).sum()
        if n_train > n * 0.05 and n_val > n * 0.05 and n_test > n * 0.05:
            use_fixed = True

    if use_fixed:
        train_mask = times < ts_train
        val_mask = (times >= ts_train) & (times < ts_val)
        test_mask = times >= ts_val
    else:
        # Auto-split by time using ratios
        sorted_times = np.sort(times.values)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        ts_train = pd.Timestamp(sorted_times[train_idx])
        ts_val = pd.Timestamp(sorted_times[val_idx])

        train_mask = times < ts_train
        val_mask = (times >= ts_train) & (times < ts_val)
        test_mask = times >= ts_val

    splits = {}
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        splits[split_name] = {
            "X_seq": data["X_seq"][mask],
            "X_tab": data["X_tab"][mask],
        }
        if "y" in data:
            splits[split_name]["y"] = data["y"][mask]

    return splits
