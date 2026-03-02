"""Feature Engine — orchestrates computation of all 127 features.

Combines price (75), derived (32), and macro (20) feature groups.
Used both for real-time serving (single row) and batch training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_server.config import FEATURE_COUNT, SEQUENCE_LENGTH, XGB_FEATURE_COUNT
from ai_server.features.price_features import compute_price_features
from ai_server.features.derived_features import compute_derived_features
from ai_server.features.macro_features import MacroContext, compute_macro_features


# Feature group sizes (must sum to FEATURE_COUNT=127)
PRICE_FEATURE_COUNT = 75
DERIVED_FEATURE_COUNT = 32
MACRO_FEATURE_COUNT = 20

# XGBoost uses a subset: last row of most important features
XGB_FEATURE_NAMES: list[str] = []  # Populated after first compute


class FeatureEngine:
    """Orchestrates computation of all 127 features.

    Usage (real-time serving):
        engine = FeatureEngine()
        all_features = engine.compute(df_ohlcv, macro=macro_ctx)
        sequential = engine.get_sequential_features(all_features)  # (200, ~100)
        tabular = engine.get_tabular_features(all_features)        # (60,)

    Usage (batch training):
        engine = FeatureEngine()
        all_features = engine.compute(df_full_history, macro=macro_ctx)
        X_seq, X_tab = engine.prepare_training_data(all_features)
    """

    def __init__(self) -> None:
        self._feature_names: list[str] = []
        self._xgb_feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def xgb_feature_names(self) -> list[str]:
        return self._xgb_feature_names

    def compute(
        self,
        df: pd.DataFrame,
        macro: MacroContext | None = None,
        account_drawdown: float = 0.0,
        days_to_news: float = 30.0,
        session_number: int = 1,
    ) -> pd.DataFrame:
        """Compute all 127 features from OHLCV data + macro context.

        Args:
            df: DataFrame with columns: time, open, high, low, close, tick_volume, spread
            macro: Current macro data context
            account_drawdown: Current account drawdown percentage
            days_to_news: Days until next major news event
            session_number: Current session number (1 or 2)

        Returns:
            DataFrame with 127 feature columns, same length as input.
        """
        n = len(df)

        # Compute each feature group
        price_feats = compute_price_features(df)
        derived_feats = compute_derived_features(
            df,
            account_drawdown=account_drawdown,
            days_to_news=days_to_news,
            session_number=session_number,
        )
        macro_feats = compute_macro_features(n, macro=macro)
        macro_feats.index = df.index

        # Combine all features
        all_features = pd.concat([price_feats, derived_feats, macro_feats], axis=1)

        # Ensure exactly FEATURE_COUNT features
        current_count = len(all_features.columns)
        if current_count > FEATURE_COUNT:
            # Trim to 127 by dropping least important extras
            all_features = all_features.iloc[:, :FEATURE_COUNT]
        elif current_count < FEATURE_COUNT:
            # Pad with zeros if we're short
            for i in range(current_count, FEATURE_COUNT):
                all_features[f"pad_{i}"] = 0.0

        self._feature_names = list(all_features.columns)

        # Select XGBoost features (most important tabular features)
        self._xgb_feature_names = self._select_xgb_features(all_features)

        return all_features

    def _select_xgb_features(self, features: pd.DataFrame) -> list[str]:
        """Select the top 60 features for XGBoost (tabular model).

        Prioritizes: indicators, ratios, and macro over raw pattern encodings.
        """
        all_cols = list(features.columns)

        # Priority features for tabular model
        priority_prefixes = [
            "ema_", "atr_", "rsi_", "macd_", "bb_", "adx_", "stoch_",
            "plus_di", "minus_di", "vwap_", "volume_ratio", "spread_",
            "dxy_", "real_yield", "vix_", "news_", "session_",
            "momentum_", "return_", "hour_", "day_", "market_structure",
            "volatility_", "range_", "close_position", "body_pct",
            "macro_", "risk_off", "gold_dxy",
        ]

        selected: list[str] = []
        for prefix in priority_prefixes:
            for col in all_cols:
                if col.startswith(prefix) and col not in selected:
                    selected.append(col)
                    if len(selected) >= XGB_FEATURE_COUNT:
                        return selected

        # Fill remaining with any unused columns
        for col in all_cols:
            if col not in selected:
                selected.append(col)
                if len(selected) >= XGB_FEATURE_COUNT:
                    break

        return selected[:XGB_FEATURE_COUNT]

    def get_sequential_features(
        self,
        features: pd.DataFrame,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> np.ndarray:
        """Extract last `sequence_length` rows as 3D array for BiLSTM.

        Returns: shape (1, sequence_length, num_features)
        """
        n = len(features)
        if n < sequence_length:
            # Pad with zeros at the beginning
            padding = pd.DataFrame(
                np.zeros((sequence_length - n, len(features.columns))),
                columns=features.columns,
            )
            features = pd.concat([padding, features], ignore_index=True)

        arr = features.iloc[-sequence_length:].values.astype(np.float32)
        return arr.reshape(1, sequence_length, -1)

    def get_tabular_features(self, features: pd.DataFrame) -> np.ndarray:
        """Extract last row of XGBoost features as 2D array.

        Returns: shape (1, XGB_FEATURE_COUNT)
        """
        xgb_cols = self._xgb_feature_names or list(features.columns[:XGB_FEATURE_COUNT])
        # Use only columns that exist
        valid_cols = [c for c in xgb_cols if c in features.columns]
        row = features[valid_cols].iloc[-1:].values.astype(np.float32)
        # Pad if needed
        if row.shape[1] < XGB_FEATURE_COUNT:
            padding = np.zeros((1, XGB_FEATURE_COUNT - row.shape[1]), dtype=np.float32)
            row = np.hstack([row, padding])
        return row

    def prepare_training_data(
        self,
        features: pd.DataFrame,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data: sequential + tabular arrays.

        Returns:
            X_seq: shape (num_samples, sequence_length, num_features)
            X_tab: shape (num_samples, XGB_FEATURE_COUNT)
        """
        n = len(features)
        num_samples = n - sequence_length + 1
        if num_samples <= 0:
            return np.empty((0, sequence_length, len(features.columns))), np.empty((0, XGB_FEATURE_COUNT))

        all_values = features.values.astype(np.float32)

        # Sequential: sliding window
        X_seq = np.zeros((num_samples, sequence_length, all_values.shape[1]), dtype=np.float32)
        for i in range(num_samples):
            X_seq[i] = all_values[i:i + sequence_length]

        # Tabular: last row of each window
        xgb_cols = self._xgb_feature_names or list(features.columns[:XGB_FEATURE_COUNT])
        valid_indices = [features.columns.get_loc(c) for c in xgb_cols if c in features.columns]
        X_tab = np.zeros((num_samples, XGB_FEATURE_COUNT), dtype=np.float32)
        for i in range(num_samples):
            row = all_values[i + sequence_length - 1, valid_indices]
            X_tab[i, :len(row)] = row

        return X_seq, X_tab

    def normalize_features(
        self,
        features: pd.DataFrame,
        stats: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
        """Z-score normalize features. Returns normalized df and stats dict.

        Args:
            features: Input feature DataFrame
            stats: Optional pre-computed (mean, std) per column. If None, computed from data.

        Returns:
            (normalized_df, stats_dict) where stats_dict maps col -> (mean, std)
        """
        if stats is None:
            stats = {}
            for col in features.columns:
                mean = features[col].mean()
                std = features[col].std()
                if std == 0:
                    std = 1.0
                stats[col] = (float(mean), float(std))

        normalized = features.copy()
        for col in features.columns:
            if col in stats:
                mean, std = stats[col]
                normalized[col] = (features[col] - mean) / max(std, 1e-8)

        return normalized, stats
