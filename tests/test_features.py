"""Tests for feature engineering pipeline — Phase 7."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_server.config import FEATURE_COUNT, SEQUENCE_LENGTH, XGB_FEATURE_COUNT
from ai_server.features.feature_engine import FeatureEngine
from ai_server.features.price_features import (
    calc_atr,
    calc_rsi,
    calc_macd,
    calc_bollinger,
    calc_adx,
    calc_stochastic,
    compute_price_features,
    encode_candle_patterns,
    encode_market_structure,
)
from ai_server.features.derived_features import compute_derived_features
from ai_server.features.macro_features import MacroContext, compute_macro_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data resembling gold prices."""
    rng = np.random.RandomState(seed)
    base_price = 3000.0
    returns = rng.normal(0, 0.001, n)
    prices = base_price * np.exp(np.cumsum(returns))

    base_time = datetime(2025, 1, 2, 8, 0, tzinfo=timezone.utc)
    data = {
        "time": [base_time + timedelta(minutes=i) for i in range(n)],
        "open": prices + rng.uniform(-2, 2, n),
        "high": prices + rng.uniform(1, 8, n),
        "low": prices - rng.uniform(1, 8, n),
        "close": prices,
        "tick_volume": rng.randint(100, 5000, n).astype(float),
        "spread": rng.randint(1, 5, n).astype(float),
        "real_volume": np.zeros(n),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Price feature tests
# ---------------------------------------------------------------------------


class TestPriceIndicators:
    def test_calc_atr(self):
        df = _make_ohlcv(200)
        atr = calc_atr(df, 14)
        assert len(atr) == 200
        assert atr.iloc[20:].isna().sum() == 0
        assert (atr.iloc[20:] > 0).all()

    def test_calc_rsi(self):
        df = _make_ohlcv(200)
        rsi = calc_rsi(df["close"], 14)
        assert len(rsi) == 200
        valid = rsi.iloc[20:].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_calc_macd(self):
        df = _make_ohlcv(200)
        line, signal, hist = calc_macd(df["close"])
        assert len(line) == 200
        assert len(signal) == 200
        assert len(hist) == 200

    def test_calc_bollinger(self):
        df = _make_ohlcv(200)
        width, pct_b = calc_bollinger(df["close"])
        assert len(width) == 200
        valid_b = pct_b.iloc[30:].dropna()
        # %B should mostly be between -1 and 2 (can exceed during strong moves)
        assert valid_b.mean() > -1 and valid_b.mean() < 2

    def test_calc_adx(self):
        df = _make_ohlcv(200)
        adx, plus_di, minus_di = calc_adx(df)
        assert len(adx) == 200
        valid_adx = adx.iloc[30:].dropna()
        assert (valid_adx >= 0).all()

    def test_calc_stochastic(self):
        df = _make_ohlcv(200)
        k, d = calc_stochastic(df)
        valid_k = k.iloc[10:].dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()


class TestCandlePatterns:
    def test_encode_returns_14_columns(self):
        df = _make_ohlcv(100)
        patterns = encode_candle_patterns(df)
        assert patterns.shape[1] == 14

    def test_pattern_values_binary(self):
        df = _make_ohlcv(100)
        patterns = encode_candle_patterns(df)
        unique_vals = set()
        for col in patterns.columns:
            unique_vals.update(patterns[col].unique())
        assert unique_vals.issubset({0.0, 1.0})


class TestMarketStructure:
    def test_encode_returns_series(self):
        df = _make_ohlcv(200)
        ms = encode_market_structure(df)
        assert len(ms) == 200
        # Values should be 0,1,2,3 (HH, HL, LH, LL)
        assert set(ms.dropna().unique()).issubset({0.0, 1.0, 2.0, 3.0})


class TestPriceFeatures:
    def test_compute_price_features_shape(self):
        df = _make_ohlcv(300)
        feats = compute_price_features(df)
        assert len(feats) == 300
        # Should produce 75 features
        assert feats.shape[1] == 75

    def test_no_nan_after_warmup(self):
        df = _make_ohlcv(500)
        feats = compute_price_features(df)
        # After fillna(0), no NaN should remain
        assert feats.isna().sum().sum() == 0

    def test_no_inf(self):
        df = _make_ohlcv(500)
        feats = compute_price_features(df)
        inf_count = np.isinf(feats.values).sum()
        assert inf_count == 0


# ---------------------------------------------------------------------------
# Derived feature tests
# ---------------------------------------------------------------------------


class TestDerivedFeatures:
    def test_compute_shape(self):
        df = _make_ohlcv(300)
        feats = compute_derived_features(df)
        assert len(feats) == 300
        assert feats.shape[1] >= 28  # Allow some flexibility

    def test_temporal_encoding(self):
        df = _make_ohlcv(100)
        feats = compute_derived_features(df)
        # Sine/cosine should be in [-1, 1]
        assert feats["hour_sin"].between(-1, 1).all()
        assert feats["hour_cos"].between(-1, 1).all()
        assert feats["day_sin"].between(-1, 1).all()
        assert feats["day_cos"].between(-1, 1).all()

    def test_no_nan(self):
        df = _make_ohlcv(300)
        feats = compute_derived_features(df)
        assert feats.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Macro feature tests
# ---------------------------------------------------------------------------


class TestMacroFeatures:
    def test_default_context(self):
        feats = compute_macro_features(10)
        assert len(feats) == 10
        assert feats.shape[1] == 20

    def test_with_context(self):
        ctx = MacroContext(
            dxy_price=104.5,
            dxy_ema_50=103.0,
            dxy_momentum_roc=0.5,
            dxy_direction="UP",
            real_yield=2.1,
            real_yield_direction="UP",
            vix=22.0,
            vix_5d_roc=-3.0,
        )
        feats = compute_macro_features(5, macro=ctx)
        assert feats["dxy_direction"].iloc[0] == 1.0  # UP
        assert feats["real_yield_direction"].iloc[0] == 1.0
        assert feats["vix_level"].iloc[0] == pytest.approx(0.22, abs=0.01)

    def test_no_nan(self):
        feats = compute_macro_features(100)
        assert feats.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Feature Engine integration tests
# ---------------------------------------------------------------------------


class TestFeatureEngine:
    def test_compute_127_features(self):
        df = _make_ohlcv(500)
        engine = FeatureEngine()
        feats = engine.compute(df)
        assert feats.shape[1] == FEATURE_COUNT

    def test_feature_names_populated(self):
        df = _make_ohlcv(300)
        engine = FeatureEngine()
        engine.compute(df)
        assert len(engine.feature_names) == FEATURE_COUNT

    def test_sequential_features_shape(self):
        df = _make_ohlcv(500)
        engine = FeatureEngine()
        feats = engine.compute(df)
        seq = engine.get_sequential_features(feats)
        assert seq.shape == (1, SEQUENCE_LENGTH, FEATURE_COUNT)
        assert seq.dtype == np.float32

    def test_tabular_features_shape(self):
        df = _make_ohlcv(500)
        engine = FeatureEngine()
        feats = engine.compute(df)
        tab = engine.get_tabular_features(feats)
        assert tab.shape == (1, XGB_FEATURE_COUNT)
        assert tab.dtype == np.float32

    def test_no_nan_in_output(self):
        df = _make_ohlcv(500)
        engine = FeatureEngine()
        feats = engine.compute(df)
        assert feats.isna().sum().sum() == 0

    def test_no_inf_in_output(self):
        df = _make_ohlcv(500)
        engine = FeatureEngine()
        feats = engine.compute(df)
        inf_count = np.isinf(feats.values).sum()
        assert inf_count == 0

    def test_sequential_padding_short_data(self):
        """If data < SEQUENCE_LENGTH, should pad with zeros."""
        df = _make_ohlcv(50)
        engine = FeatureEngine()
        feats = engine.compute(df)
        seq = engine.get_sequential_features(feats)
        assert seq.shape == (1, SEQUENCE_LENGTH, FEATURE_COUNT)
        # First rows should be zero (padding)
        assert seq[0, 0, :].sum() == 0

    def test_prepare_training_data(self):
        df = _make_ohlcv(300)
        engine = FeatureEngine()
        feats = engine.compute(df)
        X_seq, X_tab = engine.prepare_training_data(feats)
        expected_samples = 300 - SEQUENCE_LENGTH + 1
        assert X_seq.shape[0] == expected_samples
        assert X_seq.shape[1] == SEQUENCE_LENGTH
        assert X_tab.shape[0] == expected_samples
        assert X_tab.shape[1] == XGB_FEATURE_COUNT

    def test_normalize_features(self):
        df = _make_ohlcv(300)
        engine = FeatureEngine()
        feats = engine.compute(df)
        normalized, stats = engine.normalize_features(feats)
        assert len(stats) == FEATURE_COUNT
        # Normalized features should have ~0 mean for most columns
        means = normalized.mean()
        assert abs(means.mean()) < 0.5

    def test_normalize_with_precomputed_stats(self):
        df = _make_ohlcv(300)
        engine = FeatureEngine()
        feats = engine.compute(df)
        _, stats = engine.normalize_features(feats)
        # Re-normalize with same stats
        normalized2, _ = engine.normalize_features(feats, stats=stats)
        assert normalized2.shape == feats.shape

    def test_xgb_feature_selection(self):
        df = _make_ohlcv(300)
        engine = FeatureEngine()
        engine.compute(df)
        assert len(engine.xgb_feature_names) == XGB_FEATURE_COUNT

    def test_macro_context_integration(self):
        df = _make_ohlcv(300)
        engine = FeatureEngine()
        ctx = MacroContext(
            dxy_price=104.5,
            dxy_direction="UP",
            vix=25.0,
            vix_5d_roc=5.0,
        )
        feats = engine.compute(df, macro=ctx)
        assert feats.shape[1] == FEATURE_COUNT
        # DXY direction should be encoded
        assert "dxy_direction" in feats.columns
