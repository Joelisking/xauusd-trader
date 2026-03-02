"""Tests for AI training pipeline — Phase 9."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_server.training.label_generator import (
    generate_scalper_labels,
    generate_swing_labels,
    compute_class_weights,
    filter_labeled_data,
)
from ai_server.training.walk_forward import (
    generate_segments,
    WalkForwardSegment,
    WalkForwardResult,
    run_walk_forward,
)
from ai_server.training.evaluate import (
    evaluate_model,
    find_optimal_threshold,
    compare_models,
    EvaluationReport,
)
from ai_server.training.feature_selection import (
    FeatureImportanceResult,
    select_top_features,
    monitor_feature_drift,
)
from ai_server.training.train_regime import generate_regime_labels
from ai_server.training.train_nfp import prepare_nfp_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic price data with trending behavior."""
    rng = np.random.RandomState(seed)
    base = 3000.0
    # Create a trend + noise pattern
    trend = np.cumsum(rng.normal(0.05, 0.5, n))
    close = base + trend
    high = close + rng.uniform(0.5, 3.0, n)
    low = close - rng.uniform(0.5, 3.0, n)
    # Compute simple ATR
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    atr = pd.Series(tr).rolling(14).mean().values
    return pd.DataFrame({
        "close": close,
        "high": high,
        "low": low,
        "open": close + rng.normal(0, 0.3, n),
        "atr_14": atr,
    })


# ---------------------------------------------------------------------------
# Label generator tests
# ---------------------------------------------------------------------------


class TestLabelGenerator:
    def test_scalper_labels_shape(self):
        df = _make_price_data(500)
        labels = generate_scalper_labels(df, sl_pips=20, forward_bars=60)
        assert len(labels) == 500

    def test_scalper_labels_values(self):
        df = _make_price_data(500)
        labels = generate_scalper_labels(df, sl_pips=20, forward_bars=60)
        valid = labels[~np.isnan(labels)]
        assert len(valid) > 0
        assert set(np.unique(valid)).issubset({0.0, 1.0})

    def test_scalper_labels_last_bars_nan(self):
        """Last forward_bars rows should be NaN (can't look forward)."""
        df = _make_price_data(200)
        labels = generate_scalper_labels(df, forward_bars=60)
        assert np.isnan(labels[-1])

    def test_swing_labels_shape(self):
        df = _make_price_data(300)
        labels = generate_swing_labels(df, sl_pips=60, forward_bars=96)
        assert len(labels) == 300

    def test_swing_labels_values(self):
        df = _make_price_data(300)
        labels = generate_swing_labels(df, sl_pips=60, forward_bars=96)
        valid = labels[~np.isnan(labels)]
        assert len(valid) > 0
        assert set(np.unique(valid)).issubset({0.0, 1.0})

    def test_class_weights(self):
        labels = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float32)
        weights = compute_class_weights(labels)
        assert 0 in weights and 1 in weights
        assert weights[0] == 1.0

    def test_class_weights_all_nan(self):
        labels = np.full(10, np.nan)
        weights = compute_class_weights(labels)
        assert weights == {0: 1.0, 1: 1.0}

    def test_filter_labeled_data(self):
        features = np.random.randn(10, 5)
        labels = np.array([1, np.nan, 0, 1, np.nan, 0, 1, 0, np.nan, 1], dtype=np.float32)
        filt_feat, filt_labels = filter_labeled_data(features, labels)
        assert len(filt_feat) == 7
        assert len(filt_labels) == 7
        assert not np.any(np.isnan(filt_labels))


# ---------------------------------------------------------------------------
# Walk-forward tests
# ---------------------------------------------------------------------------


class TestWalkForward:
    def test_generate_segments(self):
        start = datetime(2018, 1, 1)
        end = datetime(2025, 1, 1)
        segments = generate_segments(start, end, n_segments=12)
        assert len(segments) > 0
        for seg in segments:
            assert seg["train_start"] < seg["train_end"]
            assert seg["test_start"] < seg["test_end"]
            # Gap between train end and test start
            assert seg["test_start"] > seg["train_end"]

    def test_segments_no_overlap(self):
        start = datetime(2018, 1, 1)
        end = datetime(2025, 1, 1)
        segments = generate_segments(start, end, n_segments=12)
        for seg in segments:
            # Train period must not overlap with test period
            assert seg["train_end"] <= seg["test_start"]

    def test_walk_forward_with_dummy_model(self):
        """Test walk-forward with a trivial random model."""
        n = 500
        rng = np.random.RandomState(42)
        start = datetime(2020, 1, 1)
        times = np.array([start + timedelta(days=i) for i in range(n)])
        X_seq = rng.randn(n, 10, 5).astype(np.float32)
        X_tab = rng.randn(n, 10).astype(np.float32)
        y = rng.randint(0, 2, n).astype(np.float32)

        def dummy_train(X_seq, X_tab, y):
            return None

        def dummy_predict(model, X_seq, X_tab):
            return np.random.rand(len(X_seq))

        result = run_walk_forward(
            times, X_seq, X_tab, y,
            train_fn=dummy_train,
            predict_fn=dummy_predict,
            n_segments=3,
            min_auc=0.3,
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.segments) >= 1

    def test_walk_forward_result_summary(self):
        result = WalkForwardResult(
            segments=[
                WalkForwardSegment(1, datetime(2020, 1, 1), datetime(2020, 9, 1),
                                   datetime(2020, 10, 1), datetime(2020, 12, 1),
                                   train_samples=100, test_samples=30, auc=0.72, accuracy=0.65, passed=True),
            ],
            mean_auc=0.72,
            min_auc=0.72,
            mean_accuracy=0.65,
            all_passed=True,
        )
        summary = result.summary()
        assert "0.72" in summary
        assert "PASS" in summary


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_evaluate_model_basic(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100).astype(float)
        y_proba = rng.uniform(0, 1, 100)
        report = evaluate_model(y_true, y_proba, "TestModel", "test")
        assert report.n_samples == 100
        assert 0 <= report.auc <= 1
        assert 0 <= report.accuracy <= 1

    def test_evaluate_perfect_model(self):
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        report = evaluate_model(y_true, y_proba, "Perfect", "test")
        assert report.auc == 1.0
        assert report.accuracy == 1.0

    def test_evaluate_model_summary(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 50).astype(float)
        y_proba = rng.uniform(0, 1, 50)
        report = evaluate_model(y_true, y_proba, "Test", "val")
        summary = report.summary()
        assert "Test" in summary
        assert "val" in summary

    def test_find_optimal_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        y_proba = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        threshold, score = find_optimal_threshold(y_true, y_proba)
        assert 0.3 <= threshold <= 0.8
        assert score > 0

    def test_compare_models(self):
        reports = [
            EvaluationReport("Model A", "test", auc=0.75, accuracy=0.70),
            EvaluationReport("Model B", "test", auc=0.80, accuracy=0.73),
        ]
        output = compare_models(reports)
        assert "Model A" in output
        assert "Model B" in output


# ---------------------------------------------------------------------------
# Feature selection tests
# ---------------------------------------------------------------------------


class TestFeatureSelection:
    def test_select_top_features(self):
        importance = FeatureImportanceResult(
            feature_names=[f"f{i}" for i in range(50)],
            importance_scores=np.arange(50, dtype=float),
            top_k_indices=np.argsort(np.arange(50))[::-1],
            top_k_names=[f"f{49 - i}" for i in range(50)],
        )
        selected = select_top_features(importance, k=10)
        assert len(selected) == 10
        assert 49 in selected  # Highest importance

    def test_monitor_feature_drift(self):
        baseline = FeatureImportanceResult(
            feature_names=[f"f{i}" for i in range(30)],
            importance_scores=np.arange(30, dtype=float)[::-1],
            top_k_indices=np.arange(30),
            top_k_names=[f"f{i}" for i in range(30)],
        )
        # In current, the top features dropped to bottom
        current = FeatureImportanceResult(
            feature_names=[f"f{i}" for i in range(30)],
            importance_scores=np.arange(30, dtype=float),  # Reversed
            top_k_indices=np.arange(30)[::-1],
            top_k_names=[f"f{29 - i}" for i in range(30)],
        )
        warnings = monitor_feature_drift(current, baseline, alert_threshold=20)
        # Some of baseline's top-10 should now be in current's bottom-20
        assert len(warnings) > 0
        assert "DRIFT" in warnings[0]


# ---------------------------------------------------------------------------
# Regime and NFP tests
# ---------------------------------------------------------------------------


class TestRegimeLabels:
    def test_generate_regime_labels(self):
        adx = np.array([30, 15, 22, 28, 10, 35])
        vix = np.array([12, 15, 30, 12, 28, 10])
        labels = generate_regime_labels(adx, vix)
        assert labels[0] == 0  # Trending (ADX=30)
        assert labels[1] == 1  # Ranging (ADX=15)
        assert labels[2] == 2  # Crisis (VIX=30, overrides ADX)
        assert labels[4] == 2  # Crisis (VIX=28)

    def test_all_regimes_present(self):
        adx = np.array([30, 15, 22])
        vix = np.array([12, 12, 30])
        labels = generate_regime_labels(adx, vix)
        assert set(labels) == {0, 1, 2}


class TestNFPData:
    def test_prepare_nfp_data(self):
        events = [
            {"nfp_surprise": 50, "gold_price": 3000, "dxy_level": 104, "gold_5d_return": -0.5, "gold_direction_after": 0},
            {"nfp_surprise": -100, "gold_price": 3100, "dxy_level": 103, "gold_5d_return": 1.2, "gold_direction_after": 1},
        ]
        X, y = prepare_nfp_data(events)
        assert X.shape == (2, 4)
        assert y.shape == (2,)
        assert X[0, 0] == 50  # nfp_surprise
        assert y[1] == 1  # gold_direction_after
