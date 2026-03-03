"""Tests for the full scoring interface — Phase 10."""

import numpy as np
import pytest

from ai_server.config import (
    FEATURE_COUNT,
    MODEL_VERSION,
    NEWS_HALT_THRESHOLD,
    NEWS_REDUCE_THRESHOLD,
    SCALPER_MIN_AI_SCORE,
    SWING_MIN_TREND_SCORE,
)
from ai_server.protocol import EntryCheckRequest
from ai_server.scoring import (
    init_models,
    is_fallback_mode,
    score_entry,
    _compute_news_risk,
    _extract_features,
    _fallback_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    bot: str = "scalper",
    direction: str = "BUY",
    session_hour: int = 14,
    vix_level: float = 18.0,
    current_spread: float = 1.0,
) -> EntryCheckRequest:
    return EntryCheckRequest(
        type="entry_check",
        symbol="XAUUSD",
        direction=direction,
        timeframe="M1" if bot == "scalper" else "H1",
        bot=bot,
        session_hour=session_hour,
        dxy_trend="DOWN",
        real_yield_trend="DOWN",
        vix_level=vix_level,
        current_spread=current_spread,
        atr_14=15.0,
        session_risk_used=3.0,
        account_drawdown=2.0,
        features=[0.5] * FEATURE_COUNT,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitModels:
    def test_init_enters_fallback_without_weights(self):
        """Without trained model files, init should enter fallback mode."""
        init_models()
        assert is_fallback_mode() is True


class TestFallbackScoring:
    def test_fallback_score_scalper(self):
        req = _make_request(bot="scalper")
        resp = _fallback_score(req)
        assert resp.entry_score < SCALPER_MIN_AI_SCORE  # Below threshold
        assert resp.approve is False
        assert resp.recommended_lot_multiplier == 0.0
        assert "fallback" in resp.model_version

    def test_fallback_score_swing(self):
        req = _make_request(bot="swing")
        resp = _fallback_score(req)
        assert resp.entry_score < SWING_MIN_TREND_SCORE
        assert resp.approve is False


class TestScoreEntry:
    def test_score_entry_returns_valid_response(self):
        init_models()
        req = _make_request()
        resp = score_entry(req)
        assert 0 <= resp.entry_score <= 100
        assert 0 <= resp.trend_score <= 100
        assert 0 <= resp.news_risk <= 100
        assert resp.regime in ("trending", "ranging", "crisis")
        assert resp.wyckoff_phase in ("A", "B", "C", "D", "E")
        assert resp.model_version != ""
        assert resp.latency_ms >= 0

    def test_score_entry_scalper(self):
        init_models()
        req = _make_request(bot="scalper")
        resp = score_entry(req)
        assert isinstance(resp.entry_score, int)

    def test_score_entry_swing(self):
        init_models()
        req = _make_request(bot="swing")
        resp = score_entry(req)
        assert isinstance(resp.entry_score, int)


class TestNewsRisk:
    def test_low_vix_low_spread(self):
        req = _make_request(vix_level=15.0, current_spread=0.5)
        risk = _compute_news_risk(req)
        assert risk == 0

    def test_high_vix(self):
        req = _make_request(vix_level=32.0, current_spread=0.5)
        risk = _compute_news_risk(req)
        assert risk >= 40

    def test_wide_spread(self):
        req = _make_request(vix_level=15.0, current_spread=3.5)
        risk = _compute_news_risk(req)
        assert risk >= 30

    def test_combined_risk_caps_at_100(self):
        req = _make_request(vix_level=35.0, current_spread=4.0)
        risk = _compute_news_risk(req)
        assert risk <= 100

    def test_high_news_risk_blocks_approval(self):
        """When news risk >= NEWS_HALT_THRESHOLD, approve must be False."""
        init_models()
        req = _make_request(vix_level=35.0, current_spread=4.0)
        resp = score_entry(req)
        if resp.news_risk >= NEWS_HALT_THRESHOLD:
            assert resp.approve is False
            assert resp.recommended_lot_multiplier == 0.0


class TestFeatureExtraction:
    def test_extract_features_shapes(self):
        req = _make_request()
        seq, tab = _extract_features(req)
        assert seq.shape == (200, FEATURE_COUNT)
        assert tab.shape == (60,)
        assert seq.dtype == np.float32
        assert tab.dtype == np.float32

    def test_extract_features_values(self):
        req = _make_request()
        seq, tab = _extract_features(req)
        # All features are 0.5 in test request
        assert np.allclose(tab, 0.5)
