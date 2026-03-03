"""Scoring interface — full ensemble inference (Phase 10).

Loads all models at startup and routes entry_check requests to the
appropriate ensemble (scalper or swing).  Falls back to degraded mode
with conservative dummy scores when model weights are unavailable.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ai_server.config import (
    FEATURE_COUNT,
    LOT_MULT_HIGH,
    LOT_MULT_NORMAL,
    LOT_MULT_PRIME,
    MODEL_VERSION,
    NEWS_HALT_THRESHOLD,
    NEWS_REDUCE_THRESHOLD,
    SCALPER_MIN_AI_SCORE,
    SEQUENCE_LENGTH,
    SESSION_OVERLAP_END,
    SESSION_OVERLAP_START,
    SWING_MIN_TREND_SCORE,
    XGB_FEATURE_COUNT,
)
from ai_server.features.feature_engine import FeatureEngine
from ai_server.features.macro_features import MacroContext
from ai_server.models.ensemble import EnsembleScorer
from ai_server.protocol import EntryCheckRequest, EntryCheckResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level scorer — loaded once at startup via init_models()
# ---------------------------------------------------------------------------

_scorer: EnsembleScorer | None = None
_engine: FeatureEngine = FeatureEngine()
_fallback_mode: bool = False


def init_models() -> None:
    """Load all model weights into memory.  Call once at server startup."""
    global _scorer, _fallback_mode
    _scorer = EnsembleScorer()
    try:
        _scorer.load_all_models()
        _fallback_mode = not _scorer.models_loaded
    except Exception as exc:
        logger.error("Failed to initialize models: %s — entering fallback mode", exc)
        _fallback_mode = True

    if _fallback_mode:
        logger.warning("Scoring running in FALLBACK mode (dummy scores)")
    else:
        logger.info("All models loaded — scoring fully operational")


def is_fallback_mode() -> bool:
    """True when models are not loaded and dummy scores are returned."""
    return _fallback_mode


# ---------------------------------------------------------------------------
# News risk helper
# ---------------------------------------------------------------------------


def _compute_news_risk(request: EntryCheckRequest) -> int:
    """Derive a news risk score from request context fields.

    In production this pulls from the news calendar service; here we
    use the VIX level and spread anomaly as proxies when the full
    calendar data is not embedded in the request.
    """
    risk = 0
    # VIX contribution
    if request.vix_level > 30:
        risk += 40
    elif request.vix_level > 25:
        risk += 25
    elif request.vix_level > 20:
        risk += 10

    # Spread anomaly
    if request.current_spread > 3.0:
        risk += 30
    elif request.current_spread > 2.0:
        risk += 15

    return min(risk, 100)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_features(request: EntryCheckRequest) -> tuple[np.ndarray, np.ndarray]:
    """Extract sequential and tabular feature arrays from the request.

    The EA sends 127 flat features per request.  For BiLSTM we reshape
    into (1, SEQUENCE_LENGTH, FEATURE_COUNT) by tiling the single row
    into a pseudo-sequence.  For XGBoost we take the first XGB_FEATURE_COUNT.

    NOTE: In production the EA should send the full 200-candle sequence.
    This tiling is a safe fallback for the current single-row protocol.
    """
    features = np.array(request.features, dtype=np.float32)

    # Sequential: tile the 127-feature vector across SEQUENCE_LENGTH time-steps
    sequential = np.tile(features, (SEQUENCE_LENGTH, 1))  # (200, 127)

    # Tabular: first XGB_FEATURE_COUNT features
    tabular = features[:XGB_FEATURE_COUNT]

    return sequential, tabular


# ---------------------------------------------------------------------------
# Public scoring entry point
# ---------------------------------------------------------------------------


def score_entry(request: EntryCheckRequest, latency_ms: int = 0) -> EntryCheckResponse:
    """Score an entry request using the full ensemble or fallback mode.

    This is the single entry point called by server.py for every
    entry_check message.
    """
    global _scorer
    t0 = time.monotonic()

    # --- Fallback / degraded mode ---
    if _fallback_mode or _scorer is None:
        return _fallback_score(request, latency_ms)

    # --- Extract features ---
    sequential, tabular = _extract_features(request)

    # --- Route to correct ensemble ---
    if request.bot == "scalper":
        result = _scorer.score_scalper_entry(
            sequential_features=sequential,
            tabular_features=tabular,
            session_hour=request.session_hour,
        )
    else:
        result = _scorer.score_swing_entry(
            sequential_features=sequential,
            tabular_features=tabular,
            session_hour=request.session_hour,
        )

    # --- News risk ---
    news_risk = _compute_news_risk(request)

    # --- Override approval on high news risk ---
    approve = result.approve
    lot_mult = result.lot_multiplier
    if news_risk >= NEWS_HALT_THRESHOLD:
        approve = False
        lot_mult = 0.0
    elif news_risk >= NEWS_REDUCE_THRESHOLD:
        lot_mult = lot_mult * 0.5

    elapsed = int((time.monotonic() - t0) * 1000)

    return EntryCheckResponse(
        entry_score=int(round(result.entry_score)),
        trend_score=int(round(result.trend_score)),
        news_risk=news_risk,
        wyckoff_phase=result.wyckoff_phase,
        regime=result.regime,
        approve=approve,
        recommended_lot_multiplier=round(lot_mult, 2),
        model_version=MODEL_VERSION,
        latency_ms=max(elapsed, latency_ms),
    )


# ---------------------------------------------------------------------------
# Fallback scoring (no models loaded)
# ---------------------------------------------------------------------------


def _fallback_score(request: EntryCheckRequest, latency_ms: int = 0) -> EntryCheckResponse:
    """Conservative dummy scores when models are unavailable.

    Returns below-threshold scores so the EA enters fallback mode
    (rule-based only, no AI-boosted entries).
    """
    entry_score = 50  # Below SCALPER_MIN_AI_SCORE (68) — EA won't take AI entries
    trend_score = 50  # Below SWING_MIN_TREND_SCORE (72)
    news_risk = _compute_news_risk(request)

    return EntryCheckResponse(
        entry_score=entry_score,
        trend_score=trend_score,
        news_risk=news_risk,
        wyckoff_phase="A",
        regime="ranging",
        approve=False,  # Never approve in fallback
        recommended_lot_multiplier=0.0,
        model_version=MODEL_VERSION + "-fallback",
        latency_ms=latency_ms,
    )
