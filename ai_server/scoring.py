"""Scoring interface — dummy implementation for Phase 2, real model drop-in later.

Provides the same interface that the full ensemble will implement in Phase 10.
"""

from __future__ import annotations

from ai_server.config import (
    LOT_MULT_HIGH,
    LOT_MULT_NORMAL,
    LOT_MULT_PRIME,
    MODEL_VERSION,
    NEWS_HALT_THRESHOLD,
    SCALPER_MIN_AI_SCORE,
    SESSION_OVERLAP_END,
    SESSION_OVERLAP_START,
    SWING_MIN_TREND_SCORE,
)
from ai_server.protocol import EntryCheckRequest, EntryCheckResponse


def score_entry(request: EntryCheckRequest, latency_ms: int = 0) -> EntryCheckResponse:
    """Score an entry request. Currently returns dummy scores for integration testing.

    In Phase 10 this will call real BiLSTM + XGBoost ensemble, regime classifier,
    and news risk scorer.
    """
    # --- Dummy scores ---
    entry_score = 75
    trend_score = 70
    news_risk = 10
    regime = "trending"
    wyckoff_phase = "D"

    # --- Lot multiplier logic (real — kept even with dummy scores) ---
    is_overlap = SESSION_OVERLAP_START <= request.session_hour < SESSION_OVERLAP_END
    if entry_score >= 80 and is_overlap:
        lot_mult = LOT_MULT_PRIME
    elif entry_score >= 80:
        lot_mult = LOT_MULT_HIGH
    else:
        lot_mult = LOT_MULT_NORMAL

    # --- Approval logic (real) ---
    if news_risk >= NEWS_HALT_THRESHOLD:
        approve = False
    elif request.bot == "scalper":
        approve = entry_score >= SCALPER_MIN_AI_SCORE
    else:
        approve = trend_score >= SWING_MIN_TREND_SCORE

    return EntryCheckResponse(
        entry_score=entry_score,
        trend_score=trend_score,
        news_risk=news_risk,
        wyckoff_phase=wyckoff_phase,
        regime=regime,
        approve=approve,
        recommended_lot_multiplier=lot_mult,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
    )
