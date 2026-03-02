"""Message protocol — dataclasses, validation, and JSON serialization.

Defines the contract between MT5 EAs and the Python AI server.
Transport: newline-delimited JSON over TCP socket on localhost:5001.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ai_server.config import FEATURE_COUNT

# ---------------------------------------------------------------------------
# Request dataclasses
# ---------------------------------------------------------------------------

VALID_SYMBOLS = {"XAUUSD"}
VALID_DIRECTIONS = {"BUY", "SELL"}
VALID_TIMEFRAMES = {"M1", "M5", "H1", "H4"}
VALID_BOTS = {"scalper", "swing"}
VALID_DXY_TRENDS = {"UP", "DOWN", "NEUTRAL"}


@dataclass
class EntryCheckRequest:
    type: str  # "entry_check"
    symbol: str
    direction: str
    timeframe: str
    bot: str
    session_hour: int
    dxy_trend: str = "NEUTRAL"
    real_yield_trend: str = "NEUTRAL"
    vix_level: float = 0.0
    current_spread: float = 0.0
    atr_14: float = 0.0
    session_risk_used: float = 0.0
    account_drawdown: float = 0.0
    features: list[float] = field(default_factory=list)


@dataclass
class HeartbeatRequest:
    type: str  # "heartbeat"


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EntryCheckResponse:
    entry_score: int = 0
    trend_score: int = 0
    news_risk: int = 0
    wyckoff_phase: str = "D"
    regime: str = "trending"
    approve: bool = False
    recommended_lot_multiplier: float = 1.0
    model_version: str = ""
    latency_ms: int = 0


@dataclass
class HeartbeatResponse:
    status: str = "healthy"
    uptime_seconds: int = 0
    model_version: str = ""
    predictions_today: int = 0
    avg_latency_ms: int = 0
    queue_depth: int = 0


@dataclass
class ErrorResponse:
    approve: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when request validation fails."""


def validate_entry_request(data: dict[str, Any]) -> EntryCheckRequest:
    """Validate and parse an entry_check request dict into a dataclass.

    Raises ValidationError on bad input.
    """
    # Required fields
    for key in ("symbol", "direction", "timeframe", "bot", "features"):
        if key not in data:
            raise ValidationError(f"missing_field:{key}")

    symbol = str(data["symbol"]).upper()
    if symbol not in VALID_SYMBOLS:
        raise ValidationError("invalid_symbol")

    direction = str(data["direction"]).upper()
    if direction not in VALID_DIRECTIONS:
        raise ValidationError("invalid_direction")

    timeframe = str(data["timeframe"]).upper()
    if timeframe not in VALID_TIMEFRAMES:
        raise ValidationError(f"invalid_timeframe:{timeframe}")

    bot = str(data["bot"]).lower()
    if bot not in VALID_BOTS:
        raise ValidationError(f"invalid_bot:{bot}")

    features = data["features"]
    if not isinstance(features, list):
        raise ValidationError("features_not_list")
    if len(features) != FEATURE_COUNT:
        raise ValidationError(f"invalid_features:expected_{FEATURE_COUNT}_got_{len(features)}")

    # NaN / Inf check
    for i, v in enumerate(features):
        if not isinstance(v, (int, float)):
            raise ValidationError(f"feature_not_numeric:index_{i}")
        if math.isnan(v) or math.isinf(v):
            raise ValidationError("nan_detected")

    session_hour = int(data.get("session_hour", 0))

    return EntryCheckRequest(
        type="entry_check",
        symbol=symbol,
        direction=direction,
        timeframe=timeframe,
        bot=bot,
        session_hour=session_hour,
        dxy_trend=str(data.get("dxy_trend", "NEUTRAL")).upper(),
        real_yield_trend=str(data.get("real_yield_trend", "NEUTRAL")).upper(),
        vix_level=float(data.get("vix_level", 0.0)),
        current_spread=float(data.get("current_spread", 0.0)),
        atr_14=float(data.get("atr_14", 0.0)),
        session_risk_used=float(data.get("session_risk_used", 0.0)),
        account_drawdown=float(data.get("account_drawdown", 0.0)),
        features=[float(v) for v in features],
    )


def validate_heartbeat(data: dict[str, Any]) -> HeartbeatRequest:
    return HeartbeatRequest(type="heartbeat")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def serialize(obj: Any) -> str:
    """Serialize a dataclass to a JSON string (no trailing newline)."""
    return json.dumps(asdict(obj))


def deserialize(raw: str) -> dict[str, Any]:
    """Parse a JSON string into a dict. Raises ValidationError on bad JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid_json:{exc}") from exc
