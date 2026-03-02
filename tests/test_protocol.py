"""Unit tests for message validation and serialization."""

import math
import json
import pytest

from ai_server.config import FEATURE_COUNT
from ai_server.protocol import (
    EntryCheckRequest,
    EntryCheckResponse,
    HeartbeatResponse,
    ValidationError,
    deserialize,
    serialize,
    validate_entry_request,
)


class TestValidateEntryRequest:
    def _valid_data(self, **overrides) -> dict:
        base = {
            "type": "entry_check",
            "symbol": "XAUUSD",
            "direction": "BUY",
            "timeframe": "M1",
            "bot": "scalper",
            "session_hour": 14,
            "features": [0.0] * FEATURE_COUNT,
        }
        base.update(overrides)
        return base

    def test_valid_request(self):
        req = validate_entry_request(self._valid_data())
        assert isinstance(req, EntryCheckRequest)
        assert req.symbol == "XAUUSD"
        assert req.direction == "BUY"
        assert len(req.features) == FEATURE_COUNT

    def test_invalid_symbol(self):
        with pytest.raises(ValidationError, match="invalid_symbol"):
            validate_entry_request(self._valid_data(symbol="EURUSD"))

    def test_invalid_direction(self):
        with pytest.raises(ValidationError, match="invalid_direction"):
            validate_entry_request(self._valid_data(direction="HOLD"))

    def test_invalid_timeframe(self):
        with pytest.raises(ValidationError, match="invalid_timeframe"):
            validate_entry_request(self._valid_data(timeframe="D1"))

    def test_invalid_bot(self):
        with pytest.raises(ValidationError, match="invalid_bot"):
            validate_entry_request(self._valid_data(bot="hedger"))

    def test_wrong_feature_count(self):
        with pytest.raises(ValidationError, match="invalid_features"):
            validate_entry_request(self._valid_data(features=[0.0] * 50))

    def test_nan_in_features(self):
        features = [0.0] * FEATURE_COUNT
        features[10] = float("nan")
        with pytest.raises(ValidationError, match="nan_detected"):
            validate_entry_request(self._valid_data(features=features))

    def test_inf_in_features(self):
        features = [0.0] * FEATURE_COUNT
        features[5] = float("inf")
        with pytest.raises(ValidationError, match="nan_detected"):
            validate_entry_request(self._valid_data(features=features))

    def test_missing_symbol(self):
        data = self._valid_data()
        del data["symbol"]
        with pytest.raises(ValidationError, match="missing_field:symbol"):
            validate_entry_request(data)

    def test_missing_features(self):
        data = self._valid_data()
        del data["features"]
        with pytest.raises(ValidationError, match="missing_field:features"):
            validate_entry_request(data)

    def test_case_insensitive_symbol(self):
        req = validate_entry_request(self._valid_data(symbol="xauusd"))
        assert req.symbol == "XAUUSD"

    def test_case_insensitive_direction(self):
        req = validate_entry_request(self._valid_data(direction="sell"))
        assert req.direction == "SELL"

    def test_sell_direction(self):
        req = validate_entry_request(self._valid_data(direction="SELL"))
        assert req.direction == "SELL"

    def test_swing_bot(self):
        req = validate_entry_request(self._valid_data(bot="swing", timeframe="H1"))
        assert req.bot == "swing"
        assert req.timeframe == "H1"


class TestSerialization:
    def test_serialize_response(self):
        resp = EntryCheckResponse(entry_score=78, approve=True, regime="trending")
        raw = serialize(resp)
        parsed = json.loads(raw)
        assert parsed["entry_score"] == 78
        assert parsed["approve"] is True

    def test_serialize_heartbeat(self):
        resp = HeartbeatResponse(status="healthy", uptime_seconds=3600)
        raw = serialize(resp)
        parsed = json.loads(raw)
        assert parsed["status"] == "healthy"

    def test_deserialize_valid_json(self):
        data = deserialize('{"type": "heartbeat"}')
        assert data["type"] == "heartbeat"

    def test_deserialize_invalid_json(self):
        with pytest.raises(ValidationError, match="invalid_json"):
            deserialize("not json at all")
