"""Integration tests — connect to real server, send JSON, verify responses."""

import asyncio
import json

import pytest
import pytest_asyncio

from ai_server.config import FEATURE_COUNT
from tests.conftest import make_entry_check_request, make_heartbeat_request, send_json


@pytest.mark.asyncio
async def test_heartbeat(server_port):
    resp = await send_json(server_port, make_heartbeat_request())
    assert resp["status"] == "healthy"
    assert "uptime_seconds" in resp
    assert "model_version" in resp
    assert "predictions_today" in resp


@pytest.mark.asyncio
async def test_entry_check_scalper(server_port):
    data = make_entry_check_request(direction="BUY", bot="scalper")
    resp = await send_json(server_port, data)
    assert resp["entry_score"] == 75
    assert resp["trend_score"] == 70
    assert resp["news_risk"] == 10
    assert resp["approve"] is True
    assert resp["regime"] == "trending"
    assert resp["wyckoff_phase"] == "D"
    assert resp["model_version"] != ""
    assert resp["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_entry_check_swing(server_port):
    data = make_entry_check_request(
        direction="SELL", bot="swing", timeframe="H1",
    )
    resp = await send_json(server_port, data)
    assert resp["entry_score"] == 75
    assert resp["approve"] is False  # trend_score 70 < 72 swing threshold


@pytest.mark.asyncio
async def test_invalid_symbol(server_port):
    data = make_entry_check_request(symbol="EURUSD")
    # Override after helper builds it
    data["symbol"] = "EURUSD"
    resp = await send_json(server_port, data)
    assert resp["approve"] is False
    assert "invalid_symbol" in resp.get("error", "")


@pytest.mark.asyncio
async def test_wrong_feature_count(server_port):
    data = make_entry_check_request(features=[0.0] * 50)
    resp = await send_json(server_port, data)
    assert resp["approve"] is False
    assert "invalid_features" in resp.get("error", "")


@pytest.mark.asyncio
async def test_nan_features(server_port):
    features = [0.0] * FEATURE_COUNT
    features[0] = float("nan")
    data = make_entry_check_request(features=features)
    resp = await send_json(server_port, data)
    assert resp["approve"] is False
    assert "nan_detected" in resp.get("error", "")


@pytest.mark.asyncio
async def test_malformed_json(server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)
    writer.write(b"this is not json\n")
    await writer.drain()
    raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
    resp = json.loads(raw)
    assert resp["approve"] is False
    assert "invalid_json" in resp.get("error", "")
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_unknown_message_type(server_port):
    resp = await send_json(server_port, {"type": "foobar"})
    assert "error" in resp
    assert "unknown_type" in resp["error"]


@pytest.mark.asyncio
async def test_multiple_requests_same_connection(server_port):
    """Verify persistent connections work."""
    reader, writer = await asyncio.open_connection("127.0.0.1", server_port)

    # Send heartbeat
    writer.write((json.dumps(make_heartbeat_request()) + "\n").encode())
    await writer.drain()
    raw1 = await asyncio.wait_for(reader.readline(), timeout=5.0)
    resp1 = json.loads(raw1)
    assert resp1["status"] == "healthy"

    # Send entry check on same connection
    data = make_entry_check_request()
    writer.write((json.dumps(data) + "\n").encode())
    await writer.drain()
    raw2 = await asyncio.wait_for(reader.readline(), timeout=5.0)
    resp2 = json.loads(raw2)
    assert resp2["entry_score"] == 75

    writer.close()
    await writer.wait_closed()
