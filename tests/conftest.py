"""Shared test fixtures for the XAUUSD AI Trading System."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from ai_server.config import AI_SERVER_HOST, AI_SERVER_PORT, FEATURE_COUNT
from ai_server.server import start_server


def make_entry_check_request(
    direction: str = "BUY",
    features: list[float] | None = None,
    **overrides,
) -> dict:
    """Build a valid entry_check request dict."""
    if features is None:
        features = [0.0] * FEATURE_COUNT
    base = {
        "type": "entry_check",
        "symbol": "XAUUSD",
        "direction": direction,
        "timeframe": "M1",
        "bot": "scalper",
        "session_hour": 14,
        "dxy_trend": "DOWN",
        "real_yield_trend": "DOWN",
        "vix_level": 18.5,
        "current_spread": 0.8,
        "atr_14": 15.2,
        "session_risk_used": 3.5,
        "account_drawdown": 2.1,
        "features": features,
    }
    base.update(overrides)
    return base


def make_heartbeat_request() -> dict:
    return {"type": "heartbeat"}


@pytest_asyncio.fixture
async def server_port(unused_tcp_port_factory) -> AsyncGenerator[int, None]:
    """Start the AI server on a random port and yield the port number."""
    import ai_server.config as cfg

    # Override config to use a free port
    original_port = cfg.AI_SERVER_PORT
    port = unused_tcp_port_factory()
    cfg.AI_SERVER_PORT = port

    server_task = asyncio.create_task(_run_server(cfg.AI_SERVER_HOST, port))
    await asyncio.sleep(0.1)  # give server time to bind

    yield port

    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    cfg.AI_SERVER_PORT = original_port


async def _run_server(host: str, port: int) -> None:
    from ai_server.scoring import init_models
    init_models()  # Load models (will enter fallback mode in tests)

    server = await asyncio.start_server(
        __import__("ai_server.server", fromlist=["handle_client"]).handle_client,
        host,
        port,
    )
    async with server:
        await server.serve_forever()


async def send_json(port: int, data: dict, host: str = AI_SERVER_HOST) -> dict:
    """Connect, send a JSON message, receive one JSON response."""
    reader, writer = await asyncio.open_connection(host, port)
    payload = json.dumps(data) + "\n"
    writer.write(payload.encode("utf-8"))
    await writer.drain()

    raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()
    return json.loads(raw.decode("utf-8"))
