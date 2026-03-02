"""Async TCP server — the bridge between MT5 EAs and the AI engine.

Transport: newline-delimited JSON over TCP on localhost:5001.
Run with: python -m ai_server.server
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from ai_server.config import AI_SERVER_HOST, AI_SERVER_PORT, PREDICTION_LOG_DIR
from ai_server.health import HealthTracker
from ai_server.protocol import (
    ErrorResponse,
    ValidationError,
    deserialize,
    serialize,
    validate_entry_request,
    validate_heartbeat,
)
from ai_server.scoring import score_entry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_server")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

health = HealthTracker()


# ---------------------------------------------------------------------------
# Prediction logger
# ---------------------------------------------------------------------------


def _log_prediction(request_data: dict, response_data: dict, latency_ms: int) -> None:
    """Append a prediction record to today's JSONL log file."""
    PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = PREDICTION_LOG_DIR / f"{today}.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bot": request_data.get("bot", ""),
        "direction": request_data.get("direction", ""),
        "entry_score": response_data.get("entry_score"),
        "trend_score": response_data.get("trend_score"),
        "regime": response_data.get("regime"),
        "approved": response_data.get("approve"),
        "latency_ms": latency_ms,
        "actual_outcome": None,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    addr = writer.get_extra_info("peername")
    logger.info("Connection from %s", addr)

    try:
        while True:
            raw = await asyncio.wait_for(reader.readline(), timeout=30.0)
            if not raw:
                break

            line = raw.decode("utf-8").strip()
            if not line:
                continue

            t0 = time.monotonic()

            try:
                data = deserialize(line)
            except ValidationError as exc:
                response = serialize(ErrorResponse(error=str(exc)))
                writer.write((response + "\n").encode("utf-8"))
                await writer.drain()
                continue

            msg_type = data.get("type", "")

            if msg_type == "heartbeat":
                validate_heartbeat(data)
                response_dict = health.to_dict()
                response_str = json.dumps(response_dict)

            elif msg_type == "entry_check":
                try:
                    request = validate_entry_request(data)
                except ValidationError as exc:
                    response_str = serialize(ErrorResponse(error=str(exc)))
                    writer.write((response_str + "\n").encode("utf-8"))
                    await writer.drain()
                    continue

                latency_ms = int((time.monotonic() - t0) * 1000)
                result = score_entry(request, latency_ms=latency_ms)
                latency_ms = int((time.monotonic() - t0) * 1000)
                result.latency_ms = latency_ms

                health.record_prediction(latency_ms)

                response_str = serialize(result)
                response_dict = json.loads(response_str)
                _log_prediction(data, response_dict, latency_ms)

            else:
                response_str = serialize(ErrorResponse(error=f"unknown_type:{msg_type}"))

            writer.write((response_str + "\n").encode("utf-8"))
            await writer.drain()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info("%s request processed in %dms", msg_type, elapsed)

    except asyncio.TimeoutError:
        logger.info("Client %s timed out", addr)
    except ConnectionResetError:
        logger.info("Client %s disconnected", addr)
    except Exception:
        logger.exception("Error handling client %s", addr)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        logger.info("Connection closed: %s", addr)


# ---------------------------------------------------------------------------
# Server entrypoint
# ---------------------------------------------------------------------------


async def start_server() -> None:
    server = await asyncio.start_server(handle_client, AI_SERVER_HOST, AI_SERVER_PORT)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    logger.info("AI Server listening on %s", addrs)

    async with server:
        await server.serve_forever()


def main() -> None:
    logger.info("Starting XAUUSD AI Server on %s:%d", AI_SERVER_HOST, AI_SERVER_PORT)
    asyncio.run(start_server())


if __name__ == "__main__":
    main()
