"""Alpha Vantage client — fetches DXY (US Dollar Index) data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiohttp

from ai_server.config import ALPHA_VANTAGE_API_KEY

BASE_URL = "https://www.alphavantage.co/query"

# DXY is tracked via the Invesco DB US Dollar Index Bullish Fund (UUP) as proxy,
# or FOREX rate EURUSD inverted. Alpha Vantage provides forex pairs.
# We use DX-Y.NYB or approximate via EUR/USD inverse.
DXY_SYMBOL = "UUP"  # ETF proxy for DXY


@dataclass
class DXYSnapshot:
    price: float | None = None
    ema_50: float | None = None
    direction: str = "NEUTRAL"  # UP, DOWN, NEUTRAL
    distance_from_ema: float = 0.0
    momentum_roc: float = 0.0  # 5-day rate of change %
    timestamp: datetime | None = None


class AlphaVantageClient:
    """Async client for Alpha Vantage API — DXY data."""

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key or ALPHA_VANTAGE_API_KEY
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_dxy_snapshot(self) -> DXYSnapshot:
        """Fetch current DXY snapshot using daily time series."""
        snap = DXYSnapshot(timestamp=datetime.now(timezone.utc))

        if not self._api_key:
            return snap

        session = await self._get_session()
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": DXY_SYMBOL,
            "apikey": self._api_key,
            "outputsize": "compact",  # last 100 days
        }

        try:
            async with session.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return snap
                data = await resp.json()

            ts = data.get("Time Series (Daily)", {})
            if not ts:
                return snap

            # Sort dates descending
            dates = sorted(ts.keys(), reverse=True)
            if not dates:
                return snap

            # Current price
            latest = ts[dates[0]]
            snap.price = float(latest["4. close"])

            # 50-day EMA approximation (simple average of last 50 closes)
            closes = []
            for d in dates[:50]:
                closes.append(float(ts[d]["4. close"]))

            if len(closes) >= 50:
                snap.ema_50 = sum(closes) / len(closes)
                snap.distance_from_ema = snap.price - snap.ema_50

            # Direction from 5-day trend
            if len(closes) >= 6:
                snap.momentum_roc = ((closes[0] - closes[5]) / closes[5]) * 100
                if snap.momentum_roc > 0.3:
                    snap.direction = "UP"
                elif snap.momentum_roc < -0.3:
                    snap.direction = "DOWN"

        except Exception:
            pass

        return snap
