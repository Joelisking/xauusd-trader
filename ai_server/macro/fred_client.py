"""FRED API client — fetches US 10Y Real Yield (DFII10), VIX (VIXCLS), and related series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from ai_server.config import FRED_API_KEY

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs
SERIES_REAL_YIELD = "DFII10"    # 10-Year Treasury Inflation-Indexed Security
SERIES_NOMINAL_10Y = "DGS10"    # 10-Year Treasury Constant Maturity
SERIES_VIX = "VIXCLS"           # CBOE Volatility Index


@dataclass
class FredObservation:
    date: str
    value: float


@dataclass
class MacroSnapshot:
    real_yield: float | None = None
    real_yield_direction: str = "NEUTRAL"  # UP, DOWN, NEUTRAL
    nominal_10y: float | None = None
    vix: float | None = None
    vix_5d_roc: float | None = None
    timestamp: datetime | None = None


class FredClient:
    """Async client for FRED API."""

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key or FRED_API_KEY
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_series(
        self, series_id: str, lookback_days: int = 30
    ) -> list[FredObservation]:
        """Fetch recent observations for a FRED series."""
        if not self._api_key:
            return []

        session = await self._get_session()
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "desc",
        }

        try:
            async with session.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                observations = []
                for obs in data.get("observations", []):
                    val_str = obs.get("value", ".")
                    if val_str == "." or val_str == "":
                        continue
                    observations.append(FredObservation(
                        date=obs["date"],
                        value=float(val_str),
                    ))
                return observations
        except Exception:
            return []

    async def get_snapshot(self) -> MacroSnapshot:
        """Fetch current macro snapshot: real yield, VIX, 10Y nominal."""
        snap = MacroSnapshot(timestamp=datetime.now(timezone.utc))

        # Real yield
        ry_obs = await self._fetch_series(SERIES_REAL_YIELD, lookback_days=30)
        if len(ry_obs) >= 2:
            snap.real_yield = ry_obs[0].value
            prev = ry_obs[1].value
            diff = snap.real_yield - prev
            if diff > 0.02:
                snap.real_yield_direction = "UP"
            elif diff < -0.02:
                snap.real_yield_direction = "DOWN"

        # VIX
        vix_obs = await self._fetch_series(SERIES_VIX, lookback_days=30)
        if vix_obs:
            snap.vix = vix_obs[0].value
        if len(vix_obs) >= 6:
            snap.vix_5d_roc = ((vix_obs[0].value - vix_obs[5].value) / vix_obs[5].value) * 100

        # Nominal 10Y
        nom_obs = await self._fetch_series(SERIES_NOMINAL_10Y, lookback_days=10)
        if nom_obs:
            snap.nominal_10y = nom_obs[0].value

        return snap
