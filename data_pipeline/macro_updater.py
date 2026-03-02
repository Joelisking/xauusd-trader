"""Periodic macro data refresh — runs every 15 minutes.

Fetches DXY, US 10Y Real Yield, VIX, WTI Oil from external APIs.
Writes dxy_trend.json for MQL5 DXYFilter and macro features to SQLite.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ai_server.config import DATA_DIR, PROJECT_ROOT
from ai_server.macro.alpha_vantage_client import AlphaVantageClient, DXYSnapshot
from ai_server.macro.fred_client import FredClient, MacroSnapshot

DXY_JSON_PATH = PROJECT_ROOT / "dxy_trend.json"
MACRO_DB_PATH = DATA_DIR / "macro.db"


def _init_db(db_path: Path) -> sqlite3.Connection:
    """Create macro database and table if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_snapshots (
            timestamp TEXT PRIMARY KEY,
            dxy_price REAL,
            dxy_direction TEXT,
            dxy_ema_50 REAL,
            dxy_momentum REAL,
            real_yield REAL,
            real_yield_direction TEXT,
            nominal_10y REAL,
            vix REAL,
            vix_5d_roc REAL
        )
    """)
    conn.commit()
    return conn


def write_dxy_json(dxy: DXYSnapshot, macro: MacroSnapshot) -> None:
    """Write dxy_trend.json for MQL5 DXYFilter to read."""
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dxy_trend": dxy.direction,
        "dxy_price": dxy.price,
        "dxy_ema_50": dxy.ema_50,
        "dxy_momentum_roc": dxy.momentum_roc,
        "real_yield": macro.real_yield,
        "real_yield_direction": macro.real_yield_direction,
        "vix": macro.vix,
        "vix_5d_roc": macro.vix_5d_roc,
    }
    DXY_JSON_PATH.write_text(json.dumps(data, indent=2))


def write_to_db(dxy: DXYSnapshot, macro: MacroSnapshot) -> None:
    """Persist macro snapshot to SQLite for AI training features."""
    conn = _init_db(MACRO_DB_PATH)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO macro_snapshots
               (timestamp, dxy_price, dxy_direction, dxy_ema_50, dxy_momentum,
                real_yield, real_yield_direction, nominal_10y, vix, vix_5d_roc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                dxy.price,
                dxy.direction,
                dxy.ema_50,
                dxy.momentum_roc,
                macro.real_yield,
                macro.real_yield_direction,
                macro.nominal_10y,
                macro.vix,
                macro.vix_5d_roc,
            ),
        )
        conn.commit()
    finally:
        conn.close()


async def update_once() -> None:
    """Perform a single macro data update cycle."""
    fred = FredClient()
    av = AlphaVantageClient()

    try:
        macro = await fred.get_snapshot()
        dxy = await av.get_dxy_snapshot()

        write_dxy_json(dxy, macro)
        write_to_db(dxy, macro)

        print(
            f"[MacroUpdater] Updated: DXY={dxy.direction} "
            f"RealYield={macro.real_yield_direction} VIX={macro.vix}"
        )
    finally:
        await fred.close()
        await av.close()


async def run_loop(interval_minutes: int = 15) -> None:
    """Run the macro updater in a loop."""
    print(f"[MacroUpdater] Starting — interval {interval_minutes}min")
    while True:
        try:
            await update_once()
        except Exception as exc:
            print(f"[MacroUpdater] Error: {exc}")
        await asyncio.sleep(interval_minutes * 60)


def main() -> None:
    asyncio.run(update_once())


if __name__ == "__main__":
    main()
