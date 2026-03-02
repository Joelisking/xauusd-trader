"""Export XAUUSD historical data from MT5 to Parquet files.

Runs on Windows VPS only (requires MetaTrader5 Python library).
Exports M1, M5, H1, H4 to data/XAUUSD_{timeframe}.parquet.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ai_server.config import DATA_DIR

# MT5 timeframe constants (defined here to avoid import errors on Mac)
_TIMEFRAMES = {
    "M1": 1,      # TIMEFRAME_M1
    "M5": 5,      # TIMEFRAME_M5
    "H1": 16385,  # TIMEFRAME_H1
    "H4": 16388,  # TIMEFRAME_H4
}

# How far back to export per timeframe
_YEARS_BACK = {
    "M1": 10,
    "M5": 10,
    "H1": 10,
    "H4": 10,
}

COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


def export_timeframe(symbol: str, tf_name: str, years: int) -> Path:
    """Export a single timeframe from MT5 to Parquet. Returns the output path."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package not available — run on Windows VPS", file=sys.stderr)
        raise

    tf_const = _TIMEFRAMES[tf_name]
    end = datetime.now(timezone.utc)
    start = end.replace(year=end.year - years)

    rates = mt5.copy_rates_range(symbol, tf_const, start, end)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data returned for {symbol} {tf_name}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[COLUMNS]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{symbol}_{tf_name}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")

    print(f"Exported {len(df):,} rows to {out_path}")
    return out_path


def export_all(symbol: str = "XAUUSD") -> dict[str, Path]:
    """Export all timeframes. Returns {tf_name: path}."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package not available — run on Windows VPS", file=sys.stderr)
        raise

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    try:
        results = {}
        for tf_name, years in _YEARS_BACK.items():
            results[tf_name] = export_timeframe(symbol, tf_name, years)
        return results
    finally:
        mt5.shutdown()


def main() -> None:
    paths = export_all()
    print(f"\nExported {len(paths)} timeframes:")
    for tf, path in paths.items():
        print(f"  {tf}: {path}")


if __name__ == "__main__":
    main()
