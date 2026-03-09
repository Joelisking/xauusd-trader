"""Download XAUUSD M1 historical data from Dukascopy (free, no API key needed).

Dukascopy provides tick data in bi5 format (LZMA-compressed binary).
This script downloads tick data hour by hour, aggregates to M1 OHLCV bars,
and saves as Parquet matching the MT5 export format.

Usage:
    uv run python -m data_pipeline.dukascopy_download
    uv run python -m data_pipeline.dukascopy_download --start 2016 --end 2026
    uv run python -m data_pipeline.dukascopy_download --start 2020 --end 2026 --resume
"""

from __future__ import annotations

import argparse
import io
import lzma
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ai_server.config import DATA_DIR

# Dukascopy data feed URL pattern
# Months are 0-indexed (January = 00)
_BASE_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

# XAUUSD point value — Dukascopy stores prices as integers, divide by this
_POINT_DIVISOR = 1000.0

# Tick record: 20 bytes each
# uint32: milliseconds since start of hour
# uint32: ask price (divide by _POINT_DIVISOR)
# uint32: bid price (divide by _POINT_DIVISOR)
# float32: ask volume
# float32: bid volume
_TICK_STRUCT = struct.Struct(">IIIff")
_TICK_SIZE = _TICK_STRUCT.size  # 20 bytes

# Output columns matching MT5 export format
_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]

# Cache directory for raw bi5 files (allows resuming)
_CACHE_DIR = DATA_DIR / "dukascopy_cache"


def parse_bi5(data: bytes, hour_start: datetime) -> list[dict]:
    """Parse a bi5 (LZMA-compressed) tick data file.

    Returns list of tick dicts with time, bid, ask, volume.
    """
    if not data or len(data) < 4:
        return []

    try:
        decompressed = lzma.decompress(data)
    except lzma.LZMAError:
        return []

    if len(decompressed) == 0:
        return []

    n_ticks = len(decompressed) // _TICK_SIZE
    ticks = []

    for i in range(n_ticks):
        offset = i * _TICK_SIZE
        ms, ask_raw, bid_raw, ask_vol, bid_vol = _TICK_STRUCT.unpack_from(
            decompressed, offset
        )

        tick_time = hour_start + timedelta(milliseconds=ms)
        ask = ask_raw / _POINT_DIVISOR
        bid = bid_raw / _POINT_DIVISOR
        mid = (ask + bid) / 2.0
        spread = round(ask - bid, 3)
        volume = ask_vol + bid_vol

        ticks.append({
            "time": tick_time,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "volume": volume,
        })

    return ticks


def ticks_to_m1(ticks: list[dict]) -> pd.DataFrame:
    """Aggregate tick data into M1 OHLCV bars."""
    if not ticks:
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")

    # Resample to 1-minute bars using mid price
    ohlc = df["mid"].resample("1min").ohlc()
    ohlc = ohlc.dropna()

    if ohlc.empty:
        return pd.DataFrame(columns=_COLUMNS)

    # Aggregate volume and spread
    vol = df["volume"].resample("1min").sum()
    spread_avg = df["spread"].resample("1min").mean()
    tick_count = df["mid"].resample("1min").count()

    result = pd.DataFrame({
        "time": ohlc.index,
        "open": ohlc["open"].values,
        "high": ohlc["high"].values,
        "low": ohlc["low"].values,
        "close": ohlc["close"].values,
        "tick_volume": tick_count.loc[ohlc.index].values,
        "spread": (spread_avg.loc[ohlc.index].values * _POINT_DIVISOR).astype(int),
        "real_volume": vol.loc[ohlc.index].values,
    })

    return result


def download_hour(
    symbol: str,
    dt: datetime,
    session: requests.Session,
    use_cache: bool = True,
) -> list[dict]:
    """Download tick data for a single hour. Returns list of tick dicts."""
    # Cache path
    cache_file = _CACHE_DIR / f"{dt.year}/{dt.month:02d}/{dt.day:02d}/{dt.hour:02d}.bi5"

    if use_cache and cache_file.exists():
        data = cache_file.read_bytes()
        if len(data) > 0:
            return parse_bi5(data, dt)
        return []

    # Dukascopy months are 0-indexed
    url = _BASE_URL.format(
        symbol=symbol,
        year=dt.year,
        month=dt.month - 1,  # 0-indexed!
        day=dt.day,
        hour=dt.hour,
    )

    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 0:
            # Cache the raw data
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(resp.content)
            return parse_bi5(resp.content, dt)
        else:
            # Cache empty marker so we don't retry
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(b"")
            return []
    except requests.RequestException:
        return []


def download_m1_data(
    symbol: str = "XAUUSD",
    start_year: int = 2016,
    end_year: int = 2026,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download M1 data from Dukascopy for the given year range.

    Returns DataFrame with columns matching MT5 export format.
    """
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(end_year, 3, 1, tzinfo=timezone.utc)  # Up to current date
    now = datetime.now(timezone.utc)
    if end > now:
        end = now

    # Calculate total hours for progress
    total_hours = int((end - start).total_seconds() / 3600)
    print(f"Downloading {symbol} tick data: {start.date()} to {end.date()}")
    print(f"Total hours to process: {total_hours:,}")
    print(f"Cache directory: {_CACHE_DIR}")
    print()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; trading-research/1.0)",
    })

    all_bars: list[pd.DataFrame] = []
    current = start
    processed = 0
    last_progress = time.time()
    bars_total = 0

    while current < end:
        ticks = download_hour(symbol, current, session, use_cache=use_cache)

        if ticks:
            bars = ticks_to_m1(ticks)
            if not bars.empty:
                all_bars.append(bars)
                bars_total += len(bars)

        processed += 1
        current += timedelta(hours=1)

        # Progress update every 5 seconds
        elapsed = time.time() - last_progress
        if elapsed >= 5.0 or processed == total_hours:
            pct = 100 * processed / total_hours if total_hours > 0 else 100
            # Estimate remaining time
            rate = processed / max(time.time() - (last_progress - elapsed), 1)
            remaining = (total_hours - processed) / max(rate, 0.1)
            remaining_min = remaining / 60

            print(
                f"\r  Progress: {processed:,}/{total_hours:,} hours ({pct:.1f}%) "
                f"| M1 bars: {bars_total:,} "
                f"| ETA: {remaining_min:.0f} min",
                end="",
                flush=True,
            )
            last_progress = time.time()

        # Rate limiting — be polite to Dukascopy
        # ~50ms delay between requests (20 req/sec)
        time.sleep(0.05)

    print()  # newline after progress

    if not all_bars:
        print("No data downloaded!")
        return pd.DataFrame(columns=_COLUMNS)

    # Combine all bars
    print("Combining M1 bars...")
    df = pd.concat(all_bars, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Remove duplicates (overlapping hours)
    df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

    print(f"Total M1 bars: {len(df):,}")
    print(f"Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download XAUUSD M1 data from Dukascopy")
    parser.add_argument("--start", type=int, default=2016, help="Start year (default: 2016)")
    parser.add_argument("--end", type=int, default=2026, help="End year (default: 2026)")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol (default: XAUUSD)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching (re-download all)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: data/XAUUSD_M1.parquet)")
    args = parser.parse_args()

    df = download_m1_data(
        symbol=args.symbol,
        start_year=args.start,
        end_year=args.end,
        use_cache=not args.no_cache,
    )

    if df.empty:
        print("No data to save.")
        sys.exit(1)

    # Save as Parquet
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else DATA_DIR / f"{args.symbol}_M1.parquet"

    # Back up existing file
    if out_path.exists():
        backup = out_path.with_suffix(".parquet.bak")
        out_path.rename(backup)
        print(f"Backed up existing file to {backup}")

    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\nSaved {len(df):,} M1 bars to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
