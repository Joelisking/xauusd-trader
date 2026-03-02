"""Price & Technical features — 75 features from OHLCV data.

Computes EMAs, ATR, RSI, MACD, Bollinger Bands, ADX, Stochastic,
candlestick pattern encodings, market structure labels, and VWAP distance.
All functions operate on pandas DataFrames with standard OHLCV columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal, histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Bollinger band width and %B."""
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return width, pct_b


def calc_adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX, +DI, -DI."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)

    # Zero out when the other is larger
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calc_stochastic(df: pd.DataFrame, k_period: int = 5, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    lowest = df["low"].rolling(k_period).min()
    highest = df["high"].rolling(k_period).max()
    pct_k = 100 * (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)
    pct_d = pct_k.rolling(d_period).mean()
    return pct_k, pct_d


# ---------------------------------------------------------------------------
# Candlestick pattern encodings (numerical)
# ---------------------------------------------------------------------------

def _body(o: float, c: float) -> float:
    return abs(c - o)

def _range(h: float, l: float) -> float:
    return h - l


def encode_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode 14 candlestick patterns as numerical columns (0 or +-1).

    Returns DataFrame with 14 columns: pattern_hammer, pattern_inv_hammer,
    pattern_engulf_bull, pattern_engulf_bear, pattern_pin_bull, pattern_pin_bear,
    pattern_doji, pattern_shooting_star, pattern_morning_star, pattern_evening_star,
    pattern_three_white, pattern_three_black, pattern_tweezer_top, pattern_tweezer_bottom.
    """
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    patterns = np.zeros((n, 14), dtype=np.float32)

    for i in range(2, n):
        body_i = abs(c[i] - o[i])
        range_i = h[i] - l[i]
        if range_i == 0:
            continue
        body_pct = body_i / range_i

        bull = c[i] > o[i]
        bear = c[i] < o[i]
        upper_wick = h[i] - max(c[i], o[i])
        lower_wick = min(c[i], o[i]) - l[i]

        # Hammer (bullish): small body, long lower wick
        if body_pct < 0.35 and lower_wick > 2 * body_i and upper_wick < body_i:
            patterns[i, 0] = 1.0

        # Inverted Hammer: small body, long upper wick
        if body_pct < 0.35 and upper_wick > 2 * body_i and lower_wick < body_i:
            patterns[i, 1] = 1.0

        # Bullish Engulfing
        if i >= 1:
            prev_body = abs(c[i - 1] - o[i - 1])
            if c[i - 1] < o[i - 1] and bull and body_i > prev_body:
                patterns[i, 2] = 1.0

        # Bearish Engulfing
        if i >= 1:
            prev_body = abs(c[i - 1] - o[i - 1])
            if c[i - 1] > o[i - 1] and bear and body_i > prev_body:
                patterns[i, 3] = 1.0

        # Bullish Pin Bar
        if lower_wick > 2.5 * body_i and upper_wick < 0.3 * range_i:
            patterns[i, 4] = 1.0

        # Bearish Pin Bar
        if upper_wick > 2.5 * body_i and lower_wick < 0.3 * range_i:
            patterns[i, 5] = 1.0

        # Doji
        if body_pct < 0.1:
            patterns[i, 6] = 1.0

        # Shooting Star
        if body_pct < 0.3 and upper_wick > 2 * body_i and lower_wick < 0.2 * range_i and bear:
            patterns[i, 7] = 1.0

        # Morning Star (3-candle)
        if i >= 2:
            body_prev2 = abs(c[i - 2] - o[i - 2])
            body_prev1 = abs(c[i - 1] - o[i - 1])
            if (c[i - 2] < o[i - 2] and body_prev1 < 0.3 * body_prev2
                    and bull and body_i > 0.5 * body_prev2):
                patterns[i, 8] = 1.0

        # Evening Star (3-candle)
        if i >= 2:
            body_prev2 = abs(c[i - 2] - o[i - 2])
            body_prev1 = abs(c[i - 1] - o[i - 1])
            if (c[i - 2] > o[i - 2] and body_prev1 < 0.3 * body_prev2
                    and bear and body_i > 0.5 * body_prev2):
                patterns[i, 9] = 1.0

        # Three White Soldiers
        if i >= 2:
            if (c[i] > o[i] and c[i - 1] > o[i - 1] and c[i - 2] > o[i - 2]
                    and c[i] > c[i - 1] > c[i - 2]):
                patterns[i, 10] = 1.0

        # Three Black Crows
        if i >= 2:
            if (c[i] < o[i] and c[i - 1] < o[i - 1] and c[i - 2] < o[i - 2]
                    and c[i] < c[i - 1] < c[i - 2]):
                patterns[i, 11] = 1.0

        # Tweezer Top
        if i >= 1 and abs(h[i] - h[i - 1]) < 0.1 * range_i and bear:
            patterns[i, 12] = 1.0

        # Tweezer Bottom
        if i >= 1 and abs(l[i] - l[i - 1]) < 0.1 * range_i and bull:
            patterns[i, 13] = 1.0

    cols = [
        "pattern_hammer", "pattern_inv_hammer",
        "pattern_engulf_bull", "pattern_engulf_bear",
        "pattern_pin_bull", "pattern_pin_bear",
        "pattern_doji", "pattern_shooting_star",
        "pattern_morning_star", "pattern_evening_star",
        "pattern_three_white", "pattern_three_black",
        "pattern_tweezer_top", "pattern_tweezer_bottom",
    ]
    return pd.DataFrame(patterns, columns=cols, index=df.index)


# ---------------------------------------------------------------------------
# Market structure encoding
# ---------------------------------------------------------------------------


def encode_market_structure(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """Encode market structure as 0=HH, 1=HL, 2=LH, 3=LL based on recent swing points.

    Uses 5-bar fractal swing detection.
    """
    high, low = df["high"].values, df["low"].values
    n = len(df)
    result = np.full(n, np.nan, dtype=np.float32)

    swing_highs: list[float] = []
    swing_lows: list[float] = []

    for i in range(2, n - 2):
        # 5-bar fractal high
        if high[i] > high[i - 1] and high[i] > high[i - 2] and high[i] > high[i + 1] and high[i] > high[i + 2]:
            if swing_highs and high[i] > swing_highs[-1]:
                result[i] = 0  # HH
            elif swing_highs and high[i] < swing_highs[-1]:
                result[i] = 2  # LH
            swing_highs.append(high[i])
            if len(swing_highs) > lookback:
                swing_highs.pop(0)

        # 5-bar fractal low
        if low[i] < low[i - 1] and low[i] < low[i - 2] and low[i] < low[i + 1] and low[i] < low[i + 2]:
            if swing_lows and low[i] > swing_lows[-1]:
                result[i] = 1  # HL
            elif swing_lows and low[i] < swing_lows[-1]:
                result[i] = 3  # LL
            swing_lows.append(low[i])
            if len(swing_lows) > lookback:
                swing_lows.pop(0)

    return pd.Series(result, index=df.index, name="market_structure").ffill().fillna(0)


# ---------------------------------------------------------------------------
# VWAP distance
# ---------------------------------------------------------------------------


def calc_vwap_distance(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    """Price distance from session VWAP as fraction of ATR."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df.get("tick_volume", df.get("volume", pd.Series(np.ones(len(df)), index=df.index)))
    cum_vol = volume.cumsum()
    cum_tp_vol = (typical * volume).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    distance = (df["close"] - vwap) / atr.replace(0, np.nan)
    return distance.fillna(0)


# ---------------------------------------------------------------------------
# Main: compute all 75 price features
# ---------------------------------------------------------------------------


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 75 price & technical features from OHLCV data.

    Input: DataFrame with columns: open, high, low, close, tick_volume (or volume).
    Output: DataFrame with 75 feature columns, same index as input.
    """
    features = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- OHLCV normalized (4) ---
    features["return_1"] = close.pct_change()
    features["return_5"] = close.pct_change(5)
    features["return_10"] = close.pct_change(10)
    features["return_20"] = close.pct_change(20)

    # --- EMA values and deltas (8) ---
    for period in (8, 21, 50, 200):
        e = ema(close, period)
        features[f"ema_{period}_dist"] = (close - e) / e.replace(0, np.nan)
        features[f"ema_{period}_delta"] = e.pct_change(5)

    # --- ATR (2) ---
    atr_14 = calc_atr(df, 14)
    atr_21 = calc_atr(df, 21)
    features["atr_14"] = atr_14
    features["atr_21"] = atr_21

    # --- RSI (3) ---
    features["rsi_7"] = calc_rsi(close, 7)
    features["rsi_14"] = calc_rsi(close, 14)
    features["rsi_21"] = calc_rsi(close, 21)

    # --- MACD (4) ---
    macd_line, macd_signal, macd_hist = calc_macd(close)
    features["macd_line"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_histogram"] = macd_hist
    features["macd_hist_roc"] = macd_hist.diff()

    # --- Bollinger Bands (2) ---
    bb_width, bb_pct_b = calc_bollinger(close)
    features["bb_width"] = bb_width
    features["bb_pct_b"] = bb_pct_b

    # --- ADX (3) ---
    adx, plus_di, minus_di = calc_adx(df)
    features["adx_14"] = adx
    features["plus_di"] = plus_di
    features["minus_di"] = minus_di

    # --- Stochastic (2) ---
    stoch_k, stoch_d = calc_stochastic(df)
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_d

    # --- Candlestick patterns (14) ---
    patterns = encode_candle_patterns(df)
    for col in patterns.columns:
        features[col] = patterns[col].values

    # --- Market structure (4: M5-style structure + rolling stats) ---
    ms = encode_market_structure(df)
    features["market_structure"] = ms
    features["ms_hh_count"] = (ms == 0).rolling(20).sum()
    features["ms_ll_count"] = (ms == 3).rolling(20).sum()
    features["ms_trend_score"] = (features["ms_hh_count"] - features["ms_ll_count"]) / 20

    # --- VWAP distance (1) ---
    features["vwap_distance"] = calc_vwap_distance(df, atr_14)

    # --- Additional price features (28) ---
    features["high_low_range"] = (high - low) / close
    features["body_pct"] = (close - df["open"]).abs() / (high - low).replace(0, np.nan)
    features["upper_shadow"] = (high - pd.concat([close, df["open"]], axis=1).max(axis=1)) / (high - low).replace(0, np.nan)
    features["lower_shadow"] = (pd.concat([close, df["open"]], axis=1).min(axis=1) - low) / (high - low).replace(0, np.nan)
    features["close_position"] = (close - low) / (high - low).replace(0, np.nan)
    features["volatility_ratio"] = atr_14 / atr_21.replace(0, np.nan)
    features["price_momentum_5"] = close - close.shift(5)
    features["price_momentum_20"] = close - close.shift(20)
    features["rsi_divergence"] = features["rsi_14"].diff(5) - features["return_5"] * 100
    vol_series = df.get("tick_volume", df.get("volume", pd.Series(1, index=df.index)))
    features["volume_ratio"] = vol_series / vol_series.rolling(20).mean().replace(0, np.nan)
    features["atr_ratio"] = atr_14 / atr_14.rolling(50).mean().replace(0, np.nan)
    features["range_expansion"] = (high - low) / (high - low).rolling(20).mean().replace(0, np.nan)

    # EMA crossover signals
    ema_8 = ema(close, 8)
    ema_21 = ema(close, 21)
    ema_50 = ema(close, 50)
    features["ema_8_21_cross"] = (ema_8 - ema_21) / close
    features["ema_21_50_cross"] = (ema_21 - ema_50) / close
    features["ema_stack_bull"] = ((ema_8 > ema_21) & (ema_21 > ema_50)).astype(np.float32)
    features["ema_stack_bear"] = ((ema_8 < ema_21) & (ema_21 < ema_50)).astype(np.float32)

    # Price vs key EMAs
    features["close_vs_ema200"] = (close - ema(close, 200)) / close

    # Volatility contraction/expansion
    features["bb_squeeze"] = (bb_width < bb_width.rolling(50).quantile(0.2)).astype(np.float32)
    features["atr_contraction"] = (atr_14 < atr_14.rolling(50).quantile(0.2)).astype(np.float32)

    # Momentum divergence indicators
    features["rsi_14_slope"] = features["rsi_14"].diff(3)
    features["price_slope"] = close.pct_change(3) * 100
    features["momentum_divergence"] = features["rsi_14_slope"] - features["price_slope"]

    # High/low breakout signals
    features["high_20_break"] = (close > high.rolling(20).max().shift(1)).astype(np.float32)
    features["low_20_break"] = (close < low.rolling(20).min().shift(1)).astype(np.float32)

    # Candle body direction streak
    body_dir = (close > df["open"]).astype(float) - (close < df["open"]).astype(float)
    features["body_dir_streak"] = body_dir.rolling(5).sum()

    # Volume-price relationship
    features["volume_price_corr"] = close.pct_change().rolling(20).corr(vol_series.pct_change())

    # Trend strength from ADX components
    features["di_spread"] = (plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)

    # Relative close position within recent range
    features["close_in_20bar_range"] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min()).replace(0, np.nan)

    # Fill NaN from warmup period
    features = features.fillna(0)
    return features
