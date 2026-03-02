"""Macro & Correlation features — 20 features.

DXY, real yield, VIX, oil, and economic calendar flags.
These features come from external data sources (FRED, Alpha Vantage, ForexFactory).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MacroContext:
    """Container for current macro data passed from external feeds."""

    dxy_price: float = 0.0
    dxy_ema_50: float = 0.0
    dxy_momentum_roc: float = 0.0
    dxy_direction: str = "NEUTRAL"  # UP, DOWN, NEUTRAL

    real_yield: float = 0.0
    real_yield_direction: str = "NEUTRAL"
    nominal_10y: float = 0.0

    vix: float = 0.0
    vix_5d_roc: float = 0.0

    oil_direction: str = "NEUTRAL"

    event_within_1h: bool = False
    event_within_4h: bool = False
    event_within_24h: bool = False
    next_event_impact: int = 0  # 0-3
    news_risk_score: int = 0  # 0-100


def _direction_to_num(direction: str) -> float:
    """Convert direction string to numerical value."""
    d = direction.upper()
    if d == "UP":
        return 1.0
    elif d == "DOWN":
        return -1.0
    return 0.0


def compute_macro_features(
    n_rows: int,
    macro: MacroContext | None = None,
) -> pd.DataFrame:
    """Compute 20 macro & correlation features.

    These are "static" for a given point in time — the same macro context
    applies to all rows in a batch. For real-time serving, n_rows=1.
    For batch/training, macro values are looked up per-row externally.
    """
    if macro is None:
        macro = MacroContext()

    features = pd.DataFrame(index=range(n_rows))

    # --- DXY features (5) ---
    features["dxy_direction"] = _direction_to_num(macro.dxy_direction)
    features["dxy_dist_ema50"] = (
        (macro.dxy_price - macro.dxy_ema_50) / max(macro.dxy_ema_50, 1)
        if macro.dxy_ema_50 else 0.0
    )
    features["dxy_momentum"] = macro.dxy_momentum_roc / 100.0  # normalize
    features["dxy_price_norm"] = macro.dxy_price / 100.0  # rough normalization
    features["dxy_strength"] = abs(macro.dxy_momentum_roc) / 100.0

    # --- Real Yield features (3) ---
    features["real_yield"] = macro.real_yield
    features["real_yield_direction"] = _direction_to_num(macro.real_yield_direction)
    features["nominal_10y"] = macro.nominal_10y

    # --- VIX features (3) ---
    features["vix_level"] = macro.vix / 100.0  # normalize
    features["vix_5d_roc"] = macro.vix_5d_roc / 100.0
    features["vix_regime"] = np.where(macro.vix > 25, 1.0, np.where(macro.vix > 18, 0.5, 0.0))

    # --- Oil features (1) ---
    features["oil_direction"] = _direction_to_num(macro.oil_direction)

    # --- Calendar event features (4) ---
    features["event_within_1h"] = float(macro.event_within_1h)
    features["event_within_4h"] = float(macro.event_within_4h)
    features["event_within_24h"] = float(macro.event_within_24h)
    features["next_event_impact"] = macro.next_event_impact / 3.0  # normalize 0-1

    # --- Composite macro features (4) ---
    features["news_risk_score"] = macro.news_risk_score / 100.0

    # Gold-DXY inverse correlation strength
    features["gold_dxy_inverse"] = -features["dxy_direction"]

    # Risk-off composite: high VIX + rising real yield = bearish gold
    features["risk_off_signal"] = (
        features["vix_level"] * 0.5 +
        features["real_yield_direction"] * 0.3 +
        features["dxy_direction"] * 0.2
    )

    # Macro alignment score: how aligned are macro factors for gold direction
    features["macro_alignment"] = (
        -features["dxy_direction"] +  # DXY down = gold up
        -features["real_yield_direction"] +  # yield down = gold up
        features["vix_level"] * 2  # high VIX = gold up (safe haven)
    ) / 4.0

    features = features.fillna(0)
    return features
