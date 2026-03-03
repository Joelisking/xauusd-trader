"""Stress testing on extreme historical periods.

Tests strategy performance during: 2008 crisis, 2011 gold peak,
2020 COVID crash, 2022 rate hike cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


# Extreme periods for gold (dates in YYYY-MM-DD format)
STRESS_PERIODS: dict[str, dict[str, str]] = {
    "2008_financial_crisis": {
        "start": "2008-09-01",
        "end": "2009-03-31",
        "description": "Lehman collapse, extreme volatility, gold safe-haven surge",
    },
    "2011_gold_peak": {
        "start": "2011-07-01",
        "end": "2011-12-31",
        "description": "Gold peaks at $1920, sharp reversal, high volatility",
    },
    "2020_covid_crash": {
        "start": "2020-02-15",
        "end": "2020-06-30",
        "description": "COVID crash, liquidity crisis, gold whipsaw then surge",
    },
    "2022_rate_hikes": {
        "start": "2022-03-01",
        "end": "2022-11-30",
        "description": "Aggressive Fed rate hikes, USD surge, gold decline",
    },
}


@dataclass
class StressTestResult:
    """Results for a single stress period."""
    period_name: str
    description: str
    start_date: str
    end_date: str
    n_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    profit_factor: float
    avg_trade_pnl: float
    max_consecutive_losses: int
    passed: bool  # Meets minimum thresholds

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.period_name} ({self.start_date} to {self.end_date}):\n"
            f"  {self.description}\n"
            f"  Trades: {self.n_trades}, Win Rate: {self.win_rate:.1%}\n"
            f"  P&L: {self.total_pnl:.2f}, Max DD: {self.max_drawdown:.2%}\n"
            f"  Profit Factor: {self.profit_factor:.2f}, Avg Trade: {self.avg_trade_pnl:.2f}\n"
            f"  Max Consecutive Losses: {self.max_consecutive_losses}"
        )


def run_stress_test(
    trade_results: pd.DataFrame,
    period_name: str,
    min_profit_factor: float = 1.0,
    max_drawdown_pct: float = 0.20,
) -> StressTestResult:
    """Run stress test on trade results for a specific period.

    Args:
        trade_results: DataFrame with columns: time, pnl, direction, duration
        period_name: Key from STRESS_PERIODS
        min_profit_factor: Minimum PF to pass (relaxed during stress)
        max_drawdown_pct: Maximum drawdown to pass

    Returns:
        StressTestResult
    """
    period = STRESS_PERIODS.get(period_name, {})

    if trade_results.empty:
        return StressTestResult(
            period_name=period_name,
            description=period.get("description", ""),
            start_date=period.get("start", ""),
            end_date=period.get("end", ""),
            n_trades=0, win_rate=0, total_pnl=0, max_drawdown=0,
            profit_factor=0, avg_trade_pnl=0, max_consecutive_losses=0,
            passed=False,
        )

    pnls = trade_results["pnl"].values

    # Win rate
    wins = (pnls > 0).sum()
    win_rate = wins / len(pnls) if len(pnls) > 0 else 0

    # Total P&L
    total_pnl = float(pnls.sum())

    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / np.maximum(running_max, 1)
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0

    # Profit factor
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max consecutive losses
    max_consec = _max_consecutive_losses(pnls)

    passed = (
        profit_factor >= min_profit_factor
        and max_dd <= max_drawdown_pct
        and len(pnls) >= 5  # Need enough trades for meaningful test
    )

    return StressTestResult(
        period_name=period_name,
        description=period.get("description", ""),
        start_date=period.get("start", ""),
        end_date=period.get("end", ""),
        n_trades=len(pnls),
        win_rate=win_rate,
        total_pnl=total_pnl,
        max_drawdown=max_dd,
        profit_factor=float(profit_factor),
        avg_trade_pnl=float(pnls.mean()),
        max_consecutive_losses=max_consec,
        passed=passed,
    )


def run_all_stress_tests(
    all_trades: pd.DataFrame,
    bot_type: str = "scalper",
) -> list[StressTestResult]:
    """Run stress tests across all extreme periods.

    Args:
        all_trades: DataFrame with columns: time, pnl, direction, duration
        bot_type: "scalper" or "swing" — adjusts thresholds

    Returns:
        List of StressTestResult for each period
    """
    if bot_type == "scalper":
        min_pf, max_dd = 1.0, 0.20  # Relaxed during stress
    else:
        min_pf, max_dd = 1.0, 0.25

    results = []
    for period_name, period_info in STRESS_PERIODS.items():
        # Filter trades to this period
        if "time" in all_trades.columns:
            mask = (
                (all_trades["time"] >= period_info["start"]) &
                (all_trades["time"] <= period_info["end"])
            )
            period_trades = all_trades[mask]
        else:
            period_trades = pd.DataFrame()

        result = run_stress_test(period_trades, period_name, min_pf, max_dd)
        results.append(result)

    return results


def _max_consecutive_losses(pnls: np.ndarray) -> int:
    """Count maximum consecutive losing trades."""
    max_streak = 0
    current_streak = 0
    for pnl in pnls:
        if pnl < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak
