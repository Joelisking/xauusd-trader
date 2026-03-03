"""Backtest result analyzer — parse MT5 results, generate equity curves and metrics.

Parses MT5 Strategy Tester HTML/CSV output and computes standardized
performance metrics for comparison against target thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Standardized backtest performance metrics."""
    bot_type: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Duration
    avg_trade_duration_min: float = 0.0
    max_trade_duration_min: float = 0.0

    # Equity curve
    equity_curve: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            f"=== {self.bot_type} Backtest: {self.start_date} to {self.end_date} ===",
            f"  Balance: {self.initial_balance:.0f} -> {self.final_balance:.0f}",
            f"  Trades: {self.total_trades} (W:{self.winning_trades} L:{self.losing_trades})",
            f"  Win Rate: {self.win_rate:.1%}",
            f"  Profit Factor: {self.profit_factor:.2f}",
            f"  Total P&L: {self.total_pnl:.2f}",
            f"  Avg Win: {self.avg_win:.2f}, Avg Loss: {self.avg_loss:.2f}",
            f"  Max Drawdown: {self.max_drawdown_pct:.2%} ({self.max_drawdown_abs:.2f})",
            f"  Sharpe: {self.sharpe_ratio:.2f}, Sortino: {self.sortino_ratio:.2f}",
            f"  Avg Duration: {self.avg_trade_duration_min:.0f} min",
        ]
        return "\n".join(lines)

    def check_targets(self) -> dict[str, bool]:
        """Check against target thresholds from architecture spec."""
        if self.bot_type == "scalper":
            return {
                "profit_factor": self.profit_factor > 1.5,
                "max_drawdown": self.max_drawdown_pct < 0.12,
                "win_rate": self.win_rate > 0.55,
            }
        else:  # swing
            return {
                "profit_factor": self.profit_factor > 1.7,
                "max_drawdown": self.max_drawdown_pct < 0.12,
                "win_rate": self.win_rate > 0.52,
            }


def analyze_trades(
    trades: pd.DataFrame,
    bot_type: str = "scalper",
    initial_balance: float = 10000.0,
) -> BacktestMetrics:
    """Analyze a DataFrame of trade results.

    Args:
        trades: DataFrame with columns: time, pnl, direction, duration_min (optional)
        bot_type: "scalper" or "swing"
        initial_balance: Starting balance

    Returns:
        BacktestMetrics with all computed metrics
    """
    if trades.empty or "pnl" not in trades.columns:
        return BacktestMetrics(
            bot_type=bot_type, start_date="", end_date="",
            initial_balance=initial_balance, final_balance=initial_balance,
        )

    pnls = trades["pnl"].values.astype(float)
    n = len(pnls)

    # Basic stats
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0

    # Equity curve
    equity = initial_balance + np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity)
    drawdown_pcts = drawdowns / np.maximum(running_max, 1)

    # Sharpe ratio (annualized, assuming ~252 trading days)
    daily_returns = pnls / initial_balance
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino ratio (only downside deviation)
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Calmar ratio
    max_dd_pct = float(drawdown_pcts.max()) if len(drawdown_pcts) > 0 else 0.0
    annual_return = (equity[-1] / initial_balance - 1) if n > 0 else 0.0
    calmar = annual_return / max_dd_pct if max_dd_pct > 0 else 0.0

    # Duration
    durations = trades.get("duration_min", pd.Series(dtype=float))
    avg_duration = float(durations.mean()) if not durations.empty else 0.0
    max_duration = float(durations.max()) if not durations.empty else 0.0

    # Dates
    start_date = str(trades["time"].iloc[0]) if "time" in trades.columns and n > 0 else ""
    end_date = str(trades["time"].iloc[-1]) if "time" in trades.columns and n > 0 else ""

    return BacktestMetrics(
        bot_type=bot_type,
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        final_balance=float(equity[-1]),
        total_trades=n,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / n if n > 0 else 0.0,
        total_pnl=float(pnls.sum()),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        avg_trade_pnl=float(pnls.mean()),
        avg_win=float(wins.mean()) if len(wins) > 0 else 0.0,
        avg_loss=float(losses.mean()) if len(losses) > 0 else 0.0,
        largest_win=float(wins.max()) if len(wins) > 0 else 0.0,
        largest_loss=float(losses.min()) if len(losses) > 0 else 0.0,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_abs=float(drawdowns.max()) if len(drawdowns) > 0 else 0.0,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
        avg_trade_duration_min=avg_duration,
        max_trade_duration_min=max_duration,
        equity_curve=equity,
    )


def parse_mt5_csv(csv_path: Path) -> pd.DataFrame:
    """Parse MT5 Strategy Tester CSV export into standardized DataFrame.

    MT5 CSV format varies by locale; this handles the common format:
    Deal, Open Time, Type, Size, Symbol, Price, S/L, T/P, Commission, Swap, Profit
    """
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path, sep="\t", encoding="utf-16-le", on_bad_lines="skip")

    # Standardize column names
    col_map = {
        "Open Time": "time",
        "Profit": "pnl",
        "Type": "direction",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Filter to closed trades only (exclude balance operations)
    if "direction" in df.columns:
        df = df[df["direction"].isin(["buy", "sell", "Buy", "Sell"])]

    return df
