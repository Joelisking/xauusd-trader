"""Tests for backtesting framework — Phase 13."""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.monte_carlo import run_monte_carlo, MonteCarloResult
from data_pipeline.stress_test import (
    run_stress_test,
    run_all_stress_tests,
    _max_consecutive_losses,
    STRESS_PERIODS,
)
from data_pipeline.backtest_analyzer import analyze_trades, BacktestMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades(n: int = 200, win_rate: float = 0.6, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic trade results."""
    rng = np.random.RandomState(seed)
    wins = rng.uniform(10, 50, int(n * win_rate))
    losses = -rng.uniform(10, 30, n - int(n * win_rate))
    pnls = np.concatenate([wins, losses])
    rng.shuffle(pnls)

    times = pd.date_range("2023-01-01", periods=n, freq="1h")
    durations = rng.uniform(3, 60, n)

    return pd.DataFrame({
        "time": times,
        "pnl": pnls,
        "direction": rng.choice(["BUY", "SELL"], n),
        "duration_min": durations,
    })


# ---------------------------------------------------------------------------
# Monte Carlo tests
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_basic_run(self):
        pnls = np.array([10, -5, 15, -8, 20, -3, 12, -6, 8, -4], dtype=float)
        result = run_monte_carlo(pnls, n_simulations=100, seed=42)
        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 100
        assert result.n_trades == 10
        assert result.mean_max_dd >= 0

    def test_drawdown_range(self):
        pnls = np.array([10, -5, 15, -8, 20, -3] * 20, dtype=float)
        result = run_monte_carlo(pnls, n_simulations=500, seed=42)
        assert 0 <= result.p95_max_dd <= 1
        assert result.p95_max_dd >= result.mean_max_dd
        assert result.p99_max_dd >= result.p95_max_dd

    def test_profitable_trades(self):
        pnls = np.array([10, 15, 20, -5, -3] * 40, dtype=float)
        result = run_monte_carlo(pnls, n_simulations=200, seed=42)
        assert result.mean_final_pnl > 0

    def test_summary_string(self):
        pnls = np.array([10, -5, 15, -8, 20, -3], dtype=float)
        result = run_monte_carlo(pnls, n_simulations=50, seed=42)
        summary = result.summary()
        assert "Monte Carlo" in summary
        assert "Max DD" in summary

    def test_distribution_shape(self):
        pnls = np.array([10, -5, 15, -8], dtype=float)
        result = run_monte_carlo(pnls, n_simulations=100, seed=42)
        assert len(result.max_dd_distribution) == 100


# ---------------------------------------------------------------------------
# Stress test tests
# ---------------------------------------------------------------------------


class TestStressTest:
    def test_run_with_trades(self):
        trades = _make_trades(50)
        result = run_stress_test(trades, "2020_covid_crash")
        assert result.n_trades == 50
        assert 0 <= result.win_rate <= 1
        assert isinstance(result.profit_factor, float)

    def test_run_empty_trades(self):
        trades = pd.DataFrame()
        result = run_stress_test(trades, "2020_covid_crash")
        assert result.n_trades == 0
        assert result.passed is False

    def test_all_stress_periods_defined(self):
        assert len(STRESS_PERIODS) >= 4
        for name, info in STRESS_PERIODS.items():
            assert "start" in info
            assert "end" in info
            assert "description" in info

    def test_max_consecutive_losses(self):
        pnls = np.array([10, -5, -3, -8, 20, -2, -1, 15])
        assert _max_consecutive_losses(pnls) == 3

    def test_max_consecutive_no_losses(self):
        pnls = np.array([10, 5, 3, 8])
        assert _max_consecutive_losses(pnls) == 0

    def test_summary_format(self):
        trades = _make_trades(30)
        result = run_stress_test(trades, "2008_financial_crisis")
        summary = result.summary()
        assert "2008" in summary


# ---------------------------------------------------------------------------
# Backtest analyzer tests
# ---------------------------------------------------------------------------


class TestBacktestAnalyzer:
    def test_analyze_trades_basic(self):
        trades = _make_trades(100)
        metrics = analyze_trades(trades, bot_type="scalper")
        assert metrics.total_trades == 100
        assert metrics.winning_trades + metrics.losing_trades == 100
        assert 0 <= metrics.win_rate <= 1
        assert metrics.max_drawdown_pct >= 0

    def test_equity_curve(self):
        trades = _make_trades(50)
        metrics = analyze_trades(trades)
        assert len(metrics.equity_curve) == 50
        assert metrics.equity_curve[0] != 0  # Should start near initial balance

    def test_profit_factor(self):
        # All winning trades
        trades = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=5, freq="1h"),
            "pnl": [10, 20, 15, 30, 25],
        })
        metrics = analyze_trades(trades)
        assert metrics.profit_factor == float("inf")

    def test_empty_trades(self):
        trades = pd.DataFrame()
        metrics = analyze_trades(trades)
        assert metrics.total_trades == 0
        assert metrics.final_balance == metrics.initial_balance

    def test_check_targets_scalper(self):
        trades = _make_trades(200, win_rate=0.6)
        metrics = analyze_trades(trades, bot_type="scalper")
        targets = metrics.check_targets()
        assert "profit_factor" in targets
        assert "max_drawdown" in targets
        assert "win_rate" in targets

    def test_summary_format(self):
        trades = _make_trades(100)
        metrics = analyze_trades(trades)
        summary = metrics.summary()
        assert "Backtest" in summary
        assert "Win Rate" in summary
        assert "Drawdown" in summary

    def test_sharpe_ratio(self):
        trades = _make_trades(200)
        metrics = analyze_trades(trades)
        # Sharpe should be a reasonable number
        assert -10 < metrics.sharpe_ratio < 20

    def test_durations(self):
        trades = _make_trades(50)
        metrics = analyze_trades(trades)
        assert metrics.avg_trade_duration_min > 0
        assert metrics.max_trade_duration_min >= metrics.avg_trade_duration_min
