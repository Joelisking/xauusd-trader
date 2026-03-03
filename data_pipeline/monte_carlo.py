"""Monte Carlo simulation for worst-case drawdown estimation.

Randomizes trade order 1000x to estimate the distribution of maximum
drawdown and other risk metrics.  Target: 95th percentile DD < 18%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    n_trades: int
    mean_max_dd: float
    median_max_dd: float
    p95_max_dd: float
    p99_max_dd: float
    worst_max_dd: float
    mean_final_pnl: float
    p5_final_pnl: float
    max_dd_distribution: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def summary(self) -> str:
        return (
            f"Monte Carlo ({self.n_simulations} sims, {self.n_trades} trades):\n"
            f"  Max DD — Mean: {self.mean_max_dd:.2%}, Median: {self.median_max_dd:.2%}\n"
            f"  Max DD — P95: {self.p95_max_dd:.2%}, P99: {self.p99_max_dd:.2%}\n"
            f"  Max DD — Worst: {self.worst_max_dd:.2%}\n"
            f"  Final P&L — Mean: {self.mean_final_pnl:.2f}, P5: {self.p5_final_pnl:.2f}\n"
            f"  Target P95 < 18%: {'PASS' if self.p95_max_dd < 0.18 else 'FAIL'}"
        )

    @property
    def passes_target(self) -> bool:
        return self.p95_max_dd < 0.18


def run_monte_carlo(
    trade_pnls: np.ndarray,
    n_simulations: int = 1000,
    initial_balance: float = 10000.0,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation by shuffling trade order.

    Args:
        trade_pnls: Array of individual trade P&L values (absolute, not %)
        n_simulations: Number of random permutations
        initial_balance: Starting account balance
        seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with drawdown distribution
    """
    rng = np.random.RandomState(seed)
    n_trades = len(trade_pnls)
    max_dds = np.zeros(n_simulations)
    final_pnls = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Shuffle trade order
        shuffled = rng.permutation(trade_pnls)

        # Compute equity curve
        equity = initial_balance + np.cumsum(shuffled)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max

        max_dds[i] = drawdowns.max()
        final_pnls[i] = equity[-1] - initial_balance

    return MonteCarloResult(
        n_simulations=n_simulations,
        n_trades=n_trades,
        mean_max_dd=float(np.mean(max_dds)),
        median_max_dd=float(np.median(max_dds)),
        p95_max_dd=float(np.percentile(max_dds, 95)),
        p99_max_dd=float(np.percentile(max_dds, 99)),
        worst_max_dd=float(np.max(max_dds)),
        mean_final_pnl=float(np.mean(final_pnls)),
        p5_final_pnl=float(np.percentile(final_pnls, 5)),
        max_dd_distribution=max_dds,
    )
