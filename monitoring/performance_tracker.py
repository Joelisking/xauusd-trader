"""Rolling performance metrics and report generation for the XAUUSD AI Trading System.

Stores trade records in memory (primary) and optionally persists to SQLite.
Thread-safe via asyncio.Lock.

Usage::

    tracker = PerformanceTracker()
    await tracker.record_trade({
        "symbol": "XAUUSD",
        "direction": "BUY",
        "bot": "scalper",
        "pnl_usd": 42.50,
        "pnl_pips": 8.3,
        "duration_min": 7,
        "exit_reason": "TP1",
        "ai_score": 76,
    })

    stats = tracker.get_daily_stats()
    report = tracker.generate_daily_report()
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai_server.config import LOG_DIR

logger = logging.getLogger(__name__)

# SQLite database path — sits inside the existing logs directory
PERFORMANCE_DB_PATH = LOG_DIR / "performance.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning default when denominator is zero."""
    return numerator / denominator if denominator else default


def _compute_max_drawdown(trades: list[dict[str, Any]]) -> float:
    """Compute max drawdown percentage from a running P&L equity curve.

    Iterates trades in order, tracking cumulative P&L.  The drawdown is
    measured from the highest equity peak reached so far.

    Args:
        trades: List of trade dicts each containing a 'pnl_usd' key.

    Returns:
        Maximum drawdown as a positive percentage (0-100).
    """
    if not trades:
        return 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for t in trades:
        equity += t.get("pnl_usd", 0.0)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Records trades and computes rolling performance metrics.

    Args:
        db_path:  Optional SQLite path for persistence.  If None, data
                  lives in memory only.  Pass a Path to enable persistence.
    """

    def __init__(self, db_path: Path | None = PERFORMANCE_DB_PATH) -> None:
        self._trades: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._db_path = db_path
        self._db_ready = False

        if self._db_path is not None:
            self._init_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the trades table if it doesn't already exist."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    symbol      TEXT    NOT NULL DEFAULT 'XAUUSD',
                    direction   TEXT    NOT NULL DEFAULT '',
                    bot         TEXT    NOT NULL DEFAULT '',
                    pnl_usd     REAL    NOT NULL DEFAULT 0.0,
                    pnl_pips    REAL    NOT NULL DEFAULT 0.0,
                    duration_min REAL   NOT NULL DEFAULT 0.0,
                    exit_reason TEXT    NOT NULL DEFAULT '',
                    ai_score    INTEGER NOT NULL DEFAULT 0,
                    lot         REAL    NOT NULL DEFAULT 0.0,
                    regime      TEXT    NOT NULL DEFAULT ''
                )
            """)
            conn.commit()
            conn.close()
            self._db_ready = True
        except sqlite3.Error as exc:
            logger.warning("SQLite init failed (%s) — running in memory-only mode", exc)
            self._db_ready = False

    def _persist_trade(self, trade: dict[str, Any]) -> None:
        """Insert a single trade record into SQLite (best-effort)."""
        if not self._db_ready or self._db_path is None:
            return
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(
                """
                INSERT INTO trades
                    (timestamp, symbol, direction, bot, pnl_usd, pnl_pips,
                     duration_min, exit_reason, ai_score, lot, regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.get("timestamp", _utcnow().isoformat()),
                    trade.get("symbol", "XAUUSD"),
                    trade.get("direction", ""),
                    trade.get("bot", ""),
                    float(trade.get("pnl_usd", 0.0)),
                    float(trade.get("pnl_pips", 0.0)),
                    float(trade.get("duration_min", 0.0)),
                    trade.get("exit_reason", ""),
                    int(trade.get("ai_score", 0)),
                    float(trade.get("lot", 0.0)),
                    trade.get("regime", ""),
                ),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Failed to persist trade to SQLite: %s", exc)

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    async def record_trade(self, trade_data: dict[str, Any]) -> None:
        """Record a completed trade.

        Args:
            trade_data: Dict with at minimum: pnl_usd, pnl_pips.
                        Optional fields: symbol, direction, bot, duration_min,
                        exit_reason, ai_score, lot, regime, timestamp.
        """
        async with self._lock:
            record: dict[str, Any] = {
                "timestamp": trade_data.get("timestamp", _utcnow().isoformat()),
                "symbol": trade_data.get("symbol", "XAUUSD"),
                "direction": trade_data.get("direction", ""),
                "bot": trade_data.get("bot", ""),
                "pnl_usd": float(trade_data.get("pnl_usd", 0.0)),
                "pnl_pips": float(trade_data.get("pnl_pips", 0.0)),
                "duration_min": float(trade_data.get("duration_min", 0.0)),
                "exit_reason": trade_data.get("exit_reason", ""),
                "ai_score": int(trade_data.get("ai_score", 0)),
                "lot": float(trade_data.get("lot", 0.0)),
                "regime": trade_data.get("regime", ""),
            }
            self._trades.append(record)

        self._persist_trade(record)
        logger.debug(
            "Trade recorded: %s %s P&L=%.2f",
            record["bot"],
            record["direction"],
            record["pnl_usd"],
        )

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def _trades_in_window(
        self,
        since: datetime,
        bot: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all trades within the given window, optionally filtered by bot."""
        result: list[dict[str, Any]] = []
        for t in self._trades:
            try:
                ts = datetime.fromisoformat(t["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError):
                continue
            if ts < since:
                continue
            if bot is not None and t.get("bot") != bot:
                continue
            result.append(t)
        return result

    @staticmethod
    def _compute_max_drawdown(trades: list[dict[str, Any]]) -> float:
        """Compute max drawdown percentage from a running equity curve.

        Delegates to the module-level ``_compute_max_drawdown`` function.
        """
        return _compute_max_drawdown(trades)

    def _compute_stats(
        self, trades: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute standard metrics from a list of trade records."""
        if not trades:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "avg_duration_min": 0.0,
                "best_trade_usd": 0.0,
                "worst_trade_usd": 0.0,
                "avg_win_usd": 0.0,
                "avg_loss_usd": 0.0,
                "profit_factor": 0.0,
            }

        wins = [t for t in trades if t.get("pnl_usd", 0.0) > 0]
        losses = [t for t in trades if t.get("pnl_usd", 0.0) <= 0]

        total_pnl = sum(t.get("pnl_usd", 0.0) for t in trades)
        gross_profit = sum(t.get("pnl_usd", 0.0) for t in wins)
        gross_loss = abs(sum(t.get("pnl_usd", 0.0) for t in losses))
        avg_duration = _safe_divide(
            sum(t.get("duration_min", 0.0) for t in trades),
            len(trades),
        )
        pnl_values = [t.get("pnl_usd", 0.0) for t in trades]

        return {
            "trade_count": len(trades),
            "win_rate": _safe_divide(len(wins), len(trades)) * 100.0,
            "total_pnl": total_pnl,
            "max_drawdown": self._compute_max_drawdown(trades),
            "avg_duration_min": avg_duration,
            "best_trade_usd": max(pnl_values) if pnl_values else 0.0,
            "worst_trade_usd": min(pnl_values) if pnl_values else 0.0,
            "avg_win_usd": _safe_divide(gross_profit, len(wins)),
            "avg_loss_usd": _safe_divide(gross_loss, len(losses)),
            "profit_factor": _safe_divide(gross_profit, gross_loss),
        }

    # ------------------------------------------------------------------
    # Public statistics API
    # ------------------------------------------------------------------

    def get_daily_stats(
        self,
        date: datetime | None = None,
        bot: str | None = None,
    ) -> dict[str, Any]:
        """Return statistics for the given calendar day (UTC).

        Args:
            date:  The day to query.  Defaults to today UTC.
            bot:   Filter by bot name ('scalper' / 'swing' / None = all).

        Returns:
            Dict with: trades_today, win_rate, total_pnl, max_drawdown,
            avg_duration_min, best_trade_usd, worst_trade_usd.
        """
        if date is None:
            date = _utcnow()

        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        if day_start.tzinfo is None:
            day_start = day_start.replace(tzinfo=timezone.utc)

        trades = self._trades_in_window(day_start, bot=bot)
        stats = self._compute_stats(trades)

        # Rename trade_count to trades_today for clarity in daily context
        stats["trades_today"] = stats.pop("trade_count")

        # Separate scalper/swing counts for the all-bots query
        if bot is None:
            stats["scalper_trades"] = sum(
                1 for t in trades if t.get("bot") == "scalper"
            )
            stats["swing_trades"] = sum(
                1 for t in trades if t.get("bot") == "swing"
            )
        return stats

    def get_weekly_stats(
        self,
        end_date: datetime | None = None,
        bot: str | None = None,
    ) -> dict[str, Any]:
        """Return rolling 7-day statistics.

        Args:
            end_date:  End of the 7-day window (defaults to now UTC).
            bot:       Filter by bot name.

        Returns:
            Dict with same keys as get_daily_stats(), plus a 'period_days' key.
        """
        if end_date is None:
            end_date = _utcnow()

        week_start = end_date - timedelta(days=7)

        trades = self._trades_in_window(week_start, bot=bot)
        stats = self._compute_stats(trades)
        stats["trade_count"] = stats.pop("trade_count", len(trades))
        stats["period_days"] = 7

        if bot is None:
            stats["scalper_trades"] = sum(
                1 for t in trades if t.get("bot") == "scalper"
            )
            stats["swing_trades"] = sum(
                1 for t in trades if t.get("bot") == "swing"
            )
        return stats

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_daily_report(self, date: datetime | None = None) -> str:
        """Generate a formatted daily performance report string for Telegram.

        Args:
            date: The day to report on.  Defaults to today UTC.

        Returns:
            Multi-line string ready to be sent as a Telegram message.
        """
        if date is None:
            date = _utcnow()

        stats = self.get_daily_stats(date)
        date_str = date.strftime("%Y-%m-%d")
        total_pnl = stats.get("total_pnl", 0.0)
        result_icon = "PROFIT" if total_pnl >= 0 else "LOSS"

        lines = [
            f"[DAILY REPORT] {date_str}  {result_icon}",
            "",
            f"Trades:       {stats.get('trades_today', 0)}",
            f"Win Rate:     {stats.get('win_rate', 0.0):.1f}%",
            f"Total P&L:    ${total_pnl:+.2f}",
            f"Max Drawdown: {stats.get('max_drawdown', 0.0):.1f}%",
            f"Avg Duration: {stats.get('avg_duration_min', 0.0):.0f} min",
        ]

        scalper = stats.get("scalper_trades")
        swing = stats.get("swing_trades")
        if scalper is not None:
            lines.append(f"Scalper:      {scalper} trades")
        if swing is not None:
            lines.append(f"Swing:        {swing} trades")

        best = stats.get("best_trade_usd")
        worst = stats.get("worst_trade_usd")
        if best is not None and stats.get("trades_today", 0) > 0:
            lines.append(f"Best trade:   ${best:+.2f}")
        if worst is not None and stats.get("trades_today", 0) > 0:
            lines.append(f"Worst trade:  ${worst:+.2f}")

        pf = stats.get("profit_factor", 0.0)
        if pf > 0:
            lines.append(f"Profit Factor:{pf:.2f}")

        return "\n".join(lines)

    def generate_weekly_report(
        self,
        end_date: datetime | None = None,
        model_stats: dict[str, Any] | None = None,
    ) -> str:
        """Generate a formatted weekly performance report string.

        Args:
            end_date:     End date for the 7-day window.  Defaults to now.
            model_stats:  Optional dict with AI model metrics:
                          scalper_auc, swing_auc, scalper_accuracy,
                          swing_accuracy, feature_drift.

        Returns:
            Multi-line string ready for Telegram.
        """
        if end_date is None:
            end_date = _utcnow()

        stats = self.get_weekly_stats(end_date)
        week_end_str = end_date.strftime("%Y-%m-%d")
        total_pnl = stats.get("total_pnl", 0.0)
        result_icon = "PROFIT" if total_pnl >= 0 else "LOSS"

        lines = [
            f"[WEEKLY REPORT] w/e {week_end_str}  {result_icon}",
            "",
            "-- Trading Performance --",
            f"Trades:       {stats.get('trade_count', 0)}",
            f"Win Rate:     {stats.get('win_rate', 0.0):.1f}%",
            f"Total P&L:    ${total_pnl:+.2f}",
            f"Max Drawdown: {stats.get('max_drawdown', 0.0):.1f}%",
            f"Profit Factor:{stats.get('profit_factor', 0.0):.2f}",
        ]

        scalper = stats.get("scalper_trades")
        swing = stats.get("swing_trades")
        if scalper is not None:
            lines.append(f"Scalper:      {scalper} trades")
        if swing is not None:
            lines.append(f"Swing:        {swing} trades")

        if model_stats:
            lines.append("")
            lines.append("-- Model Performance --")
            scalper_auc = model_stats.get("scalper_auc")
            swing_auc = model_stats.get("swing_auc")
            scalper_acc = model_stats.get("scalper_accuracy")
            swing_acc = model_stats.get("swing_accuracy")
            drift = model_stats.get("feature_drift")

            if scalper_auc is not None:
                status = "OK" if scalper_auc >= 0.70 else "BELOW TARGET"
                lines.append(f"Scalper AUC:  {scalper_auc:.3f} [{status}]")
            if swing_auc is not None:
                status = "OK" if swing_auc >= 0.68 else "BELOW TARGET"
                lines.append(f"Swing AUC:    {swing_auc:.3f} [{status}]")
            if scalper_acc is not None:
                lines.append(f"Scalper Acc:  {scalper_acc:.1f}%")
            if swing_acc is not None:
                lines.append(f"Swing Acc:    {swing_acc:.1f}%")
            if drift is not None:
                drift_status = "HIGH - retrain recommended" if drift > 0.15 else "OK"
                lines.append(f"Feature Drift:{drift:.3f} [{drift_status}]")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def trade_count(self) -> int:
        """Total number of trades recorded in memory."""
        return len(self._trades)

    def clear(self) -> None:
        """Clear all in-memory trade records.  Does not affect the database."""
        self._trades.clear()
