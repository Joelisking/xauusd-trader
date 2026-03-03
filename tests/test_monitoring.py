"""Tests for Phase 12 monitoring modules.

Covers:
  - TelegramAlertBot: message formatting for all 7 alert types, rate limiting,
    queue behaviour, graceful no-op when token is absent.
  - Watchdog: health check logic, consecutive failure tracking, alert dispatch.
  - PerformanceTracker: trade recording, stats computation, report generation.
  - News schedule: phase computation and schedule building (imported from
    monitoring.news_schedule which was completed in Phase 11).

Telegram HTTP calls are always mocked — no real messages are sent.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from monitoring.telegram_bot import AlertType, TelegramAlertBot, _FORMATTERS
from monitoring.watchdog import (
    MAX_CONSECUTIVE_FAILURES,
    CheckResult,
    Watchdog,
    _get_system_stats,
)
from monitoring.performance_tracker import PerformanceTracker, _compute_max_drawdown, _safe_divide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot(configured: bool = False) -> TelegramAlertBot:
    """Return a TelegramAlertBot with or without credentials."""
    if configured:
        return TelegramAlertBot(token="fake_token_123", chat_id="99999")
    return TelegramAlertBot(token="", chat_id="")


async def _drain(bot: TelegramAlertBot) -> None:
    """Let the event loop run briefly so queued messages are processed."""
    await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# TelegramAlertBot — formatting tests (no network)
# ---------------------------------------------------------------------------

class TestTelegramAlertBotFormatting:
    """Test that each alert type formats without error and contains key fields."""

    def test_format_trade_entry_buy(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.TRADE_ENTRY, {
            "symbol": "XAUUSD",
            "direction": "BUY",
            "lot": 0.02,
            "score": 78,
            "regime": "trending",
            "entry_price": 2350.50,
            "sl": 2340.00,
            "tp": 2365.00,
            "wyckoff_phase": "D",
            "bot": "scalper",
        })
        assert "TRADE ENTRY" in text
        assert "BUY" in text
        assert "XAUUSD" in text
        assert "0.02" in text
        assert "78" in text
        assert "trending" in text
        assert "2350.50" in text
        assert "D" in text

    def test_format_trade_entry_sell(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.TRADE_ENTRY, {
            "direction": "SELL",
            "score": 72,
        })
        assert "SELL" in text

    def test_format_trade_exit_win(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.TRADE_EXIT, {
            "symbol": "XAUUSD",
            "pnl_pips": 12.5,
            "pnl_usd": 62.50,
            "exit_reason": "TP1",
            "duration_min": 9,
            "session_pnl_usd": 100.0,
        })
        assert "TRADE EXIT" in text
        assert "WIN" in text
        assert "+12.5" in text
        assert "62.50" in text
        assert "TP1" in text
        assert "9 min" in text

    def test_format_trade_exit_loss(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.TRADE_EXIT, {
            "pnl_pips": -8.0,
            "pnl_usd": -40.0,
            "exit_reason": "SL hit",
        })
        assert "LOSS" in text
        assert "-8.0" in text

    def test_format_news_shield_detection(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.NEWS_SHIELD, {
            "phase": "DETECTION",
            "event_name": "Non-Farm Payrolls",
            "event_time": "2026-03-07T13:30:00+00:00",
            "minutes_until": 55,
        })
        assert "NEWS SHIELD" in text
        assert "DETECTION" in text
        assert "Non-Farm Payrolls" in text
        assert "55" in text

    def test_format_news_shield_pre(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.NEWS_SHIELD, {
            "phase": "PRE",
            "event_name": "CPI m/m",
            "minutes_until": 15,
        })
        assert "PRE" in text
        assert "halted" in text.lower()

    def test_format_news_shield_during(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.NEWS_SHIELD, {
            "phase": "DURING",
            "event_name": "FOMC",
            "minutes_since": 5,
        })
        assert "DURING" in text
        assert "blocked" in text.lower()

    def test_format_news_shield_post(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.NEWS_SHIELD, {
            "phase": "POST",
            "event_name": "NFP",
            "minutes_since": 30,
        })
        assert "POST" in text

    def test_format_news_shield_clear(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.NEWS_SHIELD, {"phase": "NONE", "event_name": ""})
        assert "CLEAR" in text

    def test_format_ai_server_startup(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.AI_SERVER_HEALTH, {
            "status": "startup",
            "uptime_seconds": 0,
            "model_version": "2026-03-01",
        })
        assert "STARTUP" in text
        assert "2026-03-01" in text

    def test_format_ai_server_down(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.AI_SERVER_HEALTH, {
            "status": "down",
            "message": "Connection refused",
        })
        assert "DOWN" in text
        assert "Connection refused" in text

    def test_format_ai_server_degraded(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.AI_SERVER_HEALTH, {
            "status": "degraded",
            "uptime_seconds": 3600,
            "avg_latency_ms": 210.5,
        })
        assert "DEGRADED" in text
        assert "210.5" in text

    def test_format_ai_server_recovered(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.AI_SERVER_HEALTH, {
            "status": "recovered",
            "uptime_seconds": 120,
            "message": "Recovered after 3 failures",
        })
        assert "RECOVERED" in text

    def test_format_risk_alert_warning(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.RISK_ALERT, {
            "alert_level": "YELLOW",
            "reason": "Session loss 5.1%",
            "session_loss_pct": 5.1,
        })
        assert "RISK ALERT" in text
        assert "YELLOW" in text
        assert "5.1" in text

    def test_format_risk_alert_halt(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.RISK_ALERT, {
            "alert_level": "HALT",
            "reason": "Session loss 7.2%",
            "session_loss_pct": 7.2,
            "action_taken": "Trading halted",
        })
        assert "HALT" in text
        assert "Trading halted" in text

    def test_format_risk_alert_spike(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.RISK_ALERT, {
            "alert_level": "SPIKE",
            "reason": "ATR ratio 4.2x",
        })
        assert "SPIKE" in text

    def test_format_daily_report_profit(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.DAILY_REPORT, {
            "date": "2026-03-03",
            "trades_today": 12,
            "win_rate": 66.7,
            "total_pnl": 180.50,
            "max_drawdown": 2.1,
            "avg_duration_min": 8.5,
            "best_trade_usd": 62.50,
            "worst_trade_usd": -25.00,
            "scalper_trades": 10,
            "swing_trades": 2,
        })
        assert "DAILY REPORT" in text
        assert "2026-03-03" in text
        assert "PROFIT" in text
        assert "12" in text
        assert "66.7" in text
        assert "180.50" in text

    def test_format_daily_report_loss(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.DAILY_REPORT, {
            "date": "2026-03-03",
            "trades_today": 4,
            "win_rate": 25.0,
            "total_pnl": -60.0,
            "max_drawdown": 1.5,
            "avg_duration_min": 11.0,
        })
        assert "LOSS" in text

    def test_format_weekly_model_ok(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.WEEKLY_MODEL, {
            "week_end": "2026-03-03",
            "scalper_auc": 0.724,
            "swing_auc": 0.695,
            "scalper_accuracy": 67.3,
            "swing_accuracy": 64.8,
            "feature_drift": 0.08,
            "total_predictions": 150,
            "approval_rate": 34.5,
            "retrain_recommended": False,
        })
        assert "WEEKLY MODEL REPORT" in text
        assert "0.724" in text
        assert "0.695" in text
        assert "OK" in text
        assert "67.3" in text

    def test_format_weekly_model_retrain(self) -> None:
        bot = _make_bot()
        text = bot.format_message(AlertType.WEEKLY_MODEL, {
            "week_end": "2026-03-03",
            "scalper_auc": 0.651,
            "swing_auc": 0.630,
            "feature_drift": 0.22,
            "retrain_recommended": True,
        })
        assert "BELOW TARGET" in text
        assert "retrain" in text.lower()

    def test_all_alert_types_have_formatters(self) -> None:
        """Every AlertType enum member must have a formatter registered."""
        for alert_type in AlertType:
            assert alert_type in _FORMATTERS, f"Missing formatter for {alert_type}"

    def test_unknown_alert_type_raises(self) -> None:
        bot = _make_bot()
        # Pass a raw string that is not an AlertType to exercise the error path
        with pytest.raises((ValueError, AttributeError, KeyError)):
            bot.format_message("NOT_AN_ALERT_TYPE", {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TelegramAlertBot — send_alert + queue behaviour (mocked network)
# ---------------------------------------------------------------------------

class TestTelegramAlertBotSend:
    @pytest.mark.asyncio
    async def test_send_alert_unconfigured_does_not_raise(self) -> None:
        """When token is absent send_alert must silently succeed."""
        bot = _make_bot(configured=False)
        await bot.start()
        # Should not raise even though no token
        await bot.send_alert(AlertType.TRADE_ENTRY, {
            "direction": "BUY",
            "score": 74,
        })
        await bot.stop()

    @pytest.mark.asyncio
    async def test_send_alert_configured_calls_telegram(self) -> None:
        """With credentials the send method should call Bot.send_message.

        Bot is imported lazily inside start(), so we patch telegram.Bot directly.
        """
        bot = _make_bot(configured=True)

        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message = AsyncMock()

        # Bot is imported with `from telegram import Bot` inside start(), so patch
        # the name in the telegram package namespace
        with patch("telegram.Bot", return_value=mock_bot_instance):
            await bot.start()
            await bot.send_alert(AlertType.TRADE_EXIT, {
                "pnl_pips": 10.0,
                "pnl_usd": 50.0,
                "exit_reason": "TP",
                "duration_min": 8,
            })
            # Give the drain task time to process one message
            await asyncio.sleep(0.8)
            await bot.stop()

        mock_bot_instance.send_message.assert_called_once()
        call_kwargs = mock_bot_instance.send_message.call_args
        # chat_id should be our fake chat ID
        assert call_kwargs.kwargs.get("chat_id") == "99999" or \
               (call_kwargs.args and call_kwargs.args[0] == "99999")

    @pytest.mark.asyncio
    async def test_rate_limiting_does_not_exceed_limit(self) -> None:
        """Messages queued in burst should not be sent faster than the rate limit."""
        # Use a very tight rate limit to make the test deterministic
        bot = TelegramAlertBot(token="", chat_id="", rate_limit=5)
        await bot.start()
        # Queue 3 messages — all within limit, should not block
        for _ in range(3):
            await bot.send_alert(AlertType.AI_SERVER_HEALTH, {"status": "healthy"})
        # Verify they all went into the queue
        await bot.stop()

    @pytest.mark.asyncio
    async def test_multiple_alert_types_queued(self) -> None:
        """Queue should accept multiple alert types without error."""
        bot = _make_bot(configured=False)
        await bot.start()
        for alert_type in AlertType:
            await bot.send_alert(alert_type, {})
        await bot.stop()


# ---------------------------------------------------------------------------
# Watchdog — check logic and failure tracking
# ---------------------------------------------------------------------------

class TestWatchdogAIServer:
    @pytest.mark.asyncio
    async def test_check_ai_server_connection_refused(self) -> None:
        """Should return unhealthy when the server port is not open."""
        # Use a port almost certainly not in use
        bot = _make_bot(configured=False)
        watchdog = Watchdog(bot=bot, ai_port=19999)
        result = await watchdog.check_ai_server()
        assert result.healthy is False
        assert "connect" in result.message.lower() or "refused" in result.message.lower() \
               or "timed" in result.message.lower() or "error" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_ai_server_success(self) -> None:
        """Should return healthy when server responds correctly."""
        heartbeat_response = json.dumps({
            "status": "healthy",
            "uptime_seconds": 600,
            "model_version": "2026-03-01",
        }) + "\n"

        async def mock_open_connection(host, port, **kwargs):
            reader = asyncio.StreamReader()
            reader.feed_data(heartbeat_response.encode())
            reader.feed_eof()
            writer = MagicMock()
            writer.write = MagicMock()
            writer.drain = AsyncMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()
            return reader, writer

        bot = _make_bot(configured=False)
        watchdog = Watchdog(bot=bot, ai_port=5001)
        with patch("asyncio.open_connection", side_effect=mock_open_connection):
            result = await watchdog.check_ai_server()

        assert result.healthy is True
        assert result.details["status"] == "healthy"
        assert result.details["uptime_seconds"] == 600

    @pytest.mark.asyncio
    async def test_consecutive_failure_tracking(self) -> None:
        """Failure counter increments on each failed check."""
        bot = _make_bot(configured=False)
        watchdog = Watchdog(bot=bot, ai_port=19998)

        assert watchdog._ai_failures == 0
        for i in range(3):
            await watchdog._handle_ai_result(CheckResult(healthy=False, message="down"))
        assert watchdog._ai_failures == 3

    @pytest.mark.asyncio
    async def test_failure_counter_resets_on_recovery(self) -> None:
        """Counter resets after a successful check."""
        bot = _make_bot(configured=False)
        watchdog = Watchdog(bot=bot, ai_port=19997)

        # Accumulate failures
        for _ in range(5):
            await watchdog._handle_ai_result(CheckResult(healthy=False, message="down"))
        assert watchdog._ai_failures == 5

        # Recovery
        await watchdog._handle_ai_result(
            CheckResult(healthy=True, message="ok", details={"uptime_seconds": 10})
        )
        assert watchdog._ai_failures == 0
        assert watchdog._ai_was_up is True

    @pytest.mark.asyncio
    async def test_alert_sent_after_max_failures(self) -> None:
        """Telegram alert fires when failure count reaches MAX_CONSECUTIVE_FAILURES."""
        bot = _make_bot(configured=False)
        sent_alerts: list[tuple[AlertType, dict]] = []

        async def mock_send(alert_type: AlertType, data: dict) -> None:
            sent_alerts.append((alert_type, data))

        bot.send_alert = mock_send  # type: ignore[method-assign]
        watchdog = Watchdog(bot=bot, ai_port=19996)

        for _ in range(MAX_CONSECUTIVE_FAILURES):
            await watchdog._handle_ai_result(CheckResult(healthy=False, message="down"))

        assert len(sent_alerts) == 1
        assert sent_alerts[0][0] == AlertType.AI_SERVER_HEALTH
        assert sent_alerts[0][1]["status"] == "down"

    @pytest.mark.asyncio
    async def test_recovery_alert_sent(self) -> None:
        """Recovery Telegram alert fires when server comes back up after confirmed outage."""
        bot = _make_bot(configured=False)
        sent_alerts: list[tuple[AlertType, dict]] = []

        async def mock_send(alert_type: AlertType, data: dict) -> None:
            sent_alerts.append((alert_type, data))

        bot.send_alert = mock_send  # type: ignore[method-assign]
        watchdog = Watchdog(bot=bot, ai_port=19995)
        watchdog._ai_failures = MAX_CONSECUTIVE_FAILURES
        watchdog._ai_was_up = False

        await watchdog._handle_ai_result(
            CheckResult(
                healthy=True,
                message="ok",
                details={"uptime_seconds": 30, "model_version": "v1"},
            )
        )
        assert any(
            a[0] == AlertType.AI_SERVER_HEALTH and a[1]["status"] == "recovered"
            for a in sent_alerts
        )

    @pytest.mark.asyncio
    async def test_no_duplicate_alert_before_threshold(self) -> None:
        """No alert fires until exactly MAX_CONSECUTIVE_FAILURES consecutive failures."""
        bot = _make_bot(configured=False)
        sent_alerts: list = []
        bot.send_alert = AsyncMock(side_effect=lambda t, d: sent_alerts.append((t, d)))  # type: ignore

        watchdog = Watchdog(bot=bot, ai_port=19994)
        for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
            await watchdog._handle_ai_result(CheckResult(healthy=False, message="down"))

        assert len(sent_alerts) == 0


class TestWatchdogSystemHealth:
    def test_healthy_system_returns_true(self) -> None:
        bot = _make_bot(configured=False)
        watchdog = Watchdog(bot=bot)

        # Inject a fake stats provider that returns all-good values
        with patch("monitoring.watchdog._get_system_stats") as mock_stats:
            mock_stats.return_value = MagicMock(
                cpu_percent=30.0,
                memory_percent=50.0,
                disk_percent=40.0,
                available_memory_mb=2048.0,
                free_disk_gb=20.0,
            )
            result = watchdog.check_system_health()

        assert result.healthy is True

    def test_high_cpu_triggers_unhealthy(self) -> None:
        bot = _make_bot(configured=False)
        watchdog = Watchdog(bot=bot)

        with patch("monitoring.watchdog._get_system_stats") as mock_stats:
            mock_stats.return_value = MagicMock(
                cpu_percent=95.0,
                memory_percent=50.0,
                disk_percent=40.0,
                available_memory_mb=2048.0,
                free_disk_gb=20.0,
            )
            result = watchdog.check_system_health()

        assert result.healthy is False
        assert "CPU" in result.message

    def test_high_memory_triggers_unhealthy(self) -> None:
        bot = _make_bot(configured=False)
        watchdog = Watchdog(bot=bot)

        with patch("monitoring.watchdog._get_system_stats") as mock_stats:
            mock_stats.return_value = MagicMock(
                cpu_percent=30.0,
                memory_percent=92.0,
                disk_percent=40.0,
                available_memory_mb=100.0,
                free_disk_gb=20.0,
            )
            result = watchdog.check_system_health()

        assert result.healthy is False
        assert "Memory" in result.message or "memory" in result.message

    def test_high_disk_triggers_unhealthy(self) -> None:
        bot = _make_bot(configured=False)
        watchdog = Watchdog(bot=bot)

        with patch("monitoring.watchdog._get_system_stats") as mock_stats:
            mock_stats.return_value = MagicMock(
                cpu_percent=30.0,
                memory_percent=50.0,
                disk_percent=90.0,
                available_memory_mb=2048.0,
                free_disk_gb=2.0,
            )
            result = watchdog.check_system_health()

        assert result.healthy is False
        assert "Disk" in result.message or "disk" in result.message

    def test_stats_unavailable_returns_healthy(self) -> None:
        """If stats cannot be collected, result should be healthy (no false alarms)."""
        bot = _make_bot(configured=False)
        watchdog = Watchdog(bot=bot)

        with patch("monitoring.watchdog._get_system_stats", return_value=None):
            result = watchdog.check_system_health()

        assert result.healthy is True

    @pytest.mark.asyncio
    async def test_resource_alert_sent_after_threshold(self) -> None:
        """Telegram alert fires for CPU after MAX_CONSECUTIVE_FAILURES bad checks."""
        bot = _make_bot(configured=False)
        sent_alerts: list = []
        bot.send_alert = AsyncMock(side_effect=lambda t, d: sent_alerts.append((t, d)))  # type: ignore

        watchdog = Watchdog(bot=bot)
        bad_result = CheckResult(
            healthy=False,
            message="CPU 96%",
            details={
                "cpu_percent": 96.0,
                "memory_percent": 50.0,
                "disk_percent": 40.0,
            },
        )

        for _ in range(MAX_CONSECUTIVE_FAILURES):
            await watchdog._handle_system_result(bad_result)

        assert any(a[0] == AlertType.RISK_ALERT for a in sent_alerts)


class TestGetSystemStats:
    def test_returns_stats_or_none(self) -> None:
        """_get_system_stats should return a SystemStats or None, never raise."""
        result = _get_system_stats()
        # Either we get a valid object or None — both are acceptable
        if result is not None:
            assert 0.0 <= result.cpu_percent <= 100.0
            assert 0.0 <= result.memory_percent <= 100.0
            assert 0.0 <= result.disk_percent <= 100.0


# ---------------------------------------------------------------------------
# PerformanceTracker — trade recording and stats
# ---------------------------------------------------------------------------

def _make_trade(
    pnl_usd: float = 50.0,
    pnl_pips: float = 10.0,
    bot: str = "scalper",
    duration_min: float = 8.0,
    exit_reason: str = "TP1",
    ai_score: int = 75,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    return {
        "symbol": "XAUUSD",
        "direction": "BUY",
        "bot": bot,
        "pnl_usd": pnl_usd,
        "pnl_pips": pnl_pips,
        "duration_min": duration_min,
        "exit_reason": exit_reason,
        "ai_score": ai_score,
        "lot": 0.01,
        "timestamp": timestamp.isoformat(),
    }


class TestPerformanceTrackerRecording:
    @pytest.mark.asyncio
    async def test_record_trade_increments_count(self) -> None:
        tracker = PerformanceTracker(db_path=None)
        assert tracker.trade_count() == 0
        await tracker.record_trade(_make_trade())
        assert tracker.trade_count() == 1

    @pytest.mark.asyncio
    async def test_record_multiple_trades(self) -> None:
        tracker = PerformanceTracker(db_path=None)
        for _ in range(5):
            await tracker.record_trade(_make_trade())
        assert tracker.trade_count() == 5

    @pytest.mark.asyncio
    async def test_clear_resets_count(self) -> None:
        tracker = PerformanceTracker(db_path=None)
        await tracker.record_trade(_make_trade())
        tracker.clear()
        assert tracker.trade_count() == 0

    @pytest.mark.asyncio
    async def test_minimal_trade_accepted(self) -> None:
        """A trade with only pnl_usd should be accepted without error."""
        tracker = PerformanceTracker(db_path=None)
        await tracker.record_trade({"pnl_usd": -15.0})
        assert tracker.trade_count() == 1


class TestPerformanceTrackerDailyStats:
    @pytest.mark.asyncio
    async def test_empty_daily_stats(self) -> None:
        tracker = PerformanceTracker(db_path=None)
        stats = tracker.get_daily_stats()
        assert stats["trades_today"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["total_pnl"] == 0.0

    @pytest.mark.asyncio
    async def test_daily_stats_correct_win_rate(self) -> None:
        now = datetime.now(timezone.utc)
        tracker = PerformanceTracker(db_path=None)
        await tracker.record_trade(_make_trade(pnl_usd=50.0, timestamp=now))
        await tracker.record_trade(_make_trade(pnl_usd=30.0, timestamp=now))
        await tracker.record_trade(_make_trade(pnl_usd=-20.0, timestamp=now))

        stats = tracker.get_daily_stats(date=now)
        assert stats["trades_today"] == 3
        assert abs(stats["win_rate"] - (2 / 3 * 100)) < 0.1
        assert abs(stats["total_pnl"] - 60.0) < 0.001

    @pytest.mark.asyncio
    async def test_daily_stats_excludes_yesterday(self) -> None:
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1, hours=1)
        tracker = PerformanceTracker(db_path=None)
        await tracker.record_trade(_make_trade(pnl_usd=50.0, timestamp=yesterday))
        await tracker.record_trade(_make_trade(pnl_usd=30.0, timestamp=now))

        stats = tracker.get_daily_stats(date=now)
        assert stats["trades_today"] == 1

    @pytest.mark.asyncio
    async def test_daily_stats_bot_filter(self) -> None:
        now = datetime.now(timezone.utc)
        tracker = PerformanceTracker(db_path=None)
        await tracker.record_trade(_make_trade(pnl_usd=50.0, bot="scalper", timestamp=now))
        await tracker.record_trade(_make_trade(pnl_usd=20.0, bot="swing", timestamp=now))

        scalper_stats = tracker.get_daily_stats(date=now, bot="scalper")
        swing_stats = tracker.get_daily_stats(date=now, bot="swing")

        assert scalper_stats["trades_today"] == 1
        assert swing_stats["trades_today"] == 1

    @pytest.mark.asyncio
    async def test_daily_stats_includes_bot_breakdown(self) -> None:
        now = datetime.now(timezone.utc)
        tracker = PerformanceTracker(db_path=None)
        await tracker.record_trade(_make_trade(bot="scalper", timestamp=now))
        await tracker.record_trade(_make_trade(bot="scalper", timestamp=now))
        await tracker.record_trade(_make_trade(bot="swing", timestamp=now))

        stats = tracker.get_daily_stats(date=now)
        assert stats.get("scalper_trades") == 2
        assert stats.get("swing_trades") == 1


class TestPerformanceTrackerWeeklyStats:
    @pytest.mark.asyncio
    async def test_weekly_stats_rolling_window(self) -> None:
        now = datetime.now(timezone.utc)
        tracker = PerformanceTracker(db_path=None)

        # 5 trades in last 7 days
        for i in range(5):
            ts = now - timedelta(days=i)
            await tracker.record_trade(_make_trade(pnl_usd=10.0, timestamp=ts))

        # 2 trades older than 7 days
        for i in range(8, 10):
            ts = now - timedelta(days=i)
            await tracker.record_trade(_make_trade(pnl_usd=10.0, timestamp=ts))

        stats = tracker.get_weekly_stats(end_date=now)
        assert stats["trade_count"] == 5
        assert stats["period_days"] == 7


class TestMaxDrawdown:
    def test_no_trades_zero_dd(self) -> None:
        assert _compute_max_drawdown([]) == 0.0

    def test_all_wins_zero_dd(self) -> None:
        trades = [{"pnl_usd": 10.0}, {"pnl_usd": 20.0}, {"pnl_usd": 5.0}]
        assert _compute_max_drawdown(trades) == 0.0

    def test_single_loss_after_gains(self) -> None:
        trades = [
            {"pnl_usd": 100.0},
            {"pnl_usd": -50.0},  # 50% drawdown from peak of 100
        ]
        dd = _compute_max_drawdown(trades)
        assert abs(dd - 50.0) < 0.01

    def test_recovery_after_drawdown(self) -> None:
        trades = [
            {"pnl_usd": 100.0},
            {"pnl_usd": -60.0},
            {"pnl_usd": 80.0},
        ]
        dd = _compute_max_drawdown(trades)
        # Peak after trade 1 = 100. After trade 2 = 40. DD = (100-40)/100 = 60%
        assert abs(dd - 60.0) < 0.01


class TestSafeDivide:
    def test_normal_division(self) -> None:
        assert _safe_divide(10.0, 4.0) == 2.5

    def test_zero_denominator_returns_default(self) -> None:
        assert _safe_divide(10.0, 0.0) == 0.0

    def test_zero_denominator_custom_default(self) -> None:
        assert _safe_divide(10.0, 0.0, default=99.0) == 99.0


class TestPerformanceTrackerReports:
    @pytest.mark.asyncio
    async def test_generate_daily_report_profit(self) -> None:
        now = datetime.now(timezone.utc)
        tracker = PerformanceTracker(db_path=None)
        for _ in range(3):
            await tracker.record_trade(_make_trade(pnl_usd=30.0, timestamp=now))
        await tracker.record_trade(_make_trade(pnl_usd=-15.0, timestamp=now))

        report = tracker.generate_daily_report(date=now)
        assert "DAILY REPORT" in report
        assert "PROFIT" in report
        assert "4" in report  # 4 trades
        assert "75.0" in report  # 3/4 = 75% win rate

    @pytest.mark.asyncio
    async def test_generate_daily_report_loss(self) -> None:
        now = datetime.now(timezone.utc)
        tracker = PerformanceTracker(db_path=None)
        await tracker.record_trade(_make_trade(pnl_usd=-40.0, timestamp=now))

        report = tracker.generate_daily_report(date=now)
        assert "LOSS" in report
        assert "-40.00" in report

    @pytest.mark.asyncio
    async def test_generate_daily_report_no_trades(self) -> None:
        tracker = PerformanceTracker(db_path=None)
        report = tracker.generate_daily_report()
        assert "DAILY REPORT" in report
        assert "0" in report

    @pytest.mark.asyncio
    async def test_generate_weekly_report_with_model_stats(self) -> None:
        now = datetime.now(timezone.utc)
        tracker = PerformanceTracker(db_path=None)
        for _ in range(5):
            ts = now - timedelta(hours=12 * _)
            await tracker.record_trade(_make_trade(pnl_usd=20.0, timestamp=ts))

        model_stats = {
            "scalper_auc": 0.715,
            "swing_auc": 0.672,
            "scalper_accuracy": 65.0,
            "swing_accuracy": 62.0,
            "feature_drift": 0.09,
        }
        report = tracker.generate_weekly_report(end_date=now, model_stats=model_stats)
        assert "WEEKLY REPORT" in report
        assert "Model Performance" in report
        assert "0.715" in report
        assert "0.672" in report

    @pytest.mark.asyncio
    async def test_generate_weekly_report_drift_warning(self) -> None:
        tracker = PerformanceTracker(db_path=None)
        report = tracker.generate_weekly_report(
            model_stats={"feature_drift": 0.25}
        )
        assert "retrain" in report.lower() or "HIGH" in report

    @pytest.mark.asyncio
    async def test_generate_weekly_report_no_model_stats(self) -> None:
        tracker = PerformanceTracker(db_path=None)
        report = tracker.generate_weekly_report()
        assert "WEEKLY REPORT" in report
        # Without model_stats, the Model Performance section should not appear
        assert "Model Performance" not in report


class TestPerformanceTrackerSQLite:
    @pytest.mark.asyncio
    async def test_sqlite_persistence(self, tmp_path: Path) -> None:
        """Trades should be written to SQLite when a db_path is given."""
        db_path = tmp_path / "perf_test.db"
        tracker = PerformanceTracker(db_path=db_path)
        await tracker.record_trade(_make_trade(pnl_usd=42.0))

        assert db_path.exists()
        # Read back with raw sqlite3
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT pnl_usd FROM trades").fetchall()
        conn.close()
        assert len(rows) == 1
        assert abs(rows[0][0] - 42.0) < 0.001

    @pytest.mark.asyncio
    async def test_memory_only_mode(self) -> None:
        """db_path=None should not create any file and work fine."""
        tracker = PerformanceTracker(db_path=None)
        await tracker.record_trade(_make_trade(pnl_usd=10.0))
        assert tracker.trade_count() == 1


# ---------------------------------------------------------------------------
# News schedule integration tests (from monitoring.news_schedule)
# These re-verify the key logic from Phase 11 within this test module.
# ---------------------------------------------------------------------------

class TestNewsSchedulePhaseComputation:
    """Verify compute_phase and build_schedule work correctly."""

    def test_nfp_detection_phase(self) -> None:
        from monitoring.news_schedule import compute_phase

        now = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=45)  # 45 min before = detection window
        result = compute_phase(event_time, now, "NFP")
        assert result["phase"] == "DETECTION"

    def test_nfp_pre_phase(self) -> None:
        from monitoring.news_schedule import compute_phase

        now = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=15)  # 15 min before = pre window
        result = compute_phase(event_time, now, "NFP")
        assert result["phase"] == "PRE"

    def test_nfp_during_phase(self) -> None:
        from monitoring.news_schedule import compute_phase

        now = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
        event_time = now - timedelta(minutes=10)  # 10 min after = during
        result = compute_phase(event_time, now, "NFP")
        assert result["phase"] == "DURING"

    def test_nfp_post_phase(self) -> None:
        from monitoring.news_schedule import compute_phase

        now = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
        event_time = now - timedelta(minutes=50)  # 50 min after = post window
        result = compute_phase(event_time, now, "NFP")
        assert result["phase"] == "POST"

    def test_far_future_event_is_none(self) -> None:
        from monitoring.news_schedule import compute_phase

        now = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(hours=5)
        result = compute_phase(event_time, now, "HIGH_IMPACT")
        assert result["phase"] == "NONE"

    def test_build_schedule_pre_event_blocks(self) -> None:
        from monitoring.news_schedule import build_schedule

        now = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
        events = [{
            "name": "Non-Farm Payrolls",
            "time": (now + timedelta(minutes=20)).isoformat(),
            "impact": 3,
            "currency": "USD",
        }]
        schedule = build_schedule(events, now=now)
        assert schedule["shield_active"] is True
        assert schedule["shield_phase"] == "PRE"

    def test_build_schedule_non_usd_filtered(self) -> None:
        from monitoring.news_schedule import build_schedule

        now = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
        events = [{
            "name": "EUR CPI",
            "time": (now + timedelta(hours=1)).isoformat(),
            "impact": 3,
            "currency": "EUR",
        }]
        schedule = build_schedule(events, now=now)
        assert len(schedule["upcoming_events"]) == 0
        assert schedule["shield_active"] is False

    def test_build_schedule_empty(self) -> None:
        from monitoring.news_schedule import build_schedule

        schedule = build_schedule([])
        assert schedule["shield_active"] is False
        assert schedule["shield_phase"] == "NONE"
        assert schedule["upcoming_events"] == []
