"""Telegram alert bot for the XAUUSD AI Trading System.

Sends structured alerts for 7 event types:
  1. Trade Entry
  2. Trade Exit
  3. News Shield phase transitions
  4. AI Server Health
  5. Risk Alert
  6. Daily Performance Report
  7. Weekly Model Performance

Rate-limited to max 20 messages/minute with an async queue for burst handling.
Configured via TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars (from ai_server.config).
Degrades gracefully when the token is not configured.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from enum import Enum, auto
from typing import Any

from ai_server.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

# Maximum messages per minute before throttling kicks in
_RATE_LIMIT_PER_MINUTE = 20
# Seconds between individual sends when draining the queue under load
_MIN_SEND_INTERVAL = 0.5


class AlertType(Enum):
    TRADE_ENTRY = auto()
    TRADE_EXIT = auto()
    NEWS_SHIELD = auto()
    AI_SERVER_HEALTH = auto()
    RISK_ALERT = auto()
    DAILY_REPORT = auto()
    WEEKLY_MODEL = auto()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_trade_entry(data: dict[str, Any]) -> str:
    """Format a Trade Entry alert message."""
    symbol = data.get("symbol", "XAUUSD")
    direction = data.get("direction", "BUY").upper()
    lot = data.get("lot", 0.0)
    score = data.get("score", 0)
    regime = data.get("regime", "unknown")
    entry_price = data.get("entry_price", 0.0)
    sl = data.get("sl", 0.0)
    tp = data.get("tp", 0.0)
    wyckoff = data.get("wyckoff_phase", "")
    bot = data.get("bot", "")

    direction_icon = "BUY" if direction == "BUY" else "SELL"
    lines = [
        f"[TRADE ENTRY] {direction_icon}",
        f"Symbol:    {symbol}",
        f"Direction: {direction}",
        f"Lot:       {lot:.2f}",
        f"Entry:     {entry_price:.2f}",
        f"SL:        {sl:.2f}",
        f"TP:        {tp:.2f}",
        f"AI Score:  {score}",
        f"Regime:    {regime}",
    ]
    if wyckoff:
        lines.append(f"Wyckoff:   {wyckoff}")
    if bot:
        lines.append(f"Bot:       {bot}")
    return "\n".join(lines)


def _format_trade_exit(data: dict[str, Any]) -> str:
    """Format a Trade Exit alert message."""
    symbol = data.get("symbol", "XAUUSD")
    pnl_pips = data.get("pnl_pips", 0.0)
    pnl_usd = data.get("pnl_usd", 0.0)
    exit_reason = data.get("exit_reason", "unknown")
    duration_min = data.get("duration_min", 0)
    session_pnl = data.get("session_pnl_usd", None)

    result = "WIN" if pnl_pips >= 0 else "LOSS"
    result_icon = "[WIN]" if pnl_pips >= 0 else "[LOSS]"
    lines = [
        f"[TRADE EXIT] {result_icon} {result}",
        f"Symbol:      {symbol}",
        f"P&L (pips):  {pnl_pips:+.1f}",
        f"P&L (USD):   ${pnl_usd:+.2f}",
        f"Exit reason: {exit_reason}",
        f"Duration:    {duration_min} min",
    ]
    if session_pnl is not None:
        lines.append(f"Session P&L: ${session_pnl:+.2f}")
    return "\n".join(lines)


def _format_news_shield(data: dict[str, Any]) -> str:
    """Format a News Shield phase transition alert."""
    phase = data.get("phase", "NONE")
    event_name = data.get("event_name", "Unknown event")
    event_time = data.get("event_time", "")
    minutes_until = data.get("minutes_until", None)
    minutes_since = data.get("minutes_since", None)

    phase_labels = {
        "DETECTION": "[DETECTION] Warning window active",
        "PRE": "[PRE-NEWS] Trading halted",
        "DURING": "[DURING] Event in progress — all entries blocked",
        "POST": "[POST-NEWS] Reduced-risk window",
        "NONE": "[CLEAR] News shield deactivated",
    }
    label = phase_labels.get(phase, f"Phase: {phase}")

    lines = [
        f"[NEWS SHIELD] {label}",
        f"Event: {event_name}",
    ]
    if event_time:
        lines.append(f"Time:  {event_time}")
    if minutes_until is not None and minutes_until > 0:
        lines.append(f"Starts in: {minutes_until:.0f} min")
    if minutes_since is not None and minutes_since > 0:
        lines.append(f"Elapsed:   {minutes_since:.0f} min")
    return "\n".join(lines)


def _format_ai_server_health(data: dict[str, Any]) -> str:
    """Format an AI Server Health alert."""
    status = data.get("status", "unknown")
    uptime_sec = data.get("uptime_seconds", 0)
    model_version = data.get("model_version", "")
    message = data.get("message", "")
    predictions_today = data.get("predictions_today", None)
    avg_latency_ms = data.get("avg_latency_ms", None)

    status_icons = {
        "startup": "[STARTUP] AI Server started",
        "shutdown": "[SHUTDOWN] AI Server stopped",
        "healthy": "[OK] AI Server healthy",
        "degraded": "[DEGRADED] AI Server in fallback mode",
        "down": "[DOWN] AI Server unreachable",
        "recovered": "[RECOVERED] AI Server back online",
    }
    header = status_icons.get(status, f"[{status.upper()}] AI Server status")

    uptime_hours = uptime_sec // 3600
    uptime_min = (uptime_sec % 3600) // 60

    lines = [header]
    if uptime_sec > 0:
        lines.append(f"Uptime:    {uptime_hours}h {uptime_min}m")
    if model_version:
        lines.append(f"Model ver: {model_version}")
    if predictions_today is not None:
        lines.append(f"Preds/day: {predictions_today}")
    if avg_latency_ms is not None:
        lines.append(f"Avg lat:   {avg_latency_ms:.1f}ms")
    if message:
        lines.append(f"Note:      {message}")
    return "\n".join(lines)


def _format_risk_alert(data: dict[str, Any]) -> str:
    """Format a Risk Alert message."""
    alert_level = data.get("alert_level", "WARNING")  # WARNING, YELLOW, RED, HALT
    reason = data.get("reason", "Unknown")
    session_loss_pct = data.get("session_loss_pct", None)
    daily_loss_pct = data.get("daily_loss_pct", None)
    action_taken = data.get("action_taken", "")

    level_icons = {
        "WARNING": "[WARNING]",
        "YELLOW": "[YELLOW ALERT]",
        "RED": "[RED ALERT]",
        "HALT": "[TRADING HALTED]",
        "SPIKE": "[SPIKE DETECTED]",
    }
    header = level_icons.get(alert_level.upper(), f"[RISK] {alert_level}")

    lines = [
        f"[RISK ALERT] {header}",
        f"Reason: {reason}",
    ]
    if session_loss_pct is not None:
        lines.append(f"Session loss: {session_loss_pct:.1f}%")
    if daily_loss_pct is not None:
        lines.append(f"Daily loss:   {daily_loss_pct:.1f}%")
    if action_taken:
        lines.append(f"Action:       {action_taken}")
    return "\n".join(lines)


def _format_daily_report(data: dict[str, Any]) -> str:
    """Format a Daily Performance Report."""
    date = data.get("date", "")
    trades_today = data.get("trades_today", 0)
    win_rate = data.get("win_rate", 0.0)
    total_pnl = data.get("total_pnl", 0.0)
    max_drawdown = data.get("max_drawdown", 0.0)
    avg_duration_min = data.get("avg_duration_min", 0.0)
    best_trade = data.get("best_trade_usd", None)
    worst_trade = data.get("worst_trade_usd", None)
    scalper_trades = data.get("scalper_trades", None)
    swing_trades = data.get("swing_trades", None)

    result_icon = "PROFIT" if total_pnl >= 0 else "LOSS"

    lines = [
        f"[DAILY REPORT] {date}  {result_icon}",
        "",
        f"Total Trades:  {trades_today}",
        f"Win Rate:      {win_rate:.1f}%",
        f"Total P&L:     ${total_pnl:+.2f}",
        f"Max Drawdown:  {max_drawdown:.1f}%",
        f"Avg Duration:  {avg_duration_min:.0f} min",
    ]
    if scalper_trades is not None:
        lines.append(f"Scalper:       {scalper_trades} trades")
    if swing_trades is not None:
        lines.append(f"Swing:         {swing_trades} trades")
    if best_trade is not None:
        lines.append(f"Best trade:    ${best_trade:+.2f}")
    if worst_trade is not None:
        lines.append(f"Worst trade:   ${worst_trade:+.2f}")
    return "\n".join(lines)


def _format_weekly_model(data: dict[str, Any]) -> str:
    """Format a Weekly Model Performance report."""
    week_end = data.get("week_end", "")
    scalper_auc = data.get("scalper_auc", None)
    swing_auc = data.get("swing_auc", None)
    scalper_accuracy = data.get("scalper_accuracy", None)
    swing_accuracy = data.get("swing_accuracy", None)
    feature_drift = data.get("feature_drift", None)
    retrain_recommended = data.get("retrain_recommended", False)
    total_predictions = data.get("total_predictions", None)
    approval_rate = data.get("approval_rate", None)

    lines = [
        f"[WEEKLY MODEL REPORT] w/e {week_end}",
        "",
    ]

    if scalper_auc is not None:
        status = "OK" if scalper_auc >= 0.70 else "BELOW TARGET"
        lines.append(f"Scalper AUC:      {scalper_auc:.3f}  [{status}]")
    if swing_auc is not None:
        status = "OK" if swing_auc >= 0.68 else "BELOW TARGET"
        lines.append(f"Swing AUC:        {swing_auc:.3f}  [{status}]")
    if scalper_accuracy is not None:
        lines.append(f"Scalper Accuracy: {scalper_accuracy:.1f}%")
    if swing_accuracy is not None:
        lines.append(f"Swing Accuracy:   {swing_accuracy:.1f}%")
    if feature_drift is not None:
        drift_status = "HIGH" if feature_drift > 0.15 else "OK"
        lines.append(f"Feature Drift:    {feature_drift:.3f}  [{drift_status}]")
    if total_predictions is not None:
        lines.append(f"Total Preds:      {total_predictions}")
    if approval_rate is not None:
        lines.append(f"Approval Rate:    {approval_rate:.1f}%")
    if retrain_recommended:
        lines.append("")
        lines.append("[ACTION] Model retrain recommended")
    return "\n".join(lines)


# Map each AlertType to its formatting function
_FORMATTERS = {
    AlertType.TRADE_ENTRY: _format_trade_entry,
    AlertType.TRADE_EXIT: _format_trade_exit,
    AlertType.NEWS_SHIELD: _format_news_shield,
    AlertType.AI_SERVER_HEALTH: _format_ai_server_health,
    AlertType.RISK_ALERT: _format_risk_alert,
    AlertType.DAILY_REPORT: _format_daily_report,
    AlertType.WEEKLY_MODEL: _format_weekly_model,
}


# ---------------------------------------------------------------------------
# Main bot class
# ---------------------------------------------------------------------------

class TelegramAlertBot:
    """Async Telegram alert bot with rate-limiting and message queue.

    Usage::

        bot = TelegramAlertBot()
        await bot.start()
        await bot.send_alert(AlertType.TRADE_ENTRY, {"symbol": "XAUUSD", ...})
        await bot.stop()

    When TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are absent the bot logs a
    warning on the first send attempt and becomes a no-op (it will never raise).
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        rate_limit: int = _RATE_LIMIT_PER_MINUTE,
    ) -> None:
        self._token: str = token or TELEGRAM_BOT_TOKEN
        self._chat_id: str = chat_id or TELEGRAM_CHAT_ID
        self._rate_limit: int = rate_limit
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._timestamps: deque[float] = deque()
        self._send_task: asyncio.Task | None = None
        self._configured: bool = bool(self._token and self._chat_id)
        self._bot = None  # telegram.Bot instance, created lazily on start()

        if not self._configured:
            logger.warning(
                "TelegramAlertBot: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. "
                "Alerts will be logged but not delivered."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background queue-drain task."""
        if self._configured and self._bot is None:
            from telegram import Bot
            self._bot = Bot(token=self._token)
        self._send_task = asyncio.create_task(self._drain_queue())
        logger.info("TelegramAlertBot started (configured=%s)", self._configured)

    async def stop(self) -> None:
        """Drain remaining messages and stop the background task."""
        if self._send_task is not None:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        logger.info("TelegramAlertBot stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_alert(self, alert_type: AlertType, data: dict[str, Any]) -> None:
        """Enqueue an alert for delivery.

        Args:
            alert_type: One of the AlertType enum members.
            data:       Dict of fields relevant to the alert type.
        """
        formatter = _FORMATTERS.get(alert_type)
        if formatter is None:
            logger.error("Unknown alert type: %s", alert_type)
            return

        try:
            text = formatter(data)
        except Exception as exc:
            logger.error("Failed to format alert %s: %s", alert_type, exc)
            return

        logger.info("Alert queued [%s]:\n%s", alert_type.name, text)
        await self._queue.put(text)

    def format_message(self, alert_type: AlertType, data: dict[str, Any]) -> str:
        """Return the formatted text for an alert without sending it.

        Useful for testing and pre-inspection.
        """
        formatter = _FORMATTERS.get(alert_type)
        if formatter is None:
            raise ValueError(f"Unknown alert type: {alert_type}")
        return formatter(data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _drain_queue(self) -> None:
        """Background task — continuously drains the message queue, respecting rate limits."""
        while True:
            text = await self._queue.get()
            await self._rate_wait()
            await self._send(text)
            self._queue.task_done()

    async def _rate_wait(self) -> None:
        """Block until we are under the per-minute rate limit."""
        now = time.monotonic()
        # Remove timestamps older than 60 seconds
        while self._timestamps and now - self._timestamps[0] > 60.0:
            self._timestamps.popleft()

        if len(self._timestamps) >= self._rate_limit:
            # Calculate how long until the oldest timestamp expires
            sleep_for = 60.0 - (now - self._timestamps[0]) + 0.05
            if sleep_for > 0:
                logger.debug("Rate limit reached — sleeping %.1fs", sleep_for)
                await asyncio.sleep(sleep_for)
        else:
            # Small courtesy gap between messages to avoid Telegram flood limits
            await asyncio.sleep(_MIN_SEND_INTERVAL)

        self._timestamps.append(time.monotonic())

    async def _send(self, text: str) -> None:
        """Actually deliver a message via the Telegram Bot API."""
        if not self._configured or self._bot is None:
            logger.debug("Telegram not configured — suppressing message: %s", text[:80])
            return
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=None,  # Plain text — avoids Markdown escaping issues
            )
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
