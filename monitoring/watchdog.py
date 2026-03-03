"""System health watchdog for the XAUUSD AI Trading System.

Runs as a separate async process.  Every 30 seconds it performs:
  1. AI server heartbeat — TCP connect to localhost:5001, send heartbeat
  2. System health check — CPU, memory and disk via the os / resource modules
     (psutil is NOT required; falls back gracefully if unavailable)

Fires Telegram alerts when:
  - AI server goes down / comes back up
  - CPU, memory or disk exceeds configured thresholds

Usage::

    import asyncio
    from monitoring.watchdog import Watchdog

    watchdog = Watchdog()
    asyncio.run(watchdog.run())
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ai_server.config import (
    AI_SERVER_HOST,
    AI_SERVER_PORT,
)
from monitoring.telegram_bot import AlertType, TelegramAlertBot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (not in config because they are monitoring-specific and rarely
# changed, but kept as module-level constants so they are easy to adjust)
# ---------------------------------------------------------------------------
CHECK_INTERVAL_SECONDS = 30
CPU_ALERT_THRESHOLD_PCT = 90.0      # percentage
MEMORY_ALERT_THRESHOLD_PCT = 85.0   # percentage
DISK_ALERT_THRESHOLD_PCT = 85.0     # percentage
MAX_CONSECUTIVE_FAILURES = 3        # alert after this many back-to-back failures
HEARTBEAT_TIMEOUT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single watchdog check."""
    healthy: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemStats:
    """System resource snapshot."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    available_memory_mb: float
    free_disk_gb: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_system_stats() -> SystemStats | None:
    """Return basic system resource stats without psutil.

    Uses the standard library only.  Returns None if the platform does
    not support the required interfaces.
    """
    try:
        # --- CPU: use os.getloadavg() as a proxy on Unix ----------------
        # getloadavg is not available on Windows, so we fall back to 0.0
        try:
            load1, _, _ = os.getloadavg()
            # Convert 1-minute load average to an approximate percentage.
            # A load of 1.0 per CPU core = 100 % usage for that core.
            cpu_count = os.cpu_count() or 1
            cpu_percent = min((load1 / cpu_count) * 100.0, 100.0)
        except (AttributeError, OSError):
            cpu_percent = 0.0

        # --- Memory: try /proc/meminfo (Linux) --------------------------
        memory_percent = 0.0
        available_memory_mb = 0.0
        try:
            with open("/proc/meminfo") as f:
                meminfo: dict[str, int] = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        try:
                            meminfo[key] = int(parts[1])
                        except ValueError:
                            pass
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            if total_kb > 0:
                memory_percent = ((total_kb - avail_kb) / total_kb) * 100.0
                available_memory_mb = avail_kb / 1024.0
        except (FileNotFoundError, OSError):
            # macOS or Windows — try resource module as fallback
            try:
                import resource
                rusage = resource.getrusage(resource.RUSAGE_SELF)
                # ru_maxrss is in bytes on macOS, kilobytes on Linux
                if hasattr(rusage, "ru_maxrss"):
                    available_memory_mb = 0.0  # cannot determine total
                    memory_percent = 0.0
            except ImportError:
                pass

        # --- Disk: os.statvfs -------------------------------------------
        disk_percent = 0.0
        free_disk_gb = 0.0
        try:
            stat = os.statvfs("/")
            total_blocks = stat.f_blocks
            free_blocks = stat.f_bfree
            if total_blocks > 0:
                used = total_blocks - free_blocks
                disk_percent = (used / total_blocks) * 100.0
                free_disk_gb = (free_blocks * stat.f_frsize) / (1024 ** 3)
        except (AttributeError, OSError):
            # os.statvfs not available on Windows
            pass

        return SystemStats(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            available_memory_mb=available_memory_mb,
            free_disk_gb=free_disk_gb,
        )

    except Exception as exc:
        logger.debug("System stats unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Watchdog class
# ---------------------------------------------------------------------------

class Watchdog:
    """Monitors the AI server and system health, fires Telegram alerts.

    Args:
        bot:              TelegramAlertBot instance.  Created automatically
                          if not provided.
        check_interval:   Seconds between checks (default 30).
        ai_host:          AI server hostname.
        ai_port:          AI server port.
    """

    def __init__(
        self,
        bot: TelegramAlertBot | None = None,
        check_interval: int = CHECK_INTERVAL_SECONDS,
        ai_host: str = AI_SERVER_HOST,
        ai_port: int = AI_SERVER_PORT,
    ) -> None:
        self._bot = bot or TelegramAlertBot()
        self._check_interval = check_interval
        self._ai_host = ai_host
        self._ai_port = ai_port

        # Consecutive failure counters
        self._ai_failures: int = 0
        self._cpu_failures: int = 0
        self._memory_failures: int = 0
        self._disk_failures: int = 0

        # Last known states (for recovery detection)
        self._ai_was_up: bool = True
        self._started_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def check_ai_server(self) -> CheckResult:
        """TCP connect to the AI server and send a heartbeat message.

        Returns CheckResult with healthy=True on success.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._ai_host, self._ai_port),
                timeout=HEARTBEAT_TIMEOUT_SECONDS,
            )
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as exc:
            return CheckResult(
                healthy=False,
                message=f"Cannot connect to AI server: {exc}",
                details={"host": self._ai_host, "port": self._ai_port},
            )

        try:
            payload = json.dumps({"type": "heartbeat"}) + "\n"
            writer.write(payload.encode("utf-8"))
            await asyncio.wait_for(writer.drain(), timeout=HEARTBEAT_TIMEOUT_SECONDS)

            raw = await asyncio.wait_for(
                reader.readline(), timeout=HEARTBEAT_TIMEOUT_SECONDS
            )
            response = json.loads(raw.decode("utf-8").strip())

            writer.close()
            await writer.wait_closed()

            uptime = response.get("uptime_seconds", 0)
            model_version = response.get("model_version", "")
            status = response.get("status", "unknown")

            return CheckResult(
                healthy=True,
                message="AI server healthy",
                details={
                    "status": status,
                    "uptime_seconds": uptime,
                    "model_version": model_version,
                },
            )
        except Exception as exc:
            try:
                writer.close()
            except Exception:
                pass
            return CheckResult(
                healthy=False,
                message=f"AI server responded with error: {exc}",
            )

    def check_system_health(self) -> CheckResult:
        """Check CPU, memory and disk usage.

        Returns CheckResult with details dict containing all metrics.
        Healthy if all metrics are below thresholds.
        """
        stats = _get_system_stats()
        if stats is None:
            return CheckResult(
                healthy=True,
                message="System stats unavailable on this platform",
                details={},
            )

        warnings: list[str] = []
        if stats.cpu_percent > CPU_ALERT_THRESHOLD_PCT:
            warnings.append(f"CPU {stats.cpu_percent:.0f}% > {CPU_ALERT_THRESHOLD_PCT}%")
        if stats.memory_percent > MEMORY_ALERT_THRESHOLD_PCT:
            warnings.append(
                f"Memory {stats.memory_percent:.0f}% > {MEMORY_ALERT_THRESHOLD_PCT}%"
            )
        if stats.disk_percent > DISK_ALERT_THRESHOLD_PCT:
            warnings.append(
                f"Disk {stats.disk_percent:.0f}% > {DISK_ALERT_THRESHOLD_PCT}%"
            )

        healthy = len(warnings) == 0
        message = "System healthy" if healthy else "; ".join(warnings)

        return CheckResult(
            healthy=healthy,
            message=message,
            details={
                "cpu_percent": stats.cpu_percent,
                "memory_percent": stats.memory_percent,
                "disk_percent": stats.disk_percent,
                "available_memory_mb": stats.available_memory_mb,
                "free_disk_gb": stats.free_disk_gb,
            },
        )

    # ------------------------------------------------------------------
    # Single check cycle
    # ------------------------------------------------------------------

    async def run_checks(self) -> dict[str, CheckResult]:
        """Run all checks once and fire alerts as needed.

        Returns a dict mapping check name to CheckResult.
        """
        results: dict[str, CheckResult] = {}

        # AI server
        ai_result = await self.check_ai_server()
        results["ai_server"] = ai_result
        await self._handle_ai_result(ai_result)

        # System health
        sys_result = self.check_system_health()
        results["system_health"] = sys_result
        await self._handle_system_result(sys_result)

        return results

    # ------------------------------------------------------------------
    # Alert dispatch
    # ------------------------------------------------------------------

    async def _handle_ai_result(self, result: CheckResult) -> None:
        """Update failure counters and send alerts on state changes."""
        if result.healthy:
            if not self._ai_was_up and self._ai_failures >= MAX_CONSECUTIVE_FAILURES:
                # Server recovered after a confirmed outage
                uptime = result.details.get("uptime_seconds", 0)
                model_ver = result.details.get("model_version", "")
                await self._bot.send_alert(
                    AlertType.AI_SERVER_HEALTH,
                    {
                        "status": "recovered",
                        "uptime_seconds": uptime,
                        "model_version": model_ver,
                        "message": f"Recovered after {self._ai_failures} failures",
                    },
                )
                logger.info("AI server recovered after %d failures", self._ai_failures)

            self._ai_failures = 0
            self._ai_was_up = True
        else:
            self._ai_failures += 1
            logger.warning(
                "AI server check failed (%d consecutive): %s",
                self._ai_failures,
                result.message,
            )
            # Alert on the first failure that crosses the threshold
            if self._ai_failures == MAX_CONSECUTIVE_FAILURES:
                await self._bot.send_alert(
                    AlertType.AI_SERVER_HEALTH,
                    {
                        "status": "down",
                        "message": result.message,
                        "uptime_seconds": 0,
                    },
                )
                self._ai_was_up = False
            elif self._ai_failures > MAX_CONSECUTIVE_FAILURES:
                # Periodic reminder every 10 failures
                if self._ai_failures % 10 == 0:
                    await self._bot.send_alert(
                        AlertType.AI_SERVER_HEALTH,
                        {
                            "status": "down",
                            "message": (
                                f"Still unreachable ({self._ai_failures} checks failed). "
                                f"{result.message}"
                            ),
                            "uptime_seconds": 0,
                        },
                    )

    async def _handle_system_result(self, result: CheckResult) -> None:
        """Fire resource alerts when thresholds are breached."""
        if result.healthy:
            # Reset counters (independent of which resource was previously bad)
            self._cpu_failures = 0
            self._memory_failures = 0
            self._disk_failures = 0
            return

        details = result.details
        cpu = details.get("cpu_percent", 0.0)
        mem = details.get("memory_percent", 0.0)
        disk = details.get("disk_percent", 0.0)

        if cpu > CPU_ALERT_THRESHOLD_PCT:
            self._cpu_failures += 1
            if self._cpu_failures == MAX_CONSECUTIVE_FAILURES:
                await self._bot.send_alert(
                    AlertType.RISK_ALERT,
                    {
                        "alert_level": "WARNING",
                        "reason": f"High CPU usage: {cpu:.0f}%",
                        "action_taken": "No automatic action — manual review required",
                    },
                )
        else:
            self._cpu_failures = 0

        if mem > MEMORY_ALERT_THRESHOLD_PCT:
            self._memory_failures += 1
            if self._memory_failures == MAX_CONSECUTIVE_FAILURES:
                await self._bot.send_alert(
                    AlertType.RISK_ALERT,
                    {
                        "alert_level": "WARNING",
                        "reason": f"High memory usage: {mem:.0f}%",
                        "action_taken": "No automatic action — manual review required",
                    },
                )
        else:
            self._memory_failures = 0

        if disk > DISK_ALERT_THRESHOLD_PCT:
            self._disk_failures += 1
            if self._disk_failures == MAX_CONSECUTIVE_FAILURES:
                await self._bot.send_alert(
                    AlertType.RISK_ALERT,
                    {
                        "alert_level": "WARNING",
                        "reason": f"High disk usage: {disk:.0f}%",
                        "action_taken": "No automatic action — manual review required",
                    },
                )
        else:
            self._disk_failures = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the watchdog main loop indefinitely.

        Starts the Telegram bot, then checks every CHECK_INTERVAL_SECONDS.
        Call this with asyncio.run() or as an asyncio task.
        """
        await self._bot.start()
        logger.info(
            "Watchdog started — checking every %ds (AI=%s:%d)",
            self._check_interval,
            self._ai_host,
            self._ai_port,
        )

        # Send startup notification
        await self._bot.send_alert(
            AlertType.AI_SERVER_HEALTH,
            {
                "status": "startup",
                "message": "Watchdog process started",
                "uptime_seconds": 0,
            },
        )

        try:
            while True:
                try:
                    results = await self.run_checks()
                    for name, res in results.items():
                        level = "OK" if res.healthy else "FAIL"
                        logger.debug("[%s] %s: %s", level, name, res.message)
                except Exception as exc:
                    logger.error("Watchdog check cycle failed: %s", exc)

                await asyncio.sleep(self._check_interval)
        finally:
            await self._bot.stop()
            logger.info("Watchdog stopped")
