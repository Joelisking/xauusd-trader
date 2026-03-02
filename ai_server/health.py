"""Health tracking — uptime, prediction count, latency, queue depth."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ai_server.config import MODEL_VERSION


@dataclass
class HealthTracker:
    """Tracks server health metrics. Thread-safe is not required — single event loop."""

    start_time: float = field(default_factory=time.time)
    predictions_today: int = 0
    _latency_sum_ms: float = 0.0
    _latency_count: int = 0
    queue_depth: int = 0
    model_version: str = MODEL_VERSION
    _last_reset_day: int = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_prediction(self, latency_ms: float) -> None:
        self._maybe_reset_daily()
        self.predictions_today += 1
        self._latency_sum_ms += latency_ms
        self._latency_count += 1

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self.start_time)

    @property
    def avg_latency_ms(self) -> int:
        if self._latency_count == 0:
            return 0
        return int(self._latency_sum_ms / self._latency_count)

    @property
    def status(self) -> str:
        if self.avg_latency_ms > 300:
            return "degraded"
        return "healthy"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "uptime_seconds": self.uptime_seconds,
            "model_version": self.model_version,
            "predictions_today": self.predictions_today,
            "avg_latency_ms": self.avg_latency_ms,
            "queue_depth": self.queue_depth,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self) -> None:
        today = time.gmtime().tm_yday
        if today != self._last_reset_day:
            self.predictions_today = 0
            self._latency_sum_ms = 0.0
            self._latency_count = 0
            self._last_reset_day = today
