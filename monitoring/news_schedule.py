"""News schedule writer — produces news_schedule.json for MQL5 EAs.

Reads from the news calendar service and writes a JSON file that the
MQL5 NewsShield includes can parse on each tick.  Run as a periodic
task (every 5 minutes) alongside the macro updater.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai_server.config import DATA_DIR

logger = logging.getLogger(__name__)

# Output file path — MQL5 DXYFilter reads from DATA_DIR
NEWS_SCHEDULE_PATH = DATA_DIR / "news_schedule.json"

# Event timing windows per event type (minutes)
EVENT_WINDOWS: dict[str, dict[str, int]] = {
    "NFP": {"detection": 60, "pre": 30, "during": 20, "post": 90},
    "CPI": {"detection": 30, "pre": 30, "during": 20, "post": 75},
    "FOMC": {"detection": 60, "pre": 60, "during": 30, "post": 120},
    "FED_SPEECH": {"detection": 30, "pre": 30, "during": 15, "post": 60},
    "HIGH_IMPACT": {"detection": 30, "pre": 30, "during": 20, "post": 75},
}


def classify_event_type(event_name: str) -> str:
    """Map event name to type key for timing windows."""
    name = event_name.upper()
    if "NON-FARM" in name or "NFP" in name or "NONFARM" in name:
        return "NFP"
    if "CPI" in name or "CONSUMER PRICE" in name:
        return "CPI"
    if "FOMC" in name or "FEDERAL OPEN MARKET" in name or "FED RATE" in name:
        return "FOMC"
    if "FED" in name and ("SPEAK" in name or "CHAIR" in name or "TESTIMONY" in name):
        return "FED_SPEECH"
    return "HIGH_IMPACT"


def compute_phase(event_time: datetime, now: datetime, event_type: str) -> dict[str, Any]:
    """Compute the current news phase for a given event.

    Returns dict with: phase, minutes_until, minutes_since, event_type.
    """
    windows = EVENT_WINDOWS.get(event_type, EVENT_WINDOWS["HIGH_IMPACT"])
    minutes_until = (event_time - now).total_seconds() / 60

    if minutes_until > windows["detection"]:
        return {"phase": "NONE", "minutes_until": minutes_until, "event_type": event_type}
    elif minutes_until > windows["pre"]:
        return {"phase": "DETECTION", "minutes_until": minutes_until, "event_type": event_type}
    elif minutes_until > 0:
        return {"phase": "PRE", "minutes_until": minutes_until, "event_type": event_type}
    elif -minutes_until <= windows["during"]:
        return {"phase": "DURING", "minutes_since": -minutes_until, "event_type": event_type}
    elif -minutes_until <= windows["during"] + windows["post"]:
        return {"phase": "POST", "minutes_since": -minutes_until, "event_type": event_type}
    else:
        return {"phase": "NONE", "minutes_since": -minutes_until, "event_type": event_type}


def build_schedule(events: list[dict], now: datetime | None = None) -> dict:
    """Build the news schedule JSON structure.

    Args:
        events: List of dicts with keys: name, time (ISO string), impact (1-3), currency
        now: Current time (defaults to UTC now)

    Returns:
        Dict ready to be written as JSON for MQL5 consumption.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Filter to next 24 hours of events affecting gold (USD-related)
    upcoming = []
    for event in events:
        currency = event.get("currency", "USD").upper()
        if currency not in ("USD", "ALL"):
            continue

        try:
            event_time = datetime.fromisoformat(event["time"])
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
        except (ValueError, KeyError):
            continue

        # Only include events in next 24 hours
        if event_time < now - timedelta(hours=2):
            continue
        if event_time > now + timedelta(hours=24):
            continue

        event_type = classify_event_type(event.get("name", ""))
        phase_info = compute_phase(event_time, now, event_type)

        upcoming.append({
            "name": event.get("name", "Unknown"),
            "time": event_time.isoformat(),
            "impact": event.get("impact", 2),
            "currency": currency,
            "event_type": event_type,
            "phase": phase_info["phase"],
            "minutes_until": phase_info.get("minutes_until", 0),
            "minutes_since": phase_info.get("minutes_since", 0),
        })

    # Sort by time
    upcoming.sort(key=lambda e: e["time"])

    # Find the most active/nearest event
    active_event = None
    for ev in upcoming:
        if ev["phase"] in ("DETECTION", "PRE", "DURING", "POST"):
            active_event = ev
            break

    # Compute overall shield status
    if active_event and active_event["phase"] == "DURING":
        shield_active = True
        shield_phase = "DURING"
    elif active_event and active_event["phase"] == "PRE":
        shield_active = True
        shield_phase = "PRE"
    elif active_event and active_event["phase"] == "DETECTION":
        shield_active = False  # Detection phase = warning only, not blocking
        shield_phase = "DETECTION"
    elif active_event and active_event["phase"] == "POST":
        shield_active = False  # Post phase = reduced risk entries allowed
        shield_phase = "POST"
    else:
        shield_active = False
        shield_phase = "NONE"

    return {
        "updated_at": now.isoformat(),
        "shield_active": shield_active,
        "shield_phase": shield_phase,
        "active_event": active_event,
        "upcoming_events": upcoming,
        "next_event_minutes": upcoming[0].get("minutes_until", 999) if upcoming else 999,
    }


def write_schedule(schedule: dict, path: Path = NEWS_SCHEDULE_PATH) -> None:
    """Write the schedule JSON file for MQL5 EAs to read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schedule, f, indent=2, default=str)
    logger.info("News schedule written to %s (%d events)", path, len(schedule.get("upcoming_events", [])))


async def update_news_schedule(
    calendar_fn=None,
    path: Path = NEWS_SCHEDULE_PATH,
) -> dict:
    """Fetch events from news calendar and write schedule file.

    Args:
        calendar_fn: Async callable that returns list of event dicts.
                    If None, uses the built-in news calendar client.
        path: Output file path.

    Returns:
        The schedule dict that was written.
    """
    if calendar_fn is None:
        from ai_server.macro.news_calendar import NewsCalendar
        calendar = NewsCalendar()
        try:
            events = await calendar.fetch_events()
        except Exception as exc:
            logger.error("Failed to fetch news calendar: %s", exc)
            events = []
    else:
        events = await calendar_fn()

    schedule = build_schedule(events)
    write_schedule(schedule, path)
    return schedule


async def run_schedule_loop(interval_seconds: int = 300) -> None:
    """Run the news schedule updater in a loop (every 5 minutes)."""
    logger.info("News schedule updater started (interval=%ds)", interval_seconds)
    while True:
        try:
            await update_news_schedule()
        except Exception as exc:
            logger.error("News schedule update failed: %s", exc)
        await asyncio.sleep(interval_seconds)
