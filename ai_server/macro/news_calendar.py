"""ForexFactory calendar parser + news risk scorer.

Fetches economic calendar XML, classifies event impact, and calculates
a composite news risk score (0-100) used by both EAs and the AI server.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

# Currencies that impact XAUUSD
GOLD_CURRENCIES = {"USD", "EUR", "CNY", "JPY", "CHF"}

# Event title keywords -> extra impact boost
HIGH_IMPACT_KEYWORDS = {
    "Non-Farm": 3,
    "NFP": 3,
    "CPI": 3,
    "FOMC": 3,
    "Fed Chair": 3,
    "Interest Rate": 3,
    "GDP": 2,
    "PPI": 2,
    "Retail Sales": 2,
    "ISM": 2,
    "PCE": 2,
    "Unemployment": 2,
}


@dataclass
class NewsEvent:
    time: datetime
    currency: str
    impact: int       # 0=none, 1=low, 2=medium, 3=high
    title: str = ""
    forecast: str = ""
    previous: str = ""

    @property
    def is_high_impact(self) -> bool:
        return self.impact >= 3

    @property
    def affects_gold(self) -> bool:
        return self.currency in GOLD_CURRENCIES


@dataclass
class NewsRiskResult:
    score: int = 0                    # 0-100
    nearest_event: NewsEvent | None = None
    minutes_to_nearest: float = 999
    phase: str = "none"               # none, detection, pre, during, post


class NewsCalendar:
    """Economic calendar with ForexFactory XML parsing and risk scoring."""

    def __init__(self) -> None:
        self._events: list[NewsEvent] = []
        self._last_fetch: datetime | None = None
        self._fetch_interval = timedelta(minutes=30)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def update(self) -> None:
        """Fetch latest calendar data from ForexFactory XML feed."""
        now = datetime.now(timezone.utc)
        if self._last_fetch and (now - self._last_fetch) < self._fetch_interval:
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    FOREX_FACTORY_URL,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        return
                    text = await resp.text()

            self._events = self._parse_xml(text)
            self._last_fetch = now
        except Exception:
            pass  # Keep existing events on fetch failure

    def _parse_xml(self, xml_text: str) -> list[NewsEvent]:
        """Parse ForexFactory XML into NewsEvent list."""
        events = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return events

        for item in root.iter("event"):
            title = _text(item, "title")
            currency = _text(item, "country").upper()
            impact_str = _text(item, "impact").lower()
            date_str = _text(item, "date")
            time_str = _text(item, "time")

            # Parse impact level
            if "high" in impact_str:
                impact = 3
            elif "medium" in impact_str:
                impact = 2
            elif "low" in impact_str:
                impact = 1
            else:
                impact = 0

            # Boost impact for known high-impact events
            for keyword, boost_level in HIGH_IMPACT_KEYWORDS.items():
                if keyword.lower() in title.lower():
                    impact = max(impact, boost_level)
                    break

            # Parse datetime
            event_time = _parse_ff_datetime(date_str, time_str)
            if event_time is None:
                continue

            events.append(NewsEvent(
                time=event_time,
                currency=currency,
                impact=impact,
                title=title,
                forecast=_text(item, "forecast"),
                previous=_text(item, "previous"),
            ))

        return sorted(events, key=lambda e: e.time)

    def set_events(self, events: list[NewsEvent]) -> None:
        """Manually set events (for testing or file-based loading)."""
        self._events = sorted(events, key=lambda e: e.time)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_upcoming_events(self, hours_ahead: int = 4) -> list[NewsEvent]:
        """Return gold-affecting events within the next N hours."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        return [
            e for e in self._events
            if e.affects_gold and now <= e.time <= cutoff
        ]

    def has_event_within(self, minutes: int) -> bool:
        """Check if any high-impact gold event is within N minutes."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=minutes)
        return any(
            e.is_high_impact and e.affects_gold and now <= e.time <= cutoff
            for e in self._events
        )

    def get_nearest_event(self) -> tuple[NewsEvent | None, float]:
        """Return nearest future gold event and minutes until it.

        Returns (None, 999) if no upcoming event.
        """
        now = datetime.now(timezone.utc)
        for e in self._events:
            if e.affects_gold and e.time > now - timedelta(minutes=90):
                minutes = (e.time - now).total_seconds() / 60
                return e, minutes
        return None, 999.0

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def get_news_risk_score(self) -> int:
        """Return composite news risk 0-100.

        Score logic:
        - Base: 0 (no events nearby)
        - High-impact within 60 min: 40-80
        - High-impact within 30 min: 60-90
        - During event (0-20 min after): 90-100
        - Post-event (20-75 min after): 30-50
        - Multiple events compound
        """
        return self.get_news_risk().score

    def get_news_risk(self) -> NewsRiskResult:
        """Return detailed news risk assessment."""
        result = NewsRiskResult()
        now = datetime.now(timezone.utc)

        for event in self._events:
            if not event.affects_gold:
                continue

            minutes = (event.time - now).total_seconds() / 60

            # Skip events more than 2 hours away or more than 90 min past
            if minutes > 120 or minutes < -90:
                continue

            # Track nearest
            abs_min = abs(minutes)
            if abs_min < abs(result.minutes_to_nearest):
                result.nearest_event = event
                result.minutes_to_nearest = minutes

            # Score based on proximity and impact
            impact_mult = event.impact / 3.0  # 0.0 to 1.0

            if 0 <= minutes <= 20:
                # During: T-0 to T+20
                score = int(85 + 15 * impact_mult)
                result.phase = "during"
            elif -20 < minutes < 0:
                # Just passed but within during window
                score = int(80 + 15 * impact_mult)
                result.phase = "during"
            elif 0 < minutes <= 30:
                # Pre-news: T-30 to T-0
                score = int(55 + 35 * impact_mult)
                result.phase = "pre"
            elif 0 < minutes <= 60:
                # Detection: T-60 to T-30
                score = int(30 + 30 * impact_mult)
                result.phase = "detection"
            elif -75 <= minutes < -20:
                # Post-news: T+20 to T+75
                score = int(25 + 25 * impact_mult)
                result.phase = "post"
            else:
                score = int(10 * impact_mult)

            result.score = max(result.score, score)

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    return child.text.strip() if child is not None and child.text else ""


def _parse_ff_datetime(date_str: str, time_str: str) -> datetime | None:
    """Parse ForexFactory date/time strings into UTC datetime."""
    if not date_str:
        return None

    # ForexFactory uses formats like "01-03-2026" and "8:30am"
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    else:
        return None

    if time_str and time_str.lower() not in ("", "all day", "tentative"):
        try:
            # "8:30am" or "2:00pm"
            t = datetime.strptime(time_str.strip(), "%I:%M%p")
            dt = dt.replace(hour=t.hour, minute=t.minute)
        except ValueError:
            pass

    return dt.replace(tzinfo=timezone.utc)
