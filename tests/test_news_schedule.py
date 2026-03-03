"""Tests for news schedule writer — Phase 11."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import tempfile

import pytest

from monitoring.news_schedule import (
    build_schedule,
    classify_event_type,
    compute_phase,
    write_schedule,
)


class TestClassifyEventType:
    def test_nfp(self):
        assert classify_event_type("Non-Farm Payrolls") == "NFP"
        assert classify_event_type("NFP Release") == "NFP"

    def test_cpi(self):
        assert classify_event_type("CPI m/m") == "CPI"
        assert classify_event_type("Consumer Price Index") == "CPI"

    def test_fomc(self):
        assert classify_event_type("FOMC Statement") == "FOMC"
        assert classify_event_type("Federal Open Market Committee") == "FOMC"
        assert classify_event_type("Fed Rate Decision") == "FOMC"

    def test_fed_speech(self):
        assert classify_event_type("Fed Chair Speaks") == "FED_SPEECH"

    def test_default_high_impact(self):
        assert classify_event_type("GDP q/q") == "HIGH_IMPACT"
        assert classify_event_type("Retail Sales") == "HIGH_IMPACT"


class TestComputePhase:
    def test_none_phase_far_event(self):
        now = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(hours=5)
        result = compute_phase(event_time, now, "HIGH_IMPACT")
        assert result["phase"] == "NONE"

    def test_detection_phase(self):
        now = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=45)  # 45 min before NFP
        result = compute_phase(event_time, now, "NFP")
        assert result["phase"] == "DETECTION"

    def test_pre_phase(self):
        now = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=15)
        result = compute_phase(event_time, now, "NFP")
        assert result["phase"] == "PRE"

    def test_during_phase(self):
        now = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
        event_time = now - timedelta(minutes=5)
        result = compute_phase(event_time, now, "NFP")
        assert result["phase"] == "DURING"

    def test_post_phase(self):
        now = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
        event_time = now - timedelta(minutes=30)
        result = compute_phase(event_time, now, "NFP")
        assert result["phase"] == "POST"


class TestBuildSchedule:
    def test_empty_events(self):
        schedule = build_schedule([])
        assert schedule["shield_active"] is False
        assert schedule["shield_phase"] == "NONE"
        assert len(schedule["upcoming_events"]) == 0

    def test_upcoming_event(self):
        now = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
        events = [{
            "name": "Non-Farm Payrolls",
            "time": (now + timedelta(minutes=20)).isoformat(),
            "impact": 3,
            "currency": "USD",
        }]
        schedule = build_schedule(events, now=now)
        assert len(schedule["upcoming_events"]) == 1
        assert schedule["upcoming_events"][0]["event_type"] == "NFP"
        assert schedule["upcoming_events"][0]["phase"] == "PRE"
        assert schedule["shield_active"] is True

    def test_during_event_blocks(self):
        now = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
        events = [{
            "name": "CPI m/m",
            "time": (now - timedelta(minutes=5)).isoformat(),
            "impact": 3,
            "currency": "USD",
        }]
        schedule = build_schedule(events, now=now)
        assert schedule["shield_active"] is True
        assert schedule["shield_phase"] == "DURING"

    def test_filters_non_usd_events(self):
        now = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
        events = [
            {"name": "EUR CPI", "time": (now + timedelta(hours=1)).isoformat(), "impact": 3, "currency": "EUR"},
            {"name": "US GDP", "time": (now + timedelta(hours=2)).isoformat(), "impact": 3, "currency": "USD"},
        ]
        schedule = build_schedule(events, now=now)
        assert len(schedule["upcoming_events"]) == 1
        assert schedule["upcoming_events"][0]["name"] == "US GDP"

    def test_filters_old_events(self):
        now = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
        events = [{
            "name": "Old Event",
            "time": (now - timedelta(hours=5)).isoformat(),
            "impact": 3,
            "currency": "USD",
        }]
        schedule = build_schedule(events, now=now)
        assert len(schedule["upcoming_events"]) == 0


class TestWriteSchedule:
    def test_write_to_file(self):
        schedule = build_schedule([])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "news_schedule.json"
            write_schedule(schedule, path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "shield_active" in data
            assert "upcoming_events" in data
