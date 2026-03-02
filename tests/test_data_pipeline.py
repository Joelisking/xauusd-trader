"""Tests for data pipeline — validator, macro clients, news calendar."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.data_validator import ValidationReport, validate_parquet
from ai_server.macro.news_calendar import NewsCalendar, NewsEvent


# ---------------------------------------------------------------------------
# Data Validator tests
# ---------------------------------------------------------------------------


class TestDataValidator:
    def _make_parquet(self, tmp_path: Path, **kwargs) -> Path:
        """Create a minimal valid Parquet file."""
        n = kwargs.get("n", 100)
        base_time = datetime(2025, 1, 2, 8, 0, tzinfo=timezone.utc)
        data = {
            "time": [base_time + timedelta(minutes=i) for i in range(n)],
            "open": np.random.uniform(3000, 3100, n),
            "high": np.random.uniform(3050, 3150, n),
            "low": np.random.uniform(2950, 3050, n),
            "close": np.random.uniform(3000, 3100, n),
            "tick_volume": np.random.randint(100, 5000, n),
            "spread": np.random.randint(1, 5, n),
            "real_volume": np.zeros(n),
        }
        df = pd.DataFrame(data)
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)
        return path

    def test_valid_file(self, tmp_path):
        path = self._make_parquet(tmp_path)
        report = validate_parquet(path)
        assert report.nan_count == 0
        assert report.inf_count == 0
        assert report.total_rows == 100

    def test_nan_detection(self, tmp_path):
        path = self._make_parquet(tmp_path)
        df = pd.read_parquet(path)
        df.loc[5, "close"] = np.nan
        df.loc[10, "open"] = np.nan
        df.to_parquet(path, index=False)

        report = validate_parquet(path)
        assert report.nan_count == 2
        assert not report.is_valid

    def test_inf_detection(self, tmp_path):
        path = self._make_parquet(tmp_path)
        df = pd.read_parquet(path)
        df.loc[0, "high"] = np.inf
        df.to_parquet(path, index=False)

        report = validate_parquet(path)
        assert report.inf_count >= 1

    def test_time_gap_detection(self, tmp_path):
        """Insert a 3-hour gap during market hours — should be flagged."""
        n = 100
        base_time = datetime(2025, 1, 6, 8, 0, tzinfo=timezone.utc)  # Monday
        times = [base_time + timedelta(minutes=i) for i in range(50)]
        # 3-hour gap
        times += [base_time + timedelta(minutes=50 + 180 + i) for i in range(50)]

        data = {
            "time": times,
            "open": np.random.uniform(3000, 3100, n),
            "high": np.random.uniform(3050, 3150, n),
            "low": np.random.uniform(2950, 3050, n),
            "close": np.random.uniform(3000, 3100, n),
            "tick_volume": np.random.randint(100, 5000, n),
            "spread": np.random.randint(1, 5, n),
            "real_volume": np.zeros(n),
        }
        df = pd.DataFrame(data)
        path = tmp_path / "gaps.parquet"
        df.to_parquet(path, index=False)

        report = validate_parquet(path)
        assert report.time_gaps >= 1
        assert report.max_gap_minutes >= 180

    def test_report_summary(self, tmp_path):
        path = self._make_parquet(tmp_path)
        report = validate_parquet(path)
        summary = report.summary()
        assert "PASS" in summary
        assert "100" in summary


# ---------------------------------------------------------------------------
# News Calendar tests
# ---------------------------------------------------------------------------


class TestNewsCalendar:
    def _make_event(self, minutes_from_now: float, impact: int = 3, currency: str = "USD") -> NewsEvent:
        return NewsEvent(
            time=datetime.now(timezone.utc) + timedelta(minutes=minutes_from_now),
            currency=currency,
            impact=impact,
            title="Test Event",
        )

    def test_no_events_score_zero(self):
        cal = NewsCalendar()
        assert cal.get_news_risk_score() == 0

    def test_high_impact_near_scores_high(self):
        cal = NewsCalendar()
        cal.set_events([self._make_event(10, impact=3)])  # 10 min away
        score = cal.get_news_risk_score()
        assert score >= 55  # pre-news phase

    def test_during_event_scores_highest(self):
        cal = NewsCalendar()
        cal.set_events([self._make_event(-5, impact=3)])  # 5 min ago
        score = cal.get_news_risk_score()
        assert score >= 80

    def test_non_gold_currency_ignored(self):
        cal = NewsCalendar()
        cal.set_events([self._make_event(10, impact=3, currency="AUD")])
        assert cal.get_news_risk_score() == 0

    def test_has_event_within(self):
        cal = NewsCalendar()
        cal.set_events([self._make_event(20, impact=3)])
        assert cal.has_event_within(30) is True
        assert cal.has_event_within(10) is False

    def test_get_upcoming_events(self):
        cal = NewsCalendar()
        cal.set_events([
            self._make_event(30, impact=3),
            self._make_event(300, impact=2),  # 5 hours away
        ])
        upcoming = cal.get_upcoming_events(hours_ahead=1)
        assert len(upcoming) == 1

    def test_nearest_event(self):
        cal = NewsCalendar()
        e1 = self._make_event(60, impact=3)
        e2 = self._make_event(120, impact=2)
        cal.set_events([e1, e2])
        nearest, minutes = cal.get_nearest_event()
        assert nearest is not None
        assert 55 < minutes < 65

    def test_post_news_phase(self):
        cal = NewsCalendar()
        cal.set_events([self._make_event(-40, impact=3)])  # 40 min ago
        risk = cal.get_news_risk()
        assert risk.phase == "post"
        assert 20 <= risk.score <= 60
