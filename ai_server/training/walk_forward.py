"""Walk-forward validation framework.

12 segments over 5-year test period. Each segment:
- 8 months train, 2 months test, 1-month gap (no data leakage).
- Average performance must exceed minimum benchmarks across ALL 12 segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


@dataclass
class WalkForwardSegment:
    """Results for a single walk-forward segment."""
    segment_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int = 0
    test_samples: int = 0
    auc: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    win_rate: float = 0.0
    passed: bool = False


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward validation results."""
    segments: list[WalkForwardSegment] = field(default_factory=list)
    mean_auc: float = 0.0
    mean_accuracy: float = 0.0
    min_auc: float = 0.0
    all_passed: bool = False

    def summary(self) -> str:
        lines = [
            f"Walk-Forward Validation: {len(self.segments)} segments",
            f"  Mean AUC: {self.mean_auc:.4f}",
            f"  Min AUC:  {self.min_auc:.4f}",
            f"  Mean Acc: {self.mean_accuracy:.4f}",
            f"  All Pass: {self.all_passed}",
        ]
        for seg in self.segments:
            status = "PASS" if seg.passed else "FAIL"
            lines.append(
                f"  Seg {seg.segment_id:2d}: AUC={seg.auc:.4f} "
                f"Acc={seg.accuracy:.4f} WR={seg.win_rate:.1%} "
                f"Train={seg.train_samples} Test={seg.test_samples} [{status}]"
            )
        return "\n".join(lines)


def generate_segments(
    start_date: datetime,
    end_date: datetime,
    n_segments: int = 12,
    train_months: int = 8,
    test_months: int = 2,
    gap_months: int = 1,
) -> list[dict[str, datetime]]:
    """Generate walk-forward segment date ranges.

    Returns list of dicts with keys: train_start, train_end, test_start, test_end
    """
    total_months = train_months + gap_months + test_months
    segments = []

    for i in range(n_segments):
        offset_months = i * test_months  # Slide by test_months each segment
        seg_start = _add_months(start_date, offset_months)
        train_end = _add_months(seg_start, train_months)
        test_start = _add_months(train_end, gap_months)
        test_end = _add_months(test_start, test_months)

        if test_end > end_date:
            break

        segments.append({
            "train_start": seg_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

    return segments


def run_walk_forward(
    times: np.ndarray,
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    train_fn: Callable,
    predict_fn: Callable,
    n_segments: int = 12,
    min_auc: float = 0.65,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation.

    Args:
        times: Array of timestamps for each sample
        X_seq: Sequential features (samples, seq_len, features)
        X_tab: Tabular features (samples, features)
        y: Binary labels
        train_fn: Callable(X_seq_train, X_tab_train, y_train) -> model
        predict_fn: Callable(model, X_seq_test, X_tab_test) -> probabilities
        n_segments: Number of walk-forward segments
        min_auc: Minimum AUC threshold for passing
        start_date: Override start date
        end_date: Override end date

    Returns:
        WalkForwardResult with all segment metrics
    """
    times_dt = pd.to_datetime(times)

    if start_date is None:
        start_date = times_dt.min().to_pydatetime()
    if end_date is None:
        end_date = times_dt.max().to_pydatetime()

    segments = generate_segments(start_date, end_date, n_segments)
    result = WalkForwardResult()

    for i, seg_dates in enumerate(segments):
        seg = WalkForwardSegment(
            segment_id=i + 1,
            train_start=seg_dates["train_start"],
            train_end=seg_dates["train_end"],
            test_start=seg_dates["test_start"],
            test_end=seg_dates["test_end"],
        )

        # Split data by time
        train_mask = (times_dt >= seg.train_start) & (times_dt < seg.train_end)
        test_mask = (times_dt >= seg.test_start) & (times_dt < seg.test_end)

        X_seq_train = X_seq[train_mask]
        X_tab_train = X_tab[train_mask]
        y_train = y[train_mask]

        X_seq_test = X_seq[test_mask]
        X_tab_test = X_tab[test_mask]
        y_test = y[test_mask]

        seg.train_samples = len(y_train)
        seg.test_samples = len(y_test)

        # Skip if insufficient data
        if seg.train_samples < 100 or seg.test_samples < 20:
            result.segments.append(seg)
            continue

        # Skip if only one class in train or test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            result.segments.append(seg)
            continue

        # Train and predict
        model = train_fn(X_seq_train, X_tab_train, y_train)
        y_pred_proba = predict_fn(model, X_seq_test, X_tab_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        seg.auc = roc_auc_score(y_test, y_pred_proba)
        seg.accuracy = accuracy_score(y_test, y_pred)
        seg.precision = precision_score(y_test, y_pred, zero_division=0)
        seg.recall = recall_score(y_test, y_pred, zero_division=0)
        seg.win_rate = y_pred[y_pred == 1].sum() / max(len(y_pred), 1) if len(y_pred) > 0 else 0
        seg.passed = seg.auc >= min_auc

        result.segments.append(seg)

    # Aggregate
    aucs = [s.auc for s in result.segments if s.test_samples >= 20]
    accs = [s.accuracy for s in result.segments if s.test_samples >= 20]

    result.mean_auc = float(np.mean(aucs)) if aucs else 0.0
    result.min_auc = float(np.min(aucs)) if aucs else 0.0
    result.mean_accuracy = float(np.mean(accs)) if accs else 0.0
    result.all_passed = all(s.passed for s in result.segments if s.test_samples >= 20)

    return result


def _add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime."""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, 28)  # Safe day
    return dt.replace(year=year, month=month, day=day)
