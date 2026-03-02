"""Data validation — NaN/inf/outlier/gap detection for OHLCV Parquet files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    file_path: str = ""
    total_rows: int = 0
    nan_count: int = 0
    inf_count: int = 0
    outlier_count: int = 0
    time_gaps: int = 0
    max_gap_minutes: float = 0.0
    completeness_pct: float = 0.0
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0

    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        lines = [
            f"[{status}] {self.file_path}",
            f"  Rows: {self.total_rows:,}",
            f"  NaN: {self.nan_count}, Inf: {self.inf_count}, Outliers (>5σ): {self.outlier_count}",
            f"  Time gaps (>1h in market hours): {self.time_gaps}, max gap: {self.max_gap_minutes:.0f} min",
            f"  Completeness: {self.completeness_pct:.1f}%",
        ]
        for issue in self.issues:
            lines.append(f"  WARNING: {issue}")
        return "\n".join(lines)


# Market hours: Sunday 22:00 UTC to Friday 22:00 UTC (gold/forex)
def _is_market_hours(dt: pd.Timestamp) -> bool:
    wd = dt.weekday()  # Mon=0 .. Sun=6
    if wd == 5:  # Saturday
        return False
    if wd == 6 and dt.hour < 22:  # Sunday before 22:00
        return False
    if wd == 4 and dt.hour >= 22:  # Friday after 22:00
        return False
    return True


def validate_parquet(path: Path | str, max_gap_minutes: float = 60.0) -> ValidationReport:
    """Validate a single Parquet OHLCV file.

    Args:
        path: Path to the Parquet file.
        max_gap_minutes: Flag gaps larger than this during market hours.
    """
    path = Path(path)
    report = ValidationReport(file_path=str(path))

    df = pd.read_parquet(path)
    report.total_rows = len(df)

    if report.total_rows == 0:
        report.issues.append("Empty file")
        return report

    # NaN check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report.nan_count = int(df[numeric_cols].isna().sum().sum())
    if report.nan_count > 0:
        per_col = df[numeric_cols].isna().sum()
        bad = per_col[per_col > 0]
        for col, cnt in bad.items():
            report.issues.append(f"NaN in {col}: {cnt}")

    # Inf check
    inf_mask = df[numeric_cols].apply(lambda s: np.isinf(s.astype(float)))
    report.inf_count = int(inf_mask.sum().sum())
    if report.inf_count > 0:
        report.issues.append(f"Infinity values found: {report.inf_count}")

    # Outlier check (>5 sigma from mean)
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 100:
            continue
        mean, std = s.mean(), s.std()
        if std == 0:
            continue
        outliers = ((s - mean).abs() > 5 * std).sum()
        report.outlier_count += int(outliers)
        if outliers > 0:
            report.issues.append(f"Outliers in {col}: {outliers} (>5σ)")

    # Time gap detection
    if "time" in df.columns:
        times = pd.to_datetime(df["time"], utc=True)
        diffs = times.diff().dropna()

        for i, gap in enumerate(diffs):
            gap_min = gap.total_seconds() / 60
            ts = times.iloc[i + 1]

            if gap_min > max_gap_minutes and _is_market_hours(ts):
                report.time_gaps += 1
                if gap_min > report.max_gap_minutes:
                    report.max_gap_minutes = gap_min

        # Completeness: estimate expected bars
        total_span = (times.iloc[-1] - times.iloc[0]).total_seconds() / 60
        if total_span > 0:
            # Rough: 5 days/week * fraction of total span
            expected = total_span * (5 / 7)  # exclude weekends
            report.completeness_pct = min(100.0, (report.total_rows / max(1, expected)) * 100)

    # Timezone check
    if "time" in df.columns:
        sample = pd.to_datetime(df["time"].iloc[0], utc=True)
        if sample.tzinfo is None:
            report.issues.append("Timestamps have no timezone — expected UTC")

    return report


def validate_all(data_dir: Path | str, symbol: str = "XAUUSD") -> list[ValidationReport]:
    """Validate all Parquet files for a symbol."""
    data_dir = Path(data_dir)
    reports = []
    for tf in ("M1", "M5", "H1", "H4"):
        path = data_dir / f"{symbol}_{tf}.parquet"
        if path.exists():
            reports.append(validate_parquet(path))
        else:
            r = ValidationReport(file_path=str(path))
            r.issues.append("File not found")
            reports.append(r)
    return reports


def main() -> None:
    from ai_server.config import DATA_DIR

    reports = validate_all(DATA_DIR)
    for r in reports:
        print(r.summary())
        print()


if __name__ == "__main__":
    main()
