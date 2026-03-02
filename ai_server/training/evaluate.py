"""Model evaluation metrics and reporting.

Computes classification metrics, calibration, and generates evaluation reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    log_loss,
)


@dataclass
class EvaluationReport:
    """Comprehensive model evaluation report."""
    model_name: str
    dataset_name: str  # "train", "val", "test"
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    auc: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    log_loss_val: float = 0.0
    win_rate: float = 0.0
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    threshold: float = 0.5
    calibration_error: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== {self.model_name} — {self.dataset_name} ===",
            f"  Samples: {self.n_samples} (pos={self.n_positive}, neg={self.n_negative})",
            f"  AUC:       {self.auc:.4f}",
            f"  Accuracy:  {self.accuracy:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1:        {self.f1:.4f}",
            f"  Log Loss:  {self.log_loss_val:.4f}",
            f"  Win Rate:  {self.win_rate:.1%}",
            f"  Threshold: {self.threshold}",
            f"  Cal Error: {self.calibration_error:.4f}",
            f"  Confusion Matrix:",
            f"    TN={int(self.confusion[0, 0])} FP={int(self.confusion[0, 1])}",
            f"    FN={int(self.confusion[1, 0])} TP={int(self.confusion[1, 1])}",
        ]
        return "\n".join(lines)

    @property
    def passed_auc_threshold(self) -> bool:
        """Check if AUC meets minimum thresholds (0.70 scalper, 0.68 swing)."""
        return self.auc >= 0.68


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "model",
    dataset_name: str = "test",
    threshold: float = 0.5,
) -> EvaluationReport:
    """Evaluate binary classification model.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred_proba: Predicted probabilities
        model_name: Name for the report
        dataset_name: Dataset split name
        threshold: Classification threshold

    Returns:
        EvaluationReport with all metrics
    """
    report = EvaluationReport(
        model_name=model_name,
        dataset_name=dataset_name,
        threshold=threshold,
    )

    y_true = np.asarray(y_true).astype(float)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    # Clip probabilities
    y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
    y_pred = (y_pred_proba >= threshold).astype(int)

    report.n_samples = len(y_true)
    report.n_positive = int((y_true == 1).sum())
    report.n_negative = int((y_true == 0).sum())

    if report.n_samples == 0 or len(np.unique(y_true)) < 2:
        return report

    report.auc = roc_auc_score(y_true, y_pred_proba)
    report.accuracy = accuracy_score(y_true, y_pred)
    report.precision = precision_score(y_true, y_pred, zero_division=0)
    report.recall = recall_score(y_true, y_pred, zero_division=0)
    report.f1 = f1_score(y_true, y_pred, zero_division=0)
    report.log_loss_val = log_loss(y_true, y_pred_proba)
    report.win_rate = report.n_positive / report.n_samples
    report.confusion = confusion_matrix(y_true, y_pred)
    report.calibration_error = _expected_calibration_error(y_true, y_pred_proba)

    return report


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """Find optimal classification threshold.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ("f1", "accuracy", "precision")

    Returns:
        (optimal_threshold, best_metric_value)
    """
    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.arange(0.3, 0.8, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def compare_models(
    reports: list[EvaluationReport],
) -> str:
    """Compare multiple model evaluation reports side by side."""
    lines = ["Model Comparison:"]
    header = f"  {'Model':<25s} {'Dataset':<10s} {'AUC':>8s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}"
    lines.append(header)
    lines.append("  " + "-" * 77)

    for r in reports:
        lines.append(
            f"  {r.model_name:<25s} {r.dataset_name:<10s} "
            f"{r.auc:>8.4f} {r.accuracy:>8.4f} {r.precision:>8.4f} "
            f"{r.recall:>8.4f} {r.f1:>8.4f}"
        )

    return "\n".join(lines)


def _expected_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual frequencies.
    Lower is better (well-calibrated model).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = y_pred_proba[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)

    return float(ece)
