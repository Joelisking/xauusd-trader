"""SHAP-based feature importance ranking and selection.

Uses SHAP values to identify the most predictive features.
Target: limit to 40 most important features (per architecture spec).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureImportanceResult:
    """Feature importance analysis results."""
    feature_names: list[str]
    importance_scores: np.ndarray
    top_k_indices: np.ndarray
    top_k_names: list[str]

    def summary(self, top_n: int = 20) -> str:
        lines = [f"Feature Importance (top {top_n}):"]
        sorted_idx = np.argsort(self.importance_scores)[::-1]
        for rank, idx in enumerate(sorted_idx[:top_n], 1):
            lines.append(f"  {rank:3d}. {self.feature_names[idx]:30s} {self.importance_scores[idx]:.6f}")
        return "\n".join(lines)


def compute_shap_importance(
    model: object,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 1000,
) -> FeatureImportanceResult:
    """Compute SHAP-based feature importance for an XGBoost model.

    Args:
        model: Trained model (XGBoost or any tree-based model with predict)
        X: Feature matrix (samples, features)
        feature_names: Optional feature name list
        max_samples: Max samples to use for SHAP computation (speed)

    Returns:
        FeatureImportanceResult with scores and rankings
    """
    import shap

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    # Subsample for speed
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    # Use TreeExplainer for tree-based models, KernelExplainer as fallback
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, X_sample[:100])
        shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP values per feature
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification, use class 1
    importance = np.abs(shap_values).mean(axis=0)

    return FeatureImportanceResult(
        feature_names=feature_names,
        importance_scores=importance,
        top_k_indices=np.argsort(importance)[::-1],
        top_k_names=[feature_names[i] for i in np.argsort(importance)[::-1]],
    )


def select_top_features(
    importance: FeatureImportanceResult,
    k: int = 40,
) -> list[int]:
    """Select top-k feature indices by SHAP importance.

    Per architecture: limit to 40 most important features.
    More features = more overfitting.
    """
    return importance.top_k_indices[:k].tolist()


def compute_xgb_native_importance(
    model: object,
    feature_names: list[str] | None = None,
) -> FeatureImportanceResult:
    """Compute feature importance using XGBoost's built-in gain metric.

    Faster than SHAP but less accurate. Good for quick screening.
    """
    try:
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
    except AttributeError:
        # Model might not be XGBoost
        n_features = len(feature_names) if feature_names else 0
        return FeatureImportanceResult(
            feature_names=feature_names or [],
            importance_scores=np.zeros(n_features),
            top_k_indices=np.arange(n_features),
            top_k_names=feature_names or [],
        )

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(scores))]

    # Map XGBoost feature names (f0, f1, ...) to importance values
    importance = np.zeros(len(feature_names))
    for key, value in scores.items():
        try:
            idx = int(key.replace("f", ""))
            if idx < len(importance):
                importance[idx] = value
        except (ValueError, IndexError):
            pass

    sorted_idx = np.argsort(importance)[::-1]
    return FeatureImportanceResult(
        feature_names=feature_names,
        importance_scores=importance,
        top_k_indices=sorted_idx,
        top_k_names=[feature_names[i] for i in sorted_idx],
    )


def monitor_feature_drift(
    current_importance: FeatureImportanceResult,
    baseline_importance: FeatureImportanceResult,
    alert_threshold: int = 20,
) -> list[str]:
    """Detect if top features have drifted significantly.

    Per architecture: if top-10 feature drops to bottom-20, investigate regime change.

    Returns list of warning strings.
    """
    warnings = []

    # Get baseline top-10
    baseline_top10 = set(baseline_importance.top_k_names[:10])

    # Get current bottom-N
    n_features = len(current_importance.feature_names)
    current_bottom = set(current_importance.top_k_names[n_features - alert_threshold:])

    # Check for drift
    drifted = baseline_top10 & current_bottom
    for feat in drifted:
        old_rank = baseline_importance.top_k_names.index(feat) + 1
        new_rank = current_importance.top_k_names.index(feat) + 1
        warnings.append(
            f"DRIFT: '{feat}' dropped from rank {old_rank} to {new_rank}. "
            f"Possible regime change."
        )

    return warnings
