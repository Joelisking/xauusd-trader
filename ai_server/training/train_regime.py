"""Regime classifier training pipeline.

Trains a model to classify market regime: Trending / Ranging / Crisis.
Labels derived from ADX and VIX thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_server.config import REGIME_CLF_PATH


def generate_regime_labels(
    adx: np.ndarray,
    vix: np.ndarray,
) -> np.ndarray:
    """Generate regime labels from ADX and VIX values.

    Classes:
        0 = Trending (ADX > 25)
        1 = Ranging (ADX < 20)
        2 = Crisis (VIX > 25)

    Crisis takes precedence over ADX-based classification.
    """
    labels = np.ones(len(adx), dtype=np.int32)  # Default: ranging

    labels[adx > 25] = 0  # Trending
    labels[adx < 20] = 1  # Ranging
    labels[vix > 25] = 2  # Crisis (overrides)

    return labels


def train_regime_classifier(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    save_path: Path = REGIME_CLF_PATH,
) -> Any:
    """Train regime classifier model.

    Args:
        X: Features relevant to regime (ATR, ADX, VIX, Bollinger width, etc.)
        y: Regime labels (0=trending, 1=ranging, 2=crisis)
        X_val: Validation features
        y_val: Validation labels
        epochs: Training epochs
        batch_size: Batch size
        save_path: Model save path

    Returns:
        Trained model
    """
    from ai_server.models.regime_classifier import RegimeClassifier

    model = RegimeClassifier()
    num_features = X.shape[1]
    model.build_model(num_features)

    model.train(
        X, y,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[TrainRegime] Model saved to {save_path}")
    return model


def extract_regime_features(features: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Extract the subset of features relevant to regime classification.

    Selects: ADX, ATR, VIX, Bollinger width, volatility ratio, volume ratio.
    """
    regime_keywords = [
        "adx", "atr", "vix", "bb_width", "volatility",
        "volume_ratio", "spread_ratio", "range_expansion",
    ]

    indices = []
    for i, name in enumerate(feature_names):
        if any(kw in name.lower() for kw in regime_keywords):
            indices.append(i)

    if not indices:
        # Fallback: use first 10 features
        indices = list(range(min(10, features.shape[1])))

    return features[:, indices]
