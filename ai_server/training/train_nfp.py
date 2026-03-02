"""NFP direction model training pipeline.

Simple XGBoost model for post-NFP gold direction prediction.
Input: previous NFP surprise, gold price level, DXY level, preceding 5-day gold trend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_server.config import NFP_MODEL_PATH


NFP_FEATURE_NAMES = [
    "nfp_surprise",      # Actual - Forecast (thousands)
    "gold_price_level",  # Current gold price
    "dxy_level",         # Current DXY level
    "gold_5d_trend",     # 5-day gold return %
]


def train_nfp_model(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    save_path: Path = NFP_MODEL_PATH,
) -> Any:
    """Train NFP direction model.

    Args:
        X: NFP features (samples, 4)
        y: Binary labels (0=down, 1=up after NFP)
        X_val: Validation features
        y_val: Validation labels
        save_path: Model save path

    Returns:
        Trained model
    """
    from ai_server.models.nfp_model import NFPDirectionModel

    model = NFPDirectionModel()
    model.build_model()
    model.train(X, y, X_val=X_val, y_val=y_val)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[TrainNFP] Model saved to {save_path}")
    return model


def prepare_nfp_data(
    nfp_events: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare NFP training data from historical event records.

    Each event dict should have:
        - nfp_surprise: float (actual - forecast, thousands)
        - gold_price: float
        - dxy_level: float
        - gold_5d_return: float (%)
        - gold_direction_after: int (0=down, 1=up)

    Returns:
        (X, y) arrays
    """
    X = []
    y = []

    for event in nfp_events:
        X.append([
            event.get("nfp_surprise", 0),
            event.get("gold_price", 3000),
            event.get("dxy_level", 104),
            event.get("gold_5d_return", 0),
        ])
        y.append(event.get("gold_direction_after", 0))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
