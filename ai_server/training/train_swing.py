"""Swing model training pipeline.

Trains BiLSTM + XGBoost ensemble for swing entry and trend scoring.
Same architecture as scalper but trained on H1 data with different labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_server.config import (
    BILSTM_WEIGHT,
    SWING_BILSTM_PATH,
    SWING_XGB_PATH,
    XGB_WEIGHT,
)
from ai_server.training.label_generator import compute_class_weights
from ai_server.training.evaluate import evaluate_model


def train_swing_bilstm(
    X_seq: np.ndarray,
    y: np.ndarray,
    X_val_seq: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    save_path: Path = SWING_BILSTM_PATH,
) -> Any:
    """Train swing BiLSTM model."""
    from ai_server.models.swing_bilstm import SwingBiLSTM

    model = SwingBiLSTM()
    num_features = X_seq.shape[2]
    model.build_model(num_features)

    class_weights = compute_class_weights(y)
    model.train(
        X_seq, y,
        X_val=X_val_seq, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[TrainSwing] BiLSTM saved to {save_path}")
    return model


def train_swing_xgb(
    X_tab: np.ndarray,
    y: np.ndarray,
    X_val_tab: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_trials: int = 100,
    save_path: Path = SWING_XGB_PATH,
) -> Any:
    """Train swing XGBoost model with Optuna tuning."""
    from ai_server.models.xgboost_models import SwingXGB

    model = SwingXGB()

    if n_trials > 0 and X_val_tab is not None and y_val is not None:
        model.tune_hyperparameters(X_tab, y, X_val_tab, y_val, n_trials=n_trials)

    model.train(X_tab, y, X_val=X_val_tab, y_val=y_val)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[TrainSwing] XGBoost saved to {save_path}")
    return model


def run_swing_training(
    X_seq_train: np.ndarray,
    X_tab_train: np.ndarray,
    y_train: np.ndarray,
    X_seq_val: np.ndarray,
    X_tab_val: np.ndarray,
    y_val: np.ndarray,
    X_seq_test: np.ndarray,
    X_tab_test: np.ndarray,
    y_test: np.ndarray,
    bilstm_epochs: int = 100,
    xgb_trials: int = 100,
) -> dict[str, Any]:
    """Run complete swing training pipeline."""
    print("[TrainSwing] Starting full training pipeline...")

    bilstm = train_swing_bilstm(
        X_seq_train, y_train,
        X_val_seq=X_seq_val, y_val=y_val,
        epochs=bilstm_epochs,
    )

    xgb = train_swing_xgb(
        X_tab_train, y_train,
        X_val_tab=X_tab_val, y_val=y_val,
        n_trials=xgb_trials,
    )

    # Evaluate
    bilstm_pred_test = np.array([bilstm.predict(x.reshape(1, *x.shape)) for x in X_seq_test])
    xgb_pred_test = np.array([xgb.predict(x.reshape(1, -1)) for x in X_tab_test])
    ensemble_test = BILSTM_WEIGHT * bilstm_pred_test + XGB_WEIGHT * xgb_pred_test

    reports = {
        "bilstm_test": evaluate_model(y_test, bilstm_pred_test, "Swing BiLSTM", "test"),
        "xgb_test": evaluate_model(y_test, xgb_pred_test, "Swing XGBoost", "test"),
        "ensemble_test": evaluate_model(y_test, ensemble_test, "Swing Ensemble", "test"),
    }

    for name, report in reports.items():
        print(f"\n{report.summary()}")

    return {"bilstm": bilstm, "xgb": xgb, "reports": reports}
