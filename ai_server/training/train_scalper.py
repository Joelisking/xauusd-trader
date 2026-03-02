"""Scalper model training pipeline.

Trains BiLSTM + XGBoost ensemble for scalper entry quality scoring.
Uses time-based split, class weighting, and walk-forward validation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ai_server.config import (
    BILSTM_WEIGHT,
    DATA_DIR,
    MODEL_DIR,
    SCALPER_BILSTM_PATH,
    SCALPER_XGB_PATH,
    SEQUENCE_LENGTH,
    XGB_WEIGHT,
)
from ai_server.training.label_generator import (
    compute_class_weights,
    filter_labeled_data,
    generate_scalper_labels,
)
from ai_server.training.evaluate import evaluate_model, EvaluationReport
from ai_server.training.walk_forward import run_walk_forward, WalkForwardResult


def train_scalper_bilstm(
    X_seq: np.ndarray,
    y: np.ndarray,
    X_val_seq: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    save_path: Path = SCALPER_BILSTM_PATH,
) -> Any:
    """Train scalper BiLSTM model.

    Args:
        X_seq: Sequential features (samples, seq_len, features)
        y: Binary labels
        X_val_seq: Validation sequential features
        y_val: Validation labels
        epochs: Max training epochs
        batch_size: Training batch size
        save_path: Where to save trained model

    Returns:
        Trained Keras model
    """
    from ai_server.models.scalper_bilstm import ScalperBiLSTM

    model = ScalperBiLSTM()
    num_features = X_seq.shape[2]
    model.build_model(num_features)

    class_weights = compute_class_weights(y)

    model.train(
        X_seq, y,
        X_val=X_val_seq,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[TrainScalper] BiLSTM saved to {save_path}")
    return model


def train_scalper_xgb(
    X_tab: np.ndarray,
    y: np.ndarray,
    X_val_tab: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_trials: int = 100,
    save_path: Path = SCALPER_XGB_PATH,
) -> Any:
    """Train scalper XGBoost model with Optuna tuning.

    Args:
        X_tab: Tabular features (samples, features)
        y: Binary labels
        X_val_tab: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials
        save_path: Where to save trained model

    Returns:
        Trained XGBoost model
    """
    from ai_server.models.xgboost_models import ScalperXGB

    model = ScalperXGB()

    if n_trials > 0 and X_val_tab is not None and y_val is not None:
        model.tune_hyperparameters(X_tab, y, X_val_tab, y_val, n_trials=n_trials)

    model.train(X_tab, y, X_val=X_val_tab, y_val=y_val)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[TrainScalper] XGBoost saved to {save_path}")
    return model


def run_scalper_training(
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
    """Run complete scalper training pipeline.

    Returns dict with models, reports, and ensemble evaluation.
    """
    print("[TrainScalper] Starting full training pipeline...")

    # Train BiLSTM
    bilstm = train_scalper_bilstm(
        X_seq_train, y_train,
        X_val_seq=X_seq_val, y_val=y_val,
        epochs=bilstm_epochs,
    )

    # Train XGBoost
    xgb = train_scalper_xgb(
        X_tab_train, y_train,
        X_val_tab=X_tab_val, y_val=y_val,
        n_trials=xgb_trials,
    )

    # Evaluate individually
    bilstm_pred_val = np.array([bilstm.predict(x.reshape(1, *x.shape)) for x in X_seq_val])
    xgb_pred_val = np.array([xgb.predict(x.reshape(1, -1)) for x in X_tab_val])

    bilstm_pred_test = np.array([bilstm.predict(x.reshape(1, *x.shape)) for x in X_seq_test])
    xgb_pred_test = np.array([xgb.predict(x.reshape(1, -1)) for x in X_tab_test])

    # Ensemble
    ensemble_val = BILSTM_WEIGHT * bilstm_pred_val + XGB_WEIGHT * xgb_pred_val
    ensemble_test = BILSTM_WEIGHT * bilstm_pred_test + XGB_WEIGHT * xgb_pred_test

    # Reports
    reports = {
        "bilstm_val": evaluate_model(y_val, bilstm_pred_val, "Scalper BiLSTM", "val"),
        "bilstm_test": evaluate_model(y_test, bilstm_pred_test, "Scalper BiLSTM", "test"),
        "xgb_val": evaluate_model(y_val, xgb_pred_val, "Scalper XGBoost", "val"),
        "xgb_test": evaluate_model(y_test, xgb_pred_test, "Scalper XGBoost", "test"),
        "ensemble_val": evaluate_model(y_val, ensemble_val, "Scalper Ensemble", "val"),
        "ensemble_test": evaluate_model(y_test, ensemble_test, "Scalper Ensemble", "test"),
    }

    for name, report in reports.items():
        print(f"\n{report.summary()}")

    return {
        "bilstm": bilstm,
        "xgb": xgb,
        "reports": reports,
    }
