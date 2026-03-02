"""XGBoost model wrappers for scalper and swing entry scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ai_server.config import (
    SCALPER_XGB_PATH,
    SWING_XGB_PATH,
    XGB_FEATURE_COUNT,
)
from ai_server.models.base import BaseModel

logger = logging.getLogger(__name__)

# Default XGBoost hyper-parameters used before Optuna tuning.
_DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}


class _BaseXGB(BaseModel):
    """Shared implementation for scalper and swing XGBoost classifiers.

    Subclasses only need to override ``name`` and the default *model_path*.
    """

    # Subclasses must set this at class level
    _default_path: Path = Path()

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params: dict[str, Any] = {**_DEFAULT_XGB_PARAMS, **(params or {})}
        self._model: Any | None = None  # xgboost.XGBClassifier

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # overridden by subclasses
        return "_BaseXGB"

    def build_model(self) -> Any:
        """Create and return a new, untrained XGBClassifier."""
        from xgboost import XGBClassifier  # local import — keep module-level clean

        model = XGBClassifier(**self._params)
        logger.debug("%s: created XGBClassifier with params=%s", self.name, self._params)
        return model

    def load(self, path: str | None = None) -> None:
        """Load a trained model from a JSON file.

        Falls back to the subclass ``_default_path`` when *path* is None.
        Initialises an untrained model if the file does not exist so that
        tests and dev workflows succeed without trained weights.
        """
        target = Path(path) if path else self._default_path
        self._model = self.build_model()

        if target.exists():
            self._model.load_model(str(target))
            logger.info("%s: model loaded from %s", self.name, target)
        else:
            logger.warning(
                "%s: model file not found at %s — running with untrained model.",
                self.name,
                target,
            )

    def save(self, path: str | None = None) -> None:
        """Save the trained model to a JSON file.

        Parameters
        ----------
        path:
            Destination path.  Falls back to the subclass ``_default_path``.
        """
        if self._model is None:
            raise RuntimeError(f"{self.name}.save called before model was built or loaded.")

        target = Path(path) if path else self._default_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(target))
        logger.info("%s: model saved to %s", self.name, target)

    def predict(self, features: np.ndarray) -> float:
        """Return entry quality probability 0.0–1.0.

        Parameters
        ----------
        features:
            Array of shape ``(XGB_FEATURE_COUNT,)`` or ``(1, XGB_FEATURE_COUNT)``.
        """
        if self._model is None:
            self.load()

        arr = np.array(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if arr.shape[1] != XGB_FEATURE_COUNT:
            raise ValueError(
                f"{self.name}.predict expects {XGB_FEATURE_COUNT} features, "
                f"got {arr.shape[1]}"
            )

        proba: np.ndarray = self._model.predict_proba(arr)
        # Column 1 = probability of class 1 (entry win)
        return float(proba[0, 1])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        save_path: str | None = None,
    ) -> Any:
        """Fit the XGBClassifier with early stopping.

        Parameters
        ----------
        X_train:
            Tabular features, shape ``(n_samples, XGB_FEATURE_COUNT)``.
        y_train:
            Binary labels.
        X_val:
            Validation features.
        y_val:
            Validation labels.
        save_path:
            Where to save the trained model.

        Returns
        -------
        xgboost.XGBClassifier
            The fitted model instance.
        """
        from xgboost import XGBClassifier

        # Compute scale_pos_weight for class imbalance (mirrors class weighting
        # strategy used in the LSTM models — no SMOTE)
        n_pos = int(np.sum(y_train))
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        train_params = {
            **self._params,
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_rounds": 30,
        }

        self._model = XGBClassifier(**train_params)
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        best_iter = getattr(self._model, "best_iteration", None)
        best_score = getattr(self._model, "best_score", None)
        logger.info(
            "%s: training complete — best_iteration=%s, best_score=%s",
            self.name,
            best_iter,
            best_score,
        )

        self.save(save_path)
        return self._model

    # ------------------------------------------------------------------
    # Hyper-parameter tuning
    # ------------------------------------------------------------------

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        cv_splits: int = 5,
        save_path: str | None = None,
    ) -> dict[str, Any]:
        """Use Optuna to find the best hyper-parameters, maximising AUC.

        A time-series cross-validation split is used to avoid look-ahead
        bias consistent with the walk-forward methodology described in the
        architecture docs.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, XGB_FEATURE_COUNT)``.
        y:
            Binary labels.
        n_trials:
            Number of Optuna trials.
        cv_splits:
            Number of CV folds (TimeSeriesSplit).
        save_path:
            Where to save the best model after tuning.

        Returns
        -------
        dict[str, Any]
            Best hyper-parameter dict.
        """
        import optuna
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit
        from xgboost import XGBClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "eval_metric": "auc",
                "use_label_encoder": False,
                "random_state": 42,
                "n_jobs": -1,
            }

            auc_scores: list[float] = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_vl = X[train_idx], X[val_idx]
                y_tr, y_vl = y[train_idx], y[val_idx]

                n_pos = int(np.sum(y_tr))
                n_neg = int(len(y_tr) - n_pos)
                spw = n_neg / n_pos if n_pos > 0 else 1.0

                clf = XGBClassifier(**params, scale_pos_weight=spw)
                clf.fit(X_tr, y_tr, verbose=False)

                proba = clf.predict_proba(X_vl)[:, 1]
                if len(np.unique(y_vl)) > 1:
                    auc_scores.append(roc_auc_score(y_vl, proba))

            return float(np.mean(auc_scores)) if auc_scores else 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = {**_DEFAULT_XGB_PARAMS, **study.best_params}
        logger.info(
            "%s: tuning complete — best AUC=%.4f, best_params=%s",
            self.name,
            study.best_value,
            best_params,
        )

        # Retrain on full dataset with best params
        self._params = best_params
        self._model = XGBClassifier(**self._params)

        n_pos = int(np.sum(y))
        n_neg = int(len(y) - n_pos)
        self._model.set_params(scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0)
        self._model.fit(X, y, verbose=False)

        self.save(save_path)
        return best_params


# ---------------------------------------------------------------------------
# Concrete subclasses — differ only in path and display name
# ---------------------------------------------------------------------------


class ScalperXGB(_BaseXGB):
    """XGBoost entry scorer for the Gold Scalper bot."""

    _default_path: Path = Path(SCALPER_XGB_PATH)

    @property
    def name(self) -> str:
        return "ScalperXGB"


class SwingXGB(_BaseXGB):
    """XGBoost entry scorer for the Gold Swing Rider bot."""

    _default_path: Path = Path(SWING_XGB_PATH)

    @property
    def name(self) -> str:
        return "SwingXGB"
