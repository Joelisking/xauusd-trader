"""Post-NFP gold direction model (XGBoost classifier).

Activated T+20 after Non-Farm Payrolls for short-window scalper entries.

Input features (4)
------------------
  0 — previous_nfp_surprise    : actual vs forecast, z-scored
  1 — gold_price_level         : XAUUSD close normalised by ATR
  2 — dxy_level                : DXY index value normalised by 50-EMA distance
  3 — preceding_5d_gold_trend  : slope of last 5 daily closes (z-scored)

Output
------
  direction   : "up" or "down"
  confidence  : float 0.0–1.0 (probability of predicted direction)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ai_server.config import NFP_MODEL_PATH
from ai_server.models.base import BaseModel

logger = logging.getLogger(__name__)

# Fixed feature count for the NFP model
NFP_FEATURE_COUNT: int = 4

# Direction labels
_DIRECTION_LABELS: dict[int, str] = {0: "down", 1: "up"}

# Default XGBoost params (lighter than the main model — small training set)
_NFP_DEFAULT_PARAMS: dict[str, Any] = {
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}


class NFPDirectionModel(BaseModel):
    """XGBoost model predicting post-NFP gold direction.

    Trained on 15 years of post-NFP gold behaviour (first-hour move after
    release).  Predicts whether gold will rally or fall in the T+20 to T+90
    window following the NFP print.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params: dict[str, Any] = {**_NFP_DEFAULT_PARAMS, **(params or {})}
        self._model: Any | None = None  # xgboost.XGBClassifier

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "NFPDirectionModel"

    def load(self, path: str | None = None) -> None:
        """Load model from a JSON file.

        Falls back to ``NFP_MODEL_PATH`` from config.  Initialises an
        untrained model if the file is absent so tests pass without weights.
        """
        from xgboost import XGBClassifier

        target = Path(path) if path else Path(NFP_MODEL_PATH)

        self._model = XGBClassifier(**self._params)

        if target.exists():
            self._model.load_model(str(target))
            logger.info("%s: model loaded from %s", self.name, target)
        else:
            logger.warning(
                "%s: model file not found at %s — running with untrained model.",
                self.name,
                target,
            )

    def predict(self, features: np.ndarray) -> float:  # type: ignore[override]
        """Return probability of class 1 (gold up) — satisfies BaseModel.

        For typed usage prefer ``predict_direction`` which returns a named
        tuple with the direction label.
        """
        _, confidence = self.predict_direction(features)
        return confidence

    # ------------------------------------------------------------------
    # Rich predict interface
    # ------------------------------------------------------------------

    def predict_direction(
        self, features: np.ndarray
    ) -> tuple[str, float]:
        """Predict post-NFP gold direction with confidence.

        Parameters
        ----------
        features:
            Array of shape ``(NFP_FEATURE_COUNT,)`` or
            ``(1, NFP_FEATURE_COUNT)`` with values:
            [nfp_surprise, gold_price_level, dxy_level, 5d_trend].

        Returns
        -------
        tuple[str, float]
            ``(direction, confidence)`` where direction is ``"up"`` or
            ``"down"`` and confidence is the probability 0.0–1.0 of the
            predicted class.
        """
        if self._model is None:
            self.load()

        arr = np.array(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if arr.shape[1] != NFP_FEATURE_COUNT:
            raise ValueError(
                f"{self.name}.predict expects {NFP_FEATURE_COUNT} features, "
                f"got {arr.shape[1]}"
            )

        proba: np.ndarray = self._model.predict_proba(arr)  # (1, 2)
        class_idx = int(np.argmax(proba[0]))
        confidence = float(proba[0, class_idx])
        direction = _DIRECTION_LABELS[class_idx]
        return direction, confidence

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """Save model to JSON file."""
        if self._model is None:
            raise RuntimeError(f"{self.name}.save called before model was built or loaded.")

        target = Path(path) if path else Path(NFP_MODEL_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(target))
        logger.info("%s: model saved to %s", self.name, target)

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
        """Fit the model with early stopping.

        Parameters
        ----------
        X_train:
            Feature matrix, shape ``(n_samples, NFP_FEATURE_COUNT)``.
        y_train:
            Binary direction labels (0=down, 1=up).
        X_val, y_val:
            Validation split for early stopping.
        save_path:
            Destination for saved model JSON.

        Returns
        -------
        xgboost.XGBClassifier
            Fitted model instance.
        """
        from xgboost import XGBClassifier

        n_pos = int(np.sum(y_train))
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        train_params = {
            **self._params,
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_rounds": 20,
        }

        self._model = XGBClassifier(**train_params)
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=20,
        )

        logger.info(
            "%s: training complete — best_iteration=%s",
            self.name,
            getattr(self._model, "best_iteration", None),
        )

        self.save(save_path)
        return self._model
