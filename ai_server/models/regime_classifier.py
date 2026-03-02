"""Market regime classifier: Trending / Ranging / Crisis.

Architecture
------------
Dense(64) -> ReLU -> Dropout(0.3) -> Dense(32) -> ReLU -> Dense(3, softmax)

Regime labels
-------------
  0 — Trending : ADX > 25
  1 — Ranging  : ADX < 20
  2 — Crisis   : VIX > 25

The regime drives bot mode switching in the ensemble scorer:
  - Trending  : normal operation
  - Ranging   : scalper pauses, swing tightens stops
  - Crisis    : 50 % sizes, tighter stops, swing paused
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ai_server.config import REGIME_CLF_PATH

logger = logging.getLogger(__name__)

# Class index -> human-readable label
_REGIME_NAMES: dict[int, str] = {
    0: "trending",
    1: "ranging",
    2: "crisis",
}

# Minimum number of regime-relevant features fed to this model.
# Subset: ATR, ADX, +DI, -DI, VIX, BB width, RSI(14), MACD histogram,
#         spread ratio, volume ratio, hour-of-day, price momentum ROC,
#         DXY ROC, real-yield level.  Exact count is 14 — extend as needed.
REGIME_FEATURE_COUNT: int = 14


class RegimeClassifier:
    """Feed-forward regime classifier.

    This class is intentionally *not* a subclass of ``BaseModel`` because it
    has a richer ``predict`` signature (returns both name and probabilities)
    and takes a smaller feature vector than the BiLSTM models.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._feature_count: int = REGIME_FEATURE_COUNT

    @property
    def name(self) -> str:
        return "RegimeClassifier"

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self, feature_count: int | None = None) -> Any:
        """Build and compile the feed-forward regime model.

        Parameters
        ----------
        feature_count:
            Input dimensionality.  Defaults to ``REGIME_FEATURE_COUNT``.

        Returns
        -------
        keras.Model
        """
        from tensorflow import keras
        from tensorflow.keras import layers

        if feature_count is not None:
            self._feature_count = feature_count

        inputs = keras.Input(shape=(self._feature_count,), name="regime_input")
        x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        x = layers.Dense(32, activation="relu", name="dense_2")(x)
        outputs = layers.Dense(3, activation="softmax", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="RegimeClassifier")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(
            "%s: built model — params=%d", self.name, model.count_params()
        )
        self._model = model
        return model

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load(self, path: str | None = None) -> None:
        """Load weights from an .h5 file.

        Falls back to ``REGIME_CLF_PATH`` from config.  Builds an untrained
        model when the file does not exist so that tests succeed without
        trained weights.
        """
        target = Path(path) if path else Path(REGIME_CLF_PATH)

        if self._model is None:
            self.build_model()

        if target.exists():
            self._model.load_weights(str(target))
            logger.info("%s: weights loaded from %s", self.name, target)
        else:
            logger.warning(
                "%s: weight file not found at %s — running with random weights.",
                self.name,
                target,
            )

    def save(self, path: str | None = None) -> None:
        """Save model weights."""
        if self._model is None:
            raise RuntimeError(f"{self.name}.save called before model was built.")

        target = Path(path) if path else Path(REGIME_CLF_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_weights(str(target))
        logger.info("%s: weights saved to %s", self.name, target)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> tuple[str, np.ndarray]:
        """Classify the current market regime.

        Parameters
        ----------
        features:
            Array of shape ``(REGIME_FEATURE_COUNT,)`` or
            ``(1, REGIME_FEATURE_COUNT)``.

        Returns
        -------
        tuple[str, np.ndarray]
            ``(regime_name, probabilities)`` where *probabilities* has shape
            ``(3,)`` and sums to 1.0.
        """
        if self._model is None:
            self.load()

        arr = np.array(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if arr.shape[1] != self._feature_count:
            raise ValueError(
                f"{self.name}.predict expects {self._feature_count} features, "
                f"got {arr.shape[1]}"
            )

        import tensorflow as tf

        proba = self._model(tf.constant(arr), training=False).numpy()  # (1, 3)
        proba_flat = proba[0]  # (3,)
        class_idx = int(np.argmax(proba_flat))
        regime_name = self.get_regime_name(class_idx)
        return regime_name, proba_flat

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_regime_name(class_idx: int) -> str:
        """Map class index to regime label.

        Parameters
        ----------
        class_idx:
            0 = trending, 1 = ranging, 2 = crisis.

        Returns
        -------
        str
        """
        return _REGIME_NAMES.get(class_idx, "unknown")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        save_path: str | None = None,
    ) -> Any:
        """Train with early stopping on validation accuracy.

        Parameters
        ----------
        X_train:
            Feature matrix, shape ``(n_samples, feature_count)``.
        y_train:
            Integer class labels (0/1/2).
        X_val, y_val:
            Validation split.
        epochs, batch_size:
            Training configuration.
        save_path:
            Where to save best weights.

        Returns
        -------
        keras.callbacks.History
        """
        from tensorflow import keras

        if self._model is None:
            self.build_model(feature_count=X_train.shape[-1])

        save_target = str(save_path or REGIME_CLF_PATH)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                mode="max",
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=save_target,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ]

        history = self._model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info(
            "%s: training complete — best val_accuracy=%.4f",
            self.name,
            max(history.history.get("val_accuracy", [0.0])),
        )
        return history
