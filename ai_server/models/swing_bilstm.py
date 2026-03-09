"""Swing BiLSTM + Multi-Head Attention model for H1/H4 entry scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ai_server.config import (
    FEATURE_COUNT,
    SEQUENCE_LENGTH,
    SWING_BILSTM_PATH,
    SWING_MIN_TREND_SCORE,
    TREND_EXHAUSTION_SCORE,
)
from ai_server.models.base import BaseModel

logger = logging.getLogger(__name__)


class SwingBiLSTM(BaseModel):
    """BiLSTM + Multi-Head Attention model for swing entry and trend scoring.

    Processes H1 sequences of (SEQUENCE_LENGTH, num_features) and returns a
    probability 0.0–1.0.  The ensemble scorer converts this to a 0–100 score
    which must exceed ``SWING_MIN_TREND_SCORE`` (72) to approve an entry, and
    a mid-trade score below ``TREND_EXHAUSTION_SCORE`` (45) triggers exposure
    reduction.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._num_features: int = FEATURE_COUNT
        self._min_trend_score: int = SWING_MIN_TREND_SCORE
        self._exhaustion_score: int = TREND_EXHAUSTION_SCORE

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "SwingBiLSTM"

    def load(self, path: str | None = None) -> None:
        """Load weights from an .h5 file.

        Falls back to ``SWING_BILSTM_PATH`` from config if *path* is None.
        Builds an untrained model when the file does not exist so that unit
        tests and dev workflows can proceed without trained weights.
        """
        target = Path(path) if path else Path(SWING_BILSTM_PATH)

        if not self._model:
            self._model = self.build_model(self._num_features)

        if target.exists():
            self._model.load_weights(str(target))
            logger.info("%s: weights loaded from %s", self.name, target)
        else:
            logger.warning(
                "%s: weight file not found at %s — running with random weights.",
                self.name,
                target,
            )

    def predict(self, features: np.ndarray) -> float:
        """Return trend-quality probability 0.0–1.0.

        Parameters
        ----------
        features:
            Array of shape ``(SEQUENCE_LENGTH, num_features)`` or
            ``(1, SEQUENCE_LENGTH, num_features)``.
        """
        if self._model is None:
            self.load()

        arr = np.array(features, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        if arr.shape[1:] != (SEQUENCE_LENGTH, self._num_features):
            raise ValueError(
                f"{self.name}.predict expects shape (batch, {SEQUENCE_LENGTH}, "
                f"{self._num_features}), got {arr.shape}"
            )

        import tensorflow as tf

        raw = self._model(tf.constant(arr), training=False)
        return float(np.squeeze(raw.numpy()))

    # ------------------------------------------------------------------
    # Architecture builder — identical topology to ScalperBiLSTM but
    # holds separate weights trained on H1 sequences
    # ------------------------------------------------------------------

    def build_model(self, num_features: int) -> Any:
        """Construct and compile the BiLSTM + Attention architecture.

        Parameters
        ----------
        num_features:
            Width of the feature dimension.

        Returns
        -------
        keras.Model
            Compiled model.
        """
        from tensorflow import keras
        from tensorflow.keras import layers

        self._num_features = num_features

        inputs = keras.Input(shape=(SEQUENCE_LENGTH, num_features), name="sequence_input")

        # First BiLSTM block
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True), name="bilstm_1"
        )(inputs)
        x = layers.Dropout(0.3, name="dropout_1")(x)

        # Second BiLSTM block
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True), name="bilstm_2"
        )(x)

        # Multi-Head Attention
        attn_out = layers.MultiHeadAttention(
            num_heads=8, key_dim=64, name="mha"
        )(x, x)

        x = layers.Add(name="residual_add")([x, attn_out])
        x = layers.LayerNormalization(name="layer_norm")(x)

        x = layers.GlobalAveragePooling1D(name="gap")(x)

        x = layers.Dense(64, activation="relu", name="dense_1")(x)
        x = layers.Dropout(0.2, name="dropout_2")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="SwingBiLSTM")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.AUC(name="auc"),
                keras.metrics.BinaryAccuracy(name="accuracy"),
            ],
        )

        logger.info(
            "%s: built model — params=%d", self.name, model.count_params()
        )
        self._model = model
        return model

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
        batch_size: int = 32,
        save_path: str | None = None,
    ) -> Any:
        """Train with early stopping, class weighting, and best-model saving.

        Parameters
        ----------
        X_train:
            Training sequences, shape ``(n_samples, SEQUENCE_LENGTH, features)``.
        y_train:
            Binary labels.
        X_val:
            Validation sequences.
        y_val:
            Validation labels.
        epochs:
            Maximum epochs.
        batch_size:
            Mini-batch size.
        save_path:
            Destination for best weights; falls back to ``SWING_BILSTM_PATH``.

        Returns
        -------
        keras.callbacks.History
        """
        from tensorflow import keras

        if self._model is None:
            n_features = X_train.shape[-1]
            self._model = self.build_model(n_features)

        n_pos = int(np.sum(y_train))
        n_neg = int(len(y_train) - n_pos)
        class_weight = (
            {0: 1.0, 1: n_neg / n_pos} if n_pos > 0 and n_neg > 0 else {0: 1.0, 1: 1.0}
        )

        save_target = str(save_path or SWING_BILSTM_PATH)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=10,
                mode="max",
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=save_target,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                factor=0.5,
                patience=5,
                mode="max",
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = self._model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info(
            "%s: training complete — best val_auc=%.4f",
            self.name,
            max(history.history.get("val_auc", [0.0])),
        )
        return history
