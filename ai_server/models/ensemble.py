"""Ensemble scorer — combines all models into a single trade-quality score.

Scoring formula
---------------
  entry_score = (bilstm_prob * BILSTM_WEIGHT + xgb_prob * XGB_WEIGHT) * 100

Lot-multiplier tiers
--------------------
  0.8  — score 68–79   (LOT_MULT_NORMAL)
  1.0  — score 80–99   (LOT_MULT_HIGH)
  1.2  — score 80+ AND London-NY overlap session active  (LOT_MULT_PRIME)

All thresholds are sourced from ``ai_server.config`` — never hardcoded here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ai_server.config import (
    BILSTM_WEIGHT,
    ENTRY4_MIN_SCORE,
    FEATURE_COUNT,
    LOT_MULT_HIGH,
    LOT_MULT_NORMAL,
    LOT_MULT_PRIME,
    NFP_MODEL_PATH,
    REGIME_CLF_PATH,
    SCALPER_BILSTM_PATH,
    SCALPER_MIN_AI_SCORE,
    SCALPER_XGB_PATH,
    SEQUENCE_LENGTH,
    SESSION_OVERLAP_END,
    SESSION_OVERLAP_START,
    SWING_BILSTM_PATH,
    SWING_MIN_TREND_SCORE,
    SWING_XGB_PATH,
    TREND_EXHAUSTION_SCORE,
    XGB_FEATURE_COUNT,
    XGB_WEIGHT,
)

logger = logging.getLogger(__name__)


@dataclass
class ScalperEntryResult:
    """Structured output from ``EnsembleScorer.score_scalper_entry``."""

    entry_score: float
    trend_score: float
    regime: str
    wyckoff_phase: str
    approve: bool
    lot_multiplier: float
    bilstm_prob: float
    xgb_prob: float
    regime_probs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    latency_ms: float = 0.0


@dataclass
class SwingEntryResult:
    """Structured output from ``EnsembleScorer.score_swing_entry``."""

    entry_score: float
    trend_score: float
    regime: str
    wyckoff_phase: str
    approve: bool
    lot_multiplier: float
    bilstm_prob: float
    xgb_prob: float
    regime_probs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    latency_ms: float = 0.0


class EnsembleScorer:
    """Weighted ensemble of BiLSTM, XGBoost, regime, and NFP models.

    Typical usage
    -------------
    ::

        scorer = EnsembleScorer()
        scorer.load_all_models()

        result = scorer.score_scalper_entry(
            sequential_features=np.zeros((SEQUENCE_LENGTH, FEATURE_COUNT)),
            tabular_features=np.zeros(XGB_FEATURE_COUNT),
            session_hour=14,
        )
        if result.approve:
            execute_trade(result.lot_multiplier)
    """

    def __init__(self) -> None:
        # Lazy-imported model instances
        self._scalper_bilstm: Any | None = None
        self._swing_bilstm: Any | None = None
        self._scalper_xgb: Any | None = None
        self._swing_xgb: Any | None = None
        self._regime_clf: Any | None = None
        self._nfp_model: Any | None = None

        self._models_loaded: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def models_loaded(self) -> bool:
        """True when all model files were found and weights were loaded."""
        return self._models_loaded

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_all_models(self) -> None:
        """Attempt to load all model weights from paths defined in config.

        Individual models that are missing from disk emit a warning but do
        not raise — the scorer falls back to dummy values for missing models.
        Calling ``models_loaded`` afterward indicates full readiness.
        """
        from ai_server.models.nfp_model import NFPDirectionModel
        from ai_server.models.regime_classifier import RegimeClassifier
        from ai_server.models.scalper_bilstm import ScalperBiLSTM
        from ai_server.models.swing_bilstm import SwingBiLSTM
        from ai_server.models.xgboost_models import ScalperXGB, SwingXGB

        self._scalper_bilstm = ScalperBiLSTM()
        self._swing_bilstm = SwingBiLSTM()
        self._scalper_xgb = ScalperXGB()
        self._swing_xgb = SwingXGB()
        self._regime_clf = RegimeClassifier()
        self._nfp_model = NFPDirectionModel()

        all_paths_exist = all(
            Path(p).exists()
            for p in (
                SCALPER_BILSTM_PATH,
                SWING_BILSTM_PATH,
                SCALPER_XGB_PATH,
                SWING_XGB_PATH,
                REGIME_CLF_PATH,
                NFP_MODEL_PATH,
            )
        )

        # Load each model; failures are logged but do not abort startup.
        for label, model in [
            ("scalper_bilstm", self._scalper_bilstm),
            ("swing_bilstm", self._swing_bilstm),
            ("scalper_xgb", self._scalper_xgb),
            ("swing_xgb", self._swing_xgb),
            ("regime_clf", self._regime_clf),
            ("nfp_model", self._nfp_model),
        ]:
            try:
                model.load()
            except Exception as exc:  # pragma: no cover
                logger.error("EnsembleScorer: failed to load %s — %s", label, exc)

        self._models_loaded = all_paths_exist
        if self._models_loaded:
            logger.info("EnsembleScorer: all models loaded successfully.")
        else:
            logger.warning(
                "EnsembleScorer: one or more model files missing — "
                "running in degraded mode with random weights."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Auto-load models on first use if not already loaded."""
        if self._scalper_bilstm is None:
            self.load_all_models()

    @staticmethod
    def _compute_ensemble_score(bilstm_prob: float, xgb_prob: float) -> float:
        """Apply weighted ensemble formula.

        Returns
        -------
        float
            Score in range 0–100.
        """
        raw = bilstm_prob * BILSTM_WEIGHT + xgb_prob * XGB_WEIGHT
        return float(np.clip(raw * 100.0, 0.0, 100.0))

    @staticmethod
    def _compute_lot_multiplier(score: float, session_hour: int) -> float:
        """Determine position-size multiplier based on score and session.

        Parameters
        ----------
        score:
            Ensemble score 0–100.
        session_hour:
            Current UTC hour.

        Returns
        -------
        float
            ``LOT_MULT_NORMAL`` (0.8), ``LOT_MULT_HIGH`` (1.0), or
            ``LOT_MULT_PRIME`` (1.2).
        """
        in_overlap = SESSION_OVERLAP_START <= session_hour < SESSION_OVERLAP_END
        if score >= ENTRY4_MIN_SCORE and in_overlap:
            return LOT_MULT_PRIME
        if score >= ENTRY4_MIN_SCORE:
            return LOT_MULT_HIGH
        return LOT_MULT_NORMAL

    @staticmethod
    def _infer_wyckoff_phase(trend_score: float) -> str:
        """Heuristic Wyckoff phase from trend score.

        Without a dedicated Wyckoff model (Phase 9+), we derive a coarse
        phase estimate from the trend strength score so downstream logic
        always receives a valid label.

        Mapping
        -------
        score >= 80  -> "D" (Sign of Strength / Mark-up in progress)
        score >= 65  -> "C" (Spring / Upthrust — best entry timing)
        score >= 50  -> "B" (Consolidation / Testing)
        score < 50   -> "A" (Preliminary supply/demand)
        """
        if trend_score >= 80.0:
            return "D"
        if trend_score >= 65.0:
            return "C"
        if trend_score >= 50.0:
            return "B"
        return "A"

    def _safe_bilstm_predict(
        self,
        model: Any,
        sequential_features: np.ndarray,
        fallback: float = 0.5,
    ) -> float:
        """Call BiLSTM predict with a fallback on any exception."""
        try:
            arr = np.array(sequential_features, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            return float(model._model.predict(arr, verbose=0).squeeze())  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("BiLSTM predict failed (%s) — using fallback=%.2f", exc, fallback)
            return fallback

    def _safe_xgb_predict(
        self,
        model: Any,
        tabular_features: np.ndarray,
        fallback: float = 0.5,
    ) -> float:
        """Call XGBoost predict with a fallback on any exception."""
        try:
            arr = np.array(tabular_features, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            return float(model._model.predict_proba(arr)[0, 1])  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("XGB predict failed (%s) — using fallback=%.2f", exc, fallback)
            return fallback

    def _safe_regime_predict(
        self,
        regime_features: np.ndarray,
        fallback_regime: str = "trending",
    ) -> tuple[str, np.ndarray]:
        """Run regime classifier with graceful fallback."""
        try:
            return self._regime_clf.predict(regime_features)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "Regime predict failed (%s) — defaulting to '%s'", exc, fallback_regime
            )
            probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return fallback_regime, probs

    # ------------------------------------------------------------------
    # Public scoring interface
    # ------------------------------------------------------------------

    def score_scalper_entry(
        self,
        sequential_features: np.ndarray,
        tabular_features: np.ndarray,
        session_hour: int = 14,
        regime_features: np.ndarray | None = None,
    ) -> ScalperEntryResult:
        """Score a scalper entry candidate.

        Parameters
        ----------
        sequential_features:
            Shape ``(SEQUENCE_LENGTH, FEATURE_COUNT)`` — last 200 M1 candles.
        tabular_features:
            Shape ``(XGB_FEATURE_COUNT,)`` — last-candle tabular features.
        session_hour:
            Current UTC hour, used for lot-multiplier tier selection.
        regime_features:
            Optional subset of features for the regime classifier.  When
            ``None`` the first ``REGIME_FEATURE_COUNT`` elements of
            *tabular_features* are used.

        Returns
        -------
        ScalperEntryResult
        """
        t0 = time.perf_counter()
        self._ensure_loaded()

        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT

        # --- BiLSTM inference ---
        bilstm_prob = self._safe_bilstm_predict(self._scalper_bilstm, sequential_features)

        # --- XGBoost inference ---
        xgb_prob = self._safe_xgb_predict(self._scalper_xgb, tabular_features)

        # --- Ensemble score ---
        entry_score = self._compute_ensemble_score(bilstm_prob, xgb_prob)

        # --- Regime classification ---
        tab = np.array(tabular_features, dtype=np.float32).flatten()
        reg_feats = (
            np.array(regime_features, dtype=np.float32)
            if regime_features is not None
            else tab[:REGIME_FEATURE_COUNT]
        )
        regime, regime_probs = self._safe_regime_predict(reg_feats)

        # --- Trend score (same BiLSTM * 100 for scalper) ---
        trend_score = float(bilstm_prob * 100.0)

        # --- Wyckoff phase ---
        wyckoff_phase = self._infer_wyckoff_phase(trend_score)

        # --- Approval gate ---
        approve = (
            entry_score >= SCALPER_MIN_AI_SCORE
            and regime != "crisis"
        )

        # --- Lot multiplier ---
        lot_multiplier = self._compute_lot_multiplier(entry_score, session_hour)
        if not approve:
            lot_multiplier = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ScalperEntryResult(
            entry_score=entry_score,
            trend_score=trend_score,
            regime=regime,
            wyckoff_phase=wyckoff_phase,
            approve=approve,
            lot_multiplier=lot_multiplier,
            bilstm_prob=bilstm_prob,
            xgb_prob=xgb_prob,
            regime_probs=regime_probs,
            latency_ms=latency_ms,
        )

    def score_swing_entry(
        self,
        sequential_features: np.ndarray,
        tabular_features: np.ndarray,
        session_hour: int = 14,
        regime_features: np.ndarray | None = None,
    ) -> SwingEntryResult:
        """Score a swing entry candidate.

        Parameters
        ----------
        sequential_features:
            Shape ``(SEQUENCE_LENGTH, FEATURE_COUNT)`` — last 200 H1 candles.
        tabular_features:
            Shape ``(XGB_FEATURE_COUNT,)`` — last-candle tabular features.
        session_hour:
            Current UTC hour.
        regime_features:
            Optional regime classifier input.

        Returns
        -------
        SwingEntryResult
        """
        t0 = time.perf_counter()
        self._ensure_loaded()

        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT

        bilstm_prob = self._safe_bilstm_predict(self._swing_bilstm, sequential_features)
        xgb_prob = self._safe_xgb_predict(self._swing_xgb, tabular_features)

        entry_score = self._compute_ensemble_score(bilstm_prob, xgb_prob)
        trend_score = float(bilstm_prob * 100.0)
        wyckoff_phase = self._infer_wyckoff_phase(trend_score)

        tab = np.array(tabular_features, dtype=np.float32).flatten()
        reg_feats = (
            np.array(regime_features, dtype=np.float32)
            if regime_features is not None
            else tab[:REGIME_FEATURE_COUNT]
        )
        regime, regime_probs = self._safe_regime_predict(reg_feats)

        # Swing requires higher minimum trend score
        approve = (
            entry_score >= SWING_MIN_TREND_SCORE
            and trend_score >= SWING_MIN_TREND_SCORE
            and regime != "crisis"
        )

        lot_multiplier = self._compute_lot_multiplier(entry_score, session_hour)
        if not approve:
            lot_multiplier = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SwingEntryResult(
            entry_score=entry_score,
            trend_score=trend_score,
            regime=regime,
            wyckoff_phase=wyckoff_phase,
            approve=approve,
            lot_multiplier=lot_multiplier,
            bilstm_prob=bilstm_prob,
            xgb_prob=xgb_prob,
            regime_probs=regime_probs,
            latency_ms=latency_ms,
        )

    def is_trend_exhausted(self, trend_score: float) -> bool:
        """Return True when a mid-trade trend score signals exhaustion.

        When this returns ``True`` the swing EA should reduce exposure by 50 %
        and trail the remaining position.
        """
        return trend_score < TREND_EXHAUSTION_SCORE
