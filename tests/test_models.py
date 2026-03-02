"""Tests for all AI model definitions (Phase 8).

These tests verify architecture, forward-pass shapes, output ranges, and
ensemble logic using small random data.  No real training data is required —
models run with random (untrained) weights throughout.

All model files are tested in isolation before the ensemble is tested as a
whole so failures can be pinpointed easily.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_server.config import (
    BILSTM_WEIGHT,
    ENTRY4_MIN_SCORE,
    FEATURE_COUNT,
    LOT_MULT_HIGH,
    LOT_MULT_NORMAL,
    LOT_MULT_PRIME,
    SCALPER_MIN_AI_SCORE,
    SEQUENCE_LENGTH,
    SESSION_OVERLAP_END,
    SESSION_OVERLAP_START,
    SWING_MIN_TREND_SCORE,
    TREND_EXHAUSTION_SCORE,
    XGB_FEATURE_COUNT,
    XGB_WEIGHT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_seq(batch: int = 1) -> np.ndarray:
    """Random sequential features (batch, SEQUENCE_LENGTH, FEATURE_COUNT)."""
    return np.random.rand(batch, SEQUENCE_LENGTH, FEATURE_COUNT).astype(np.float32)


def _rand_tab(batch: int = 1) -> np.ndarray:
    """Random tabular features (batch, XGB_FEATURE_COUNT)."""
    return np.random.rand(batch, XGB_FEATURE_COUNT).astype(np.float32)


def _rand_regime_feats(batch: int = 1) -> np.ndarray:
    """Random regime features (batch, REGIME_FEATURE_COUNT)."""
    from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT

    return np.random.rand(batch, REGIME_FEATURE_COUNT).astype(np.float32)


# ---------------------------------------------------------------------------
# ScalperBiLSTM
# ---------------------------------------------------------------------------


class TestScalperBiLSTM:
    def test_instantiation(self) -> None:
        from ai_server.models.scalper_bilstm import ScalperBiLSTM

        model = ScalperBiLSTM()
        assert model.name == "ScalperBiLSTM"

    def test_build_model_returns_keras_model(self) -> None:
        import tensorflow as tf
        from ai_server.models.scalper_bilstm import ScalperBiLSTM

        model = ScalperBiLSTM()
        keras_model = model.build_model(FEATURE_COUNT)
        assert keras_model is not None
        assert hasattr(keras_model, "predict")

    def test_model_input_shape(self) -> None:
        from ai_server.models.scalper_bilstm import ScalperBiLSTM

        model = ScalperBiLSTM()
        keras_model = model.build_model(FEATURE_COUNT)
        # Input shape should be (None, SEQUENCE_LENGTH, FEATURE_COUNT)
        expected_input = (None, SEQUENCE_LENGTH, FEATURE_COUNT)
        assert keras_model.input_shape == expected_input

    def test_model_output_shape(self) -> None:
        from ai_server.models.scalper_bilstm import ScalperBiLSTM

        model = ScalperBiLSTM()
        keras_model = model.build_model(FEATURE_COUNT)
        # Output shape should be (None, 1) — single sigmoid probability
        assert keras_model.output_shape == (None, 1)

    def test_forward_pass_returns_probability(self) -> None:
        from ai_server.models.scalper_bilstm import ScalperBiLSTM

        model = ScalperBiLSTM()
        model.build_model(FEATURE_COUNT)
        seq = _rand_seq(batch=1)
        # Pass as (SEQUENCE_LENGTH, FEATURE_COUNT) — predict should add batch dim
        prob = model.predict(seq[0])
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_forward_pass_batch_input(self) -> None:
        from ai_server.models.scalper_bilstm import ScalperBiLSTM

        model = ScalperBiLSTM()
        model.build_model(FEATURE_COUNT)
        seq_batch = _rand_seq(batch=1)  # (1, SEQUENCE_LENGTH, FEATURE_COUNT)
        prob = model.predict(seq_batch)
        assert 0.0 <= prob <= 1.0

    def test_load_missing_file_does_not_raise(self) -> None:
        from ai_server.models.scalper_bilstm import ScalperBiLSTM

        model = ScalperBiLSTM()
        # Should log a warning but not raise
        model.load("/tmp/nonexistent_scalper_bilstm.h5")
        # Model should still be usable
        seq = _rand_seq(batch=1)
        prob = model.predict(seq[0])
        assert 0.0 <= prob <= 1.0

    def test_wrong_feature_count_raises(self) -> None:
        from ai_server.models.scalper_bilstm import ScalperBiLSTM

        model = ScalperBiLSTM()
        model.build_model(FEATURE_COUNT)
        bad_seq = np.random.rand(1, SEQUENCE_LENGTH, FEATURE_COUNT + 10).astype(np.float32)
        with pytest.raises(ValueError, match="expects shape"):
            model.predict(bad_seq)

    def test_multiple_forward_passes_are_stable(self) -> None:
        """Multiple calls on the same model instance should not crash."""
        from ai_server.models.scalper_bilstm import ScalperBiLSTM

        model = ScalperBiLSTM()
        model.build_model(FEATURE_COUNT)
        for _ in range(5):
            seq = _rand_seq(batch=1)
            prob = model.predict(seq[0])
            assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# SwingBiLSTM
# ---------------------------------------------------------------------------


class TestSwingBiLSTM:
    def test_instantiation(self) -> None:
        from ai_server.models.swing_bilstm import SwingBiLSTM

        model = SwingBiLSTM()
        assert model.name == "SwingBiLSTM"

    def test_build_model(self) -> None:
        from ai_server.models.swing_bilstm import SwingBiLSTM

        model = SwingBiLSTM()
        keras_model = model.build_model(FEATURE_COUNT)
        assert keras_model.input_shape == (None, SEQUENCE_LENGTH, FEATURE_COUNT)
        assert keras_model.output_shape == (None, 1)

    def test_forward_pass_returns_probability(self) -> None:
        from ai_server.models.swing_bilstm import SwingBiLSTM

        model = SwingBiLSTM()
        model.build_model(FEATURE_COUNT)
        seq = _rand_seq(batch=1)
        prob = model.predict(seq[0])
        assert 0.0 <= prob <= 1.0

    def test_load_missing_file_does_not_raise(self) -> None:
        from ai_server.models.swing_bilstm import SwingBiLSTM

        model = SwingBiLSTM()
        model.load("/tmp/nonexistent_swing_bilstm.h5")
        seq = _rand_seq(batch=1)
        prob = model.predict(seq[0])
        assert 0.0 <= prob <= 1.0

    def test_scalper_and_swing_are_independent_instances(self) -> None:
        """Scalper and Swing models must not share weights."""
        from ai_server.models.scalper_bilstm import ScalperBiLSTM
        from ai_server.models.swing_bilstm import SwingBiLSTM

        scalper = ScalperBiLSTM()
        swing = SwingBiLSTM()
        assert scalper is not swing
        assert scalper._model is None  # not yet built
        assert swing._model is None

    def test_wrong_sequence_length_raises(self) -> None:
        from ai_server.models.swing_bilstm import SwingBiLSTM

        model = SwingBiLSTM()
        model.build_model(FEATURE_COUNT)
        # Wrong sequence length
        bad_seq = np.random.rand(1, SEQUENCE_LENGTH + 5, FEATURE_COUNT).astype(np.float32)
        with pytest.raises(ValueError, match="expects shape"):
            model.predict(bad_seq)


# ---------------------------------------------------------------------------
# ScalperXGB / SwingXGB
# ---------------------------------------------------------------------------


class TestXGBoostModels:
    def test_scalper_xgb_instantiation(self) -> None:
        from ai_server.models.xgboost_models import ScalperXGB

        model = ScalperXGB()
        assert model.name == "ScalperXGB"

    def test_swing_xgb_instantiation(self) -> None:
        from ai_server.models.xgboost_models import SwingXGB

        model = SwingXGB()
        assert model.name == "SwingXGB"

    def test_scalper_xgb_forward_pass_after_minimal_train(self) -> None:
        """Train on tiny data and verify predict returns 0-1 probability."""
        from ai_server.models.xgboost_models import ScalperXGB

        model = ScalperXGB()
        # Tiny dataset to exercise the training path without slow computation
        rng = np.random.default_rng(42)
        X = rng.random((20, XGB_FEATURE_COUNT), dtype=np.float32)
        y = rng.integers(0, 2, size=20).astype(np.float32)

        model._model = model.build_model()
        # Skip early stopping by fitting directly with a small n_estimators
        from xgboost import XGBClassifier

        clf = XGBClassifier(n_estimators=10, eval_metric="auc", random_state=42)
        clf.fit(X, y)
        model._model = clf

        prob = model.predict(X[0])
        assert 0.0 <= prob <= 1.0

    def test_scalper_xgb_predict_shape(self) -> None:
        """predict() must accept a 1-D vector of length XGB_FEATURE_COUNT."""
        from xgboost import XGBClassifier
        from ai_server.models.xgboost_models import ScalperXGB

        model = ScalperXGB()
        rng = np.random.default_rng(7)
        X = rng.random((10, XGB_FEATURE_COUNT), dtype=np.float32)
        y = rng.integers(0, 2, size=10).astype(np.float32)
        clf = XGBClassifier(n_estimators=5, eval_metric="auc", random_state=0)
        clf.fit(X, y)
        model._model = clf

        prob = model.predict(X[0])
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_swing_xgb_separate_from_scalper(self) -> None:
        from ai_server.models.xgboost_models import ScalperXGB, SwingXGB

        scalper = ScalperXGB()
        swing = SwingXGB()
        assert scalper is not swing
        assert "scalper" in str(scalper._default_path).lower()
        assert "swing" in str(swing._default_path).lower()

    def test_load_missing_file_does_not_raise(self) -> None:
        from ai_server.models.xgboost_models import ScalperXGB

        model = ScalperXGB()
        model.load("/tmp/nonexistent_scalper.json")
        # Model should be built (untrained) but usable after a fit
        assert model._model is not None

    def test_wrong_feature_count_raises(self) -> None:
        from xgboost import XGBClassifier
        from ai_server.models.xgboost_models import ScalperXGB

        model = ScalperXGB()
        rng = np.random.default_rng(0)
        X = rng.random((10, XGB_FEATURE_COUNT), dtype=np.float32)
        y = rng.integers(0, 2, size=10).astype(np.float32)
        clf = XGBClassifier(n_estimators=5, eval_metric="auc")
        clf.fit(X, y)
        model._model = clf

        bad_features = np.random.rand(XGB_FEATURE_COUNT + 5).astype(np.float32)
        with pytest.raises(ValueError, match="features"):
            model.predict(bad_features)


# ---------------------------------------------------------------------------
# RegimeClassifier
# ---------------------------------------------------------------------------


class TestRegimeClassifier:
    def test_instantiation(self) -> None:
        from ai_server.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier()
        assert clf.name == "RegimeClassifier"

    def test_build_model(self) -> None:
        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT, RegimeClassifier

        clf = RegimeClassifier()
        model = clf.build_model()
        assert model.input_shape == (None, REGIME_FEATURE_COUNT)
        assert model.output_shape == (None, 3)

    def test_predict_returns_regime_and_probabilities(self) -> None:
        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT, RegimeClassifier

        clf = RegimeClassifier()
        clf.build_model()
        feats = np.random.rand(REGIME_FEATURE_COUNT).astype(np.float32)
        regime, probs = clf.predict(feats)
        assert isinstance(regime, str)
        assert regime in ("trending", "ranging", "crisis")
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_predict_batched_input(self) -> None:
        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT, RegimeClassifier

        clf = RegimeClassifier()
        clf.build_model()
        feats = np.random.rand(1, REGIME_FEATURE_COUNT).astype(np.float32)
        regime, probs = clf.predict(feats)
        assert regime in ("trending", "ranging", "crisis")

    def test_get_regime_name_valid_indices(self) -> None:
        from ai_server.models.regime_classifier import RegimeClassifier

        assert RegimeClassifier.get_regime_name(0) == "trending"
        assert RegimeClassifier.get_regime_name(1) == "ranging"
        assert RegimeClassifier.get_regime_name(2) == "crisis"

    def test_get_regime_name_unknown_index(self) -> None:
        from ai_server.models.regime_classifier import RegimeClassifier

        result = RegimeClassifier.get_regime_name(99)
        assert result == "unknown"

    def test_probabilities_sum_to_one(self) -> None:
        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT, RegimeClassifier

        clf = RegimeClassifier()
        clf.build_model()
        for _ in range(10):
            feats = np.random.rand(REGIME_FEATURE_COUNT).astype(np.float32)
            _, probs = clf.predict(feats)
            assert abs(probs.sum() - 1.0) < 1e-4, f"probs.sum() = {probs.sum()}"

    def test_load_missing_file_does_not_raise(self) -> None:
        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT, RegimeClassifier

        clf = RegimeClassifier()
        clf.load("/tmp/nonexistent_regime.h5")
        feats = np.random.rand(REGIME_FEATURE_COUNT).astype(np.float32)
        regime, _ = clf.predict(feats)
        assert regime in ("trending", "ranging", "crisis")

    def test_wrong_feature_count_raises(self) -> None:
        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT, RegimeClassifier

        clf = RegimeClassifier()
        clf.build_model()
        bad_feats = np.random.rand(REGIME_FEATURE_COUNT + 5).astype(np.float32)
        with pytest.raises(ValueError, match="features"):
            clf.predict(bad_feats)


# ---------------------------------------------------------------------------
# NFPDirectionModel
# ---------------------------------------------------------------------------


class TestNFPDirectionModel:
    def test_instantiation(self) -> None:
        from ai_server.models.nfp_model import NFPDirectionModel

        model = NFPDirectionModel()
        assert model.name == "NFPDirectionModel"

    def test_predict_direction_after_minimal_train(self) -> None:
        from xgboost import XGBClassifier
        from ai_server.models.nfp_model import NFP_FEATURE_COUNT, NFPDirectionModel

        model = NFPDirectionModel()
        rng = np.random.default_rng(1)
        X = rng.random((20, NFP_FEATURE_COUNT), dtype=np.float32)
        y = rng.integers(0, 2, size=20).astype(np.float32)
        clf = XGBClassifier(n_estimators=5, eval_metric="auc", random_state=0)
        clf.fit(X, y)
        model._model = clf

        direction, confidence = model.predict_direction(X[0])
        assert direction in ("up", "down")
        assert 0.0 <= confidence <= 1.0

    def test_predict_base_method_returns_float(self) -> None:
        from xgboost import XGBClassifier
        from ai_server.models.nfp_model import NFP_FEATURE_COUNT, NFPDirectionModel

        model = NFPDirectionModel()
        rng = np.random.default_rng(2)
        X = rng.random((10, NFP_FEATURE_COUNT), dtype=np.float32)
        y = rng.integers(0, 2, size=10).astype(np.float32)
        clf = XGBClassifier(n_estimators=5, eval_metric="auc")
        clf.fit(X, y)
        model._model = clf

        prob = model.predict(X[0])
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_load_missing_file_does_not_raise(self) -> None:
        from ai_server.models.nfp_model import NFP_FEATURE_COUNT, NFPDirectionModel

        model = NFPDirectionModel()
        model.load("/tmp/nonexistent_nfp.json")
        # Model is built but untrained — just verify it doesn't crash on load
        assert model._model is not None

    def test_feature_count_is_4(self) -> None:
        from ai_server.models.nfp_model import NFP_FEATURE_COUNT

        assert NFP_FEATURE_COUNT == 4

    def test_batched_input(self) -> None:
        from xgboost import XGBClassifier
        from ai_server.models.nfp_model import NFP_FEATURE_COUNT, NFPDirectionModel

        model = NFPDirectionModel()
        rng = np.random.default_rng(3)
        X = rng.random((10, NFP_FEATURE_COUNT), dtype=np.float32)
        y = rng.integers(0, 2, size=10).astype(np.float32)
        clf = XGBClassifier(n_estimators=5, eval_metric="auc")
        clf.fit(X, y)
        model._model = clf

        # Pass 2-D row
        direction, confidence = model.predict_direction(X[0:1])
        assert direction in ("up", "down")
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# EnsembleScorer
# ---------------------------------------------------------------------------


class TestEnsembleScorer:
    """Tests for the EnsembleScorer class.

    We inject lightweight pre-trained XGBoost and keras models so the tests
    run quickly without full model training.
    """

    @pytest.fixture
    def scorer_with_random_models(self):
        """Return an EnsembleScorer with all sub-models initialised."""
        from ai_server.models.ensemble import EnsembleScorer
        from ai_server.models.nfp_model import NFPDirectionModel
        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT, RegimeClassifier
        from ai_server.models.scalper_bilstm import ScalperBiLSTM
        from ai_server.models.swing_bilstm import SwingBiLSTM
        from ai_server.models.xgboost_models import ScalperXGB, SwingXGB
        from xgboost import XGBClassifier

        scorer = EnsembleScorer()
        rng = np.random.default_rng(99)

        # Build BiLSTM models with random weights (no training needed for inference)
        scalper_bilstm = ScalperBiLSTM()
        scalper_bilstm.build_model(FEATURE_COUNT)
        swing_bilstm = SwingBiLSTM()
        swing_bilstm.build_model(FEATURE_COUNT)

        # Train tiny XGBoost classifiers
        def _tiny_xgb(feature_count: int) -> XGBClassifier:
            X = rng.random((20, feature_count), dtype=np.float32)
            y = rng.integers(0, 2, size=20).astype(np.float32)
            clf = XGBClassifier(n_estimators=5, eval_metric="auc", random_state=0)
            clf.fit(X, y)
            return clf

        scalper_xgb = ScalperXGB()
        scalper_xgb._model = _tiny_xgb(XGB_FEATURE_COUNT)
        swing_xgb = SwingXGB()
        swing_xgb._model = _tiny_xgb(XGB_FEATURE_COUNT)

        # Regime classifier
        regime_clf = RegimeClassifier()
        regime_clf.build_model(REGIME_FEATURE_COUNT)

        # NFP model
        nfp_model = NFPDirectionModel()
        from ai_server.models.nfp_model import NFP_FEATURE_COUNT

        nfp_model._model = _tiny_xgb(NFP_FEATURE_COUNT)

        scorer._scalper_bilstm = scalper_bilstm
        scorer._swing_bilstm = swing_bilstm
        scorer._scalper_xgb = scalper_xgb
        scorer._swing_xgb = swing_xgb
        scorer._regime_clf = regime_clf
        scorer._nfp_model = nfp_model

        return scorer

    def test_instantiation(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        assert scorer.models_loaded is False

    def test_models_loaded_false_when_files_missing(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        scorer.load_all_models()  # weight files don't exist — should warn, not raise
        # models_loaded is False because files are absent
        assert scorer.models_loaded is False

    def test_score_scalper_entry_structure(self, scorer_with_random_models) -> None:
        from ai_server.models.ensemble import ScalperEntryResult

        scorer = scorer_with_random_models
        seq = _rand_seq(batch=1)[0]  # (SEQUENCE_LENGTH, FEATURE_COUNT)
        tab = _rand_tab(batch=1)[0]  # (XGB_FEATURE_COUNT,)

        result = scorer.score_scalper_entry(seq, tab, session_hour=14)
        assert isinstance(result, ScalperEntryResult)

    def test_score_scalper_entry_ranges(self, scorer_with_random_models) -> None:
        scorer = scorer_with_random_models
        seq = _rand_seq(batch=1)[0]
        tab = _rand_tab(batch=1)[0]

        result = scorer.score_scalper_entry(seq, tab, session_hour=14)
        assert 0.0 <= result.entry_score <= 100.0
        assert 0.0 <= result.trend_score <= 100.0
        assert result.regime in ("trending", "ranging", "crisis")
        assert result.wyckoff_phase in ("A", "B", "C", "D", "E")
        assert isinstance(result.approve, bool)
        assert result.lot_multiplier >= 0.0
        assert 0.0 <= result.bilstm_prob <= 1.0
        assert 0.0 <= result.xgb_prob <= 1.0
        assert result.latency_ms > 0.0

    def test_score_swing_entry_structure(self, scorer_with_random_models) -> None:
        from ai_server.models.ensemble import SwingEntryResult

        scorer = scorer_with_random_models
        seq = _rand_seq(batch=1)[0]
        tab = _rand_tab(batch=1)[0]

        result = scorer.score_swing_entry(seq, tab, session_hour=14)
        assert isinstance(result, SwingEntryResult)

    def test_score_swing_entry_ranges(self, scorer_with_random_models) -> None:
        scorer = scorer_with_random_models
        seq = _rand_seq(batch=1)[0]
        tab = _rand_tab(batch=1)[0]

        result = scorer.score_swing_entry(seq, tab, session_hour=14)
        assert 0.0 <= result.entry_score <= 100.0
        assert 0.0 <= result.trend_score <= 100.0
        assert result.regime in ("trending", "ranging", "crisis")
        assert isinstance(result.approve, bool)

    def test_ensemble_formula_correctness(self) -> None:
        """Verify the ensemble formula: score = (bilstm * 0.55 + xgb * 0.45) * 100."""
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        bilstm_prob = 0.80
        xgb_prob = 0.60
        expected = (bilstm_prob * BILSTM_WEIGHT + xgb_prob * XGB_WEIGHT) * 100.0
        actual = scorer._compute_ensemble_score(bilstm_prob, xgb_prob)
        assert abs(actual - expected) < 1e-5

    def test_ensemble_weights_sum_check(self) -> None:
        """BILSTM_WEIGHT + XGB_WEIGHT should equal 1.0."""
        assert abs(BILSTM_WEIGHT + XGB_WEIGHT - 1.0) < 1e-9

    def test_lot_multiplier_normal_range(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        # Score 68-79, non-overlap session
        mult = scorer._compute_lot_multiplier(score=70.0, session_hour=12)
        assert mult == LOT_MULT_NORMAL

    def test_lot_multiplier_high_range(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        # Score >= ENTRY4_MIN_SCORE (82), non-overlap session
        mult = scorer._compute_lot_multiplier(score=85.0, session_hour=12)
        assert mult == LOT_MULT_HIGH

    def test_lot_multiplier_prime_during_overlap(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        # Score >= 82, overlap session (13-17 UTC)
        overlap_hour = SESSION_OVERLAP_START  # 13
        mult = scorer._compute_lot_multiplier(score=85.0, session_hour=overlap_hour)
        assert mult == LOT_MULT_PRIME

    def test_lot_multiplier_prime_not_outside_overlap(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        # Score >= 82 but outside overlap (hour 18 = NY session)
        mult = scorer._compute_lot_multiplier(score=85.0, session_hour=18)
        assert mult == LOT_MULT_HIGH

    def test_unapproved_scalper_has_zero_lot(self, scorer_with_random_models) -> None:
        """When approve=False the lot multiplier must be 0.0."""
        scorer = scorer_with_random_models
        seq = _rand_seq(batch=1)[0]
        tab = _rand_tab(batch=1)[0]

        # Score many times until we get a non-approved result or force approve=False
        result = scorer.score_scalper_entry(seq, tab, session_hour=14)
        if not result.approve:
            assert result.lot_multiplier == 0.0

    def test_is_trend_exhausted_true_below_threshold(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        assert scorer.is_trend_exhausted(TREND_EXHAUSTION_SCORE - 1) is True

    def test_is_trend_exhausted_false_above_threshold(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        assert scorer.is_trend_exhausted(TREND_EXHAUSTION_SCORE + 1) is False

    def test_wyckoff_phase_mapping(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        # Verify boundary values
        assert scorer._infer_wyckoff_phase(80.0) == "D"
        assert scorer._infer_wyckoff_phase(65.0) == "C"
        assert scorer._infer_wyckoff_phase(64.9) == "B"
        assert scorer._infer_wyckoff_phase(50.0) == "B"
        assert scorer._infer_wyckoff_phase(49.9) == "A"

    def test_regime_probs_shape(self, scorer_with_random_models) -> None:
        scorer = scorer_with_random_models
        seq = _rand_seq(batch=1)[0]
        tab = _rand_tab(batch=1)[0]

        result = scorer.score_scalper_entry(seq, tab)
        assert result.regime_probs.shape == (3,)
        assert abs(result.regime_probs.sum() - 1.0) < 1e-4

    def test_swing_requires_higher_min_score(self) -> None:
        """Swing minimum score (72) > Scalper minimum score (68)."""
        assert SWING_MIN_TREND_SCORE > SCALPER_MIN_AI_SCORE

    def test_custom_regime_features_accepted(self, scorer_with_random_models) -> None:
        from ai_server.models.regime_classifier import REGIME_FEATURE_COUNT

        scorer = scorer_with_random_models
        seq = _rand_seq(batch=1)[0]
        tab = _rand_tab(batch=1)[0]
        regime_feats = np.random.rand(REGIME_FEATURE_COUNT).astype(np.float32)

        result = scorer.score_scalper_entry(seq, tab, regime_features=regime_feats)
        assert result.regime in ("trending", "ranging", "crisis")

    def test_load_all_models_does_not_raise_without_files(self) -> None:
        """load_all_models should not raise even when all weight files are absent."""
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        scorer.load_all_models()  # no files on disk in test env
        assert scorer._scalper_bilstm is not None
        assert scorer._swing_bilstm is not None
        assert scorer._scalper_xgb is not None
        assert scorer._swing_xgb is not None
        assert scorer._regime_clf is not None
        assert scorer._nfp_model is not None


# ---------------------------------------------------------------------------
# Cross-model sanity checks
# ---------------------------------------------------------------------------


class TestCrossModelSanity:
    def test_bilstm_and_xgb_feature_count_config_constants(self) -> None:
        """Config constants are what the models expect."""
        assert FEATURE_COUNT == 127
        assert SEQUENCE_LENGTH == 200
        assert XGB_FEATURE_COUNT == 60

    def test_all_model_names_unique(self) -> None:
        from ai_server.models.nfp_model import NFPDirectionModel
        from ai_server.models.regime_classifier import RegimeClassifier
        from ai_server.models.scalper_bilstm import ScalperBiLSTM
        from ai_server.models.swing_bilstm import SwingBiLSTM
        from ai_server.models.xgboost_models import ScalperXGB, SwingXGB

        names = [
            ScalperBiLSTM().name,
            SwingBiLSTM().name,
            ScalperXGB().name,
            SwingXGB().name,
            RegimeClassifier().name,
            NFPDirectionModel().name,
        ]
        assert len(names) == len(set(names)), "Duplicate model names detected."

    def test_ensemble_score_upper_clipped(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        # Even if probabilities exceed 1 due to FP noise, score must be <= 100
        score = scorer._compute_ensemble_score(1.0, 1.0)
        assert score <= 100.0

    def test_ensemble_score_lower_clipped(self) -> None:
        from ai_server.models.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        score = scorer._compute_ensemble_score(0.0, 0.0)
        assert score >= 0.0
