"""AI model definitions for the XAUUSD trading system."""

from ai_server.models.base import BaseModel
from ai_server.models.ensemble import EnsembleScorer, ScalperEntryResult, SwingEntryResult
from ai_server.models.nfp_model import NFPDirectionModel
from ai_server.models.regime_classifier import RegimeClassifier
from ai_server.models.scalper_bilstm import ScalperBiLSTM
from ai_server.models.swing_bilstm import SwingBiLSTM
from ai_server.models.xgboost_models import ScalperXGB, SwingXGB

__all__ = [
    "BaseModel",
    "ScalperBiLSTM",
    "SwingBiLSTM",
    "ScalperXGB",
    "SwingXGB",
    "RegimeClassifier",
    "NFPDirectionModel",
    "EnsembleScorer",
    "ScalperEntryResult",
    "SwingEntryResult",
]
