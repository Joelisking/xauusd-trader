"""System-wide configuration constants for the XAUUSD AI Trading System."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
PREDICTION_LOG_DIR = LOG_DIR / "predictions"
TRADE_LOG_DIR = LOG_DIR / "trades"
DATA_DIR = PROJECT_ROOT / "data"
BACKTEST_DIR = PROJECT_ROOT / "backtest_results"

# ---------------------------------------------------------------------------
# AI Server
# ---------------------------------------------------------------------------
AI_SERVER_HOST = "127.0.0.1"
AI_SERVER_PORT = 5001
MAX_REQUESTS_PER_SECOND = 100

# ---------------------------------------------------------------------------
# Model file paths
# ---------------------------------------------------------------------------
SCALPER_BILSTM_PATH = MODEL_DIR / "gold_scalper_bilstm.h5"
SWING_BILSTM_PATH = MODEL_DIR / "gold_swing_bilstm.h5"
SCALPER_XGB_PATH = MODEL_DIR / "gold_scalper_xgb.json"
SWING_XGB_PATH = MODEL_DIR / "gold_swing_xgb.json"
REGIME_CLF_PATH = MODEL_DIR / "regime_classifier.h5"
NFP_MODEL_PATH = MODEL_DIR / "nfp_direction_xgb.json"
MODEL_VERSION = "2026-03-01"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FEATURE_COUNT = 127
SEQUENCE_LENGTH = 200       # BiLSTM lookback window (candles)
XGB_FEATURE_COUNT = 60      # Tabular features for XGBoost

# ---------------------------------------------------------------------------
# Ensemble weights
# ---------------------------------------------------------------------------
BILSTM_WEIGHT = 0.55
XGB_WEIGHT = 0.45

# ---------------------------------------------------------------------------
# Scalper thresholds
# ---------------------------------------------------------------------------
SCALPER_MIN_AI_SCORE = 68
SCALPER_MAX_AI_SCORE = 82       # Threshold for cascade Entry 4
SCALPER_MAX_SPREAD = 2.0        # pips
SCALPER_RISK_PCT = 1.5          # per-trade %
SESSION_RISK_CAP = 10.0         # max session loss %
SESSION_HALT_PCT = 7.0          # non-overridable halt %
DAILY_HALT_PCT = 8.0            # full daily stop
MAX_TRADE_DURATION_MIN = 15
BREAKEVEN_TRIGGER_PIPS = 10.0
TRAILING_DISTANCE_PIPS = 12.0
HARD_STOP_PIPS = 20.0
TP_MULTIPLIER = 1.5             # ATR multiplier
MOMENTUM_TP_MULTIPLIER = 2.5
MOMENTUM_TRIGGER_PIPS = 12.0
MOMENTUM_TRIGGER_MINUTES = 4
MIN_ATR_M5_PIPS = 5
MAX_LOT_PCT = 5.0              # hard cap: 5% of account

# ---------------------------------------------------------------------------
# Swing thresholds
# ---------------------------------------------------------------------------
SWING_MIN_TREND_SCORE = 72
SWING_RISK_PCT = 2.0
SWING_TP1_RR = 1.5
SWING_TP2_RR = 3.0
SWING_TP1_CLOSE_PCT = 40.0
SWING_MAX_HOLD_HOURS = 72
SWING_VOLUME_SPIKE_MULT = 1.5
SWING_RSI_LOW = 42
SWING_RSI_HIGH = 55
SWING_NEWS_BLACKOUT_HOURS = 4
SWING_MAX_SPREAD = 2.5

# ---------------------------------------------------------------------------
# AI scoring thresholds
# ---------------------------------------------------------------------------
ENTRY4_MIN_SCORE = 82
TREND_EXHAUSTION_SCORE = 45
NEWS_HALT_THRESHOLD = 75
NEWS_REDUCE_THRESHOLD = 50

# ---------------------------------------------------------------------------
# Lot multiplier tiers
# ---------------------------------------------------------------------------
LOT_MULT_NORMAL = 0.8       # score 68-79
LOT_MULT_HIGH = 1.0         # score 80-99
LOT_MULT_PRIME = 1.2        # score 80+ during London-NY overlap

# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------
SPIKE_ATR_RATIO = 3.0
SPIKE_COOLDOWN_MINUTES = 10

# ---------------------------------------------------------------------------
# Session hours (UTC)
# ---------------------------------------------------------------------------
SESSION_ASIAN_START = 0
SESSION_ASIAN_END = 7
SESSION_LONDON_OPEN_START = 7
SESSION_LONDON_OPEN_END = 10
SESSION_LONDON_START = 10
SESSION_LONDON_END = 13
SESSION_OVERLAP_START = 13
SESSION_OVERLAP_END = 17
SESSION_NY_START = 17
SESSION_NY_END = 21
SESSION_NY_CLOSE_START = 21
SESSION_NY_CLOSE_END = 24

# ---------------------------------------------------------------------------
# Macro API keys (set via environment or .env)
# ---------------------------------------------------------------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# News shield timing (minutes)
# ---------------------------------------------------------------------------
NEWS_DETECTION_MINUTES = 60
NEWS_PRE_MINUTES = 30
NEWS_DURING_MINUTES = 20
NEWS_POST_MINUTES = 75
NFP_POST_MINUTES = 90
