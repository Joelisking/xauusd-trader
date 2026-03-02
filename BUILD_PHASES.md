# XAUUSD AI Trading System — Complete Build Phases

> This document defines every build phase, what each produces, and the exact files to create.
> Designed for session continuity — any fresh Claude Code session can read this and pick up where the last left off.

---

## Project Context

- **What**: Dual-bot AI trading system for XAUUSD (Gold) on MetaTrader 5
- **Two bots**: Scalper (M1/M5) + Swing Rider (H1/H4)
- **AI layer**: BiLSTM + Attention + XGBoost ensemble (Python), communicates with MQL5 EAs via TCP socket
- **Dev environment**: macOS (local development) -> Git push -> Windows VPS (MT5 + production)
- **Source blueprint**: `XAUUSD_AI_Bot_Blueprint_V2.docx` (in project root)
- **Architecture docs**: `ARCHITECTURE.md` + `architecture/*.md` (6 detailed sub-docs)
- **Blueprint sections to reference**: The docx covers Sections 1-11. Architecture docs decompose these into implementable specs.

---

## Current Status Tracker

> Update this section as phases complete.

| Phase | Status | Notes |
|---|---|---|
| Phase 0: Architecture Docs | COMPLETE | ARCHITECTURE.md + 6 sub-docs created |
| Phase 1: Project Scaffolding | COMPLETE | .gitignore, requirements.txt, config.py (all constants), package structure, directories |
| Phase 2: Communication Protocol | COMPLETE | protocol.py (dataclasses + validation), server.py (async TCP :5001), health.py, scoring.py (dummy), models/base.py, macro/news_calendar.py stub, 27 tests passing |
| Phase 3: MQL5 Shared Libraries | COMPLETE | 8 shared includes: Constants, AIClient, MarketStructure, CandlePatterns, NewsShield, SpikeDetector, SessionManager, VWAPCalculator |
| Phase 4: Gold Scalper EA | COMPLETE | GoldScalper.mq5 + DirectionLayer, EntryLayer, ExitManager, RiskManager — full implementations |
| Phase 5: Gold Swing EA | COMPLETE | GoldSwingRider.mq5 + H4DirectionSystem, H1ExecutionEngine, SwingExitManager, SwingRiskManager, DXYFilter — full implementations. Shared includes copied from scalper. |
| Phase 6: Data Pipeline | COMPLETE | mt5_export.py, data_validator.py, macro_updater.py, fred_client.py, alpha_vantage_client.py, news_calendar.py (full), 40 tests passing |
| Phase 7: Feature Engineering | COMPLETE | feature_engine.py (orchestrator), price_features.py (75), derived_features.py (32), macro_features.py (20) = 127 total, feature_pipeline.py (batch), 30 tests passing |
| Phase 8: AI Model Definitions | COMPLETE | ScalperBiLSTM, SwingBiLSTM, ScalperXGB, SwingXGB, RegimeClassifier, NFPDirectionModel, EnsembleScorer — 61 tests passing |
| Phase 9: AI Training Pipeline | COMPLETE | label_generator.py, train_scalper.py, train_swing.py, train_regime.py, train_nfp.py, walk_forward.py (12-seg), feature_selection.py (SHAP), evaluate.py — 22 tests passing |
| Phase 10: AI Server (Full) | NOT STARTED | |
| Phase 11: News Shield | NOT STARTED | |
| Phase 12: Monitoring & Alerts | NOT STARTED | |
| Phase 13: Backtesting Framework | NOT STARTED | |
| Phase 14: Integration Testing | NOT STARTED | |
| Phase 15: Demo Deployment | NOT STARTED | |

---

## PHASE 1: Project Scaffolding

**Goal**: Set up directory structure, dependencies, Git config, and shared configuration.

**Files to create:**

```
.gitignore
requirements.txt
ai_server/__init__.py
ai_server/config.py                    # All system constants: ports, thresholds, paths, model params
ai_server/models/__init__.py
ai_server/features/__init__.py
ai_server/macro/__init__.py
ai_server/training/__init__.py
data_pipeline/__init__.py
monitoring/__init__.py
models/.gitkeep                        # Trained model weights go here (Git LFS)
backtest_results/.gitkeep
logs/.gitkeep
```

**`ai_server/config.py` must define:**
- `AI_SERVER_HOST = "127.0.0.1"`, `AI_SERVER_PORT = 5001`
- Scalper thresholds: `SCALPER_MIN_AI_SCORE = 68`, `SCALPER_MAX_SPREAD = 2.0`, `SCALPER_RISK_PCT = 1.5`, `SESSION_RISK_CAP = 10.0`, `SESSION_HALT_PCT = 7.0`
- Swing thresholds: `SWING_MIN_TREND_SCORE = 72`, `SWING_RISK_PCT = 2.0`, `SWING_TP1_RR = 1.5`, `SWING_TP2_RR = 3.0`
- AI thresholds: `ENTRY4_MIN_SCORE = 82`, `TREND_EXHAUSTION_SCORE = 45`, `NEWS_HALT_THRESHOLD = 75`, `NEWS_REDUCE_THRESHOLD = 50`
- Spike detection: `SPIKE_ATR_RATIO = 3.0`, `SPIKE_COOLDOWN_MINUTES = 10`
- Session hours (UTC): London open 07-10, London 10-13, London-NY overlap 13-17, NY 17-21, NY close 21-00
- Paths: model file paths, log directories, data directories
- Macro API keys placeholders: FRED, Alpha Vantage

**`.gitignore` must include:**
- `*.ex5`, `*.ex4` (compiled MQL5)
- `models/*.h5`, `models/*.json` (use Git LFS instead)
- `logs/`, `__pycache__/`, `.env`, `*.pyc`
- `backtest_results/`
- `.vscode/`, `.idea/`

**`requirements.txt`:**
```
tensorflow>=2.16
xgboost>=2.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
ta-lib>=0.4.28
aiohttp>=3.9
python-telegram-bot>=20.0
shap>=0.43
optuna>=3.4
MetaTrader5>=5.0.45
websockets>=12.0
uvicorn>=0.27
pyarrow>=15.0
```

**Verification**: `python -c "import ai_server.config; print('Config OK')"` succeeds.

---

## PHASE 2: Communication Protocol

**Goal**: Build the TCP socket server and JSON message protocol — the contract between MT5 and Python.

**Reference**: `architecture/communication-protocol.md`

**Files to create:**

```
ai_server/protocol.py                  # Message dataclasses, validation, serialization
ai_server/server.py                    # Async TCP server on port 5001
ai_server/health.py                    # Health check + performance metrics
tests/test_protocol.py                 # Unit tests for message validation
tests/test_server.py                   # Integration test: send JSON, get response
```

**`ai_server/protocol.py` must define:**
- `EntryCheckRequest` dataclass: type, symbol, direction, timeframe, bot, session_hour, dxy_trend, real_yield_trend, vix_level, current_spread, atr_14, session_risk_used, account_drawdown, features (list of 127 floats)
- `EntryCheckResponse` dataclass: entry_score, trend_score, news_risk, wyckoff_phase, regime, approve, recommended_lot_multiplier, model_version, latency_ms
- `HeartbeatRequest` / `HeartbeatResponse` dataclasses
- `validate_request()` — check feature count = 127, no NaN/inf, valid symbol, valid direction
- `serialize()` / `deserialize()` — JSON conversion

**`ai_server/server.py` must implement:**
- Async TCP server using `asyncio.start_server` on `config.AI_SERVER_HOST:AI_SERVER_PORT`
- Accept connection, read JSON until newline delimiter
- Route by `type` field: `"entry_check"` -> scoring handler, `"heartbeat"` -> health handler
- Return JSON response + newline delimiter
- Log every request/response pair to `logs/predictions/YYYY-MM-DD.jsonl`
- Handle malformed JSON gracefully (return error response, don't crash)
- **For now**: scoring handler returns dummy scores `{entry_score: 75, trend_score: 70, news_risk: 10, regime: "trending", wyckoff_phase: "D", approve: true, recommended_lot_multiplier: 1.0}`

**`ai_server/health.py` must implement:**
- Track: uptime_seconds, predictions_today, avg_latency_ms, queue_depth
- Return health status: "healthy" / "degraded" / "error"
- Track model_version string (from config, updated on model reload)

**Verification**:
1. Run `python -m ai_server.server` on Mac
2. `echo '{"type":"heartbeat"}\n' | nc localhost 5001` returns health JSON
3. `echo '{"type":"entry_check","symbol":"XAUUSD","direction":"BUY","timeframe":"M1","bot":"scalper","session_hour":14,"features":[0.0,...127 values...]}\n' | nc localhost 5001` returns scoring JSON
4. `pytest tests/test_protocol.py tests/test_server.py` passes

---

## PHASE 3: MQL5 Shared Libraries

**Goal**: Build the reusable MQL5 include files shared between scalper and swing EAs.

**Reference**: `architecture/scalper-bot.md` (sections 5-6), `architecture/swing-bot.md` (section 7), `architecture/communication-protocol.md`

**Files to create:**

```
gold_scalper_ea/Include/Constants.mqh
gold_scalper_ea/Include/AIClient.mqh
gold_scalper_ea/Include/MarketStructure.mqh
gold_scalper_ea/Include/CandlePatterns.mqh
gold_scalper_ea/Include/NewsShield.mqh
gold_scalper_ea/Include/SpikeDetector.mqh
gold_scalper_ea/Include/SessionManager.mqh
gold_scalper_ea/Include/VWAPCalculator.mqh
```

**`Constants.mqh` must define:**
- Enums: `ENUM_DIRECTION {DIRECTION_NONE, DIRECTION_BULL, DIRECTION_BEAR}`
- Enums: `ENUM_MARKET_STRUCTURE {STRUCT_HH, STRUCT_HL, STRUCT_LH, STRUCT_LL, STRUCT_NONE}`
- Enums: `ENUM_CANDLE_PATTERN {PATTERN_NONE, PATTERN_HAMMER, PATTERN_ENGULFING_BULL, PATTERN_ENGULFING_BEAR, PATTERN_PIN_BAR_BULL, PATTERN_PIN_BAR_BEAR, PATTERN_DOJI, PATTERN_SHOOTING_STAR, PATTERN_MORNING_STAR, PATTERN_EVENING_STAR, ...14 total}`
- Enums: `ENUM_NEWS_PHASE {NEWS_PHASE_NONE, NEWS_PHASE_DETECTION, NEWS_PHASE_PRE, NEWS_PHASE_DURING, NEWS_PHASE_POST}`
- Enums: `ENUM_CASCADE_STATE {CASCADE_IDLE, CASCADE_PILOT, CASCADE_CORE, CASCADE_ADD, CASCADE_MAX, CASCADE_CANCELLED}`
- Enums: `ENUM_SESSION {SESSION_ASIAN, SESSION_LONDON_OPEN, SESSION_LONDON, SESSION_LONDON_NY_OVERLAP, SESSION_NY, SESSION_NY_CLOSE}`
- Enums: `ENUM_REGIME {REGIME_TRENDING, REGIME_RANGING, REGIME_CRISIS}`
- All input parameters for both EAs (shared defaults)

**`AIClient.mqh` — class `CAIClient`:**
- `int m_socket` — socket handle
- `bool Connect()` — connect to localhost:5001, 5s timeout
- `bool SendRequest(string json)` — send JSON + newline
- `string ReceiveResponse()` — read until newline, 3s timeout (scalper) / 5s (swing)
- `bool ScoreEntry(direction, timeframe, bot, features[127], &entry_score, &trend_score, &news_risk, &regime, &wyckoff_phase, &approve, &lot_multiplier)` — builds JSON, sends, parses response
- `bool Heartbeat(&status, &uptime, &model_version)` — heartbeat check
- `int m_consecutive_failures` — tracks failures for fallback mode
- `bool IsInFallbackMode()` — true if 3+ consecutive failures
- `void ResetFailures()` — called on successful response

**`MarketStructure.mqh` — class `CMarketStructure`:**
- `ENUM_MARKET_STRUCTURE DetectSwingPoint(string symbol, ENUM_TIMEFRAMES tf, int shift)` — identifies HH/HL/LH/LL
- `ENUM_DIRECTION GetTrend(string symbol, ENUM_TIMEFRAMES tf, int lookback)` — 2+ consecutive HH/HL = BULL, 2+ LL/LH = BEAR
- Uses fractal-based swing point detection (3-bar or 5-bar fractal)
- Returns DIRECTION_NONE if ambiguous

**`CandlePatterns.mqh` — class `CCandlePatterns`:**
- `ENUM_CANDLE_PATTERN Detect(string symbol, ENUM_TIMEFRAMES tf, int shift)` — returns dominant pattern at given candle
- Implements 14 patterns: hammer, inverted hammer, bullish engulfing, bearish engulfing, bullish pin bar, bearish pin bar, doji, shooting star, morning star, evening star, three white soldiers, three black crows, tweezer top, tweezer bottom
- Each pattern has body/wick ratio thresholds from standard definitions
- Returns `PATTERN_NONE` if no clear pattern

**`NewsShield.mqh` — class `CNewsShield`:**
- `bool IsActive()` — returns true if any phase active (blocks trading)
- `ENUM_NEWS_PHASE GetCurrentPhase()` — which phase we're in
- `void Update()` — called each tick, manages phase transitions based on time
- `bool IsPostNewsWindow()` — true during T+20 to T+75 (reduced-risk entries allowed)
- `void SetNextEvent(datetime event_time, int impact_level)` — set from Python calendar data
- For Phase 1: reads news schedule from a CSV/JSON file updated by Python macro feed
- Full protocol: Detection (T-60) -> Pre-News (T-30) -> During (T-0 to T+20) -> Post-News (T+20 to T+75)

**`SpikeDetector.mqh` — class `CSpikeDetector`:**
- `bool IsActive()` — true if spike detected and cooldown not expired
- `void Update(string symbol)` — check ATR(1) / ATR(20) ratio on M1. If > 3.0: activate, start 10-min cooldown timer.
- `datetime GetCooldownEnd()` — when entries resume

**`SessionManager.mqh` — class `CSessionManager`:**
- `ENUM_SESSION GetCurrentSession()` — based on UTC hour
- `bool IsScalperActive()` — true during London Open, London, London-NY, NY
- `bool IsSwingEntryAllowed()` — true during London Open (if overnight setup), London, London-NY
- `double GetPositionSizeMultiplier()` — 0.5 for London Open, 1.0 for London/Overlap, 0.75 for NY, 0.0 for Asian/Close

**`VWAPCalculator.mqh` — class `CVWAPCalculator`:**
- `double GetVWAP(string symbol)` — session-reset VWAP (resets at session open)
- `void Update(string symbol)` — accumulate price * volume / cumulative volume
- Custom implementation — not standard in MT5
- Uses tick volume as proxy for real volume

**Verification**: Each .mqh file compiles when `#include`'d from a minimal test EA in MetaEditor on VPS.

---

## PHASE 4: Gold Scalper EA

**Goal**: Complete scalper EA with M5 direction system, M1 cascade entry, all exit types, and risk management.

**Reference**: `architecture/scalper-bot.md` (full document)

**Files to create:**

```
gold_scalper_ea/GoldScalper.mq5
gold_scalper_ea/Include/DirectionLayer.mqh
gold_scalper_ea/Include/EntryLayer.mqh
gold_scalper_ea/Include/ExitManager.mqh
gold_scalper_ea/Include/RiskManager.mqh
```

**`GoldScalper.mq5` — main EA:**
- Input parameters (from Constants.mqh): RiskPercent=1.5, SessionRiskCap=10.0, SessionHaltPct=7.0, AIMinScore=68, MaxSpreadPips=2.0, MaxTradeDuration=15, BreakevenTrigger=10.0, TrailingDistance=12.0, HardStopPips=20.0, TPMultiplier=1.5, AIServerPort=5001
- `OnInit()`: initialize all classes, connect AIClient, log startup
- `OnTick()`: sequential gate checks (NewsShield -> SpikeDetector -> RiskManager -> SessionManager -> Spread -> Direction -> Entry -> AI Score -> Execute), then ExitManager.ManageOpenTrades()
- `OnDeinit()`: close socket, log shutdown stats
- `OnTimer()`: 60-second heartbeat to AI server

**`DirectionLayer.mqh` — class `CDirectionLayer`:**
- `ENUM_DIRECTION Get5MBias()` — returns current M5 direction, updated each M5 candle close
- Checks ALL five indicators: EMA stack (21>50>200 = BULL), Market Structure (2+ HH/HL), MACD histogram (positive/negative), ATR(14) > 5 pips gate, VWAP filter (above=long, below=short during overlap)
- All five must agree. Any disagreement = DIRECTION_NONE.
- Caches result between M5 candle closes (does not flicker on tick)

**`EntryLayer.mqh` — class `CEntryLayer`:**
- `bool HasSignal(ENUM_DIRECTION direction)` — checks M1 conditions: pullback to 21 EMA + rejection candle + RSI(7) in range
- `ENUM_CASCADE_STATE GetCascadeState()` — current state machine position
- Cascade state machine: IDLE -> PILOT_ENTERED -> CORE_ENTERED -> ADD_ENTERED -> MAX_ENTERED
- **Critical rule**: if Pilot is negative before next entry triggers, cancel entire cascade
- Entry 1 (Pilot): 0.01 lot, SL below M1 swing low
- Entry 2 (Core): 0.02 lot, next candle closes in direction, Pilot must be positive
- Entry 3 (Add): 0.02 lot, momentum candle (body > 70% of range)
- Entry 4 (Max): 0.01 lot, only if AI score >= 82 AND London-NY overlap active
- `void Reset()` — return to IDLE state

**`ExitManager.mqh` — class `CExitManager`:**
- `void ManageOpenTrades()` — called every tick for all open positions
- Exit types in priority order:
  1. Hard stop: 20 pips max adverse
  2. Direction flip: M5 EMA stack reversal -> close ALL
  3. Time exit: 15 minutes elapsed -> close at market
  4. Initial SL hit
  5. VWAP rejection: 2 consecutive rejection candles at VWAP -> close 50%, SL to BE
  6. Primary TP: ATR(14) * 1.5 from Entry 1 price
  7. Momentum TP: +12 pips in 4 minutes -> extend to ATR * 2.5, SL to BE
  8. Trailing stop: after BE set, trail 12 pips behind
- Breakeven move: at +10 pips, move SL to entry price

**`RiskManager.mqh` — class `CRiskManager`:**
- `double CalcLotSize(double sl_pips)` — `Lot = (AccountBalance * 0.015) / (sl_pips * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE))`
- `bool SessionCapReached()` — true if cumulative session loss >= SessionRiskCap
- `void TrackTrade(double pnl)` — update session P&L
- Session halt at 7% cumulative loss (non-overridable)
- Daily cap: max 2 sessions. Session 2 only if Session 1 net positive.
- Winning session bonus: 6%+ profit -> next session at 1.0% risk
- Hard cap: max lot = 5% of account regardless of formula

**Verification**:
1. Compile in MetaEditor on VPS with 0 errors
2. Attach to XAUUSD M1 chart in Strategy Tester (visual mode)
3. Verify: EMA stack direction prints to journal, spread gate blocks during wide spread, session gate blocks outside hours
4. With Python server running: verify AI score requests/responses appear in logs

---

## PHASE 5: Gold Swing EA

**Goal**: Complete swing EA with H4 direction system, H1 execution, partial exits, and macro filters.

**Reference**: `architecture/swing-bot.md` (full document)

**Files to create:**

```
gold_swing_ea/GoldSwingRider.mq5
gold_swing_ea/Include/H4DirectionSystem.mqh
gold_swing_ea/Include/H1ExecutionEngine.mqh
gold_swing_ea/Include/SwingExitManager.mqh
gold_swing_ea/Include/SwingRiskManager.mqh
gold_swing_ea/Include/DXYFilter.mqh
```

(Plus copy shared includes from Phase 3: Constants.mqh, AIClient.mqh, MarketStructure.mqh, NewsShield.mqh, SessionManager.mqh)

**`GoldSwingRider.mq5` — main EA:**
- Inputs: RiskPercent=2.0, AIMinTrendScore=72, AIExhaustionScore=45, TP1_RR=1.5, TP2_RR=3.0, TP1_ClosePct=40.0, MaxHoldHours=72, VolumeSpikeMult=1.5, RSI_Low=42, RSI_High=55, NewsBlackoutHours=4
- `OnTick()`: gate checks then H4/H1 evaluation logic
- Evaluates primarily on candle closes (H4 and H1), not every tick
- Every tick: manage existing positions (check structural breakdowns, TP hits)

**`H4DirectionSystem.mqh` — class `CH4DirectionSystem`:**
- `ENUM_DIRECTION GetH4Trend()` — evaluates all 6 indicators
- Market Structure: 2+ HH/HL (bull) or LL/LH (bear) on H4
- 200 EMA: above = long only, below = short only (binary, no exceptions)
- Weekly alignment: check weekly chart direction matches H4
- RSI(14) on H4: above 50 = bull, below 50 = bear
- AI Trend Strength Score >= 72 required
- Wyckoff phase from AI (Phase C/D = highest priority)

**`H1ExecutionEngine.mqh` — class `CH1ExecutionEngine`:**
- `bool HasEntry(ENUM_DIRECTION h4_direction)` — checks all 7 H1 conditions
- Pullback to 50 EMA or AI-identified S/D zone
- RSI(14) pulled back to 42-55
- Confirmation candle closes back in H4 direction
- Volume spike > 1.5x 20-bar average
- No news within 4 hours
- DXY not in strong opposing move

**`SwingExitManager.mqh` — class `CSwingExitManager`:**
- Mandatory structural closes (non-negotiable):
  - H4 close below last HL (long) = 100% close
  - H4 close above last LH (short) = 100% close
  - H4 200 EMA flip = 100% close
- AI-driven: trend score drops from >72 to <45 = close 50%, trail rest
- News: NFP/CPI/FOMC within 2h = reduce to 50%, SL to BE
- DXY headwind: 3-candle H4 rally against position = close 50%
- TP management: TP1 hit -> close 40%, SL to BE. TP2 hit -> close remaining 60%.
- Time limit: 72 hours max hold
- **Critical design**: exits only on H4 candle CLOSES, never on H1 noise or tick-level P&L

**`SwingRiskManager.mqh` — class `CSwingRiskManager`:**
- `double CalcLotSize(double sl_pips)` — `Lot = (AccountBalance * 0.02) / (sl_pips * pip_value)`
- SL: below last H4 swing low (longs) / above last H4 swing high (shorts). Typically 40-80 pips.
- TP1: entry + (SL_distance * 1.5). Close 40%.
- TP2: entry + (SL_distance * 3.0). Close remaining 60%.

**`DXYFilter.mqh` — class `CDXYFilter`:**
- Reads `dxy_trend.json` file (written by Python macro feed every 15 min)
- `ENUM_DIRECTION GetDXYTrend()` — UP/DOWN/NEUTRAL
- `bool IsMacroHeadwind(ENUM_DIRECTION gold_direction)` — true if DXY opposes gold position
- `bool HasThreeCandleRally(ENUM_DIRECTION against)` — detect 3-candle H4 DXY rally

**Verification**: Same as Phase 4 — compile, attach to H1 chart in Strategy Tester, verify H4 direction logging and AI scoring.

---

## PHASE 6: Data Pipeline

**Goal**: Export MT5 historical data to Parquet, validate it, and set up macro data feeds.

**Reference**: `architecture/ai-engine.md` (section 6.1), `architecture/infrastructure.md` (Phase 1 Week 2)

**Files to create:**

```
data_pipeline/mt5_export.py            # Export XAUUSD M1/M5/H1/H4 to Parquet
data_pipeline/data_validator.py        # NaN/inf/outlier/gap detection
data_pipeline/macro_updater.py         # Periodic DXY/yield/VIX data refresh
ai_server/macro/fred_client.py         # FRED API client (real yield, VIX)
ai_server/macro/alpha_vantage_client.py # Alpha Vantage client (DXY)
ai_server/macro/news_calendar.py       # ForexFactory XML parser + risk scorer
```

**`mt5_export.py`:**
- Connect to MT5 via `MetaTrader5` Python library
- Export XAUUSD: M1 (10 years), M5, H1, H4 to Parquet files in `data/` directory
- Each file: columns = time, open, high, low, close, tick_volume, spread, real_volume
- Save as `data/XAUUSD_M1.parquet`, `data/XAUUSD_M5.parquet`, etc.
- Run on VPS only (where MT5 is installed)

**`data_validator.py`:**
- Check for NaN, infinity, outliers (>5 sigma)
- Detect time gaps (>1 hour in M1 during market hours = flag)
- Verify timezone consistency
- Report data completeness percentage
- Output validation report

**`macro_updater.py`:**
- Runs as scheduled task (every 15 minutes)
- Fetches DXY, US 10Y Real Yield, VIX, WTI Oil
- Writes `dxy_trend.json` file for MQL5 DXYFilter to read
- Writes macro features to SQLite for AI model input

**`fred_client.py`:** FRED API for real yield (DGS10, DFII10), VIX (VIXCLS)
**`alpha_vantage_client.py`:** Alpha Vantage for DXY
**`news_calendar.py`:** ForexFactory XML feed parser, event impact classification, news risk score calculation

**Verification**: Run `mt5_export.py` on VPS, verify Parquet files. Run `data_validator.py`, verify report shows <1% gaps. Run `macro_updater.py`, verify `dxy_trend.json` written with valid data.

---

## PHASE 7: Feature Engineering

**Goal**: Build the 127-feature calculation pipeline, both for batch (training) and real-time (serving).

**Reference**: `architecture/ai-engine.md` (section 4)

**Files to create:**

```
ai_server/features/feature_engine.py   # Orchestrator: assembles all 127 features
ai_server/features/price_features.py   # 75 price/technical features
ai_server/features/derived_features.py # 32 derived/temporal features
ai_server/features/macro_features.py   # 20 macro/correlation features
data_pipeline/feature_pipeline.py      # Batch feature calculation for training data
tests/test_features.py                 # Validate feature count, no NaN, distributions
```

**Feature groups (127 total):**

*Price & Technical (75):*
- OHLCV multi-timeframe (M1, M5, H1, H4)
- EMA(8), EMA(21), EMA(50), EMA(200) — values AND rates of change (deltas)
- ATR(14), ATR(21)
- RSI(7), RSI(14), RSI(21)
- MACD(12/26/9): line, signal, histogram, histogram ROC
- Bollinger Bands: width, %B
- ADX(14), +DI, -DI
- Stochastic(5,3)
- 14 candlestick pattern numerical encodings
- Market structure labels (HH/HL/LH/LL encoded 0-3 for M5 and H4)
- VWAP distance (price distance from VWAP as fraction of ATR)

*Derived & Temporal (32):*
- Hour of day (0-23), day of week (0-4)
- Days to next major news event (1-30)
- Distance from round $100 levels (3200, 3300, etc.)
- Spread vs 20-period average ratio
- Tick volume vs 20-bar average ratio
- Session number (1st or 2nd)
- Current drawdown from account peak

*Macro & Correlation (20):*
- DXY: trend direction, distance from 50 EMA, momentum (ROC)
- US 10Y Real Yield: direction, level
- VIX: level, 5-day ROC
- WTI Oil: direction
- Economic calendar: binary flags for events within 1H, 4H, 24H
- Upcoming event impact score (0-3)

**`feature_pipeline.py`** runs batch calculation over entire Parquet dataset, producing training-ready feature matrices.

**Verification**: `pytest tests/test_features.py` — verify 127 features, no NaN, distributions within expected ranges. Load sample Parquet, run pipeline, check output shape.

---

## PHASE 8: AI Model Definitions

**Goal**: Define model architectures in code (not yet trained).

**Reference**: `architecture/ai-engine.md` (sections 2-3, 5)

**Files to create:**

```
ai_server/models/base.py               # Abstract model interface
ai_server/models/scalper_bilstm.py     # Scalper BiLSTM+Attention architecture
ai_server/models/swing_bilstm.py       # Swing BiLSTM+Attention architecture
ai_server/models/xgboost_models.py     # XGBoost model wrappers (scalper + swing)
ai_server/models/regime_classifier.py  # Trending/Ranging/Crisis classifier
ai_server/models/nfp_model.py          # Post-NFP direction model (XGBoost)
ai_server/models/ensemble.py           # Weighted ensemble combiner
```

**BiLSTM architecture (both scalper and swing, same structure):**
```
Input(shape=(200, num_features))
-> BiLSTM(128, return_sequences=True)
-> Dropout(0.3)
-> BiLSTM(64, return_sequences=True)
-> MultiHeadAttention(num_heads=8, key_dim=64)
-> GlobalAveragePooling1D()
-> Dense(1, activation='sigmoid')
```

**XGBoost:** Trained on last-candle tabular features (60 features). Tuned with Optuna.

**Ensemble:** `Final = (BiLSTM * 0.55) + (XGBoost * 0.45)`. Weights configurable.

**Regime classifier:** Separate Keras model. Classes: Trending (ADX>25), Ranging (ADX<20), Crisis (VIX>25).

**NFP model:** Simple XGBoost on: previous NFP surprise, gold price level, DXY level, preceding 5-day gold trend.

**Verification**: Each model can be instantiated and do a forward pass with random data of correct shape.

---

## PHASE 9: AI Training Pipeline

**Goal**: Build complete training pipeline with label generation, walk-forward validation, and model evaluation.

**Reference**: `architecture/ai-engine.md` (section 6)

**Files to create:**

```
ai_server/training/label_generator.py   # Forward-looking TP/SL binary labels
ai_server/training/train_scalper.py     # Full scalper training pipeline
ai_server/training/train_swing.py       # Full swing training pipeline
ai_server/training/train_regime.py      # Regime classifier training
ai_server/training/train_nfp.py         # NFP direction model training
ai_server/training/walk_forward.py      # 12-segment walk-forward validation
ai_server/training/feature_selection.py # SHAP-based feature importance
ai_server/training/evaluate.py          # Model evaluation metrics, plots
```

**Label generation:**
- Scalper: for each M1 candle with signal, look forward 60 candles. TP hit before SL = 1, SL first = 0.
- Swing: same on H1, forward 96 candles (4 days).
- Min samples: 15,000 scalper, 3,000 swing.

**Training config:**
- Time-based split: 2015-2022 train, 2022-2023 val, 2023-2025 test (sacred)
- Class weighting (NOT SMOTE): `class_weight = {0: 1.0, 1: wins/losses}`
- Adam optimizer, lr=0.001, batch=32, early stopping patience=10
- Target: AUC > 0.70 (scalper), > 0.68 (swing)
- XGBoost: Optuna 100 trials, maximize validation AUC

**Walk-forward:** 12 segments, 8 months train / 2 months test / 1 month gap each.

**Verification**: Run training on a small data subset (1 year). Verify model saves, loads, and predicts correctly. Walk-forward produces 12 evaluation segments.

---

## PHASE 10: AI Server (Full)

**Goal**: Replace dummy scores with real model inference. Full production server.

**Reference**: `architecture/ai-engine.md` (section 7), `architecture/communication-protocol.md`

**Files to modify:**

```
ai_server/server.py                    # Replace dummy scoring with real model calls
ai_server/scoring.py                   # Full ensemble scoring with all models
```

**`scoring.py` must implement:**
- Load all models at startup (not per-request)
- `score_scalper_entry(features)` — BiLSTM + XGBoost ensemble, regime, news risk
- `score_swing_entry(features)` — same with swing models, trend strength, Wyckoff phase
- Ensemble: `(bilstm_prob * 0.55 + xgb_prob * 0.45) * 100`
- Lot multiplier: 0.8 (score 68-79), 1.0 (score 80-99), 1.2 (score 80+ in overlap)

**Latency targets:** P95 < 150ms (scalper), < 300ms (swing). If exceeded: optimize with TF Serving.

**Verification**: Stress test 500 concurrent requests. Verify P95 < 150ms. Run 48-hour integration test with live M1 data feed.

---

## PHASE 11: News Shield (Full Implementation)

**Goal**: Complete the 4-phase news protocol with calendar integration and NFP-specific logic.

**Reference**: `architecture/risk-management.md` (section 5)

**Files to modify/create:**

```
ai_server/macro/news_calendar.py       # Full ForexFactory parser + risk scoring
gold_scalper_ea/Include/NewsShield.mqh  # Full 4-phase implementation
gold_swing_ea/Include/NewsShield.mqh    # Swing-specific news handling
monitoring/news_schedule.py             # Write news schedule file for EAs
```

**Full protocol per event type:**
- NFP: T-60 lockdown, T+20 post-NFP scalper with dedicated AI model, T+90 normal
- CPI: T-30 lockdown, inverse reaction pattern handling
- FOMC: T-60 lockdown, swing stays flat entire FOMC day
- Fed speeches: T-30 lockdown
- Other high-impact: T-30 standard protocol

**Verification**: Simulate NFP event manually. Verify all 4 phases activate with correct timing. Verify Telegram alerts fire for each phase transition.

---

## PHASE 12: Monitoring & Alerts

**Goal**: Telegram bot with all 7 alert types, watchdog process, Grafana dashboard configs.

**Reference**: `architecture/infrastructure.md` (section 3)

**Files to create:**

```
monitoring/telegram_bot.py             # Full alert bot with all 7 types
monitoring/watchdog.py                 # MT5 + AI server heartbeat monitor
monitoring/performance_tracker.py      # Rolling metrics, daily report generation
monitoring/grafana/dashboards/equity.json
monitoring/grafana/dashboards/performance.json
monitoring/grafana/dashboards/ai_health.json
```

**7 alert types:** Trade Entry, Trade Exit, News Shield, AI Server Health, Risk Alert, Daily Performance Report, Weekly Model Performance.

**Watchdog:** Every 30 seconds: check MT5 heartbeat, AI server heartbeat, open position integrity (all have server-side stops), VPS system health.

**Verification**: Send test alerts for all 7 types. Verify receipt on Telegram. Run watchdog, verify it detects simulated MT5 down and AI server down.

---

## PHASE 13: Backtesting Framework

**Goal**: Set up Strategy Tester configurations, walk-forward scripts, Monte Carlo simulation, stress testing.

**Reference**: `architecture/infrastructure.md` (Phase 4: Weeks 9-11)

**Files to create:**

```
backtest_results/configs/scalper_backtest.set   # MT5 Strategy Tester preset
backtest_results/configs/swing_backtest.set     # MT5 Strategy Tester preset
data_pipeline/monte_carlo.py                    # Randomize trade order 1000x, estimate worst-case DD
data_pipeline/stress_test.py                    # Run on 2008, 2011, 2020 extreme periods
data_pipeline/backtest_analyzer.py              # Parse MT5 backtest results, generate equity curves
```

**Targets:**
- Scalper: PF > 1.5, Max DD < 12% (with 2-pip slippage, real spread, Every Tick mode)
- Swing: PF > 1.7
- Walk-forward: all 12 segments must pass minimum thresholds
- Monte Carlo: 95th percentile DD < 18%

**Verification**: Run scalper backtest Jan 2020-Dec 2024. Compare results to target thresholds. Run Monte Carlo, verify DD distribution.

---

## PHASE 14: Integration Testing

**Goal**: End-to-end system test — both EAs + AI server + monitoring running together.

**Tasks (no new files, testing existing):**
1. Start Python AI server on VPS
2. Attach both EAs to XAUUSD charts (Scalper on M1, Swing on H1)
3. Run for 48 hours continuous on demo account
4. Verify: AI requests/responses logged, trades executed per rules, News Shield activates, Telegram alerts received, no crashes
5. Measure: AI response latency, trade execution latency, memory usage over time
6. Verify fallback mode: kill AI server -> EA enters fallback -> restart server -> EA recovers

**Verification**: 48-hour clean run. Zero crashes. All alert types received. Fallback mode tested and recovered.

---

## PHASE 15: Demo Deployment

**Goal**: 6-week minimum demo trading period before any live capital.

**Requirements before going live:**
- [ ] 300+ scalper trades completed in demo
- [ ] 30+ swing trades completed in demo
- [ ] Survived 1 NFP event
- [ ] Survived 1 CPI event
- [ ] Survived 1 FOMC event
- [ ] Demo metrics within +/-15% of backtest expectations
- [ ] Win rate: scalper > 55%, swing > 52%
- [ ] Profit factor > 1.4
- [ ] Max drawdown < 12%
- [ ] All monitoring systems verified operational for full 6 weeks
- [ ] Emergency stop plan tested: both EAs disabled + all positions closed in < 45 seconds

---

## Cross-Phase Dependencies

```
Phase 1 (Scaffolding)
  |
  v
Phase 2 (Protocol) -----> Phase 3 (Shared MQL5 Libs)
  |                              |
  v                              v
Phase 6 (Data Pipeline)    Phase 4 (Scalper EA)
  |                              |
  v                              v
Phase 7 (Features)         Phase 5 (Swing EA)
  |                              |
  v                              |
Phase 8 (Model Defs)            |
  |                              |
  v                              |
Phase 9 (Training)              |
  |                              |
  v                              v
Phase 10 (Full AI Server) <---- Both EAs ready
  |
  v
Phase 11 (News Shield) + Phase 12 (Monitoring)
  |
  v
Phase 13 (Backtesting)
  |
  v
Phase 14 (Integration)
  |
  v
Phase 15 (Demo Deploy)
```

Python-side phases (2, 6, 7, 8, 9, 10) can progress on Mac without MT5.
MQL5-side phases (3, 4, 5) need compilation on VPS but code can be written on Mac.
Phase 11+ requires both sides operational.
