# XAUUSD AI Trading System — Architecture Overview

> Version 2.0 | March 2026
> Platform: MetaTrader 5 | Languages: MQL5 + Python 3.11

---

## 1. System Purpose

A dual-bot AI-powered algorithmic trading system for XAUUSD (Gold/USD) that combines
rule-based technical analysis with deep learning ensemble scoring. The system trades
two complementary strategies — scalping (M1/M5) and swing trading (H1/H4) — unified
under a shared risk management and AI intelligence layer.

**Target performance**: 5-9% monthly return, <7% max drawdown, Profit Factor >1.7.

---

## 2. High-Level Architecture

```
+-------------------------------------------------------------------+
|                        LAYER 3: DATA & MONITORING                  |
|  SQLite DB | Parquet Files | FRED/AlphaVantage APIs | Telegram    |
|  Grafana Dashboard | Performance Tracker | Model Storage (Git LFS) |
+-------------------------------------------------------------------+
        |                    |                      |
        v                    v                      v
+-------------------------------------------------------------------+
|                     LAYER 2: AI INTELLIGENCE                       |
|                      Python 3.11 Server                            |
|                                                                    |
|  +-------------------+  +------------------+  +-----------------+  |
|  | Scalper Ensemble  |  | Swing Ensemble   |  | Regime Clf      |  |
|  | BiLSTM+Attn(0.55) |  | BiLSTM+Attn(0.55)|  | Trending/       |  |
|  | XGBoost    (0.45) |  | XGBoost    (0.45)|  | Ranging/Crisis  |  |
|  +-------------------+  +------------------+  +-----------------+  |
|                                                                    |
|  +-------------------+  +------------------+                       |
|  | NFP Direction     |  | News Calendar    |                       |
|  | Model (XGBoost)   |  | + Risk Scorer    |                       |
|  +-------------------+  +------------------+                       |
|                                                                    |
|  Endpoints: /score (entry check) | /health | /performance          |
|  Transport: TCP socket :5001 | JSON | async (asyncio + uvicorn)    |
+-------------------------------------------------------------------+
        ^                    ^                      ^
        |     JSON req/resp  |                      |
        |     < 150ms P95    |                      |
+-------------------------------------------------------------------+
|                     LAYER 1: EXECUTION                             |
|                   MQL5 Expert Advisors (MT5)                       |
|                                                                    |
|  +-----------------------------+  +-----------------------------+  |
|  | BOT A: GOLD SCALPER         |  | BOT B: GOLD SWING RIDER    |  |
|  | Timeframes: M1 / M5        |  | Timeframes: H1 / H4        |  |
|  |                             |  |                             |  |
|  | CDirectionLayer (M5 bias)  |  | CH4DirectionSystem          |  |
|  | CEntryLayer (M1 cascade)   |  | CH1ExecutionEngine          |  |
|  | CExitManager (TP/SL/trail) |  | CSwingExitManager           |  |
|  | CRiskManager (session cap) |  | CSwingRiskManager           |  |
|  | CSpikeDetector             |  | CDXYFilter                  |  |
|  | CNewsShield                |  | CNewsShield                 |  |
|  | CAIClient (socket)         |  | CAIClient (socket)          |  |
|  +-----------------------------+  +-----------------------------+  |
|                                                                    |
|  Shared: Session scheduler | Spread gate | Lot size calculator     |
+-------------------------------------------------------------------+
        |                                           |
        v                                           v
+-------------------------------------------------------------------+
|                     BROKER / MARKET                                 |
|  ECN/STP: IC Markets RAW | Pepperstone Razor | FP Markets Raw     |
|  XAUUSD spread: 0.4-1.2 pips | Execution: < 5ms to broker        |
+-------------------------------------------------------------------+
```

---

## 3. Component Index

| Component | Location | Language | Purpose |
|---|---|---|---|
| Gold Scalper EA | `gold_scalper_ea/` | MQL5 | M1/M5 scalping execution |
| Gold Swing EA | `gold_swing_ea/` | MQL5 | H1/H4 swing execution |
| AI Server | `ai_server/` | Python | Model serving, scoring, health |
| Data Pipeline | `data_pipeline/` | Python | Feature engineering, data export |
| Models | `models/` | Binary | Trained model weights (.h5, .json) |
| Backtest Results | `backtest_results/` | Mixed | Equity curves, performance logs |
| Monitoring | `monitoring/` | Python | Telegram bot, Grafana config |

---

## 4. Data Flow — Scalper Trade Lifecycle

```
1. OnTick() fires in MT5
         |
2. Gate checks (sequential, any failure = skip):
   a. NewsShield.IsActive()?          --> yes: return
   b. SpikeDetector.IsActive()?       --> yes: return
   c. RiskManager.SessionCapReached()? --> yes: return
   d. IsWithinTradingHours()?         --> no:  return
   e. GetCurrentSpread() > 2.0 pips?  --> yes: return
         |
3. DirectionLayer.Get5MBias()
   - EMA stack (21/50/200) alignment
   - Market structure (HH/HL or LL/LH)
   - MACD histogram sign
   - ATR(14) > 5 pips gate
   - VWAP filter (London-NY overlap)
   Result: BULL | BEAR | NONE --> NONE: return
         |
4. EntryLayer.HasSignal(direction)
   - M1 pullback to 21 EMA
   - Rejection candle pattern (hammer/engulfing/pin bar)
   - RSI(7) range check (38-55 longs, 45-62 shorts)
   Result: true | false --> false: return
         |
5. Build 127-feature vector, send to AI server via TCP :5001
         |
6. AI Server processes:
   - BiLSTM(128->64) + Attention(8 heads) --> probability
   - XGBoost on tabular features --> probability
   - Ensemble: (LSTM * 0.55) + (XGB * 0.45) * 100 = entry_score
   - Regime classifier, news risk score
   - Returns JSON response < 150ms
         |
7. If entry_score >= 68 AND approve == true:
   - RiskManager.CalcLotSize() * recommended_lot_multiplier
   - CascadeEntry.Execute(direction, lot, score)
     Entry 1 (Pilot): 0.01 lot, SL below swing low
     Entry 2 (Core):  0.02 lot, after confirmation candle
     Entry 3 (Add):   0.02 lot, on momentum candle
     Entry 4 (Max):   0.01 lot, only if score >= 82 + London-NY overlap
         |
8. ExitManager.ManageOpenTrades() every tick:
   - ATR(14) * 1.5 TP from Entry 1 price
   - Momentum TP extension (ATR * 2.5 if +12 pips in 4 min)
   - Breakeven at +10 pips
   - Trailing stop at 12 pips
   - Time exit at 15 minutes
   - Direction flip exit (M5 EMA stack reversal)
   - VWAP rejection exit (50% close)
   - Hard stop at 20 pips adverse
```

---

## 5. Data Flow — Swing Trade Lifecycle

```
1. H4 candle closes -> evaluate direction system
         |
2. H4 Direction assessment:
   - Market structure: 2+ consecutive HH/HL (bull) or LL/LH (bear)
   - 200 EMA position (above = long only, below = short only)
   - Weekly trend alignment
   - RSI(14) on H4 (above/below 50)
   - Wyckoff phase detection (AI)
   - AI Trend Strength Score >= 72 required
         |
3. H1 pullback detected:
   - Price pulls back to 50 EMA or AI-identified S/D zone
   - RSI(14) retraces to 42-55
   - Confirmation candle closes back in H4 trend direction
   - Volume spike > 1.5x 20-bar average
   - No major news within 4 hours
   - DXY not in strong opposing move
         |
4. AI scoring via TCP :5001 (< 300ms acceptable)
   - Trend Strength Score, Entry Quality Score
   - Wyckoff Phase, Regime, News Risk
         |
5. If approved: single position entry
   - Risk 2% of account
   - SL below last H4 swing low (40-80 pips)
   - TP1 at 1:1.5 R:R -> close 40%
   - TP2 at 1:3.0 R:R -> close 60%
         |
6. Position management:
   - After TP1 hit -> SL to breakeven
   - Max hold: 72 hours
   - Exit on H4 structural breakdown ONLY (not H1 noise)
   - H4 close below last HL (long) = immediate 100% close
   - AI Trend Strength drops below 45 = close 50%
   - Major news within 2h = reduce to 50%, SL to breakeven
   - DXY 3-candle rally against position = close 50%
```

---

## 6. Session Schedule (UTC)

| Session | UTC | Scalper | Swing |
|---|---|---|---|
| Asian | 00:00-07:00 | Offline | Monitor open trades only |
| London Open | 07:00-10:00 | Active (50% size) | Entry allowed if overnight setup confirmed |
| London | 10:00-13:00 | Fully active | Fully active |
| London-NY Overlap | 13:00-17:00 | **PRIME** — full capacity | **PRIME** — full capacity |
| NY | 17:00-21:00 | Active (75% size) | Hold positions, no new entries after 19:00 |
| NY Close | 21:00-00:00 | Close remaining scalps | Evaluate H4 close for structure breaks |

---

## 7. Technology Stack

| Component | Technology | Version/Notes |
|---|---|---|
| Trading Terminal | MetaTrader 5 | Latest build, VPS-hosted |
| EA Language | MQL5 | Compiled to .ex5 |
| AI Framework | TensorFlow + Keras | 2.16+, GPU for training only |
| Gradient Boosting | XGBoost | 2.0+ |
| Data Processing | Pandas + NumPy + TA-Lib | Latest stable |
| Socket Server | Python asyncio + uvicorn | Non-blocking, 2000+ req/s |
| Database | SQLite3 + Parquet | Zero-config, Parquet for ML data |
| Macro Data | FRED API + Alpha Vantage | DXY, real yield, VIX, CPI |
| Economic Calendar | ForexFactory XML | Polled every 30 min |
| Backtesting | MT5 Strategy Tester | Every Tick mode, real spread |
| VPS | Windows Server 2022 | 4GB RAM, 2 vCPU, SSD |
| Monitoring | Telegram Bot + Grafana | Real-time alerts + dashboards |
| Version Control | Git + GitHub (private) | Model weights in Git LFS |

---

## 8. Architecture Sub-Documents

Detailed designs for each subsystem:

- **[Scalper Bot Architecture](architecture/scalper-bot.md)** — M1/M5 direction, cascade entry, exits, session risk
- **[Swing Bot Architecture](architecture/swing-bot.md)** — H4 direction, H1 execution, partial exits, structural closes
- **[AI Engine Architecture](architecture/ai-engine.md)** — BiLSTM+XGBoost ensemble, 127 features, training pipeline, model serving
- **[Risk Management Architecture](architecture/risk-management.md)** — Session caps, spike detection, news shield, failure modes
- **[Communication Protocol](architecture/communication-protocol.md)** — TCP socket spec, JSON schemas, heartbeat, fallback
- **[Infrastructure & Ops](architecture/infrastructure.md)** — VPS, broker config, monitoring, alerts, deployment, build plan

---

## 9. Project Directory Structure

```
xauusd-trader/
|-- ARCHITECTURE.md                    # This file
|-- XAUUSD_AI_Bot_Blueprint_V2.docx   # Source blueprint document
|-- architecture/                      # Detailed architecture docs
|   |-- scalper-bot.md
|   |-- swing-bot.md
|   |-- ai-engine.md
|   |-- risk-management.md
|   |-- communication-protocol.md
|   |-- infrastructure.md
|
|-- gold_scalper_ea/                   # MQL5 Scalper EA source
|   |-- GoldScalper.mq5               # Main EA file
|   |-- Include/
|   |   |-- DirectionLayer.mqh        # M5 EMA stack + structure
|   |   |-- EntryLayer.mqh            # M1 cascade entry system
|   |   |-- ExitManager.mqh           # TP/SL/trail/time management
|   |   |-- RiskManager.mqh           # Position sizing + session cap
|   |   |-- AIClient.mqh              # TCP socket client
|   |   |-- NewsShield.mqh            # Economic calendar integration
|   |   |-- SpikeDetector.mqh         # Flash crash detection
|   |   |-- SessionManager.mqh        # Trading hours gate
|   |   |-- VWAPCalculator.mqh        # Custom session-reset VWAP
|   |   |-- CandlePatterns.mqh        # Pattern recognition (14 patterns)
|   |   |-- MarketStructure.mqh       # HH/HL/LH/LL detection
|   |   +-- Constants.mqh             # Shared constants and enums
|
|-- gold_swing_ea/                     # MQL5 Swing EA source
|   |-- GoldSwingRider.mq5            # Main EA file
|   |-- Include/
|   |   |-- H4DirectionSystem.mqh     # H4 trend analysis
|   |   |-- H1ExecutionEngine.mqh     # H1 entry logic
|   |   |-- SwingExitManager.mqh      # Structural exit rules
|   |   |-- SwingRiskManager.mqh      # 2% per-trade risk
|   |   |-- AIClient.mqh              # Shared socket client
|   |   |-- NewsShield.mqh            # Shared news protocol
|   |   |-- DXYFilter.mqh             # Dollar correlation filter
|   |   |-- MarketStructure.mqh       # Shared structure detector
|   |   +-- Constants.mqh             # Shared constants
|
|-- ai_server/                         # Python AI server
|   |-- server.py                      # Main async TCP server
|   |-- scoring.py                     # Ensemble scoring logic
|   |-- models/
|   |   |-- scalper_bilstm.py         # Scalper BiLSTM+Attention model def
|   |   |-- swing_bilstm.py           # Swing BiLSTM+Attention model def
|   |   |-- xgboost_models.py         # XGBoost model wrappers
|   |   |-- regime_classifier.py      # Trending/Ranging/Crisis classifier
|   |   +-- nfp_model.py              # Post-NFP direction model
|   |-- features/
|   |   |-- feature_engine.py         # 127-feature calculation pipeline
|   |   |-- price_features.py         # OHLCV, EMA, ATR, RSI, MACD, BB, ADX
|   |   |-- derived_features.py       # Temporal, round numbers, session
|   |   +-- macro_features.py         # DXY, real yield, VIX, calendar
|   |-- macro/
|   |   |-- fred_client.py            # FRED API (real yield, VIX)
|   |   |-- alpha_vantage_client.py   # Alpha Vantage (DXY)
|   |   +-- news_calendar.py          # ForexFactory XML parser
|   |-- training/
|   |   |-- train_scalper.py          # Scalper model training pipeline
|   |   |-- train_swing.py            # Swing model training pipeline
|   |   |-- train_regime.py           # Regime classifier training
|   |   |-- label_generator.py        # Forward-looking label creation
|   |   |-- walk_forward.py           # Walk-forward validation
|   |   +-- feature_selection.py      # SHAP-based feature importance
|   |-- config.py                      # Server configuration
|   +-- health.py                      # Health check & performance endpoints
|
|-- data_pipeline/                     # Data acquisition and processing
|   |-- mt5_export.py                  # MT5 -> Parquet data export
|   |-- feature_pipeline.py           # Batch feature engineering
|   |-- data_validator.py             # NaN/infinity/outlier checks
|   +-- macro_updater.py              # Periodic macro data refresh
|
|-- models/                            # Trained model weights (Git LFS)
|   |-- gold_scalper_bilstm.h5
|   |-- gold_swing_bilstm.h5
|   |-- gold_scalper_xgb.json
|   |-- gold_swing_xgb.json
|   |-- regime_classifier.h5
|   +-- nfp_direction_xgb.json
|
|-- backtest_results/                  # Backtest outputs
|   |-- scalper/
|   |-- swing/
|   +-- walk_forward/
|
|-- monitoring/                        # Monitoring and alerting
|   |-- telegram_bot.py               # Alert bot
|   |-- grafana/
|   |   +-- dashboards/               # JSON dashboard configs
|   +-- watchdog.py                    # MT5 heartbeat monitor
|
+-- logs/                              # Runtime logs
    |-- ai_server/
    |-- trades/
    +-- predictions/
```

---

## 10. Version Roadmap

| Version | Timeline | Scope |
|---|---|---|
| **1.0** | Months 1-3 | XAUUSD only. Both bots. BiLSTM+XGBoost. MT5 on VPS. Telegram + Grafana. |
| **1.5** | Months 4-6 | Add XAGUSD. FinBERT NLP sentiment. Interactive web dashboard. Challenger model system. |
| **2.0** | Months 7-12 | Reinforcement Learning (PPO). Portfolio mode. EUR/USD expansion. Order flow analysis (CME L2). |
