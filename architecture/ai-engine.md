# AI Engine — Detailed Architecture

> Architecture: Bidirectional LSTM + Multi-Head Attention + XGBoost Ensemble
> Role: Filter and confidence scorer — NEVER issues BUY/SELL signals
> Outputs: Entry Quality Score, Trend Strength Score, News Risk Score, Wyckoff Phase

---

## 1. Design Philosophy

The AI layer is a **filter**, not a decision maker. The rule-based logic in the MQL5 EAs determines direction and entry/exit criteria. The AI modifies:

- **Confidence** — should this setup be taken at all?
- **Position sizing** — how much to risk on this setup?
- **Risk posture** — how tight should stops be?

| AI Capability | Status | Usage |
|---|---|---|
| Classify market regime | Works well | Drives bot mode switching |
| Score entry quality on pattern similarity | Works well | Entry Quality Score 0-100 |
| Detect institutional order flow zones | Works well | Supply/demand zone input to H1 |
| Predict exact price at future time | **Fails** | Strictly prohibited in design |
| Generate 90%+ win rate by itself | **Fails** | AI is a filter only |
| Detect news behavioral anomalies | Works well | News Risk Score 0-100 |
| Identify Wyckoff phase | Works moderately | Improves swing timing |
| Operate without rule-based risk management | **Catastrophic** | AI never overrides risk rules |

---

## 2. Model Architecture

### 2.1 Ensemble Overview

```
                    127 Features
                    /           \
                   /             \
    +-------------+               +-------------+
    | Sequential  |               | Tabular     |
    | Features    |               | Features    |
    | (200, ~100) |               | (1, 60)     |
    +------+------+               +------+------+
           |                             |
           v                             v
    +-------------+               +-------------+
    | BiLSTM      |               | XGBoost     |
    | Layer 1:128 |               | Gradient    |
    | Layer 2:64  |               | Boosting    |
    | +Dropout0.3 |               | (Optuna     |
    | +Attention  |               |  tuned)     |
    | (8 heads)   |               |             |
    +------+------+               +------+------+
           |                             |
           v                             v
       probability                   probability
       (0.0 - 1.0)                  (0.0 - 1.0)
           |                             |
           +----------+    +-------------+
                      |    |
                      v    v
               +------+----+------+
               | Ensemble Combiner |
               | LSTM * 0.55      |
               | XGB  * 0.45      |
               +--------+---------+
                        |
                        v
                  Score (0 - 100)
```

### 2.2 BiLSTM + Attention Network

```
Input Shape: (batch, 200, num_features)
    |
    v
BiLSTM Layer 1: 128 units (both directions)
    - Processes M1 sequence of last 200 candles forward AND backward
    |
    v
Dropout: 0.3 (prevents overfitting)
    |
    v
BiLSTM Layer 2: 64 units
    |
    v
Multi-Head Attention: 8 heads
    - Identifies which historical candles are most predictive
    - Critical for pattern recognition in gold
    |
    v
Dense Output: 1 unit, sigmoid activation
    - Output: probability 0.0 to 1.0
```

### 2.3 XGBoost Model

- Trained on same features but as **tabular data** (not sequence)
- Input: last-candle features only (60 features)
- Optimized on technical indicator interactions
- Hyperparameter tuning: Optuna, 100 trials, maximizing validation AUC
- Faster inference than LSTM — serves as a complementary view

### 2.4 Ensemble Weights

```
Final Score = (BiLSTM probability * 0.55) + (XGBoost probability * 0.45)
```

Weights determined by cross-validation performance. BiLSTM gets higher weight due to superior sequential pattern recognition on gold time series.

### 2.5 Why Not Other Architectures (V1)?

| Architecture | Reason Rejected for V1 |
|---|---|
| Full Transformer (GPT-style) | 10x more data/compute needed. Overkill for single instrument. BiLSTM+Attention achieves comparable accuracy with 5x less compute. |
| Reinforcement Learning | Requires live data collection phase. Planned for V2.0 after 6 months of live operation. |
| Pure CNN | Good for pattern detection but lacks sequential memory. Misses temporal dependencies critical for multi-session gold patterns. |

---

## 3. AI Outputs — Four Scores

| Output | Range | Consumer | Gate |
|---|---|---|---|
| **Entry Quality Score** | 0-100 | Scalper: execute if >= 68. Swing: execute if >= 72. | Score 80-100: max lot. Score 68-79: 80% of max lot. |
| **Trend Strength Score** | 0-100 | Swing Bot direction filter (>= 72 required). Controls trailing stop tightness. | Score < 45 mid-trade = reduce exposure 50%. |
| **News Risk Score** | 0-100 | Both bots. 100 = maximum risk. | > 75: HALT all trading. 50-75: halve sizes + widen stops 20%. < 50: normal. |
| **Wyckoff Phase** | A/B/C/D/E | Swing Bot entry timing. | Phase C (Spring/Upthrust) + Phase D (SoS/SoW) = highest priority. Phase A/B/E = reduced sizing. |

---

## 4. Input Features — 127 Total

### 4.1 Price & Technical Features (75 features)

| Feature Group | Count | Details |
|---|---|---|
| OHLCV multi-timeframe | ~20 | Last 200 candles on M1, M5, H1, H4 |
| EMA values + deltas | 8 | EMA(8), EMA(21), EMA(50), EMA(200) — values AND rates of change |
| ATR | 2 | ATR(14), ATR(21) — gold volatility context |
| RSI multi-period | 3 | RSI(7), RSI(14), RSI(21) — three timeframe momentum |
| MACD | 4 | Line, signal, histogram, histogram rate of change |
| Bollinger Bands | 2 | Band width, %B position — squeeze detection |
| ADX | 3 | ADX(14), +DI, -DI — trend strength |
| Stochastic | 2 | Stochastic(5,3) — momentum divergence |
| Candlestick patterns | 14 | Numerical encoding: hammer, engulfing, doji, pin bar, morning/evening star, etc. |
| Market structure | 4 | HH/HL/LH/LL encoded (0-3) for M5 and H4 |
| VWAP distance | 1 | Price distance from VWAP as fraction of ATR |
| Other | ~12 | Various derived price features |

### 4.2 Derived & Temporal Features (32 features)

| Feature | Notes |
|---|---|
| Hour of day (0-23) | Top 10 predictor — gold behaves distinctly each session |
| Day of week (0-4) | Monday gaps, Friday profit-taking are real patterns |
| Days to next major news (1-30) | Forward-looking calendar feature |
| Distance from round numbers | $100 levels (3200, 3300, etc.) — gold respects these |
| Spread vs 20-period avg | > 1.5x = anomaly flag |
| Tick volume vs 20-bar avg | 1.5x+ = institutional activity |
| Session number | 1st or 2nd session of day |
| Current drawdown from peak | > 5% = risk-off mode |
| Other derived | ~24 additional temporal/session features |

### 4.3 Macro & Correlation Features (20 features)

| Feature | Source | Notes |
|---|---|---|
| DXY trend direction | Alpha Vantage | Gold's strongest short-term correlation |
| DXY distance from 50 EMA | Alpha Vantage | |
| DXY momentum (ROC) | Alpha Vantage | |
| US 10Y Real Yield direction | FRED API | Gold's strongest fundamental driver |
| US 10Y Real Yield level | FRED API | |
| VIX level | FRED API | Risk-off detector |
| VIX 5-day rate of change | FRED API | |
| WTI Oil direction | External feed | Inflation expectations proxy |
| Econ calendar: event within 1H | ForexFactory | Binary flag |
| Econ calendar: event within 4H | ForexFactory | Binary flag |
| Econ calendar: event within 24H | ForexFactory | Binary flag |
| Upcoming event impact score | ForexFactory | 0=none, 1=low, 2=medium, 3=high |
| Other macro | Various | ~8 additional macro features |

---

## 5. Auxiliary Models

### 5.1 Regime Classifier

- **Architecture**: Separate model (Keras)
- **Classes**: Trending / Ranging / Crisis
- **Labels**: ADX > 25 = Trending, ADX < 20 = Ranging, VIX > 25 = Crisis
- **Usage**: Each regime activates different bot parameters
  - Trending: normal operation
  - Ranging: scalper pauses, swing tightens stops
  - Crisis: 50% sizes, tighter stops, swing paused

### 5.2 NFP Direction Model

- **Architecture**: XGBoost (simple, fast)
- **Input**: Previous NFP surprise, gold price level, DXY level, preceding 5-day gold trend
- **Output**: Post-NFP gold direction prediction
- **Usage**: Activated T+20 after NFP for post-news scalper entries
- **Training data**: 15 years of post-NFP gold behavior

---

## 6. Training Pipeline

### 6.1 Data Requirements

| Requirement | Specification |
|---|---|
| Data source | MT5 historical + Dukascopy tick data + Refinitiv/TradingView macro |
| Volume | 10 years M1 = ~4 million rows |
| Split | 70% train / 15% val / 15% test. **Time-based split, NEVER random.** |
| Train period | 2015-2022 |
| Validation period | 2022-2023 |
| Test period | 2023-2025 (sacred — evaluated ONCE at the end) |

### 6.2 Label Generation

**Scalper labels:**
- For each M1 candle with a scalper signal, look forward up to 60 candles
- If TP hit before SL: label = 1 (win)
- If SL hit first: label = 0 (loss)
- Minimum 15,000 labeled scalper entries required

**Swing labels:**
- Same logic on H1 timeframe, looking forward up to 96 H1 candles (4 days)
- Minimum 3,000 labeled swing entries required

### 6.3 Class Imbalance

- Gold typically has 58/42 win/loss split in labeled data
- Use **class weighting** (not SMOTE): `class_weight = {0: 1.0, 1: wins/losses}`
- SMOTE on time series creates data leakage — strictly prohibited

### 6.4 Feature Sequences

- Reshape tabular data into `(samples, 200, features)` for LSTM input
- XGBoost uses last-candle tabular features `(samples, 60)`

### 6.5 Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Early stopping patience | 10 epochs |
| Target validation AUC | > 0.70 (scalper), > 0.68 (swing) |
| XGBoost tuning | Optuna, 100 trials, maximize validation AUC |
| Ensemble weight optimization | Grid search on validation set |

### 6.6 Walk-Forward Validation

- 12 segments over 5-year test period
- Each segment: 8 months train, 2 months test, 1-month gap (no data leakage)
- Average performance must exceed minimum benchmarks across ALL 12 segments

### 6.7 Retraining Schedule

- **Monthly retraining is MANDATORY**
- Add last 90 days of live trading data
- Retrain overnight
- Validate new model beats current model on last 6 months before deploying
- Challenger model system: always keep production + newly retrained challenger running in paper mode

---

## 7. Model Serving Architecture

### 7.1 Server Design

```python
# Async TCP server on port 5001
# All models loaded at startup (not per-request — critical for latency)

Models loaded:
  - scalper_bilstm (gold_scalper_bilstm.h5)
  - swing_bilstm   (gold_swing_bilstm.h5)
  - scalper_xgb    (gold_scalper_xgb.json)
  - swing_xgb      (gold_swing_xgb.json)
  - regime_clf     (regime_classifier.h5)

Endpoints:
  /score         - Entry quality check (main endpoint)
  /health        - System status
  /performance   - Rolling metrics
```

### 7.2 Scoring Flow

```
1. Receive 127 features from MT5 EA
2. Extract sequential features -> reshape to (1, 200, ~100)
3. Extract tabular features -> reshape to (1, 60)
4. Run BiLSTM prediction (GPU if available, CPU acceptable for inference)
5. Run XGBoost prediction
6. Ensemble: (LSTM * 0.55) + (XGB * 0.45) * 100
7. Run regime classifier
8. Fetch current news risk score from calendar
9. Build response JSON
10. Return within 150ms (scalper) / 300ms (swing)
```

### 7.3 Latency Targets

| Metric | Target |
|---|---|
| P95 response time (scalper) | < 150ms |
| P95 response time (swing) | < 300ms |
| Concurrent request capacity | 500+ requests |
| Server restart latency | < 15 seconds |

### 7.4 Performance Tracking

- Log every prediction with actual market outcome
- Calculate rolling 30-day accuracy in real-time
- Alert if rolling accuracy drops below 58%
- Weekly feature importance monitoring (SHAP values)
- If top-10 feature drops to bottom-20, investigate regime change

---

## 8. Overfitting Prevention

| Defense | Implementation |
|---|---|
| Dropout | 0.3 between LSTM layers |
| Early stopping | Halt when val loss stops improving for 10 epochs |
| Feature selection | Limit to 40 most important (SHAP-based). More features = more overfitting. |
| Walk-forward validation | Never evaluate on training/tuning data |
| Robustness testing | Strategy must survive +/-20% parameter variation |
| Out-of-sample | Last 15% (2024-2025) evaluated ONCE, at the very end |
| Emergency retrain | If live win rate < 55% for 200 consecutive trades |
| Simplicity | Target max 12 adjustable parameters system-wide |

---

## 9. Python Module Structure

```
ai_server/
|-- server.py                   # Main async TCP server (asyncio + uvicorn)
|-- scoring.py                  # Ensemble scoring logic
|-- config.py                   # Server configuration, model paths, thresholds
|-- health.py                   # Health check + performance metrics endpoints
|
|-- models/
|   |-- scalper_bilstm.py      # Scalper BiLSTM+Attention model definition
|   |-- swing_bilstm.py        # Swing BiLSTM+Attention model definition
|   |-- xgboost_models.py      # XGBoost model wrappers
|   |-- regime_classifier.py   # Trending/Ranging/Crisis classifier
|   +-- nfp_model.py           # Post-NFP direction model
|
|-- features/
|   |-- feature_engine.py      # 127-feature calculation pipeline
|   |-- price_features.py      # OHLCV, EMA, ATR, RSI, MACD, BB, ADX, Stoch
|   |-- derived_features.py    # Temporal, round numbers, session, drawdown
|   +-- macro_features.py      # DXY, real yield, VIX, calendar events
|
|-- macro/
|   |-- fred_client.py         # FRED API (US 10Y real yield, VIX)
|   |-- alpha_vantage_client.py # Alpha Vantage (DXY)
|   +-- news_calendar.py       # ForexFactory XML parser + risk scorer
|
+-- training/
    |-- train_scalper.py       # Scalper model training pipeline
    |-- train_swing.py         # Swing model training pipeline
    |-- train_regime.py        # Regime classifier training
    |-- label_generator.py     # Forward-looking label creation
    |-- walk_forward.py        # Walk-forward validation framework
    +-- feature_selection.py   # SHAP-based feature importance ranking
```
