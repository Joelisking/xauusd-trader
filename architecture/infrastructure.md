# Infrastructure & Operations — Detailed Architecture

> Deployment: Windows VPS (primary + backup)
> Monitoring: Telegram Bot + Grafana
> Build timeline: 16 weeks across 5 phases

---

## 1. Production Infrastructure

### 1.1 VPS Configuration

| Component | Primary VPS | Backup VPS |
|---|---|---|
| OS | Windows Server 2022 | Windows Server 2022 |
| RAM | 4GB | 2GB (lighter config) |
| CPU | 2 vCPU | 2 vCPU |
| Storage | 60GB SSD | 40GB SSD |
| Provider | Beeks FX (London LD4) or ForexVPS.net (NY4) | Different data center from primary |
| Latency to broker | < 5ms | < 10ms |
| SLA | 99.95%+ | 99.95%+ |
| Failover | — | Automatic activation within 60 seconds of primary failure |

### 1.2 Software Stack on VPS

```
Windows Server 2022
|
|-- MetaTrader 5 (latest build)
|   |-- GoldScalper.ex5 (compiled EA)
|   |-- GoldSwingRider.ex5 (compiled EA)
|   +-- Data files (history, presets)
|
|-- Python 3.11 (Conda environment, pinned deps)
|   |-- AI Server (running as Windows Service)
|   |-- Macro data updater (scheduled task, every 15 min)
|   |-- Watchdog process (separate service)
|   +-- Telegram bot (separate service)
|
|-- SQLite databases
|   |-- trades.db (trade log)
|   |-- predictions.db (AI prediction log)
|   +-- performance.db (rolling metrics)
|
+-- Backup agent
    +-- Daily backup to Backblaze B2
```

### 1.3 Network Architecture

```
+------------------+     < 5ms      +------------------+
|  Primary VPS     | <------------> |  ECN Broker      |
|  (Beeks FX LD4)  |                |  (IC Markets)    |
+------------------+                +------------------+
        |
        | localhost:5001
        |
+------------------+
|  Python AI Server|
|  (same VPS)      |
+------------------+
        |
        | HTTPS (outbound only)
        |
+------------------+     +------------------+     +------------------+
|  FRED API        |     |  Alpha Vantage   |     |  ForexFactory    |
|  (real yield,VIX)|     |  (DXY)           |     |  (calendar XML)  |
+------------------+     +------------------+     +------------------+
        |
        | HTTPS (outbound only)
        |
+------------------+
|  Telegram API    |
|  (alerts)        |
+------------------+
```

---

## 2. Broker Configuration

### 2.1 Recommended Brokers

| Broker | Account Type | XAUUSD Spread | Verdict |
|---|---|---|---|
| **IC Markets** | RAW Spread | avg 0.4 pips | **TOP CHOICE** for scalper |
| **Pepperstone** | Razor | avg 0.7 pips | Excellent for both bots |
| **FP Markets** | Raw | avg 0.8 pips | Solid backup broker |

### 2.2 Prohibited Broker Types

- Any broker with "guaranteed fills" or "no slippage" marketing
- Dealing desk brokers
- Market makers
- Brokers with "dealer approval" clauses in ToS

### 2.3 Multi-Broker Strategy

- Run same bot on 2 brokers simultaneously with small positions
- Compare fill quality monthly
- If average slippage > 0.8 pips over 100 trades on any broker: switch

### 2.4 Symbol Verification

On first install, verify:
- XAUUSD is available with correct symbol naming (some brokers use GOLD, XAUUSDm, etc.)
- Pip value matches expected: 1 pip = $1 per 0.01 lot at standard gold price
- Minimum lot size >= 0.01
- Maximum lot size >= 1.0

---

## 3. Monitoring Stack

### 3.1 Telegram Alerts

Seven alert types, all sent within 500ms of trigger:

| Alert Type | Content | Priority |
|---|---|---|
| **Trade Entry** | Symbol, direction, lot, entry price, SL, TP, AI score, regime, Wyckoff phase | Normal |
| **Trade Exit** | P&L (pips + $), exit reason, session running P&L | Normal |
| **News Shield** | Activation reason, event name, timer. Deactivation confirmation. | High |
| **AI Server Health** | Online/offline status. Heartbeat every 60s. | Critical |
| **Risk Alert** | 5% session loss = YELLOW. 7% = RED halt. 8% daily = FULL STOP. | Critical |
| **Performance Report** | Daily at 22:00 UTC: trades, win rate, P&L, max DD, AI accuracy | Normal |
| **Model Performance** | Weekly: rolling 30-day win rate vs baseline. Monthly retrain trigger. | Normal |

### 3.2 Grafana Dashboard

Dashboard panels:

| Panel | Data Source | Update Frequency |
|---|---|---|
| Equity curve | trades.db | Real-time |
| Daily P&L bar chart | trades.db | End of day |
| Win rate (rolling 30-day) | trades.db | After each trade |
| AI server uptime | health checks | Every 60s |
| AI prediction accuracy | predictions.db | After each trade outcome |
| News shield activations | event log | Real-time |
| Spread distribution | tick log | Every 5 min |
| Current drawdown | trades.db | Real-time |
| Session profitability rate | trades.db | End of session |
| Regime classification history | predictions.db | Every 15 min |

### 3.3 Watchdog Process

Separate lightweight process running as a Windows Service:

```
Every 30 seconds:
  1. Check MT5 heartbeat (is the terminal responding?)
     - No response: send Telegram alert, attempt MT5 restart
  2. Check AI server heartbeat (ping localhost:5001/health)
     - No response: send Telegram alert, restart AI service
  3. Check open position integrity
     - Any position without a server-side stop? ALERT immediately
  4. Check VPS system health
     - RAM > 90%? Disk > 85%? CPU sustained > 95%? ALERT
```

### 3.4 UptimeRobot

- External monitoring (free tier)
- Alert via SMS + Telegram if VPS goes offline for > 2 minutes
- Monitors VPS IP reachability

---

## 4. Backup Strategy

| What | Frequency | Destination | Retention |
|---|---|---|---|
| MT5 data folder | Daily | Backblaze B2 ($0.006/GB/month) | 90 days |
| Model files (.h5, .json) | After each retrain | Git LFS + B2 | All versions |
| SQLite databases | Daily | B2 | 90 days |
| EA source code | Every commit | GitHub private repo | Permanent |
| Python source | Every commit | GitHub private repo | Permanent |
| Grafana dashboards | Weekly | Git repo (JSON export) | Permanent |
| Trade logs | Daily | B2 + local archive | 2 years |

---

## 5. Development Environment

### 5.1 Local Setup

| Tool | Purpose |
|---|---|
| MetaTrader 5 | EA development, backtesting, strategy tester |
| MetaEditor | MQL5 IDE (built into MT5) |
| VS Code / PyCharm | Python AI development |
| Python 3.11 + Conda | AI training and server development |
| GPU (RTX 3060+) or Colab Pro | Model training (NOT on VPS) |
| Git + GitHub | Version control |

### 5.2 Dependencies (requirements.txt)

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
```

All versions pinned in production. Never auto-update packages on VPS.

---

## 6. Build Plan — 16 Weeks

### Phase 1: Foundation (Weeks 1-2)

**Week 1:**
- [ ] Install MT5 from IC Markets / Pepperstone (RAW/Razor account)
- [ ] Verify XAUUSD available with correct symbol naming
- [ ] Download 10 years XAUUSD data: M1, M5, H1, H4 via MT5 History Center
- [ ] Verify data completeness (no gaps > 1 hour in M1)
- [ ] Install Python 3.11 + all dependencies
- [ ] Set up project directory structure
- [ ] Write skeleton MQL5 EA: EMA stack reader with print statements

**Week 2:**
- [ ] Build Python TCP socket server (asyncio): listen :5001, return dummy scores
- [ ] Integrate socket client into both MQL5 EAs
- [ ] Test round-trip: EA sends -> Python logs -> EA receives + prints score
- [ ] Export XAUUSD data to Parquet via MT5 Python library
- [ ] Build feature engineering pipeline: calculate all 127 features
- [ ] Validate feature distributions (NaN, infinity, outliers)
- [ ] Set up macro data feeds: FRED API, Alpha Vantage. Test API calls.

### Phase 2: Bot Logic (Weeks 3-4)

**Week 3 — Scalper Bot:**
- [ ] Implement M5 EMA stack direction reader. Paper trade 3 days.
- [ ] Add M1 entry trigger: pullback detector, RSI check, candle pattern recognizer (9+ patterns)
- [ ] Implement cascade entry model (4 entries, pilot cancellation)
- [ ] Implement custom session-reset VWAP in MQL5
- [ ] Implement exit logic: ATR TP, SL, breakeven, trailing, time exit, direction flip
- [ ] Implement risk manager: dynamic lot sizing, session cap, daily cap
- [ ] Implement spike detector: ATR(1) vs ATR(20)

**Week 4 — Swing Bot:**
- [ ] Implement H4 market structure analyzer (HH/HL/LH/LL with swing points)
- [ ] Add H1 entry logic: 50 EMA pullback, confirmation candle, volume check
- [ ] Implement single-position management with partial close (40%/60%)
- [ ] Implement structural breakdown exit: H4 close below last HL/above last LH
- [ ] Add DXY and real yield trend inputs (file-based from Python)
- [ ] Begin paper trading swing bot (3+ weeks needed)

### Phase 3: AI Development (Weeks 5-8)

**Week 5 — Data Preparation:**
- [ ] Generate scalper labels: forward-looking TP/SL binary labels on M1
- [ ] Generate swing labels: same logic on H1 (up to 96 candles forward)
- [ ] Apply time-based train/val/test split (2015-2022 / 2022-2023 / 2023-2025)
- [ ] Handle class imbalance with class weights
- [ ] Build feature sequences: reshape to (samples, 200, features) for LSTM

**Weeks 6-7 — Model Training:**
- [ ] Build BiLSTM architecture in Keras
- [ ] Build XGBoost model. Tune with Optuna (100 trials).
- [ ] Train BiLSTM: Adam, lr=0.001, batch=32, early stopping patience=10
- [ ] Target: validation AUC > 0.70 (scalper), > 0.68 (swing)
- [ ] Build ensemble: weighted combination, optimize weights via grid search
- [ ] Train regime classifier (Trending/Ranging/Crisis)
- [ ] Train NFP direction model

**Week 8 — AI Server & Integration:**
- [ ] Build full async AI server with all models loaded at startup
- [ ] Expose /score, /health, /performance endpoints
- [ ] Replace dummy scores in EAs with real AI responses
- [ ] Integration test: run full system on live M1 feed for 48 hours
- [ ] Latency test: 500 concurrent requests, verify P95 < 150ms
- [ ] Build prediction logging with actual outcome tracking

### Phase 4: Backtesting & Validation (Weeks 9-11)

- [ ] Scalper backtest: MT5 Strategy Tester, Every Tick, Jan 2020-Dec 2024. Real spread + 2-pip slippage. Target: PF > 1.5, Max DD < 12%.
- [ ] Swing backtest: same period and mode. Target: PF > 1.7.
- [ ] Walk-forward optimization: 12 segments over 5 years. Average must exceed minimums across ALL segments.
- [ ] Stress test: 2008, 2011, 2020 (COVID) extreme volatility periods.
- [ ] Monte Carlo simulation: 1000 randomized trade orders. 95th percentile DD < 18%.
- [ ] Document all results. Export equity curves to PDF.

### Phase 5: Live Deployment (Weeks 12-16)

- [ ] Deploy on MT5 Demo account. Run minimum 6 weeks.
- [ ] Must complete: 1 NFP, 1 CPI, 1 FOMC event in demo.
- [ ] Must achieve: 300+ scalper trades, 30+ swing trades in demo.
- [ ] Compare live vs backtest metrics weekly.
- [ ] Set up production VPS. Install MT5 + Python environment.
- [ ] Deploy AI server as Windows Service with auto-restart.
- [ ] Set up Telegram bot. Test all 7 alert types manually.
- [ ] Set up Grafana dashboard.
- [ ] Start live trading: minimum $1,000 (0.01 micro lots max).
- [ ] Monitor daily for first 2 weeks.
- [ ] Scale to $2,000-$5,000 only after 30 profitable live days matching demo within +/-15%.

---

## 7. Cost Analysis

| Component | Self-Built | Hiring Developer |
|---|---|---|
| MQL5 EAs (both bots) | 8-12 weeks | $2,500-$6,000 |
| Python AI system | 6-10 weeks | $4,000-$10,000 |
| Integration & testing | 4 weeks | $1,500-$3,000 |
| VPS (annual) | $200-$500/yr | $200-$500/yr |
| Macro data APIs | Free (FRED/AlphaVantage) | $0-$200/yr |
| GPU training (cloud) | $50-$200 one-time | $50-$200 one-time |
| **Total** | **~5-6 months** | **$8,000-$20,000** |

---

## 8. Pre-Launch Checklist

- [ ] 10+ years XAUUSD M1, M5, H1, H4 data downloaded and verified (no gaps, correct timezone)
- [ ] All 127 AI features validated — no NaN, no lookahead bias, distributions checked
- [ ] Both EAs backtested in Every Tick mode with real spread + 2-pip slippage, passing minimum thresholds
- [ ] AI models trained with walk-forward validation. Scalper AUC > 0.70, Swing AUC > 0.68.
- [ ] AI server stress tested: 500 concurrent requests, P95 < 150ms
- [ ] Spike detector tested: manually inject 3x ATR candle. Verify bot freezes entries.
- [ ] News Shield tested: trigger manual NFP simulation. Verify all 4 phases activate.
- [ ] Telegram alerts tested: received all 7 alert types on phone.
- [ ] VPS deployed: MT5 + Python AI running as services with auto-restart. Uptime monitoring active.
- [ ] Demo trading: minimum 6 weeks, 400+ scalper trades, 40+ swing trades, 1 NFP, 1 CPI, 1 FOMC
- [ ] Live account with ECN broker. Initial capital $1,000-$2,500.
- [ ] Risk parameters triple-checked: 1.5% per trade, 7% session halt, 10% session cap, 1.5x ATR TP
- [ ] Emergency stop plan documented: both EAs disabled + all positions closed in < 45 seconds
