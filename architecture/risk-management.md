# Risk Management — Detailed Architecture

> Principle: Capital preservation is non-negotiable. No single trade, session, or event can cause catastrophic loss.
> The AI NEVER overrides risk rules.

---

## 1. Risk Hierarchy

```
+-------------------------------------------------------+
|  LEVEL 1: PER-TRADE RISK                               |
|  Scalper: 1.5% of account | Swing: 2.0% of account     |
|  Dynamic lot sizing using current pip value              |
+-------------------------------------------------------+
         |
         v
+-------------------------------------------------------+
|  LEVEL 2: SESSION RISK (Scalper Only)                   |
|  10% cap per session | 7% halt trigger                  |
|  Max 2 sessions/day | Session 2 only if Session 1 > 0   |
+-------------------------------------------------------+
         |
         v
+-------------------------------------------------------+
|  LEVEL 3: DAILY RISK                                    |
|  8% daily loss = FULL STOP                              |
|  Winning session bonus: next session at 1.0% risk       |
+-------------------------------------------------------+
         |
         v
+-------------------------------------------------------+
|  LEVEL 4: SYSTEM-WIDE SAFETY NETS                       |
|  Spike detector | News Shield | VIX circuit breaker     |
|  AI fallback mode | Max concurrent positions            |
+-------------------------------------------------------+
         |
         v
+-------------------------------------------------------+
|  LEVEL 5: INFRASTRUCTURE SAFETY                         |
|  Server-side hard stops | VPS failover | Watchdog       |
|  Emergency manual shutdown (< 45 seconds)               |
+-------------------------------------------------------+
```

---

## 2. Per-Trade Risk

### 2.1 Scalper Lot Sizing

```
Lot = (Account_Balance * 0.015) / (SL_pips * Pip_Value)

Where:
  - Account_Balance: current equity (recalculated EVERY trade)
  - 0.015: 1.5% risk per trade
  - SL_pips: distance to stop loss in pips (typically 10-16 for gold scalper)
  - Pip_Value: SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE) — NEVER hardcoded

Example ($10,000 account, 12 pip SL):
  Lot = (10000 * 0.015) / (12 * 1.0) = 150 / 12 = 12.5 -> 0.12 lots
```

### 2.2 Swing Lot Sizing

```
Lot = (Account_Balance * 0.02) / (SL_pips * Pip_Value)

Where:
  - 0.02: 2% risk per trade
  - SL_pips: typically 40-80 pips for gold swing

Example ($10,000 account, 60 pip SL):
  Lot = (10000 * 0.02) / (60 * 1.0) = 200 / 60 = 3.33 -> 0.03 lots
```

### 2.3 AI Score Adjustment

| AI Entry Score | Lot Multiplier |
|---|---|
| 68-79 | 0.80 (80% of calculated lot) |
| 80-100 | 1.00 (full calculated lot) |
| 80-100 + London-NY overlap | 1.20 (120% — size up on high confidence) |

### 2.4 Hard Caps

- **Maximum lot size = 5% of account** regardless of formula output (absolute safety net)
- Lot size is recalculated every trade, not once at bot start
- Pip value uses `SYMBOL_TRADE_TICK_VALUE` — accounts for gold price level changes

---

## 3. Session Risk Management (Scalper)

| Parameter | Value | Rationale |
|---|---|---|
| Session Risk Budget | 10% of account at session start | $10,000 account = $1,000 max session loss |
| Per-Trade Risk | 1.5% = $150 per trade on $10K | Max ~6 losing trades before session halt |
| Session Halt Trigger | Cumulative session loss hits **7%** | Immediate halt, bot disables itself, no override |
| Daily Cap | Max **2 sessions** per day | Session 2 ONLY if Session 1 was net positive |
| Winning Session Bonus | Session profitable at 6%+ | Next session's per-trade risk reduced to 1.0% (protect gains) |

### Session Risk Flow

```
Session Start
    |
    v
Track cumulative P&L per session
    |
    +-- Each trade: check remaining budget >= 1.5%
    |       |-- No: session ends
    |       +-- Yes: allow trade
    |
    +-- After each trade close:
    |       |-- Cumulative loss >= 7%?  --> HALT (no override)
    |       |-- Cumulative loss >= 9%?  --> should never reach here (7% halts first)
    |       +-- Continue
    |
    +-- Session ends naturally (time window closes)
            |
            +-- Was session net positive?
                    |-- Yes: Session 2 eligible
                    +-- No: No Session 2 today
```

---

## 4. Spike Detector

Protects against flash crashes and sudden 150+ pip moves.

### 4.1 Detection Logic

```
ATR_1  = ATR(1) on M1   // single-candle range
ATR_20 = ATR(20) on M1  // 20-candle average range

If ATR_1 / ATR_20 > 3.0:
    SPIKE DETECTED
    -> Halt ALL new entries for 10 minutes
    -> Send Telegram alert
    -> Log spike details (time, magnitude, direction)
```

### 4.2 Additional Spike Rules

- Maximum concurrent open scalp positions = **4** (the cascade)
- Never open a second cascade while first is active
- Outside primary trading hours (thin liquidity): scalper operates at **50% max position size**
- All SL orders are **hard server-side stops** — broker fills them even in extreme conditions

---

## 5. News Shield — Four-Phase Protocol

### 5.1 Phase Overview

```
T-60 min          T-30 min          T-0            T+20 min         T+75 min
    |                 |                |                |                |
    v                 v                v                v                v
[DETECTION]    [PRE-NEWS]        [DURING]       [POST-NEWS]      [NORMAL]
 Calendar       Close scalps      Zero trading    Careful re-entry   Full ops
 Spread watch   Eval swings       Log spike       60% sizing         resume
 VIX monitor    Block entries     Track spread    Post-news AI
```

### 5.2 Phase 1: Detection (T-60 minutes)

- ForexFactory XML feed polled every 30 minutes
- Flag all HIGH impact events affecting USD, Gold, or global risk
- **Spread anomaly detector**: if XAUUSD spread > 3x 20-period average = immediate News Shield activation regardless of calendar
- **VIX real-time monitor**: VIX jump > 2.5 in 30 minutes = geopolitical shock protocol

### 5.3 Phase 2: Pre-News Protocol (T-30 minutes)

| Bot | Action |
|---|---|
| **Scalper** | Close ALL open positions. No new entries. Hard block. |
| **Swing** | If in profit >= 60% of TP2 distance: close 50%. Move all SLs to breakeven. |
| **System** | Set `NEWS_SHIELD_ACTIVE` flag. Send Telegram alert. Pre-place bracket orders at key S/R levels. |

### 5.4 Phase 3: During Event (T-0 to T+20 minutes)

- **Zero trading activity.** All entries blocked by hard gate.
- Log direction and magnitude of initial spike
- Spread monitor: if spread > 5 pips when window opens, extend blackout by 10 minutes

### 5.5 Phase 4: Post-News Entry Window (T+20 to T+75 minutes)

| Condition | Rule |
|---|---|
| Spread gate | Must be < 2.5 pips |
| Direction | M5 must show 3+ consecutive candles in clear post-news direction |
| Entry type | Scalper ONLY for first 45 min. No swing entries until H1 closes confirming direction. |
| Position size | 60% of normal (0.9% per trade instead of 1.5%) |
| AI model | **Post-news AI model** activated (separately trained on post-news patterns) |
| Time window | T+20 to T+75 only. Normal ops resume after 75 min. |

### 5.6 NFP-Specific Protocol

NFP is gold's most predictable high-impact event. Historical 68% directional continuation rate post-NFP.

| Time | Action |
|---|---|
| T-60 (12:30 UTC) | ALL positions closed. Full lockdown. Every 1st Friday. |
| T+0 | Record initial direction of first 30-second spike and magnitude |
| T+5 to T+15 | Log consolidation vs extension (whipsaw vs directional) |
| T+20 | If spread < 2.5 pips AND M5 shows 3 candles in clear direction: activate Post-NFP Scalper |
| T+20 to T+90 | Post-NFP Scalper at 60% sizing. TP = 1.5x normal (gold's NFP moves are larger) |
| T+90 | Return to full normal operation |

### 5.7 Event Impact Table

| Event | Typical XAUUSD Move | Protocol |
|---|---|---|
| NFP | 300-600 pips | Full lockdown T-60 to T+20 |
| CPI / PPI | 150-400 pips | Lockdown T-30. Inverse reaction pattern. |
| FOMC Decision | 200-500 pips | Lockdown T-60. Swing bot stays flat entire FOMC day. |
| Fed Speeches (Powell) | 50-200 pips | Lockdown T-30. |
| ISM PMI / GDP / Retail | 30-100 pips | Standard lockdown T-30 / post-news T+15. |
| Geopolitical Events | Unpredictable | VIX spike detection handles this. |

---

## 6. VIX Circuit Breaker

| VIX Condition | System Response |
|---|---|
| VIX > 25 | Swing bot pauses new longs (market in crisis mode) |
| VIX > 30 for 3+ consecutive days | **Reduced Mode**: 50% sizes, tighter stops, swing bot paused |
| VIX jump > 2.5 in 30 minutes | Geopolitical shock — News Shield activates |
| VIX jump > 3 in 1 hour | All systems enter Reduced Mode |

---

## 7. AI Fallback Mode

When the Python AI server is unresponsive:

| Stage | Condition | Action |
|---|---|---|
| Warning | 1 failed request | Retry after 5 seconds |
| Alert | 2 consecutive failed requests | Send Telegram: "AI SERVER WARNING" |
| Fallback | 3 consecutive failed requests | Switch to **Conservative Fallback Mode** |

### Conservative Fallback Mode

- No AI scores used
- Only take entries where ALL rule-based filters score 100%
- Entry criteria are far stricter than normal
- Continue operating until AI server recovers
- Send Telegram alert every 5 minutes while in fallback

### AI Server Auto-Recovery

- Python AI server runs as Windows Service / systemd
- Auto-restart on crash, latency < 15 seconds
- Heartbeat endpoint pinged every 60 seconds by EA
- Any timeout > 5 seconds triggers alert

---

## 8. The 12 Failure Modes — Prevention Matrix

| # | Failure Mode | Detection | Primary Fix |
|---|---|---|---|
| 1 | **Overfitting** | Val accuracy > 8% below train | Walk-forward validation, Dropout(0.3), early stopping, SHAP feature selection (top 40) |
| 2 | **Regime Shift** | 30-day live perf > 20% below 6-month backtest avg | Regime classifier, rolling retrain (90-day window, 3x recent weight), VIX circuit breaker |
| 3 | **Slippage/Spread** | Backtest vs live fill comparison | 2.5-pip slippage in all backtests, spread gate < 2.0 pips, ECN brokers only, real-tick backtesting |
| 4 | **VPS Downtime** | Watchdog heartbeat failure | Primary + backup VPS, 60s failover, server-side hard stops, Telegram alert on heartbeat miss |
| 5 | **AI Server Loss** | 3 consecutive request failures | Fallback mode, auto-restart service, heartbeat endpoint, health dashboard |
| 6 | **Flash Crash/Spike** | ATR(1) > 3x ATR(20) | Max 4 concurrent scalps, 10-min entry halt, server-side stops, reduced size outside prime hours |
| 7 | **Model Decay** | Rolling 30-day accuracy < 58% | Monthly mandatory retrain, prediction logging, challenger model system, feature importance monitoring |
| 8 | **Over-Optimization** | Parameter sensitivity > +/-20% breaks strategy | Set params from logic not backtests, robustness testing, sacred out-of-sample, max 12 adjustable params |
| 9 | **Broker Manipulation** | Avg slippage > 0.8 pips over 100 trades | ECN only, fill price logging, multi-broker redundancy, ToS audit |
| 10 | **Psychological Interference** | Manual override journal shows 5+ wrong interventions | Pre-defined thresholds, no real-time P&L dashboard, drawdown education, intervention journal |
| 11 | **Insufficient Demo** | Deploy before 6 weeks / 300+ scalper trades / 30+ swings | Min 6 weeks demo including 1 NFP + 1 CPI + 1 FOMC |
| 12 | **Wrong Lot Size** | Account wipe on single trade | Dynamic pip value from `SYMBOL_TRADE_TICK_VALUE`, 5% hard cap, recalculate every trade |

---

## 9. Emergency Stop Plan

Must be able to disable BOTH EAs and close ALL positions in under **45 seconds**.

### 9.1 Steps

1. Press EA global disable button in MT5 (disables all EAs instantly)
2. Right-click Trade tab -> "Close All Positions"
3. Verify all positions closed in Journal tab
4. Send manual Telegram: "EMERGENCY STOP EXECUTED"

### 9.2 Automated Emergency Triggers

| Trigger | Automated Response |
|---|---|
| Daily loss >= 8% | Full stop — both EAs disable, close all positions |
| AI server down > 10 minutes | Scalper halts completely. Swing manages existing only (no new entries). |
| Spread > 10 pips sustained > 5 min | Both EAs pause all activity |
| Account equity < 70% of starting balance | Permanent halt — manual review required |

---

## 10. Risk Parameters Summary

| Parameter | Scalper | Swing |
|---|---|---|
| Per-trade risk | 1.5% | 2.0% |
| Max concurrent positions | 4 (cascade) | 1 per direction |
| Stop loss range | 10-16 pips | 40-80 pips |
| Max adverse excursion | 20 pips (hard stop) | SL level only |
| Session risk cap | 10% | N/A |
| Session halt | 7% loss | N/A |
| Daily halt | 8% | 8% |
| Max hold time | 15 minutes | 72 hours |
| AI min score | 68 | 72 |
| Spread gate | < 2.0 pips | < 2.5 pips |
| News blackout | 30 minutes | 4 hours |
