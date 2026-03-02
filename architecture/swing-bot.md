# Bot B: Gold Swing Rider — Detailed Architecture

> Timeframes: H1 (execution) / H4 (direction anchor)
> Strategy: Trend-following swing trades aligned with institutional H4 structure
> Hold duration: 24-72 hours
> Exit trigger: H4 structural breakdown only — never H1 noise

---

## 1. Architecture Overview

```
+------------------------------------------+
|  H4 DIRECTION SYSTEM                      |
|  "The Institutional View"                 |
|  Updated every H4 candle close            |
|  Output: BULL_TREND | BEAR_TREND | FLAT   |
+------------------------------------------+
         |
         | trend direction + strength
         v
+------------------------------------------+
|  H1 EXECUTION ENGINE                      |
|  Evaluated every H1 candle close          |
|  Output: entry signal with S/D zone       |
+------------------------------------------+
         |
         | features vector (127)
         v
+------------------------------------------+
|  AI FILTER (real-time)                    |
|  Trend Strength Score >= 72 required      |
|  Wyckoff Phase detection                  |
+------------------------------------------+
         |
         | approved + phase + lot mult
         v
+------------------------------------------+
|  POSITION MANAGEMENT                      |
|  Single entry, partial exit model         |
|  TP1 (40%) -> TP2 (60%) -> structural     |
+------------------------------------------+
```

---

## 2. H4 Direction System — The Institutional View

The 4-hour chart represents institutional positioning. Hedge funds and central bank operations leave visible footprints in H4 structure. The swing bot identifies these footprints and aligns with them.

### 2.1 H4 Direction Indicators

| Tool | BULL TREND | BEAR TREND | NO TRADE |
|---|---|---|---|
| **Market Structure** (Primary) | 2+ consecutive HH/HL on H4 | 2+ consecutive LL/LH on H4 | Ambiguous / mixed structure |
| **200 EMA on H4** | Price above 200 EMA = long bias ONLY | Price below 200 EMA = short bias ONLY | Price oscillating around EMA |
| **Weekly Trend Alignment** | Weekly chart direction aligns with H4 | Weekly chart direction aligns with H4 | Weekly opposes H4 = skip |
| **RSI(14) on H4** | Above 50 | Below 50 | Overbought/oversold = wait for pullback |
| **Wyckoff Phase (AI)** | Accumulation Phase C/D detected | Distribution Phase B/C detected | Phase A/B/E = reduced sizing |
| **AI Trend Strength Score** | Score >= 72 | Score >= 72 | Score < 72 = no entry |

### 2.2 Key Principle: Weekly Alignment

Entering in alignment with the weekly trend increases win rate by ~18% in XAUUSD backtests. The H4 direction system checks weekly trend as a mandatory confirmation — never trades against it.

### 2.3 The 200 EMA Rule

This single filter eliminates ~40% of losing swing setups. It is binary:
- Price above H4 200 EMA = **only long entries allowed**
- Price below H4 200 EMA = **only short entries allowed**
- No exceptions, no discretion

---

## 3. H1 Execution Logic

### 3.1 Entry Conditions (ALL Must Be Met)

1. H4 trend confirmed + AI Trend Strength Score >= 72
2. H1 price pulls back to 50 EMA **or** AI-identified institutional supply/demand zone
3. H1 RSI(14) has pulled back to 42-55 range (trend continuation, not reversal)
4. H1 confirmation candle closes BACK in H4 trend direction after pullback
5. Volume spike on confirmation candle: tick volume > 1.5x 20-bar average (institutional entry proxy)
6. No major news within 4 hours (News Shield active check)
7. DXY trend is NOT in strong opposing move (macro alignment required)

### 3.2 The Biggest Swing Trading Mistake — Addressed

The hardest problem: preventing premature exits during NORMAL H1 pullbacks within an H4 uptrend.

A 25-pip pullback on H1 is completely normal during a gold uptrend.

**Solution**: Use ONLY **H4 CANDLE CLOSES** as structural breakdown triggers — never H1 candles, ticks, or unrealized P&L. The bot ignores H1 noise and anchors to H4 reality.

---

## 4. Position Management — Single Entry, Partial Exit

### 4.1 Entry Sizing

| Parameter | Value | Rationale |
|---|---|---|
| Position Size | Risk exactly 2% of account | Swing trades need larger stops — intentional |
| Lot Calculation | `Lot = (Account * 0.02) / (SL_pips * pip_value)` | Recalculated per trade |
| Stop Loss | Below last H4 swing low (longs) / above last H4 swing high (shorts) | Typically 40-80 pips |

### 4.2 Take Profit Levels

| Level | R:R Ratio | Action |
|---|---|---|
| **TP1** | 1:1.5 | Close 40% of position. Banks profit, converts to free ride. |
| **TP2** | 1:3.0 | Close remaining 60%. Gold's larger ranges make 1:3 achievable regularly. |

### 4.3 After TP1 Hit

- Move SL to **breakeven** immediately
- Worst case from here: breakeven (zero loss)
- Best case: TP2 hit for full 1:3 R:R on 60% of position

### 4.4 Time Limit

- Maximum **72 hours** hold time
- If TP2 not hit and H4 structure still intact after 72h: re-evaluate or trail SL aggressively

---

## 5. Exit Rules — When the Story Changes

### 5.1 Mandatory Structural Close Triggers

These are **non-negotiable**. When any fires, the bot executes immediately.

| Trigger | Condition | Action |
|---|---|---|
| **H4 Structure Break (Long)** | H4 candle CLOSES below the last Higher Low | Close 100% immediately |
| **H4 Structure Break (Short)** | H4 candle CLOSES above the last Lower High | Close 100% immediately |
| **H4 200 EMA Flip** | Price closes below 200 EMA in a long trade | Close 100% — institutional support gone |
| **AI Trend Exhaustion** | Trend Strength Score drops from >72 to below 45 | Close 50% immediately, trail rest tightly |
| **Major News Proximity** | NFP/CPI/FOMC within 2 hours | Reduce to 50%, move SL to breakeven. No exceptions. |
| **DXY Macro Headwind** | DXY makes 3-candle rally on H4 against gold long | Close 50% |

### 5.2 Exit Priority Order

```
1. H4 structural breakdown (any type) — immediate 100% close
2. H4 200 EMA flip — immediate 100% close
3. 72-hour time limit
4. Initial SL hit
5. Major news proximity — reduce to 50%
6. DXY headwind — reduce to 50%
7. AI trend exhaustion — close 50%, trail rest
8. TP1 hit — close 40%, SL to breakeven
9. TP2 hit — close remaining 60%
```

### 5.3 What Does NOT Trigger an Exit

- H1 pullback (even 25-40 pips)
- H1 candle patterns against the position
- Unrealized P&L drawdown within SL range
- Intraday noise, tick-level fluctuations

---

## 6. Macro Filters

### 6.1 DXY (US Dollar Index) Integration

Gold moves inversely to USD. The swing bot reads DXY trend data from a file updated every 15 minutes by the Python macro feed.

| DXY Condition | Swing Bot Response |
|---|---|
| DXY in strong downtrend | Favor gold longs — wind at our back |
| DXY in strong uptrend | Favor gold shorts — or reduce long sizing |
| DXY making 3-candle H4 rally against position | Close 50% of gold position |
| DXY neutral/ranging | Normal operation |

### 6.2 US 10-Year Real Yield

Gold's strongest fundamental driver. Moves inversely to real yield.

| Yield Condition | Swing Bot Response |
|---|---|
| Real yield rising | Headwind for gold longs — reduce sizing or favor shorts |
| Real yield falling | Tailwind for gold longs — normal or increased sizing |
| Direction input to AI | Real yield direction and level are AI input features |

### 6.3 VIX (Volatility Index)

| VIX Condition | Swing Bot Response |
|---|---|
| VIX > 25 | Pause new swing longs — market in crisis mode |
| VIX > 30 for 3+ days | Reduced Mode: 50% sizes, tighter stops |
| VIX jump > 2.5 in 30 min | Geopolitical shock — activate News Shield |

---

## 7. MQL5 Class Structure

```
GoldSwingRider.mq5 (main EA)
|
|-- CH4DirectionSystem
|   |-- Market structure analyzer (HH/HL/LH/LL on H4)
|   |-- Swing point identification algorithm
|   |-- 200 EMA position checker
|   |-- Weekly trend alignment checker
|   |-- RSI(14) on H4 evaluator
|   +-- Direction state (BULL_TREND / BEAR_TREND / FLAT)
|
|-- CH1ExecutionEngine
|   |-- 50 EMA pullback detector
|   |-- Supply/demand zone reader (from AI)
|   |-- RSI(14) range validator (42-55 for continuation)
|   |-- Confirmation candle classifier
|   |-- Volume spike detector (1.5x 20-bar average)
|   +-- Entry signal generator
|
|-- CSwingExitManager
|   |-- H4 structural breakdown detector
|   |-- 200 EMA flip monitor
|   |-- TP1 / TP2 management
|   |-- Partial close executor (40% / 60% split)
|   |-- Breakeven mover (after TP1)
|   |-- Time limit enforcer (72 hours)
|   |-- AI trend score degradation handler
|   +-- DXY headwind detector
|
|-- CSwingRiskManager
|   |-- Lot size calculator: Lot = (Account * 0.02) / (SL_pips * pip_value)
|   |-- SL placement: below last H4 swing low/high
|   |-- TP1 calculator: entry + (SL_distance * 1.5)
|   |-- TP2 calculator: entry + (SL_distance * 3.0)
|   +-- Max position count enforcer
|
|-- CDXYFilter
|   |-- File reader (DXY data written by Python every 15 min)
|   |-- DXY trend direction parser
|   |-- 3-candle rally detector
|   +-- Macro alignment scorer
|
|-- CAIClient (shared with scalper)
|   |-- TCP socket to localhost:5001
|   |-- JSON request/response
|   +-- Fallback mode
|
|-- CNewsShield (shared with scalper)
|   |-- 4-phase protocol
|   |-- Swing-specific: evaluate open trades, partial close if near TP
|   +-- FOMC day: stay flat entirely
|
+-- CSessionManager
    |-- H4 candle close event handler
    |-- H1 candle close event handler
    +-- Weekend position evaluation
```

---

## 8. Swing Bot Input Parameters

```mql5
input double RiskPercent       = 2.0;    // Per-trade risk %
input int    AIMinTrendScore   = 72;     // Min AI trend strength
input int    AIExhaustionScore = 45;     // Score below this = reduce
input double TP1_RR            = 1.5;    // TP1 risk:reward ratio
input double TP2_RR            = 3.0;    // TP2 risk:reward ratio
input double TP1_ClosePct      = 40.0;   // % to close at TP1
input int    MaxHoldHours      = 72;     // Max position hold time
input double VolumeSpikeMult   = 1.5;    // Tick volume vs 20-bar avg
input int    RSI_Low           = 42;     // RSI pullback low bound
input int    RSI_High          = 55;     // RSI pullback high bound
input int    NewsBlackoutHours = 4;      // No entry if news within N hours
input int    AIServerPort      = 5001;   // Python AI server port
input string DXYDataFile       = "dxy_trend.json"; // Macro data file path
```

---

## 9. State Machine — Swing Trade Lifecycle

```
              [SCANNING]
                  |
                  | H4 trend confirmed + H1 signal + AI approved
                  v
            [POSITION_OPEN]
                  |
        +---------+---------+
        |                   |
    TP1 hit           structural break
        |                   |
        v                   v
  [PARTIAL_CLOSE]      [FULL_CLOSE]
  (40% closed,              |
   SL to breakeven)         v
        |              [SCANNING]
        |
   +----+----+
   |         |
  TP2 hit  structural break / time limit / macro exit
   |         |
   v         v
[FULL_CLOSE] [FULL_CLOSE]
   |              |
   v              v
[SCANNING]   [SCANNING]
```

---

## 10. Interaction with Scalper Bot

The swing bot and scalper bot operate independently but share:

1. **News Shield** — same calendar, same protocol
2. **AI Client** — same TCP connection to Python server (different request types)
3. **Session Manager** — swing bot respects the same UTC schedule

They do NOT share:
- Risk budget (separate per-trade % and separate position tracking)
- Direction signals (M5 direction and H4 direction can disagree — this is normal)
- Exit logic (completely independent)

**Version 2.0 addition**: Portfolio mode where both bots share a unified risk budget, preventing simultaneous opposite positions.
