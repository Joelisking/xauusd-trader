# Bot A: Gold Scalper — Detailed Architecture

> Timeframes: M1 (execution) / M5 (direction)
> Strategy: Rapid precision entries on pullbacks, cascade position building
> Active window: London Open (07:00-10:00 UTC) + London-NY Overlap (13:00-17:00 UTC)
> Max trade duration: 15 minutes

---

## 1. Two-Layer Architecture

The scalper operates on two distinct layers that must agree before any entry:

```
+------------------------------------------+
|  DIRECTION LAYER (M5)                     |
|  Updated every M5 candle close            |
|  Output: BULL | BEAR | NONE              |
+------------------------------------------+
         |
         | direction signal
         v
+------------------------------------------+
|  EXECUTION LAYER (M1)                     |
|  Evaluated every tick                     |
|  Output: cascade entry sequence           |
+------------------------------------------+
         |
         | features vector (127)
         v
+------------------------------------------+
|  AI FILTER (real-time)                    |
|  Entry Quality Score 0-100               |
|  Gate: score >= 68 to execute             |
+------------------------------------------+
         |
         | approved + lot multiplier
         v
+------------------------------------------+
|  RISK CONTROL (per-session)               |
|  Session cap: 10% across max 7 trades     |
|  Per-trade: 1.5% of account               |
+------------------------------------------+
```

---

## 2. M5 Direction System

The highest-leverage decision. A wrong direction reading causes a cascade of losing trades.

### 2.1 Primary Direction Indicators

All five must align. Any disagreement = `DIRECTION_NONE` = no trade.

| Indicator | BULLISH Condition | BEARISH Condition |
|---|---|---|
| **EMA Stack** (21/50/200) | Price above all three, 21 > 50 > 200 (stacked) | Price below all three, 21 < 50 < 200 (inverse stack) |
| **Market Structure** | 2+ consecutive HH/HL on M5 | 2+ consecutive LL/LH on M5 |
| **MACD Histogram** | Positive | Negative |
| **ATR(14) Gate** | ATR(14) > 5 pips (market alive) | ATR(14) > 5 pips (market alive) |
| **VWAP Filter** | Price above session VWAP (London-NY only) | Price below session VWAP (London-NY only) |

### 2.2 Direction Update Cadence

- Direction is **recalculated on every M5 candle close** (not on tick)
- Direction persists between M5 candles — does not flicker
- If direction flips mid-cascade, **all open positions close immediately**

### 2.3 EMA Lag Mitigation

We never trade the crossover itself. We trade in the **confirmed direction AFTER** the EMA stack has aligned, entering on the **first M1 pullback** to the 21 EMA. Typical entry occurs 3-8 candles after the M5 direction locks in.

---

## 3. M1 Cascade Entry Model

The core innovation. Instead of one large position, we build into confirmed moves with 3-4 small positions, each requiring additional confirmation.

### 3.1 Full Entry Condition Checklist

**ALL must be met before Entry 1 (Pilot):**

1. M5 EMA stack confirms BULL or BEAR (updated each M5 candle close)
2. M1 price has pulled back to 21 EMA (retracement entry, not breakout chasing)
3. M1 candle shows clear rejection pattern:
   - Longs: bullish engulfing, hammer, or pin bar
   - Shorts: bearish engulfing, shooting star
4. RSI(7) on M1: 38-55 for longs, 45-62 for shorts (prevents momentum exhaustion entry)
5. Spread < 2.0 pips (gold-specific tight gate)
6. AI Entry Quality Score >= 68/100
7. Session risk remaining > 1.5% (enough budget for this trade)
8. No high-impact news within 30 minutes (News Shield check)
9. Current time within active session window

### 3.2 Cascade Entry Sequence

| Entry | Trigger | Lot Size | Stop Loss |
|---|---|---|---|
| **Entry 1: Pilot** | M1 touches 21 EMA + rejection candle + RSI in range | 0.01 micro lot | Below M1 swing low (10-14 pips typical) |
| **Entry 2: Core** | Next M1 candle closes in trade direction. Entry 1 must be in positive territory. | 0.02 lots | Same SL as Entry 1 |
| **Entry 3: Add** | Momentum candle: body > 70% of total range. Price accelerating. | 0.02 lots | SL moved to Entry 2 level (locks breakeven on Entry 1) |
| **Entry 4: Max** | Only if AI score >= 82 AND London-NY overlap active. Especially clean structure. | 0.01 lots | Trailing SL 10 pips behind price |

### 3.3 Critical Rule: Pilot Governs All

> If Entry 1 (Pilot) hits its stop loss before Entry 2 triggers, the **ENTIRE cascade is CANCELLED immediately**. Zero exceptions. Never add to a losing position.

This is the single most important rule in the scalping model.

---

## 4. Exit Strategy

### 4.1 Take Profit Logic

| Exit Type | Condition | Action |
|---|---|---|
| **Primary TP** | Price reaches ATR(14) x 1.5 from Entry 1 price. Typical: 22-25 pips on gold M1. | Close all positions |
| **Momentum TP** | +12 pips in 4 minutes | Move SL to breakeven, extend TP to ATR x 2.5 (~37+ pips) |
| **Time Exit** | 15 minutes elapsed | Close at market — setup has likely failed |
| **Direction Flip** | M5 EMA stack flips direction mid-trade | Close ALL positions immediately regardless of P&L |
| **VWAP Rejection** | Price stalls at VWAP with 2 consecutive rejection candles | Close 50%, move SL to breakeven |

### 4.2 Stop Loss Logic

| SL Type | Condition | Placement |
|---|---|---|
| **Initial SL** | At entry | 2 pips beyond M1 swing low (longs) or swing high (shorts). Typical 10-16 pips. |
| **Breakeven Move** | Trade at +10 pips | Move SL to entry price |
| **Trailing Stop** | After breakeven set | Trail 12 pips behind price |
| **Hard Stop** | Always active | 20 pips max adverse excursion — close and reassess |

### 4.3 Exit Priority Order

```
1. Hard Stop (20 pips) — absolute safety net, never overridden
2. Direction Flip — immediate full close
3. Time Exit (15 min) — mandatory
4. Initial SL hit
5. VWAP Rejection — partial close
6. Primary TP hit
7. Momentum TP extension
8. Trailing stop trigger
```

---

## 5. MQL5 Class Structure

```
GoldScalper.mq5 (main EA)
|
|-- CDirectionLayer
|   |-- EMA stack reader (21/50/200 on M5)
|   |-- Market structure detector (HH/HL/LH/LL)
|   |-- MACD histogram evaluator
|   |-- ATR gate (14-period, M5)
|   +-- VWAP calculator (session-reset, custom impl)
|
|-- CEntryLayer
|   |-- EMA pullback detector (M1, 21 EMA)
|   |-- Candle pattern recognizer (14 patterns)
|   |-- RSI range validator (7-period, M1)
|   +-- Cascade state machine (tracking entries 1-4)
|
|-- CExitManager
|   |-- ATR-scaled TP calculator
|   |-- Momentum TP detector (+12 pips in 4 min)
|   |-- Breakeven mover
|   |-- Trailing stop manager (12 pips)
|   |-- Time exit timer (15 min)
|   |-- Direction flip detector
|   +-- VWAP rejection detector
|
|-- CRiskManager
|   |-- Dynamic lot size calculator: Lot = (Account * 0.015) / (SL_pips * pip_value)
|   |-- Session risk budget tracker (10% cap)
|   |-- Session halt trigger (7% cumulative loss = halt)
|   |-- Daily cap enforcer (max 2 sessions, session 2 only if session 1 net positive)
|   +-- Winning session reducer (6%+ profit -> next session at 1.0% risk)
|
|-- CAIClient
|   |-- TCP socket connection to localhost:5001
|   |-- JSON request builder (127 features)
|   |-- Response parser
|   |-- Timeout handler (5s max)
|   +-- Fallback mode (if AI unresponsive for 3 consecutive requests)
|
|-- CNewsShield
|   |-- ForexFactory calendar reader
|   |-- 4-phase protocol (Detection -> Pre -> During -> Post)
|   |-- Spread anomaly detector (3x 20-period average)
|   +-- Post-news entry window manager
|
|-- CSpikeDetector
|   |-- ATR(1) vs ATR(20) ratio monitor
|   |-- Trigger: ratio > 3x = halt entries for 10 minutes
|   +-- Cooldown timer
|
+-- CSessionManager
    |-- UTC time-based session identification
    |-- Position size multiplier per session
    +-- Weekend/holiday gate
```

---

## 6. Scalper Input Parameters

```mql5
input double RiskPercent       = 1.5;    // Per-trade risk %
input double SessionRiskCap    = 10.0;   // Max session loss %
input double SessionHaltPct    = 7.0;    // Halt trigger %
input int    AIMinScore        = 68;     // Min AI entry score
input int    AIMaxScore        = 82;     // Score threshold for Entry 4
input double MaxSpreadPips     = 2.0;    // Spread gate (gold-specific)
input int    MaxTradeDuration  = 15;     // Minutes
input double BreakevenTrigger  = 10.0;   // Pips to move SL to BE
input double TrailingDistance   = 12.0;   // Trailing SL distance in pips
input double HardStopPips      = 20.0;   // Max adverse excursion
input double TPMultiplier      = 1.5;    // ATR multiplier for TP
input double MomentumTPMult    = 2.5;    // Extended TP multiplier
input double MomentumTrigger   = 12.0;   // Pips in 4 min for momentum
input int    MinATR_M5         = 5;      // Min ATR(14) on M5 in pips
input int    AIServerPort      = 5001;   // Python AI server port
```

---

## 7. State Machine — Cascade Entry

```
         [IDLE]
            |
            | direction + signal + AI approval
            v
     [PILOT_ENTERED]
       /          \
   SL hit       confirmation candle
      |               |
   [CANCELLED]   [CORE_ENTERED]
                   /          \
              no momentum    momentum candle
                  |               |
           [HOLD_2_POS]    [ADD_ENTERED]
                              /        \
                       score < 82   score >= 82 + overlap
                          |               |
                    [HOLD_3_POS]    [MAX_ENTERED]
                                        |
                                   [HOLD_4_POS]
                                        |
                              ExitManager handles all
                                        |
                                     [CLOSED]
                                        |
                                     [IDLE]
```

Each state transition is guarded by the **Pilot Rule**: if Pilot is negative at any point before the next entry triggers, the entire cascade cancels and all positions close.
