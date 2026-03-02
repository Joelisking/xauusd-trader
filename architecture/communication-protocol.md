# Communication Protocol — MT5 ↔ Python AI Server

> Transport: TCP socket on localhost:5001
> Format: JSON (UTF-8)
> Latency target: < 150ms P95 (scalper), < 300ms P95 (swing)

---

## 1. Architecture

```
+---------------------------+          TCP :5001          +---------------------------+
|  MT5 Expert Advisor       | --------------------------> |  Python AI Server         |
|  (MQL5)                   |                             |  (asyncio + uvicorn)      |
|                           | <-------------------------- |                           |
|  CAIClient class          |          JSON response      |  score_entry()            |
|  - Builds feature vector  |                             |  - BiLSTM + XGBoost       |
|  - Sends JSON request     |                             |  - Regime classifier      |
|  - Parses response        |                             |  - News risk scorer       |
|  - Handles timeouts       |                             |  - Wyckoff phase          |
|  - Manages fallback mode  |                             |  - Health endpoint        |
+---------------------------+                             +---------------------------+
```

---

## 2. Message Types

### 2.1 Entry Check Request (MT5 -> Python)

Sent when the EA has a candidate entry that passed all rule-based filters.

```json
{
  "type": "entry_check",
  "symbol": "XAUUSD",
  "direction": "BUY",
  "timeframe": "M1",
  "bot": "scalper",
  "session_hour": 14,
  "dxy_trend": "DOWN",
  "real_yield_trend": "DOWN",
  "vix_level": 18.5,
  "current_spread": 0.8,
  "atr_14": 15.2,
  "session_risk_used": 3.5,
  "account_drawdown": 2.1,
  "features": [... 127 float values ...]
}
```

| Field | Type | Description |
|---|---|---|
| type | string | `"entry_check"` — primary scoring request |
| symbol | string | Always `"XAUUSD"` for V1 |
| direction | string | `"BUY"` or `"SELL"` (determined by rule-based direction layer) |
| timeframe | string | `"M1"` (scalper) or `"H1"` (swing) |
| bot | string | `"scalper"` or `"swing"` — determines which model ensemble to use |
| session_hour | int | Current UTC hour (0-23) |
| dxy_trend | string | `"UP"`, `"DOWN"`, or `"NEUTRAL"` |
| real_yield_trend | string | `"UP"`, `"DOWN"`, or `"NEUTRAL"` |
| vix_level | float | Current VIX index value |
| current_spread | float | Current XAUUSD spread in pips |
| atr_14 | float | ATR(14) on the entry timeframe |
| session_risk_used | float | % of session risk budget already consumed |
| account_drawdown | float | Current drawdown from account peak in % |
| features | float[] | 127 pre-calculated feature values |

### 2.2 Entry Check Response (Python -> MT5)

```json
{
  "entry_score": 78,
  "trend_score": 82,
  "news_risk": 12,
  "wyckoff_phase": "D",
  "regime": "trending",
  "approve": true,
  "recommended_lot_multiplier": 1.0,
  "model_version": "2026-03-01",
  "latency_ms": 45
}
```

| Field | Type | Description |
|---|---|---|
| entry_score | int | 0-100. Scalper gate: >= 68. Swing gate: >= 72. |
| trend_score | int | 0-100. Swing bot direction filter. |
| news_risk | int | 0-100. > 75: halt all. 50-75: halve sizes. |
| wyckoff_phase | string | `"A"`, `"B"`, `"C"`, `"D"`, `"E"`. Phase C/D = highest priority. |
| regime | string | `"trending"`, `"ranging"`, `"crisis"` |
| approve | bool | Server-side final check (score + news risk combined) |
| recommended_lot_multiplier | float | 0.8 for normal, 1.0 for high confidence, 1.2 for very high |
| model_version | string | Date of current production model |
| latency_ms | int | Server-side processing time |

### 2.3 Heartbeat Request (MT5 -> Python)

Sent every 60 seconds to verify server health.

```json
{
  "type": "heartbeat"
}
```

### 2.4 Heartbeat Response (Python -> MT5)

```json
{
  "status": "healthy",
  "uptime_seconds": 86400,
  "model_version": "2026-03-01",
  "predictions_today": 142,
  "avg_latency_ms": 38,
  "queue_depth": 0
}
```

| Field | Type | Description |
|---|---|---|
| status | string | `"healthy"`, `"degraded"`, `"error"` |
| uptime_seconds | int | Server uptime since last restart |
| model_version | string | Currently loaded model date |
| predictions_today | int | Scoring requests handled today |
| avg_latency_ms | int | Average response time today |
| queue_depth | int | Pending requests in queue |

---

## 3. Connection Management

### 3.1 Connection Lifecycle

```
EA starts
    |
    v
Connect to localhost:5001
    |
    +-- Success: normal operation
    |
    +-- Failure: retry 3 times with 2s backoff
            |
            +-- Still failing: enter FALLBACK MODE
            +-- Send Telegram: "AI SERVER UNREACHABLE"
```

### 3.2 Timeout Handling

| Timeout | Value | Action |
|---|---|---|
| Connection timeout | 5 seconds | Retry once, then increment failure counter |
| Read timeout (scalper) | 3 seconds | Retry once, then skip this entry |
| Read timeout (swing) | 5 seconds | Retry once, then skip this entry |
| Heartbeat timeout | 5 seconds | Send Telegram warning |

### 3.3 Fallback Mode Transitions

```
NORMAL MODE
    |
    | 1 failed request
    v
WARNING (retry after 5s)
    |
    | 2 consecutive failures
    v
ALERT (Telegram: "AI SERVER WARNING")
    |
    | 3 consecutive failures
    v
FALLBACK MODE
    - No AI scores used
    - Only 100% rule-based entries (stricter criteria)
    - Telegram alert every 5 minutes
    |
    | successful heartbeat response
    v
RECOVERY (1 successful scoring request required)
    |
    | confirmed working
    v
NORMAL MODE
```

---

## 4. Latency Requirements

### 4.1 End-to-End Budget (Scalper)

```
Feature extraction (MQL5):    ~10ms
TCP send:                      ~1ms
Server processing:            ~80ms  (BiLSTM + XGBoost + ensemble)
TCP receive:                   ~1ms
Response parsing (MQL5):       ~5ms
                              -------
Total:                        ~97ms  (target < 150ms P95)
```

### 4.2 Stress Test Requirements

Before deployment:
- Send **500 concurrent requests** to AI server
- Verify **P95 response time < 150ms**
- If above target: optimize with TensorFlow Serving or TorchServe
- Server must handle **2000+ requests/second** without degradation

---

## 5. Data Integrity

### 5.1 Feature Vector Validation

The Python server validates every incoming request:

| Check | Rule | On Failure |
|---|---|---|
| Feature count | Exactly 127 values | Return `{"approve": false, "error": "invalid_features"}` |
| NaN/Inf check | No NaN or infinity values | Return `{"approve": false, "error": "nan_detected"}` |
| Range check | Each feature within expected bounds | Log warning, proceed with clipped values |
| Symbol check | Must be "XAUUSD" | Return `{"approve": false, "error": "invalid_symbol"}` |

### 5.2 Prediction Logging

Every request/response pair is logged for model performance tracking:

```
predictions/
  YYYY-MM-DD.jsonl

Each line:
{
  "timestamp": "2026-03-02T14:23:45Z",
  "request_hash": "abc123",
  "bot": "scalper",
  "direction": "BUY",
  "entry_score": 78,
  "trend_score": 82,
  "regime": "trending",
  "approved": true,
  "latency_ms": 45,
  "actual_outcome": null  // filled later by outcome tracker
}
```

The `actual_outcome` field is populated asynchronously when the trade closes, enabling rolling accuracy computation.

---

## 6. Security Considerations

- Server listens on **localhost only** (127.0.0.1:5001) — no external access
- No authentication required (same-machine communication)
- If deployed across machines (future): add TLS + API key authentication
- Log all connections and disconnections
- Rate limit: max 100 requests/second per client (prevents accidental flood)
