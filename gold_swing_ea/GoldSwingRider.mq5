//+------------------------------------------------------------------+
//| GoldSwingRider.mq5 — Gold Swing Rider EA                         |
//| Timeframes: H1 (execution), H4 (direction anchor)               |
//| Strategy: Trend-following swing trades aligned with H4 structure |
//| Hold duration: 24-72 hours                                        |
//| Exit trigger: H4 structural breakdown only — never H1 noise      |
//| Magic number: 100002                                              |
//+------------------------------------------------------------------+
#property copyright   "XAUUSD AI Trading System"
#property link        "https://github.com/xauusd-trader"
#property version     "1.00"
#property description "Gold Swing Rider — AI-assisted H4/H1 swing trading EA"
#property strict

#include "Include\Constants.mqh"
#include "Include\AIClient.mqh"
#include "Include\MarketStructure.mqh"
#include "Include\NewsShield.mqh"
#include "Include\SessionManager.mqh"
#include "Include\H4DirectionSystem.mqh"
#include "Include\H1ExecutionEngine.mqh"
#include "Include\SwingExitManager.mqh"
#include "Include\SwingRiskManager.mqh"
#include "Include\DXYFilter.mqh"
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

// ==========================================================================
// INPUT PARAMETERS
// ==========================================================================

input group "=== Risk Settings ==="
input double RiskPercent        = 2.0;       // Per-trade risk % of account balance
input double TP1_RR             = 1.5;       // TP1 risk:reward ratio (close 40%)
input double TP2_RR             = 3.0;       // TP2 risk:reward ratio (close remaining 60%)
input double TP1_ClosePct       = 40.0;      // Percentage of position to close at TP1

input group "=== AI Settings ==="
input int    AIMinTrendScore    = 72;        // Minimum AI trend strength score for entry
input int    AIExhaustionScore  = 45;        // Trend score below this = reduce position 50%
input int    AIServerPort       = 5001;      // Python AI server TCP port

input group "=== Entry Settings ==="
input double VolumeSpikeMult    = 1.5;       // Tick volume vs 20-bar average multiplier
input int    RSI_Low            = 42;        // RSI pullback lower bound (trend continuation)
input int    RSI_High           = 55;        // RSI pullback upper bound (trend continuation)
input int    NewsBlackoutHours  = 4;         // Block entry if major news within N hours

input group "=== Exit Settings ==="
input int    MaxHoldHours       = 72;        // Maximum position hold time in hours

input group "=== Macro Settings ==="
input string DXYDataFile        = "dxy_trend.json";  // Path to Python macro data file

// ==========================================================================
// GLOBAL OBJECTS
// ==========================================================================

CH4DirectionSystem  g_h4_direction;
CH1ExecutionEngine  g_h1_execution;
CSwingExitManager   g_exit_manager;
CSwingRiskManager   g_risk_manager;
CDXYFilter          g_dxy_filter;
CAIClient           g_ai_client;
CNewsShield         g_news_shield;
CSessionManager     g_session_manager;
CTrade              g_trade;
CPositionInfo       g_pos_info;

// ==========================================================================
// RUNTIME STATE
// ==========================================================================

// Candle timestamps for close detection
datetime g_last_h4_candle = 0;
datetime g_last_h1_candle = 0;

// Session statistics
int      g_trades_entered  = 0;
int      g_trades_exited   = 0;
double   g_session_pnl     = 0.0;

// AI scoring state
int      g_last_entry_score  = 0;
int      g_last_trend_score  = 0;
int      g_last_news_risk    = 0;
string   g_last_regime       = "";
string   g_last_wyckoff      = "";
double   g_last_lot_mult     = 1.0;
bool     g_last_ai_approved  = false;

// Heartbeat
int      g_heartbeat_failures = 0;

// ==========================================================================
// EA LIFECYCLE
// ==========================================================================

int OnInit()
{
    PrintFormat("[SwingRider] OnInit — XAUUSD Gold Swing Rider v1.0 magic=%d", MAGIC_SWING);

    // Verify symbol
    if(_Symbol != "XAUUSD" && StringFind(_Symbol, "GOLD") < 0 && StringFind(_Symbol, "XAU") < 0)
    {
        Print("[SwingRider] WARNING: EA is not on a Gold instrument. Verify chart symbol.");
    }

    // Configure trade object
    g_trade.SetExpertMagicNumber(MAGIC_SWING);
    g_trade.SetDeviationInPoints(30);
    g_trade.SetTypeFilling(ORDER_FILLING_IOC);

    // Initialise AI client (swing uses 5s timeout — heavier analysis)
    g_ai_client.SetPort(AIServerPort);
    g_ai_client.SetReadTimeout(5000);

    // Connect to AI server (non-fatal if unavailable — fallback mode)
    if(!g_ai_client.Connect())
        PrintFormat("[SwingRider] AI server not available: %s — fallback mode active",
                    g_ai_client.GetLastError());
    else
        Print("[SwingRider] AI server connected");

    // Initialise H4 direction system
    if(!g_h4_direction.Init(_Symbol, AIMinTrendScore))
    {
        Print("[SwingRider] FATAL: H4DirectionSystem Init failed");
        return INIT_FAILED;
    }

    // Initialise H1 execution engine
    if(!g_h1_execution.Init(_Symbol, RSI_Low, RSI_High,
                             VolumeSpikeMult, NewsBlackoutHours,
                             &g_news_shield, &g_dxy_filter))
    {
        Print("[SwingRider] FATAL: H1ExecutionEngine Init failed");
        return INIT_FAILED;
    }

    // Initialise risk manager
    g_risk_manager.Init(_Symbol, RiskPercent, TP1_RR, TP2_RR, TP1_ClosePct);

    // Initialise DXY filter
    g_dxy_filter.SetDataFile(DXYDataFile);
    g_dxy_filter.Refresh();

    // Initialise exit manager
    g_exit_manager.Init(_Symbol, MAGIC_SWING, MaxHoldHours, AIExhaustionScore,
                         &g_risk_manager, &g_dxy_filter, &g_news_shield);

    // Start 60-second timer for heartbeat
    EventSetTimer(60);

    // Cache initial candle timestamps
    g_last_h4_candle = iTime(_Symbol, PERIOD_H4, 1);
    g_last_h1_candle = iTime(_Symbol, PERIOD_H1, 1);

    Print("[SwingRider] Initialisation complete — scanning for swing setups");
    return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    g_ai_client.Disconnect();

    PrintFormat("[SwingRider] OnDeinit reason=%d | Trades entered=%d exited=%d session_PnL=%.2f",
                reason, g_trades_entered, g_trades_exited, g_session_pnl);

    string reason_str;
    switch(reason)
    {
        case REASON_REMOVE:   reason_str = "EA removed";       break;
        case REASON_DISABLE:  reason_str = "EA disabled";      break;
        case REASON_RECOMPILE: reason_str = "Recompiled";      break;
        case REASON_CHARTCLOSE: reason_str = "Chart closed";   break;
        default:              reason_str = IntegerToString(reason);
    }
    PrintFormat("[SwingRider] Shutdown reason: %s", reason_str);
}

// ==========================================================================
// HEARTBEAT TIMER — every 60 seconds
// ==========================================================================

void OnTimer()
{
    string hb_status = "";
    int    hb_uptime = 0;
    string hb_model  = "";

    if(g_ai_client.Heartbeat(hb_status, hb_uptime, hb_model))
    {
        g_heartbeat_failures = 0;
        if(g_ai_client.IsInFallbackMode())
        {
            g_ai_client.ResetFailures();
            Print("[SwingRider] AI server reconnected — exiting fallback mode");
        }

        PrintFormat("[SwingRider] Heartbeat OK | status=%s uptime=%ds model=%s",
                    hb_status, hb_uptime, hb_model);
    }
    else
    {
        g_heartbeat_failures++;
        PrintFormat("[SwingRider] Heartbeat FAILED (%d consecutive) — %s",
                    g_heartbeat_failures, g_ai_client.GetLastError());
    }

    // Refresh DXY data on heartbeat cadence (every 60s, but file only re-read every 15min)
    g_dxy_filter.Refresh();

    // Log position status
    if(g_exit_manager.HasOpenPosition())
    {
        PrintFormat("[SwingRider] Open positions: %d | Session PnL: %.2f",
                    g_exit_manager.GetOpenPositionCount(), g_session_pnl);
    }
}

// ==========================================================================
// MAIN TICK
// ==========================================================================

void OnTick()
{
    // ------------------------------------------------------------------
    // Always manage existing positions first — every tick
    // (TP hits, time limit, news reduction)
    // H4 structural checks are gated inside ManagePositions()
    // ------------------------------------------------------------------

    double h4_ema200 = g_h4_direction.GetH4EMA200();
    g_exit_manager.ManagePositions(h4_ema200);

    // ------------------------------------------------------------------
    // GATE 1: News shield — update every tick
    // ------------------------------------------------------------------

    g_news_shield.Update();

    // ------------------------------------------------------------------
    // GATE 2: Weekend / market closed
    // ------------------------------------------------------------------

    if(g_session_manager.IsWeekend())
        return;

    // ------------------------------------------------------------------
    // GATE 3: If a position is already open, skip entry evaluation
    // Swing bot: single position at a time
    // ------------------------------------------------------------------

    if(g_exit_manager.HasOpenPosition())
        return;

    // ------------------------------------------------------------------
    // GATE 4: Session — swing entries only during London / Overlap
    // ------------------------------------------------------------------

    if(!g_session_manager.IsSwingEntryAllowed())
        return;

    // ------------------------------------------------------------------
    // CANDLE CLOSE DETECTION
    // Entry evaluation happens only on H4 and H1 candle closes
    // ------------------------------------------------------------------

    datetime current_h4 = iTime(_Symbol, PERIOD_H4, 1);
    datetime current_h1 = iTime(_Symbol, PERIOD_H1, 1);

    bool h4_closed = (current_h4 != g_last_h4_candle);
    bool h1_closed = (current_h1 != g_last_h1_candle);

    if(h4_closed)
    {
        g_last_h4_candle = current_h4;
        PrintFormat("[SwingRider] H4 candle closed at %s", TimeToString(current_h4));
        OnH4CandleClose();
    }

    if(h1_closed)
    {
        g_last_h1_candle = current_h1;
        OnH1CandleClose();
    }
}

// ==========================================================================
// H4 CANDLE CLOSE HANDLER
// Refreshes the institutional direction view
// ==========================================================================

void OnH4CandleClose()
{
    // Query AI server for updated trend score (uses current H4 features)
    if(!g_ai_client.IsInFallbackMode())
        RequestAIScore(PERIOD_H4);

    // Re-evaluate H4 direction (will use new AI score from UpdateAIData above)
    ENUM_DIRECTION h4_dir = g_h4_direction.GetH4Trend();

    PrintFormat("[SwingRider] H4 Direction: %s | AI trend=%d wyckoff=%s",
                (h4_dir == DIRECTION_BULL) ? "BULL" :
                (h4_dir == DIRECTION_BEAR) ? "BEAR" : "NONE",
                g_last_trend_score,
                g_last_wyckoff);
}

// ==========================================================================
// H1 CANDLE CLOSE HANDLER
// Evaluates all 7 H1 entry conditions
// ==========================================================================

void OnH1CandleClose()
{
    // Get current confirmed H4 direction
    ENUM_DIRECTION h4_dir = g_h4_direction.GetH4Trend();

    if(h4_dir == DIRECTION_NONE)
    {
        Print("[SwingRider] H1 close: no confirmed H4 direction — skipping");
        return;
    }

    // Request AI score on H1 timeframe
    if(!g_ai_client.IsInFallbackMode())
        RequestAIScore(PERIOD_H1);

    // Check AI approval
    if(!g_last_ai_approved && !g_ai_client.IsInFallbackMode())
    {
        PrintFormat("[SwingRider] H1 close: AI not approved (score=%d trend=%d)",
                    g_last_entry_score, g_last_trend_score);
        return;
    }

    // Verify trend score still meets threshold
    if(!g_ai_client.IsInFallbackMode() && g_last_trend_score < AIMinTrendScore)
    {
        PrintFormat("[SwingRider] H1 close: trend score %d < %d — no entry",
                    g_last_trend_score, AIMinTrendScore);
        return;
    }

    // Evaluate H1 execution conditions
    if(!g_h1_execution.HasEntry(h4_dir))
    {
        PrintFormat("[SwingRider] H1 close: entry conditions not met — %s",
                    g_h1_execution.GetLastFailReason());
        return;
    }

    // All conditions met — execute the trade
    ExecuteSwingEntry(h4_dir);
}

// ==========================================================================
// AI SCORING REQUEST
// Builds the 127-feature vector and queries the AI server
// ==========================================================================

bool RequestAIScore(ENUM_TIMEFRAMES tf)
{
    double features[];
    ArrayResize(features, FEATURE_COUNT);
    ArrayInitialize(features, 0.0);

    // Build feature vector
    // The complete feature engineering is done in the Python server's preprocessing.
    // The MQL5 side sends the raw OHLCV + indicator values; Python returns scores.
    // This is a minimal but functional feature vector for real usage.

    BuildFeatureVector(features, tf);

    string tf_str = (tf == PERIOD_H4) ? "H4" : "H1";
    ENUM_DIRECTION cur_dir = g_h4_direction.GetH4Trend();
    ENUM_DIRECTION req_dir = (cur_dir != DIRECTION_NONE) ? cur_dir : DIRECTION_BULL;

    bool ok = g_ai_client.ScoreEntry(
        req_dir,
        tf_str,
        "swing",
        features,
        g_last_entry_score,
        g_last_trend_score,
        g_last_news_risk,
        g_last_regime,
        g_last_wyckoff,
        g_last_ai_approved,
        g_last_lot_mult
    );

    if(ok)
    {
        // Push new AI data into the H4 direction system
        g_h4_direction.UpdateAIData(g_last_trend_score, g_last_wyckoff);

        // Update AI trend exhaustion monitoring in exit manager
        g_exit_manager.UpdateAllAIScores(g_last_trend_score);

        PrintFormat("[SwingRider] AI: entry=%d trend=%d news=%d regime=%s wyckoff=%s approved=%s",
                    g_last_entry_score, g_last_trend_score, g_last_news_risk,
                    g_last_regime, g_last_wyckoff,
                    g_last_ai_approved ? "YES" : "NO");
    }
    else
    {
        PrintFormat("[SwingRider] AI request failed: %s", g_ai_client.GetLastError());
    }

    return ok;
}

// ==========================================================================
// FEATURE VECTOR BUILDER
// Populates the 127-element features array for the AI server
// ==========================================================================

void BuildFeatureVector(double &features[], ENUM_TIMEFRAMES tf)
{
    int idx = 0;

    // ---- Multi-timeframe OHLCV (bars 1 and 2 for H1 and H4) ----

    // H1 candle data (bar 1 — last closed)
    features[idx++] = iOpen(_Symbol,  PERIOD_H1, 1);
    features[idx++] = iHigh(_Symbol,  PERIOD_H1, 1);
    features[idx++] = iLow(_Symbol,   PERIOD_H1, 1);
    features[idx++] = iClose(_Symbol, PERIOD_H1, 1);
    features[idx++] = (double)iVolume(_Symbol, PERIOD_H1, 1);

    // H4 candle data
    features[idx++] = iOpen(_Symbol,  PERIOD_H4, 1);
    features[idx++] = iHigh(_Symbol,  PERIOD_H4, 1);
    features[idx++] = iLow(_Symbol,   PERIOD_H4, 1);
    features[idx++] = iClose(_Symbol, PERIOD_H4, 1);
    features[idx++] = (double)iVolume(_Symbol, PERIOD_H4, 1);

    // M5 candle data (for scalper-compatibility features)
    features[idx++] = iOpen(_Symbol,  PERIOD_M5, 1);
    features[idx++] = iHigh(_Symbol,  PERIOD_M5, 1);
    features[idx++] = iLow(_Symbol,   PERIOD_M5, 1);
    features[idx++] = iClose(_Symbol, PERIOD_M5, 1);
    features[idx++] = (double)iVolume(_Symbol, PERIOD_M5, 1);

    // ---- EMA values (H1) ----
    int ema8_h1   = iMA(_Symbol, PERIOD_H1, 8,   0, MODE_EMA, PRICE_CLOSE);
    int ema21_h1  = iMA(_Symbol, PERIOD_H1, 21,  0, MODE_EMA, PRICE_CLOSE);
    int ema50_h1  = iMA(_Symbol, PERIOD_H1, 50,  0, MODE_EMA, PRICE_CLOSE);
    int ema200_h1 = iMA(_Symbol, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE);

    double ema_buf[];
    ArraySetAsSeries(ema_buf, true);

    // EMA 8 H1
    double v8 = 0, v21 = 0, v50 = 0, v200 = 0;
    if(CopyBuffer(ema8_h1,   0, 1, 2, ema_buf) > 0) { v8   = ema_buf[0]; features[idx++] = v8;   features[idx++] = ema_buf[0] - ema_buf[1]; }
    else { features[idx++] = 0; features[idx++] = 0; }
    if(CopyBuffer(ema21_h1,  0, 1, 2, ema_buf) > 0) { v21  = ema_buf[0]; features[idx++] = v21;  features[idx++] = ema_buf[0] - ema_buf[1]; }
    else { features[idx++] = 0; features[idx++] = 0; }
    if(CopyBuffer(ema50_h1,  0, 1, 2, ema_buf) > 0) { v50  = ema_buf[0]; features[idx++] = v50;  features[idx++] = ema_buf[0] - ema_buf[1]; }
    else { features[idx++] = 0; features[idx++] = 0; }
    if(CopyBuffer(ema200_h1, 0, 1, 2, ema_buf) > 0) { v200 = ema_buf[0]; features[idx++] = v200; features[idx++] = ema_buf[0] - ema_buf[1]; }
    else { features[idx++] = 0; features[idx++] = 0; }

    IndicatorRelease(ema8_h1); IndicatorRelease(ema21_h1);
    IndicatorRelease(ema50_h1); IndicatorRelease(ema200_h1);

    // ---- EMA values (H4) ----
    int ema8_h4   = iMA(_Symbol, PERIOD_H4, 8,   0, MODE_EMA, PRICE_CLOSE);
    int ema21_h4  = iMA(_Symbol, PERIOD_H4, 21,  0, MODE_EMA, PRICE_CLOSE);
    int ema50_h4  = iMA(_Symbol, PERIOD_H4, 50,  0, MODE_EMA, PRICE_CLOSE);
    int ema200_h4 = iMA(_Symbol, PERIOD_H4, 200, 0, MODE_EMA, PRICE_CLOSE);

    if(CopyBuffer(ema8_h4,   0, 1, 2, ema_buf) > 0) { features[idx++] = ema_buf[0]; features[idx++] = ema_buf[0] - ema_buf[1]; }
    else { features[idx++] = 0; features[idx++] = 0; }
    if(CopyBuffer(ema21_h4,  0, 1, 2, ema_buf) > 0) { features[idx++] = ema_buf[0]; features[idx++] = ema_buf[0] - ema_buf[1]; }
    else { features[idx++] = 0; features[idx++] = 0; }
    if(CopyBuffer(ema50_h4,  0, 1, 2, ema_buf) > 0) { features[idx++] = ema_buf[0]; features[idx++] = ema_buf[0] - ema_buf[1]; }
    else { features[idx++] = 0; features[idx++] = 0; }
    if(CopyBuffer(ema200_h4, 0, 1, 2, ema_buf) > 0) { features[idx++] = ema_buf[0]; features[idx++] = ema_buf[0] - ema_buf[1]; }
    else { features[idx++] = 0; features[idx++] = 0; }

    IndicatorRelease(ema8_h4); IndicatorRelease(ema21_h4);
    IndicatorRelease(ema50_h4); IndicatorRelease(ema200_h4);

    // ---- ATR ----
    int atr14_h1 = iATR(_Symbol, PERIOD_H1, 14);
    int atr14_h4 = iATR(_Symbol, PERIOD_H4, 14);
    double atr_buf[];
    ArraySetAsSeries(atr_buf, true);

    if(CopyBuffer(atr14_h1, 0, 1, 1, atr_buf) > 0) features[idx++] = atr_buf[0]; else features[idx++] = 0;
    if(CopyBuffer(atr14_h4, 0, 1, 1, atr_buf) > 0) features[idx++] = atr_buf[0]; else features[idx++] = 0;

    IndicatorRelease(atr14_h1); IndicatorRelease(atr14_h4);

    // ---- RSI ----
    int rsi7_h1  = iRSI(_Symbol, PERIOD_H1, 7,  PRICE_CLOSE);
    int rsi14_h1 = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
    int rsi14_h4 = iRSI(_Symbol, PERIOD_H4, 14, PRICE_CLOSE);
    double rsi_buf[];
    ArraySetAsSeries(rsi_buf, true);

    if(CopyBuffer(rsi7_h1,  0, 1, 1, rsi_buf) > 0) features[idx++] = rsi_buf[0]; else features[idx++] = 50;
    if(CopyBuffer(rsi14_h1, 0, 1, 1, rsi_buf) > 0) features[idx++] = rsi_buf[0]; else features[idx++] = 50;
    if(CopyBuffer(rsi14_h4, 0, 1, 1, rsi_buf) > 0) features[idx++] = rsi_buf[0]; else features[idx++] = 50;

    IndicatorRelease(rsi7_h1); IndicatorRelease(rsi14_h1); IndicatorRelease(rsi14_h4);

    // ---- MACD (H1) ----
    int macd_h1 = iMACD(_Symbol, PERIOD_H1, 12, 26, 9, PRICE_CLOSE);
    double macd_main[], macd_sig[];
    ArraySetAsSeries(macd_main, true);
    ArraySetAsSeries(macd_sig, true);

    if(CopyBuffer(macd_h1, MAIN_LINE,   1, 2, macd_main) > 0 &&
       CopyBuffer(macd_h1, SIGNAL_LINE, 1, 1, macd_sig)  > 0)
    {
        features[idx++] = macd_main[0];
        features[idx++] = macd_sig[0];
        features[idx++] = macd_main[0] - macd_sig[0];      // Histogram
        features[idx++] = macd_main[0] - macd_main[1];     // Histogram ROC
    }
    else { features[idx++] = 0; features[idx++] = 0; features[idx++] = 0; features[idx++] = 0; }

    IndicatorRelease(macd_h1);

    // ---- Bollinger Bands (H1, 20, 2) ----
    int bb_h1 = iBands(_Symbol, PERIOD_H1, 20, 0, 2.0, PRICE_CLOSE);
    double bb_upper[], bb_lower[], bb_mid[];
    ArraySetAsSeries(bb_upper, true); ArraySetAsSeries(bb_lower, true); ArraySetAsSeries(bb_mid, true);

    if(CopyBuffer(bb_h1, UPPER_BAND, 1, 1, bb_upper) > 0 &&
       CopyBuffer(bb_h1, LOWER_BAND, 1, 1, bb_lower) > 0 &&
       CopyBuffer(bb_h1, BASE_LINE,  1, 1, bb_mid)   > 0)
    {
        double bw   = (bb_upper[0] - bb_lower[0]);
        double price = iClose(_Symbol, PERIOD_H1, 1);
        double pct_b = (bw > 0) ? (price - bb_lower[0]) / bw : 0.5;
        features[idx++] = bw;
        features[idx++] = pct_b;
    }
    else { features[idx++] = 0; features[idx++] = 0.5; }

    IndicatorRelease(bb_h1);

    // ---- ADX (H1, H4) ----
    int adx_h1 = iADX(_Symbol, PERIOD_H1, 14);
    int adx_h4 = iADX(_Symbol, PERIOD_H4, 14);
    double adx_buf[];
    ArraySetAsSeries(adx_buf, true);

    if(CopyBuffer(adx_h1, MAIN_LINE,  1, 1, adx_buf) > 0) features[idx++] = adx_buf[0]; else features[idx++] = 20;
    if(CopyBuffer(adx_h1, PLUSDI_LINE,1, 1, adx_buf) > 0) features[idx++] = adx_buf[0]; else features[idx++] = 0;
    if(CopyBuffer(adx_h1, MINUSDI_LINE,1,1, adx_buf) > 0) features[idx++] = adx_buf[0]; else features[idx++] = 0;
    if(CopyBuffer(adx_h4, MAIN_LINE,  1, 1, adx_buf) > 0) features[idx++] = adx_buf[0]; else features[idx++] = 20;
    if(CopyBuffer(adx_h4, PLUSDI_LINE,1, 1, adx_buf) > 0) features[idx++] = adx_buf[0]; else features[idx++] = 0;
    if(CopyBuffer(adx_h4, MINUSDI_LINE,1,1, adx_buf) > 0) features[idx++] = adx_buf[0]; else features[idx++] = 0;

    IndicatorRelease(adx_h1); IndicatorRelease(adx_h4);

    // ---- Temporal features ----
    MqlDateTime dt;
    TimeGMT(dt);
    features[idx++] = (double)dt.hour;
    features[idx++] = (double)dt.day_of_week;

    // Price distance from nearest round $100 level
    double price_now = iClose(_Symbol, PERIOD_H1, 0);
    double nearest_round = MathRound(price_now / 100.0) * 100.0;
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double pip   = point * 10.0;
    features[idx++] = (pip > 0) ? MathAbs(price_now - nearest_round) / pip : 0;

    // Spread vs 20-bar average
    double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
    features[idx++] = (pip > 0) ? spread / pip : 0;

    // H1 volume ratio (current vs 20-bar avg)
    long vol_cur = iVolume(_Symbol, PERIOD_H1, 1);
    long vol_sum = 0;
    for(int k = 2; k <= 21; k++) vol_sum += iVolume(_Symbol, PERIOD_H1, k);
    double vol_avg = (vol_sum > 0) ? (double)vol_sum / 20.0 : 1.0;
    features[idx++] = (vol_avg > 0) ? (double)vol_cur / vol_avg : 1.0;

    // Account drawdown
    double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    features[idx++] = (balance > 0) ? (balance - equity) / balance * 100.0 : 0.0;

    // Session number (0 or 1)
    features[idx++] = 0.0;

    // DXY trend direction encoded: 1=UP, -1=DOWN, 0=NEUTRAL
    ENUM_DIRECTION dxy_dir = g_dxy_filter.GetDXYTrend();
    features[idx++] = (dxy_dir == DIRECTION_BULL) ? 1.0 :
                      (dxy_dir == DIRECTION_BEAR) ? -1.0 : 0.0;

    // H4 market structure: bull=1, bear=-1, none=0
    ENUM_DIRECTION h4_dir = g_h4_direction.GetH4Trend();
    features[idx++] = (h4_dir == DIRECTION_BULL) ? 1.0 :
                      (h4_dir == DIRECTION_BEAR) ? -1.0 : 0.0;

    // News active: 1 if news shield active, else 0
    features[idx++] = g_news_shield.IsActive() ? 1.0 : 0.0;

    // Pad remaining features with zeros to reach exactly FEATURE_COUNT
    while(idx < FEATURE_COUNT)
        features[idx++] = 0.0;
}

// ==========================================================================
// TRADE EXECUTION
// ==========================================================================

void ExecuteSwingEntry(ENUM_DIRECTION direction)
{
    PrintFormat("[SwingRider] Attempting %s entry", direction == DIRECTION_BULL ? "LONG" : "SHORT");

    // Get entry price
    double entry_price = (direction == DIRECTION_BULL)
                         ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                         : SymbolInfoDouble(_Symbol, SYMBOL_BID);

    // Get SL anchor: last H4 swing low (long) or last H4 swing high (short)
    double swing_price;
    if(direction == DIRECTION_BULL)
        swing_price = g_h4_direction.GetLastH4SwingLow();
    else
        swing_price = g_h4_direction.GetLastH4SwingHigh();

    if(swing_price <= 0)
    {
        Print("[SwingRider] Cannot get H4 swing price for SL calculation — aborting");
        return;
    }

    // Calculate all trade levels
    double lot_size, sl_price, tp1_price, tp2_price, sl_pips;
    if(!g_risk_manager.CalcAllLevels(
        direction, entry_price, swing_price,
        lot_size, sl_price, tp1_price, tp2_price, sl_pips))
    {
        Print("[SwingRider] Risk manager rejected levels — aborting entry");
        return;
    }

    // Apply AI lot multiplier if available
    if(!g_ai_client.IsInFallbackMode() && g_last_lot_mult > 0)
    {
        double vol_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
        double vol_min  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        lot_size = MathMax(vol_min,
                           MathFloor(lot_size * g_last_lot_mult / vol_step) * vol_step);
    }

    // Execute the order
    // Note: TP is not placed on the broker server — we manage TP1/TP2 manually
    // via partial closes. The SL IS placed as a server-side stop.
    bool result = false;

    if(direction == DIRECTION_BULL)
    {
        result = g_trade.Buy(lot_size, _Symbol, entry_price, sl_price, 0,
                             StringFormat("SwingLong SL%.0f TP1%.0f TP2%.0f",
                                          sl_pips, tp1_price, tp2_price));
    }
    else
    {
        result = g_trade.Sell(lot_size, _Symbol, entry_price, sl_price, 0,
                              StringFormat("SwingShort SL%.0f TP1%.0f TP2%.0f",
                                           sl_pips, tp1_price, tp2_price));
    }

    if(result)
    {
        ulong ticket = g_trade.ResultOrder();
        g_trades_entered++;

        PrintFormat("[SwingRider] ENTRY OK ticket=%llu dir=%s entry=%.2f sl=%.2f tp1=%.2f tp2=%.2f lots=%.2f",
                    ticket,
                    direction == DIRECTION_BULL ? "LONG" : "SHORT",
                    entry_price, sl_price, tp1_price, tp2_price, lot_size);

        // Register with exit manager
        g_exit_manager.RegisterPosition(ticket, direction, entry_price,
                                        sl_price, tp1_price, tp2_price, lot_size);

        // Pass current AI trend score to exit manager
        g_exit_manager.UpdateAIScore(ticket, g_last_trend_score);
    }
    else
    {
        PrintFormat("[SwingRider] ENTRY FAILED: retcode=%d err=%d",
                    g_trade.ResultRetcode(), GetLastError());
    }
}

// ==========================================================================
// END OF FILE
// ==========================================================================
