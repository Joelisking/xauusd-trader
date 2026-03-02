//+------------------------------------------------------------------+
//| GoldScalper.mq5 — Gold Scalper EA                                |
//|                                                                   |
//| Timeframes: M1 (execution), M5 (direction)                       |
//| Magic number: 100001                                              |
//|                                                                   |
//| Architecture:                                                     |
//|   OnTick gate sequence:                                           |
//|     1. NewsShield  — block during news events                     |
//|     2. SpikeDetector — block during flash crash / spike           |
//|     3. RiskManager  — session halt / cap / daily limit            |
//|     4. SessionManager — trading hours gate                        |
//|     5. Spread gate — max 2.0 pips                                 |
//|     6. DirectionLayer — M5 consensus (5 indicators)               |
//|     7. EntryLayer — M1 signal (pullback + rejection + RSI)        |
//|     8. AI Score gate — min score 68                               |
//|     9. Execute cascade (Pilot entry via EntryLayer)               |
//|   Always: ExitManager.ManageOpenTrades()                          |
//|   Always: EntryLayer.ManageCascade() (state machine transitions)  |
//+------------------------------------------------------------------+

#property copyright "XAUUSD AI Trading System"
#property version   "1.00"
#property strict

// ------------------------------------------------------------------
// Includes
// ------------------------------------------------------------------
#include "Include\Constants.mqh"
#include "Include\AIClient.mqh"
#include "Include\MarketStructure.mqh"
#include "Include\CandlePatterns.mqh"
#include "Include\NewsShield.mqh"
#include "Include\SpikeDetector.mqh"
#include "Include\SessionManager.mqh"
#include "Include\VWAPCalculator.mqh"
#include "Include\DirectionLayer.mqh"
#include "Include\EntryLayer.mqh"
#include "Include\ExitManager.mqh"
#include "Include\RiskManager.mqh"

// ------------------------------------------------------------------
// Input Parameters
// ------------------------------------------------------------------

// Risk parameters
input group              "=== Risk Management ==="
input double RiskPercent       = 1.5;    // Per-trade risk percentage
input double SessionRiskCap    = 10.0;   // Max cumulative session loss %
input double SessionHaltPct    = 7.0;    // Non-overridable halt threshold %

// AI filter
input group              "=== AI Filter ==="
input int    AIMinScore        = 68;     // Minimum AI entry score (0-100)
input int    AIMaxScore        = 82;     // Score required for Entry 4 (Max)
input int    AIServerPort      = 5001;   // Python AI server TCP port

// Trade execution
input group              "=== Trade Execution ==="
input double MaxSpreadPips     = 2.0;    // Maximum allowed spread in pips
input int    MaxTradeDuration  = 15;     // Maximum trade duration in minutes

// Exit management
input group              "=== Exit Management ==="
input double BreakevenTrigger  = 10.0;   // Pips profit to move SL to breakeven
input double TrailingDistance  = 12.0;   // Trailing stop distance in pips
input double HardStopPips      = 20.0;   // Hard stop maximum adverse excursion
input double TPMultiplier      = 1.5;    // ATR multiplier for primary TP
input double MomentumTPMult    = 2.5;    // ATR multiplier for extended momentum TP
input double MomentumTrigger   = 12.0;   // Pips in 4 minutes to activate momentum TP

// Direction layer
input group              "=== Direction Layer ==="
input int    MinATR_M5         = 5;      // Minimum ATR(14) on M5 in pips

// ------------------------------------------------------------------
// Global object pointers
// ------------------------------------------------------------------
CAIClient        *g_ai_client       = NULL;
CMarketStructure *g_market_struct   = NULL;
CCandlePatterns  *g_patterns        = NULL;
CNewsShield      *g_news_shield     = NULL;
CSpikeDetector   *g_spike_detector  = NULL;
CSessionManager  *g_session         = NULL;
CVWAPCalculator  *g_vwap            = NULL;
CDirectionLayer  *g_direction       = NULL;
CEntryLayer      *g_entry           = NULL;
CExitManager     *g_exit_manager    = NULL;
CRiskManager     *g_risk_manager    = NULL;

// ------------------------------------------------------------------
// State
// ------------------------------------------------------------------
bool     g_init_ok             = false;
int      g_heartbeat_counter   = 0;
datetime g_last_heartbeat_time = 0;

// For tracking closed trades to update RiskManager
double   g_prev_balance        = 0.0;
ulong    g_last_deal_ticket    = 0;

// Feature vector (127 elements) for AI scoring
double   g_features[FEATURE_COUNT];

// ------------------------------------------------------------------
// Helper: calculate spread in pips
// ------------------------------------------------------------------
double GetSpreadPips()
{
    double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double pip    = SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 10.0;
    if(pip == 0.0) return 0.0;
    return spread / pip;
}

// ------------------------------------------------------------------
// Helper: build the 127-feature vector for AI scoring
// This is a partial implementation — in production Phase 7 will
// generate all 127 features via the Python feature engine.
// Here we populate with available MQL5 indicators.
// Features are indexed 0-126 per the architecture spec.
// ------------------------------------------------------------------
bool BuildFeatureVector(ENUM_DIRECTION direction)
{
    ArrayInitialize(g_features, 0.0);

    // ---- Price features (indices 0-14): M1 OHLCV ----
    g_features[0]  = iOpen(_Symbol,  PERIOD_M1, 1);
    g_features[1]  = iHigh(_Symbol,  PERIOD_M1, 1);
    g_features[2]  = iLow(_Symbol,   PERIOD_M1, 1);
    g_features[3]  = iClose(_Symbol, PERIOD_M1, 1);
    g_features[4]  = (double)iTickVolume(_Symbol, PERIOD_M1, 1);

    // ---- M5 OHLCV ----
    g_features[5]  = iOpen(_Symbol,  PERIOD_M5, 1);
    g_features[6]  = iHigh(_Symbol,  PERIOD_M5, 1);
    g_features[7]  = iLow(_Symbol,   PERIOD_M5, 1);
    g_features[8]  = iClose(_Symbol, PERIOD_M5, 1);
    g_features[9]  = (double)iTickVolume(_Symbol, PERIOD_M5, 1);

    // ---- M1 EMA values ----
    int ema8_h  = iMA(_Symbol, PERIOD_M1, 8,   0, MODE_EMA, PRICE_CLOSE);
    int ema21_h = iMA(_Symbol, PERIOD_M1, 21,  0, MODE_EMA, PRICE_CLOSE);
    int ema50_h = iMA(_Symbol, PERIOD_M1, 50,  0, MODE_EMA, PRICE_CLOSE);
    double ema_buf[1];
    if(CopyBuffer(ema8_h,  0, 1, 1, ema_buf) > 0) g_features[10] = ema_buf[0];
    if(CopyBuffer(ema21_h, 0, 1, 1, ema_buf) > 0) g_features[11] = ema_buf[0];
    if(CopyBuffer(ema50_h, 0, 1, 1, ema_buf) > 0) g_features[12] = ema_buf[0];
    IndicatorRelease(ema8_h);
    IndicatorRelease(ema21_h);
    IndicatorRelease(ema50_h);

    // ---- ATR(14) on M1 and M5 ----
    int atr14_m1 = iATR(_Symbol, PERIOD_M1, 14);
    int atr14_m5 = iATR(_Symbol, PERIOD_M5, 14);
    double atr_buf[1];
    if(CopyBuffer(atr14_m1, 0, 1, 1, atr_buf) > 0) g_features[13] = atr_buf[0];
    if(CopyBuffer(atr14_m5, 0, 1, 1, atr_buf) > 0) g_features[14] = atr_buf[0];
    IndicatorRelease(atr14_m1);
    IndicatorRelease(atr14_m5);

    // ---- RSI(7) and RSI(14) on M1 ----
    int rsi7_h  = iRSI(_Symbol, PERIOD_M1, 7,  PRICE_CLOSE);
    int rsi14_h = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
    double rsi_buf[1];
    if(CopyBuffer(rsi7_h,  0, 1, 1, rsi_buf) > 0) g_features[15] = rsi_buf[0];
    if(CopyBuffer(rsi14_h, 0, 1, 1, rsi_buf) > 0) g_features[16] = rsi_buf[0];
    IndicatorRelease(rsi7_h);
    IndicatorRelease(rsi14_h);

    // ---- MACD histogram on M1 ----
    int macd_h = iMACD(_Symbol, PERIOD_M1, 12, 26, 9, PRICE_CLOSE);
    double macd_buf[1];
    if(CopyBuffer(macd_h, 2, 1, 1, macd_buf) > 0) g_features[17] = macd_buf[0];  // histogram
    if(CopyBuffer(macd_h, 0, 1, 1, macd_buf) > 0) g_features[18] = macd_buf[0];  // MACD line
    if(CopyBuffer(macd_h, 1, 1, 1, macd_buf) > 0) g_features[19] = macd_buf[0];  // signal
    IndicatorRelease(macd_h);

    // ---- Bollinger Bands on M1 ----
    int bb_h = iBands(_Symbol, PERIOD_M1, 20, 0, 2.0, PRICE_CLOSE);
    double bb_buf[1];
    double bb_upper = 0.0, bb_lower = 0.0, bb_mid = 0.0;
    if(CopyBuffer(bb_h, 1, 1, 1, bb_buf) > 0) bb_upper = bb_buf[0];
    if(CopyBuffer(bb_h, 2, 1, 1, bb_buf) > 0) bb_lower = bb_buf[0];
    if(CopyBuffer(bb_h, 0, 1, 1, bb_buf) > 0) bb_mid   = bb_buf[0];
    IndicatorRelease(bb_h);
    if(bb_upper - bb_lower > 0)
        g_features[20] = (iClose(_Symbol, PERIOD_M1, 1) - bb_lower) / (bb_upper - bb_lower);  // %B
    g_features[21] = bb_upper - bb_lower;  // BB width

    // ---- ADX(14) on M1 ----
    int adx_h = iADX(_Symbol, PERIOD_M1, 14);
    double adx_buf[1];
    if(CopyBuffer(adx_h, 0, 1, 1, adx_buf) > 0) g_features[22] = adx_buf[0];  // ADX
    if(CopyBuffer(adx_h, 1, 1, 1, adx_buf) > 0) g_features[23] = adx_buf[0];  // +DI
    if(CopyBuffer(adx_h, 2, 1, 1, adx_buf) > 0) g_features[24] = adx_buf[0];  // -DI
    IndicatorRelease(adx_h);

    // ---- Direction encoding ----
    g_features[25] = (direction == DIRECTION_BULL) ? 1.0 :
                     (direction == DIRECTION_BEAR) ? -1.0 : 0.0;

    // ---- VWAP distance ----
    double vwap = g_vwap->GetVWAP(_Symbol);
    double close = iClose(_Symbol, PERIOD_M1, 1);
    g_features[26] = (vwap > 0.0) ? (close - vwap) : 0.0;

    // ---- Session info ----
    MqlDateTime dt;
    TimeGMT(dt);
    g_features[27] = (double)dt.hour;
    g_features[28] = (double)dt.day_of_week;
    g_features[29] = g_session->IsOverlap() ? 1.0 : 0.0;

    // ---- Risk state ----
    g_features[30] = g_risk_manager->GetSessionPnL();
    g_features[31] = (double)g_risk_manager->GetTradesThisSession();

    // ---- Spread ----
    g_features[32] = GetSpreadPips();

    // ---- Candle pattern encoding ----
    ENUM_CANDLE_PATTERN p = g_patterns->Detect(_Symbol, PERIOD_M1, 1);
    g_features[33] = (double)p;

    // ---- Market structure ----
    ENUM_DIRECTION ms = g_market_struct->GetTrend(_Symbol, PERIOD_M5, 60);
    g_features[34] = (ms == DIRECTION_BULL) ? 1.0 : (ms == DIRECTION_BEAR) ? -1.0 : 0.0;

    // Features 35-126: remaining 92 features are provided as 0.0 placeholders.
    // Phase 7 will implement the full 127-feature pipeline via Python feature
    // engine, making these available through a shared memory or file channel.
    // The AI server handles partial feature vectors in Phase 2 (dummy scoring).

    return true;
}

// ------------------------------------------------------------------
// Helper: log a gate failure message (avoids log spam with throttle)
// ------------------------------------------------------------------
static datetime s_last_gate_log = 0;

void LogGateBlock(string gate_name)
{
    datetime now = TimeCurrent();
    if(now - s_last_gate_log >= 60)  // Log at most once per minute
    {
        PrintFormat("[GoldScalper] Gate blocked: %s", gate_name);
        s_last_gate_log = now;
    }
}

// ------------------------------------------------------------------
// Helper: scan for newly closed deals and update RiskManager
// ------------------------------------------------------------------
void UpdateRiskFromClosedDeals()
{
    HistorySelect(TimeCurrent() - 86400, TimeCurrent());  // Last 24 hours
    int deals = HistoryDealsTotal();
    if(deals == 0) return;

    for(int i = deals - 1; i >= 0; i--)
    {
        ulong ticket = HistoryDealGetTicket(i);
        if(ticket == 0) continue;
        if(ticket <= g_last_deal_ticket) break;  // Already processed

        long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
        if(magic != MAGIC_SCALPER) continue;

        ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket, DEAL_ENTRY);
        if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT) continue;

        double pnl = HistoryDealGetDouble(ticket, DEAL_PROFIT) +
                     HistoryDealGetDouble(ticket, DEAL_COMMISSION) +
                     HistoryDealGetDouble(ticket, DEAL_SWAP);

        g_risk_manager->TrackTrade(pnl);
        g_last_deal_ticket = ticket;

        PrintFormat("[GoldScalper] Closed deal: ticket=%d pnl=%.2f", (int)ticket, pnl);
    }
}

// ------------------------------------------------------------------
// OnInit
// ------------------------------------------------------------------
int OnInit()
{
    Print("==========================================================");
    Print("[GoldScalper] Initializing Gold Scalper EA v1.0");
    Print("[GoldScalper] Symbol: ", _Symbol, " | Magic: ", MAGIC_SCALPER);
    Print("==========================================================");

    // Validate symbol
    if(_Symbol != "XAUUSD" && StringFind(_Symbol, "GOLD") < 0 && StringFind(_Symbol, "XAUUSD") < 0)
    {
        Print("[GoldScalper] WARNING: EA designed for XAUUSD. Current symbol: ", _Symbol);
    }

    // Allocate objects
    g_market_struct   = new CMarketStructure();
    g_patterns        = new CCandlePatterns();
    g_news_shield     = new CNewsShield();
    g_spike_detector  = new CSpikeDetector();
    g_session         = new CSessionManager();
    g_vwap            = new CVWAPCalculator();
    g_direction       = new CDirectionLayer();
    g_entry           = new CEntryLayer();
    g_exit_manager    = new CExitManager();
    g_risk_manager    = new CRiskManager();
    g_ai_client       = new CAIClient();

    // Verify allocations
    if(g_market_struct  == NULL || g_patterns     == NULL ||
       g_news_shield    == NULL || g_spike_detector == NULL ||
       g_session        == NULL || g_vwap          == NULL ||
       g_direction      == NULL || g_entry         == NULL ||
       g_exit_manager   == NULL || g_risk_manager  == NULL ||
       g_ai_client      == NULL)
    {
        Print("[GoldScalper] FATAL: Object allocation failed");
        return INIT_FAILED;
    }

    // ---- Initialize SpikeDetector ----
    if(!g_spike_detector->Init(_Symbol))
    {
        Print("[GoldScalper] WARNING: SpikeDetector init failed");
        // Non-fatal — spike detection will be inactive
    }

    // ---- Initialize DirectionLayer ----
    if(!g_direction->Init(_Symbol, g_market_struct, g_vwap, g_session, (double)MinATR_M5))
    {
        Print("[GoldScalper] FATAL: DirectionLayer init failed");
        return INIT_FAILED;
    }

    // ---- Initialize EntryLayer ----
    if(!g_entry->Init(_Symbol, MAGIC_SCALPER, g_patterns, g_session, AIMaxScore))
    {
        Print("[GoldScalper] FATAL: EntryLayer init failed");
        return INIT_FAILED;
    }

    // ---- Initialize ExitManager ----
    if(!g_exit_manager->Init(_Symbol, MAGIC_SCALPER,
                              g_direction, g_entry, g_vwap, g_patterns,
                              HardStopPips, MaxTradeDuration,
                              BreakevenTrigger, TrailingDistance,
                              TPMultiplier, MomentumTPMult, MomentumTrigger))
    {
        Print("[GoldScalper] FATAL: ExitManager init failed");
        return INIT_FAILED;
    }

    // ---- Initialize RiskManager ----
    if(!g_risk_manager->Init(_Symbol, RiskPercent, SessionRiskCap, SessionHaltPct))
    {
        Print("[GoldScalper] FATAL: RiskManager init failed");
        return INIT_FAILED;
    }

    // ---- Connect AI client ----
    g_ai_client->SetPort(AIServerPort);
    bool ai_connected = g_ai_client->Connect();
    if(!ai_connected)
    {
        PrintFormat("[GoldScalper] WARNING: AI server not reachable on port %d — fallback mode active",
                    AIServerPort);
        // Non-fatal: EA will use fallback scoring (block entry when AI unreachable)
    }
    else
    {
        Print("[GoldScalper] AI server connected successfully");
    }

    // ---- VWAP initial load ----
    g_vwap->Update(_Symbol);

    // ---- Set 60-second heartbeat timer ----
    EventSetTimer(60);

    // ---- Record initial balance ----
    g_prev_balance   = AccountInfoDouble(ACCOUNT_BALANCE);
    g_last_deal_ticket = 0;

    g_init_ok = true;

    PrintFormat("[GoldScalper] Init complete: risk=%.1f%% session_cap=%.1f%% AI_min=%d spread_max=%.1f",
                RiskPercent, SessionRiskCap, AIMinScore, MaxSpreadPips);

    return INIT_SUCCEEDED;
}

// ------------------------------------------------------------------
// OnDeinit
// ------------------------------------------------------------------
void OnDeinit(const int reason)
{
    EventKillTimer();

    // Log final session stats
    if(g_risk_manager != NULL)
    {
        PrintFormat("[GoldScalper] Shutdown stats: session P&L=%.2f%%, trades=%d",
                    g_risk_manager->GetSessionPnL(),
                    g_risk_manager->GetTradesThisSession());
        g_risk_manager->CloseSession();
    }

    // Disconnect AI client
    if(g_ai_client != NULL)
    {
        g_ai_client->Disconnect();
    }

    // Free all objects
    if(g_ai_client      != NULL) { delete g_ai_client;       g_ai_client      = NULL; }
    if(g_direction      != NULL) { delete g_direction;       g_direction      = NULL; }
    if(g_entry          != NULL) { delete g_entry;           g_entry          = NULL; }
    if(g_exit_manager   != NULL) { delete g_exit_manager;    g_exit_manager   = NULL; }
    if(g_risk_manager   != NULL) { delete g_risk_manager;    g_risk_manager   = NULL; }
    if(g_vwap           != NULL) { delete g_vwap;            g_vwap           = NULL; }
    if(g_session        != NULL) { delete g_session;         g_session        = NULL; }
    if(g_spike_detector != NULL) { delete g_spike_detector;  g_spike_detector = NULL; }
    if(g_news_shield    != NULL) { delete g_news_shield;     g_news_shield    = NULL; }
    if(g_patterns       != NULL) { delete g_patterns;        g_patterns       = NULL; }
    if(g_market_struct  != NULL) { delete g_market_struct;   g_market_struct  = NULL; }

    Print("[GoldScalper] Shutdown complete. Reason: ", reason);
}

// ------------------------------------------------------------------
// OnTimer — 60-second heartbeat to AI server
// ------------------------------------------------------------------
void OnTimer()
{
    if(!g_init_ok || g_ai_client == NULL) return;

    string status       = "";
    int    uptime       = 0;
    string model_ver    = "";

    bool ok = g_ai_client->Heartbeat(status, uptime, model_ver);
    if(ok)
    {
        PrintFormat("[GoldScalper] AI heartbeat OK: status=%s uptime=%ds model=%s",
                    status, uptime, model_ver);
        // If we were in fallback, we're now recovered
    }
    else
    {
        PrintFormat("[GoldScalper] AI heartbeat FAILED (failures=%d/%s)",
                    g_ai_client->GetFailureCount(),
                    g_ai_client->IsInFallbackMode() ? "FALLBACK" : "ok");

        // Attempt reconnect if not in fallback
        if(!g_ai_client->IsInFallbackMode())
            g_ai_client->Connect();
    }

    // Update risk from any closed deals
    UpdateRiskFromClosedDeals();

    // Keep session P&L current
    g_risk_manager->UpdateFromBalance();
}

// ------------------------------------------------------------------
// OnTick — main execution loop
// ------------------------------------------------------------------
void OnTick()
{
    if(!g_init_ok) return;

    // ---- Update always-on services ----
    g_news_shield->Update();
    g_spike_detector->Update(_Symbol);
    g_vwap->Update(_Symbol);

    // ---- Always: manage open trades (highest priority) ----
    g_exit_manager->ManageOpenTrades();

    // ---- Always: advance cascade state machine ----
    if(g_entry->IsActive())
        g_entry->ManageCascade();

    // =====================================================
    // NEW ENTRY GATE SEQUENCE
    // Any failed gate: log and return (no new entry)
    // =====================================================

    // Skip new entries if cascade already active
    if(g_entry->IsActive())
        return;

    // ---- Gate 1: NewsShield ----
    if(g_news_shield->BlocksNewEntries())
    {
        LogGateBlock("NewsShield (pre/during news)");
        return;
    }

    // ---- Gate 2: SpikeDetector ----
    if(g_spike_detector->IsActive())
    {
        LogGateBlock(StringFormat("SpikeDetector (cooldown until %s)",
                     TimeToString(g_spike_detector->GetCooldownEnd(), TIME_MINUTES)));
        return;
    }

    // ---- Gate 3: RiskManager ----
    if(g_risk_manager->IsSessionHalted())
    {
        LogGateBlock(StringFormat("RiskManager (SESSION HALTED at %.1f%%)", SessionHaltPct));
        return;
    }
    if(g_risk_manager->SessionCapReached())
    {
        LogGateBlock(StringFormat("RiskManager (session cap %.1f%% reached)", SessionRiskCap));
        return;
    }
    if(!g_risk_manager->HasBudgetForTrade())
    {
        LogGateBlock("RiskManager (insufficient session budget)");
        return;
    }

    // ---- Gate 4: SessionManager ----
    if(!g_session->IsScalperActive())
    {
        LogGateBlock(StringFormat("SessionManager (inactive session: %s)",
                     g_session->GetSessionName()));
        return;
    }

    // ---- Gate 5: Spread gate ----
    double current_spread = GetSpreadPips();
    if(current_spread > MaxSpreadPips)
    {
        LogGateBlock(StringFormat("Spread gate (%.2f > %.1f pips)", current_spread, MaxSpreadPips));
        return;
    }

    // ---- Gate 6: Direction (M5 consensus) ----
    ENUM_DIRECTION direction = g_direction->Get5MBias();
    if(direction == DIRECTION_NONE)
    {
        // Don't log this — direction disagreement is frequent and expected
        return;
    }

    // ---- Gate 7: Entry signal (M1 conditions) ----
    if(!g_entry->HasSignal(direction))
    {
        return;  // No pullback/rejection/RSI signal
    }

    // ---- Gate 8: AI Score ----
    // Build feature vector
    if(!BuildFeatureVector(direction))
    {
        Print("[GoldScalper] Feature vector build failed — skipping");
        return;
    }

    int    entry_score    = 0;
    int    trend_score    = 0;
    int    news_risk      = 0;
    string regime         = "";
    string wyckoff_phase  = "";
    bool   ai_approve     = false;
    double lot_multiplier = 1.0;

    bool ai_ok = g_ai_client->ScoreEntry(
        direction, "M1", "scalper",
        g_features,
        entry_score, trend_score, news_risk,
        regime, wyckoff_phase, ai_approve, lot_multiplier
    );

    if(!ai_ok)
    {
        if(g_ai_client->IsInFallbackMode())
        {
            // Fallback: block Entry 4 (Max) but allow entries 1-3 if score
            // cannot be obtained. Use conservative default behaviour: block all.
            // This is the safe default per the spec. Change to allow if desired.
            LogGateBlock("AI client (fallback mode — all entries blocked for safety)");
            return;
        }
        LogGateBlock(StringFormat("AI client error: %s", g_ai_client->GetLastError()));
        return;
    }

    if(!ai_approve)
    {
        PrintFormat("[GoldScalper] AI rejected entry: score=%d trend=%d news_risk=%d regime=%s",
                    entry_score, trend_score, news_risk, regime);
        return;
    }

    if(entry_score < AIMinScore)
    {
        PrintFormat("[GoldScalper] AI score below threshold: %d < %d", entry_score, AIMinScore);
        return;
    }

    // Pass AI score to EntryLayer (for Entry 4 gate)
    g_entry->SetAIScore(entry_score);

    PrintFormat("[GoldScalper] AI APPROVED: score=%d trend=%d news=%d regime=%s phase=%s lot_mult=%.2f",
                entry_score, trend_score, news_risk, regime, wyckoff_phase, lot_multiplier);

    // ---- Gate 9: Execute Pilot entry ----

    // Calculate lot size from RiskManager
    // Use approximate SL of 12 pips for initial lot calculation
    // The actual SL is set by EntryLayer based on M1 swing low/high
    double sl_pips = 12.0;  // Conservative estimate
    double base_lot = g_risk_manager->CalcLotSize(sl_pips);

    // Apply AI lot multiplier (capped at broker min/max in CalcLotSize)
    // Note: Pilot is always 0.01 lots per spec — lot multiplier
    // is for informational purposes here; the EntryLayer enforces fixed lots
    PrintFormat("[GoldScalper] Executing Pilot: direction=%s base_lot=%.2f AI_lot_mult=%.2f",
                (direction == DIRECTION_BULL) ? "BUY" : "SELL",
                base_lot, lot_multiplier);

    if(!g_entry->ExecutePilot(direction))
    {
        Print("[GoldScalper] Pilot execution FAILED");
        return;
    }

    // Set TP levels in ExitManager now that Pilot is confirmed
    double atr_m5 = g_direction->GetCurrentATR();
    g_exit_manager->SetTPLevels(g_entry->GetPilotOpenPrice(), atr_m5, direction);

    PrintFormat("[GoldScalper] Pilot CONFIRMED: ticket=%d TP=%.5f",
                (int)g_entry->GetPilotTicket(),
                g_exit_manager->GetPrimaryTP());
}

// ------------------------------------------------------------------
// OnTradeTransaction — detect when positions close
// ------------------------------------------------------------------
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
    if(!g_init_ok) return;

    // When a deal is executed (position closed)
    if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
    {
        // Let OnTimer handle deal scanning to avoid double-counting
        // (OnTimer runs every 60s and does a full history scan)

        // If cascade was fully closed (no more open positions with our magic),
        // reset the entry layer
        if(g_entry->IsActive())
        {
            ulong tickets[];
            int open_count = g_entry->GetOpenTickets(tickets);
            if(open_count == 0)
            {
                Print("[GoldScalper] All cascade positions closed — resetting entry layer");
                g_entry->Reset();
            }
        }
    }
}

// ------------------------------------------------------------------
// End of file
// ------------------------------------------------------------------
