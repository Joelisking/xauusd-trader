//+------------------------------------------------------------------+
//| DirectionLayer.mqh — M5 direction system for Gold Scalper        |
//|                                                                   |
//| Five-indicator consensus model. ALL five must agree.             |
//| Any disagreement -> DIRECTION_NONE.                               |
//|                                                                   |
//| Indicators:                                                       |
//|   1. EMA stack (21/50/200) alignment on M5                        |
//|   2. Market structure (2+ HH/HL = BULL, 2+ LL/LH = BEAR)         |
//|   3. MACD histogram sign (positive = BULL, negative = BEAR)       |
//|   4. ATR(14) > MinATR pips gate (market must have energy)         |
//|   5. VWAP filter (above VWAP = long only, during overlap only)    |
//|                                                                   |
//| Direction is cached and updated only on new M5 bar close.         |
//| Does NOT flicker between M5 candle opens.                         |
//+------------------------------------------------------------------+
#ifndef DIRECTIONLAYER_MQH
#define DIRECTIONLAYER_MQH

#include "Constants.mqh"
#include "MarketStructure.mqh"
#include "VWAPCalculator.mqh"
#include "SessionManager.mqh"

class CDirectionLayer
{
private:
    // ------------------------------------------------------------------
    // Indicator handles
    // ------------------------------------------------------------------
    int     m_ema21_handle;
    int     m_ema50_handle;
    int     m_ema200_handle;
    int     m_macd_handle;
    int     m_atr_handle;

    // ------------------------------------------------------------------
    // Dependencies
    // ------------------------------------------------------------------
    CMarketStructure  *m_market_structure;
    CVWAPCalculator   *m_vwap;
    CSessionManager   *m_session;

    // ------------------------------------------------------------------
    // State / cache
    // ------------------------------------------------------------------
    ENUM_DIRECTION  m_cached_direction;
    datetime        m_last_m5_bar_time;
    string          m_symbol;

    // ------------------------------------------------------------------
    // Input thresholds (set via Init)
    // ------------------------------------------------------------------
    double  m_min_atr_pips;     // Default 5.0 pips
    double  m_pip_size;         // XAUUSD: 0.1 (1 pip = $0.10)

    // ------------------------------------------------------------------
    // Private: pip conversion helper
    // ------------------------------------------------------------------
    double PointsToPips(double points)
    {
        // For XAUUSD: 1 pip = 10 points (price unit = 0.01, pip = 0.10)
        return points / (SymbolInfoDouble(m_symbol, SYMBOL_POINT) * 10.0);
    }

    // ------------------------------------------------------------------
    // Private: evaluate EMA stack on M5
    // Returns BULL if 21>50>200 and price above all three.
    // Returns BEAR if 21<50<200 and price below all three.
    // Returns DIRECTION_NONE otherwise.
    // ------------------------------------------------------------------
    ENUM_DIRECTION CheckEMAStack()
    {
        double ema21[1], ema50[1], ema200[1];

        if(CopyBuffer(m_ema21_handle,  0, 1, 1, ema21)  <= 0) return DIRECTION_NONE;
        if(CopyBuffer(m_ema50_handle,  0, 1, 1, ema50)  <= 0) return DIRECTION_NONE;
        if(CopyBuffer(m_ema200_handle, 0, 1, 1, ema200) <= 0) return DIRECTION_NONE;

        double close = iClose(m_symbol, PERIOD_M5, 1);

        // Bullish stack: 21 > 50 > 200 and price above all
        bool bull_stack = (ema21[0] > ema50[0]) &&
                          (ema50[0] > ema200[0]) &&
                          (close > ema21[0]);

        // Bearish stack: 21 < 50 < 200 and price below all
        bool bear_stack = (ema21[0] < ema50[0]) &&
                          (ema50[0] < ema200[0]) &&
                          (close < ema21[0]);

        if(bull_stack) return DIRECTION_BULL;
        if(bear_stack) return DIRECTION_BEAR;
        return DIRECTION_NONE;
    }

    // ------------------------------------------------------------------
    // Private: evaluate MACD histogram sign on M5
    // histogram[0] is the buffer index for MACD histogram in MT5
    // iMACD: buffer 0 = MACD line, buffer 1 = signal, buffer 2 = histogram
    // ------------------------------------------------------------------
    ENUM_DIRECTION CheckMACD()
    {
        double hist[1];
        // Buffer index 2 = MACD histogram
        if(CopyBuffer(m_macd_handle, 2, 1, 1, hist) <= 0)
            return DIRECTION_NONE;

        if(hist[0] > 0.0) return DIRECTION_BULL;
        if(hist[0] < 0.0) return DIRECTION_BEAR;
        return DIRECTION_NONE;  // Exactly zero = indeterminate
    }

    // ------------------------------------------------------------------
    // Private: ATR gate — returns true if ATR(14) on M5 > threshold
    // ------------------------------------------------------------------
    bool CheckATRGate()
    {
        double atr[1];
        if(CopyBuffer(m_atr_handle, 0, 1, 1, atr) <= 0)
            return false;

        double atr_pips = PointsToPips(atr[0]);
        return (atr_pips >= m_min_atr_pips);
    }

    // ------------------------------------------------------------------
    // Private: VWAP filter
    // During London-NY overlap: above VWAP = long only, below = short only
    // Outside overlap: VWAP filter is neutral (both directions allowed)
    // ------------------------------------------------------------------
    ENUM_DIRECTION CheckVWAP()
    {
        // VWAP filter only strictly enforced during London-NY overlap
        bool is_overlap = m_session->IsOverlap();

        double vwap = m_vwap->GetVWAP(m_symbol);

        // If VWAP not yet calculated, skip filter (return DIRECTION_NONE
        // would block all trades — instead return a neutral "pass" via a
        // special convention: we return the direction that matches price
        // regardless, which the caller treats as non-blocking)
        if(vwap == 0.0)
            return DIRECTION_NONE; // Cannot confirm, treat as disagreement

        double price = iClose(m_symbol, PERIOD_M5, 1);

        if(is_overlap)
        {
            // Strict VWAP enforcement during overlap
            if(price > vwap) return DIRECTION_BULL;
            if(price < vwap) return DIRECTION_BEAR;
            return DIRECTION_NONE;  // Exactly at VWAP
        }
        else
        {
            // Outside overlap: VWAP acts as a soft guide only
            // Return whichever direction the price is relative to VWAP
            // This allows the consensus to still work in non-overlap hours
            if(price > vwap) return DIRECTION_BULL;
            if(price < vwap) return DIRECTION_BEAR;
            return DIRECTION_NONE;
        }
    }

    // ------------------------------------------------------------------
    // Private: core calculation — called only on new M5 bar
    // ------------------------------------------------------------------
    ENUM_DIRECTION CalculateDirection()
    {
        // Gate: ATR must clear threshold or no direction
        if(!CheckATRGate())
        {
            PrintFormat("[DirectionLayer] ATR gate FAILED — market too quiet (min %.1f pips)", m_min_atr_pips);
            return DIRECTION_NONE;
        }

        // Evaluate each indicator
        ENUM_DIRECTION ema_dir  = CheckEMAStack();
        ENUM_DIRECTION ms_dir   = m_market_structure->GetTrend(m_symbol, PERIOD_M5, 60);
        ENUM_DIRECTION macd_dir = CheckMACD();
        ENUM_DIRECTION vwap_dir = CheckVWAP();

        // Log individual signals for diagnostics
        string ema_str  = (ema_dir  == DIRECTION_BULL) ? "BULL" : (ema_dir  == DIRECTION_BEAR) ? "BEAR" : "NONE";
        string ms_str   = (ms_dir   == DIRECTION_BULL) ? "BULL" : (ms_dir   == DIRECTION_BEAR) ? "BEAR" : "NONE";
        string macd_str = (macd_dir == DIRECTION_BULL) ? "BULL" : (macd_dir == DIRECTION_BEAR) ? "BEAR" : "NONE";
        string vwap_str = (vwap_dir == DIRECTION_BULL) ? "BULL" : (vwap_dir == DIRECTION_BEAR) ? "BEAR" : "NONE";

        PrintFormat("[DirectionLayer] M5 signals | EMA:%s | Structure:%s | MACD:%s | VWAP:%s | ATR:PASS",
                    ema_str, ms_str, macd_str, vwap_str);

        // All four directional indicators must agree (ATR is a gate, not directional)
        // Check for bullish consensus
        if(ema_dir  == DIRECTION_BULL &&
           ms_dir   == DIRECTION_BULL &&
           macd_dir == DIRECTION_BULL &&
           vwap_dir == DIRECTION_BULL)
        {
            Print("[DirectionLayer] BULL consensus confirmed");
            return DIRECTION_BULL;
        }

        // Check for bearish consensus
        if(ema_dir  == DIRECTION_BEAR &&
           ms_dir   == DIRECTION_BEAR &&
           macd_dir == DIRECTION_BEAR &&
           vwap_dir == DIRECTION_BEAR)
        {
            Print("[DirectionLayer] BEAR consensus confirmed");
            return DIRECTION_BEAR;
        }

        // Disagreement among indicators
        Print("[DirectionLayer] No consensus — DIRECTION_NONE");
        return DIRECTION_NONE;
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------
    CDirectionLayer()
    {
        m_ema21_handle    = INVALID_HANDLE;
        m_ema50_handle    = INVALID_HANDLE;
        m_ema200_handle   = INVALID_HANDLE;
        m_macd_handle     = INVALID_HANDLE;
        m_atr_handle      = INVALID_HANDLE;

        m_market_structure = NULL;
        m_vwap             = NULL;
        m_session          = NULL;

        m_cached_direction = DIRECTION_NONE;
        m_last_m5_bar_time = 0;
        m_symbol           = _Symbol;
        m_min_atr_pips     = 5.0;
        m_pip_size         = 0.1;   // XAUUSD: $0.10 per pip
    }

    // ------------------------------------------------------------------
    // Destructor
    // ------------------------------------------------------------------
    ~CDirectionLayer()
    {
        Deinit();
    }

    // ------------------------------------------------------------------
    // Init — create indicator handles and link dependencies
    // ------------------------------------------------------------------
    bool Init(string           symbol,
              CMarketStructure *market_structure,
              CVWAPCalculator  *vwap,
              CSessionManager  *session,
              double           min_atr_pips = 5.0)
    {
        m_symbol           = symbol;
        m_market_structure = market_structure;
        m_vwap             = vwap;
        m_session          = session;
        m_min_atr_pips     = min_atr_pips;

        // Create EMA handles on M5
        m_ema21_handle  = iMA(symbol, PERIOD_M5, 21,  0, MODE_EMA, PRICE_CLOSE);
        m_ema50_handle  = iMA(symbol, PERIOD_M5, 50,  0, MODE_EMA, PRICE_CLOSE);
        m_ema200_handle = iMA(symbol, PERIOD_M5, 200, 0, MODE_EMA, PRICE_CLOSE);

        // MACD(12, 26, 9) on M5
        m_macd_handle = iMACD(symbol, PERIOD_M5, 12, 26, 9, PRICE_CLOSE);

        // ATR(14) on M5
        m_atr_handle = iATR(symbol, PERIOD_M5, 14);

        if(m_ema21_handle  == INVALID_HANDLE ||
           m_ema50_handle  == INVALID_HANDLE ||
           m_ema200_handle == INVALID_HANDLE ||
           m_macd_handle   == INVALID_HANDLE ||
           m_atr_handle    == INVALID_HANDLE)
        {
            Print("[DirectionLayer] ERROR: Failed to create one or more indicator handles");
            return false;
        }

        Print("[DirectionLayer] Initialized successfully");
        return true;
    }

    // ------------------------------------------------------------------
    // Deinit — release indicator handles
    // ------------------------------------------------------------------
    void Deinit()
    {
        if(m_ema21_handle  != INVALID_HANDLE) { IndicatorRelease(m_ema21_handle);  m_ema21_handle  = INVALID_HANDLE; }
        if(m_ema50_handle  != INVALID_HANDLE) { IndicatorRelease(m_ema50_handle);  m_ema50_handle  = INVALID_HANDLE; }
        if(m_ema200_handle != INVALID_HANDLE) { IndicatorRelease(m_ema200_handle); m_ema200_handle = INVALID_HANDLE; }
        if(m_macd_handle   != INVALID_HANDLE) { IndicatorRelease(m_macd_handle);   m_macd_handle   = INVALID_HANDLE; }
        if(m_atr_handle    != INVALID_HANDLE) { IndicatorRelease(m_atr_handle);    m_atr_handle    = INVALID_HANDLE; }
    }

    // ------------------------------------------------------------------
    // Get5MBias — main public interface
    //
    // Updates direction ONLY when a new M5 bar has formed.
    // Between M5 bar closes, returns the cached value.
    // This eliminates tick-level flicker entirely.
    // ------------------------------------------------------------------
    ENUM_DIRECTION Get5MBias()
    {
        datetime current_m5_bar = iTime(m_symbol, PERIOD_M5, 1);

        // New M5 bar has closed since last check
        if(current_m5_bar != m_last_m5_bar_time)
        {
            m_last_m5_bar_time = current_m5_bar;
            m_cached_direction = CalculateDirection();
        }

        return m_cached_direction;
    }

    // ------------------------------------------------------------------
    // ForceRecalculate — call after session open or VWAP reset
    // ------------------------------------------------------------------
    void ForceRecalculate()
    {
        m_last_m5_bar_time = 0;  // Forces recalculation on next call
    }

    // ------------------------------------------------------------------
    // GetCurrentATR — used by other modules for TP/SL calculations
    // ------------------------------------------------------------------
    double GetCurrentATR()
    {
        double atr[1];
        if(CopyBuffer(m_atr_handle, 0, 1, 1, atr) <= 0)
            return 0.0;
        return atr[0];
    }

    // ------------------------------------------------------------------
    // GetCurrentATRPips — returns ATR in pips (divide by pip size)
    // ------------------------------------------------------------------
    double GetCurrentATRPips()
    {
        return PointsToPips(GetCurrentATR());
    }

    // ------------------------------------------------------------------
    // GetEMA21 — used by EntryLayer for pullback detection
    // ------------------------------------------------------------------
    double GetEMA21_M5()
    {
        double ema[1];
        if(CopyBuffer(m_ema21_handle, 0, 0, 1, ema) <= 0)
            return 0.0;
        return ema[0];
    }

    // ------------------------------------------------------------------
    // IsEMAStackBull / IsEMAStackBear — used by ExitManager for flip check
    // ------------------------------------------------------------------
    bool IsEMAStackBull()
    {
        return CheckEMAStack() == DIRECTION_BULL;
    }

    bool IsEMAStackBear()
    {
        return CheckEMAStack() == DIRECTION_BEAR;
    }

    // ------------------------------------------------------------------
    // GetCachedDirection — return cached value without recalculating
    // ------------------------------------------------------------------
    ENUM_DIRECTION GetCachedDirection()
    {
        return m_cached_direction;
    }
};

#endif // DIRECTIONLAYER_MQH
