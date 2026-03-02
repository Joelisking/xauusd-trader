//+------------------------------------------------------------------+
//| H4DirectionSystem.mqh — Institutional H4 trend direction filter   |
//| All 6 indicators must align for a confirmed direction             |
//| Updated only on H4 candle close — cached between closes          |
//+------------------------------------------------------------------+
#ifndef H4DIRECTIONSYSTEM_MQH
#define H4DIRECTIONSYSTEM_MQH

#include "Constants.mqh"
#include "MarketStructure.mqh"

class CH4DirectionSystem
{
private:
    // Cached direction state
    ENUM_DIRECTION  m_cached_direction;
    datetime        m_last_h4_candle;      // iTime(symbol, H4, 1) at last update
    bool            m_cache_valid;

    // Last AI response data
    int             m_last_ai_trend_score;
    string          m_last_wyckoff_phase;

    // Indicator handles — created once in Init()
    int             m_ema200_h4_handle;
    int             m_rsi14_h4_handle;
    int             m_ema200_w1_handle;    // Weekly 200 EMA for weekly trend
    int             m_rsi14_w1_handle;     // Weekly RSI for direction check

    // Market structure analyzer
    CMarketStructure m_ms;

    // EA inputs cached locally
    int             m_min_ai_trend_score;
    string          m_symbol;

    // ------------------------------------------------------------------
    // Indicator 1: Market Structure on H4
    // Requires 2+ consecutive HH/HL (bull) or LL/LH (bear)
    // ------------------------------------------------------------------

    ENUM_DIRECTION CheckMarketStructure()
    {
        m_ms.SetFractalBars(5);
        m_ms.SetLookback(60);
        return m_ms.GetTrend(m_symbol, PERIOD_H4, 60);
    }

    // ------------------------------------------------------------------
    // Indicator 2: 200 EMA on H4 — binary, no exceptions
    // Above = long ONLY, Below = short ONLY
    // ------------------------------------------------------------------

    ENUM_DIRECTION CheckEMA200()
    {
        if(m_ema200_h4_handle == INVALID_HANDLE)
            return DIRECTION_NONE;

        double ema_buf[];
        ArraySetAsSeries(ema_buf, true);
        if(CopyBuffer(m_ema200_h4_handle, 0, 1, 1, ema_buf) <= 0)
            return DIRECTION_NONE;

        double close_h4_1 = iClose(m_symbol, PERIOD_H4, 1);  // Last closed bar

        if(close_h4_1 > ema_buf[0])
            return DIRECTION_BULL;
        if(close_h4_1 < ema_buf[0])
            return DIRECTION_BEAR;

        return DIRECTION_NONE;
    }

    // ------------------------------------------------------------------
    // Indicator 3: Weekly alignment
    // Weekly chart direction must match H4 direction
    // Uses market structure on W1 plus price vs weekly 200 EMA
    // ------------------------------------------------------------------

    ENUM_DIRECTION CheckWeeklyAlignment()
    {
        // Weekly market structure
        m_ms.SetFractalBars(3);
        m_ms.SetLookback(20);
        ENUM_DIRECTION weekly_ms = m_ms.GetTrend(m_symbol, PERIOD_W1, 20);

        // Cross-check with weekly 200 EMA if handle valid
        if(m_ema200_w1_handle != INVALID_HANDLE)
        {
            double w1_ema_buf[];
            ArraySetAsSeries(w1_ema_buf, true);
            if(CopyBuffer(m_ema200_w1_handle, 0, 1, 1, w1_ema_buf) > 0)
            {
                double close_w1_1 = iClose(m_symbol, PERIOD_W1, 1);

                ENUM_DIRECTION ema_dir = DIRECTION_NONE;
                if(close_w1_1 > w1_ema_buf[0])  ema_dir = DIRECTION_BULL;
                if(close_w1_1 < w1_ema_buf[0])  ema_dir = DIRECTION_BEAR;

                // Both must agree for a confirmed weekly direction
                if(ema_dir != DIRECTION_NONE && weekly_ms != DIRECTION_NONE)
                {
                    if(ema_dir == weekly_ms)
                        return weekly_ms;
                    else
                        return DIRECTION_NONE;   // Weekly MS and EMA disagree
                }

                // One signal available — use it
                if(ema_dir != DIRECTION_NONE) return ema_dir;
            }
        }

        return weekly_ms;
    }

    // ------------------------------------------------------------------
    // Indicator 4: RSI(14) on H4
    // Above 50 = bull, below 50 = bear
    // ------------------------------------------------------------------

    ENUM_DIRECTION CheckH4RSI()
    {
        if(m_rsi14_h4_handle == INVALID_HANDLE)
            return DIRECTION_NONE;

        double rsi_buf[];
        ArraySetAsSeries(rsi_buf, true);
        if(CopyBuffer(m_rsi14_h4_handle, 0, 1, 1, rsi_buf) <= 0)
            return DIRECTION_NONE;

        if(rsi_buf[0] > 50.0)
            return DIRECTION_BULL;
        if(rsi_buf[0] < 50.0)
            return DIRECTION_BEAR;

        return DIRECTION_NONE;
    }

    // ------------------------------------------------------------------
    // Indicator 5: AI Trend Strength Score >= m_min_ai_trend_score
    // Score stored from last AI response — checked externally
    // Returns BULL or BEAR based on score threshold only; direction from AI
    // is set by the caller via UpdateAIData()
    // ------------------------------------------------------------------

    bool AIScoretThresholdMet()
    {
        return (m_last_ai_trend_score >= m_min_ai_trend_score);
    }

    // ------------------------------------------------------------------
    // Indicator 6: Wyckoff phase from AI
    // Phase C or D = highest priority confirmation
    // Phase A, B = not confirmed (wait)
    // Phase E (after distribution top) = avoid longs
    // ------------------------------------------------------------------

    bool WyckoffPhaseConfirms(ENUM_DIRECTION direction)
    {
        // If no AI data yet, don't block (treated as neutral)
        if(StringLen(m_last_wyckoff_phase) == 0)
            return true;

        string phase = m_last_wyckoff_phase;

        if(direction == DIRECTION_BULL)
        {
            // Accumulation Phase C (Spring/test) or D (SOS) = bullish
            if(phase == "C" || phase == "D")
                return true;
            // Phase E = markdown, avoid longs
            if(phase == "E")
                return false;
            // Phase A, B = still developing — allow but lower conviction
            return true;
        }

        if(direction == DIRECTION_BEAR)
        {
            // Distribution Phase B or C = bearish pressure
            if(phase == "B" || phase == "C")
                return true;
            // Phase D = sign of weakness in distribution — bearish
            if(phase == "D")
                return true;
            // Phase A = just topped — allow but be cautious
            return true;
        }

        return true;
    }

    // ------------------------------------------------------------------
    // Perform the full 6-indicator evaluation
    // All six must agree; any DIRECTION_NONE or mismatch = DIRECTION_NONE
    // ------------------------------------------------------------------

    ENUM_DIRECTION EvaluateAllIndicators()
    {
        // 1. Market Structure
        ENUM_DIRECTION ms_dir = CheckMarketStructure();
        if(ms_dir == DIRECTION_NONE)
        {
            PrintFormat("[H4Dir] FAIL: Market structure ambiguous");
            return DIRECTION_NONE;
        }

        // 2. 200 EMA (binary filter)
        ENUM_DIRECTION ema_dir = CheckEMA200();
        if(ema_dir == DIRECTION_NONE)
        {
            PrintFormat("[H4Dir] FAIL: Price at H4 200 EMA — no bias");
            return DIRECTION_NONE;
        }
        if(ema_dir != ms_dir)
        {
            PrintFormat("[H4Dir] FAIL: 200 EMA (%d) disagrees with MS (%d)", ema_dir, ms_dir);
            return DIRECTION_NONE;
        }

        // 3. Weekly alignment
        ENUM_DIRECTION weekly_dir = CheckWeeklyAlignment();
        if(weekly_dir != DIRECTION_NONE && weekly_dir != ms_dir)
        {
            PrintFormat("[H4Dir] FAIL: Weekly direction (%d) opposes H4 (%d)", weekly_dir, ms_dir);
            return DIRECTION_NONE;
        }

        // 4. RSI(14) on H4
        ENUM_DIRECTION rsi_dir = CheckH4RSI();
        if(rsi_dir == DIRECTION_NONE)
        {
            PrintFormat("[H4Dir] FAIL: H4 RSI exactly at 50 — ambiguous");
            return DIRECTION_NONE;
        }
        if(rsi_dir != ms_dir)
        {
            PrintFormat("[H4Dir] FAIL: H4 RSI (%d) disagrees with MS (%d)", rsi_dir, ms_dir);
            return DIRECTION_NONE;
        }

        // 5. AI Trend Strength Score
        if(!AIScoretThresholdMet())
        {
            PrintFormat("[H4Dir] FAIL: AI trend score %d < required %d",
                        m_last_ai_trend_score, m_min_ai_trend_score);
            return DIRECTION_NONE;
        }

        // 6. Wyckoff phase
        if(!WyckoffPhaseConfirms(ms_dir))
        {
            PrintFormat("[H4Dir] FAIL: Wyckoff phase '%s' opposes %d",
                        m_last_wyckoff_phase, ms_dir);
            return DIRECTION_NONE;
        }

        // All 6 confirmed
        PrintFormat("[H4Dir] CONFIRMED direction=%d (MS=%d EMA=%d W1=%d RSI=%d AI=%d phase=%s)",
                    ms_dir, ms_dir, ema_dir, weekly_dir, rsi_dir,
                    m_last_ai_trend_score, m_last_wyckoff_phase);

        return ms_dir;
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    CH4DirectionSystem()
    {
        m_cached_direction     = DIRECTION_NONE;
        m_last_h4_candle       = 0;
        m_cache_valid          = false;
        m_last_ai_trend_score  = 0;
        m_last_wyckoff_phase   = "";
        m_ema200_h4_handle     = INVALID_HANDLE;
        m_rsi14_h4_handle      = INVALID_HANDLE;
        m_ema200_w1_handle     = INVALID_HANDLE;
        m_rsi14_w1_handle      = INVALID_HANDLE;
        m_min_ai_trend_score   = 72;
        m_symbol               = _Symbol;
    }

    ~CH4DirectionSystem()
    {
        if(m_ema200_h4_handle != INVALID_HANDLE) IndicatorRelease(m_ema200_h4_handle);
        if(m_rsi14_h4_handle  != INVALID_HANDLE) IndicatorRelease(m_rsi14_h4_handle);
        if(m_ema200_w1_handle != INVALID_HANDLE) IndicatorRelease(m_ema200_w1_handle);
        if(m_rsi14_w1_handle  != INVALID_HANDLE) IndicatorRelease(m_rsi14_w1_handle);
    }

    // ------------------------------------------------------------------
    // Initialise — must be called from OnInit()
    // ------------------------------------------------------------------

    bool Init(string symbol, int min_ai_trend_score)
    {
        m_symbol             = symbol;
        m_min_ai_trend_score = min_ai_trend_score;

        // Create H4 indicators
        m_ema200_h4_handle = iMA(symbol, PERIOD_H4, 200, 0, MODE_EMA, PRICE_CLOSE);
        if(m_ema200_h4_handle == INVALID_HANDLE)
        {
            Print("[H4Dir] ERROR: Failed to create H4 EMA(200) handle");
            return false;
        }

        m_rsi14_h4_handle = iRSI(symbol, PERIOD_H4, 14, PRICE_CLOSE);
        if(m_rsi14_h4_handle == INVALID_HANDLE)
        {
            Print("[H4Dir] ERROR: Failed to create H4 RSI(14) handle");
            return false;
        }

        // Create weekly indicators (non-critical — warn but don't fail)
        m_ema200_w1_handle = iMA(symbol, PERIOD_W1, 200, 0, MODE_EMA, PRICE_CLOSE);
        if(m_ema200_w1_handle == INVALID_HANDLE)
            Print("[H4Dir] WARNING: Failed to create W1 EMA(200) — weekly alignment will use MS only");

        Print("[H4Dir] Initialised successfully");
        return true;
    }

    // ------------------------------------------------------------------
    // UpdateAIData — called after each successful AI server response
    // ------------------------------------------------------------------

    void UpdateAIData(int trend_score, string wyckoff_phase)
    {
        m_last_ai_trend_score = trend_score;
        m_last_wyckoff_phase  = wyckoff_phase;
        // Invalidate cache so next GetH4Trend() re-evaluates
        m_cache_valid = false;
    }

    int    GetLastAITrendScore()  { return m_last_ai_trend_score; }
    string GetLastWyckoffPhase()  { return m_last_wyckoff_phase;  }

    // ------------------------------------------------------------------
    // GetH4Trend — primary public method
    //
    // Returns cached direction unless an H4 candle has closed since the
    // last evaluation, in which case it re-evaluates all 6 indicators.
    // This prevents indicator recalculation on every tick.
    // ------------------------------------------------------------------

    ENUM_DIRECTION GetH4Trend()
    {
        // Check if an H4 candle has closed since last evaluation
        datetime current_h4_candle = iTime(m_symbol, PERIOD_H4, 1);

        bool h4_closed = (current_h4_candle != m_last_h4_candle);

        if(m_cache_valid && !h4_closed)
            return m_cached_direction;

        // New H4 candle closed — re-evaluate
        m_cached_direction = EvaluateAllIndicators();
        m_last_h4_candle   = current_h4_candle;
        m_cache_valid      = true;

        if(h4_closed)
            PrintFormat("[H4Dir] H4 candle closed. New direction: %d", m_cached_direction);

        return m_cached_direction;
    }

    // ------------------------------------------------------------------
    // ForceEvaluate — bypasses cache, used for testing and on new AI data
    // ------------------------------------------------------------------

    ENUM_DIRECTION ForceEvaluate()
    {
        m_cache_valid = false;
        return GetH4Trend();
    }

    // ------------------------------------------------------------------
    // GetLastSwingLow / GetLastSwingHigh — exposed for risk manager
    // These are the structural SL placement levels
    // ------------------------------------------------------------------

    double GetLastH4SwingLow()
    {
        m_ms.SetFractalBars(5);
        m_ms.SetLookback(60);
        return m_ms.GetLastSwingLow(m_symbol, PERIOD_H4);
    }

    double GetLastH4SwingHigh()
    {
        m_ms.SetFractalBars(5);
        m_ms.SetLookback(60);
        return m_ms.GetLastSwingHigh(m_symbol, PERIOD_H4);
    }

    double GetPreviousH4HL()
    {
        m_ms.SetFractalBars(5);
        m_ms.SetLookback(60);
        return m_ms.GetPreviousSwingLow(m_symbol, PERIOD_H4);
    }

    double GetPreviousH4LH()
    {
        m_ms.SetFractalBars(5);
        m_ms.SetLookback(60);
        return m_ms.GetPreviousSwingHigh(m_symbol, PERIOD_H4);
    }

    // ------------------------------------------------------------------
    // GetH4EMA200 — returns current H4 200 EMA value
    // Used by exit manager for EMA flip detection
    // ------------------------------------------------------------------

    double GetH4EMA200()
    {
        if(m_ema200_h4_handle == INVALID_HANDLE)
            return 0.0;

        double ema_buf[];
        ArraySetAsSeries(ema_buf, true);
        if(CopyBuffer(m_ema200_h4_handle, 0, 1, 1, ema_buf) <= 0)
            return 0.0;

        return ema_buf[0];
    }
};

#endif // H4DIRECTIONSYSTEM_MQH
