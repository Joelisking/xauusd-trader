//+------------------------------------------------------------------+
//| H1ExecutionEngine.mqh — H1 entry signal evaluation               |
//| All 7 conditions must be met for a valid entry signal            |
//| Evaluated on H1 candle close — not on every tick                 |
//+------------------------------------------------------------------+
#ifndef H1EXECUTIONENGINE_MQH
#define H1EXECUTIONENGINE_MQH

#include "Constants.mqh"
#include "NewsShield.mqh"
#include "DXYFilter.mqh"

class CH1ExecutionEngine
{
private:
    // Indicator handles
    int     m_ema50_h1_handle;      // H1 50 EMA — pullback target
    int     m_rsi14_h1_handle;      // H1 RSI(14) — pullback validation
    int     m_vol_ma20_h1_handle;   // Volume moving average (20 bars) — placeholder

    // AI supply/demand zone data (updated from last AI response)
    double  m_ai_sd_zone_upper;     // Upper bound of AI S/D zone
    double  m_ai_sd_zone_lower;     // Lower bound of AI S/D zone
    bool    m_ai_sd_zone_valid;

    // EA inputs cached locally
    int     m_rsi_low;              // 42
    int     m_rsi_high;             // 55
    double  m_volume_spike_mult;    // 1.5
    int     m_news_blackout_hours;  // 4

    string  m_symbol;

    // References to shared filters (set in Init — not owned)
    CNewsShield  *m_news_shield;
    CDXYFilter   *m_dxy_filter;

    // Last signal reason for logging
    string  m_last_fail_reason;

    // Cache — last H1 candle evaluated
    datetime m_last_h1_candle;
    bool     m_last_signal_result;
    ENUM_DIRECTION m_last_signal_direction;

    // ------------------------------------------------------------------
    // Condition 1: H4 trend confirmed AND AI trend score >= threshold
    // This is passed in as a parameter — checked by caller
    // (Included here for logging completeness)
    // ------------------------------------------------------------------

    // Condition handled by caller — H4 direction must be valid

    // ------------------------------------------------------------------
    // Condition 2: Price pulls back to H1 50 EMA or AI S/D zone
    // ------------------------------------------------------------------

    bool CheckPullbackToEMAOrZone(ENUM_DIRECTION h4_direction)
    {
        double current_price = iClose(m_symbol, PERIOD_H1, 1);   // Last closed H1 bar
        double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
        double pip   = point * 10;  // 1 pip for Gold = 10 points

        // Get H1 50 EMA value at last closed bar
        if(m_ema50_h1_handle == INVALID_HANDLE)
        {
            m_last_fail_reason = "EMA50 handle invalid";
            return false;
        }

        double ema50_buf[];
        ArraySetAsSeries(ema50_buf, true);
        if(CopyBuffer(m_ema50_h1_handle, 0, 1, 1, ema50_buf) <= 0)
        {
            m_last_fail_reason = "Cannot read H1 EMA50";
            return false;
        }

        double ema50 = ema50_buf[0];

        // Tolerance: price within 15 pips of 50 EMA counts as "at the EMA"
        double ema_tolerance = 15.0 * pip;

        bool near_ema = (MathAbs(current_price - ema50) <= ema_tolerance);

        // For bull pullback: price must be at or just below 50 EMA and recently
        // pulling back into it — i.e., price <= ema50 + tolerance
        bool ema_pullback_valid = false;
        if(h4_direction == DIRECTION_BULL)
            ema_pullback_valid = (current_price <= ema50 + ema_tolerance &&
                                   current_price >= ema50 - ema_tolerance * 2.0);
        else if(h4_direction == DIRECTION_BEAR)
            ema_pullback_valid = (current_price >= ema50 - ema_tolerance &&
                                   current_price <= ema50 + ema_tolerance * 2.0);

        if(ema_pullback_valid)
            return true;

        // Check AI supply/demand zone
        if(m_ai_sd_zone_valid &&
           current_price >= m_ai_sd_zone_lower &&
           current_price <= m_ai_sd_zone_upper)
            return true;

        m_last_fail_reason = StringFormat(
            "No pullback: price=%.2f EMA50=%.2f zone=[%.2f,%.2f]",
            current_price, ema50, m_ai_sd_zone_lower, m_ai_sd_zone_upper);

        return false;
    }

    // ------------------------------------------------------------------
    // Condition 3: RSI(14) on H1 has pulled back to 42-55 range
    // This confirms trend continuation, not exhaustion reversal
    // ------------------------------------------------------------------

    bool CheckRSIPullback()
    {
        if(m_rsi14_h1_handle == INVALID_HANDLE)
        {
            m_last_fail_reason = "RSI14 handle invalid";
            return false;
        }

        double rsi_buf[];
        ArraySetAsSeries(rsi_buf, true);
        if(CopyBuffer(m_rsi14_h1_handle, 0, 1, 1, rsi_buf) <= 0)
        {
            m_last_fail_reason = "Cannot read H1 RSI14";
            return false;
        }

        double rsi = rsi_buf[0];

        if(rsi >= m_rsi_low && rsi <= m_rsi_high)
            return true;

        m_last_fail_reason = StringFormat("RSI %.1f not in [%d,%d]", rsi, m_rsi_low, m_rsi_high);
        return false;
    }

    // ------------------------------------------------------------------
    // Condition 4: Confirmation candle closes BACK in H4 trend direction
    // Last closed H1 bar body must close in the trend direction after
    // the pullback, signalling resumption of the trend.
    // ------------------------------------------------------------------

    bool CheckConfirmationCandle(ENUM_DIRECTION h4_direction)
    {
        double open1  = iOpen(m_symbol,  PERIOD_H1, 1);
        double close1 = iClose(m_symbol, PERIOD_H1, 1);

        double body = close1 - open1;   // Positive = bull candle, negative = bear

        // Minimum body size: 30% of the full bar range
        double high1 = iHigh(m_symbol, PERIOD_H1, 1);
        double low1  = iLow(m_symbol,  PERIOD_H1, 1);
        double range = high1 - low1;

        if(range <= 0)
        {
            m_last_fail_reason = "Confirmation candle has zero range";
            return false;
        }

        double body_ratio = MathAbs(body) / range;

        if(body_ratio < 0.30)
        {
            m_last_fail_reason = StringFormat("Confirmation candle body ratio %.2f < 0.30", body_ratio);
            return false;
        }

        if(h4_direction == DIRECTION_BULL && body > 0)
            return true;

        if(h4_direction == DIRECTION_BEAR && body < 0)
            return true;

        m_last_fail_reason = StringFormat(
            "Candle body=%.2f does not close in direction %d", body, h4_direction);
        return false;
    }

    // ------------------------------------------------------------------
    // Condition 5: Volume spike — tick volume > 1.5x 20-bar average
    // Institutional entry proxy: large volume on the confirmation candle
    // ------------------------------------------------------------------

    bool CheckVolumeSpike()
    {
        // Tick volumes from the last closed H1 bar
        long vol_current = iVolume(m_symbol, PERIOD_H1, 1);
        if(vol_current <= 0)
        {
            // Volume data not available — pass condition silently
            // (Some brokers don't provide tick volume on all timeframes)
            return true;
        }

        // Calculate 20-bar average of completed bars (bars 2-21)
        long vol_sum = 0;
        int bars_counted = 0;
        for(int i = 2; i <= 21; i++)
        {
            long v = iVolume(m_symbol, PERIOD_H1, i);
            if(v > 0)
            {
                vol_sum += v;
                bars_counted++;
            }
        }

        if(bars_counted < 10)
        {
            // Not enough history — pass
            return true;
        }

        double vol_avg = (double)vol_sum / bars_counted;

        if((double)vol_current >= vol_avg * m_volume_spike_mult)
            return true;

        m_last_fail_reason = StringFormat(
            "Volume spike fail: current=%d avg=%.0f required=%.0f",
            vol_current, vol_avg, vol_avg * m_volume_spike_mult);
        return false;
    }

    // ------------------------------------------------------------------
    // Condition 6: No major news within m_news_blackout_hours
    // ------------------------------------------------------------------

    bool CheckNoNewsBlackout()
    {
        if(m_news_shield == NULL)
            return true;   // Shield not connected — allow

        if(m_news_shield->IsEventWithinHours(m_news_blackout_hours))
        {
            m_last_fail_reason = StringFormat(
                "News blackout: event within %d hours", m_news_blackout_hours);
            return false;
        }

        if(m_news_shield->BlocksNewEntries())
        {
            m_last_fail_reason = "NewsShield blocking entries";
            return false;
        }

        return true;
    }

    // ------------------------------------------------------------------
    // Condition 7: DXY not in strong opposing move
    // ------------------------------------------------------------------

    bool CheckDXYAlignment(ENUM_DIRECTION h4_direction)
    {
        if(m_dxy_filter == NULL)
            return true;   // Filter not connected — allow

        if(m_dxy_filter->IsMacroHeadwind(h4_direction))
        {
            // Also check if it's a 3-candle strong move — only block on that
            if(m_dxy_filter->HasThreeCandleRally(h4_direction))
            {
                m_last_fail_reason = "DXY 3-candle headwind against entry direction";
                return false;
            }
            // Mild headwind — allow entry but log warning
            PrintFormat("[H1Exec] WARNING: DXY headwind for direction %d (mild)", h4_direction);
        }

        return true;
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    CH1ExecutionEngine()
    {
        m_ema50_h1_handle     = INVALID_HANDLE;
        m_rsi14_h1_handle     = INVALID_HANDLE;
        m_vol_ma20_h1_handle  = INVALID_HANDLE;
        m_ai_sd_zone_upper    = 0.0;
        m_ai_sd_zone_lower    = 0.0;
        m_ai_sd_zone_valid    = false;
        m_rsi_low             = 42;
        m_rsi_high            = 55;
        m_volume_spike_mult   = 1.5;
        m_news_blackout_hours = 4;
        m_symbol              = _Symbol;
        m_news_shield         = NULL;
        m_dxy_filter          = NULL;
        m_last_fail_reason    = "";
        m_last_h1_candle      = 0;
        m_last_signal_result  = false;
        m_last_signal_direction = DIRECTION_NONE;
    }

    ~CH1ExecutionEngine()
    {
        if(m_ema50_h1_handle    != INVALID_HANDLE) IndicatorRelease(m_ema50_h1_handle);
        if(m_rsi14_h1_handle    != INVALID_HANDLE) IndicatorRelease(m_rsi14_h1_handle);
    }

    // ------------------------------------------------------------------
    // Initialise — must be called from OnInit()
    // ------------------------------------------------------------------

    bool Init(string symbol,
              int rsi_low, int rsi_high,
              double volume_spike_mult,
              int news_blackout_hours,
              CNewsShield *news_shield,
              CDXYFilter  *dxy_filter)
    {
        m_symbol              = symbol;
        m_rsi_low             = rsi_low;
        m_rsi_high            = rsi_high;
        m_volume_spike_mult   = volume_spike_mult;
        m_news_blackout_hours = news_blackout_hours;
        m_news_shield         = news_shield;
        m_dxy_filter          = dxy_filter;

        // Create H1 indicators
        m_ema50_h1_handle = iMA(symbol, PERIOD_H1, 50, 0, MODE_EMA, PRICE_CLOSE);
        if(m_ema50_h1_handle == INVALID_HANDLE)
        {
            Print("[H1Exec] ERROR: Failed to create H1 EMA(50) handle");
            return false;
        }

        m_rsi14_h1_handle = iRSI(symbol, PERIOD_H1, 14, PRICE_CLOSE);
        if(m_rsi14_h1_handle == INVALID_HANDLE)
        {
            Print("[H1Exec] ERROR: Failed to create H1 RSI(14) handle");
            return false;
        }

        Print("[H1Exec] Initialised successfully");
        return true;
    }

    // ------------------------------------------------------------------
    // UpdateAIZone — called after AI server response to set S/D zone
    // The AI server encodes supply/demand zones as optional fields
    // ------------------------------------------------------------------

    void UpdateAIZone(double zone_lower, double zone_upper)
    {
        if(zone_lower > 0 && zone_upper > zone_lower)
        {
            m_ai_sd_zone_lower = zone_lower;
            m_ai_sd_zone_upper = zone_upper;
            m_ai_sd_zone_valid = true;
            PrintFormat("[H1Exec] AI S/D zone updated: [%.2f, %.2f]",
                        zone_lower, zone_upper);
        }
        else
        {
            m_ai_sd_zone_valid = false;
        }
    }

    // ------------------------------------------------------------------
    // HasEntry — main public method
    //
    // Evaluates all 7 conditions for an H1 entry signal.
    // Only re-evaluates on H1 candle close (cached between closes).
    // Condition 1 (H4 trend) is passed as a parameter by the caller.
    // ------------------------------------------------------------------

    bool HasEntry(ENUM_DIRECTION h4_direction)
    {
        // No direction = no entry
        if(h4_direction == DIRECTION_NONE)
        {
            m_last_fail_reason = "H4 direction not confirmed";
            return false;
        }

        // Check if a new H1 candle has closed since last evaluation
        datetime current_h1_candle = iTime(m_symbol, PERIOD_H1, 1);
        bool h1_closed = (current_h1_candle != m_last_h1_candle);

        // Return cached result if same candle and same direction
        if(!h1_closed &&
           m_last_h1_candle != 0 &&
           m_last_signal_direction == h4_direction)
        {
            return m_last_signal_result;
        }

        // Evaluate all 7 conditions
        m_last_fail_reason = "";

        // Condition 1: H4 trend already confirmed by caller (h4_direction != NONE)

        // Condition 2: Pullback to EMA or S/D zone
        if(!CheckPullbackToEMAOrZone(h4_direction))
        {
            PrintFormat("[H1Exec] FAIL cond2: %s", m_last_fail_reason);
            m_last_h1_candle        = current_h1_candle;
            m_last_signal_result    = false;
            m_last_signal_direction = h4_direction;
            return false;
        }

        // Condition 3: RSI pullback to 42-55
        if(!CheckRSIPullback())
        {
            PrintFormat("[H1Exec] FAIL cond3: %s", m_last_fail_reason);
            m_last_h1_candle        = current_h1_candle;
            m_last_signal_result    = false;
            m_last_signal_direction = h4_direction;
            return false;
        }

        // Condition 4: Confirmation candle
        if(!CheckConfirmationCandle(h4_direction))
        {
            PrintFormat("[H1Exec] FAIL cond4: %s", m_last_fail_reason);
            m_last_h1_candle        = current_h1_candle;
            m_last_signal_result    = false;
            m_last_signal_direction = h4_direction;
            return false;
        }

        // Condition 5: Volume spike
        if(!CheckVolumeSpike())
        {
            PrintFormat("[H1Exec] FAIL cond5: %s", m_last_fail_reason);
            m_last_h1_candle        = current_h1_candle;
            m_last_signal_result    = false;
            m_last_signal_direction = h4_direction;
            return false;
        }

        // Condition 6: No news blackout
        if(!CheckNoNewsBlackout())
        {
            PrintFormat("[H1Exec] FAIL cond6: %s", m_last_fail_reason);
            m_last_h1_candle        = current_h1_candle;
            m_last_signal_result    = false;
            m_last_signal_direction = h4_direction;
            return false;
        }

        // Condition 7: DXY alignment
        if(!CheckDXYAlignment(h4_direction))
        {
            PrintFormat("[H1Exec] FAIL cond7: %s", m_last_fail_reason);
            m_last_h1_candle        = current_h1_candle;
            m_last_signal_result    = false;
            m_last_signal_direction = h4_direction;
            return false;
        }

        // All 7 conditions met
        PrintFormat("[H1Exec] SIGNAL confirmed for direction=%d", h4_direction);
        m_last_h1_candle        = current_h1_candle;
        m_last_signal_result    = true;
        m_last_signal_direction = h4_direction;
        return true;
    }

    // ------------------------------------------------------------------
    // Get the H1 50 EMA value — used by risk manager for SL placement
    // ------------------------------------------------------------------

    double GetH1EMA50()
    {
        if(m_ema50_h1_handle == INVALID_HANDLE)
            return 0.0;

        double buf[];
        ArraySetAsSeries(buf, true);
        if(CopyBuffer(m_ema50_h1_handle, 0, 1, 1, buf) <= 0)
            return 0.0;

        return buf[0];
    }

    // ------------------------------------------------------------------
    // Get current H1 RSI — used for position monitoring
    // ------------------------------------------------------------------

    double GetH1RSI()
    {
        if(m_rsi14_h1_handle == INVALID_HANDLE)
            return 50.0;

        double buf[];
        ArraySetAsSeries(buf, true);
        if(CopyBuffer(m_rsi14_h1_handle, 0, 0, 1, buf) <= 0)
            return 50.0;

        return buf[0];
    }

    // ------------------------------------------------------------------
    // Diagnostic
    // ------------------------------------------------------------------

    string GetLastFailReason() { return m_last_fail_reason; }
};

#endif // H1EXECUTIONENGINE_MQH
