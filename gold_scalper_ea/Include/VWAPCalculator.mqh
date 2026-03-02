//+------------------------------------------------------------------+
//| VWAPCalculator.mqh — Custom session-reset VWAP implementation     |
//| Uses tick volume as proxy for real volume. Resets at session open. |
//+------------------------------------------------------------------+
#ifndef VWAPCALCULATOR_MQH
#define VWAPCALCULATOR_MQH

#include "Constants.mqh"

class CVWAPCalculator
{
private:
    double   m_cum_pv;         // cumulative (price * volume)
    double   m_cum_vol;        // cumulative volume
    double   m_vwap;
    datetime m_session_start;
    int      m_session_start_hour;  // UTC hour when VWAP resets

    void ResetAccumulation()
    {
        m_cum_pv  = 0.0;
        m_cum_vol = 0.0;
        m_vwap    = 0.0;
    }

    datetime GetSessionStartTime()
    {
        MqlDateTime dt;
        TimeGMT(dt);

        // Determine which session we're in and its start hour
        int hour = dt.hour;
        int reset_hour;

        if(hour >= SESSION_OVERLAP_START && hour < SESSION_NY_CLOSE_END)
            reset_hour = SESSION_OVERLAP_START;    // NY session VWAP
        else if(hour >= SESSION_LONDON_OPEN_START && hour < SESSION_OVERLAP_START)
            reset_hour = SESSION_LONDON_OPEN_START; // London session VWAP
        else
            reset_hour = SESSION_ASIAN_START;       // Asian session VWAP

        dt.hour = reset_hour;
        dt.min  = 0;
        dt.sec  = 0;
        return StructToTime(dt);
    }

public:
    CVWAPCalculator()
    {
        m_cum_pv  = 0.0;
        m_cum_vol = 0.0;
        m_vwap    = 0.0;
        m_session_start = 0;
        m_session_start_hour = SESSION_LONDON_OPEN_START;
    }

    // ------------------------------------------------------------------
    // Update — call on each new bar or tick
    // ------------------------------------------------------------------

    void Update(string symbol)
    {
        datetime current_session = GetSessionStartTime();

        // Check for session reset
        if(current_session != m_session_start)
        {
            ResetAccumulation();
            m_session_start = current_session;

            // Backfill from session start to now
            ENUM_TIMEFRAMES tf = PERIOD_M1;
            int bars = iBars(symbol, tf);
            datetime start = m_session_start;

            for(int i = bars - 1; i >= 0; i--)
            {
                datetime bar_time = iTime(symbol, tf, i);
                if(bar_time < start) continue;

                double typical = (iHigh(symbol, tf, i) + iLow(symbol, tf, i) + iClose(symbol, tf, i)) / 3.0;
                long   vol = iTickVolume(symbol, tf, i);

                if(vol > 0)
                {
                    m_cum_pv  += typical * vol;
                    m_cum_vol += vol;
                }
            }

            if(m_cum_vol > 0)
                m_vwap = m_cum_pv / m_cum_vol;
        }
        else
        {
            // Incremental update with latest bar
            double typical = (iHigh(symbol, PERIOD_M1, 0) +
                             iLow(symbol, PERIOD_M1, 0) +
                             iClose(symbol, PERIOD_M1, 0)) / 3.0;
            long vol = iTickVolume(symbol, PERIOD_M1, 0);

            if(vol > 0)
            {
                m_cum_pv  += typical * vol;
                m_cum_vol += vol;
                m_vwap = m_cum_pv / m_cum_vol;
            }
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    double GetVWAP(string symbol)
    {
        return m_vwap;
    }

    double GetDistanceFromVWAP(string symbol)
    {
        if(m_vwap == 0.0) return 0.0;
        double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
        return bid - m_vwap;
    }

    double GetDistanceInATR(string symbol, double atr_value)
    {
        if(atr_value == 0.0 || m_vwap == 0.0) return 0.0;
        double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
        return (bid - m_vwap) / atr_value;
    }

    bool IsPriceAboveVWAP(string symbol)
    {
        if(m_vwap == 0.0) return false;
        return SymbolInfoDouble(symbol, SYMBOL_BID) > m_vwap;
    }
};

#endif // VWAPCALCULATOR_MQH
