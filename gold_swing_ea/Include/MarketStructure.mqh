//+------------------------------------------------------------------+
//| MarketStructure.mqh — Fractal swing point detection, HH/HL/LH/LL |
//| Identifies market structure for both scalper (M5) and swing (H4)  |
//+------------------------------------------------------------------+
#ifndef MARKETSTRUCTURE_MQH
#define MARKETSTRUCTURE_MQH

#include "Constants.mqh"

class CMarketStructure
{
private:
    int     m_fractal_bars;        // 3 or 5 bar fractal
    int     m_lookback;            // bars to scan for swing points
    double  m_swing_highs[];       // detected swing high prices
    double  m_swing_lows[];        // detected swing low prices
    int     m_swing_high_bars[];   // bar indices of swing highs
    int     m_swing_low_bars[];    // bar indices of swing lows

    // ------------------------------------------------------------------
    // Fractal detection
    // ------------------------------------------------------------------

    bool IsFractalHigh(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        // 3-bar fractal: bar is higher than bar-1 and bar+1
        double high_mid  = iHigh(symbol, tf, bar);
        double high_left = iHigh(symbol, tf, bar + 1);
        double high_right = iHigh(symbol, tf, bar - 1);

        if(m_fractal_bars == 5 && bar >= 2)
        {
            double high_left2  = iHigh(symbol, tf, bar + 2);
            double high_right2 = iHigh(symbol, tf, bar - 2);
            return (high_mid > high_left && high_mid > high_right &&
                    high_mid > high_left2 && high_mid > high_right2);
        }

        return (high_mid > high_left && high_mid > high_right);
    }

    bool IsFractalLow(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        double low_mid   = iLow(symbol, tf, bar);
        double low_left  = iLow(symbol, tf, bar + 1);
        double low_right = iLow(symbol, tf, bar - 1);

        if(m_fractal_bars == 5 && bar >= 2)
        {
            double low_left2  = iLow(symbol, tf, bar + 2);
            double low_right2 = iLow(symbol, tf, bar - 2);
            return (low_mid < low_left && low_mid < low_right &&
                    low_mid < low_left2 && low_mid < low_right2);
        }

        return (low_mid < low_left && low_mid < low_right);
    }

    // ------------------------------------------------------------------
    // Scan for swing points
    // ------------------------------------------------------------------

    void ScanSwingPoints(string symbol, ENUM_TIMEFRAMES tf)
    {
        ArrayResize(m_swing_highs, 0);
        ArrayResize(m_swing_lows, 0);
        ArrayResize(m_swing_high_bars, 0);
        ArrayResize(m_swing_low_bars, 0);

        int start = (m_fractal_bars == 5) ? 2 : 1;

        for(int i = start; i < m_lookback - start; i++)
        {
            if(IsFractalHigh(symbol, tf, i))
            {
                int sz = ArraySize(m_swing_highs);
                ArrayResize(m_swing_highs, sz + 1);
                ArrayResize(m_swing_high_bars, sz + 1);
                m_swing_highs[sz] = iHigh(symbol, tf, i);
                m_swing_high_bars[sz] = i;
            }

            if(IsFractalLow(symbol, tf, i))
            {
                int sz = ArraySize(m_swing_lows);
                ArrayResize(m_swing_lows, sz + 1);
                ArrayResize(m_swing_low_bars, sz + 1);
                m_swing_lows[sz] = iLow(symbol, tf, i);
                m_swing_low_bars[sz] = i;
            }
        }
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    CMarketStructure()
    {
        m_fractal_bars = 3;
        m_lookback = 50;
    }

    void SetFractalBars(int bars) { m_fractal_bars = bars; }
    void SetLookback(int bars)    { m_lookback = bars; }

    // ------------------------------------------------------------------
    // Detect swing point type at a given shift
    // ------------------------------------------------------------------

    ENUM_MARKET_STRUCTURE DetectSwingPoint(string symbol, ENUM_TIMEFRAMES tf, int shift)
    {
        ScanSwingPoints(symbol, tf);

        int num_highs = ArraySize(m_swing_highs);
        int num_lows  = ArraySize(m_swing_lows);

        // Need at least 2 swing highs and 2 swing lows
        if(num_highs < 2 || num_lows < 2)
            return STRUCT_NONE;

        // Compare most recent two swing highs
        double latest_high = m_swing_highs[0];
        double prev_high   = m_swing_highs[1];

        // Compare most recent two swing lows
        double latest_low  = m_swing_lows[0];
        double prev_low    = m_swing_lows[1];

        // Determine structure based on most recent swing point type
        // If latest swing high is closer (more recent) than latest swing low
        if(num_highs > 0 && num_lows > 0)
        {
            if(m_swing_high_bars[0] < m_swing_low_bars[0])
            {
                // Most recent point is a high
                if(latest_high > prev_high)
                    return STRUCT_HH;
                else if(latest_high < prev_high)
                    return STRUCT_LH;
            }
            else
            {
                // Most recent point is a low
                if(latest_low > prev_low)
                    return STRUCT_HL;
                else if(latest_low < prev_low)
                    return STRUCT_LL;
            }
        }

        return STRUCT_NONE;
    }

    // ------------------------------------------------------------------
    // Trend detection: requires 2+ consecutive matching structures
    // ------------------------------------------------------------------

    ENUM_DIRECTION GetTrend(string symbol, ENUM_TIMEFRAMES tf, int lookback = 50)
    {
        m_lookback = lookback;
        ScanSwingPoints(symbol, tf);

        int num_highs = ArraySize(m_swing_highs);
        int num_lows  = ArraySize(m_swing_lows);

        if(num_highs < 3 || num_lows < 3)
            return DIRECTION_NONE;

        // Count consecutive HH/HL for bullish
        int bull_count = 0;
        for(int i = 0; i < num_highs - 1 && i < 4; i++)
        {
            if(m_swing_highs[i] > m_swing_highs[i + 1])
                bull_count++;
            else
                break;
        }
        for(int i = 0; i < num_lows - 1 && i < 4; i++)
        {
            if(m_swing_lows[i] > m_swing_lows[i + 1])
                bull_count++;
            else
                break;
        }

        // Count consecutive LL/LH for bearish
        int bear_count = 0;
        for(int i = 0; i < num_lows - 1 && i < 4; i++)
        {
            if(m_swing_lows[i] < m_swing_lows[i + 1])
                bear_count++;
            else
                break;
        }
        for(int i = 0; i < num_highs - 1 && i < 4; i++)
        {
            if(m_swing_highs[i] < m_swing_highs[i + 1])
                bear_count++;
            else
                break;
        }

        // Need 2+ HH + 2+ HL = 4+ bull signals, or 2+ LL + 2+ LH
        if(bull_count >= 4)
            return DIRECTION_BULL;
        if(bear_count >= 4)
            return DIRECTION_BEAR;

        // Relaxed: 2 consecutive in one direction
        if(bull_count >= 2 && bear_count < 2)
            return DIRECTION_BULL;
        if(bear_count >= 2 && bull_count < 2)
            return DIRECTION_BEAR;

        return DIRECTION_NONE;
    }

    // ------------------------------------------------------------------
    // Accessors for swing points (used by exit managers)
    // ------------------------------------------------------------------

    double GetLastSwingHigh(string symbol, ENUM_TIMEFRAMES tf)
    {
        ScanSwingPoints(symbol, tf);
        if(ArraySize(m_swing_highs) > 0)
            return m_swing_highs[0];
        return 0.0;
    }

    double GetLastSwingLow(string symbol, ENUM_TIMEFRAMES tf)
    {
        ScanSwingPoints(symbol, tf);
        if(ArraySize(m_swing_lows) > 0)
            return m_swing_lows[0];
        return 0.0;
    }

    double GetPreviousSwingHigh(string symbol, ENUM_TIMEFRAMES tf)
    {
        ScanSwingPoints(symbol, tf);
        if(ArraySize(m_swing_highs) > 1)
            return m_swing_highs[1];
        return 0.0;
    }

    double GetPreviousSwingLow(string symbol, ENUM_TIMEFRAMES tf)
    {
        ScanSwingPoints(symbol, tf);
        if(ArraySize(m_swing_lows) > 1)
            return m_swing_lows[1];
        return 0.0;
    }
};

#endif // MARKETSTRUCTURE_MQH
