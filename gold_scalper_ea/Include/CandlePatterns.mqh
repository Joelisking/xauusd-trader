//+------------------------------------------------------------------+
//| CandlePatterns.mqh — 14 candlestick pattern recognition           |
//+------------------------------------------------------------------+
#ifndef CANDLEPATTERNS_MQH
#define CANDLEPATTERNS_MQH

#include "Constants.mqh"

class CCandlePatterns
{
private:
    // Body / range ratio thresholds
    double m_doji_body_pct;       // max body/range for doji
    double m_hammer_wick_ratio;   // min lower wick / body for hammer
    double m_engulf_min_pct;      // min engulfing body / prev body
    double m_pinbar_wick_pct;     // min wick / total range for pin bar

    // Helpers
    double Body(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        return MathAbs(iClose(symbol, tf, bar) - iOpen(symbol, tf, bar));
    }

    double Range(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        return iHigh(symbol, tf, bar) - iLow(symbol, tf, bar);
    }

    double UpperWick(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        double close = iClose(symbol, tf, bar);
        double open  = iOpen(symbol, tf, bar);
        return iHigh(symbol, tf, bar) - MathMax(close, open);
    }

    double LowerWick(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        double close = iClose(symbol, tf, bar);
        double open  = iOpen(symbol, tf, bar);
        return MathMin(close, open) - iLow(symbol, tf, bar);
    }

    bool IsBullish(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        return iClose(symbol, tf, bar) > iOpen(symbol, tf, bar);
    }

    bool IsBearish(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        return iClose(symbol, tf, bar) < iOpen(symbol, tf, bar);
    }

    // ------------------------------------------------------------------
    // Pattern checks
    // ------------------------------------------------------------------

    bool IsHammer(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        double range = Range(symbol, tf, bar);
        if(range == 0) return false;
        double body = Body(symbol, tf, bar);
        double lower = LowerWick(symbol, tf, bar);
        double upper = UpperWick(symbol, tf, bar);
        // Lower wick >= 2x body, upper wick small, body in upper third
        return (lower >= body * m_hammer_wick_ratio &&
                upper <= body * 0.5 &&
                body / range >= 0.15);
    }

    bool IsInvertedHammer(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        double range = Range(symbol, tf, bar);
        if(range == 0) return false;
        double body = Body(symbol, tf, bar);
        double lower = LowerWick(symbol, tf, bar);
        double upper = UpperWick(symbol, tf, bar);
        return (upper >= body * m_hammer_wick_ratio &&
                lower <= body * 0.5 &&
                body / range >= 0.15);
    }

    bool IsEngulfingBull(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        if(!IsBullish(symbol, tf, bar) || !IsBearish(symbol, tf, bar + 1))
            return false;
        double curr_body = Body(symbol, tf, bar);
        double prev_body = Body(symbol, tf, bar + 1);
        if(prev_body == 0) return false;
        return (curr_body / prev_body >= m_engulf_min_pct &&
                iClose(symbol, tf, bar) > iOpen(symbol, tf, bar + 1) &&
                iOpen(symbol, tf, bar) < iClose(symbol, tf, bar + 1));
    }

    bool IsEngulfingBear(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        if(!IsBearish(symbol, tf, bar) || !IsBullish(symbol, tf, bar + 1))
            return false;
        double curr_body = Body(symbol, tf, bar);
        double prev_body = Body(symbol, tf, bar + 1);
        if(prev_body == 0) return false;
        return (curr_body / prev_body >= m_engulf_min_pct &&
                iOpen(symbol, tf, bar) > iClose(symbol, tf, bar + 1) &&
                iClose(symbol, tf, bar) < iOpen(symbol, tf, bar + 1));
    }

    bool IsPinBarBull(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        double range = Range(symbol, tf, bar);
        if(range == 0) return false;
        double lower = LowerWick(symbol, tf, bar);
        return (lower / range >= m_pinbar_wick_pct &&
                UpperWick(symbol, tf, bar) / range <= 0.15);
    }

    bool IsPinBarBear(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        double range = Range(symbol, tf, bar);
        if(range == 0) return false;
        double upper = UpperWick(symbol, tf, bar);
        return (upper / range >= m_pinbar_wick_pct &&
                LowerWick(symbol, tf, bar) / range <= 0.15);
    }

    bool IsDoji(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        double range = Range(symbol, tf, bar);
        if(range == 0) return false;
        return (Body(symbol, tf, bar) / range <= m_doji_body_pct);
    }

    bool IsShootingStar(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        // Inverted hammer after an uptrend (at a high)
        if(!IsInvertedHammer(symbol, tf, bar)) return false;
        // Previous candle was bullish
        return IsBullish(symbol, tf, bar + 1);
    }

    bool IsMorningStar(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        // 3-candle pattern: bearish, small body (doji-like), bullish
        if(bar + 2 >= iBars(symbol, tf)) return false;
        if(!IsBearish(symbol, tf, bar + 2)) return false;
        if(!IsBullish(symbol, tf, bar)) return false;

        double mid_range = Range(symbol, tf, bar + 1);
        if(mid_range == 0) return false;
        double mid_body_pct = Body(symbol, tf, bar + 1) / mid_range;

        return (mid_body_pct <= 0.3 &&
                Body(symbol, tf, bar) > Body(symbol, tf, bar + 1));
    }

    bool IsEveningStar(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        if(bar + 2 >= iBars(symbol, tf)) return false;
        if(!IsBullish(symbol, tf, bar + 2)) return false;
        if(!IsBearish(symbol, tf, bar)) return false;

        double mid_range = Range(symbol, tf, bar + 1);
        if(mid_range == 0) return false;
        double mid_body_pct = Body(symbol, tf, bar + 1) / mid_range;

        return (mid_body_pct <= 0.3 &&
                Body(symbol, tf, bar) > Body(symbol, tf, bar + 1));
    }

    bool IsThreeWhiteSoldiers(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        if(bar + 2 >= iBars(symbol, tf)) return false;
        return (IsBullish(symbol, tf, bar) &&
                IsBullish(symbol, tf, bar + 1) &&
                IsBullish(symbol, tf, bar + 2) &&
                iClose(symbol, tf, bar) > iClose(symbol, tf, bar + 1) &&
                iClose(symbol, tf, bar + 1) > iClose(symbol, tf, bar + 2) &&
                Body(symbol, tf, bar) / Range(symbol, tf, bar) >= 0.6 &&
                Body(symbol, tf, bar + 1) / Range(symbol, tf, bar + 1) >= 0.6);
    }

    bool IsThreeBlackCrows(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        if(bar + 2 >= iBars(symbol, tf)) return false;
        return (IsBearish(symbol, tf, bar) &&
                IsBearish(symbol, tf, bar + 1) &&
                IsBearish(symbol, tf, bar + 2) &&
                iClose(symbol, tf, bar) < iClose(symbol, tf, bar + 1) &&
                iClose(symbol, tf, bar + 1) < iClose(symbol, tf, bar + 2) &&
                Body(symbol, tf, bar) / Range(symbol, tf, bar) >= 0.6 &&
                Body(symbol, tf, bar + 1) / Range(symbol, tf, bar + 1) >= 0.6);
    }

    bool IsTweezerTop(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        if(bar + 1 >= iBars(symbol, tf)) return false;
        double diff = MathAbs(iHigh(symbol, tf, bar) - iHigh(symbol, tf, bar + 1));
        double avgRange = (Range(symbol, tf, bar) + Range(symbol, tf, bar + 1)) / 2.0;
        if(avgRange == 0) return false;
        return (diff / avgRange <= 0.05 &&
                IsBullish(symbol, tf, bar + 1) &&
                IsBearish(symbol, tf, bar));
    }

    bool IsTweezerBottom(string symbol, ENUM_TIMEFRAMES tf, int bar)
    {
        if(bar + 1 >= iBars(symbol, tf)) return false;
        double diff = MathAbs(iLow(symbol, tf, bar) - iLow(symbol, tf, bar + 1));
        double avgRange = (Range(symbol, tf, bar) + Range(symbol, tf, bar + 1)) / 2.0;
        if(avgRange == 0) return false;
        return (diff / avgRange <= 0.05 &&
                IsBearish(symbol, tf, bar + 1) &&
                IsBullish(symbol, tf, bar));
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    CCandlePatterns()
    {
        m_doji_body_pct     = 0.10;   // body < 10% of range
        m_hammer_wick_ratio = 2.0;    // lower wick >= 2x body
        m_engulf_min_pct    = 1.0;    // engulfing body >= prev body
        m_pinbar_wick_pct   = 0.65;   // wick >= 65% of total range
    }

    // ------------------------------------------------------------------
    // Main detection — returns dominant pattern at given bar
    // ------------------------------------------------------------------

    ENUM_CANDLE_PATTERN Detect(string symbol, ENUM_TIMEFRAMES tf, int shift)
    {
        // Multi-candle patterns first (higher priority)
        if(IsMorningStar(symbol, tf, shift))       return PATTERN_MORNING_STAR;
        if(IsEveningStar(symbol, tf, shift))       return PATTERN_EVENING_STAR;
        if(IsThreeWhiteSoldiers(symbol, tf, shift)) return PATTERN_THREE_WHITE;
        if(IsThreeBlackCrows(symbol, tf, shift))   return PATTERN_THREE_BLACK;
        if(IsEngulfingBull(symbol, tf, shift))     return PATTERN_ENGULFING_BULL;
        if(IsEngulfingBear(symbol, tf, shift))     return PATTERN_ENGULFING_BEAR;
        if(IsTweezerTop(symbol, tf, shift))        return PATTERN_TWEEZER_TOP;
        if(IsTweezerBottom(symbol, tf, shift))     return PATTERN_TWEEZER_BOTTOM;

        // Single-candle patterns
        if(IsShootingStar(symbol, tf, shift))      return PATTERN_SHOOTING_STAR;
        if(IsPinBarBull(symbol, tf, shift))        return PATTERN_PIN_BAR_BULL;
        if(IsPinBarBear(symbol, tf, shift))        return PATTERN_PIN_BAR_BEAR;
        if(IsHammer(symbol, tf, shift))            return PATTERN_HAMMER;
        if(IsInvertedHammer(symbol, tf, shift))    return PATTERN_INVERTED_HAMMER;
        if(IsDoji(symbol, tf, shift))              return PATTERN_DOJI;

        return PATTERN_NONE;
    }

    // ------------------------------------------------------------------
    // Convenience: is the pattern bullish?
    // ------------------------------------------------------------------

    bool IsBullishPattern(ENUM_CANDLE_PATTERN p)
    {
        return (p == PATTERN_HAMMER ||
                p == PATTERN_ENGULFING_BULL ||
                p == PATTERN_PIN_BAR_BULL ||
                p == PATTERN_MORNING_STAR ||
                p == PATTERN_THREE_WHITE ||
                p == PATTERN_TWEEZER_BOTTOM ||
                p == PATTERN_INVERTED_HAMMER);
    }

    bool IsBearishPattern(ENUM_CANDLE_PATTERN p)
    {
        return (p == PATTERN_SHOOTING_STAR ||
                p == PATTERN_ENGULFING_BEAR ||
                p == PATTERN_PIN_BAR_BEAR ||
                p == PATTERN_EVENING_STAR ||
                p == PATTERN_THREE_BLACK ||
                p == PATTERN_TWEEZER_TOP);
    }
};

#endif // CANDLEPATTERNS_MQH
