//+------------------------------------------------------------------+
//| EntryLayer.mqh — M1 cascade entry system for Gold Scalper        |
//|                                                                   |
//| Manages the 4-entry cascade state machine.                        |
//|                                                                   |
//| State machine:                                                    |
//|   IDLE -> PILOT -> CORE -> ADD -> MAX -> (exits handled by        |
//|                                           ExitManager)            |
//|   Any state -> CANCELLED if Pilot goes negative                   |
//|                                                                   |
//| Entry conditions:                                                 |
//|   Entry 1 (Pilot): M1 pullback to 21 EMA + rejection + RSI gate  |
//|   Entry 2 (Core):  Next M1 candle closes in direction +           |
//|                    Pilot must be positive                         |
//|   Entry 3 (Add):   Momentum candle (body > 70% of range)         |
//|   Entry 4 (Max):   AI score >= 82 AND London-NY overlap           |
//|                                                                   |
//| CRITICAL: If Pilot is negative before any subsequent entry        |
//|           triggers, the ENTIRE cascade is cancelled immediately.  |
//+------------------------------------------------------------------+
#ifndef ENTRYLAYER_MQH
#define ENTRYLAYER_MQH

#include "Constants.mqh"
#include "CandlePatterns.mqh"
#include "SessionManager.mqh"
#include <Trade\Trade.mqh>

class CEntryLayer
{
private:
    // ------------------------------------------------------------------
    // Indicator handles (M1)
    // ------------------------------------------------------------------
    int     m_ema21_m1_handle;
    int     m_rsi7_m1_handle;

    // ------------------------------------------------------------------
    // Dependencies
    // ------------------------------------------------------------------
    CCandlePatterns *m_patterns;
    CSessionManager *m_session;

    // ------------------------------------------------------------------
    // Trade object (for order execution)
    // ------------------------------------------------------------------
    CTrade  m_trade;

    // ------------------------------------------------------------------
    // Cascade state
    // ------------------------------------------------------------------
    ENUM_CASCADE_STATE  m_state;
    ENUM_DIRECTION      m_cascade_direction;
    ulong               m_pilot_ticket;   // Ticket of Entry 1 (Pilot)
    ulong               m_core_ticket;    // Ticket of Entry 2
    ulong               m_add_ticket;     // Ticket of Entry 3
    ulong               m_max_ticket;     // Ticket of Entry 4
    double              m_pilot_open_price;
    double              m_cascade_sl;     // Initial stop loss price
    datetime            m_pilot_open_time;
    datetime            m_last_m1_bar_checked; // Track M1 bars for state transitions

    // ------------------------------------------------------------------
    // Configuration parameters
    // ------------------------------------------------------------------
    string  m_symbol;
    int     m_magic;
    double  m_pip_point;          // 10 * _Point for XAUUSD (1 pip = 0.10)
    int     m_ai_score_for_max;   // Default 82
    int     m_ai_last_score;      // Last AI score, provided externally

    // RSI range thresholds
    double  m_rsi_long_min;       // 38
    double  m_rsi_long_max;       // 55
    double  m_rsi_short_min;      // 45
    double  m_rsi_short_max;      // 62

    // EMA proximity threshold: how close to EMA counts as "touch"
    // Expressed as a fraction of ATR(14) on M1
    double  m_ema_proximity_atr_fraction;  // Default 0.3

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    double PipToPrice(double pips)
    {
        return pips * m_pip_point;
    }

    double PriceToPips(double price_diff)
    {
        if(m_pip_point == 0.0) return 0.0;
        return MathAbs(price_diff) / m_pip_point;
    }

    // Get M1 ATR to calibrate EMA proximity tolerance
    double GetM1ATR()
    {
        int atr_handle = iATR(m_symbol, PERIOD_M1, 14);
        if(atr_handle == INVALID_HANDLE) return 0.0;
        double atr[1];
        CopyBuffer(atr_handle, 0, 1, 1, atr);
        IndicatorRelease(atr_handle);
        return atr[0];
    }

    // ------------------------------------------------------------------
    // Check M1 pullback to 21 EMA
    //
    // "Pullback" means:
    //   - The candle's low (for long) or high (for short) touched the EMA
    //     within a tolerance of (ATR * m_ema_proximity_atr_fraction)
    //   - The candle itself closed back in the direction (bullish for long,
    //     bearish for short)
    // ------------------------------------------------------------------
    bool IsAtEMAPullback(ENUM_DIRECTION direction)
    {
        double ema21[1];
        if(CopyBuffer(m_ema21_m1_handle, 0, 1, 1, ema21) <= 0)
            return false;

        double atr    = GetM1ATR();
        double tol    = (atr > 0) ? atr * m_ema_proximity_atr_fraction : PipToPrice(0.5);
        double low1   = iLow(m_symbol,  PERIOD_M1, 1);
        double high1  = iHigh(m_symbol, PERIOD_M1, 1);
        double close1 = iClose(m_symbol, PERIOD_M1, 1);
        double open1  = iOpen(m_symbol,  PERIOD_M1, 1);

        if(direction == DIRECTION_BULL)
        {
            // Candle low must have touched or come within tolerance of 21 EMA
            // Candle must close bullish (close > open)
            bool touched_ema = (low1 <= ema21[0] + tol) && (low1 >= ema21[0] - tol * 2.0);
            bool closed_bull = (close1 > open1);
            return (touched_ema && closed_bull);
        }
        else if(direction == DIRECTION_BEAR)
        {
            // Candle high must have touched or come within tolerance of 21 EMA
            // Candle must close bearish (close < open)
            bool touched_ema = (high1 >= ema21[0] - tol) && (high1 <= ema21[0] + tol * 2.0);
            bool closed_bear = (close1 < open1);
            return (touched_ema && closed_bear);
        }

        return false;
    }

    // ------------------------------------------------------------------
    // Check rejection candle pattern on M1 bar 1 (last closed bar)
    // For longs: hammer, bullish engulfing, bullish pin bar
    // For shorts: shooting star, bearish engulfing, bearish pin bar
    // ------------------------------------------------------------------
    bool IsRejectionCandle(ENUM_DIRECTION direction)
    {
        ENUM_CANDLE_PATTERN pattern = m_patterns->Detect(m_symbol, PERIOD_M1, 1);

        if(direction == DIRECTION_BULL)
        {
            return (pattern == PATTERN_HAMMER         ||
                    pattern == PATTERN_ENGULFING_BULL  ||
                    pattern == PATTERN_PIN_BAR_BULL    ||
                    pattern == PATTERN_MORNING_STAR    ||
                    pattern == PATTERN_TWEEZER_BOTTOM);
        }
        else if(direction == DIRECTION_BEAR)
        {
            return (pattern == PATTERN_SHOOTING_STAR  ||
                    pattern == PATTERN_ENGULFING_BEAR  ||
                    pattern == PATTERN_PIN_BAR_BEAR    ||
                    pattern == PATTERN_EVENING_STAR    ||
                    pattern == PATTERN_TWEEZER_TOP);
        }

        return false;
    }

    // ------------------------------------------------------------------
    // Check RSI(7) range on M1
    // Longs:  38-55 (not overbought)
    // Shorts: 45-62 (not oversold)
    // ------------------------------------------------------------------
    bool IsRSIInRange(ENUM_DIRECTION direction)
    {
        double rsi[1];
        if(CopyBuffer(m_rsi7_m1_handle, 0, 1, 1, rsi) <= 0)
            return false;

        if(direction == DIRECTION_BULL)
            return (rsi[0] >= m_rsi_long_min && rsi[0] <= m_rsi_long_max);
        else if(direction == DIRECTION_BEAR)
            return (rsi[0] >= m_rsi_short_min && rsi[0] <= m_rsi_short_max);

        return false;
    }

    // ------------------------------------------------------------------
    // Check if last closed M1 candle closed in trade direction
    // (Used for Entry 2 / Core trigger)
    // ------------------------------------------------------------------
    bool CandleClosedInDirection(ENUM_DIRECTION direction)
    {
        double close = iClose(m_symbol, PERIOD_M1, 1);
        double open  = iOpen(m_symbol,  PERIOD_M1, 1);

        if(direction == DIRECTION_BULL) return (close > open);
        if(direction == DIRECTION_BEAR) return (close < open);
        return false;
    }

    // ------------------------------------------------------------------
    // Check if last closed M1 candle is a momentum candle
    // Momentum: body > 70% of total range
    // (Used for Entry 3 / Add trigger)
    // ------------------------------------------------------------------
    bool IsMomentumCandle()
    {
        double high  = iHigh(m_symbol,  PERIOD_M1, 1);
        double low   = iLow(m_symbol,   PERIOD_M1, 1);
        double close = iClose(m_symbol, PERIOD_M1, 1);
        double open  = iOpen(m_symbol,  PERIOD_M1, 1);

        double range = high - low;
        if(range <= 0.0) return false;

        double body = MathAbs(close - open);
        return (body / range >= 0.70);
    }

    // ------------------------------------------------------------------
    // Check if Pilot position is currently in positive territory
    // Returns true if Pilot P&L > 0
    // ------------------------------------------------------------------
    bool IsPilotPositive()
    {
        if(m_pilot_ticket == 0) return false;

        if(!PositionSelectByTicket(m_pilot_ticket))
            return false;

        return (PositionGetDouble(POSITION_PROFIT) > 0.0);
    }

    // ------------------------------------------------------------------
    // Check if Pilot still exists (not been stopped out)
    // ------------------------------------------------------------------
    bool IsPilotOpen()
    {
        if(m_pilot_ticket == 0) return false;
        return PositionSelectByTicket(m_pilot_ticket);
    }

    // ------------------------------------------------------------------
    // Find M1 swing low (for long SL) or swing high (for short SL)
    // Looks back m_sl_lookback bars to find the most recent swing point
    // ------------------------------------------------------------------
    double FindM1SwingLow(int lookback = 10)
    {
        double lowest = iLow(m_symbol, PERIOD_M1, 1);
        for(int i = 1; i <= lookback; i++)
        {
            double l = iLow(m_symbol, PERIOD_M1, i);
            if(l < lowest) lowest = l;
        }
        return lowest;
    }

    double FindM1SwingHigh(int lookback = 10)
    {
        double highest = iHigh(m_symbol, PERIOD_M1, 1);
        for(int i = 1; i <= lookback; i++)
        {
            double h = iHigh(m_symbol, PERIOD_M1, i);
            if(h > highest) highest = h;
        }
        return highest;
    }

    // ------------------------------------------------------------------
    // Execute a market order
    // Returns ticket on success, 0 on failure
    // ------------------------------------------------------------------
    ulong ExecuteOrder(ENUM_DIRECTION direction, double lots, double sl_price, double comment_lot)
    {
        double ask = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
        double bid = SymbolInfoDouble(m_symbol, SYMBOL_BID);

        string comment = StringFormat("Scalper cascade L%.2f", comment_lot);

        bool ok = false;
        if(direction == DIRECTION_BULL)
            ok = m_trade.Buy(lots, m_symbol, ask, sl_price, 0.0, comment);
        else if(direction == DIRECTION_BEAR)
            ok = m_trade.Sell(lots, m_symbol, bid, sl_price, 0.0, comment);

        if(ok)
            return m_trade.ResultOrder();

        PrintFormat("[EntryLayer] Order FAILED: %s", m_trade.ResultComment());
        return 0;
    }

    // ------------------------------------------------------------------
    // Cancel entire cascade — close all open positions from this cascade
    // ------------------------------------------------------------------
    void CancelCascade(string reason)
    {
        PrintFormat("[EntryLayer] CASCADE CANCELLED: %s", reason);

        // Close all cascade positions that are still open
        ulong tickets[4];
        tickets[0] = m_pilot_ticket;
        tickets[1] = m_core_ticket;
        tickets[2] = m_add_ticket;
        tickets[3] = m_max_ticket;

        for(int i = 0; i < 4; i++)
        {
            if(tickets[i] != 0 && PositionSelectByTicket(tickets[i]))
            {
                m_trade.PositionClose(tickets[i]);
            }
        }

        Reset();
        m_state = CASCADE_CANCELLED;
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------
    CEntryLayer()
    {
        m_ema21_m1_handle  = INVALID_HANDLE;
        m_rsi7_m1_handle   = INVALID_HANDLE;
        m_patterns         = NULL;
        m_session          = NULL;

        m_state             = CASCADE_IDLE;
        m_cascade_direction = DIRECTION_NONE;
        m_pilot_ticket      = 0;
        m_core_ticket       = 0;
        m_add_ticket        = 0;
        m_max_ticket        = 0;
        m_pilot_open_price  = 0.0;
        m_cascade_sl        = 0.0;
        m_pilot_open_time   = 0;
        m_last_m1_bar_checked = 0;

        m_symbol            = _Symbol;
        m_magic             = MAGIC_SCALPER;
        m_pip_point         = 0.0;
        m_ai_score_for_max  = 82;
        m_ai_last_score     = 0;

        m_rsi_long_min      = 38.0;
        m_rsi_long_max      = 55.0;
        m_rsi_short_min     = 45.0;
        m_rsi_short_max     = 62.0;

        m_ema_proximity_atr_fraction = 0.3;
    }

    // ------------------------------------------------------------------
    // Destructor
    // ------------------------------------------------------------------
    ~CEntryLayer()
    {
        Deinit();
    }

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------
    bool Init(string           symbol,
              int              magic,
              CCandlePatterns  *patterns,
              CSessionManager  *session,
              int              ai_score_for_max = 82)
    {
        m_symbol           = symbol;
        m_magic            = magic;
        m_patterns         = patterns;
        m_session          = session;
        m_ai_score_for_max = ai_score_for_max;

        // XAUUSD: 1 pip = $0.10 = 10 points
        m_pip_point = SymbolInfoDouble(symbol, SYMBOL_POINT) * 10.0;

        // Create M1 indicators
        m_ema21_m1_handle = iMA(symbol, PERIOD_M1, 21, 0, MODE_EMA, PRICE_CLOSE);
        m_rsi7_m1_handle  = iRSI(symbol, PERIOD_M1, 7, PRICE_CLOSE);

        if(m_ema21_m1_handle == INVALID_HANDLE || m_rsi7_m1_handle == INVALID_HANDLE)
        {
            Print("[EntryLayer] ERROR: Failed to create indicator handles");
            return false;
        }

        m_trade.SetExpertMagicNumber(magic);
        m_trade.SetDeviationInPoints(30);     // 3 pips slippage tolerance
        m_trade.SetTypeFilling(ORDER_FILLING_FOK);

        Print("[EntryLayer] Initialized successfully");
        return true;
    }

    // ------------------------------------------------------------------
    // Deinit
    // ------------------------------------------------------------------
    void Deinit()
    {
        if(m_ema21_m1_handle != INVALID_HANDLE) { IndicatorRelease(m_ema21_m1_handle); m_ema21_m1_handle = INVALID_HANDLE; }
        if(m_rsi7_m1_handle  != INVALID_HANDLE) { IndicatorRelease(m_rsi7_m1_handle);  m_rsi7_m1_handle  = INVALID_HANDLE; }
    }

    // ------------------------------------------------------------------
    // SetAIScore — called from main EA after receiving AI response
    // ------------------------------------------------------------------
    void SetAIScore(int score) { m_ai_last_score = score; }

    // ------------------------------------------------------------------
    // HasSignal — check M1 entry conditions
    //
    // Returns true if ALL three M1 conditions are met for the given direction:
    //   1. Price has pulled back to 21 EMA
    //   2. Rejection candle present
    //   3. RSI(7) in valid range
    //
    // Only evaluates when NOT already in a cascade (state must be IDLE).
    // ------------------------------------------------------------------
    bool HasSignal(ENUM_DIRECTION direction)
    {
        // Only look for new signals when idle
        if(m_state != CASCADE_IDLE && m_state != CASCADE_CANCELLED)
            return false;

        if(direction == DIRECTION_NONE)
            return false;

        bool pullback  = IsAtEMAPullback(direction);
        bool rejection = IsRejectionCandle(direction);
        bool rsi_ok    = IsRSIInRange(direction);

        if(pullback && rejection && rsi_ok)
        {
            PrintFormat("[EntryLayer] Signal CONFIRMED: dir=%s pullback=%s rejection=%s rsi=%s",
                        (direction == DIRECTION_BULL) ? "BULL" : "BEAR",
                        pullback  ? "YES" : "NO",
                        rejection ? "YES" : "NO",
                        rsi_ok    ? "YES" : "NO");
            return true;
        }

        return false;
    }

    // ------------------------------------------------------------------
    // ExecutePilot — Entry 1: 0.01 lot, SL below M1 swing low
    //
    // Returns true on successful order placement.
    // Transitions state to CASCADE_PILOT.
    // ------------------------------------------------------------------
    bool ExecutePilot(ENUM_DIRECTION direction)
    {
        if(m_state != CASCADE_IDLE && m_state != CASCADE_CANCELLED)
            return false;

        // Calculate stop loss price
        double sl_price;
        double sl_buffer = m_pip_point * 2.0;  // 2 pip buffer beyond swing

        if(direction == DIRECTION_BULL)
        {
            sl_price = FindM1SwingLow(10) - sl_buffer;
        }
        else
        {
            sl_price = FindM1SwingHigh(10) + sl_buffer;
        }

        // Lot size: fixed 0.01 for Pilot
        double lots = 0.01;

        // Normalize to broker's lot step
        double lot_step = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
        lots = MathMax(SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN),
                       MathRound(lots / lot_step) * lot_step);

        ulong ticket = ExecuteOrder(direction, lots, sl_price, lots);
        if(ticket == 0)
            return false;

        m_pilot_ticket      = ticket;
        m_cascade_direction = direction;
        m_cascade_sl        = sl_price;
        m_pilot_open_time   = TimeCurrent();

        // Capture entry price from the position
        if(PositionSelectByTicket(ticket))
            m_pilot_open_price = PositionGetDouble(POSITION_PRICE_OPEN);

        m_state = CASCADE_PILOT;

        PrintFormat("[EntryLayer] PILOT entered: ticket=%d dir=%s lots=%.2f sl=%.5f entry=%.5f",
                    (int)ticket,
                    (direction == DIRECTION_BULL) ? "BUY" : "SELL",
                    lots, sl_price, m_pilot_open_price);
        return true;
    }

    // ------------------------------------------------------------------
    // ManageCascade — call every tick to manage state transitions
    //
    // Handles:
    //   - Pilot negative check (cancel entire cascade)
    //   - Entry 2 (Core) trigger
    //   - Entry 3 (Add) trigger
    //   - Entry 4 (Max) trigger
    // ------------------------------------------------------------------
    void ManageCascade()
    {
        if(m_state == CASCADE_IDLE || m_state == CASCADE_CANCELLED)
            return;

        // ---- Guard: Pilot gone means it was stopped out ----
        // If Pilot ticket no longer selectable, position was closed by broker SL
        if(m_pilot_ticket != 0 && !PositionSelectByTicket(m_pilot_ticket))
        {
            // Pilot hit its stop — cancel everything
            CancelCascade("Pilot SL hit — position closed by broker");
            return;
        }

        // ---- Critical rule: if Pilot is negative, cancel cascade ----
        if(m_state == CASCADE_PILOT || m_state == CASCADE_CORE || m_state == CASCADE_ADD)
        {
            if(!IsPilotPositive())
            {
                // Only cancel if we are waiting to ADD more — not if Pilot
                // just entered (give it a few ticks to settle)
                // Check time: Pilot has been open for at least 30 seconds
                if(TimeCurrent() - m_pilot_open_time > 30)
                {
                    CancelCascade("Pilot is negative — Pilot Rule triggered");
                    return;
                }
            }
        }

        // ---- Only process state transitions on new M1 bar close ----
        datetime current_bar = iTime(m_symbol, PERIOD_M1, 1);
        if(current_bar == m_last_m1_bar_checked)
            return;
        m_last_m1_bar_checked = current_bar;

        // ---- Entry 2: Core (0.02 lots) ----
        if(m_state == CASCADE_PILOT)
        {
            // Condition: next M1 candle closes in direction AND Pilot is positive
            if(CandleClosedInDirection(m_cascade_direction) && IsPilotPositive())
            {
                double lots = 0.02;
                double lot_step = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
                lots = MathMax(SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN),
                               MathRound(lots / lot_step) * lot_step);

                ulong ticket = ExecuteOrder(m_cascade_direction, lots, m_cascade_sl, lots);
                if(ticket != 0)
                {
                    m_core_ticket = ticket;
                    m_state = CASCADE_CORE;
                    PrintFormat("[EntryLayer] CORE entered: ticket=%d lots=%.2f", (int)ticket, lots);
                }
            }
            return;
        }

        // ---- Entry 3: Add (0.02 lots) ----
        if(m_state == CASCADE_CORE)
        {
            // Condition: momentum candle (body > 70% range) AND Pilot positive
            if(IsMomentumCandle() && IsPilotPositive())
            {
                double lots = 0.02;
                double lot_step = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
                lots = MathMax(SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN),
                               MathRound(lots / lot_step) * lot_step);

                ulong ticket = ExecuteOrder(m_cascade_direction, lots, m_cascade_sl, lots);
                if(ticket != 0)
                {
                    m_add_ticket = ticket;
                    m_state = CASCADE_ADD;
                    PrintFormat("[EntryLayer] ADD entered: ticket=%d lots=%.2f", (int)ticket, lots);
                }
            }
            return;
        }

        // ---- Entry 4: Max (0.01 lots) ----
        if(m_state == CASCADE_ADD)
        {
            // Conditions: AI score >= 82 AND London-NY overlap AND Pilot positive
            bool ai_qualifies = (m_ai_last_score >= m_ai_score_for_max);
            bool is_overlap   = m_session->IsOverlap();

            if(ai_qualifies && is_overlap && IsPilotPositive())
            {
                double lots = 0.01;
                double lot_step = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
                lots = MathMax(SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN),
                               MathRound(lots / lot_step) * lot_step);

                ulong ticket = ExecuteOrder(m_cascade_direction, lots, m_cascade_sl, lots);
                if(ticket != 0)
                {
                    m_max_ticket = ticket;
                    m_state = CASCADE_MAX;
                    PrintFormat("[EntryLayer] MAX entered: ticket=%d lots=%.2f AI score=%d",
                                (int)ticket, lots, m_ai_last_score);
                }
            }
            // If conditions not met, remain in ADD state (max 3 positions)
            return;
        }

        // In CASCADE_MAX state, ExitManager handles everything
    }

    // ------------------------------------------------------------------
    // Reset — return to IDLE, clear all state
    // ------------------------------------------------------------------
    void Reset()
    {
        m_state             = CASCADE_IDLE;
        m_cascade_direction = DIRECTION_NONE;
        m_pilot_ticket      = 0;
        m_core_ticket       = 0;
        m_add_ticket        = 0;
        m_max_ticket        = 0;
        m_pilot_open_price  = 0.0;
        m_cascade_sl        = 0.0;
        m_pilot_open_time   = 0;
        m_last_m1_bar_checked = 0;
    }

    // ------------------------------------------------------------------
    // Public accessors
    // ------------------------------------------------------------------
    ENUM_CASCADE_STATE  GetCascadeState()     { return m_state; }
    ENUM_DIRECTION      GetCascadeDirection() { return m_cascade_direction; }
    ulong               GetPilotTicket()      { return m_pilot_ticket; }
    ulong               GetCoreTicket()       { return m_core_ticket; }
    ulong               GetAddTicket()        { return m_add_ticket; }
    ulong               GetMaxTicket()        { return m_max_ticket; }
    double              GetPilotOpenPrice()   { return m_pilot_open_price; }
    double              GetCascadeSL()        { return m_cascade_sl; }
    datetime            GetPilotOpenTime()    { return m_pilot_open_time; }

    bool IsActive()
    {
        return (m_state == CASCADE_PILOT ||
                m_state == CASCADE_CORE  ||
                m_state == CASCADE_ADD   ||
                m_state == CASCADE_MAX);
    }

    // Return all open cascade tickets
    int GetOpenTickets(ulong &tickets[])
    {
        ArrayResize(tickets, 0);
        ulong all[4] = { m_pilot_ticket, m_core_ticket, m_add_ticket, m_max_ticket };
        for(int i = 0; i < 4; i++)
        {
            if(all[i] != 0 && PositionSelectByTicket(all[i]))
            {
                int sz = ArraySize(tickets);
                ArrayResize(tickets, sz + 1);
                tickets[sz] = all[i];
            }
        }
        return ArraySize(tickets);
    }

    // Get total open lots across all cascade positions
    double GetTotalOpenLots()
    {
        double total = 0.0;
        ulong tickets[];
        int count = GetOpenTickets(tickets);
        for(int i = 0; i < count; i++)
        {
            if(PositionSelectByTicket(tickets[i]))
                total += PositionGetDouble(POSITION_VOLUME);
        }
        return total;
    }

    // Get total unrealised PnL across all cascade positions
    double GetTotalPnL()
    {
        double total = 0.0;
        ulong tickets[];
        int count = GetOpenTickets(tickets);
        for(int i = 0; i < count; i++)
        {
            if(PositionSelectByTicket(tickets[i]))
                total += PositionGetDouble(POSITION_PROFIT);
        }
        return total;
    }
};

#endif // ENTRYLAYER_MQH
