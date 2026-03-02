//+------------------------------------------------------------------+
//| ExitManager.mqh — All exit logic for Gold Scalper cascade        |
//|                                                                   |
//| Manages 8 exit types in strict priority order:                   |
//|   1. Hard stop (20 pips max adverse excursion)                   |
//|   2. Direction flip (M5 EMA stack reversal -> close ALL)         |
//|   3. Time exit (15 minutes elapsed -> close at market)           |
//|   4. Initial SL hit (managed by broker order)                    |
//|   5. VWAP rejection (2 consecutive rejection candles -> 50%      |
//|      close, SL to BE)                                            |
//|   6. Primary TP (ATR(14) * 1.5 from Entry 1 price)              |
//|   7. Momentum TP (+12 pips in 4 minutes -> extend to ATR*2.5,   |
//|      SL to BE)                                                    |
//|   8. Trailing stop (12 pips behind after BE set)                 |
//|                                                                   |
//| Breakeven: at +10 pips, move SL to entry price                   |
//+------------------------------------------------------------------+
#ifndef EXITMANAGER_MQH
#define EXITMANAGER_MQH

#include "Constants.mqh"
#include "DirectionLayer.mqh"
#include "EntryLayer.mqh"
#include "VWAPCalculator.mqh"
#include "CandlePatterns.mqh"
#include <Trade\Trade.mqh>

class CExitManager
{
private:
    // ------------------------------------------------------------------
    // Dependencies
    // ------------------------------------------------------------------
    CDirectionLayer *m_direction;
    CEntryLayer     *m_entry;
    CVWAPCalculator *m_vwap;
    CCandlePatterns *m_patterns;

    // ------------------------------------------------------------------
    // Trade object
    // ------------------------------------------------------------------
    CTrade  m_trade;

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------
    string  m_symbol;
    double  m_hard_stop_pips;          // 20
    int     m_max_trade_duration_min;  // 15
    double  m_breakeven_trigger_pips;  // 10
    double  m_trailing_distance_pips;  // 12
    double  m_tp_multiplier;           // 1.5
    double  m_momentum_tp_mult;        // 2.5
    double  m_momentum_trigger_pips;   // 12

    // ------------------------------------------------------------------
    // State tracking
    // ------------------------------------------------------------------
    bool    m_breakeven_set;         // Has SL been moved to BE?
    bool    m_momentum_tp_extended;  // Has TP been extended to ATR*2.5?

    // Momentum TP: track price at the moment +12 pips was detected
    double  m_momentum_start_price;
    datetime m_momentum_start_time;
    bool    m_momentum_tracking;

    // VWAP rejection tracking: count consecutive rejection candles at VWAP
    int     m_vwap_rejection_count;
    datetime m_last_vwap_check_bar;

    // Computed exit levels (set when Pilot enters, updated on momentum)
    double  m_primary_tp;      // Entry1 + ATR * 1.5
    double  m_extended_tp;     // Entry1 + ATR * 2.5 (after momentum)
    double  m_atr_at_entry;    // ATR(14) value when Entry 1 was placed

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    double PipToPrice(double pips)
    {
        return pips * SymbolInfoDouble(m_symbol, SYMBOL_POINT) * 10.0;
    }

    double PriceToPips(double price_diff)
    {
        double pip = SymbolInfoDouble(m_symbol, SYMBOL_POINT) * 10.0;
        if(pip == 0.0) return 0.0;
        return MathAbs(price_diff) / pip;
    }

    double NormalizePrice(double price)
    {
        double tick = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
        if(tick == 0.0) return price;
        return MathRound(price / tick) * tick;
    }

    // Modify SL for a single position by ticket
    bool ModifySL(ulong ticket, double new_sl)
    {
        if(!PositionSelectByTicket(ticket)) return false;

        double current_sl = PositionGetDouble(POSITION_SL);
        double current_tp = PositionGetDouble(POSITION_TP);
        new_sl = NormalizePrice(new_sl);

        // Avoid modifying if already at same level (prevent spam)
        double tick = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
        if(MathAbs(new_sl - current_sl) < tick) return true;

        return m_trade.PositionModify(ticket, new_sl, current_tp);
    }

    // Set SL for ALL open cascade positions
    void SetSLForAll(double new_sl)
    {
        ulong tickets[];
        int count = m_entry->GetOpenTickets(tickets);
        for(int i = 0; i < count; i++)
        {
            if(!ModifySL(tickets[i], new_sl))
            {
                PrintFormat("[ExitManager] ModifySL failed for ticket %d: %s",
                            (int)tickets[i], m_trade.ResultComment());
            }
        }
    }

    // Close a percentage of positions (by total open lots)
    // pct = 0.50 means close 50% of open volume
    void ClosePercentage(double pct)
    {
        ulong tickets[];
        int count = m_entry->GetOpenTickets(tickets);

        double total_lots = m_entry->GetTotalOpenLots();
        double lots_to_close = total_lots * pct;

        // Close smallest positions first, leaving the biggest open
        // Sort by volume (ascending) for clean partial close
        for(int i = 0; i < count && lots_to_close > 0.001; i++)
        {
            if(!PositionSelectByTicket(tickets[i])) continue;

            double pos_vol = PositionGetDouble(POSITION_VOLUME);
            double close_vol = MathMin(pos_vol, lots_to_close);
            double lot_step  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
            close_vol = MathMax(SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN),
                                MathRound(close_vol / lot_step) * lot_step);

            if(close_vol >= pos_vol)
            {
                // Close entire position
                m_trade.PositionClose(tickets[i]);
                lots_to_close -= pos_vol;
            }
            else
            {
                // Partial close
                ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
                double price = (pos_type == POSITION_TYPE_BUY) ?
                               SymbolInfoDouble(m_symbol, SYMBOL_BID) :
                               SymbolInfoDouble(m_symbol, SYMBOL_ASK);
                m_trade.PositionClosePartial(tickets[i], close_vol);
                lots_to_close -= close_vol;
            }
        }
    }

    // Close ALL open cascade positions
    void CloseAll(string reason)
    {
        PrintFormat("[ExitManager] CLOSING ALL: %s", reason);
        ulong tickets[];
        int count = m_entry->GetOpenTickets(tickets);
        for(int i = 0; i < count; i++)
        {
            m_trade.PositionClose(tickets[i]);
        }
        // Reset cascade state after full close
        m_entry->Reset();
        ResetExitState();
    }

    // ------------------------------------------------------------------
    // Reset internal exit tracking state
    // ------------------------------------------------------------------
    void ResetExitState()
    {
        m_breakeven_set         = false;
        m_momentum_tp_extended  = false;
        m_momentum_start_price  = 0.0;
        m_momentum_start_time   = 0;
        m_momentum_tracking     = false;
        m_vwap_rejection_count  = 0;
        m_last_vwap_check_bar   = 0;
        m_primary_tp            = 0.0;
        m_extended_tp           = 0.0;
        m_atr_at_entry          = 0.0;
    }

    // ------------------------------------------------------------------
    // Get current price relative to position type
    // For BUY: current bid is the exit price
    // For SELL: current ask is the exit price
    // ------------------------------------------------------------------
    double GetCurrentPrice(ENUM_POSITION_TYPE pos_type)
    {
        if(pos_type == POSITION_TYPE_BUY)
            return SymbolInfoDouble(m_symbol, SYMBOL_BID);
        return SymbolInfoDouble(m_symbol, SYMBOL_ASK);
    }

    // ------------------------------------------------------------------
    // Get pip profit for a position from its open price
    // Positive = profitable, negative = loss
    // ------------------------------------------------------------------
    double GetPipProfit(ulong ticket)
    {
        if(!PositionSelectByTicket(ticket)) return 0.0;

        ENUM_POSITION_TYPE type  = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        double open_price        = PositionGetDouble(POSITION_PRICE_OPEN);
        double current           = GetCurrentPrice(type);

        double diff = (type == POSITION_TYPE_BUY) ? current - open_price : open_price - current;
        return PriceToPips(diff);
    }

    // ------------------------------------------------------------------
    // Exit 1: Hard Stop Check
    // Close ALL positions if ANY position is 20+ pips adverse
    // ------------------------------------------------------------------
    bool CheckHardStop()
    {
        ulong tickets[];
        int count = m_entry->GetOpenTickets(tickets);
        if(count == 0) return false;

        for(int i = 0; i < count; i++)
        {
            double pip_profit = GetPipProfit(tickets[i]);
            if(pip_profit <= -m_hard_stop_pips)
            {
                PrintFormat("[ExitManager] HARD STOP triggered: %.1f pips adverse on ticket %d",
                            MathAbs(pip_profit), (int)tickets[i]);
                CloseAll(StringFormat("Hard stop (%.0f pips adverse)", m_hard_stop_pips));
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Exit 2: Direction Flip Check
    // Close ALL if M5 EMA stack has reversed vs cascade direction
    // ------------------------------------------------------------------
    bool CheckDirectionFlip()
    {
        ENUM_DIRECTION cascade_dir = m_entry->GetCascadeDirection();
        if(cascade_dir == DIRECTION_NONE) return false;

        // Check live EMA stack (not cached — we want real-time flip detection)
        bool flipped = false;
        if(cascade_dir == DIRECTION_BULL && m_direction->IsEMAStackBear())
            flipped = true;
        if(cascade_dir == DIRECTION_BEAR && m_direction->IsEMAStackBull())
            flipped = true;

        if(flipped)
        {
            Print("[ExitManager] DIRECTION FLIP — closing all cascade positions immediately");
            CloseAll("M5 EMA stack direction reversal");
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Exit 3: Time Exit Check
    // Close ALL if 15+ minutes elapsed since Pilot entry
    // ------------------------------------------------------------------
    bool CheckTimeExit()
    {
        datetime pilot_time = m_entry->GetPilotOpenTime();
        if(pilot_time == 0) return false;

        int elapsed_min = (int)(TimeCurrent() - pilot_time) / 60;
        if(elapsed_min >= m_max_trade_duration_min)
        {
            PrintFormat("[ExitManager] TIME EXIT: %d minutes elapsed (max %d)",
                        elapsed_min, m_max_trade_duration_min);
            CloseAll(StringFormat("Time exit (%d min elapsed)", elapsed_min));
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Exit 5: VWAP Rejection
    // 2 consecutive rejection candles at VWAP -> close 50%, SL to BE
    // ------------------------------------------------------------------
    bool CheckVWAPRejection()
    {
        if(m_entry->GetCascadeState() == CASCADE_IDLE) return false;

        // Only check on new M1 bar
        datetime current_bar = iTime(m_symbol, PERIOD_M1, 1);
        if(current_bar == m_last_vwap_check_bar) return false;
        m_last_vwap_check_bar = current_bar;

        double vwap = m_vwap->GetVWAP(m_symbol);
        if(vwap == 0.0) return false;

        double high1 = iHigh(m_symbol, PERIOD_M1, 1);
        double low1  = iLow(m_symbol,  PERIOD_M1, 1);

        // Check if candle interacted with VWAP (candle range overlaps VWAP)
        bool touched_vwap = (low1 <= vwap && high1 >= vwap);
        if(!touched_vwap)
        {
            m_vwap_rejection_count = 0;
            return false;
        }

        // Check if this is a rejection candle against the trade direction
        ENUM_DIRECTION cascade_dir = m_entry->GetCascadeDirection();
        ENUM_CANDLE_PATTERN pattern = m_patterns->Detect(m_symbol, PERIOD_M1, 1);

        bool is_rejection = false;
        if(cascade_dir == DIRECTION_BULL && m_patterns->IsBearishPattern(pattern))
            is_rejection = true;
        if(cascade_dir == DIRECTION_BEAR && m_patterns->IsBullishPattern(pattern))
            is_rejection = true;

        if(is_rejection)
        {
            m_vwap_rejection_count++;
            PrintFormat("[ExitManager] VWAP rejection candle #%d", m_vwap_rejection_count);
        }
        else
        {
            m_vwap_rejection_count = 0;
        }

        if(m_vwap_rejection_count >= 2)
        {
            Print("[ExitManager] VWAP REJECTION — closing 50%, moving SL to BE");
            ClosePercentage(0.50);
            m_vwap_rejection_count = 0;

            // Move SL to breakeven for remaining positions
            MoveSLToBreakeven();
            return true;
        }

        return false;
    }

    // ------------------------------------------------------------------
    // Exit 6: Primary TP Check
    // Close ALL when price reaches ATR(14) * 1.5 from Entry 1 price
    // ------------------------------------------------------------------
    bool CheckPrimaryTP()
    {
        if(m_primary_tp == 0.0) return false;

        ulong tickets[];
        int count = m_entry->GetOpenTickets(tickets);
        if(count == 0) return false;

        // Check against Pilot position
        ulong pilot = m_entry->GetPilotTicket();
        if(pilot == 0 || !PositionSelectByTicket(pilot)) return false;

        ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        double current = GetCurrentPrice(type);

        bool tp_reached = false;
        if(type == POSITION_TYPE_BUY  && current >= m_primary_tp) tp_reached = true;
        if(type == POSITION_TYPE_SELL && current <= m_primary_tp) tp_reached = true;

        if(tp_reached)
        {
            PrintFormat("[ExitManager] PRIMARY TP hit: %.5f (ATR*%.1f)", m_primary_tp, m_tp_multiplier);
            CloseAll(StringFormat("Primary TP (ATR*%.1f)", m_tp_multiplier));
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Exit 7: Momentum TP Check
    // +12 pips in 4 minutes -> extend TP to ATR*2.5, move SL to BE
    // ------------------------------------------------------------------
    bool CheckMomentumTP()
    {
        if(m_momentum_tp_extended) return false;

        ulong pilot = m_entry->GetPilotTicket();
        if(pilot == 0) return false;

        double pip_profit = GetPipProfit(pilot);

        // Start momentum tracking if +12 pips profit
        if(!m_momentum_tracking && pip_profit >= m_momentum_trigger_pips)
        {
            m_momentum_tracking     = true;
            m_momentum_start_time   = TimeCurrent();
            m_momentum_start_price  = GetCurrentPrice(
                (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE));

            PrintFormat("[ExitManager] Momentum tracking started: +%.1f pips at %s",
                        pip_profit, TimeToString(m_momentum_start_time, TIME_MINUTES));
        }

        if(m_momentum_tracking)
        {
            int elapsed_sec = (int)(TimeCurrent() - m_momentum_start_time);

            // If +12 pips sustained for <=4 minutes, extend TP
            if(elapsed_sec <= 240 && pip_profit >= m_momentum_trigger_pips)
            {
                // Extend TP to ATR * 2.5
                ENUM_DIRECTION cascade_dir = m_entry->GetCascadeDirection();
                double entry_price = m_entry->GetPilotOpenPrice();

                double tp_distance = m_atr_at_entry * m_momentum_tp_mult;
                if(cascade_dir == DIRECTION_BULL)
                    m_extended_tp = entry_price + tp_distance;
                else
                    m_extended_tp = entry_price - tp_distance;

                m_extended_tp = NormalizePrice(m_extended_tp);

                // Move primary TP to extended level
                m_primary_tp            = m_extended_tp;
                m_momentum_tp_extended  = true;
                m_momentum_tracking     = false;

                PrintFormat("[ExitManager] MOMENTUM TP extended to %.5f (ATR*%.1f)",
                            m_extended_tp, m_momentum_tp_mult);

                // Move SL to breakeven
                MoveSLToBreakeven();
                return false;  // Don't close yet — let price run to extended TP
            }
            else if(elapsed_sec > 240)
            {
                // 4 minutes elapsed without sustaining the move — cancel momentum tracking
                m_momentum_tracking = false;
                m_momentum_start_time = 0;
                PrintFormat("[ExitManager] Momentum tracking expired (pip profit=%.1f)", pip_profit);
            }
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Breakeven check: move SL to entry when +10 pips in profit
    // ------------------------------------------------------------------
    void CheckBreakeven()
    {
        if(m_breakeven_set) return;

        ulong pilot = m_entry->GetPilotTicket();
        if(pilot == 0) return;

        double pip_profit = GetPipProfit(pilot);
        if(pip_profit >= m_breakeven_trigger_pips)
        {
            PrintFormat("[ExitManager] BREAKEVEN triggered at +%.1f pips", pip_profit);
            MoveSLToBreakeven();
        }
    }

    void MoveSLToBreakeven()
    {
        ulong pilot = m_entry->GetPilotTicket();
        if(pilot == 0 || !PositionSelectByTicket(pilot)) return;

        double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
        SetSLForAll(NormalizePrice(entry_price));
        m_breakeven_set = true;
        Print("[ExitManager] SL moved to breakeven for all cascade positions");
    }

    // ------------------------------------------------------------------
    // Exit 8: Trailing Stop — 12 pips behind current price (after BE set)
    // ------------------------------------------------------------------
    void CheckTrailingStop()
    {
        if(!m_breakeven_set) return;

        ulong tickets[];
        int count = m_entry->GetOpenTickets(tickets);
        if(count == 0) return;

        ENUM_DIRECTION cascade_dir = m_entry->GetCascadeDirection();
        double trail_distance = PipToPrice(m_trailing_distance_pips);

        for(int i = 0; i < count; i++)
        {
            if(!PositionSelectByTicket(tickets[i])) continue;

            ENUM_POSITION_TYPE type  = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            double current_sl        = PositionGetDouble(POSITION_SL);
            double entry_price       = PositionGetDouble(POSITION_PRICE_OPEN);
            double current_price     = GetCurrentPrice(type);

            double new_sl = 0.0;

            if(type == POSITION_TYPE_BUY)
            {
                new_sl = NormalizePrice(current_price - trail_distance);
                // Only move SL up, never down. Must be above breakeven.
                if(new_sl > current_sl && new_sl >= entry_price)
                    ModifySL(tickets[i], new_sl);
            }
            else if(type == POSITION_TYPE_SELL)
            {
                new_sl = NormalizePrice(current_price + trail_distance);
                // Only move SL down, never up. Must be below breakeven.
                if((current_sl == 0.0 || new_sl < current_sl) && new_sl <= entry_price)
                    ModifySL(tickets[i], new_sl);
            }
        }
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------
    CExitManager()
    {
        m_direction              = NULL;
        m_entry                  = NULL;
        m_vwap                   = NULL;
        m_patterns               = NULL;

        m_symbol                 = _Symbol;
        m_hard_stop_pips         = 20.0;
        m_max_trade_duration_min = 15;
        m_breakeven_trigger_pips = 10.0;
        m_trailing_distance_pips = 12.0;
        m_tp_multiplier          = 1.5;
        m_momentum_tp_mult       = 2.5;
        m_momentum_trigger_pips  = 12.0;

        ResetExitState();
    }

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------
    bool Init(string           symbol,
              int              magic,
              CDirectionLayer  *direction,
              CEntryLayer      *entry,
              CVWAPCalculator  *vwap,
              CCandlePatterns  *patterns,
              double           hard_stop_pips         = 20.0,
              int              max_duration_min        = 15,
              double           breakeven_trigger_pips  = 10.0,
              double           trailing_distance_pips  = 12.0,
              double           tp_multiplier           = 1.5,
              double           momentum_tp_mult        = 2.5,
              double           momentum_trigger_pips   = 12.0)
    {
        m_symbol                 = symbol;
        m_direction              = direction;
        m_entry                  = entry;
        m_vwap                   = vwap;
        m_patterns               = patterns;
        m_hard_stop_pips         = hard_stop_pips;
        m_max_trade_duration_min = max_duration_min;
        m_breakeven_trigger_pips = breakeven_trigger_pips;
        m_trailing_distance_pips = trailing_distance_pips;
        m_tp_multiplier          = tp_multiplier;
        m_momentum_tp_mult       = momentum_tp_mult;
        m_momentum_trigger_pips  = momentum_trigger_pips;

        m_trade.SetExpertMagicNumber(magic);
        m_trade.SetDeviationInPoints(30);
        m_trade.SetTypeFilling(ORDER_FILLING_FOK);

        Print("[ExitManager] Initialized");
        return true;
    }

    // ------------------------------------------------------------------
    // SetTPLevels — called when Pilot entry is confirmed
    // Calculates and stores primary TP and records ATR at entry
    // ------------------------------------------------------------------
    void SetTPLevels(double pilot_price, double atr_value, ENUM_DIRECTION direction)
    {
        m_atr_at_entry = atr_value;
        double tp_dist = atr_value * m_tp_multiplier;

        if(direction == DIRECTION_BULL)
            m_primary_tp = NormalizePrice(pilot_price + tp_dist);
        else if(direction == DIRECTION_BEAR)
            m_primary_tp = NormalizePrice(pilot_price - tp_dist);

        PrintFormat("[ExitManager] TP levels set: primary=%.5f (ATR=%.5f * %.1f)",
                    m_primary_tp, atr_value, m_tp_multiplier);

        // Reset exit tracking for new trade
        ResetExitState();
        m_atr_at_entry = atr_value;   // Restore after reset
        m_primary_tp   = NormalizePrice((direction == DIRECTION_BULL) ?
                          pilot_price + atr_value * m_tp_multiplier :
                          pilot_price - atr_value * m_tp_multiplier);
    }

    // ------------------------------------------------------------------
    // ManageOpenTrades — MAIN ENTRY POINT called every tick
    //
    // Checks exits in strict priority order.
    // Returns early as soon as an exit action is taken.
    // ------------------------------------------------------------------
    void ManageOpenTrades()
    {
        // Nothing to manage if no cascade active
        if(!m_entry->IsActive())
            return;

        // ---- PRIORITY 1: Hard stop ----
        if(CheckHardStop())
            return;

        // ---- PRIORITY 2: Direction flip ----
        if(CheckDirectionFlip())
            return;

        // ---- PRIORITY 3: Time exit ----
        if(CheckTimeExit())
            return;

        // ---- PRIORITY 4: Initial SL — handled by broker orders ----
        // (We check if Pilot was closed, which is done in EntryLayer.ManageCascade)

        // ---- PRIORITY 5: VWAP rejection ----
        if(CheckVWAPRejection())
            return;

        // ---- PRIORITY 6: Primary TP ----
        if(CheckPrimaryTP())
            return;

        // ---- PRIORITY 7: Momentum TP extension ----
        CheckMomentumTP();   // Does not close, only extends TP and moves BE

        // ---- Breakeven check (independent, not an exit) ----
        CheckBreakeven();

        // ---- PRIORITY 8: Trailing stop ----
        CheckTrailingStop();
    }

    // ------------------------------------------------------------------
    // Public accessors for monitoring
    // ------------------------------------------------------------------
    bool IsBreakevenSet()       { return m_breakeven_set; }
    bool IsMomentumExtended()   { return m_momentum_tp_extended; }
    double GetPrimaryTP()       { return m_primary_tp; }
    double GetExtendedTP()      { return m_extended_tp; }
    double GetATRAtEntry()      { return m_atr_at_entry; }
};

#endif // EXITMANAGER_MQH
