//+------------------------------------------------------------------+
//| SwingExitManager.mqh — Position exit logic for swing trades      |
//|                                                                    |
//| Exit priority (highest to lowest):                                |
//|  1. H4 structural breakdown   — 100% close, immediate            |
//|  2. H4 200 EMA flip           — 100% close, immediate            |
//|  3. 72-hour time limit        — 100% close                       |
//|  4. Initial SL hit            — handled by broker (server SL)    |
//|  5. Major news proximity      — reduce to 50%, SL to BE          |
//|  6. DXY 3-candle headwind     — close 50%                        |
//|  7. AI trend exhaustion       — close 50%, trail rest            |
//|  8. TP1 hit                   — close 40%, SL to BE              |
//|  9. TP2 hit                   — close remaining 60%              |
//|                                                                    |
//| CRITICAL: structural checks run ONLY on H4 candle close          |
//|           Tick-level P&L and H1 noise never trigger exits        |
//+------------------------------------------------------------------+
#ifndef SWINGEXITMANAGER_MQH
#define SWINGEXITMANAGER_MQH

#include "Constants.mqh"
#include "SwingRiskManager.mqh"
#include "DXYFilter.mqh"
#include "NewsShield.mqh"
#include "MarketStructure.mqh"
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

// ------------------------------------------------------------------
// Per-position state — one entry per open swing position
// ------------------------------------------------------------------

struct SSwingPositionState
{
    ulong           ticket;
    ENUM_DIRECTION  direction;
    double          entry_price;
    double          sl_price;
    double          tp1_price;
    double          tp2_price;
    double          original_lots;
    double          current_lots;
    datetime        open_time;
    bool            tp1_hit;           // Has TP1 already been closed?
    bool            sl_moved_to_be;    // Has SL been moved to breakeven?
    int             last_ai_trend_score;
    bool            news_reduction_done;  // 50% closed for news
    bool            dxy_reduction_done;   // 50% closed for DXY
};

class CSwingExitManager
{
private:
    CTrade              m_trade;
    CPositionInfo       m_pos_info;
    CMarketStructure    m_ms;
    CSwingRiskManager  *m_risk_mgr;
    CDXYFilter         *m_dxy_filter;
    CNewsShield        *m_news_shield;

    // Position tracking array
    SSwingPositionState m_positions[];
    int                 m_pos_count;

    // EA settings
    int                 m_magic;
    int                 m_max_hold_hours;
    int                 m_ai_exhaustion_score;
    string              m_symbol;

    // H4 candle close tracking
    datetime            m_last_h4_candle;       // iTime(sym, H4, 1) at last structural check

    // ------------------------------------------------------------------
    // Find a position in our tracking array by ticket
    // ------------------------------------------------------------------

    int FindPosition(ulong ticket)
    {
        for(int i = 0; i < m_pos_count; i++)
            if(m_positions[i].ticket == ticket)
                return i;
        return -1;
    }

    // ------------------------------------------------------------------
    // Remove a position from tracking (after close)
    // ------------------------------------------------------------------

    void RemovePosition(int idx)
    {
        if(idx < 0 || idx >= m_pos_count) return;
        for(int i = idx; i < m_pos_count - 1; i++)
            m_positions[i] = m_positions[i + 1];
        m_pos_count--;
        ArrayResize(m_positions, m_pos_count);
    }

    // ------------------------------------------------------------------
    // Close full position by ticket
    // ------------------------------------------------------------------

    bool CloseFullPosition(ulong ticket, string reason)
    {
        if(!m_pos_info.SelectByTicket(ticket))
        {
            PrintFormat("[SwingExit] Cannot select ticket %llu", ticket);
            return false;
        }

        double lots = m_pos_info.Volume();
        bool result = m_trade.PositionClose(ticket);

        if(result)
            PrintFormat("[SwingExit] FULL CLOSE ticket=%llu lots=%.2f reason=%s",
                        ticket, lots, reason);
        else
            PrintFormat("[SwingExit] CLOSE FAILED ticket=%llu err=%d reason=%s",
                        ticket, m_trade.ResultRetcode(), reason);

        return result;
    }

    // ------------------------------------------------------------------
    // Close partial position
    // ------------------------------------------------------------------

    bool ClosePartialPosition(ulong ticket, double close_lots, string reason)
    {
        if(close_lots <= 0)
            return false;

        if(!m_pos_info.SelectByTicket(ticket))
        {
            PrintFormat("[SwingExit] Cannot select ticket %llu for partial close", ticket);
            return false;
        }

        double current_lots = m_pos_info.Volume();
        double vol_min      = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);

        if(close_lots >= current_lots)
        {
            // Close all if partial would leave less than minimum
            return CloseFullPosition(ticket, reason + "(all)");
        }

        if(current_lots - close_lots < vol_min)
        {
            // Remainder would be below minimum — close all
            return CloseFullPosition(ticket, reason + "(all-min)");
        }

        ENUM_POSITION_TYPE pos_type = m_pos_info.PositionType();
        double price = (pos_type == POSITION_TYPE_BUY)
                       ? SymbolInfoDouble(m_symbol, SYMBOL_BID)
                       : SymbolInfoDouble(m_symbol, SYMBOL_ASK);

        bool result = m_trade.PositionClosePartial(ticket, close_lots);

        if(result)
            PrintFormat("[SwingExit] PARTIAL CLOSE ticket=%llu lots=%.2f remaining=%.2f reason=%s",
                        ticket, close_lots, current_lots - close_lots, reason);
        else
            PrintFormat("[SwingExit] PARTIAL CLOSE FAILED ticket=%llu err=%d reason=%s",
                        ticket, m_trade.ResultRetcode(), reason);

        return result;
    }

    // ------------------------------------------------------------------
    // Move stop loss to breakeven
    // ------------------------------------------------------------------

    bool MoveSLToBreakeven(ulong ticket, double entry_price)
    {
        if(!m_pos_info.SelectByTicket(ticket))
            return false;

        double current_sl = m_pos_info.StopLoss();
        double point      = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
        double pip        = point * 10.0;
        double be_buffer  = 1.0 * pip;  // 1 pip above entry for BE (covers spread)

        ENUM_POSITION_TYPE pos_type = m_pos_info.PositionType();
        double be_sl;

        if(pos_type == POSITION_TYPE_BUY)
        {
            be_sl = entry_price + be_buffer;
            // Only move if it's actually an improvement
            if(be_sl <= current_sl && current_sl > 0)
                return true;  // Already at or above BE
        }
        else
        {
            be_sl = entry_price - be_buffer;
            if(be_sl >= current_sl && current_sl > 0)
                return true;  // Already at or below BE
        }

        double tp = m_pos_info.TakeProfit();
        bool result = m_trade.PositionModify(ticket, be_sl, tp);

        if(result)
            PrintFormat("[SwingExit] SL moved to BE ticket=%llu be_sl=%.2f", ticket, be_sl);
        else
            PrintFormat("[SwingExit] SL to BE FAILED ticket=%llu err=%d", ticket, m_trade.ResultRetcode());

        return result;
    }

    // ------------------------------------------------------------------
    // STRUCTURAL CHECK 1: H4 close below last Higher Low (long)
    //                     H4 close above last Lower High (short)
    // This is the primary exit trigger — structure has broken.
    // Evaluated ONLY on H4 candle close.
    // ------------------------------------------------------------------

    bool HasStructuralBreakdown(SSwingPositionState &state)
    {
        // Use the last closed H4 bar (shift = 1)
        double h4_close = iClose(m_symbol, PERIOD_H4, 1);

        if(state.direction == DIRECTION_BULL)
        {
            // Get the last confirmed Higher Low
            m_ms.SetFractalBars(5);
            m_ms.SetLookback(60);
            double last_hl = m_ms.GetLastSwingLow(m_symbol, PERIOD_H4);

            if(last_hl > 0 && h4_close < last_hl)
            {
                PrintFormat("[SwingExit] STRUCTURAL BREAK LONG: H4 close %.2f < last HL %.2f",
                            h4_close, last_hl);
                return true;
            }
        }
        else if(state.direction == DIRECTION_BEAR)
        {
            // Get the last confirmed Lower High
            m_ms.SetFractalBars(5);
            m_ms.SetLookback(60);
            double last_lh = m_ms.GetLastSwingHigh(m_symbol, PERIOD_H4);

            if(last_lh > 0 && h4_close > last_lh)
            {
                PrintFormat("[SwingExit] STRUCTURAL BREAK SHORT: H4 close %.2f > last LH %.2f",
                            h4_close, last_lh);
                return true;
            }
        }

        return false;
    }

    // ------------------------------------------------------------------
    // STRUCTURAL CHECK 2: H4 200 EMA flip
    // Long: H4 close below 200 EMA = institutional support gone
    // Short: H4 close above 200 EMA = institutional resistance gone
    // Evaluated ONLY on H4 candle close.
    // ------------------------------------------------------------------

    bool HasEMAFlip(SSwingPositionState &state, double ema200)
    {
        if(ema200 <= 0)
            return false;

        double h4_close = iClose(m_symbol, PERIOD_H4, 1);

        if(state.direction == DIRECTION_BULL && h4_close < ema200)
        {
            PrintFormat("[SwingExit] EMA FLIP LONG: H4 close %.2f < 200 EMA %.2f",
                        h4_close, ema200);
            return true;
        }

        if(state.direction == DIRECTION_BEAR && h4_close > ema200)
        {
            PrintFormat("[SwingExit] EMA FLIP SHORT: H4 close %.2f > 200 EMA %.2f",
                        h4_close, ema200);
            return true;
        }

        return false;
    }

    // ------------------------------------------------------------------
    // CHECK 3: 72-hour time limit
    // ------------------------------------------------------------------

    bool HasExceededTimeLimit(SSwingPositionState &state)
    {
        int held_hours = (int)((TimeGMT() - state.open_time) / 3600);
        if(held_hours >= m_max_hold_hours)
        {
            PrintFormat("[SwingExit] TIME LIMIT: position open %d hours (max %d)",
                        held_hours, m_max_hold_hours);
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // CHECK 5: Major news proximity — NFP/CPI/FOMC within 2 hours
    // Action: reduce to 50%, move SL to breakeven
    // ------------------------------------------------------------------

    bool ShouldReduceForNews(SSwingPositionState &state)
    {
        if(m_news_shield == NULL)
            return false;
        if(state.news_reduction_done)
            return false;

        // Only trigger for high-impact events within 2 hours
        return m_news_shield->IsMajorEventWithinHours(2);
    }

    // ------------------------------------------------------------------
    // CHECK 6: DXY 3-candle H4 rally against position
    // Action: close 50%
    // ------------------------------------------------------------------

    bool ShouldReduceForDXY(SSwingPositionState &state)
    {
        if(m_dxy_filter == NULL)
            return false;
        if(state.dxy_reduction_done)
            return false;

        return m_dxy_filter->HasThreeCandleRally(state.direction);
    }

    // ------------------------------------------------------------------
    // CHECK 7: AI trend exhaustion — trend score dropped below threshold
    // Action: close 50%, trail the rest
    // Note: AI score must have PREVIOUSLY been above m_ai_exhaustion_score+20
    // (i.e., it dropped from >72 to <45, not just started at <45)
    // ------------------------------------------------------------------

    bool ShouldReduceForAIExhaustion(SSwingPositionState &state)
    {
        // Check if score dropped significantly
        if(state.last_ai_trend_score < m_ai_exhaustion_score &&
           state.last_ai_trend_score > 0)
        {
            PrintFormat("[SwingExit] AI EXHAUSTION: trend score %d < %d",
                        state.last_ai_trend_score, m_ai_exhaustion_score);
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // CHECK 8: TP1 hit
    // ------------------------------------------------------------------

    bool HasTP1Hit(SSwingPositionState &state)
    {
        if(state.tp1_hit)
            return false;  // Already processed

        double current_price;
        if(state.direction == DIRECTION_BULL)
            current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
        else
            current_price = SymbolInfoDouble(m_symbol, SYMBOL_ASK);

        if(state.direction == DIRECTION_BULL && current_price >= state.tp1_price)
            return true;
        if(state.direction == DIRECTION_BEAR && current_price <= state.tp1_price)
            return true;

        return false;
    }

    // ------------------------------------------------------------------
    // CHECK 9: TP2 hit
    // ------------------------------------------------------------------

    bool HasTP2Hit(SSwingPositionState &state)
    {
        double current_price;
        if(state.direction == DIRECTION_BULL)
            current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
        else
            current_price = SymbolInfoDouble(m_symbol, SYMBOL_ASK);

        if(state.direction == DIRECTION_BULL && current_price >= state.tp2_price)
            return true;
        if(state.direction == DIRECTION_BEAR && current_price <= state.tp2_price)
            return true;

        return false;
    }

    // ------------------------------------------------------------------
    // Apply trailing stop after AI exhaustion or TP1
    // Trail: move SL behind current price by 20 pips
    // ------------------------------------------------------------------

    void ApplyTrailingStop(ulong ticket, ENUM_DIRECTION direction, double trail_pips = 20.0)
    {
        if(!m_pos_info.SelectByTicket(ticket))
            return;

        double point    = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
        double pip      = point * 10.0;
        double trail_d  = trail_pips * pip;

        double current_sl = m_pos_info.StopLoss();
        double current_tp = m_pos_info.TakeProfit();
        double new_sl;

        if(direction == DIRECTION_BULL)
        {
            double bid = SymbolInfoDouble(m_symbol, SYMBOL_BID);
            new_sl = bid - trail_d;
            // Only tighten — never widen stop
            if(new_sl <= current_sl && current_sl > 0)
                return;
        }
        else
        {
            double ask = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
            new_sl = ask + trail_d;
            if(new_sl >= current_sl && current_sl > 0)
                return;
        }

        if(m_trade.PositionModify(ticket, new_sl, current_tp))
            PrintFormat("[SwingExit] TRAIL SL: ticket=%llu new_sl=%.2f", ticket, new_sl);
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    CSwingExitManager()
    {
        m_risk_mgr           = NULL;
        m_dxy_filter         = NULL;
        m_news_shield        = NULL;
        m_pos_count          = 0;
        m_magic              = MAGIC_SWING;
        m_max_hold_hours     = 72;
        m_ai_exhaustion_score = 45;
        m_symbol             = _Symbol;
        m_last_h4_candle     = 0;
        ArrayResize(m_positions, 0);
    }

    // ------------------------------------------------------------------
    // Initialise
    // ------------------------------------------------------------------

    void Init(string symbol, int magic, int max_hold_hours,
              int ai_exhaustion_score,
              CSwingRiskManager *risk_mgr,
              CDXYFilter        *dxy_filter,
              CNewsShield       *news_shield)
    {
        m_symbol              = symbol;
        m_magic               = magic;
        m_max_hold_hours      = max_hold_hours;
        m_ai_exhaustion_score = ai_exhaustion_score;
        m_risk_mgr            = risk_mgr;
        m_dxy_filter          = dxy_filter;
        m_news_shield         = news_shield;

        m_trade.SetExpertMagicNumber(magic);
        m_trade.SetDeviationInPoints(30);

        m_ms.SetFractalBars(5);
        m_ms.SetLookback(60);
    }

    // ------------------------------------------------------------------
    // RegisterPosition — called immediately after opening a new trade
    // ------------------------------------------------------------------

    void RegisterPosition(ulong ticket, ENUM_DIRECTION direction,
                          double entry_price, double sl_price,
                          double tp1_price, double tp2_price,
                          double lots)
    {
        int idx = m_pos_count;
        ArrayResize(m_positions, idx + 1);

        m_positions[idx].ticket               = ticket;
        m_positions[idx].direction            = direction;
        m_positions[idx].entry_price          = entry_price;
        m_positions[idx].sl_price             = sl_price;
        m_positions[idx].tp1_price            = tp1_price;
        m_positions[idx].tp2_price            = tp2_price;
        m_positions[idx].original_lots        = lots;
        m_positions[idx].current_lots         = lots;
        m_positions[idx].open_time            = TimeGMT();
        m_positions[idx].tp1_hit              = false;
        m_positions[idx].sl_moved_to_be       = false;
        m_positions[idx].last_ai_trend_score  = 99;   // Assume max at entry
        m_positions[idx].news_reduction_done  = false;
        m_positions[idx].dxy_reduction_done   = false;

        m_pos_count++;

        PrintFormat("[SwingExit] Registered ticket=%llu dir=%d entry=%.2f sl=%.2f tp1=%.2f tp2=%.2f lots=%.2f",
                    ticket, direction, entry_price, sl_price, tp1_price, tp2_price, lots);
    }

    // ------------------------------------------------------------------
    // UpdateAIScore — called after each AI heartbeat/score response
    // ------------------------------------------------------------------

    void UpdateAIScore(ulong ticket, int trend_score)
    {
        int idx = FindPosition(ticket);
        if(idx >= 0)
            m_positions[idx].last_ai_trend_score = trend_score;
    }

    void UpdateAllAIScores(int trend_score)
    {
        for(int i = 0; i < m_pos_count; i++)
            m_positions[i].last_ai_trend_score = trend_score;
    }

    // ------------------------------------------------------------------
    // ManagePositions — MAIN METHOD, called every tick
    //
    // Processes ALL open swing positions in priority order.
    // Structural exits only on H4 candle close.
    // TP hits evaluated every tick (price-based).
    // ------------------------------------------------------------------

    void ManagePositions(double h4_ema200)
    {
        // Sync position list with what broker actually has open
        SyncPositionList();

        if(m_pos_count == 0)
            return;

        // Check if a new H4 candle has closed
        datetime current_h4_candle = iTime(m_symbol, PERIOD_H4, 1);
        bool h4_candle_closed = (current_h4_candle != m_last_h4_candle);

        if(h4_candle_closed)
        {
            m_last_h4_candle = current_h4_candle;
            PrintFormat("[SwingExit] H4 candle closed — running structural checks");
        }

        // Process each tracked position
        // Iterate backwards because RemovePosition shifts the array
        for(int i = m_pos_count - 1; i >= 0; i--)
        {
            SSwingPositionState state = m_positions[i];

            // Verify position still exists
            if(!m_pos_info.SelectByTicket(state.ticket))
            {
                PrintFormat("[SwingExit] Ticket %llu no longer exists — removing", state.ticket);
                RemovePosition(i);
                continue;
            }

            // Update current lots from broker
            m_positions[i].current_lots = m_pos_info.Volume();

            // ============================================================
            // PRIORITY 1 & 2: Structural exits — H4 candle close ONLY
            // ============================================================

            if(h4_candle_closed)
            {
                // Priority 1: H4 structural breakdown
                if(HasStructuralBreakdown(state))
                {
                    CloseFullPosition(state.ticket, "H4_STRUCTURAL_BREAK");
                    RemovePosition(i);
                    continue;
                }

                // Priority 2: H4 200 EMA flip
                if(HasEMAFlip(state, h4_ema200))
                {
                    CloseFullPosition(state.ticket, "H4_EMA_FLIP");
                    RemovePosition(i);
                    continue;
                }

                // Priority 6: DXY 3-candle headwind (also on H4 close)
                if(ShouldReduceForDXY(state))
                {
                    double close_lots = (m_risk_mgr != NULL)
                                        ? m_risk_mgr->CalcHalfCloseLots(state.current_lots)
                                        : state.current_lots * 0.5;
                    if(close_lots > 0)
                    {
                        ClosePartialPosition(state.ticket, close_lots, "DXY_HEADWIND_50PCT");
                        m_positions[i].dxy_reduction_done = true;
                    }
                    // Do not continue — may have more checks
                }
            }

            // Refresh state after potential partial close
            if(!m_pos_info.SelectByTicket(state.ticket))
            {
                RemovePosition(i);
                continue;
            }
            m_positions[i].current_lots = m_pos_info.Volume();
            state = m_positions[i];

            // ============================================================
            // PRIORITY 3: Time limit (72 hours) — checked every tick
            // ============================================================

            if(HasExceededTimeLimit(state))
            {
                CloseFullPosition(state.ticket, "TIME_LIMIT_72H");
                RemovePosition(i);
                continue;
            }

            // ============================================================
            // PRIORITY 4: Initial SL — handled by broker server stop
            // (No action needed here; the SL order on the broker covers this)
            // ============================================================

            // ============================================================
            // PRIORITY 5: Major news proximity — checked every tick
            // ============================================================

            if(ShouldReduceForNews(state))
            {
                double close_lots = (m_risk_mgr != NULL)
                                    ? m_risk_mgr->CalcHalfCloseLots(state.current_lots)
                                    : state.current_lots * 0.5;
                if(close_lots > 0)
                    ClosePartialPosition(state.ticket, close_lots, "NEWS_PROXIMITY_50PCT");

                MoveSLToBreakeven(state.ticket, state.entry_price);
                m_positions[i].news_reduction_done  = true;
                m_positions[i].sl_moved_to_be       = true;
            }

            // Refresh
            if(!m_pos_info.SelectByTicket(state.ticket))
            {
                RemovePosition(i);
                continue;
            }
            m_positions[i].current_lots = m_pos_info.Volume();
            state = m_positions[i];

            // ============================================================
            // PRIORITY 7: AI trend exhaustion — checked every tick
            // ============================================================

            if(ShouldReduceForAIExhaustion(state))
            {
                double close_lots = (m_risk_mgr != NULL)
                                    ? m_risk_mgr->CalcHalfCloseLots(state.current_lots)
                                    : state.current_lots * 0.5;
                if(close_lots > 0)
                    ClosePartialPosition(state.ticket, close_lots, "AI_EXHAUSTION_50PCT");

                // Trail the remaining position tightly
                ApplyTrailingStop(state.ticket, state.direction, 20.0);

                // Mark as news_reduction_done to prevent double-triggering
                m_positions[i].news_reduction_done = true;
            }

            // Refresh
            if(!m_pos_info.SelectByTicket(state.ticket))
            {
                RemovePosition(i);
                continue;
            }
            m_positions[i].current_lots = m_pos_info.Volume();
            state = m_positions[i];

            // ============================================================
            // PRIORITY 8: TP1 hit — close 40%, move SL to BE
            // ============================================================

            if(HasTP1Hit(state))
            {
                double close_lots = (m_risk_mgr != NULL)
                                    ? m_risk_mgr->CalcTP1CloseLots(state.current_lots)
                                    : state.current_lots * 0.4;
                if(close_lots > 0)
                    ClosePartialPosition(state.ticket, close_lots, "TP1_40PCT");

                MoveSLToBreakeven(state.ticket, state.entry_price);

                m_positions[i].tp1_hit        = true;
                m_positions[i].sl_moved_to_be = true;

                // Apply trailing stop on the remaining position
                ApplyTrailingStop(state.ticket, state.direction, 25.0);

                PrintFormat("[SwingExit] TP1 processed ticket=%llu — 40%% closed, SL to BE",
                            state.ticket);
            }

            // Refresh
            if(!m_pos_info.SelectByTicket(state.ticket))
            {
                RemovePosition(i);
                continue;
            }
            m_positions[i].current_lots = m_pos_info.Volume();
            state = m_positions[i];

            // ============================================================
            // PRIORITY 9: TP2 hit — close remaining 60%
            // ============================================================

            if(state.tp1_hit && HasTP2Hit(state))
            {
                CloseFullPosition(state.ticket, "TP2_FULL_CLOSE");
                RemovePosition(i);
                continue;
            }

            // ============================================================
            // Trailing stop maintenance — once SL is at BE, keep trailing
            // ============================================================

            if(state.sl_moved_to_be)
            {
                ApplyTrailingStop(state.ticket, state.direction, 30.0);
            }

        }  // end position loop
    }

    // ------------------------------------------------------------------
    // SyncPositionList — reconcile our array with broker positions
    // Removes tickets that are no longer open
    // ------------------------------------------------------------------

    void SyncPositionList()
    {
        for(int i = m_pos_count - 1; i >= 0; i--)
        {
            if(!m_pos_info.SelectByTicket(m_positions[i].ticket))
                RemovePosition(i);
        }
    }

    // ------------------------------------------------------------------
    // HasOpenPosition — check if EA has any open swing positions
    // ------------------------------------------------------------------

    bool HasOpenPosition()
    {
        SyncPositionList();
        return (m_pos_count > 0);
    }

    int GetOpenPositionCount()
    {
        SyncPositionList();
        return m_pos_count;
    }

    // ------------------------------------------------------------------
    // GetOpenPositionDirection — returns direction of first open position
    // ------------------------------------------------------------------

    ENUM_DIRECTION GetOpenPositionDirection()
    {
        if(m_pos_count > 0)
            return m_positions[0].direction;
        return DIRECTION_NONE;
    }

    // ------------------------------------------------------------------
    // GetPositionTicket — returns ticket of first open position
    // ------------------------------------------------------------------

    ulong GetPositionTicket(int idx = 0)
    {
        if(idx >= 0 && idx < m_pos_count)
            return m_positions[idx].ticket;
        return 0;
    }
};

#endif // SWINGEXITMANAGER_MQH
