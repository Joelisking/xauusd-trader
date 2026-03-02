//+------------------------------------------------------------------+
//| RiskManager.mqh — Position sizing and session risk control       |
//|                                                                   |
//| Key rules:                                                        |
//|   - Per-trade lot: (Balance * 0.015) / (SL_pips * pip_value)    |
//|   - Hard cap: max lot = 5% account regardless of formula         |
//|   - Session halt at 7% cumulative loss (non-overridable)          |
//|   - Session cap: 10% max cumulative loss per session             |
//|   - Daily cap: max 2 sessions. Session 2 only if Session 1 > 0  |
//|   - Winning session bonus: 6%+ profit -> next session at 1.0%   |
//|                             risk instead of 1.5%                 |
//+------------------------------------------------------------------+
#ifndef RISKMANAGER_MQH
#define RISKMANAGER_MQH

#include "Constants.mqh"

class CRiskManager
{
private:
    // ------------------------------------------------------------------
    // Configuration (set via Init)
    // ------------------------------------------------------------------
    string  m_symbol;
    double  m_risk_pct;              // Default 1.5% per trade
    double  m_session_risk_cap;      // Default 10.0% cumulative loss
    double  m_session_halt_pct;      // Default 7.0% — non-overridable halt
    double  m_hard_lot_cap_pct;      // Default 5.0% — max lot as % of balance
    double  m_winning_bonus_trigger; // Default 6.0% — session profit to get bonus
    double  m_winning_bonus_risk;    // Default 1.0% — reduced risk after big win

    // ------------------------------------------------------------------
    // Session state
    // ------------------------------------------------------------------
    int     m_session_number;        // 1 or 2 within the trading day
    double  m_session_pnl;           // Cumulative P&L for current session (account currency)
    double  m_session_start_balance; // Account balance at session start
    bool    m_session_halted;        // True if 7% loss triggered
    bool    m_session_1_was_positive;// True if Session 1 closed with net positive P&L

    // Daily tracking
    int     m_daily_session_count;   // Sessions traded today (max 2)
    datetime m_day_start;            // Timestamp of current trading day start

    // ------------------------------------------------------------------
    // Trade tracking
    // ------------------------------------------------------------------
    double  m_closed_pnl_session;    // Closed P&L accumulated this session
    int     m_trades_this_session;   // Number of trades opened this session

    // ------------------------------------------------------------------
    // Private: calculate pip value for 1 standard lot on _Symbol
    // For XAUUSD: $1 per 0.01 lot per pip (tick value approach)
    // pip_value = (tick_value / tick_size) * pip_size
    // ------------------------------------------------------------------
    double GetPipValue()
    {
        double tick_value = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_VALUE);
        double tick_size  = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
        double point      = SymbolInfoDouble(m_symbol, SYMBOL_POINT);

        if(tick_size == 0.0 || point == 0.0) return 1.0;

        // 1 pip = 10 points for XAUUSD
        double pip_size = point * 10.0;

        // pip_value per lot = (tick_value / tick_size) * pip_size
        return (tick_value / tick_size) * pip_size;
    }

    // ------------------------------------------------------------------
    // Private: check if a new trading day has started
    // Reset daily counters if day has rolled over
    // ------------------------------------------------------------------
    void CheckDayRollover()
    {
        MqlDateTime dt;
        TimeGMT(dt);

        // New day: reset at midnight UTC (or on first call)
        MqlDateTime day_dt;
        TimeToStruct(m_day_start, day_dt);

        if(dt.day != day_dt.day || m_day_start == 0)
        {
            if(m_day_start != 0)
            {
                PrintFormat("[RiskManager] New trading day — resetting daily counters");
            }

            m_daily_session_count    = 0;
            m_session_1_was_positive = false;

            // Record day start time as midnight of current day
            dt.hour = 0;
            dt.min  = 0;
            dt.sec  = 0;
            m_day_start = StructToTime(dt);
        }
    }

    // ------------------------------------------------------------------
    // Private: get effective risk percent for this session
    // Applies winning session bonus if applicable
    // ------------------------------------------------------------------
    double GetEffectiveRiskPct()
    {
        // After a big winning session (6%+), next session uses reduced risk
        if(m_session_number == 2 && m_session_1_was_positive)
        {
            double session1_return = GetSession1ReturnPct();
            if(session1_return >= m_winning_bonus_trigger)
            {
                PrintFormat("[RiskManager] Winning session bonus applied: %.1f%% risk (was %.1f%%)",
                            m_winning_bonus_risk, m_risk_pct);
                return m_winning_bonus_risk;
            }
        }
        return m_risk_pct;
    }

    // Placeholder: session 1 return is tracked externally
    // In practice, this would be stored from the session close
    double m_session1_return_pct;

    double GetSession1ReturnPct()
    {
        return m_session1_return_pct;
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------
    CRiskManager()
    {
        m_symbol                 = _Symbol;
        m_risk_pct               = 1.5;
        m_session_risk_cap       = 10.0;
        m_session_halt_pct       = 7.0;
        m_hard_lot_cap_pct       = 5.0;
        m_winning_bonus_trigger  = 6.0;
        m_winning_bonus_risk     = 1.0;

        m_session_number         = 1;
        m_session_pnl            = 0.0;
        m_session_start_balance  = 0.0;
        m_session_halted         = false;
        m_session_1_was_positive = false;

        m_daily_session_count    = 0;
        m_day_start              = 0;

        m_closed_pnl_session     = 0.0;
        m_trades_this_session    = 0;
        m_session1_return_pct    = 0.0;
    }

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------
    bool Init(string symbol,
              double risk_pct              = 1.5,
              double session_risk_cap      = 10.0,
              double session_halt_pct      = 7.0)
    {
        m_symbol           = symbol;
        m_risk_pct         = risk_pct;
        m_session_risk_cap = session_risk_cap;
        m_session_halt_pct = session_halt_pct;

        CheckDayRollover();
        StartNewSession();

        PrintFormat("[RiskManager] Initialized: risk=%.1f%% session_cap=%.1f%% halt=%.1f%%",
                    m_risk_pct, m_session_risk_cap, m_session_halt_pct);
        return true;
    }

    // ------------------------------------------------------------------
    // StartNewSession — call at the start of each trading session
    // ------------------------------------------------------------------
    void StartNewSession()
    {
        CheckDayRollover();

        m_session_start_balance  = AccountInfoDouble(ACCOUNT_BALANCE);
        m_session_pnl            = 0.0;
        m_closed_pnl_session     = 0.0;
        m_trades_this_session    = 0;
        m_session_halted         = false;

        m_daily_session_count++;
        m_session_number = m_daily_session_count;

        PrintFormat("[RiskManager] New session started: session #%d, balance=%.2f",
                    m_session_number, m_session_start_balance);
    }

    // ------------------------------------------------------------------
    // CalcLotSize — main lot size formula
    //
    // Lot = (AccountBalance * risk_pct/100) / (sl_pips * pip_value)
    // Capped at 5% of account balance worth of risk
    // ------------------------------------------------------------------
    double CalcLotSize(double sl_pips)
    {
        if(sl_pips <= 0.0)
        {
            Print("[RiskManager] CalcLotSize: invalid sl_pips <= 0");
            return SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
        }

        double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
        double pip_value  = GetPipValue();  // Per 1 lot

        if(pip_value <= 0.0)
        {
            Print("[RiskManager] CalcLotSize: invalid pip_value");
            return SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
        }

        double effective_risk = GetEffectiveRiskPct();
        double risk_amount    = balance * (effective_risk / 100.0);

        // Lot = risk_amount / (sl_pips * pip_value)
        double raw_lot = risk_amount / (sl_pips * pip_value);

        // Hard cap: max lot = 5% of account balance / pip_value
        // Translate account balance to max loss: 5% of balance
        double max_risk_amount = balance * (m_hard_lot_cap_pct / 100.0);
        double max_lot = max_risk_amount / (sl_pips * pip_value);

        double lot = MathMin(raw_lot, max_lot);

        // Normalize to broker's lot step
        double lot_min  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
        double lot_max  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MAX);
        double lot_step = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);

        lot = MathRound(lot / lot_step) * lot_step;
        lot = MathMax(lot_min, MathMin(lot_max, lot));

        PrintFormat("[RiskManager] CalcLotSize: SL=%.1f pips, risk=%.1f%%, pip_value=%.4f -> lot=%.2f (max=%.2f)",
                    sl_pips, effective_risk, pip_value, lot, max_lot);

        return lot;
    }

    // ------------------------------------------------------------------
    // TrackTrade — record a closed trade's P&L
    //
    // Call this when a position closes (from OnTradeTransaction or
    // by scanning closed trades in OnTick).
    // ------------------------------------------------------------------
    void TrackTrade(double pnl)
    {
        m_closed_pnl_session += pnl;
        m_trades_this_session++;

        // Update session P&L
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        if(m_session_start_balance > 0)
        {
            double pnl_pct = ((balance - m_session_start_balance) / m_session_start_balance) * 100.0;
            m_session_pnl = pnl_pct;
        }

        PrintFormat("[RiskManager] Trade recorded: P&L=%.2f, session_pnl=%.2f%% (%d trades)",
                    pnl, m_session_pnl, m_trades_this_session);

        // Check halt condition
        if(m_session_pnl <= -m_session_halt_pct && !m_session_halted)
        {
            m_session_halted = true;
            PrintFormat("[RiskManager] SESSION HALT triggered at %.2f%% loss (threshold=%.1f%%)",
                        m_session_pnl, m_session_halt_pct);
        }
    }

    // ------------------------------------------------------------------
    // SessionCapReached — true if cumulative session loss >= SessionRiskCap
    // ------------------------------------------------------------------
    bool SessionCapReached()
    {
        // Always update session P&L before checking
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        if(m_session_start_balance > 0)
        {
            m_session_pnl = ((balance - m_session_start_balance) / m_session_start_balance) * 100.0;
        }

        return (m_session_pnl <= -m_session_risk_cap);
    }

    // ------------------------------------------------------------------
    // IsSessionHalted — non-overridable halt at 7% loss
    // ------------------------------------------------------------------
    bool IsSessionHalted()
    {
        // Re-evaluate on every call
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        if(m_session_start_balance > 0)
        {
            double pnl_pct = ((balance - m_session_start_balance) / m_session_start_balance) * 100.0;
            if(pnl_pct <= -m_session_halt_pct)
            {
                if(!m_session_halted)
                {
                    m_session_halted = true;
                    PrintFormat("[RiskManager] SESSION HALT: %.2f%% cumulative loss", pnl_pct);
                }
                return true;
            }
        }
        return m_session_halted;
    }

    // ------------------------------------------------------------------
    // IsDailyCapReached — true if 2 sessions already traded today,
    // OR if attempting Session 2 but Session 1 was net negative
    // ------------------------------------------------------------------
    bool IsDailyCapReached()
    {
        CheckDayRollover();

        // Hard limit: 2 sessions per day
        if(m_daily_session_count >= 2)
        {
            Print("[RiskManager] Daily cap reached: 2 sessions already traded");
            return true;
        }

        // Session 2 only allowed if Session 1 was net positive
        if(m_daily_session_count == 1 && m_session_number == 1)
        {
            // We're in Session 1 — not yet at the daily cap question
            return false;
        }

        // About to start Session 2 — check if Session 1 was positive
        if(m_session_number >= 2 && !m_session_1_was_positive)
        {
            Print("[RiskManager] Session 2 blocked: Session 1 was not net positive");
            return true;
        }

        return false;
    }

    // ------------------------------------------------------------------
    // CloseSession — call at end of each session to record outcome
    // ------------------------------------------------------------------
    void CloseSession()
    {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double session_return_pct = 0.0;

        if(m_session_start_balance > 0)
            session_return_pct = ((balance - m_session_start_balance) / m_session_start_balance) * 100.0;

        PrintFormat("[RiskManager] Session %d closed: return=%.2f%%, trades=%d",
                    m_session_number, session_return_pct, m_trades_this_session);

        // Record Session 1 outcome for Session 2 gate
        if(m_session_number == 1)
        {
            m_session_1_was_positive = (session_return_pct > 0.0);
            m_session1_return_pct    = session_return_pct;
        }
    }

    // ------------------------------------------------------------------
    // HasBudgetForTrade — check if enough session risk budget remains
    // Budget remaining > 1.5% (enough for another full-risk trade)
    // ------------------------------------------------------------------
    bool HasBudgetForTrade()
    {
        if(IsSessionHalted()) return false;
        if(SessionCapReached()) return false;

        // Session budget remaining as a percentage
        double used_pct = MathAbs(m_session_pnl < 0 ? m_session_pnl : 0.0);
        double remaining = m_session_risk_cap - used_pct;

        double effective_risk = GetEffectiveRiskPct();
        bool enough = (remaining >= effective_risk);

        if(!enough)
        {
            PrintFormat("[RiskManager] Insufficient budget: %.2f%% remaining, need %.2f%%",
                        remaining, effective_risk);
        }
        return enough;
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    double  GetSessionPnL()         { return m_session_pnl; }
    double  GetSessionPnLAmount()   { return m_closed_pnl_session; }
    int     GetSessionNumber()      { return m_session_number; }
    int     GetTradesThisSession()  { return m_trades_this_session; }
    double  GetSessionStartBalance(){ return m_session_start_balance; }
    int     GetDailySessionCount()  { return m_daily_session_count; }
    double  GetRiskPct()            { return m_risk_pct; }
    double  GetEffectiveRisk()      { return GetEffectiveRiskPct(); }

    // ------------------------------------------------------------------
    // UpdateFromBalance — call periodically to keep session P&L current
    // ------------------------------------------------------------------
    void UpdateFromBalance()
    {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        if(m_session_start_balance > 0)
        {
            m_session_pnl = ((balance - m_session_start_balance) / m_session_start_balance) * 100.0;
        }
    }
};

#endif // RISKMANAGER_MQH
