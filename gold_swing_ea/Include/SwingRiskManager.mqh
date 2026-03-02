//+------------------------------------------------------------------+
//| SwingRiskManager.mqh — Position sizing, SL/TP calculation        |
//| Lot = (AccountBalance * 0.02) / (sl_pips * pip_value)           |
//| SL: below last H4 swing low (longs) / above H4 swing high (shorts)|
//| TP1: entry + (SL_distance * 1.5), close 40%                     |
//| TP2: entry + (SL_distance * 3.0), close remaining 60%           |
//+------------------------------------------------------------------+
#ifndef SWINGRISKMANAGER_MQH
#define SWINGRISKMANAGER_MQH

#include "Constants.mqh"
#include <Trade\Trade.mqh>

class CSwingRiskManager
{
private:
    double  m_risk_percent;         // Default 2.0%
    double  m_tp1_rr;               // Default 1.5
    double  m_tp2_rr;               // Default 3.0
    double  m_tp1_close_pct;        // Default 40.0%
    double  m_max_lot_pct;          // Hard cap: 5% of account
    string  m_symbol;

    // Calculated trade parameters — stored after CalcLotSize()
    double  m_last_lot_size;
    double  m_last_sl_price;
    double  m_last_tp1_price;
    double  m_last_tp2_price;
    double  m_last_entry_price;
    double  m_last_sl_pips;

    // ------------------------------------------------------------------
    // Get pip value for the symbol
    // For XAUUSD: 1 pip = $1 per 0.01 lot = $100 per 1.0 lot per pip
    // MT5 reports tick value per tick (1 point). For gold pips = 10 points.
    // ------------------------------------------------------------------

    double GetPipValue()
    {
        double tick_value = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_VALUE);
        double tick_size  = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
        double point      = SymbolInfoDouble(m_symbol, SYMBOL_POINT);

        if(tick_size <= 0 || point <= 0)
            return 1.0;

        // 1 pip for XAUUSD = 10 points
        double pip_in_points = 10.0;
        double pip_value = tick_value * (pip_in_points * point / tick_size);

        return pip_value;
    }

    // ------------------------------------------------------------------
    // Get account balance
    // ------------------------------------------------------------------

    double GetAccountBalance()
    {
        return AccountInfoDouble(ACCOUNT_BALANCE);
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    CSwingRiskManager()
    {
        m_risk_percent      = 2.0;
        m_tp1_rr            = 1.5;
        m_tp2_rr            = 3.0;
        m_tp1_close_pct     = 40.0;
        m_max_lot_pct       = 5.0;
        m_symbol            = _Symbol;
        m_last_lot_size     = 0.01;
        m_last_sl_price     = 0.0;
        m_last_tp1_price    = 0.0;
        m_last_tp2_price    = 0.0;
        m_last_entry_price  = 0.0;
        m_last_sl_pips      = 0.0;
    }

    // ------------------------------------------------------------------
    // Initialise
    // ------------------------------------------------------------------

    void Init(string symbol, double risk_pct, double tp1_rr, double tp2_rr, double tp1_close_pct)
    {
        m_symbol        = symbol;
        m_risk_percent  = risk_pct;
        m_tp1_rr        = tp1_rr;
        m_tp2_rr        = tp2_rr;
        m_tp1_close_pct = tp1_close_pct;
    }

    // ------------------------------------------------------------------
    // CalcLotSize — core position sizing formula
    //
    // Lot = (AccountBalance * RiskPct/100) / (sl_pips * pip_value)
    //
    // sl_pips: distance from entry to stop loss in pips
    // Returns the lot size, capped at m_max_lot_pct of account margin
    // ------------------------------------------------------------------

    double CalcLotSize(double sl_pips)
    {
        if(sl_pips <= 0)
        {
            Print("[SwingRisk] ERROR: sl_pips <= 0, using minimum lot");
            return SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
        }

        double balance   = GetAccountBalance();
        double pip_value = GetPipValue();

        if(pip_value <= 0)
        {
            Print("[SwingRisk] ERROR: pip_value <= 0");
            return SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
        }

        double risk_amount = balance * (m_risk_percent / 100.0);
        double raw_lot     = risk_amount / (sl_pips * pip_value);

        // Apply hard cap: max lot = 5% account value at current price
        // Approximate max lot: (balance * 0.05) / (price * contract_size * margin_rate)
        // Simplified: cap at account balance * max_lot_pct / (price * pip_value * 100)
        double vol_step = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
        double vol_min  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
        double vol_max  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MAX);

        // Round to symbol's volume step
        double lot = MathFloor(raw_lot / vol_step) * vol_step;

        // Enforce min
        if(lot < vol_min)
            lot = vol_min;

        // Hard cap: never exceed 5% of account balance in notional exposure
        // For gold at ~3000: 1 lot = $3000 * 100 (contract size) = $300,000 notional
        // 5% cap = balance * 0.05 / pip_value per pip / 100 pips
        // Practical approximation: cap at 20x the risk_amount lot equivalent
        double max_lot_cap = (balance * (m_max_lot_pct / 100.0)) / (50.0 * pip_value);
        max_lot_cap = MathFloor(max_lot_cap / vol_step) * vol_step;
        if(max_lot_cap < vol_min) max_lot_cap = vol_min;

        if(lot > max_lot_cap)
        {
            PrintFormat("[SwingRisk] Lot %.2f capped to %.2f (5%% account cap)",
                        lot, max_lot_cap);
            lot = max_lot_cap;
        }

        // Enforce exchange max
        if(lot > vol_max)
            lot = vol_max;

        m_last_lot_size  = lot;
        m_last_sl_pips   = sl_pips;

        PrintFormat("[SwingRisk] CalcLot: balance=%.2f risk=%.2f sl_pips=%.1f pip_val=%.4f lot=%.2f",
                    balance, risk_amount, sl_pips, pip_value, lot);

        return lot;
    }

    // ------------------------------------------------------------------
    // CalcSLPrice — SL below last H4 swing low (long) / above high (short)
    // Adds a 5-pip buffer below the swing point
    // ------------------------------------------------------------------

    double CalcSLPrice(ENUM_DIRECTION direction, double swing_price)
    {
        double point  = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
        double pip    = point * 10.0;
        double buffer = 5.0 * pip;  // 5 pip buffer beyond swing point

        if(direction == DIRECTION_BULL)
            return swing_price - buffer;   // SL below swing low
        else
            return swing_price + buffer;   // SL above swing high
    }

    // ------------------------------------------------------------------
    // CalcAllLevels — calculates lot size and all TP/SL levels at once
    //
    // direction:     BULL = long, BEAR = short
    // entry_price:   planned entry (current ask/bid)
    // swing_price:   last H4 swing low (long) or swing high (short)
    //
    // Outputs: lot_size, sl_price, tp1_price, tp2_price, sl_pips
    // Returns: true if calculation valid
    // ------------------------------------------------------------------

    bool CalcAllLevels(
        ENUM_DIRECTION direction,
        double         entry_price,
        double         swing_price,
        double        &lot_size,
        double        &sl_price,
        double        &tp1_price,
        double        &tp2_price,
        double        &sl_pips_out
    )
    {
        double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
        double pip   = point * 10.0;

        // Calculate SL price
        sl_price = CalcSLPrice(direction, swing_price);

        // Distance from entry to SL in pips
        double sl_distance = MathAbs(entry_price - sl_price);
        double sl_pips     = sl_distance / pip;

        // Validate SL distance: must be at least 20 pips, at most 150 pips
        if(sl_pips < 20.0)
        {
            PrintFormat("[SwingRisk] SL too tight: %.1f pips (min 20)", sl_pips);
            // Use minimum 20-pip SL
            sl_pips = 20.0;
            if(direction == DIRECTION_BULL)
                sl_price = entry_price - (sl_pips * pip);
            else
                sl_price = entry_price + (sl_pips * pip);
        }

        if(sl_pips > 150.0)
        {
            PrintFormat("[SwingRisk] SL too wide: %.1f pips (max 150) — skipping entry", sl_pips);
            return false;
        }

        // Lot size
        lot_size = CalcLotSize(sl_pips);

        // TP levels
        if(direction == DIRECTION_BULL)
        {
            tp1_price = entry_price + (sl_distance * m_tp1_rr);
            tp2_price = entry_price + (sl_distance * m_tp2_rr);
        }
        else
        {
            tp1_price = entry_price - (sl_distance * m_tp1_rr);
            tp2_price = entry_price - (sl_distance * m_tp2_rr);
        }

        sl_pips_out = sl_pips;

        // Cache
        m_last_entry_price = entry_price;
        m_last_sl_price    = sl_price;
        m_last_tp1_price   = tp1_price;
        m_last_tp2_price   = tp2_price;

        PrintFormat("[SwingRisk] Levels: entry=%.2f sl=%.2f (%.1f pips) tp1=%.2f tp2=%.2f lot=%.2f",
                    entry_price, sl_price, sl_pips, tp1_price, tp2_price, lot_size);

        return true;
    }

    // ------------------------------------------------------------------
    // CalcPartialCloseLots — how many lots to close at TP1 (40%)
    // Remainder (60%) stays open for TP2
    // ------------------------------------------------------------------

    double CalcTP1CloseLots(double total_lots)
    {
        double vol_step = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
        double vol_min  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);

        double close_lots = MathFloor((total_lots * m_tp1_close_pct / 100.0) / vol_step) * vol_step;

        // Must close at least minimum and leave at least minimum
        if(close_lots < vol_min)
            close_lots = vol_min;
        if(close_lots >= total_lots)
            close_lots = MathFloor((total_lots - vol_min) / vol_step) * vol_step;
        if(close_lots < vol_min)
            close_lots = 0;  // Cannot split — must close all or none

        return close_lots;
    }

    // ------------------------------------------------------------------
    // CalcHalfCloseLots — 50% close for macro/news/AI exhaustion triggers
    // ------------------------------------------------------------------

    double CalcHalfCloseLots(double total_lots)
    {
        double vol_step = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
        double vol_min  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);

        double close_lots = MathFloor((total_lots * 0.50) / vol_step) * vol_step;

        if(close_lots < vol_min) close_lots = vol_min;
        if(close_lots >= total_lots)
            close_lots = MathFloor((total_lots - vol_min) / vol_step) * vol_step;
        if(close_lots < vol_min) close_lots = 0;

        return close_lots;
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    double GetLastLotSize()    { return m_last_lot_size;    }
    double GetLastSLPrice()    { return m_last_sl_price;    }
    double GetLastTP1Price()   { return m_last_tp1_price;   }
    double GetLastTP2Price()   { return m_last_tp2_price;   }
    double GetLastEntryPrice() { return m_last_entry_price; }
    double GetLastSLPips()     { return m_last_sl_pips;     }
    double GetTP1ClosePct()    { return m_tp1_close_pct;    }
};

#endif // SWINGRISKMANAGER_MQH
