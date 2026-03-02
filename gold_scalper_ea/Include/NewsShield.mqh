//+------------------------------------------------------------------+
//| NewsShield.mqh — 4-phase news protocol                            |
//| Detection (T-60) -> Pre (T-30) -> During (T+0..T+20) -> Post     |
//+------------------------------------------------------------------+
#ifndef NEWSSHIELD_MQH
#define NEWSSHIELD_MQH

#include "Constants.mqh"

class CNewsShield
{
private:
    ENUM_NEWS_PHASE m_phase;
    datetime        m_next_event_time;
    int             m_next_event_impact;    // 0-3
    string          m_next_event_title;
    bool            m_active;

    // Phase durations in seconds
    int m_detection_seconds;    // 3600 = 60 min
    int m_pre_seconds;          // 1800 = 30 min
    int m_during_seconds;       // 1200 = 20 min
    int m_post_seconds;         // 3300 = 55 min (T+20 to T+75)

    // Spread anomaly detection
    double m_spread_history[];
    int    m_spread_history_size;
    int    m_spread_idx;
    double m_spread_anomaly_mult;   // 3.0x = trigger

    void UpdateSpreadHistory(double spread)
    {
        if(ArraySize(m_spread_history) < m_spread_history_size)
        {
            int sz = ArraySize(m_spread_history);
            ArrayResize(m_spread_history, sz + 1);
            m_spread_history[sz] = spread;
        }
        else
        {
            m_spread_history[m_spread_idx % m_spread_history_size] = spread;
        }
        m_spread_idx++;
    }

    double GetAverageSpread()
    {
        int sz = ArraySize(m_spread_history);
        if(sz == 0) return 0.0;
        double sum = 0;
        for(int i = 0; i < sz; i++)
            sum += m_spread_history[i];
        return sum / sz;
    }

public:
    CNewsShield()
    {
        m_phase = NEWS_PHASE_NONE;
        m_next_event_time = 0;
        m_next_event_impact = 0;
        m_next_event_title = "";
        m_active = false;

        m_detection_seconds   = 3600;
        m_pre_seconds         = 1800;
        m_during_seconds      = 1200;
        m_post_seconds        = 3300;

        m_spread_history_size = 20;
        m_spread_idx = 0;
        m_spread_anomaly_mult = 3.0;

        ArrayResize(m_spread_history, 0);
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    void SetNextEvent(datetime event_time, int impact_level, string title = "")
    {
        m_next_event_time   = event_time;
        m_next_event_impact = impact_level;
        m_next_event_title  = title;
        PrintFormat("[NewsShield] Next event: %s (impact %d) at %s",
                    title, impact_level, TimeToString(event_time));
    }

    // ------------------------------------------------------------------
    // Update — call every tick
    // ------------------------------------------------------------------

    void Update()
    {
        datetime now = TimeGMT();

        // Update spread tracking
        double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
        if(point > 0)
        {
            double spread_pips = spread / (point * 10);  // Gold: 1 pip = 10 points
            UpdateSpreadHistory(spread_pips);

            // Spread anomaly detection: immediate activation
            double avg = GetAverageSpread();
            if(avg > 0 && spread_pips > avg * m_spread_anomaly_mult)
            {
                if(m_phase == NEWS_PHASE_NONE)
                {
                    m_phase = NEWS_PHASE_DURING;
                    m_active = true;
                    Print("[NewsShield] SPREAD ANOMALY detected — activating During phase");
                    return;
                }
            }
        }

        // No event scheduled
        if(m_next_event_time == 0 || m_next_event_impact < 2)
        {
            if(m_phase != NEWS_PHASE_NONE && m_phase != NEWS_PHASE_POST)
            {
                m_phase = NEWS_PHASE_NONE;
                m_active = false;
            }
            return;
        }

        int seconds_to_event = (int)(m_next_event_time - now);
        int seconds_after    = (int)(now - m_next_event_time);

        // Determine phase based on time
        if(seconds_to_event > m_detection_seconds)
        {
            // Before detection window
            m_phase  = NEWS_PHASE_NONE;
            m_active = false;
        }
        else if(seconds_to_event > m_pre_seconds)
        {
            // Detection phase: T-60 to T-30
            m_phase  = NEWS_PHASE_DETECTION;
            m_active = true;
        }
        else if(seconds_to_event > 0)
        {
            // Pre-news: T-30 to T-0
            m_phase  = NEWS_PHASE_PRE;
            m_active = true;
        }
        else if(seconds_after < m_during_seconds)
        {
            // During: T-0 to T+20
            m_phase  = NEWS_PHASE_DURING;
            m_active = true;
        }
        else if(seconds_after < m_during_seconds + m_post_seconds)
        {
            // Post-news: T+20 to T+75
            m_phase  = NEWS_PHASE_POST;
            m_active = true;
        }
        else
        {
            // Past the post window — back to normal
            m_phase  = NEWS_PHASE_NONE;
            m_active = false;
            // Clear event
            m_next_event_time = 0;
            m_next_event_impact = 0;
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    bool IsActive()                  { return m_active; }
    ENUM_NEWS_PHASE GetCurrentPhase(){ return m_phase; }
    string GetEventTitle()           { return m_next_event_title; }
    int    GetEventImpact()          { return m_next_event_impact; }

    bool BlocksNewEntries()
    {
        return (m_phase == NEWS_PHASE_PRE ||
                m_phase == NEWS_PHASE_DURING);
    }

    bool IsPostNewsWindow()
    {
        return (m_phase == NEWS_PHASE_POST);
    }

    bool IsDetectionPhase()
    {
        return (m_phase == NEWS_PHASE_DETECTION);
    }

    double GetPostNewsLotMultiplier()
    {
        // 60% of normal during post-news
        if(m_phase == NEWS_PHASE_POST)
            return 0.6;
        return 1.0;
    }
};

#endif // NEWSSHIELD_MQH
