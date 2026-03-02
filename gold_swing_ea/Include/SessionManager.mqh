//+------------------------------------------------------------------+
//| SessionManager.mqh — UTC session identification + activity gates   |
//+------------------------------------------------------------------+
#ifndef SESSIONMANAGER_MQH
#define SESSIONMANAGER_MQH

#include "Constants.mqh"

class CSessionManager
{
private:
    ENUM_SESSION m_current_session;
    int          m_current_hour;

public:
    CSessionManager()
    {
        m_current_session = SESSION_ASIAN;
        m_current_hour = 0;
    }

    // ------------------------------------------------------------------
    // Update — call periodically
    // ------------------------------------------------------------------

    void Update()
    {
        MqlDateTime dt;
        TimeGMT(dt);
        m_current_hour = dt.hour;
        m_current_session = GetSessionForHour(m_current_hour);
    }

    // ------------------------------------------------------------------
    // Session identification
    // ------------------------------------------------------------------

    ENUM_SESSION GetCurrentSession()
    {
        Update();
        return m_current_session;
    }

    ENUM_SESSION GetSessionForHour(int utc_hour)
    {
        if(utc_hour >= SESSION_ASIAN_START && utc_hour < SESSION_LONDON_OPEN_START)
            return SESSION_ASIAN;
        if(utc_hour >= SESSION_LONDON_OPEN_START && utc_hour < SESSION_LONDON_START_HOUR)
            return SESSION_LONDON_OPEN;
        if(utc_hour >= SESSION_LONDON_START_HOUR && utc_hour < SESSION_OVERLAP_START)
            return SESSION_LONDON;
        if(utc_hour >= SESSION_OVERLAP_START && utc_hour < SESSION_NY_START)
            return SESSION_LONDON_NY_OVERLAP;
        if(utc_hour >= SESSION_NY_START && utc_hour < SESSION_NY_CLOSE_START)
            return SESSION_NY;

        return SESSION_NY_CLOSE;
    }

    // ------------------------------------------------------------------
    // Scalper activity gate
    // ------------------------------------------------------------------

    bool IsScalperActive()
    {
        Update();
        // Active: London Open, London, London-NY Overlap, NY
        return (m_current_session == SESSION_LONDON_OPEN ||
                m_current_session == SESSION_LONDON ||
                m_current_session == SESSION_LONDON_NY_OVERLAP ||
                m_current_session == SESSION_NY);
    }

    // ------------------------------------------------------------------
    // Swing entry gate
    // ------------------------------------------------------------------

    bool IsSwingEntryAllowed()
    {
        Update();
        // London Open (overnight setup), London, London-NY
        return (m_current_session == SESSION_LONDON_OPEN ||
                m_current_session == SESSION_LONDON ||
                m_current_session == SESSION_LONDON_NY_OVERLAP);
    }

    // ------------------------------------------------------------------
    // Position size multiplier (scalper)
    // ------------------------------------------------------------------

    double GetPositionSizeMultiplier()
    {
        Update();
        switch(m_current_session)
        {
            case SESSION_LONDON_OPEN:       return 0.50;
            case SESSION_LONDON:            return 1.00;
            case SESSION_LONDON_NY_OVERLAP: return 1.00;
            case SESSION_NY:                return 0.75;
            default:                        return 0.00;  // Asian, NY Close
        }
    }

    // ------------------------------------------------------------------
    // Convenience
    // ------------------------------------------------------------------

    bool IsOverlap()
    {
        Update();
        return (m_current_session == SESSION_LONDON_NY_OVERLAP);
    }

    int GetCurrentHour() { return m_current_hour; }

    bool IsWeekend()
    {
        MqlDateTime dt;
        TimeGMT(dt);
        // Saturday=6, Sunday=0
        return (dt.day_of_week == 0 || dt.day_of_week == 6);
    }

    string GetSessionName()
    {
        switch(m_current_session)
        {
            case SESSION_ASIAN:              return "Asian";
            case SESSION_LONDON_OPEN:        return "London Open";
            case SESSION_LONDON:             return "London";
            case SESSION_LONDON_NY_OVERLAP:  return "London-NY Overlap";
            case SESSION_NY:                 return "New York";
            case SESSION_NY_CLOSE:           return "NY Close";
            default:                         return "Unknown";
        }
    }
};

#endif // SESSIONMANAGER_MQH
