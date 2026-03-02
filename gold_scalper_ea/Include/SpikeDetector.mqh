//+------------------------------------------------------------------+
//| SpikeDetector.mqh — ATR(1)/ATR(20) spike detection + cooldown     |
//+------------------------------------------------------------------+
#ifndef SPIKEDETECTOR_MQH
#define SPIKEDETECTOR_MQH

#include "Constants.mqh"

class CSpikeDetector
{
private:
    double   m_ratio_threshold;    // 3.0
    int      m_cooldown_seconds;   // 600 = 10 minutes
    datetime m_cooldown_end;
    bool     m_spike_active;
    int      m_atr1_handle;
    int      m_atr20_handle;
    double   m_last_ratio;

public:
    CSpikeDetector()
    {
        m_ratio_threshold  = 3.0;
        m_cooldown_seconds = 600;
        m_cooldown_end     = 0;
        m_spike_active     = false;
        m_atr1_handle      = INVALID_HANDLE;
        m_atr20_handle     = INVALID_HANDLE;
        m_last_ratio       = 0.0;
    }

    ~CSpikeDetector()
    {
        if(m_atr1_handle != INVALID_HANDLE)
            IndicatorRelease(m_atr1_handle);
        if(m_atr20_handle != INVALID_HANDLE)
            IndicatorRelease(m_atr20_handle);
    }

    bool Init(string symbol)
    {
        m_atr1_handle  = iATR(symbol, PERIOD_M1, 1);
        m_atr20_handle = iATR(symbol, PERIOD_M1, 20);

        if(m_atr1_handle == INVALID_HANDLE || m_atr20_handle == INVALID_HANDLE)
        {
            Print("[SpikeDetector] Failed to create ATR indicators");
            return false;
        }
        return true;
    }

    void SetThreshold(double ratio) { m_ratio_threshold = ratio; }
    void SetCooldown(int seconds)   { m_cooldown_seconds = seconds; }

    // ------------------------------------------------------------------
    // Update — call every tick
    // ------------------------------------------------------------------

    void Update(string symbol)
    {
        datetime now = TimeGMT();

        // Check if cooldown expired
        if(m_spike_active && now >= m_cooldown_end)
        {
            m_spike_active = false;
            Print("[SpikeDetector] Cooldown expired — entries resumed");
        }

        if(m_atr1_handle == INVALID_HANDLE || m_atr20_handle == INVALID_HANDLE)
            return;

        double atr1[1], atr20[1];
        if(CopyBuffer(m_atr1_handle, 0, 0, 1, atr1) <= 0) return;
        if(CopyBuffer(m_atr20_handle, 0, 0, 1, atr20) <= 0) return;

        if(atr20[0] == 0.0) return;

        m_last_ratio = atr1[0] / atr20[0];

        if(m_last_ratio > m_ratio_threshold && !m_spike_active)
        {
            m_spike_active = true;
            m_cooldown_end = now + m_cooldown_seconds;
            PrintFormat("[SpikeDetector] SPIKE detected! ATR ratio=%.2f, cooldown until %s",
                        m_last_ratio, TimeToString(m_cooldown_end));
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    bool     IsActive()       { return m_spike_active; }
    datetime GetCooldownEnd() { return m_cooldown_end; }
    double   GetLastRatio()   { return m_last_ratio; }
};

#endif // SPIKEDETECTOR_MQH
