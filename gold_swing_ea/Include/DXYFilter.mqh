//+------------------------------------------------------------------+
//| DXYFilter.mqh — US Dollar Index macro filter                      |
//| Reads dxy_trend.json written by Python macro feed every 15 min   |
//| Gold moves inversely to USD — DXY headwinds reduce/close longs   |
//+------------------------------------------------------------------+
#ifndef DXYFILTER_MQH
#define DXYFILTER_MQH

#include "Constants.mqh"

class CDXYFilter
{
private:
    string          m_data_file;           // Path to dxy_trend.json
    ENUM_DIRECTION  m_dxy_trend;           // Cached DXY direction
    double          m_dxy_momentum;        // Rate of change, positive = up
    double          m_dxy_ema_distance;    // Distance from 50 EMA (normalised)
    datetime        m_last_read_time;      // When file was last read
    int             m_read_interval_sec;   // Minimum seconds between file reads (900 = 15 min)
    bool            m_valid;               // True if last read succeeded
    string          m_last_error;

    // H4 DXY candle tracking for 3-candle rally detection
    double          m_h4_dxy_closes[];     // Ring buffer of H4 DXY close prices
    int             m_h4_ring_idx;
    int             m_h4_ring_size;        // 6 bars — enough to detect 3-candle sequences
    datetime        m_last_h4_candle;      // Timestamp of last H4 bar read

    // ------------------------------------------------------------------
    // JSON helpers (no external lib in MQL5)
    // ------------------------------------------------------------------

    string GetJsonString(const string &json, const string &key)
    {
        string search = "\"" + key + "\"";
        int pos = StringFind(json, search);
        if(pos < 0) return "";

        int colon = StringFind(json, ":", pos + StringLen(search));
        if(colon < 0) return "";

        int start = StringFind(json, "\"", colon + 1);
        if(start < 0) return "";
        start++;

        int end = StringFind(json, "\"", start);
        if(end < 0) return "";

        return StringSubstr(json, start, end - start);
    }

    double GetJsonDouble(const string &json, const string &key)
    {
        string search = "\"" + key + "\"";
        int pos = StringFind(json, search);
        if(pos < 0) return 0.0;

        int colon = StringFind(json, ":", pos + StringLen(search));
        if(colon < 0) return 0.0;

        string remainder = StringSubstr(json, colon + 1);
        StringTrimLeft(remainder);

        string numStr = "";
        for(int i = 0; i < StringLen(remainder); i++)
        {
            ushort ch = StringGetCharacter(remainder, i);
            if((ch >= '0' && ch <= '9') || ch == '-' || ch == '.')
                numStr += CharToString((uchar)ch);
            else if(StringLen(numStr) > 0)
                break;
        }

        return StringToDouble(numStr);
    }

    // ------------------------------------------------------------------
    // Read and parse the JSON file
    // ------------------------------------------------------------------

    bool ReadDXYFile()
    {
        int handle = FileOpen(m_data_file, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
        if(handle == INVALID_HANDLE)
        {
            // Try without FILE_COMMON (local data folder)
            handle = FileOpen(m_data_file, FILE_READ | FILE_TXT | FILE_ANSI);
            if(handle == INVALID_HANDLE)
            {
                m_last_error = "Cannot open " + m_data_file + " err=" + IntegerToString(GetLastError());
                m_valid = false;
                return false;
            }
        }

        string content = "";
        while(!FileIsEnding(handle))
            content += FileReadString(handle);
        FileClose(handle);

        if(StringLen(content) == 0)
        {
            m_last_error = "Empty file: " + m_data_file;
            m_valid = false;
            return false;
        }

        // Parse trend field: "UP", "DOWN", "NEUTRAL"
        string trend_str = GetJsonString(content, "trend");
        if(trend_str == "UP")
            m_dxy_trend = DIRECTION_BULL;
        else if(trend_str == "DOWN")
            m_dxy_trend = DIRECTION_BEAR;
        else
            m_dxy_trend = DIRECTION_NONE;

        // Parse momentum and ema_distance (optional fields — default 0 if absent)
        m_dxy_momentum    = GetJsonDouble(content, "momentum");
        m_dxy_ema_distance = GetJsonDouble(content, "ema_distance");

        m_last_read_time = TimeGMT();
        m_valid = true;
        m_last_error = "";

        PrintFormat("[DXYFilter] DXY trend=%s momentum=%.4f ema_dist=%.4f",
                    trend_str, m_dxy_momentum, m_dxy_ema_distance);

        return true;
    }

    // ------------------------------------------------------------------
    // Update the H4 DXY candle ring buffer
    // Reads DXY H4 closes from file field "h4_closes" array if present,
    // or uses the m_dxy_momentum trend proxy when no history available.
    // ------------------------------------------------------------------

    void UpdateH4CandleBuffer()
    {
        // Read fresh file data for H4 array
        int handle = FileOpen(m_data_file, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
        if(handle == INVALID_HANDLE)
            handle = FileOpen(m_data_file, FILE_READ | FILE_TXT | FILE_ANSI);
        if(handle == INVALID_HANDLE)
            return;

        string content = "";
        while(!FileIsEnding(handle))
            content += FileReadString(handle);
        FileClose(handle);

        // Look for "h4_closes" array in JSON: "h4_closes":[103.1,103.3,103.5,...]
        string key = "\"h4_closes\"";
        int pos = StringFind(content, key);
        if(pos < 0)
            return;  // Field not present; 3-candle detection falls back to momentum

        int arr_start = StringFind(content, "[", pos);
        int arr_end   = StringFind(content, "]", arr_start);
        if(arr_start < 0 || arr_end < 0)
            return;

        string arr_str = StringSubstr(content, arr_start + 1, arr_end - arr_start - 1);

        // Parse comma-separated doubles
        string parts[];
        int count = StringSplit(arr_str, ',', parts);
        if(count <= 0)
            return;

        int write_count = MathMin(count, m_h4_ring_size);
        ArrayResize(m_h4_dxy_closes, write_count);
        for(int i = 0; i < write_count; i++)
        {
            StringTrimLeft(parts[i]);
            StringTrimRight(parts[i]);
            m_h4_dxy_closes[i] = StringToDouble(parts[i]);
        }
    }

public:
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    CDXYFilter()
    {
        m_data_file         = "dxy_trend.json";
        m_dxy_trend         = DIRECTION_NONE;
        m_dxy_momentum      = 0.0;
        m_dxy_ema_distance  = 0.0;
        m_last_read_time    = 0;
        m_read_interval_sec = 900;   // 15 minutes
        m_valid             = false;
        m_last_error        = "";
        m_h4_ring_idx       = 0;
        m_h4_ring_size      = 6;
        m_last_h4_candle    = 0;
        ArrayResize(m_h4_dxy_closes, 0);
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    void SetDataFile(string path)         { m_data_file = path; }
    void SetReadInterval(int seconds)     { m_read_interval_sec = seconds; }
    bool IsValid()                        { return m_valid; }
    string GetLastError()                 { return m_last_error; }

    // ------------------------------------------------------------------
    // Refresh — reads file if interval has elapsed or first call
    // ------------------------------------------------------------------

    void Refresh()
    {
        datetime now = TimeGMT();
        if(!m_valid || (now - m_last_read_time) >= m_read_interval_sec)
        {
            ReadDXYFile();
            UpdateH4CandleBuffer();
        }
    }

    // ------------------------------------------------------------------
    // GetDXYTrend — returns UP / DOWN / NEUTRAL
    // Always refreshes if stale before returning
    // ------------------------------------------------------------------

    ENUM_DIRECTION GetDXYTrend()
    {
        Refresh();
        return m_dxy_trend;
    }

    // ------------------------------------------------------------------
    // IsMacroHeadwind — true if DXY direction opposes the gold position
    //
    // Gold moves inversely to USD:
    //   Gold LONG  + DXY rising  = headwind  -> true
    //   Gold SHORT + DXY falling = headwind  -> true
    //   All other combos         = tailwind  -> false
    // ------------------------------------------------------------------

    bool IsMacroHeadwind(ENUM_DIRECTION gold_direction)
    {
        Refresh();

        if(m_dxy_trend == DIRECTION_NONE)
            return false;   // Neutral DXY — not a headwind

        if(gold_direction == DIRECTION_BULL && m_dxy_trend == DIRECTION_BULL)
            return true;    // DXY rising against gold long

        if(gold_direction == DIRECTION_BEAR && m_dxy_trend == DIRECTION_BEAR)
            return true;    // DXY falling (USD weak) against gold short

        return false;
    }

    // ------------------------------------------------------------------
    // HasThreeCandleRally — detects 3 consecutive H4 DXY closes moving
    // in the direction that opposes 'against' (gold's direction).
    //
    // For a gold LONG (against = DIRECTION_BULL):
    //   We look for 3 consecutive H4 DXY closes that are HIGHER than the
    //   previous close — i.e., DXY rallying against the gold long.
    //
    // For a gold SHORT (against = DIRECTION_BEAR):
    //   We look for 3 consecutive H4 DXY closes that are LOWER — DXY
    //   falling, which is a tailwind for gold (headwind for short).
    // ------------------------------------------------------------------

    bool HasThreeCandleRally(ENUM_DIRECTION against)
    {
        Refresh();

        int sz = ArraySize(m_h4_dxy_closes);

        // Need at least 4 closes to compare 3 consecutive moves
        if(sz < 4)
        {
            // Fallback: use momentum field if available
            // Strong momentum (absolute value > threshold) in opposing direction
            if(against == DIRECTION_BULL && m_dxy_momentum > 0.3)
                return true;
            if(against == DIRECTION_BEAR && m_dxy_momentum < -0.3)
                return true;
            return false;
        }

        // Count consecutive closes moving against the gold position
        // Array index 0 = most recent close, index 1 = one bar earlier, etc.
        int rally_count = 0;

        if(against == DIRECTION_BULL)
        {
            // Gold is long — DXY rally = rising closes = headwind
            for(int i = 0; i < sz - 1 && rally_count < 3; i++)
            {
                if(m_h4_dxy_closes[i] > m_h4_dxy_closes[i + 1])
                    rally_count++;
                else
                    break;
            }
        }
        else if(against == DIRECTION_BEAR)
        {
            // Gold is short — DXY decline = falling closes = headwind for short
            for(int i = 0; i < sz - 1 && rally_count < 3; i++)
            {
                if(m_h4_dxy_closes[i] < m_h4_dxy_closes[i + 1])
                    rally_count++;
                else
                    break;
            }
        }

        return (rally_count >= 3);
    }

    // ------------------------------------------------------------------
    // Diagnostic
    // ------------------------------------------------------------------

    string GetStatusString()
    {
        string trend_str = "NEUTRAL";
        if(m_dxy_trend == DIRECTION_BULL) trend_str = "UP";
        if(m_dxy_trend == DIRECTION_BEAR) trend_str = "DOWN";

        return StringFormat("DXY[trend=%s mom=%.3f ema_dist=%.3f valid=%s last_read=%s]",
                            trend_str,
                            m_dxy_momentum,
                            m_dxy_ema_distance,
                            m_valid ? "YES" : "NO",
                            TimeToString(m_last_read_time, TIME_DATE | TIME_MINUTES));
    }
};

#endif // DXYFILTER_MQH
