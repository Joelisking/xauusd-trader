//+------------------------------------------------------------------+
//| AIClient.mqh — TCP socket client to Python AI server              |
//| Sends JSON requests, parses responses. Manages fallback mode.     |
//+------------------------------------------------------------------+
#ifndef AICLIENT_MQH
#define AICLIENT_MQH

#include "Constants.mqh"
#include <Trade\Trade.mqh>

class CAIClient
{
private:
    int     m_socket;
    string  m_host;
    int     m_port;
    int     m_consecutive_failures;
    int     m_max_failures;          // 3 = enter fallback
    int     m_connect_timeout_ms;    // 5000
    int     m_read_timeout_ms;       // 3000 scalper, 5000 swing
    string  m_last_error;

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    bool SendRaw(string data)
    {
        if(m_socket == INVALID_HANDLE)
            return false;

        char req[];
        int len = StringToCharArray(data, req, 0, WHOLE_ARRAY, CP_UTF8);
        // StringToCharArray adds null terminator, we want raw bytes
        if(len <= 0)
            return false;

        int sent = SocketSend(m_socket, req, len - 1); // exclude null
        return (sent == len - 1);
    }

    string ReceiveRaw()
    {
        if(m_socket == INVALID_HANDLE)
            return "";

        char rsp[];
        string result = "";
        uint timeout = GetTickCount() + (uint)m_read_timeout_ms;

        while(GetTickCount() < timeout)
        {
            uint available = SocketIsReadable(m_socket);
            if(available > 0)
            {
                int read = SocketRead(m_socket, rsp, (int)available, 500);
                if(read > 0)
                {
                    result += CharArrayToString(rsp, 0, read, CP_UTF8);
                    // Check for newline delimiter
                    if(StringFind(result, "\n") >= 0)
                        break;
                }
            }
            Sleep(10);
        }

        // Strip trailing newline
        StringTrimRight(result);
        return result;
    }

    // Simple JSON value extraction helpers (no external JSON lib in MQL5)
    string GetJsonString(const string &json, const string &key)
    {
        string search = "\"" + key + "\"";
        int pos = StringFind(json, search);
        if(pos < 0) return "";

        // Find the colon after the key
        int colon = StringFind(json, ":", pos + StringLen(search));
        if(colon < 0) return "";

        // Find opening quote of value
        int start = StringFind(json, "\"", colon + 1);
        if(start < 0) return "";
        start++;

        int end = StringFind(json, "\"", start);
        if(end < 0) return "";

        return StringSubstr(json, start, end - start);
    }

    int GetJsonInt(const string &json, const string &key)
    {
        string search = "\"" + key + "\"";
        int pos = StringFind(json, search);
        if(pos < 0) return 0;

        int colon = StringFind(json, ":", pos + StringLen(search));
        if(colon < 0) return 0;

        // Extract numeric value after colon
        string remainder = StringSubstr(json, colon + 1);
        StringTrimLeft(remainder);

        // Read until non-numeric
        string numStr = "";
        for(int i = 0; i < StringLen(remainder); i++)
        {
            ushort ch = StringGetCharacter(remainder, i);
            if((ch >= '0' && ch <= '9') || ch == '-')
                numStr += CharToString((uchar)ch);
            else if(StringLen(numStr) > 0)
                break;
        }

        return (int)StringToInteger(numStr);
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

    bool GetJsonBool(const string &json, const string &key)
    {
        string search = "\"" + key + "\"";
        int pos = StringFind(json, search);
        if(pos < 0) return false;

        int colon = StringFind(json, ":", pos + StringLen(search));
        if(colon < 0) return false;

        string remainder = StringSubstr(json, colon + 1);
        StringTrimLeft(remainder);

        return (StringFind(remainder, "true") == 0);
    }

    // Build JSON feature array string
    string BuildFeatureArray(const double &features[])
    {
        string result = "[";
        int count = ArraySize(features);
        for(int i = 0; i < count; i++)
        {
            result += DoubleToString(features[i], 6);
            if(i < count - 1)
                result += ",";
        }
        result += "]";
        return result;
    }

public:
    // ------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------

    CAIClient()
    {
        m_socket = INVALID_HANDLE;
        m_host = AI_SERVER_HOST;
        m_port = AI_SERVER_DEFAULT_PORT;
        m_consecutive_failures = 0;
        m_max_failures = 3;
        m_connect_timeout_ms = 5000;
        m_read_timeout_ms = 3000;
        m_last_error = "";
    }

    ~CAIClient()
    {
        Disconnect();
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    void SetPort(int port)               { m_port = port; }
    void SetReadTimeout(int ms)          { m_read_timeout_ms = ms; }
    void SetMaxFailures(int n)           { m_max_failures = n; }
    string GetLastError()                { return m_last_error; }

    // ------------------------------------------------------------------
    // Connection management
    // ------------------------------------------------------------------

    bool Connect()
    {
        if(m_socket != INVALID_HANDLE)
            Disconnect();

        m_socket = SocketCreate();
        if(m_socket == INVALID_HANDLE)
        {
            m_last_error = "SocketCreate failed: " + IntegerToString(GetLastError());
            return false;
        }

        if(!SocketConnect(m_socket, m_host, m_port, m_connect_timeout_ms))
        {
            m_last_error = "SocketConnect failed to " + m_host + ":" + IntegerToString(m_port);
            SocketClose(m_socket);
            m_socket = INVALID_HANDLE;
            return false;
        }

        PrintFormat("[AIClient] Connected to %s:%d", m_host, m_port);
        return true;
    }

    void Disconnect()
    {
        if(m_socket != INVALID_HANDLE)
        {
            SocketClose(m_socket);
            m_socket = INVALID_HANDLE;
            Print("[AIClient] Disconnected");
        }
    }

    bool IsConnected() { return m_socket != INVALID_HANDLE; }

    // ------------------------------------------------------------------
    // Fallback mode
    // ------------------------------------------------------------------

    bool IsInFallbackMode()  { return m_consecutive_failures >= m_max_failures; }
    void ResetFailures()     { m_consecutive_failures = 0; }
    int  GetFailureCount()   { return m_consecutive_failures; }

    // ------------------------------------------------------------------
    // Entry scoring
    // ------------------------------------------------------------------

    bool ScoreEntry(
        ENUM_DIRECTION direction,
        string         timeframe,
        string         bot,
        const double  &features[],
        int           &entry_score,
        int           &trend_score,
        int           &news_risk,
        string        &regime,
        string        &wyckoff_phase,
        bool          &approve,
        double        &lot_multiplier
    )
    {
        if(IsInFallbackMode())
        {
            m_last_error = "In fallback mode";
            return false;
        }

        // Reconnect if needed
        if(!IsConnected())
        {
            if(!Connect())
            {
                m_consecutive_failures++;
                return false;
            }
        }

        // Build JSON request
        string dir_str = (direction == DIRECTION_BULL) ? "BUY" : "SELL";

        MqlDateTime dt;
        TimeGMT(dt);

        string json = "{";
        json += "\"type\":\"entry_check\",";
        json += "\"symbol\":\"XAUUSD\",";
        json += "\"direction\":\"" + dir_str + "\",";
        json += "\"timeframe\":\"" + timeframe + "\",";
        json += "\"bot\":\"" + bot + "\",";
        json += "\"session_hour\":" + IntegerToString(dt.hour) + ",";
        json += "\"current_spread\":" + DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID), 2) + ",";
        json += "\"features\":" + BuildFeatureArray(features);
        json += "}\n";

        // Send
        if(!SendRaw(json))
        {
            m_consecutive_failures++;
            m_last_error = "Send failed";
            Disconnect();
            return false;
        }

        // Receive
        string response = ReceiveRaw();
        if(StringLen(response) == 0)
        {
            m_consecutive_failures++;
            m_last_error = "No response (timeout)";
            Disconnect();
            return false;
        }

        // Check for error
        string error = GetJsonString(response, "error");
        if(StringLen(error) > 0)
        {
            m_last_error = "Server error: " + error;
            m_consecutive_failures++;
            return false;
        }

        // Parse response
        entry_score    = GetJsonInt(response, "entry_score");
        trend_score    = GetJsonInt(response, "trend_score");
        news_risk      = GetJsonInt(response, "news_risk");
        regime         = GetJsonString(response, "regime");
        wyckoff_phase  = GetJsonString(response, "wyckoff_phase");
        approve        = GetJsonBool(response, "approve");
        lot_multiplier = GetJsonDouble(response, "recommended_lot_multiplier");

        // Success — reset failure counter
        ResetFailures();
        return true;
    }

    // ------------------------------------------------------------------
    // Heartbeat
    // ------------------------------------------------------------------

    bool Heartbeat(string &status, int &uptime, string &model_version)
    {
        if(!IsConnected())
        {
            if(!Connect())
            {
                m_consecutive_failures++;
                return false;
            }
        }

        string json = "{\"type\":\"heartbeat\"}\n";

        if(!SendRaw(json))
        {
            m_consecutive_failures++;
            Disconnect();
            return false;
        }

        string response = ReceiveRaw();
        if(StringLen(response) == 0)
        {
            m_consecutive_failures++;
            Disconnect();
            return false;
        }

        status        = GetJsonString(response, "status");
        uptime        = GetJsonInt(response, "uptime_seconds");
        model_version = GetJsonString(response, "model_version");

        ResetFailures();
        return true;
    }
};

#endif // AICLIENT_MQH
