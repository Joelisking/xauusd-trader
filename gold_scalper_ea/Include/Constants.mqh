//+------------------------------------------------------------------+
//| Constants.mqh — Shared enums, constants, and input parameters     |
//| Used by both Gold Scalper EA and Gold Swing Rider EA              |
//+------------------------------------------------------------------+
#ifndef CONSTANTS_MQH
#define CONSTANTS_MQH

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

enum ENUM_DIRECTION
{
    DIRECTION_NONE = 0,
    DIRECTION_BULL = 1,
    DIRECTION_BEAR = 2
};

enum ENUM_MARKET_STRUCTURE
{
    STRUCT_NONE = 0,
    STRUCT_HH   = 1,   // Higher High
    STRUCT_HL   = 2,   // Higher Low
    STRUCT_LH   = 3,   // Lower High
    STRUCT_LL   = 4    // Lower Low
};

enum ENUM_CANDLE_PATTERN
{
    PATTERN_NONE             = 0,
    PATTERN_HAMMER           = 1,
    PATTERN_INVERTED_HAMMER  = 2,
    PATTERN_ENGULFING_BULL   = 3,
    PATTERN_ENGULFING_BEAR   = 4,
    PATTERN_PIN_BAR_BULL     = 5,
    PATTERN_PIN_BAR_BEAR     = 6,
    PATTERN_DOJI             = 7,
    PATTERN_SHOOTING_STAR    = 8,
    PATTERN_MORNING_STAR     = 9,
    PATTERN_EVENING_STAR     = 10,
    PATTERN_THREE_WHITE      = 11,  // Three White Soldiers
    PATTERN_THREE_BLACK      = 12,  // Three Black Crows
    PATTERN_TWEEZER_TOP      = 13,
    PATTERN_TWEEZER_BOTTOM   = 14
};

enum ENUM_NEWS_PHASE
{
    NEWS_PHASE_NONE      = 0,
    NEWS_PHASE_DETECTION = 1,  // T-60 min
    NEWS_PHASE_PRE       = 2,  // T-30 min
    NEWS_PHASE_DURING    = 3,  // T-0 to T+20 min
    NEWS_PHASE_POST      = 4   // T+20 to T+75 min
};

enum ENUM_CASCADE_STATE
{
    CASCADE_IDLE      = 0,
    CASCADE_PILOT     = 1,
    CASCADE_CORE      = 2,
    CASCADE_ADD       = 3,
    CASCADE_MAX       = 4,
    CASCADE_CANCELLED = 5
};

enum ENUM_SESSION
{
    SESSION_ASIAN           = 0,
    SESSION_LONDON_OPEN     = 1,
    SESSION_LONDON          = 2,
    SESSION_LONDON_NY_OVERLAP = 3,
    SESSION_NY              = 4,
    SESSION_NY_CLOSE        = 5
};

enum ENUM_REGIME
{
    REGIME_TRENDING = 0,
    REGIME_RANGING  = 1,
    REGIME_CRISIS   = 2
};

// ---------------------------------------------------------------------------
// Shared Constants
// ---------------------------------------------------------------------------

#define AI_SERVER_HOST       "127.0.0.1"
#define AI_SERVER_DEFAULT_PORT 5001
#define FEATURE_COUNT        127
#define MAX_CASCADE_POSITIONS 4
#define MAGIC_SCALPER        100001
#define MAGIC_SWING          100002

// ---------------------------------------------------------------------------
// Session hours (UTC)
// ---------------------------------------------------------------------------

#define SESSION_ASIAN_START         0
#define SESSION_ASIAN_END           7
#define SESSION_LONDON_OPEN_START   7
#define SESSION_LONDON_OPEN_END     10
#define SESSION_LONDON_START_HOUR   10
#define SESSION_LONDON_END_HOUR     13
#define SESSION_OVERLAP_START       13
#define SESSION_OVERLAP_END         17
#define SESSION_NY_START            17
#define SESSION_NY_END              21
#define SESSION_NY_CLOSE_START      21
#define SESSION_NY_CLOSE_END        24

#endif // CONSTANTS_MQH
