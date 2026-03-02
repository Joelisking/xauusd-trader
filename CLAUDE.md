# CLAUDE.md

## Commands

```bash
uv run pytest tests/ -v          # Run all tests
uv run pytest tests/test_X.py    # Run one file
uv run python -m ai_server       # Start AI server (dev)
uv add <pkg>                     # Add dependency
uv sync                          # Install/sync deps
```

## Project Structure

```
ai_server/           Python AI server (async TCP :5001)
  config.py          Single source of truth for ALL constants — always use this, never hardcode
  protocol.py        JSON message contracts + validation
  server.py          Async TCP server entrypoint
  scoring.py         Entry scoring (dummy now, real models Phase 10)
  models/            Model definitions (BiLSTM, XGBoost, regime, ensemble)
  features/          127-feature calculation pipeline
  macro/             FRED, Alpha Vantage, ForexFactory clients
  training/          Training pipelines, walk-forward, SHAP
gold_scalper_ea/     MQL5 Scalper EA — M1/M5, cascade entry, 15-min max
gold_swing_ea/       MQL5 Swing EA — H1/H4, partial exits, 72h max
data_pipeline/       MT5 export, feature pipeline, macro updater
monitoring/          Telegram bot, watchdog, Grafana
tests/               pytest tests — mirrors ai_server/ structure
architecture/        6 detailed spec docs (read these before modifying anything)
BUILD_PHASES.md      Build roadmap + status tracker (update after completing phases)
```

## Conventions

**Python:**
- Python 3.13+, managed with `uv` (never use pip directly)
- Always run commands via `uv run` (e.g. `uv run pytest`, not bare `pytest`)
- Write pytest tests for every Python module. Tests go in `tests/test_<module>.py`
- Use `pytest-asyncio` for async tests. Mark with `@pytest.mark.asyncio`
- Shared test fixtures live in `tests/conftest.py`
- Import constants from `ai_server.config` — never hardcode thresholds, ports, or paths
- Dataclasses for message types, `asyncio` for networking
- Type hints on all function signatures

**MQL5:**
- Cannot compile or run on macOS — only on Windows VPS with MetaEditor
- Shared includes (Constants, AIClient, MarketStructure, CandlePatterns, NewsShield, SpikeDetector, SessionManager, VWAPCalculator) are **duplicated** in both `gold_scalper_ea/Include/` and `gold_swing_ea/Include/` — MT5 requires includes relative to EA. Keep both copies in sync.
- All classes use `C` prefix (CDirectionLayer, CExitManager, etc.)
- Guard all .mqh files with `#ifndef`/`#define`/`#endif`
- Magic numbers: Scalper = 100001, Swing = 100002

## Architecture (quick ref)

- **AI is a filter**, not a decision maker. Entry/exit logic is rule-based in MQL5. AI scores confidence 0-100.
- TCP socket localhost:5001, newline-delimited JSON. Two message types: `entry_check` (127 features -> score) and `heartbeat`.
- Latency: P95 < 150ms scalper, < 300ms swing.
- Risk: per-trade (1.5%/2%) -> session (10% cap, 7% halt) -> daily (8% halt) -> system (spike/news/VIX/fallback).
- Critical: Pilot negative = cancel cascade. Swing exits on H4 closes only. 3 AI failures = fallback mode. News Shield = 4-phase protocol.

## Reference docs

Read these BEFORE modifying the relevant subsystem:
- `architecture/scalper-bot.md` — cascade entry, M5 direction, 8 exit types
- `architecture/swing-bot.md` — H4 direction (6 indicators), H1 entry (7 conditions), structural exits
- `architecture/ai-engine.md` — 127 features, BiLSTM+XGBoost, ensemble weights, training pipeline
- `architecture/risk-management.md` — 5-level hierarchy, 12 failure modes, spike detector, news shield
- `architecture/communication-protocol.md` — JSON schemas, validation rules, fallback transitions
- `architecture/infrastructure.md` — VPS, broker, monitoring, deployment plan

## Workflow

- Dev on macOS, push via Git, pull on Windows VPS for MT5 compilation/testing
- Build progress tracked in `BUILD_PHASES.md` — update status after completing each phase
- Full implementations preferred over stubs
