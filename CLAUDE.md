# SignalForge

Multi-asset buy/sell signal system that generates price targets for US stocks, crypto, and futures.

## Architecture

```
Massive API → Data Cache (Parquet) → Prediction Engines → Ensemble → Price Targets → Paper Trading
```

### Data Layer
- **MassiveProvider** (Polygon API): Unified provider for stocks, crypto, futures
  - Stocks: direct ticker pass-through
  - Crypto: `BTC/USDT` → `X:BTCUSD` conversion
  - Futures: `ES=F` → `SPY` ETF proxy mapping
- **IncrementalFetcher**: Cache-first with rate-limit-aware incremental updates, cancel-aware 0.5s polling
  - Cache freshness: stocks/futures must have latest trading day's bar; crypto allows 2 days
- **Trading Calendar** (`data/calendar.py`): NYSE holiday-aware `last_trading_day()` with market-close cutoff (UTC 22:00 = ET 18:00) — avoids marking data as stale before daily bars are available
- **DataStore**: Parquet storage with dedup, `~/.signalforge/data/{stock,crypto,futures}/`
- Rate limiting: 5 req/min sliding window, HTTP 429 retry, consecutive failure detection

### Prediction Engines (all with graceful fallbacks)
1. **LSTM** (35%): PyTorch seq2seq, MC dropout, model persistence in `~/.signalforge/models/lstm/`
2. **GBM** (35%): LightGBM/sklearn gradient boosting, 35+ alpha factors, persistence in `~/.signalforge/models/gbm/`
3. **Technical** (20%): RSI/MACD/Bollinger/S&R via pandas-ta
4. **TradingAgents** (10%): Google Gemini multi-analyst LLM review (top 10 signals only)
5. **Kronos** (disabled): Foundation model forecasting → linear regression fallback
6. **Qlib** (disabled): Factor prediction → Ridge regression fallback
7. **Chronos** (disabled): Probabilistic time series → Holt's smoothing fallback

### Factor Module
- `factors/library.py`: 35 alpha factors (momentum, mean-reversion, volatility, crypto-specific, options-proxy)
- `factors/operators.py`: DSL for factor expressions
- `factors/preprocess.py`: Winsorize, z-score normalization
- `factors/evaluate.py`: IC/IR evaluation
- Integrated into GBM engine for feature augmentation

### Evolution Layer
- **RD-Agent**: Automated factor discovery + model evolution via LLM-powered R&D loop
- `evolution/factor_registry.py`: Stores discovered factors for GBM integration

### Asset Discovery & Ordering
- `discovery.py`: Merges config symbols + dynamically discovered symbols
- Assets grouped by category: all stocks → all crypto → all futures → all options
- Config symbols always first within each category, discovered appended after
- `config_only` mode: skip discovery, use only config-defined symbols (for watchlist scan)

### Ensemble & Targets
- `ensemble/combiner.py`: Weighted signal fusion with dynamic re-normalization
- `ensemble/targets.py`: BUY/SELL/HOLD with entry, target, stop-loss, R:R ratio
- **Asset-aware trading params** (`config.py: TradingParams`): Per-asset-class stop/target/horizon
  - Stocks: horizon=5d, stop=[1.5%,8%], target=[2%,12%], ATR×1.5 stop / ATR×2.5 target
  - Crypto: horizon=3d, stop=[3%,15%], target=[4%,25%], ATR×1.0 stop / ATR×2.0 target
- **Entry = market price**: Signals provide direction + confidence, not precise entry points
- **Stop/target rescaling**: Auto Build recalculates stop/target from actual entry price, preserving R:R ratio

### Paper Trading
- HTTP server (`paper/server.py`) serving single-file dashboard (`paper/dashboard.html`)
- Multi-account with per-account asset categories (stored in portfolio JSON, editable via modal)
- **Auto Build**: Stratified selection (each asset class gets ≥1 slot, BUY+SELL unified ranking, capped at top_n total) → Half-Kelly allocation → positions opened at real-time market price → stop/target rescaled to actual entry
  - Clears value history on build for clean chart; dedup snapshots to avoid duplicate data points
  - Reuses server's background price cache; only fetches missing symbols (near-instant)
  - Top N value persisted in sessionStorage across refreshes
- **Pending Orders** (`pending_orders.py`): Stocks/futures queued when market closed, execute at next market open
  - Cash escrowed via `reserved_cash` on Portfolio; `open_position(from_reserved=)` for atomic release+open
  - `reconcile_reserved_cash()` runs every 30s cycle to fix stale state from crashes
  - Slippage guard: skip if live price already beyond signal stop/target
  - 48-hour expiry, crash-safe incremental persistence, per-account RLock synchronization
  - Dashboard: pending orders card with cancel buttons, reserved cash in summary bar
  - API: `GET /api/pending-orders`, `POST /api/pending-orders/cancel`
- **Signal Cache**: Per-account JSON persistence (`signal_cache.py`), survives server restart. Separate full/watchlist caches per account
- **Price Fetching** (`prices.py`): Proxy-aware with auto-detection (env vars → Clash 7897/7890)
  - Crypto fallback chain: CoinGecko (primary, no geo-block) → Binance → Polygon prev close
  - Parallel fetching: crypto/stock/option price requests run concurrently via ThreadPoolExecutor
  - Stocks/Futures (market hours): Yahoo Finance (real-time intraday) → Polygon prev close fallback
  - Stocks/Futures (off hours): Polygon prev close only
  - Options: Polygon snapshot API
  - 66 CoinGecko ID mappings for major crypto tokens
- **Background Price Updater**: Server-side thread updates ALL accounts every 30s, frontend auto-refreshes in sync
- **Price Status**: `/api/price-status` endpoint for monitoring update health
- **Transaction Fees**: Asset-aware via `TradingParams.calculate_fee()` — Stocks: $0 (zero commission), Crypto: 0.1% (Binance taker), Futures: $1.25/contract. Actual open fees recorded per position (`open_fee` field); close fees estimated at current price
- Signal filtering by account categories with confidence slider (sessionStorage-persisted)
- **Scan Progress**: Backend-computed `phase_pct` (data 40% → prices 10% → engines 50%) for accurate cross-phase progress bar; dedicated `prices` stage for live price fetching
- **Scan UX**: Stop/Restart with immediate UI feedback, progress panel persists after stop
- **Watchlist**: Edit config symbols via modal (`GET/POST /api/watchlist`), scan config-only with `config_only` param
- **Scan All** vs **Watchlist Scan**: full discovery scan or config-symbols-only quick scan
- Progress log: per-symbol stages (Discovery/Cached/Data/Skipped/Error) with category labels (Stock/Crypto/Futures)
- Cancel-aware data fetching: 0.5s polling in IncrementalFetcher for responsive stop
- Custom confirm modals (no native browser dialogs), collapsible positions panel with summary header

## Development Environment
- **Server runs in WSL2**: User starts `signalforge paper dashboard` manually in WSL2 — never attempt to install deps or start/restart from Claude Code
- **Claude Code runs in sandbox mode**: Sandboxed `curl`/network calls to localhost fail (connection refused)
- **Accessing localhost:8787**:
  1. **Playwright** (preferred for UI testing) — browser can reach localhost
  2. **Bash with `dangerouslyDisableSandbox: true`** — for curl/API checks
- **No external network**: External APIs (yfinance, ccxt, etc.) are blocked in sandbox. Ask user to run commands manually with `!` prefix when external data is needed

## Environment Variables
- `MASSIVE_API_KEY` (required): Polygon/Massive API key for all market data
- `GEMINI_API_KEY` (optional): Google Gemini for TradingAgents engine

## Commands
```bash
signalforge scan                        # Scan all configured assets
signalforge scan AAPL BTC/USDT ES=F     # Scan specific symbols
signalforge top -n 10 -t crypto         # Top crypto signals
signalforge fetch AAPL MSFT             # Pre-fetch and cache data
signalforge train                       # Train LSTM/GBM models
signalforge predict AAPL --horizon 10   # Kronos prediction
signalforge evolve --mode factor -n 20  # Factor evolution
signalforge paper dashboard             # Paper trading web UI
signalforge setup                       # Check dependencies
```

## Project Structure
```
src/signalforge/
├── cli.py              # Typer CLI
├── config.py           # YAML config loader with env var expansion
├── pipeline.py         # Orchestration: data → engines → ensemble → targets
├── data/
│   ├── models.py       # Bar, Asset, Signal, TradeTarget, Portfolio
│   ├── providers.py    # MassiveProvider (Polygon API)
│   ├── store.py        # Parquet storage
│   ├── discovery.py    # Dynamic asset discovery
│   └── incremental.py  # Rate-limited incremental fetcher
├── engines/            # LSTM, GBM, Technical, Agents, Kronos, Qlib, Chronos
├── ensemble/           # Signal combiner + target calculator
├── factors/            # 35 alpha factors, operators, preprocessing
├── paper/              # Paper trading server, dashboard, portfolio management
├── evolution/          # RD-Agent factor/model evolution
└── output/             # Report generation
```

## Installation
```bash
pip install -e .                  # Core
pip install -e ".[all]"           # Everything including optional engines
./deploy.sh                       # Full deployment (install + data download + server)
```

## Tech Stack
- Python 3.10+, PyTorch, scikit-learn, LightGBM
- pandas, pandas-ta, pyarrow (Parquet)
- requests (Massive/Polygon API)
- Typer + Rich for CLI
- loguru for logging, YAML configuration
