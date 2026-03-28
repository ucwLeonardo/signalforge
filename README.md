# SignalForge

**Multi-asset buy/sell signal system** that generates actionable price targets for US stocks, crypto, and futures.

SignalForge combines multiple prediction engines into a weighted ensemble, producing entry prices, profit targets, stop-losses, and confidence scores. Includes a paper trading dashboard for simulated portfolio management.

## How It Works

```
Market Data (Massive API) → Prediction Engines → Weighted Ensemble → Buy/Sell Targets → Paper Trading
```

### Prediction Engines

| Engine | Weight | What It Does |
|--------|--------|-------------|
| **LSTM** (35%) | Sequence-to-sequence price forecasting | PyTorch LSTM with Monte Carlo dropout |
| **GBM** (35%) | Gradient-boosted return prediction | LightGBM/sklearn with 35+ alpha factors |
| **Technical** (20%) | RSI, MACD, Bollinger, support/resistance | pandas-ta + custom S/R clustering |
| **TradingAgents** (10%) | LLM multi-analyst review (top 10 signals) | Google Gemini (bull/bear/risk debate) |

Disabled engines (available but off by default): Kronos, Qlib, Chronos-2.

Every engine has a **built-in fallback** — the system runs immediately after install, no GPU or API keys required.

### Data Source

All market data (stocks, crypto, futures) comes from **Massive** (Polygon.io API) through a unified `MassiveProvider`:

| Asset Type | SignalForge Format | Polygon Ticker | Example |
|------------|-------------------|----------------|---------|
| **US Stocks** | `AAPL` | `AAPL` | Direct pass-through |
| **Crypto** | `BTC/USDT` | `X:BTCUSD` | Auto-converted |
| **Futures** | `ES=F` | `SPY` | Mapped to ETF proxy |

Futures use ETF proxies: ES=F→SPY, NQ=F→QQQ, GC=F→GLD, CL=F→USO.

### Portfolio Allocation

Auto-build uses **Half-Kelly Criterion** for position sizing:
- Kelly fraction: `f = confidence × (R:R - 1) / R:R`, halved for conservatism
- Total allocation: 80% of cash (20% reserve)
- Single position cap: 30%
- Short positions: separate 10% budget

## Quick Start

```bash
# Install
git clone https://github.com/ucwLeonardo/signalforge.git
cd signalforge
pip install -e .

# Set up Massive API key (free at https://massive.com)
export MASSIVE_API_KEY="your_key_here"

# Deploy (installs deps, downloads all data, starts server)
./deploy.sh

# Or manually:
signalforge setup              # Check dependencies
signalforge scan               # Scan all default assets
signalforge paper dashboard    # Launch paper trading dashboard
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MASSIVE_API_KEY` | **Yes** | Polygon/Massive API key for market data |
| `GEMINI_API_KEY` | No | Google Gemini for TradingAgents LLM engine |

## Paper Trading Dashboard

Web-based dashboard at `http://localhost:8787`:

- **Multi-account**: Create accounts with specific asset categories (stocks, crypto, futures)
- **Scan All**: Full scan — config symbols + dynamically discovered assets (~344 symbols)
- **Watchlist Scan**: Quick scan — only config-defined symbols (~38 symbols), much faster
- **Watchlist Editor**: Edit config symbols per category (US Stocks, Crypto, Futures, Options) directly from the dashboard
- **Stop/Restart**: Responsive scan cancellation (stops within 0.5s), progress panel stays visible with Restart button
- **Signals**: Filtered by account type — crypto account only sees crypto signals
- **Trade**: Manual position entry with entry/stop/target from signals
- **Auto-Build**: Kelly-criterion portfolio construction from top signals

```bash
# Start dashboard
signalforge paper dashboard --port 8787

# Or via deploy script
./deploy.sh
```

## CLI Commands

```bash
# Signal generation
signalforge scan                              # Scan all configured assets
signalforge scan AAPL BTC/USDT ES=F          # Scan specific symbols
signalforge scan --engine technical           # Single engine only
signalforge scan --format json                # Output as JSON/CSV

# Top signals
signalforge top                               # Top 5 buy signals
signalforge top -n 10 -a sell -t crypto       # Top 10 crypto sell signals

# Data & models
signalforge fetch AAPL MSFT --days 730        # Pre-fetch and cache data
signalforge train --categories us_stocks      # Train LSTM/GBM models
signalforge predict AAPL --horizon 10         # Kronos single-symbol prediction

# Paper trading
signalforge paper init --account myaccount    # Create account
signalforge paper status                      # Portfolio status
signalforge paper auto -n 5                   # Auto-trade top 5 signals
signalforge paper dashboard                   # Web dashboard

# System
signalforge setup                             # Check dependencies
signalforge evolve --mode factor -n 20        # Auto-discover alpha factors
```

## Default Asset Universe

| Category | Count | Symbols |
|----------|-------|---------|
| **US Stocks** | 10 | AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, AMD, AVGO, TSM |
| **Crypto** | 24 | BTC, ETH, SOL, BNB, LINK, UNI, AAVE, MKR, AVAX, DOT, MATIC, ARB, OP, SUI, SEI, TIA, FET, RNDR, TAO, DOGE, SHIB, PEPE, WIF, BONK |
| **Futures** | 4 | ES=F (S&P 500), NQ=F (Nasdaq), GC=F (Gold), CL=F (Crude Oil) |

Dynamic discovery expands this to ~128 stocks (S&P 500 fallback) + ~200 crypto pairs via Massive API.

Dynamic discovery expands stocks to ~300+ (S&P 500 subset). Assets are grouped by category (all stocks first, then crypto, then futures) — no interleaving.

Edit `config/default.yaml` to customize, or use the Watchlist Editor in the dashboard.

## Installation

### Core

```bash
pip install -e .
```

### Optional Engines

```bash
pip install -e ".[all]"          # Everything

# Or individually:
pip install pyqlib               # Qlib factor models
pip install chronos-forecasting  # Amazon Chronos-2
pip install google-genai         # Gemini for TradingAgents
pip install rdagent              # Automated factor evolution
pip install streamlit            # Streamlit dashboard
```

### Kronos (not on PyPI)

```bash
git clone https://github.com/shiyu-coder/Kronos.git ~/Kronos
cd ~/Kronos && pip install -r requirements.txt
export PYTHONPATH=~/Kronos:$PYTHONPATH
```

## Project Structure

```
signalforge/
├── config/default.yaml           # Default configuration
├── deploy.sh                     # Deployment script (install + data + server)
├── pyproject.toml                # Package definition
├── src/signalforge/
│   ├── cli.py                    # CLI entry point (Typer)
│   ├── config.py                 # YAML config loader (env var expansion)
│   ├── pipeline.py               # Orchestration: data → engines → ensemble → targets
│   ├── data/
│   │   ├── models.py             # Domain models (Bar, Asset, Signal, TradeTarget)
│   │   ├── providers.py          # MassiveProvider (Polygon API, unified for all assets)
│   │   ├── store.py              # Parquet-based data cache
│   │   ├── discovery.py          # Dynamic asset discovery (S&P 500, crypto)
│   │   └── incremental.py        # Incremental fetcher with rate-limit handling
│   ├── engines/
│   │   ├── lstm_engine.py        # LSTM with model persistence
│   │   ├── gbm_engine.py         # GBM with alpha factor integration
│   │   ├── technical.py          # Technical analysis + S/R detection
│   │   ├── agents_engine.py      # Gemini LLM multi-analyst
│   │   ├── kronos_engine.py      # Kronos foundation model (disabled)
│   │   ├── qlib_engine.py        # Qlib ML factors (disabled)
│   │   └── chronos_engine.py     # Amazon Chronos-2 (disabled)
│   ├── ensemble/
│   │   ├── combiner.py           # Weighted signal fusion
│   │   └── targets.py            # BUY/SELL/HOLD with entry/target/stop
│   ├── factors/
│   │   ├── library.py            # 35 alpha factors (momentum, mean-rev, volatility)
│   │   ├── compute.py            # Factor computation engine
│   │   ├── evaluate.py           # IC/IR evaluation
│   │   ├── operators.py          # Factor DSL operators
│   │   └── preprocess.py         # Winsorize, z-score normalization
│   ├── paper/
│   │   ├── server.py             # HTTP server (paper trading dashboard)
│   │   ├── dashboard.html        # Single-file web UI
│   │   ├── models.py             # Portfolio, Position, Trade dataclasses
│   │   ├── portfolio.py          # Account & portfolio management
│   │   ├── simulator.py          # Signal generation + Kelly allocation
│   │   └── executor.py           # Position execution logic
│   ├── evolution/
│   │   ├── rdagent_runner.py     # RD-Agent factor/model evolution
│   │   └── factor_registry.py    # Discovered factor storage
│   └── output/
│       └── report.py             # Rich table / JSON / CSV output
└── tests/                        # pytest test suite
```

## Data Storage

```
~/.signalforge/
├── data/
│   ├── stock/          # AAPL_1d.parquet, MSFT_1d.parquet, ...
│   ├── crypto/         # BTC-USDT_1d.parquet, ETH-USDT_1d.parquet, ...
│   └── futures/        # ES=F_1d.parquet, ...
├── models/
│   ├── lstm/           # Per-symbol .pt checkpoints
│   └── gbm/            # Per-symbol .pkl models
├── cache/
└── results/
```

OHLCV data is stored as Parquet (timestamp, open, high, low, close, volume). Models persist between scans for fast inference.

## Rate Limiting

Massive API free tier: **5 requests/minute**. The provider includes:
- Sliding-window rate limiter (auto-sleeps between batches)
- HTTP 429 retry with backoff
- Consecutive failure detection (skips network after 10 real failures)

First full scan of all assets takes ~40 minutes; subsequent scans use cached data (daily bars cached up to 4 days). Watchlist-only scans with cached data complete in seconds.

## Testing

```bash
pip install pytest pytest-cov
PYTHONPATH=src pytest tests/ -v
PYTHONPATH=src pytest tests/ --cov=signalforge
```

## Disclaimer

SignalForge generates signals for informational and research purposes only. It does not constitute financial advice. Always do your own research and consult with qualified professionals before making investment decisions.

## License

MIT
