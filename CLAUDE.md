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
- **IncrementalFetcher**: Cache-first with rate-limit-aware incremental updates
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

### Ensemble & Targets
- `ensemble/combiner.py`: Weighted signal fusion with dynamic re-normalization
- `ensemble/targets.py`: BUY/SELL/HOLD with entry, target, stop-loss, R:R ratio

### Paper Trading
- HTTP server (`paper/server.py`) serving single-file dashboard (`paper/dashboard.html`)
- Multi-account with per-account asset categories (stored in portfolio JSON)
- Auto-build: Half-Kelly criterion portfolio allocation
- Signal filtering by account categories
- Live scan with progress tracking

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
