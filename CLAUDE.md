# SignalForge

Multi-asset buy/sell signal system that generates price targets for US stocks, crypto, and futures.

## Architecture

```
Data Layer → Prediction Engines → Signal Ensemble → Price Targets → Report
```

### Prediction Engines
- **Kronos** (primary): Foundation model for financial OHLCV forecasting
- **Technical Analysis**: pandas-ta based indicators + support/resistance
- **Qlib** (Phase 2): ML factor models (LightGBM, LSTM, Transformer)
- **TradingAgents** (Phase 3): Multi-agent LLM qualitative analysis
- **Chronos-2** (Phase 2): Amazon's time series forecasting

### Data Sources
- yfinance: US stocks, ETFs, futures
- ccxt: Crypto from Binance and 100+ exchanges
- Parquet storage for caching

## Tech Stack
- Python 3.10+, PyTorch, HuggingFace Transformers
- pandas, pandas-ta for data and technical analysis
- Typer + Rich for CLI
- loguru for logging
- YAML configuration with env var expansion

## Commands
```bash
signalforge scan                    # Scan all configured assets
signalforge scan AAPL BTC/USDT     # Scan specific symbols
signalforge predict AAPL            # Kronos prediction for one symbol
signalforge fetch AAPL MSFT         # Pre-fetch and cache data
signalforge setup                   # Check dependencies
```

## Project Structure
```
src/signalforge/
├── config.py          # YAML config loader
├── cli.py             # Typer CLI
├── pipeline.py        # Main orchestration pipeline
├── data/
│   ├── models.py      # Bar, Asset, Signal dataclasses
│   ├── providers.py   # yfinance, ccxt data providers
│   └── store.py       # Parquet storage
├── engines/
│   ├── base.py        # PredictionEngine ABC
│   ├── kronos_engine.py  # Kronos foundation model
│   └── technical.py   # Technical analysis engine
├── ensemble/
│   ├── combiner.py    # Weighted signal combination
│   └── targets.py     # Buy/sell price target calculation
├── output/
│   └── report.py      # Rich table / JSON / CSV output
└── evolution/         # Phase 4: RD-Agent factor evolution
```
