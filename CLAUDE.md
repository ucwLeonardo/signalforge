# SignalForge

Multi-asset buy/sell signal system that generates price targets for US stocks, crypto, and futures.
Leverages 5 open-source projects: Kronos, Qlib, Chronos-2, TradingAgents, and RD-Agent.

## Architecture

```
Data Layer → 5 Prediction Engines → Signal Ensemble → Price Targets → Report/Dashboard
```

### 5 Prediction Engines (all with graceful fallbacks)
1. **Kronos** (primary): OHLCV foundation model forecasting → linear regression fallback
2. **Qlib**: ML factor prediction (Alpha158 features, LightGBM/LSTM) → Ridge regression on momentum factors
3. **Chronos-2**: Amazon's probabilistic time series → Holt's exponential smoothing fallback
4. **TradingAgents**: Multi-agent LLM debate (bull/bear/risk) → price-action rule-based fallback
5. **Technical**: RSI/MACD/Bollinger/S&R via pandas-ta → built-in numpy implementations

### Evolution Layer
- **RD-Agent**: Automated factor discovery + model evolution via LLM-powered R&D loop

### Data Sources
- yfinance: US stocks, ETFs, futures (free)
- ccxt: Crypto from Binance and 100+ exchanges (free)
- Parquet storage for caching

## Tech Stack
- Python 3.10+, PyTorch, HuggingFace Transformers
- pandas, pandas-ta, scikit-learn
- Typer + Rich for CLI, Streamlit for dashboard
- loguru for logging, YAML configuration

## Commands
```bash
signalforge scan                        # Scan all configured assets
signalforge scan AAPL BTC/USDT ES=F     # Scan specific symbols
signalforge predict AAPL --horizon 10   # Kronos prediction for one symbol
signalforge fetch AAPL MSFT             # Pre-fetch and cache data
signalforge evolve --mode factor -n 20  # Run RD-Agent factor evolution
signalforge dashboard                   # Launch Streamlit dashboard
signalforge setup                       # Check all dependencies
```

## Project Structure
```
src/signalforge/
├── config.py              # YAML config loader with env var expansion
├── cli.py                 # Typer CLI (scan, predict, fetch, setup, evolve, dashboard)
├── pipeline.py            # Full orchestration: data → engines → ensemble → targets
├── data/
│   ├── models.py          # Bar, Asset, Signal, CombinedSignal, TradeTarget
│   ├── providers.py       # yfinance, ccxt data providers
│   └── store.py           # Parquet storage with dedup
├── engines/
│   ├── base.py            # PredictionEngine ABC
│   ├── kronos_engine.py   # Kronos foundation model
│   ├── qlib_engine.py     # Qlib ML factor models
│   ├── chronos_engine.py  # Amazon Chronos-2 forecasting
│   ├── agents_engine.py   # TradingAgents multi-agent LLM
│   └── technical.py       # Technical analysis + S/R detection
├── ensemble/
│   ├── combiner.py        # Weighted 5-engine signal fusion
│   └── targets.py         # BUY/SELL/HOLD with entry/target/stop
├── output/
│   └── report.py          # Rich table / JSON / CSV
├── dashboard/
│   └── app.py             # Streamlit interactive dashboard
└── evolution/
    └── rdagent_runner.py  # RD-Agent factor/model evolution
```

## Installation
```bash
pip install -e .                  # Core (yfinance, ccxt, pandas-ta, torch)
pip install -e ".[dashboard]"     # + Streamlit dashboard
pip install -e ".[all]"           # Everything including Qlib, Chronos, TradingAgents

# For Kronos (not on PyPI):
git clone https://github.com/shiyu-coder/Kronos.git ~/Kronos
pip install -r ~/Kronos/requirements.txt
export PYTHONPATH=~/Kronos:$PYTHONPATH
```
