# SignalForge

**Multi-asset buy/sell signal system** that generates actionable price targets for US stocks, options, futures, and crypto.

SignalForge combines 5 prediction engines into a weighted ensemble, producing entry prices, profit targets, stop-losses, and confidence scores — without executing any trades.

## How It Works

```
Market Data → 5 Prediction Engines → Weighted Ensemble → Buy/Sell Price Targets
```

### Prediction Engines

| Engine | Type | What It Predicts | Powered By |
|--------|------|-----------------|------------|
| **Kronos** (40%) | Price Forecast | Future OHLCV candles | [Kronos](https://github.com/shiyu-coder/Kronos) foundation model (102M params) |
| **Qlib** (20%) | Factor Model | Expected returns from 158 alpha factors | [Microsoft Qlib](https://github.com/microsoft/qlib) (LightGBM/LSTM/Transformer) |
| **Chronos** (15%) | Time Series | Probabilistic price forecasts with intervals | [Amazon Chronos-2](https://github.com/amazon-science/chronos-forecasting) |
| **Technical** (15%) | Indicators | RSI, MACD, Bollinger, support/resistance | pandas-ta + custom S/R clustering |
| **TradingAgents** (10%) | LLM Analysis | Directional signal from multi-agent debate | [TradingAgents](https://github.com/TauricResearch/TradingAgents) |

Every engine has a **built-in fallback** — the system runs immediately after install, no GPU or API keys required.

### Supported Asset Types

| Type | Format | Example | Data Source |
|------|--------|---------|-------------|
| **US Stocks** | Ticker | `AAPL`, `NVDA` | yfinance |
| **Crypto** | `BASE/QUOTE` | `BTC/USDT`, `SOL/USDT` | ccxt (Binance) |
| **Futures** | `TICKER=F` | `ES=F`, `GC=F` | yfinance |
| **Options** | Human or OCC | `AAPL 2026-06-19 200 C` | yfinance (underlying + chain Greeks/IV) |

Options use the underlying stock's OHLCV enriched with current option chain data (implied volatility, volume, open interest, bid/ask).

## Quick Start

```bash
# Install
git clone https://github.com/ucwLeonardo/signalforge.git
cd signalforge
pip install -e .

# Check setup
signalforge setup

# Scan all default assets (10 stocks + 24 crypto + 4 futures)
signalforge scan

# Scan specific symbols
signalforge scan AAPL BTC/USDT ES=F "AAPL 2026-06-19 200 C"
```

### Example Output

```
┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Symbol   ┃ Action ┃  Entry ┃ Target ┃   Stop ┃  R:R ┃ Confidence ┃ Rationale              ┃
┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ AAPL     │  BUY   │ 218.50 │ 235.20 │ 210.80 │ 2.17 │        78% │ Bullish consensus ...  │
│ BTC/USDT │  SELL  │ 67800  │ 62500  │ 71200  │ 1.56 │        65% │ Bearish divergence ... │
│ ES=F     │  HOLD  │ 5420   │ 5420   │ 5420   │ 0.00 │        92% │ Neutral signal ...     │
└──────────┴────────┴────────┴────────┴────────┴──────┴────────────┴────────────────────────┘
```

## Use Case Workflows

### Find Top 5 Most Confident Crypto Buy Signals

```bash
signalforge top -t crypto -n 5
```

### Find Top 5 US Stock Buy Signals

```bash
signalforge top -t stock -n 5
```

### Find Top Sell Signals Across All Assets

```bash
signalforge top -a sell -n 10
```

### Scan Trending / New Crypto for Hyper-Growth

```bash
signalforge scan PEPE/USDT WIF/USDT BONK/USDT SUI/USDT TIA/USDT TAO/USDT \
  --horizon 10 --format json
```

Use `--horizon 10` for longer-term growth potential. Pipe through `jq` or sort by `risk_reward_ratio` for asymmetric bets.

### Scan Options

```bash
# Human-readable format
signalforge scan "AAPL 2026-06-19 200 C" "NVDA 2026-06-19 150 P"

# OCC format
signalforge scan AAPL260619C00200000
```

### Deep Dive on a Single Asset

```bash
signalforge predict NVDA --horizon 10
```

Shows predicted OHLCV candles for the next 10 bars with price range and direction.

### Export for Downstream Processing

```bash
# JSON for scripts/APIs
signalforge top -t crypto -n 20 -f json > signals.json

# CSV for spreadsheets
signalforge scan --format csv > signals.csv
```

## CLI Commands

```bash
signalforge scan                              # Scan all configured assets
signalforge scan AAPL BTC/USDT ES=F          # Scan specific symbols
signalforge scan --engine kronos              # Use only one engine
signalforge scan --format json                # Output as JSON/CSV

signalforge top                               # Top 5 buy signals (all assets)
signalforge top -n 10 -a sell -t crypto       # Top 10 sell signals for crypto
signalforge top -a all -t stock               # Top 5 buy+sell for stocks
signalforge top -t options                    # Top 5 buy signals for options

signalforge predict AAPL --horizon 10         # Kronos prediction for one symbol
signalforge fetch AAPL MSFT --days 730        # Pre-fetch and cache data
signalforge evolve --mode factor -n 20        # Auto-discover alpha factors
signalforge dashboard                         # Launch Streamlit web UI
signalforge setup                             # Check dependencies
```

## Default Asset Universe

### Crypto (24 pairs)

| Category | Symbols |
|----------|---------|
| **Major** | BTC, ETH, SOL, BNB |
| **DeFi** | LINK, UNI, AAVE, MKR |
| **L2 / Alt L1** | AVAX, DOT, MATIC, ARB, OP, SUI, SEI, TIA |
| **AI Tokens** | FET, RNDR, TAO |
| **Meme** | DOGE, SHIB, PEPE, WIF, BONK |

### Stocks (10)

AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, AMD, AVGO, TSM

### Futures (4)

ES=F (S&P 500), NQ=F (Nasdaq), GC=F (Gold), CL=F (Crude Oil)

Edit `config/default.yaml` to customize.

## Installation

### Core (works immediately)

```bash
pip install -e .
```

All engines run with built-in fallbacks — no GPU, API keys, or optional deps required.

| Engine | Fallback (no extra deps) |
|--------|--------------------------|
| Kronos | Linear regression |
| Qlib | Ridge regression on momentum/mean-reversion factors |
| Chronos | Holt's linear trend exponential smoothing |
| TradingAgents | Rule-based RSI + price-action sentiment |
| Technical | Built-in RSI/MACD/BBands/ATR (numpy/pandas) |

### Unlock Real Engines

```bash
# Kronos — OHLCV foundation model (recommended, especially with GPU)
git clone https://github.com/shiyu-coder/Kronos.git ~/Kronos
cd ~/Kronos && pip install -r requirements.txt
export PYTHONPATH=~/Kronos:$PYTHONPATH

# Qlib — ML factor models
pip install pyqlib

# Chronos-2 — Amazon time series forecasting
pip install chronos-forecasting

# TradingAgents — Multi-agent LLM analysis
pip install tradingagents
export ANTHROPIC_API_KEY=sk-...  # or OPENAI_API_KEY

# RD-Agent — Automated factor evolution
pip install rdagent

# Dashboard
pip install streamlit

# Or install everything at once
pip install -e ".[all]"
```

## Configuration

Edit `config/default.yaml`:

```yaml
# Asset universe
assets:
  us_stocks: [AAPL, MSFT, NVDA, TSLA, GOOGL]
  crypto: [BTC/USDT, ETH/USDT, SOL/USDT]
  futures: [ES=F, NQ=F, GC=F]
  options: ["AAPL 2026-06-19 200 C"]

# Engine weights for ensemble
ensemble:
  kronos_weight: 0.40
  qlib_weight: 0.20
  chronos_weight: 0.15
  agents_weight: 0.10
  technical_weight: 0.15

# Kronos settings
engines:
  kronos:
    enabled: true
    model: "NeoQuasar/Kronos-base"
    pred_len: 5
    device: cuda
```

## GPU Support

SignalForge leverages your GPU for:
- **Kronos** inference (~1-2 GB VRAM)
- **Qlib** LSTM/Transformer training (~4 GB VRAM)
- **Chronos-2** inference (~2 GB VRAM)

An RTX 5090 (32 GB) can run all engines simultaneously.

## Project Structure

```
signalforge/
├── config/default.yaml        # Default configuration
├── pyproject.toml             # Package definition
├── src/signalforge/
│   ├── cli.py                 # CLI entry point (Typer)
│   ├── config.py              # YAML config loader
│   ├── pipeline.py            # Orchestration pipeline
│   ├── data/
│   │   ├── models.py          # Frozen dataclasses (Bar, Asset, Signal, TradeTarget, OptionContract)
│   │   ├── providers.py       # Stock, Crypto, Futures, Options data providers
│   │   └── store.py           # Parquet-based data cache
│   ├── engines/
│   │   ├── kronos_engine.py   # Kronos foundation model
│   │   ├── qlib_engine.py     # Qlib ML factor models
│   │   ├── chronos_engine.py  # Amazon Chronos-2
│   │   ├── agents_engine.py   # TradingAgents LLM
│   │   └── technical.py       # Technical analysis + S/R
│   ├── ensemble/
│   │   ├── combiner.py        # Weighted signal fusion
│   │   └── targets.py         # Price target calculation
│   ├── output/
│   │   └── report.py          # Rich/JSON/CSV formatting
│   ├── dashboard/
│   │   └── app.py             # Streamlit web dashboard
│   └── evolution/
│       └── rdagent_runner.py  # RD-Agent factor evolution
└── tests/                     # 127 tests (pytest)
```

## Testing

```bash
pip install pytest pytest-cov
PYTHONPATH=src pytest tests/ -v            # Run all tests
PYTHONPATH=src pytest tests/ --cov=signalforge  # With coverage
```

## Disclaimer

SignalForge generates signals for informational and research purposes only. It does not constitute financial advice. Always do your own research and consult with qualified professionals before making investment decisions. Past performance of any model or signal does not guarantee future results.

## License

MIT
