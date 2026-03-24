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
| **Kronos** | Price Forecast | Future OHLCV candles | [Kronos](https://github.com/shiyu-coder/Kronos) foundation model (102M params, trained on 12B+ candles from 45 exchanges) |
| **Qlib** | Factor Model | Expected returns from 158 alpha factors | [Microsoft Qlib](https://github.com/microsoft/qlib) ML pipeline (LightGBM/LSTM/Transformer) |
| **Chronos** | Time Series | Probabilistic price forecasts with confidence intervals | [Amazon Chronos-2](https://github.com/amazon-science/chronos-forecasting) |
| **TradingAgents** | LLM Analysis | Directional signal from multi-agent debate | [TradingAgents](https://github.com/TauricResearch/TradingAgents) (bull/bear/risk analysts) |
| **Technical** | Indicators | RSI, MACD, Bollinger, support/resistance levels | pandas-ta + custom S/R clustering |

### Evolution Layer

| Tool | What It Does |
|------|-------------|
| **RD-Agent** | Automated alpha factor discovery via LLM-powered R&D loop ([Microsoft RD-Agent](https://github.com/microsoft/RD-Agent)) |

Every engine has a **built-in fallback** that works without external dependencies, so the system runs immediately after install.

## Quick Start

```bash
# Install
git clone https://github.com/ucwLeonardo/signalforge.git
cd signalforge
pip install -e .

# Generate signals
signalforge scan AAPL NVDA TSLA BTC/USDT ES=F

# Deep prediction for a single asset
signalforge predict AAPL --horizon 10

# Interactive dashboard
pip install streamlit
signalforge dashboard
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

## Installation

### Core (works immediately)

```bash
pip install -e .
```

This installs yfinance, ccxt, pandas-ta, torch, and other core deps. All engines run with baseline fallbacks.

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

## CLI Commands

```bash
signalforge scan                          # Scan all configured assets
signalforge scan AAPL BTC/USDT ES=F      # Scan specific symbols
signalforge scan --engine kronos          # Use only one engine
signalforge scan --format json            # Output as JSON
signalforge predict AAPL --horizon 10     # Kronos prediction for one symbol
signalforge fetch AAPL MSFT --days 730    # Pre-fetch and cache data
signalforge evolve --mode factor -n 20    # Auto-discover alpha factors
signalforge dashboard                     # Launch Streamlit web UI
signalforge setup                         # Check dependencies
```

## Configuration

Edit `config/default.yaml` to customize:

```yaml
# Asset universe
assets:
  us_stocks: [AAPL, MSFT, NVDA, TSLA, GOOGL]
  crypto: [BTC/USDT, ETH/USDT, SOL/USDT]
  futures: [ES=F, NQ=F, GC=F]

# Engine weights for ensemble
ensemble:
  kronos_weight: 0.35
  qlib_weight: 0.20
  chronos_weight: 0.15
  agents_weight: 0.10
  technical_weight: 0.20

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
- **Kronos** fine-tuning and inference (~1-2 GB VRAM)
- **Qlib** LSTM/Transformer model training (~4 GB VRAM)
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
│   │   ├── models.py          # Frozen dataclasses (Bar, Asset, Signal, TradeTarget)
│   │   ├── providers.py       # yfinance + ccxt data providers
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
└── notebooks/                 # Jupyter notebooks (coming soon)
```

## Disclaimer

SignalForge generates signals for informational and research purposes only. It does not constitute financial advice. Always do your own research and consult with qualified professionals before making investment decisions. Past performance of any model or signal does not guarantee future results.

## License

MIT
