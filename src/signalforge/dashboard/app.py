"""SignalForge Dashboard - Interactive signal review with Streamlit.

Run with: streamlit run src/signalforge/dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure signalforge is importable
_src = str(Path(__file__).parent.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    import streamlit as st
except ImportError:
    raise SystemExit(
        "Streamlit not installed. Install with: pip install streamlit\n"
        "Then run: streamlit run src/signalforge/dashboard/app.py"
    )


def main() -> None:
    st.set_page_config(
        page_title="SignalForge",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 SignalForge - Multi-Asset Signal Dashboard")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        # Asset selection
        asset_type = st.selectbox("Asset Type", ["US Stocks", "Crypto", "Futures", "Custom"])

        if asset_type == "US Stocks":
            default_symbols = "AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, AMD"
        elif asset_type == "Crypto":
            default_symbols = "BTC/USDT, ETH/USDT, SOL/USDT"
        elif asset_type == "Futures":
            default_symbols = "ES=F, NQ=F, GC=F, CL=F"
        else:
            default_symbols = ""

        symbols_input = st.text_area(
            "Symbols (comma-separated)",
            value=default_symbols,
        )
        symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

        st.divider()

        # Engine selection
        st.subheader("Engines")
        use_kronos = st.checkbox("Kronos (Price Forecast)", value=True)
        use_qlib = st.checkbox("Qlib (Factor Model)", value=False)
        use_chronos = st.checkbox("Chronos (Time Series)", value=False)
        use_agents = st.checkbox("TradingAgents (LLM)", value=False)
        use_technical = st.checkbox("Technical Analysis", value=True)

        st.divider()

        # Parameters
        pred_len = st.slider("Forecast Horizon (bars)", 1, 30, 5)
        interval = st.selectbox("Interval", ["1d", "1h", "4h", "1w"], index=0)
        confidence_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.3)

        run_scan = st.button("🔍 Run Scan", type="primary", use_container_width=True)

    # --- Main content ---
    if run_scan and symbols:
        _run_scan(
            symbols=symbols,
            pred_len=pred_len,
            interval=interval,
            confidence_threshold=confidence_threshold,
            use_kronos=use_kronos,
            use_qlib=use_qlib,
            use_chronos=use_chronos,
            use_agents=use_agents,
            use_technical=use_technical,
        )
    else:
        _show_welcome()


def _show_welcome() -> None:
    """Show the welcome/landing page."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Engines Available", "5")
        st.caption("Kronos, Qlib, Chronos, TradingAgents, Technical")

    with col2:
        st.metric("Asset Classes", "3")
        st.caption("US Stocks, Crypto, Futures")

    with col3:
        st.metric("Data Sources", "2+")
        st.caption("yfinance, ccxt, custom")

    st.info(
        "Configure symbols and engines in the sidebar, then click **Run Scan** "
        "to generate buy/sell signals with price targets."
    )

    st.markdown("""
    ### How It Works

    1. **Data Fetch** - Pulls OHLCV data from yfinance (stocks/futures) or ccxt (crypto)
    2. **Multi-Engine Prediction** - Runs enabled prediction engines in parallel
    3. **Signal Ensemble** - Combines predictions using weighted voting
    4. **Price Targets** - Calculates entry, target, and stop-loss prices
    5. **Report** - Displays actionable signals with confidence scores

    ### Engine Details

    | Engine | Type | What It Does |
    |--------|------|-------------|
    | **Kronos** | Price Forecast | Foundation model predicting future OHLCV candles |
    | **Qlib** | Factor Model | ML-based alpha factors (momentum, mean-reversion) |
    | **Chronos** | Time Series | Probabilistic forecasting with prediction intervals |
    | **TradingAgents** | LLM Analysis | Multi-agent debate for qualitative assessment |
    | **Technical** | Indicators | RSI, MACD, Bollinger, Support/Resistance |
    """)


def _run_scan(
    symbols: list[str],
    pred_len: int,
    interval: str,
    confidence_threshold: float,
    use_kronos: bool,
    use_qlib: bool,
    use_chronos: bool,
    use_agents: bool,
    use_technical: bool,
) -> None:
    """Execute the scan and display results."""
    from signalforge.config import load_config
    from signalforge.data.models import TradeTarget

    cfg = load_config()

    # Build engine list
    engines = []
    if use_kronos:
        engines.append("kronos")
    if use_qlib:
        engines.append("qlib")
    if use_chronos:
        engines.append("chronos")
    if use_agents:
        engines.append("agents")
    if use_technical:
        engines.append("technical")

    if not engines:
        st.warning("No engines selected. Enable at least one engine in the sidebar.")
        return

    # Progress
    progress = st.progress(0, text="Initializing...")
    status = st.empty()

    targets: list[TradeTarget] = []
    errors: list[str] = []

    for i, symbol in enumerate(symbols):
        progress.progress((i + 1) / len(symbols), text=f"Processing {symbol}...")
        status.text(f"Analyzing {symbol} with {len(engines)} engines...")

        try:
            from signalforge.pipeline import run_pipeline

            result = run_pipeline(
                symbols=[symbol],
                config=cfg,
                interval=interval,
                pred_len=pred_len,
                engines=engines,
            )
            targets.extend(result)
        except Exception as e:
            errors.append(f"{symbol}: {e}")

    progress.empty()
    status.empty()

    if errors:
        with st.expander(f"⚠️ {len(errors)} errors", expanded=False):
            for err in errors:
                st.error(err)

    if not targets:
        st.warning("No signals generated. Check data availability and engine configuration.")
        return

    # --- Display results ---
    _display_signal_table(targets, confidence_threshold)
    _display_signal_chart(targets)


def _display_signal_table(targets: list, confidence_threshold: float) -> None:
    """Display the signal table with color coding."""
    st.subheader("📋 Trade Signals")

    rows = []
    for t in targets:
        action = t.action.value if hasattr(t.action, "value") else str(t.action)
        rows.append({
            "Symbol": t.symbol,
            "Action": action,
            "Entry": f"${t.entry_price:.2f}",
            "Target": f"${t.target_price:.2f}",
            "Stop": f"${t.stop_loss:.2f}",
            "R:R": f"{t.risk_reward_ratio:.2f}",
            "Confidence": f"{t.confidence:.0%}",
            "Horizon": f"{t.horizon_days}d",
            "Rationale": t.rationale,
        })

    df = pd.DataFrame(rows)

    # Color-code by action
    def _style_action(val: str) -> str:
        if val == "BUY":
            return "background-color: #1a472a; color: #4ade80"
        if val == "SELL":
            return "background-color: #472a1a; color: #f87171"
        return "background-color: #333; color: #999"

    styled = df.style.map(_style_action, subset=["Action"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary metrics
    buys = sum(1 for t in targets if str(t.action) in ("BUY", "TradeAction.BUY"))
    sells = sum(1 for t in targets if str(t.action) in ("SELL", "TradeAction.SELL"))
    holds = len(targets) - buys - sells
    avg_conf = np.mean([t.confidence for t in targets]) if targets else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BUY Signals", buys)
    col2.metric("SELL Signals", sells)
    col3.metric("HOLD", holds)
    col4.metric("Avg Confidence", f"{avg_conf:.0%}")


def _display_signal_chart(targets: list) -> None:
    """Display a visual chart of signals."""
    st.subheader("📊 Signal Overview")

    if not targets:
        return

    chart_data = []
    for t in targets:
        action_str = t.action.value if hasattr(t.action, "value") else str(t.action)
        direction_val = 1 if action_str == "BUY" else (-1 if action_str == "SELL" else 0)
        chart_data.append({
            "Symbol": t.symbol,
            "Direction": direction_val * t.confidence,
            "Confidence": t.confidence,
            "R:R Ratio": t.risk_reward_ratio,
        })

    chart_df = pd.DataFrame(chart_data)

    if len(chart_df) > 1:
        st.bar_chart(chart_df.set_index("Symbol")["Direction"])


if __name__ == "__main__":
    main()
