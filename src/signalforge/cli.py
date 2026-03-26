"""SignalForge CLI - Generate buy/sell signals for stocks, crypto, and futures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="signalforge",
    help="Multi-asset buy/sell signal generator",
    no_args_is_help=True,
)
paper_app = typer.Typer(
    name="paper",
    help="Paper trading with virtual portfolio",
    no_args_is_help=True,
)
app.add_typer(paper_app, name="paper")
console = Console()


@app.command()
def scan(
    symbols: Optional[list[str]] = typer.Argument(
        None, help="Symbols to scan (e.g., AAPL BTC/USDT ES=F). Uses config if empty."
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config YAML"
    ),
    interval: str = typer.Option("1d", "--interval", "-i", help="Data interval (1d, 1h, etc.)"),
    pred_len: int = typer.Option(5, "--horizon", "-h", help="Prediction horizon in bars"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output: table, json, csv"),
    engine: str = typer.Option("all", "--engine", "-e", help="Engine: kronos, qlib, chronos, agents, lstm, gbm, technical, all"),
) -> None:
    """Scan assets and generate buy/sell signals with price targets."""
    from signalforge.config import load_config
    from signalforge.pipeline import run_pipeline

    cfg = load_config(config_path)

    # Determine symbols to scan
    if not symbols:
        all_symbols = cfg.us_stocks + cfg.crypto + cfg.futures + cfg.options
    else:
        all_symbols = list(symbols)

    if not all_symbols:
        console.print("[red]No symbols specified. Use arguments or configure in YAML.[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Scanning {len(all_symbols)} assets[/bold]\n"
            f"Horizon: {pred_len} bars | Interval: {interval} | Engine: {engine}",
            title="SignalForge",
            border_style="blue",
        )
    )

    targets = run_pipeline(
        symbols=all_symbols,
        config=cfg,
        interval=interval,
        pred_len=pred_len,
        engines=[engine] if engine != "all" else None,
    )

    # Output results
    from signalforge.output.report import ReportGenerator

    generator = ReportGenerator()
    output = generator.generate_report(targets, fmt=output_format)
    console.print(output)


@app.command()
def fetch(
    symbols: list[str] = typer.Argument(..., help="Symbols to fetch data for"),
    interval: str = typer.Option("1d", "--interval", "-i"),
    days: int = typer.Option(365, "--days", "-d", help="Days of history to fetch"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Fetch and cache market data for symbols."""
    from datetime import datetime, timedelta

    from signalforge.config import load_config
    from signalforge.data.providers import get_provider
    from signalforge.data.store import DataStore

    cfg = load_config(config_path)
    store = DataStore(cfg.data_dir)

    end = datetime.now()
    start = end - timedelta(days=days)

    for symbol in symbols:
        console.print(f"Fetching {symbol}...", end=" ")
        try:
            provider = get_provider(symbol)
            df = provider.fetch(symbol, interval, start, end)
            store.save(symbol, interval, df)
            console.print(f"[green]{len(df)} bars saved[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def predict(
    symbol: str = typer.Argument(..., help="Symbol to predict"),
    horizon: int = typer.Option(5, "--horizon", "-h", help="Prediction horizon in bars"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run Kronos prediction for a single symbol."""
    from datetime import datetime, timedelta

    from rich.table import Table

    from signalforge.config import load_config
    from signalforge.data.providers import get_provider
    from signalforge.engines.kronos_engine import KronosEngine

    cfg = load_config(config_path)

    console.print(f"[bold]Predicting {symbol} for {horizon} bars...[/bold]")

    # Fetch recent data
    provider = get_provider(symbol)
    end = datetime.now()
    start = end - timedelta(days=cfg.kronos.lookback * 2)
    df = provider.fetch(symbol, "1d", start, end)

    if df.empty:
        console.print(f"[red]No data for {symbol}[/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(df)} bars, last: {df.index[-1]}")

    # Run Kronos
    engine = KronosEngine(cfg.kronos)
    result = engine.predict(df, pred_len=horizon)

    # Display predictions
    table = Table(title=f"Predicted Candles for {symbol}")
    table.add_column("Date", style="cyan")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right", style="green")
    table.add_column("Low", justify="right", style="red")
    table.add_column("Close", justify="right", style="bold")
    table.add_column("Volume", justify="right", style="dim")

    for idx, row in result.iterrows():
        table.add_row(
            str(idx)[:10],
            f"{row.get('open', 0):.2f}",
            f"{row.get('high', 0):.2f}",
            f"{row.get('low', 0):.2f}",
            f"{row.get('close', 0):.2f}",
            f"{row.get('volume', 0):.0f}",
        )

    console.print(table)

    # Show buy/sell interpretation
    last_close = df["close"].iloc[-1]
    pred_close = result["close"].iloc[-1]
    pred_high = result["high"].max()
    pred_low = result["low"].min()
    change_pct = (pred_close - last_close) / last_close * 100

    direction = "[green]BULLISH[/green]" if change_pct > 0 else "[red]BEARISH[/red]"
    console.print(f"\nCurrent: {last_close:.2f} → Predicted close: {pred_close:.2f} ({change_pct:+.2f}%) {direction}")
    console.print(f"Predicted range: [red]{pred_low:.2f}[/red] - [green]{pred_high:.2f}[/green]")


@app.command()
def setup() -> None:
    """Set up SignalForge: create directories and check dependencies."""
    from signalforge.config import load_config

    cfg = load_config()

    for dir_path in [cfg.data_dir, cfg.cache_dir, cfg.results_dir]:
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created[/green] {p}")

    # Check dependencies
    checks = {
        "yfinance": "pip install yfinance",
        "ccxt": "pip install ccxt",
        "pandas_ta": "pip install pandas-ta",
        "torch": "pip install torch",
    }

    console.print("\n[bold]Dependency Check:[/bold]")
    for module, install_cmd in checks.items():
        try:
            __import__(module)
            console.print(f"  [green]OK[/green] {module}")
        except ImportError:
            console.print(f"  [red]MISSING[/red] {module} → {install_cmd}")

    # Check Kronos
    try:
        from model import Kronos  # noqa: F401
        console.print("  [green]OK[/green] Kronos")
    except ImportError:
        console.print(
            "  [yellow]OPTIONAL[/yellow] Kronos → "
            "git clone https://github.com/shiyu-coder/Kronos && pip install -r Kronos/requirements.txt"
        )

    # Check ML engine dependencies
    ml_checks = {
        "qlib": "pip install pyqlib",
        "chronos": "pip install chronos-forecasting",
        "lightgbm": "pip install lightgbm",
        "tradingagents": "pip install tradingagents",
        "rdagent": "pip install rdagent",
    }
    console.print("\n[bold]ML Engine Dependencies:[/bold]")
    for module, install_cmd in ml_checks.items():
        try:
            __import__(module)
            console.print(f"  [green]OK[/green] {module}")
        except ImportError:
            console.print(f"  [yellow]OPTIONAL[/yellow] {module} → {install_cmd}")

    # Built-in ML engines (always available via PyTorch/sklearn)
    console.print("\n[bold]Built-in ML Engines:[/bold]")
    console.print("  [green]OK[/green] LSTM (PyTorch seq2seq)")
    console.print("  [green]OK[/green] GBM (LightGBM or sklearn fallback)")

    console.print("\n[bold green]Setup complete![/bold green]")


@app.command()
def evolve(
    mode: str = typer.Option("factor", "--mode", "-m", help="Evolution mode: factor, model, joint"),
    iterations: int = typer.Option(20, "--iterations", "-n", help="Max iterations"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run automated factor/model evolution using RD-Agent."""
    from signalforge.evolution import EvolutionConfig, FactorEvolver

    console.print(
        Panel(
            f"[bold]Factor Evolution[/bold]\n"
            f"Mode: {mode} | Max iterations: {iterations}",
            title="SignalForge Evolution",
            border_style="magenta",
        )
    )

    evo_config = EvolutionConfig(
        enabled=True,
        mode=mode,
        max_iterations=iterations,
    )

    evolver = FactorEvolver(evo_config)
    result = evolver.run()

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Iterations completed: {result.iterations_completed}")
    console.print(f"  Factors discovered: {len(result.factors_discovered)}")
    console.print(f"  Models improved: {len(result.models_improved)}")

    if result.factors_discovered:
        from rich.table import Table

        table = Table(title="Discovered Factors")
        table.add_column("Name", style="cyan")
        table.add_column("Window", justify="right")
        table.add_column("Status")

        for factor in result.factors_discovered[:20]:  # Show top 20
            table.add_row(
                factor.get("name", ""),
                str(factor.get("window", "")),
                factor.get("status", ""),
            )
        console.print(table)


@app.command()
def top(
    n: int = typer.Option(5, "--n", "-n", help="Number of top signals to show"),
    action_filter: str = typer.Option("buy", "--action", "-a", help="Filter: buy, sell, all"),
    asset_type: str = typer.Option("all", "--type", "-t", help="Asset type: stock, crypto, futures, options, all"),
    symbols: Optional[list[str]] = typer.Argument(
        None, help="Symbols to scan. Uses config defaults if empty."
    ),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    interval: str = typer.Option("1d", "--interval", "-i"),
    pred_len: int = typer.Option(5, "--horizon", "-h"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output: table, json, csv"),
) -> None:
    """Find top N most confident buy/sell signals across assets.

    Examples:
        signalforge top                          # Top 5 buy signals across all assets
        signalforge top -n 10 -a sell -t crypto  # Top 10 sell signals for crypto
        signalforge top -a all -t stock          # Top 5 signals (buy+sell) for stocks
        signalforge top AAPL NVDA TSLA -n 3      # Top 3 from specific symbols
    """
    from signalforge.config import load_config
    from signalforge.data.models import TradeAction, classify_symbol
    from signalforge.pipeline import run_pipeline

    cfg = load_config(config_path)

    # Determine symbols
    if symbols:
        all_symbols = list(symbols)
    elif asset_type == "stock":
        all_symbols = cfg.us_stocks
    elif asset_type == "crypto":
        all_symbols = cfg.crypto
    elif asset_type == "futures":
        all_symbols = cfg.futures
    elif asset_type == "options":
        all_symbols = cfg.options
    else:
        all_symbols = cfg.us_stocks + cfg.crypto + cfg.futures + cfg.options

    if not all_symbols:
        console.print(f"[red]No {asset_type} symbols configured.[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Finding top {n} {action_filter.upper()} signals[/bold]\n"
            f"Scanning {len(all_symbols)} {asset_type} assets | Horizon: {pred_len} bars",
            title="SignalForge Top",
            border_style="green",
        )
    )

    targets = run_pipeline(
        symbols=all_symbols,
        config=cfg,
        interval=interval,
        pred_len=pred_len,
    )

    # Filter by action
    action_upper = action_filter.upper()
    if action_upper == "BUY":
        targets = [t for t in targets if t.action == TradeAction.BUY]
    elif action_upper == "SELL":
        targets = [t for t in targets if t.action == TradeAction.SELL]
    # "all" keeps everything

    # Sort by confidence descending, then by risk-reward ratio
    targets.sort(key=lambda t: (t.confidence, t.risk_reward_ratio), reverse=True)

    # Take top N
    top_targets = targets[:n]

    if not top_targets:
        console.print(f"[yellow]No {action_filter.upper()} signals found.[/yellow]")
        raise typer.Exit(0)

    from signalforge.output.report import ReportGenerator

    generator = ReportGenerator()
    output = generator.generate_report(top_targets, fmt=output_format)
    console.print(output)

    # Summary line
    if top_targets:
        best = top_targets[0]
        console.print(
            f"\n[bold green]Best signal:[/bold green] {best.symbol} "
            f"[bold]{best.action.value}[/bold] at {best.entry_price:.2f} "
            f"→ {best.target_price:.2f} (confidence: {best.confidence:.0%}, R:R {best.risk_reward_ratio:.1f})"
        )


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit port"),
) -> None:
    """Launch the interactive Streamlit dashboard."""
    import subprocess

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    console.print(f"[bold]Launching dashboard on port {port}...[/bold]")
    subprocess.run(
        ["streamlit", "run", str(dashboard_path), "--server.port", str(port)],
        check=True,
    )


# ---------------------------------------------------------------------------
# Paper Trading Commands
# ---------------------------------------------------------------------------
@paper_app.command()
def init(
    balance: float = typer.Option(5000.0, "--balance", "-b", help="Starting balance in USD"),
    portfolio_path: Optional[Path] = typer.Option(None, "--path", "-p", help="Portfolio JSON path"),
) -> None:
    """Initialize a new paper trading portfolio."""
    from signalforge.paper.portfolio import PortfolioManager

    mgr = PortfolioManager(portfolio_path)
    if mgr.exists():
        console.print("[yellow]Portfolio already exists. Use --path for a new one, or delete the existing file.[/yellow]")
        console.print(f"  Path: {mgr.path}")
        raise typer.Exit(1)

    portfolio = mgr.init(balance=balance)
    console.print(
        Panel(
            f"[bold green]Paper portfolio created![/bold green]\n\n"
            f"  Balance: [bold]${portfolio.cash:,.2f}[/bold]\n"
            f"  Path:    {mgr.path}",
            title="SignalForge Paper Trading",
            border_style="green",
        )
    )


@paper_app.command()
def status(
    portfolio_path: Optional[Path] = typer.Option(None, "--path", "-p"),
) -> None:
    """Show current portfolio status, positions, and P&L."""
    from rich.table import Table

    from signalforge.paper.portfolio import PortfolioManager

    mgr = PortfolioManager(portfolio_path)
    if not mgr.exists():
        console.print("[red]No portfolio found. Run 'signalforge paper init' first.[/red]")
        raise typer.Exit(1)

    # Update positions with live prices
    from signalforge.paper.simulator import LIVE_PRICES_20260326

    p = mgr.load()
    held_symbols = {pos.symbol for pos in p.positions}
    price_updates = {s: px for s, px in LIVE_PRICES_20260326.items() if s in held_symbols}
    if price_updates:
        mgr.update_prices(price_updates)
        p = mgr.load()

    # Summary
    pnl_style = "green" if p.total_pnl >= 0 else "red"
    console.print(
        Panel(
            f"  Initial Balance: ${p.initial_balance:,.2f}\n"
            f"  Cash:            ${p.cash:,.2f}\n"
            f"  Positions Value: ${p.positions_value:,.2f}\n"
            f"  Total Value:     [bold]${p.total_value:,.2f}[/bold]\n"
            f"  Unrealized P&L:  [{pnl_style}]${p.unrealized_pnl:,.2f}[/{pnl_style}]\n"
            f"  Realized P&L:    [{pnl_style}]${p.realized_pnl:,.2f}[/{pnl_style}]\n"
            f"  Total P&L:       [{pnl_style}]${p.total_pnl:,.2f} ({p.total_pnl_pct:+.2f}%)[/{pnl_style}]",
            title="Portfolio Summary",
            border_style="cyan",
        )
    )

    # Open positions
    if p.positions:
        table = Table(title="Open Positions", show_lines=True)
        table.add_column("Symbol", style="bold")
        table.add_column("Side", justify="center")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Target", justify="right", style="green")
        table.add_column("Stop", justify="right", style="red")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")

        for pos in p.positions:
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            table.add_row(
                pos.symbol,
                f"[bold {'green' if pos.side == 'long' else 'red'}]{pos.side.upper()}[/bold {'green' if pos.side == 'long' else 'red'}]",
                f"{pos.qty:.4g}",
                f"${pos.entry_price:,.2f}",
                f"${pos.current_price:,.2f}",
                f"${pos.target_price:,.2f}",
                f"${pos.stop_loss:,.2f}",
                f"[{pnl_color}]${pos.unrealized_pnl:,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pos.pnl_pct:+.2f}%[/{pnl_color}]",
            )
        console.print(table)
    else:
        console.print("[dim]No open positions.[/dim]")


@paper_app.command()
def auto(
    n: int = typer.Option(5, "--n", "-n", help="Max signals to trade"),
    portfolio_path: Optional[Path] = typer.Option(None, "--path", "-p"),
) -> None:
    """Auto-trade top signals using position sizing rules (max 20% per trade)."""
    from signalforge.paper.executor import execute_signals
    from signalforge.paper.portfolio import PortfolioManager
    from signalforge.paper.simulator import generate_live_signals

    mgr = PortfolioManager(portfolio_path)
    if not mgr.exists():
        console.print("[red]No portfolio found. Run 'signalforge paper init' first.[/red]")
        raise typer.Exit(1)

    # Generate signals from live prices
    signals = generate_live_signals()
    if not signals:
        console.print("[yellow]No signals generated.[/yellow]")
        raise typer.Exit(0)

    # Show signals before trading
    console.print(
        Panel(
            f"[bold]Generated {len(signals)} signals from live prices[/bold]",
            title="SignalForge Auto-Trade",
            border_style="green",
        )
    )

    from signalforge.output.report import ReportGenerator

    report = ReportGenerator()
    console.print(report.generate_report(signals[:n], fmt="table"))

    # Execute top N signals
    opened = execute_signals(signals[:n], mgr)

    if opened:
        console.print(f"\n[bold green]Opened {len(opened)} positions:[/bold green]")
        for pos in opened:
            console.print(
                f"  {pos.side.upper()} {pos.symbol}: "
                f"{pos.qty:.4g} shares @ ${pos.entry_price:,.2f} "
                f"(${pos.qty * pos.entry_price:,.2f})"
            )
    else:
        console.print("\n[yellow]No new positions opened (signals filtered or insufficient cash).[/yellow]")

    # Show portfolio after
    p = mgr.load()
    console.print(f"\n  Cash remaining: ${p.cash:,.2f} | Positions: {len(p.positions)} | Total: ${p.total_value:,.2f}")


@paper_app.command()
def close(
    symbol: str = typer.Argument(..., help="Symbol to close"),
    price: Optional[float] = typer.Option(None, "--price", help="Exit price (uses entry if omitted)"),
    reason: str = typer.Option("manual", "--reason", "-r", help="Reason: manual, target_hit, stop_hit"),
    portfolio_path: Optional[Path] = typer.Option(None, "--path", "-p"),
) -> None:
    """Close an open position."""
    from signalforge.paper.portfolio import PortfolioManager

    mgr = PortfolioManager(portfolio_path)
    if not mgr.exists():
        console.print("[red]No portfolio found.[/red]")
        raise typer.Exit(1)

    p = mgr.load()
    pos = next((pos for pos in p.positions if pos.symbol == symbol), None)
    if pos is None:
        console.print(f"[red]No open position for {symbol}[/red]")
        raise typer.Exit(1)

    exit_price = price if price is not None else pos.current_price
    trade = mgr.close_position(symbol, exit_price=exit_price, reason=reason)

    pnl_color = "green" if trade.pnl >= 0 else "red"
    console.print(
        f"[bold]Closed {symbol}:[/bold] {trade.side.upper()} "
        f"{trade.qty:.4g} @ ${trade.exit_price:,.2f} "
        f"[{pnl_color}]P&L: ${trade.pnl:,.2f} ({trade.pnl_pct:+.2f}%)[/{pnl_color}]"
    )


@paper_app.command()
def history(
    portfolio_path: Optional[Path] = typer.Option(None, "--path", "-p"),
) -> None:
    """Show completed trade history."""
    from rich.table import Table

    from signalforge.paper.portfolio import PortfolioManager

    mgr = PortfolioManager(portfolio_path)
    if not mgr.exists():
        console.print("[red]No portfolio found.[/red]")
        raise typer.Exit(1)

    p = mgr.load()
    if not p.trades:
        console.print("[dim]No completed trades yet.[/dim]")
        return

    table = Table(title="Trade History", show_lines=True)
    table.add_column("Symbol", style="bold")
    table.add_column("Side", justify="center")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")
    table.add_column("Reason")
    table.add_column("Closed", style="dim")

    total_pnl = 0.0
    for t in p.trades:
        pnl_color = "green" if t.pnl >= 0 else "red"
        total_pnl += t.pnl
        table.add_row(
            t.symbol,
            t.side.upper(),
            f"{t.qty:.4g}",
            f"${t.entry_price:,.2f}",
            f"${t.exit_price:,.2f}",
            f"[{pnl_color}]${t.pnl:,.2f}[/{pnl_color}]",
            f"[{pnl_color}]{t.pnl_pct:+.2f}%[/{pnl_color}]",
            t.reason,
            t.closed_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    pnl_color = "green" if total_pnl >= 0 else "red"
    console.print(f"\n[bold]Total realized P&L: [{pnl_color}]${total_pnl:,.2f}[/{pnl_color}][/bold]")


@paper_app.command(name="dashboard")
def paper_dashboard(
    port: int = typer.Option(8787, "--port", help="Server port"),
    portfolio_path: Optional[Path] = typer.Option(None, "--path", "-p"),
) -> None:
    """Launch the paper trading web dashboard."""
    from signalforge.paper.server import main as server_main

    args = [f"--port={port}"]
    if portfolio_path:
        args.append(f"--path={portfolio_path}")

    console.print(
        Panel(
            f"[bold]Paper Trading Dashboard[/bold]\n\n"
            f"  URL:  http://localhost:{port}/\n"
            f"  Press Ctrl+C to stop.",
            title="SignalForge",
            border_style="green",
        )
    )
    server_main(args)


if __name__ == "__main__":
    app()
