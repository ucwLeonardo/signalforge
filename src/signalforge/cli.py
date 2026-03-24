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
    engine: str = typer.Option("all", "--engine", "-e", help="Engine: kronos, technical, all"),
) -> None:
    """Scan assets and generate buy/sell signals with price targets."""
    from signalforge.config import load_config
    from signalforge.pipeline import run_pipeline

    cfg = load_config(config_path)

    # Determine symbols to scan
    if not symbols:
        all_symbols = cfg.us_stocks + cfg.crypto + cfg.futures
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

    # Check Phase 2 dependencies
    phase2_checks = {
        "qlib": "pip install pyqlib",
        "chronos": "pip install chronos-forecasting",
        "tradingagents": "pip install tradingagents",
        "rdagent": "pip install rdagent",
    }
    console.print("\n[bold]Phase 2+ Dependencies:[/bold]")
    for module, install_cmd in phase2_checks.items():
        try:
            __import__(module)
            console.print(f"  [green]OK[/green] {module}")
        except ImportError:
            console.print(f"  [yellow]OPTIONAL[/yellow] {module} → {install_cmd}")

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


if __name__ == "__main__":
    app()
