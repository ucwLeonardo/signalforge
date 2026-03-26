"""Signal generation for paper trading — uses the real SignalForge pipeline
(LSTM, GBM, Technical, TradingAgents) with incremental data fetching."""

from __future__ import annotations

import sys

from signalforge.data.models import TradeAction, TradeTarget


def _get_symbols_for_categories(
    categories: list[str],
    config: "Config | None" = None,
) -> list[str]:
    """Get symbol list from config filtered by requested categories."""
    if config is None:
        from signalforge.config import load_config
        config = load_config()

    symbols: list[str] = []
    if "us_stocks" in categories:
        symbols.extend(config.us_stocks)
    if "crypto" in categories:
        symbols.extend(config.crypto)
    if "futures" in categories:
        symbols.extend(config.futures)
    if "options" in categories:
        symbols.extend(config.options)
    return symbols


def generate_real_signals(
    categories: list[str] | None = None,
    config: "Config | None" = None,
    progress_cb: "Callable[[dict], None] | None" = None,
) -> list[TradeTarget]:
    """Generate signals using the real pipeline (LSTM, GBM, Technical, TradingAgents).

    Uses incremental data fetching — first run downloads full history,
    subsequent runs only fetch new bars.

    Returns signals sorted by confidence descending.
    """
    from signalforge.config import load_config
    from signalforge.pipeline import run_pipeline

    if config is None:
        config = load_config()
    if categories is None:
        categories = ["us_stocks", "crypto"]

    symbols = _get_symbols_for_categories(categories, config)
    if not symbols:
        return []

    sys.stderr.write(
        f"[SignalForge] Generating signals for {len(symbols)} assets "
        f"(categories: {', '.join(categories)})...\n"
    )

    targets = run_pipeline(
        symbols=symbols,
        config=config,
        use_store=True,
        progress_cb=progress_cb,
    )

    # Sort by confidence descending
    targets.sort(key=lambda t: (t.confidence, t.risk_reward_ratio), reverse=True)

    sys.stderr.write(
        f"[SignalForge] Generated {len(targets)} signals from {len(symbols)} assets\n"
    )
    return targets


def generate_live_signals(
    categories: list[str] | None = None,
) -> list[TradeTarget]:
    """Generate signals using the real pipeline. Alias for generate_real_signals."""
    return generate_real_signals(categories=categories)


# ---------------------------------------------------------------------------
# Symbol classification for category filtering
# ---------------------------------------------------------------------------

def _classify_symbol_category(symbol: str) -> str:
    """Classify a symbol into asset category using the pipeline's classifier."""
    from signalforge.pipeline import _classify_symbol

    sym_type = _classify_symbol(symbol)
    # Map pipeline type names to config category names
    return {"stock": "us_stocks", "crypto": "crypto", "futures": "futures", "options": "options"}.get(
        sym_type, "us_stocks"
    )


def filter_signals_by_categories(
    signals: list[TradeTarget],
    categories: list[str],
) -> list[TradeTarget]:
    """Filter signals to only include symbols in the given categories."""
    return [s for s in signals if _classify_symbol_category(s.symbol) in categories]


# ---------------------------------------------------------------------------
# Portfolio allocation algorithms
# ---------------------------------------------------------------------------

def _kelly_allocate(
    signals: list[TradeTarget],
    total_budget_pct: float = 80.0,
    max_single_pct: float = 30.0,
    short_budget_pct: float = 10.0,
) -> dict[str, float]:
    """Allocate using a half-Kelly criterion based on signal confidence and R:R.

    The Kelly fraction for each signal is:
        f = confidence * (rr_ratio - 1) / rr_ratio

    where rr_ratio = risk_reward_ratio. This is the edge/odds formula from
    the Kelly criterion, adapted for trading signals.

    We use half-Kelly (f/2) for conservatism, then normalize so total
    allocation = total_budget_pct (default 80%, keeping 20% cash reserve).

    BUY signals share the main budget; SELL signals get a smaller short budget.
    Each position is capped at max_single_pct.
    """
    buy_signals = [s for s in signals if s.action == TradeAction.BUY]
    sell_signals = [s for s in signals if s.action == TradeAction.SELL]

    if not buy_signals and not sell_signals:
        return {}

    alloc: dict[str, float] = {}

    # --- BUY allocation via half-Kelly ---
    if buy_signals:
        kelly_scores: dict[str, float] = {}
        for s in buy_signals:
            rr = max(s.risk_reward_ratio, 0.01)
            # Kelly: f = p * (b-1)/b  where p=confidence, b=rr_ratio
            # For b>1 (favorable R:R), f is positive
            f = s.confidence * (rr - 1) / rr if rr > 1 else s.confidence * 0.1
            kelly_scores[s.symbol] = max(f / 2, 0.01)  # half-Kelly, floor at 1%

        total_score = sum(kelly_scores.values())
        for sym, score in kelly_scores.items():
            pct = (score / total_score) * total_budget_pct
            alloc[sym] = round(min(pct, max_single_pct), 1)

    # --- SELL allocation (small fixed budget) ---
    if sell_signals:
        per_short = short_budget_pct / len(sell_signals)
        for s in sell_signals:
            alloc[s.symbol] = round(min(per_short, 10.0), 1)

    return alloc


def auto_build_portfolio(
    manager: "PortfolioManager",
    categories: list[str],
    top_n: int = 5,
    **_kwargs: object,
) -> dict:
    """Auto-build a portfolio: generate signals → top N by confidence → Kelly allocation → open positions.

    Signal selection uses the project's existing engines (confidence-ranked).
    Allocation uses the half-Kelly criterion based on each signal's confidence and R:R ratio.

    Returns a summary dict of what was done.
    """
    from signalforge.paper.portfolio import PortfolioManager

    portfolio = manager.load()
    balance = portfolio.cash

    # 1. Generate real signals via pipeline (already sorted by confidence desc)
    all_signals = generate_real_signals(categories=categories)
    if not all_signals:
        return {"error": "No signals for selected categories", "positions_opened": []}

    # 2. Take top N by confidence (signals are pre-sorted)
    signals = all_signals[:top_n]

    # 4. Allocate via half-Kelly criterion
    allocation = _kelly_allocate(signals)
    allocation_method = "half_kelly"

    if not allocation:
        return {"error": "Could not determine allocation", "positions_opened": []}

    # 3. Open positions
    opened = []
    errors = []
    for signal in signals:
        pct = allocation.get(signal.symbol)
        if not pct or pct <= 0:
            continue

        amount = balance * (pct / 100)
        entry = signal.entry_price
        if entry <= 0:
            continue

        # Calculate quantity
        raw_qty = amount / entry
        is_crypto = "/" in signal.symbol
        qty = round(raw_qty, 6) if is_crypto else int(raw_qty)
        if qty <= 0:
            continue

        side = "short" if signal.action == TradeAction.SELL else "long"
        try:
            pos = manager.open_position(
                symbol=signal.symbol,
                side=side,
                qty=qty,
                entry_price=entry,
                stop_loss=signal.stop_loss,
                target_price=signal.target_price,
            )
            opened.append({
                "symbol": pos.symbol,
                "side": side,
                "qty": pos.qty,
                "entry_price": entry,
                "allocation_pct": pct,
                "cost": round(qty * entry, 2),
            })
        except ValueError as exc:
            errors.append({"symbol": signal.symbol, "error": str(exc)})

    portfolio = manager.load()
    return {
        "allocation_method": allocation_method,
        "allocation": allocation,
        "positions_opened": opened,
        "errors": errors,
        "cash_remaining": round(portfolio.cash, 2),
        "total_value": round(portfolio.total_value, 2),
        "all_signals": all_signals,  # full signal list for server to cache
    }
