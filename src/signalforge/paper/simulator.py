"""Signal generation for paper trading — uses the real SignalForge pipeline
(LSTM, GBM, Technical, TradingAgents) with incremental data fetching."""

from __future__ import annotations

import sys

from signalforge.data.models import TradeAction, TradeTarget


def _fetch_live_prices_for_build(symbols: list[str]) -> dict[str, float]:
    """Fetch current market prices at build time for accurate entry pricing.

    Raises PriceFetchError for unsupported asset types (e.g. options).
    Logs warnings for individual fetch failures but continues.
    """
    from signalforge.paper.prices import fetch_prices

    prices = fetch_prices(symbols)
    missing = [s for s in symbols if s not in prices]
    if missing:
        sys.stderr.write(
            f"[Build] WARNING: No live price for {', '.join(missing)} — "
            "will fall back to signal entry_price\n"
        )
    else:
        sys.stderr.write(
            f"[Build] Fetched live prices for all {len(prices)} symbols\n"
        )
    return prices


def _get_config_symbols(
    categories: list[str],
    config: "Config",
) -> list[str]:
    """Return only the config-defined symbols (no discovery)."""
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


def _get_symbols_for_categories(
    categories: list[str],
    config: "Config | None" = None,
    config_only: bool = False,
) -> list[str]:
    """Get symbol list using dynamic discovery, falling back to config.

    If *config_only* is True, skip discovery and return only config symbols.
    """
    if config is None:
        from signalforge.config import load_config
        config = load_config()

    if config_only:
        return _get_config_symbols(categories, config)

    # Try dynamic discovery first
    try:
        from signalforge.data.discovery import discover_all

        return discover_all(categories, config)
    except Exception:
        pass

    return _get_config_symbols(categories, config)


def generate_real_signals(
    categories: list[str] | None = None,
    config: "Config | None" = None,
    progress_cb: "Callable[[dict], None] | None" = None,
    cancel_flag: "Callable[[], bool] | None" = None,
    config_only: bool = False,
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

    # Report discovery phase
    if progress_cb:
        progress_cb({"total": 0, "completed": 0, "symbol": "",
                      "stage": "discovery", "detail": f"Discovering assets for {', '.join(categories)}..."})

    symbols = _get_symbols_for_categories(categories, config, config_only=config_only)
    if not symbols:
        return []

    sys.stderr.write(
        f"[SignalForge] Generating signals for {len(symbols)} assets "
        f"(categories: {', '.join(categories)})...\n"
    )

    if progress_cb:
        progress_cb({"total": len(symbols), "completed": 0, "symbol": "",
                      "stage": "discovery", "detail": f"Discovered {len(symbols)} assets, starting pipeline..."})

    targets = run_pipeline(
        symbols=symbols,
        config=config,
        use_store=True,
        progress_cb=progress_cb,
        cancel_flag=cancel_flag,
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
) -> dict[str, float]:
    """Allocate using a half-Kelly criterion based on signal confidence and R:R.

    The Kelly fraction for each signal is:
        f = confidence * (rr_ratio - 1) / rr_ratio

    where rr_ratio = risk_reward_ratio. This is the edge/odds formula from
    the Kelly criterion, adapted for trading signals.

    We use half-Kelly (f/2) for conservatism, then normalize so total
    allocation = total_budget_pct (default 80%, keeping 20% cash reserve).

    Both BUY and SELL signals share the same Kelly-weighted pool.
    Each position is capped at max_single_pct.
    """
    if not signals:
        return {}

    kelly_scores: dict[str, float] = {}
    for s in signals:
        rr = max(s.risk_reward_ratio, 0.01)
        # Kelly: f = p * (b-1)/b  where p=confidence, b=rr_ratio
        # For b>1 (favorable R:R), f is positive
        f = s.confidence * (rr - 1) / rr if rr > 1 else s.confidence * 0.1
        kelly_scores[s.symbol] = max(f / 2, 0.01)  # half-Kelly, floor at 1%

    total_score = sum(kelly_scores.values())
    alloc: dict[str, float] = {}
    for sym, score in kelly_scores.items():
        pct = (score / total_score) * total_budget_pct
        alloc[sym] = round(min(pct, max_single_pct), 1)

    return alloc


def _stratified_select(
    signals: list[TradeTarget],
    categories: list[str],
    top_n: int = 5,
) -> list[TradeTarget]:
    """Select top N signals with cross-asset diversification.

    Strategy (stratified selection):
      1. Group BUY signals by asset category.
      2. Guarantee each category at least 1 slot (if it has signals).
      3. Remaining slots go to the highest-confidence signals globally.
      4. SELL signals are appended after (they use a separate short budget).

    This ensures the portfolio isn't concentrated in one asset class
    while still prioritising the strongest signals.
    """
    buy_signals = [s for s in signals if s.action == TradeAction.BUY]
    sell_signals = [s for s in signals if s.action == TradeAction.SELL]

    # Group buys by category
    by_cat: dict[str, list[TradeTarget]] = {}
    for s in buy_signals:
        cat = _classify_symbol_category(s.symbol)
        by_cat.setdefault(cat, []).append(s)

    # Each list is already sorted by confidence desc (from scan)
    selected: list[TradeTarget] = []
    seen: set[str] = set()

    # Phase 1: one best signal per category (guaranteed diversity)
    for cat in categories:
        if cat in by_cat and by_cat[cat]:
            best = by_cat[cat][0]
            selected.append(best)
            seen.add(best.symbol)

    # Phase 2: fill remaining slots from global ranking
    remaining = top_n - len(selected)
    if remaining > 0:
        for s in buy_signals:
            if s.symbol not in seen:
                selected.append(s)
                seen.add(s.symbol)
                remaining -= 1
                if remaining <= 0:
                    break

    # Append sell signals (separate short budget, don't count toward top_n)
    for s in sell_signals:
        if s.symbol not in seen:
            selected.append(s)
            seen.add(s.symbol)

    return selected


def build_from_cached_signals(
    manager: "PortfolioManager",
    cached_signals: list[TradeTarget],
    categories: list[str],
    top_n: int = 5,
) -> dict:
    """Build portfolio from pre-scanned signals — no re-scanning needed.

    Uses stratified selection for cross-asset diversification,
    then allocates via half-Kelly criterion.

    Returns a summary dict including allocation preview and opened positions.
    """
    from signalforge.paper.portfolio import PortfolioManager

    portfolio = manager.load()
    balance = portfolio.cash

    # Filter by account categories
    filtered = filter_signals_by_categories(cached_signals, categories)
    if not filtered:
        return {"error": "No signals match account categories", "positions_opened": []}

    # Stratified selection: diversify across categories, then fill by confidence
    signals = _stratified_select(filtered, categories, top_n)

    # Allocate via half-Kelly
    allocation = _kelly_allocate(signals)
    if not allocation:
        return {"error": "Could not determine allocation", "positions_opened": []}

    # Fetch real-time prices for selected symbols (market order simulation)
    live_prices = _fetch_live_prices_for_build([s.symbol for s in signals])

    # Open positions using real-time prices (not stale signal entry_price)
    opened = []
    errors = []
    for signal in signals:
        pct = allocation.get(signal.symbol)
        if not pct or pct <= 0:
            continue

        # Use live price if available, fall back to signal entry_price
        entry = live_prices.get(signal.symbol, signal.entry_price)
        if entry <= 0:
            continue

        amount = balance * (pct / 100)
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
                "signal_entry": signal.entry_price,
                "allocation_pct": pct,
                "cost": round(qty * entry, 2),
            })
        except ValueError as exc:
            errors.append({"symbol": signal.symbol, "error": str(exc)})

    portfolio = manager.load()

    # Category breakdown of selected signals
    cat_breakdown: dict[str, int] = {}
    for s in signals:
        cat = _classify_symbol_category(s.symbol)
        cat_breakdown[cat] = cat_breakdown.get(cat, 0) + 1

    return {
        "allocation_method": "half_kelly",
        "selection_method": "stratified",
        "allocation": allocation,
        "positions_opened": opened,
        "errors": errors,
        "signals_considered": len(filtered),
        "signals_selected": len(signals),
        "category_breakdown": cat_breakdown,
        "cash_remaining": round(portfolio.cash, 2),
        "total_value": round(portfolio.total_value, 2),
    }


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
