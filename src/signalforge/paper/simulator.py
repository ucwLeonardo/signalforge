"""Signal generation for paper trading — uses the real SignalForge pipeline
(LSTM, GBM, Technical, TradingAgents) with incremental data fetching."""

from __future__ import annotations

import sys

from signalforge.data.models import TradeAction, TradeTarget


def _rescale_stop_target(
    signal_entry: float,
    signal_stop: float,
    signal_target: float,
    actual_entry: float,
    side: str,
) -> tuple[float, float]:
    """Recalculate stop/target based on actual entry price.

    Preserves the percentage distances from the original signal so that
    risk/reward ratio stays the same regardless of entry price drift.
    """
    if signal_entry <= 0:
        return signal_stop, signal_target

    if side == "long":
        stop_pct = (signal_entry - signal_stop) / signal_entry
        target_pct = (signal_target - signal_entry) / signal_entry
        actual_stop = round(actual_entry * (1 - stop_pct), 6)
        actual_target = round(actual_entry * (1 + target_pct), 6)
    else:
        # Short: stop is above entry, target is below
        stop_pct = (signal_stop - signal_entry) / signal_entry
        target_pct = (signal_entry - signal_target) / signal_entry
        actual_stop = round(actual_entry * (1 + stop_pct), 6)
        actual_target = round(actual_entry * (1 - target_pct), 6)

    return actual_stop, actual_target


def _fetch_live_prices_for_build(symbols: list[str]) -> dict[str, float]:
    """Get current market prices for build, reusing server's cached prices.

    Uses the background price updater cache first, only fetches missing symbols.
    """
    from signalforge.paper.server import _live_prices as cached_prices

    # Use cached prices from the background updater
    prices = {s: cached_prices[s] for s in symbols if s in cached_prices}
    missing = [s for s in symbols if s not in prices]

    if missing:
        # Only fetch symbols not in cache
        from signalforge.paper.prices import fetch_prices

        fresh = fetch_prices(missing)
        prices.update(fresh)
        still_missing = [s for s in symbols if s not in prices]
        if still_missing:
            sys.stderr.write(
                f"[Build] WARNING: No live price for {', '.join(still_missing)} — "
                "will fall back to signal entry_price\n"
            )

    sys.stderr.write(
        f"[Build] Prices ready: {len(prices)}/{len(symbols)} "
        f"({len(prices) - len(missing)} cached, {len(missing)} fetched)\n"
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
                      "stage": "discovery", "detail": f"Discovering assets for {', '.join(categories)}...",
                      "phase_pct": 0})

    symbols = _get_symbols_for_categories(categories, config, config_only=config_only)
    if not symbols:
        return []

    sys.stderr.write(
        f"[SignalForge] Generating signals for {len(symbols)} assets "
        f"(categories: {', '.join(categories)})...\n"
    )

    if progress_cb:
        progress_cb({"total": len(symbols), "completed": 0, "symbol": "",
                      "stage": "discovery", "detail": f"Discovered {len(symbols)} assets, starting pipeline...",
                      "phase_pct": 0})

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
    total_budget_pct: float = 92.0,
    max_single_pct: float = 35.0,
) -> dict[str, float]:
    """Allocate using a half-Kelly criterion based on signal confidence and R:R.

    The Kelly fraction for each signal is:
        f = confidence * (rr_ratio - 1) / rr_ratio

    where rr_ratio = risk_reward_ratio. This is the edge/odds formula from
    the Kelly criterion, adapted for trading signals.

    We use half-Kelly (f/2) for conservatism, then normalize so total
    allocation = total_budget_pct (default 92%, keeping 8% cash reserve).

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
      1. Group all signals (BUY + SELL) by asset category.
      2. Guarantee each category at least 1 slot (highest confidence, regardless of direction).
      3. Remaining slots go to the highest-confidence signals globally.
      Total positions are capped at top_n.

    This ensures the portfolio isn't concentrated in one asset class
    while still prioritising the strongest signals.
    """
    # Sort all signals by confidence descending
    all_sorted = sorted(signals, key=lambda s: s.confidence, reverse=True)

    # Group by category
    by_cat: dict[str, list[TradeTarget]] = {}
    for s in all_sorted:
        cat = _classify_symbol_category(s.symbol)
        by_cat.setdefault(cat, []).append(s)

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
        for s in all_sorted:
            if s.symbol not in seen:
                selected.append(s)
                seen.add(s.symbol)
                remaining -= 1
                if remaining <= 0:
                    break

    return selected


def build_from_cached_signals(
    manager: "PortfolioManager",
    cached_signals: list[TradeTarget],
    categories: list[str],
    top_n: int = 5,
) -> dict:
    """Build/rebalance portfolio from pre-scanned signals.

    Uses stratified selection for cross-asset diversification,
    then allocates via half-Kelly criterion.

    Rebalance logic:
    - New signals not in portfolio → open new position
    - Existing positions in new signals → add or reduce to match target allocation
    - Existing positions NOT in new signals → close entirely

    Returns a summary dict including allocation preview and actions taken.
    """
    from signalforge.paper.portfolio import PortfolioManager

    portfolio = manager.load()
    total_value = portfolio.total_value  # cash + positions value

    # Filter by account categories
    filtered = filter_signals_by_categories(cached_signals, categories)
    if not filtered:
        return {"error": "No signals match account categories", "positions_opened": []}

    # Stratified selection: diversify across categories, then fill by confidence
    signals = _stratified_select(filtered, categories, top_n)

    # Allocate via half-Kelly based on total portfolio value (not just cash)
    allocation = _kelly_allocate(signals)
    if not allocation:
        return {"error": "Could not determine allocation", "positions_opened": []}

    # Fetch real-time prices for all relevant symbols
    existing_symbols = {p.symbol for p in portfolio.positions}
    all_symbols = list({s.symbol for s in signals} | existing_symbols)
    live_prices = _fetch_live_prices_for_build(all_symbols)

    selected_symbols = {s.symbol for s in signals}

    # Phase 1: Close positions that are NOT in the new signal set
    closed = []
    for pos in list(portfolio.positions):
        if pos.symbol not in selected_symbols:
            price = live_prices.get(pos.symbol, pos.current_price)
            try:
                manager.close_position(pos.symbol, price, reason="rebalance")
                closed.append({"symbol": pos.symbol, "side": pos.side,
                               "qty": pos.qty, "exit_price": price})
            except ValueError:
                pass

    # Reload after closes to get updated cash
    portfolio = manager.load()
    total_value = portfolio.total_value
    existing_map = {p.symbol: p for p in portfolio.positions}

    # Phase 2: Rebalance existing + open new
    opened = []
    added = []
    reduced = []
    errors = []

    for signal in signals:
        pct = allocation.get(signal.symbol)
        if not pct or pct <= 0:
            continue

        price = live_prices.get(signal.symbol, signal.entry_price)
        if price <= 0:
            continue

        target_value = total_value * (pct / 100)
        is_crypto = "/" in signal.symbol
        side = "short" if signal.action == TradeAction.SELL else "long"

        if signal.symbol in existing_map:
            # Existing position — adjust size
            pos = existing_map[signal.symbol]
            current_value = pos.qty * price

            if pos.side != side:
                # Direction flipped — close old, open new
                try:
                    manager.close_position(pos.symbol, price, reason="rebalance_flip")
                    closed.append({"symbol": pos.symbol, "side": pos.side,
                                   "qty": pos.qty, "exit_price": price})
                except ValueError as exc:
                    errors.append({"symbol": signal.symbol, "error": str(exc)})
                    continue
                # Reload and fall through to open new
                portfolio = manager.load()
                total_value = portfolio.total_value
                target_value = total_value * (pct / 100)
                del existing_map[signal.symbol]
            else:
                diff_value = target_value - current_value
                if diff_value > price * (0.5 if is_crypto else 1):
                    # Need to add — buy more
                    qty_add = diff_value / price
                    qty_add = round(qty_add, 6) if is_crypto else int(qty_add)
                    if qty_add > 0:
                        try:
                            manager.add_to_position(signal.symbol, qty_add, price)
                            added.append({"symbol": signal.symbol, "qty_added": qty_add,
                                          "price": price, "cost": round(qty_add * price, 2)})
                        except ValueError as exc:
                            errors.append({"symbol": signal.symbol, "error": str(exc)})
                    continue
                elif diff_value < -price * (0.5 if is_crypto else 1):
                    # Need to reduce — sell some
                    qty_reduce = abs(diff_value) / price
                    qty_reduce = round(qty_reduce, 6) if is_crypto else int(qty_reduce)
                    if qty_reduce > 0:
                        try:
                            manager.reduce_position(signal.symbol, qty_reduce, price, reason="rebalance")
                            reduced.append({"symbol": signal.symbol, "qty_reduced": qty_reduce,
                                            "price": price, "proceeds": round(qty_reduce * price, 2)})
                        except ValueError as exc:
                            errors.append({"symbol": signal.symbol, "error": str(exc)})
                    continue
                else:
                    # Close enough — no action needed
                    continue

        # New position — open fresh
        if signal.symbol not in existing_map:
            raw_qty = target_value / price
            # Crypto: 6 decimal places; stocks/futures: 2 decimals (fractional shares for paper)
            qty = round(raw_qty, 6) if is_crypto else round(raw_qty, 2)
            if qty <= 0:
                continue

            # Recalculate stop/target based on actual entry price (not signal's entry).
            # Preserve the percentage distances from the signal.
            actual_stop, actual_target = _rescale_stop_target(
                signal_entry=signal.entry_price,
                signal_stop=signal.stop_loss,
                signal_target=signal.target_price,
                actual_entry=price,
                side=side,
            )

            try:
                pos = manager.open_position(
                    symbol=signal.symbol,
                    side=side,
                    qty=qty,
                    entry_price=price,
                    stop_loss=actual_stop,
                    target_price=actual_target,
                )
                opened.append({
                    "symbol": pos.symbol, "side": side, "qty": pos.qty,
                    "entry_price": price, "signal_entry": signal.entry_price,
                    "allocation_pct": pct, "cost": round(qty * price, 2),
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
        "selection_method": "stratified_rebalance",
        "allocation": allocation,
        "positions_opened": opened,
        "positions_added": added,
        "positions_reduced": reduced,
        "positions_closed": closed,
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
        qty = round(raw_qty, 6) if is_crypto else round(raw_qty, 2)
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
