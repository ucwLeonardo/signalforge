"""Trade executor — converts TradeTargets into paper positions with position sizing."""

from __future__ import annotations

from signalforge.data.models import TradeAction, TradeTarget
from signalforge.paper.models import Position
from signalforge.paper.portfolio import PortfolioManager

# Max percentage of portfolio value per position
MAX_POSITION_PCT = 0.20
# Minimum confidence to accept a signal
MIN_CONFIDENCE = 0.30


def compute_position_size(
    portfolio_value: float,
    entry_price: float,
    max_pct: float = MAX_POSITION_PCT,
) -> float:
    """Calculate the quantity to buy/sell based on position sizing rules.

    Returns the number of shares/units (can be fractional for crypto).
    """
    max_notional = portfolio_value * max_pct
    if entry_price <= 0:
        return 0.0
    qty = max_notional / entry_price
    # For stocks, round down to whole shares
    if qty >= 1.0:
        qty = int(qty)
    return qty


def execute_signals(
    targets: list[TradeTarget],
    manager: PortfolioManager,
    max_pct: float = MAX_POSITION_PCT,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[Position]:
    """Execute a list of trade signals against the portfolio.

    Skips:
    - HOLD signals
    - Signals below min_confidence
    - Symbols with an existing open position
    - Signals that would exceed available cash

    Returns list of newly opened positions.
    """
    opened: list[Position] = []
    portfolio = manager.load()
    held_symbols = {p.symbol for p in portfolio.positions}

    for target in targets:
        # Skip HOLD signals
        if target.action == TradeAction.HOLD:
            continue
        # Skip low confidence
        if target.confidence < min_confidence:
            continue
        # Skip if already holding
        if target.symbol in held_symbols:
            continue

        # Reload portfolio for latest cash
        portfolio = manager.load()
        side = "long" if target.action == TradeAction.BUY else "short"

        qty = compute_position_size(
            portfolio_value=portfolio.total_value,
            entry_price=target.entry_price,
            max_pct=max_pct,
        )
        if qty <= 0:
            continue

        cost = qty * target.entry_price
        if cost > portfolio.cash:
            continue

        try:
            position = manager.open_position(
                symbol=target.symbol,
                side=side,
                qty=qty,
                entry_price=target.entry_price,
                stop_loss=target.stop_loss,
                target_price=target.target_price,
            )
            opened.append(position)
            held_symbols.add(target.symbol)
        except ValueError:
            continue

    return opened
