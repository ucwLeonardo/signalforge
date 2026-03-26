"""Portfolio manager with JSON persistence."""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from signalforge.paper.models import (
    Portfolio,
    Position,
    Trade,
    position_from_dict,
    trade_from_dict,
)

_DEFAULT_PATH = Path.home() / ".signalforge" / "paper_portfolio.json"


class PortfolioManager:
    """Manage a paper trading portfolio with JSON file persistence."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def init(self, balance: float = 5000.0) -> Portfolio:
        """Create a fresh portfolio with the given cash balance."""
        portfolio = Portfolio(
            cash=balance,
            positions=[],
            trades=[],
            initial_balance=balance,
            created_at=datetime.now(),
        )
        self._save(portfolio)
        return portfolio

    def load(self) -> Portfolio:
        """Load portfolio from JSON file."""
        with open(self._path) as f:
            data = json.load(f)
        return Portfolio(
            cash=data["cash"],
            positions=[position_from_dict(p) for p in data["positions"]],
            trades=[trade_from_dict(t) for t in data["trades"]],
            initial_balance=data["initial_balance"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def _save(self, portfolio: Portfolio) -> None:
        """Write portfolio to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cash": portfolio.cash,
            "positions": [p.to_dict() for p in portfolio.positions],
            "trades": [t.to_dict() for t in portfolio.trades],
            "initial_balance": portfolio.initial_balance,
            "created_at": portfolio.created_at.isoformat(),
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def open_position(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        stop_loss: float,
        target_price: float,
    ) -> Position:
        """Open a new position. Deducts cost from cash."""
        portfolio = self.load()
        # Check for duplicate
        for p in portfolio.positions:
            if p.symbol == symbol:
                raise ValueError(f"You already have an open position in {symbol}")
        # Check cash
        cost = qty * entry_price
        if cost > portfolio.cash:
            raise ValueError(
                f"Insufficient cash: need ${cost:.2f}, have ${portfolio.cash:.2f}"
            )
        position = Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            opened_at=datetime.now(),
        )
        portfolio.cash -= cost
        portfolio.positions.append(position)
        self._save(portfolio)
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Trade:
        """Close an open position. Returns the completed Trade."""
        portfolio = self.load()
        pos = None
        pos_idx = -1
        for i, p in enumerate(portfolio.positions):
            if p.symbol == symbol:
                pos = p
                pos_idx = i
                break
        if pos is None:
            raise ValueError(f"No open position for {symbol}")

        trade = Trade(
            symbol=pos.symbol,
            side=pos.side,
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            opened_at=pos.opened_at,
            closed_at=datetime.now(),
            reason=reason,
        )
        # Add proceeds back to cash
        proceeds = pos.qty * exit_price
        portfolio.cash += proceeds
        portfolio.positions.pop(pos_idx)
        portfolio.trades.append(trade)
        self._save(portfolio)
        return trade

    def update_prices(self, prices: dict[str, float]) -> Portfolio:
        """Update current prices for open positions."""
        portfolio = self.load()
        updated_positions = []
        for pos in portfolio.positions:
            new_price = prices.get(pos.symbol, pos.current_price)
            updated_positions.append(replace(pos, current_price=new_price))
        portfolio.positions = updated_positions
        self._save(portfolio)
        return portfolio
