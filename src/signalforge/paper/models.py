"""Paper trading domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Position:
    """An open paper trading position."""

    symbol: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    current_price: float
    stop_loss: float
    target_price: float
    opened_at: datetime
    open_fee: float = 0.0  # actual fee paid when opening

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.qty
        return (self.entry_price - self.current_price) * self.qty

    @property
    def market_value(self) -> float:
        if self.side == "long":
            return self.current_price * self.qty
        # Short: collateral (cost basis) + unrealized P&L
        return self.entry_price * self.qty + self.unrealized_pnl

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.qty

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis * 100

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "opened_at": self.opened_at.isoformat(),
            "open_fee": round(self.open_fee, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "market_value": round(self.market_value, 2),
            "pnl_pct": round(self.pnl_pct, 2),
        }


@dataclass(frozen=True)
class Trade:
    """A completed (closed) paper trade."""

    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    opened_at: datetime
    closed_at: datetime
    reason: str  # "target_hit", "stop_hit", "manual"

    @property
    def pnl(self) -> float:
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.qty
        return (self.entry_price - self.exit_price) * self.qty

    @property
    def pnl_pct(self) -> float:
        cost = self.entry_price * self.qty
        if cost == 0:
            return 0.0
        return self.pnl / cost * 100

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "reason": self.reason,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
        }


@dataclass
class Portfolio:
    """Virtual paper trading portfolio."""

    cash: float
    positions: list[Position] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    initial_balance: float = 5000.0
    created_at: datetime = field(default_factory=datetime.now)
    asset_categories: list[str] = field(default_factory=lambda: ["us_stocks", "crypto"])

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions)

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_balance

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_balance == 0:
            return 0.0
        return self.total_pnl / self.initial_balance * 100

    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions)


def position_from_dict(d: dict) -> Position:
    return Position(
        symbol=d["symbol"],
        side=d["side"],
        qty=d["qty"],
        entry_price=d["entry_price"],
        current_price=d["current_price"],
        stop_loss=d["stop_loss"],
        target_price=d["target_price"],
        opened_at=datetime.fromisoformat(d["opened_at"]),
        open_fee=d.get("open_fee", 0.0),
    )


def trade_from_dict(d: dict) -> Trade:
    return Trade(
        symbol=d["symbol"],
        side=d["side"],
        qty=d["qty"],
        entry_price=d["entry_price"],
        exit_price=d["exit_price"],
        opened_at=datetime.fromisoformat(d["opened_at"]),
        closed_at=datetime.fromisoformat(d["closed_at"]),
        reason=d["reason"],
    )
