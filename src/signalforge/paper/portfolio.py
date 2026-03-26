"""Portfolio manager with JSON persistence and multi-account support."""

from __future__ import annotations

import json
import re
import shutil
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

_ACCOUNTS_DIR = Path.home() / ".signalforge" / "accounts"
_LEGACY_PATH = Path.home() / ".signalforge" / "paper_portfolio.json"
_LEGACY_HISTORY = Path.home() / ".signalforge" / "paper_value_history.json"
_ACCOUNT_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,49}$")


def _validate_account_name(name: str) -> None:
    """Validate account name: alphanumeric, hyphens, underscores, 1-50 chars."""
    if not _ACCOUNT_NAME_RE.match(name):
        raise ValueError(
            f"Invalid account name '{name}': use letters, digits, hyphens, "
            "underscores only (1-50 chars, must start with alphanumeric)"
        )


def _account_dir(name: str) -> Path:
    _validate_account_name(name)
    return _ACCOUNTS_DIR / name


class PortfolioManager:
    """Manage a paper trading portfolio with JSON file persistence."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (_ACCOUNTS_DIR / "default" / "portfolio.json")

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def init(self, balance: float = 5000.0, force: bool = False) -> Portfolio:
        """Create a fresh portfolio. If force=True, overwrite existing."""
        if self._path.exists() and not force:
            raise FileExistsError(
                f"Portfolio already exists at {self._path}. Use force=True to overwrite."
            )
        portfolio = Portfolio(
            cash=balance,
            positions=[],
            trades=[],
            initial_balance=balance,
            created_at=datetime.now(),
        )
        self._save(portfolio)
        # Also clear value history
        history_path = self._path.parent / "paper_value_history.json"
        if history_path.exists():
            history_path.unlink()
        return portfolio

    def delete(self) -> None:
        """Delete portfolio and history files."""
        if self._path.exists():
            self._path.unlink()
        history_path = self._path.parent / "paper_value_history.json"
        if history_path.exists():
            history_path.unlink()

    def reset(self, balance: float | None = None) -> Portfolio:
        """Reset portfolio to initial state. Uses stored initial_balance if no balance given."""
        if balance is None and self._path.exists():
            old = self.load()
            balance = old.initial_balance
        return self.init(balance=balance or 5000.0, force=True)

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


class AccountManager:
    """Manage multiple named paper trading accounts.

    Each account is stored in ~/.signalforge/accounts/{name}/ with:
      - portfolio.json (positions, trades, cash)
      - paper_value_history.json (value snapshots)
    """

    def __init__(self) -> None:
        self._migrate_legacy()

    def _migrate_legacy(self) -> None:
        """Move legacy single-file portfolio into accounts/default/."""
        if not _LEGACY_PATH.exists():
            return
        default_dir = _ACCOUNTS_DIR / "default"
        if (default_dir / "portfolio.json").exists():
            return  # already migrated
        default_dir.mkdir(parents=True, exist_ok=True)
        _LEGACY_PATH.rename(default_dir / "portfolio.json")
        if _LEGACY_HISTORY.exists():
            _LEGACY_HISTORY.rename(default_dir / "paper_value_history.json")

    def list_accounts(self) -> list[str]:
        """Return sorted list of account names."""
        if not _ACCOUNTS_DIR.exists():
            return []
        return sorted(
            d.name
            for d in _ACCOUNTS_DIR.iterdir()
            if d.is_dir() and (d / "portfolio.json").exists()
        )

    def account_exists(self, name: str) -> bool:
        _validate_account_name(name)
        return (_account_dir(name) / "portfolio.json").exists()

    def get_manager(self, name: str) -> PortfolioManager:
        """Return a PortfolioManager for the named account."""
        _validate_account_name(name)
        return PortfolioManager(path=_account_dir(name) / "portfolio.json")

    def create_account(
        self, name: str, balance: float = 5000.0, force: bool = False
    ) -> PortfolioManager:
        """Create (or force-recreate) a named account."""
        mgr = self.get_manager(name)
        mgr.init(balance=balance, force=force)
        return mgr

    def delete_account(self, name: str) -> None:
        """Delete an account and all its data."""
        _validate_account_name(name)
        acct_dir = _account_dir(name)
        if not acct_dir.exists():
            raise ValueError(f"Account '{name}' does not exist")
        shutil.rmtree(acct_dir)

    def reset_account(self, name: str, balance: float | None = None) -> Portfolio:
        """Reset an account to initial state."""
        mgr = self.get_manager(name)
        return mgr.reset(balance=balance)

    def get_account_summary(self, name: str) -> dict:
        """Return a summary dict for one account."""
        mgr = self.get_manager(name)
        if not mgr.exists():
            raise ValueError(f"Account '{name}' does not exist")
        p = mgr.load()
        return {
            "name": name,
            "cash": round(p.cash, 2),
            "initial_balance": round(p.initial_balance, 2),
            "total_value": round(p.total_value, 2),
            "total_pnl": round(p.total_pnl, 2),
            "total_pnl_pct": round(p.total_pnl_pct, 2),
            "positions_count": len(p.positions),
            "trades_count": len(p.trades),
            "created_at": p.created_at.isoformat(),
        }
