"""Portfolio manager with JSON persistence and multi-account support."""

from __future__ import annotations

import json
import re
import shutil
import threading
from collections import defaultdict
from datetime import datetime
from dataclasses import replace
from pathlib import Path

# Per-account lock — keyed by resolved account directory path.
# Serialises all load→mutate→save sequences on portfolio.json and
# pending_orders.json for the same account, preventing races between
# the background price updater thread and HTTP request threads.
_account_locks: dict[Path, threading.RLock] = defaultdict(threading.RLock)


def account_lock(account_dir: Path) -> threading.RLock:
    """Return the per-account reentrant lock for the given account directory."""
    return _account_locks[account_dir.resolve()]

from signalforge.paper.models import (
    Portfolio,
    Position,
    Trade,
    position_from_dict,
    trade_from_dict,
)


def _fee_for_symbol(symbol: str, qty: float, price: float) -> float:
    """Calculate transaction fee using asset-type-aware parameters."""
    from signalforge.config import get_trading_params

    if "/" in symbol:
        asset_type = "crypto"
    elif symbol.endswith("=F"):
        asset_type = "futures"
    else:
        asset_type = "stock"
    return get_trading_params(asset_type).calculate_fee(qty, price)


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
        self._lock = account_lock(self._path.parent)

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def init(
        self,
        balance: float = 5000.0,
        force: bool = False,
        asset_categories: list[str] | None = None,
    ) -> Portfolio:
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
            asset_categories=asset_categories if asset_categories is not None else ["us_stocks", "crypto"],
        )
        self._save(portfolio)
        # Also clear value history and pending orders
        history_path = self._path.parent / "paper_value_history.json"
        if history_path.exists():
            history_path.unlink()
        pending_path = self._path.parent / "pending_orders.json"
        if pending_path.exists():
            pending_path.unlink()
        return portfolio

    def delete(self) -> None:
        """Delete portfolio and history files."""
        if self._path.exists():
            self._path.unlink()
        history_path = self._path.parent / "paper_value_history.json"
        if history_path.exists():
            history_path.unlink()

    def deposit(self, amount: float) -> Portfolio:
        """Add funds to the account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        with self._lock:
            portfolio = self.load()
            portfolio = Portfolio(
                cash=portfolio.cash + amount,
                positions=portfolio.positions,
                trades=portfolio.trades,
                initial_balance=portfolio.initial_balance + amount,
                created_at=portfolio.created_at,
                asset_categories=portfolio.asset_categories,
                reserved_cash=portfolio.reserved_cash,
            )
            self._save(portfolio)
            return portfolio

    def reset(self, balance: float | None = None) -> Portfolio:
        """Reset portfolio to initial state. Uses stored initial_balance if no balance given."""
        old_categories: list[str] | None = None
        if self._path.exists():
            old = self.load()
            if balance is None:
                balance = old.initial_balance
            old_categories = old.asset_categories
        return self.init(balance=balance or 5000.0, force=True, asset_categories=old_categories)

    def load(self) -> Portfolio:
        """Load portfolio from JSON file. Re-initializes if file is corrupt.

        Acquires the account RLock to prevent torn reads from a concurrent
        ``_save`` in another thread.
        """
        with self._lock:
            try:
                with open(self._path) as f:
                    data = json.load(f)
                return Portfolio(
                    cash=data["cash"],
                    positions=[position_from_dict(p) for p in data["positions"]],
                    trades=[trade_from_dict(t) for t in data["trades"]],
                    initial_balance=data["initial_balance"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    asset_categories=data.get("asset_categories", ["us_stocks", "crypto"]),
                    reserved_cash=data.get("reserved_cash", 0.0),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                import sys
                sys.stderr.write(
                    f"[Portfolio] Corrupt portfolio at {self._path}: {exc}. Re-initializing.\n"
                )
                return self.init(balance=5000.0, force=True)

    def _save(self, portfolio: Portfolio) -> None:
        """Write portfolio to JSON file. Caller should hold self._lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cash": portfolio.cash,
            "positions": [p.to_dict() for p in portfolio.positions],
            "trades": [t.to_dict() for t in portfolio.trades],
            "initial_balance": portfolio.initial_balance,
            "created_at": portfolio.created_at.isoformat(),
            "asset_categories": portfolio.asset_categories,
            "reserved_cash": portfolio.reserved_cash,
        }
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._path)  # atomic on POSIX

    def open_position(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        from_reserved: float = 0.0,
    ) -> Position:
        """Open a new position. Deducts cost + fee from cash.

        If *from_reserved* > 0 the given amount is moved out of
        ``reserved_cash`` in the same atomic save that opens the
        position, so a crash between "release" and "open" is impossible.
        """
        with self._lock:
            portfolio = self.load()
            for p in portfolio.positions:
                if p.symbol == symbol:
                    raise ValueError(f"You already have an open position in {symbol}")
            cost = qty * entry_price
            fee = _fee_for_symbol(symbol, qty, entry_price)
            total_cost = cost + fee
            # When executing a pending order the reserved amount is freed
            # into available_cash first (within this single save).
            effective_available = portfolio.available_cash + min(from_reserved, portfolio.reserved_cash)
            if total_cost > effective_available:
                raise ValueError(
                    f"Insufficient cash: need ${total_cost:.2f} (incl. ${fee:.2f} fee), "
                    f"have ${effective_available:.2f} available"
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
                open_fee=fee,
            )
            portfolio.cash -= total_cost
            if from_reserved > 0:
                portfolio.reserved_cash = max(0.0, portfolio.reserved_cash - from_reserved)
            portfolio.positions.append(position)
            self._save(portfolio)
            return position

    def add_to_position(
        self,
        symbol: str,
        qty_add: float,
        price: float,
    ) -> Position:
        """Add to an existing position. Deducts cost + fee from cash."""
        with self._lock:
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

            cost = qty_add * price
            fee = _fee_for_symbol(symbol, qty_add, price)
            if cost + fee > portfolio.available_cash:
                raise ValueError(
                    f"Insufficient cash: need ${cost + fee:.2f}, have ${portfolio.available_cash:.2f}"
                )

            old_value = pos.qty * pos.entry_price
            new_value = qty_add * price
            new_qty = pos.qty + qty_add
            avg_entry = (old_value + new_value) / new_qty if new_qty > 0 else price

            updated = replace(pos, qty=new_qty, entry_price=round(avg_entry, 6),
                              current_price=price, open_fee=pos.open_fee + fee)
            portfolio.positions[pos_idx] = updated
            portfolio.cash -= (cost + fee)
            self._save(portfolio)
            return updated

    def reduce_position(
        self,
        symbol: str,
        qty_reduce: float,
        price: float,
        reason: str = "rebalance",
    ) -> Trade | None:
        """Reduce (partial close) an existing position."""
        with self._lock:
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

            if qty_reduce >= pos.qty:
                return self._close_position_locked(portfolio, symbol, price, reason=reason)

            proceeds = qty_reduce * price
            fee = _fee_for_symbol(symbol, qty_reduce, price)
            trade = Trade(
                symbol=pos.symbol,
                side=pos.side,
                qty=qty_reduce,
                entry_price=pos.entry_price,
                exit_price=price,
                opened_at=pos.opened_at,
                closed_at=datetime.now(),
                reason=reason,
            )
            remaining_qty = pos.qty - qty_reduce
            updated = replace(pos, qty=remaining_qty, current_price=price)
            portfolio.positions[pos_idx] = updated
            portfolio.cash += proceeds - fee
            portfolio.trades.append(trade)
            self._save(portfolio)
            return trade

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Trade:
        """Close an open position. Returns the completed Trade."""
        with self._lock:
            portfolio = self.load()
            return self._close_position_locked(portfolio, symbol, exit_price, reason)

    def _close_position_locked(
        self, portfolio: Portfolio, symbol: str, exit_price: float, reason: str,
    ) -> Trade:
        """Close position — caller must hold self._lock and provide loaded portfolio."""
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
        proceeds = pos.qty * exit_price
        fee = _fee_for_symbol(symbol, pos.qty, exit_price)
        portfolio.cash += proceeds - fee
        portfolio.positions.pop(pos_idx)
        portfolio.trades.append(trade)
        self._save(portfolio)
        return trade

    def reserve_cash(self, amount: float) -> None:
        """Reserve cash for a pending order (escrow)."""
        with self._lock:
            portfolio = self.load()
            if amount > portfolio.available_cash:
                raise ValueError(
                    f"Cannot reserve ${amount:.2f}: only ${portfolio.available_cash:.2f} available"
                )
            portfolio.reserved_cash += amount
            self._save(portfolio)

    def release_cash(self, amount: float) -> None:
        """Release reserved cash (pending order executed or cancelled)."""
        with self._lock:
            portfolio = self.load()
            portfolio.reserved_cash = max(0.0, portfolio.reserved_cash - amount)
            self._save(portfolio)

    def update_prices(self, prices: dict[str, float]) -> Portfolio:
        """Update current prices for open positions."""
        with self._lock:
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
        self,
        name: str,
        balance: float = 5000.0,
        force: bool = False,
        asset_categories: list[str] | None = None,
    ) -> PortfolioManager:
        """Create (or force-recreate) a named account."""
        mgr = self.get_manager(name)
        mgr.init(balance=balance, force=force, asset_categories=asset_categories)
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

    def deposit(self, name: str, amount: float) -> Portfolio:
        """Add funds to an account."""
        mgr = self.get_manager(name)
        if not mgr.exists():
            raise ValueError(f"Account '{name}' does not exist")
        return mgr.deposit(amount)

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
            "asset_categories": p.asset_categories,
        }
