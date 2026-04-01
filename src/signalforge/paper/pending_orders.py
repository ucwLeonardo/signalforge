"""Pending order queue for paper trading.

When Auto Build runs outside US market hours, stock/futures orders are queued
as pending and execute automatically when the market opens.  Crypto orders
execute immediately (24/7 market).

Storage: ~/.signalforge/accounts/{name}/pending_orders.json
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PendingOrder:
    """A deferred position that will open at next market open."""

    id: str
    symbol: str
    side: str  # "long" or "short"
    qty: float
    signal_entry: float  # price when signal was generated
    signal_stop: float
    signal_target: float
    allocation_pct: float
    reserved_amount: float  # cash escrowed (cost + estimated fee)
    created_at: datetime
    status: str = "pending"  # "pending", "executed", "cancelled"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "signal_entry": self.signal_entry,
            "signal_stop": self.signal_stop,
            "signal_target": self.signal_target,
            "allocation_pct": self.allocation_pct,
            "reserved_amount": round(self.reserved_amount, 2),
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }


def _order_from_dict(d: dict) -> PendingOrder:
    return PendingOrder(
        id=d["id"],
        symbol=d["symbol"],
        side=d["side"],
        qty=d["qty"],
        signal_entry=d["signal_entry"],
        signal_stop=d["signal_stop"],
        signal_target=d["signal_target"],
        allocation_pct=d["allocation_pct"],
        reserved_amount=d["reserved_amount"],
        created_at=datetime.fromisoformat(d["created_at"]),
        status=d.get("status", "pending"),
    )


class PendingOrderManager:
    """Per-account pending order persistence."""

    def __init__(self, account_dir: Path) -> None:
        self._path = account_dir / "pending_orders.json"
        from signalforge.paper.portfolio import account_lock

        self._lock = account_lock(account_dir)

    def load(self) -> list[PendingOrder]:
        with self._lock:
            if not self._path.exists():
                return []
            with open(self._path) as f:
                data = json.load(f)
            return [_order_from_dict(d) for d in data.get("orders", [])]

    def save(self, orders: list[PendingOrder]) -> None:
        """Write orders to JSON. Caller should hold self._lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"orders": [o.to_dict() for o in orders]}
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._path)

    def get_pending(self) -> list[PendingOrder]:
        return [o for o in self.load() if o.status == "pending"]

    def add(self, order: PendingOrder) -> None:
        with self._lock:
            orders = self.load()
            orders.append(order)
            self.save(orders)

    def cancel(self, order_id: str) -> PendingOrder | None:
        """Cancel a pending order by ID. Returns the cancelled order or None."""
        from dataclasses import replace

        with self._lock:
            orders = self.load()
            for i, o in enumerate(orders):
                if o.id == order_id and o.status == "pending":
                    cancelled = replace(o, status="cancelled")
                    orders[i] = cancelled
                    self.save(orders)
                    return cancelled
        return None

    def cancel_all(self) -> list[PendingOrder]:
        """Cancel all pending orders. Returns list of cancelled orders."""
        from dataclasses import replace

        with self._lock:
            orders = self.load()
            cancelled = []
            for i, o in enumerate(orders):
                if o.status == "pending":
                    orders[i] = replace(o, status="cancelled")
                    cancelled.append(orders[i])
            if cancelled:
                self.save(orders)
            return cancelled

    def clear_completed(self) -> None:
        """Remove executed/cancelled orders from file."""
        with self._lock:
            orders = [o for o in self.load() if o.status == "pending"]
            self.save(orders)


def create_pending_order(
    symbol: str,
    side: str,
    qty: float,
    signal_entry: float,
    signal_stop: float,
    signal_target: float,
    allocation_pct: float,
    reserved_amount: float,
) -> PendingOrder:
    """Create a new pending order with a unique ID."""
    return PendingOrder(
        id=uuid.uuid4().hex[:12],
        symbol=symbol,
        side=side,
        qty=qty,
        signal_entry=signal_entry,
        signal_stop=signal_stop,
        signal_target=signal_target,
        allocation_pct=allocation_pct,
        reserved_amount=reserved_amount,
        created_at=datetime.now(),
    )


def _try_execute_order(
    order: PendingOrder,
    idx: int,
    price: float,
    all_orders: list[PendingOrder],
    manager: "PortfolioManager",
    results: list[dict],
) -> bool:
    """Attempt to execute a single pending order. Returns True if state changed.

    Does NOT call release_cash — the caller reconciles reserved_cash in
    a single pass after all orders are processed, which avoids double-release
    bugs when reserved_cash is a single aggregate (not per-order).
    """
    from dataclasses import replace as _replace
    from signalforge.paper.simulator import MARKET_CLOSED, _check_slippage, _rescale_stop_target
    from signalforge.data.models import TradeAction, TradeTarget

    action = TradeAction.SELL if order.side == "short" else TradeAction.BUY
    signal = TradeTarget(
        symbol=order.symbol,
        action=action,
        entry_price=order.signal_entry,
        target_price=order.signal_target,
        stop_loss=order.signal_stop,
        risk_reward_ratio=0,
        confidence=0,
        horizon_days=0,
    )

    slip_reason = _check_slippage(signal, price, order.side)
    if slip_reason and not slip_reason.startswith(MARKET_CLOSED):
        all_orders[idx] = _replace(order, status="cancelled")
        results.append({
            "symbol": order.symbol, "action": "cancelled",
            "reason": slip_reason,
        })
        sys.stderr.write(f"[PendingOrders] Cancelled {order.symbol}: {slip_reason}\n")
        return True

    actual_stop, actual_target = _rescale_stop_target(
        signal_entry=order.signal_entry,
        signal_stop=order.signal_stop,
        signal_target=order.signal_target,
        actual_entry=price,
        side=order.side,
    )

    try:
        pos = manager.open_position(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            entry_price=price,
            stop_loss=actual_stop,
            target_price=actual_target,
            from_reserved=order.reserved_amount,
        )
        all_orders[idx] = _replace(order, status="executed")
        results.append({
            "symbol": pos.symbol, "action": "executed",
            "entry_price": price, "signal_entry": order.signal_entry,
            "qty": pos.qty, "side": order.side,
        })
        sys.stderr.write(
            f"[PendingOrders] Executed {order.symbol} @ ${price:.2f} "
            f"(signal was ${order.signal_entry:.2f})\n"
        )
        return True
    except ValueError as exc:
        err = str(exc)
        if "already have an open position" in err:
            # Position exists — crash recovery or user manually opened it.
            # Mark executed; reconciliation will fix reserved_cash.
            all_orders[idx] = _replace(order, status="executed")
            results.append({
                "symbol": order.symbol, "action": "executed",
                "reason": "position already open",
            })
            sys.stderr.write(
                f"[PendingOrders] {order.symbol}: position already open\n"
            )
            return True
        # Genuine failure — leave pending for retry next cycle
        results.append({"symbol": order.symbol, "action": "error", "reason": err})
        sys.stderr.write(f"[PendingOrders] Failed to open {order.symbol}: {exc}\n")
        return False


def reconcile_reserved_cash(
    manager: "PortfolioManager",
    pending_mgr: PendingOrderManager,
) -> None:
    """Ensure reserved_cash matches the sum of pending order reservations.

    Safe to call at any time (market open or closed, with or without
    pending orders).  Corrects stale reserved_cash left by prior crashes.
    """
    from signalforge.paper.portfolio import account_lock

    with account_lock(manager.path.parent):
        all_orders = pending_mgr.load()
        still_pending = sum(
            o.reserved_amount for o in all_orders if o.status == "pending"
        )
        portfolio = manager.load()
        if abs(portfolio.reserved_cash - still_pending) > 0.001:
            portfolio.reserved_cash = round(still_pending, 2)
            manager._save(portfolio)


def execute_pending_orders(
    manager: "PortfolioManager",
    pending_mgr: PendingOrderManager,
) -> list[dict]:
    """Execute pending orders if US market is now open.

    For each pending order:
      1. Fetch live price
      2. Check if price is beyond signal stop/target (cancel if so)
      3. Rescale stop/target to actual entry price
      4. Open position, release reserved cash

    The entire load→mutate→save cycle is held under the account lock
    to prevent races with cancel/rebuild from the HTTP thread.

    Returns list of action dicts (executed/cancelled with reasons).
    """
    from signalforge.paper.prices import _is_us_market_hours

    if not _is_us_market_hours():
        return []

    # Fetch prices outside the lock (network I/O, may be slow).
    # We'll re-check order state inside the lock.
    preliminary_orders = pending_mgr.load()
    preliminary_pending = [o for o in preliminary_orders if o.status == "pending"]
    if not preliminary_pending:
        return []

    from signalforge.paper.simulator import _fetch_live_prices_for_build

    symbols = [o.symbol for o in preliminary_pending]
    live_prices = _fetch_live_prices_for_build(symbols)

    from dataclasses import replace as _replace
    from signalforge.paper.portfolio import account_lock

    lock = account_lock(manager.path.parent)
    results = []
    max_age_hours = 48

    # Hold the account lock for the entire mutate→save sequence.
    # RLock allows inner calls (open_position) to re-enter.
    with lock:
        all_orders = pending_mgr.load()
        pending = [o for o in all_orders if o.status == "pending"]
        if not pending:
            return []
        order_map = {o.id: i for i, o in enumerate(all_orders)}

        for order in pending:
            idx = order_map[order.id]
            dirty = False

            age_hours = (datetime.now() - order.created_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                all_orders[idx] = _replace(order, status="cancelled")
                results.append({
                    "symbol": order.symbol, "action": "expired",
                    "reason": f"pending order expired after {age_hours:.0f}h",
                })
                sys.stderr.write(
                    f"[PendingOrders] Expired {order.symbol} (age {age_hours:.0f}h)\n"
                )
                dirty = True
            elif order.signal_entry <= 0:
                all_orders[idx] = _replace(order, status="cancelled")
                results.append({"symbol": order.symbol, "action": "cancelled",
                                "reason": "invalid signal entry price"})
                dirty = True
            else:
                price = live_prices.get(order.symbol, 0.0)
                if price > 0:
                    dirty = _try_execute_order(
                        order, idx, price, all_orders, manager, results,
                    )

            # Persist after each state change so a crash never loses progress
            if dirty:
                pending_mgr.save(all_orders)

    # Reconcile after processing (outside the main lock scope to keep
    # the critical section short; reconcile_reserved_cash acquires its own).
    reconcile_reserved_cash(manager, pending_mgr)

    return results
