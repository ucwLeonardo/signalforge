# Paper Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add paper trading to SignalForge so users can simulate trades with a virtual $5,000 portfolio based on signal recommendations.

**Architecture:** New `paper/` subpackage with frozen dataclasses for positions/trades, a PortfolioManager for state persistence (JSON file), a TradeExecutor for position sizing and trade logic, and a mock signal simulator for offline demo. CLI subcommands added via Typer sub-app.

**Tech Stack:** Python 3.10+, Typer, Rich, JSON persistence, existing SignalForge models (TradeTarget, TradeAction)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/signalforge/paper/__init__.py` | Package exports |
| `src/signalforge/paper/models.py` | Position, Trade, Portfolio dataclasses |
| `src/signalforge/paper/portfolio.py` | PortfolioManager: load/save JSON, open/close positions, P&L |
| `src/signalforge/paper/executor.py` | TradeExecutor: position sizing (max 20%), execute from TradeTargets |
| `src/signalforge/paper/simulator.py` | Generate mock TradeTargets with realistic prices for offline demo |
| `src/signalforge/cli.py` | Add `paper` sub-app with init/status/auto/close/history commands |
| `tests/test_paper_models.py` | Unit tests for paper models |
| `tests/test_paper_portfolio.py` | Unit tests for PortfolioManager |
| `tests/test_paper_executor.py` | Unit tests for TradeExecutor |

---

### Task 1: Paper Trading Models

**Files:**
- Create: `src/signalforge/paper/__init__.py`
- Create: `src/signalforge/paper/models.py`
- Create: `tests/test_paper_models.py`

- [ ] **Step 1: Write failing tests for Position, Trade, Portfolio**

```python
# tests/test_paper_models.py
"""Tests for paper trading models."""
import pytest
from datetime import datetime


def test_position_unrealized_pnl_long():
    from signalforge.paper.models import Position
    pos = Position(
        symbol="AAPL", side="long", qty=10, entry_price=150.0,
        current_price=160.0, stop_loss=140.0, target_price=170.0,
        opened_at=datetime(2026, 3, 26),
    )
    assert pos.unrealized_pnl == pytest.approx(100.0)  # (160-150)*10


def test_position_unrealized_pnl_short():
    from signalforge.paper.models import Position
    pos = Position(
        symbol="NVDA", side="short", qty=5, entry_price=200.0,
        current_price=190.0, stop_loss=210.0, target_price=180.0,
        opened_at=datetime(2026, 3, 26),
    )
    assert pos.unrealized_pnl == pytest.approx(50.0)  # (200-190)*5


def test_position_market_value():
    from signalforge.paper.models import Position
    pos = Position(
        symbol="AAPL", side="long", qty=10, entry_price=150.0,
        current_price=160.0, stop_loss=140.0, target_price=170.0,
        opened_at=datetime(2026, 3, 26),
    )
    assert pos.market_value == pytest.approx(1600.0)


def test_trade_pnl():
    from signalforge.paper.models import Trade
    trade = Trade(
        symbol="AAPL", side="long", qty=10,
        entry_price=150.0, exit_price=165.0,
        opened_at=datetime(2026, 3, 25), closed_at=datetime(2026, 3, 26),
        reason="target_hit",
    )
    assert trade.pnl == pytest.approx(150.0)


def test_portfolio_total_value():
    from signalforge.paper.models import Portfolio, Position
    pos = Position(
        symbol="AAPL", side="long", qty=10, entry_price=150.0,
        current_price=160.0, stop_loss=140.0, target_price=170.0,
        opened_at=datetime(2026, 3, 26),
    )
    portfolio = Portfolio(
        cash=3400.0,
        positions=[pos],
        trades=[],
        initial_balance=5000.0,
    )
    assert portfolio.total_value == pytest.approx(5000.0)  # 3400 + 1600


def test_portfolio_total_pnl():
    from signalforge.paper.models import Portfolio
    portfolio = Portfolio(
        cash=5200.0, positions=[], trades=[],
        initial_balance=5000.0,
    )
    assert portfolio.total_pnl == pytest.approx(200.0)


def test_position_to_dict_roundtrip():
    from signalforge.paper.models import Position, position_from_dict
    pos = Position(
        symbol="AAPL", side="long", qty=10, entry_price=150.0,
        current_price=160.0, stop_loss=140.0, target_price=170.0,
        opened_at=datetime(2026, 3, 26),
    )
    d = pos.to_dict()
    restored = position_from_dict(d)
    assert restored.symbol == pos.symbol
    assert restored.qty == pos.qty
    assert restored.entry_price == pos.entry_price
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_paper_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'signalforge.paper'`

- [ ] **Step 3: Implement models**

```python
# src/signalforge/paper/__init__.py
"""Paper trading module for SignalForge."""

# src/signalforge/paper/models.py
"""Paper trading domain models."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Position:
    symbol: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    current_price: float
    stop_loss: float
    target_price: float
    opened_at: datetime

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.qty
        return (self.entry_price - self.current_price) * self.qty

    @property
    def market_value(self) -> float:
        return self.current_price * self.qty

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.qty

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "side": self.side, "qty": self.qty,
            "entry_price": self.entry_price, "current_price": self.current_price,
            "stop_loss": self.stop_loss, "target_price": self.target_price,
            "opened_at": self.opened_at.isoformat(),
        }


@dataclass(frozen=True)
class Trade:
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

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "side": self.side, "qty": self.qty,
            "entry_price": self.entry_price, "exit_price": self.exit_price,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "reason": self.reason,
        }


@dataclass
class Portfolio:
    cash: float
    positions: list[Position] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    initial_balance: float = 5000.0
    created_at: datetime = field(default_factory=datetime.now)

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


def position_from_dict(d: dict) -> Position:
    return Position(
        symbol=d["symbol"], side=d["side"], qty=d["qty"],
        entry_price=d["entry_price"], current_price=d["current_price"],
        stop_loss=d["stop_loss"], target_price=d["target_price"],
        opened_at=datetime.fromisoformat(d["opened_at"]),
    )


def trade_from_dict(d: dict) -> Trade:
    return Trade(
        symbol=d["symbol"], side=d["side"], qty=d["qty"],
        entry_price=d["entry_price"], exit_price=d["exit_price"],
        opened_at=datetime.fromisoformat(d["opened_at"]),
        closed_at=datetime.fromisoformat(d["closed_at"]),
        reason=d["reason"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_paper_models.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/signalforge/paper/ tests/test_paper_models.py
git commit -m "feat: add paper trading domain models (Position, Trade, Portfolio)"
```

---

### Task 2: Portfolio Manager (JSON Persistence)

**Files:**
- Create: `src/signalforge/paper/portfolio.py`
- Create: `tests/test_paper_portfolio.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_paper_portfolio.py
"""Tests for PortfolioManager."""
import pytest
from datetime import datetime
from pathlib import Path


@pytest.fixture
def portfolio_path(tmp_path):
    return tmp_path / "portfolio.json"


def test_init_creates_portfolio(portfolio_path):
    from signalforge.paper.portfolio import PortfolioManager
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    p = mgr.load()
    assert p.cash == 5000.0
    assert p.initial_balance == 5000.0
    assert p.positions == []
    assert p.trades == []


def test_open_position_deducts_cash(portfolio_path):
    from signalforge.paper.portfolio import PortfolioManager
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="AAPL", side="long", qty=5, entry_price=200.0,
        stop_loss=190.0, target_price=220.0,
    )
    p = mgr.load()
    assert p.cash == pytest.approx(4000.0)  # 5000 - 5*200
    assert len(p.positions) == 1
    assert p.positions[0].symbol == "AAPL"


def test_close_position_adds_cash(portfolio_path):
    from signalforge.paper.portfolio import PortfolioManager
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="AAPL", side="long", qty=5, entry_price=200.0,
        stop_loss=190.0, target_price=220.0,
    )
    mgr.close_position("AAPL", exit_price=210.0, reason="manual")
    p = mgr.load()
    assert p.cash == pytest.approx(5050.0)  # 4000 + 5*210
    assert len(p.positions) == 0
    assert len(p.trades) == 1
    assert p.trades[0].pnl == pytest.approx(50.0)


def test_close_short_position(portfolio_path):
    from signalforge.paper.portfolio import PortfolioManager
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="NVDA", side="short", qty=3, entry_price=100.0,
        stop_loss=110.0, target_price=85.0,
    )
    mgr.close_position("NVDA", exit_price=90.0, reason="target_hit")
    p = mgr.load()
    # Short: cash starts 5000, open deducts 300 (margin), close adds 300 + pnl(30)
    assert len(p.trades) == 1
    assert p.trades[0].pnl == pytest.approx(30.0)  # (100-90)*3


def test_cannot_open_duplicate_position(portfolio_path):
    from signalforge.paper.portfolio import PortfolioManager
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="AAPL", side="long", qty=5, entry_price=200.0,
        stop_loss=190.0, target_price=220.0,
    )
    with pytest.raises(ValueError, match="already have"):
        mgr.open_position(
            symbol="AAPL", side="long", qty=3, entry_price=205.0,
            stop_loss=195.0, target_price=225.0,
        )


def test_cannot_close_nonexistent_position(portfolio_path):
    from signalforge.paper.portfolio import PortfolioManager
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    with pytest.raises(ValueError, match="No open position"):
        mgr.close_position("AAPL", exit_price=210.0, reason="manual")


def test_insufficient_cash(portfolio_path):
    from signalforge.paper.portfolio import PortfolioManager
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=100.0)
    with pytest.raises(ValueError, match="Insufficient cash"):
        mgr.open_position(
            symbol="AAPL", side="long", qty=5, entry_price=200.0,
            stop_loss=190.0, target_price=220.0,
        )


def test_save_load_roundtrip(portfolio_path):
    from signalforge.paper.portfolio import PortfolioManager
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="AAPL", side="long", qty=5, entry_price=200.0,
        stop_loss=190.0, target_price=220.0,
    )
    # Reload from disk
    mgr2 = PortfolioManager(portfolio_path)
    p = mgr2.load()
    assert p.cash == pytest.approx(4000.0)
    assert len(p.positions) == 1
    assert p.positions[0].symbol == "AAPL"


def test_update_prices(portfolio_path):
    from signalforge.paper.portfolio import PortfolioManager
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="AAPL", side="long", qty=5, entry_price=200.0,
        stop_loss=190.0, target_price=220.0,
    )
    mgr.update_prices({"AAPL": 215.0})
    p = mgr.load()
    assert p.positions[0].current_price == pytest.approx(215.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_paper_portfolio.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PortfolioManager**

See implementation in `src/signalforge/paper/portfolio.py`. Key methods:
- `init(balance)` — create fresh portfolio JSON
- `load()` — read JSON, return Portfolio
- `save(portfolio)` — write Portfolio to JSON
- `open_position(symbol, side, qty, entry_price, stop_loss, target_price)` — validate cash, add position
- `close_position(symbol, exit_price, reason)` — remove position, create Trade, add proceeds to cash
- `update_prices(prices: dict[str, float])` — update current_price on open positions

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_paper_portfolio.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/signalforge/paper/portfolio.py tests/test_paper_portfolio.py
git commit -m "feat: add PortfolioManager with JSON persistence"
```

---

### Task 3: Trade Executor (Position Sizing + Signal Execution)

**Files:**
- Create: `src/signalforge/paper/executor.py`
- Create: `tests/test_paper_executor.py`

- [ ] **Step 1: Write failing tests**

Tests for: max 20% position sizing, converting TradeTarget to position, skip HOLD signals, skip symbols already held.

- [ ] **Step 2-5: Implement, test, commit**

---

### Task 4: Mock Signal Simulator

**Files:**
- Create: `src/signalforge/paper/simulator.py`

Generates realistic TradeTarget objects with plausible March 2026 prices when network is unavailable.

---

### Task 5: CLI Commands

**Files:**
- Modify: `src/signalforge/cli.py`

Add Typer sub-app `paper` with commands: `init`, `status`, `auto`, `close`, `history`.

---

### Task 6: Integration — Run Auto-Trade Demo

Execute `signalforge paper init` → `signalforge paper auto` → `signalforge paper status` to demonstrate the full workflow with $5,000.
