"""Tests for TradeExecutor."""

import pytest
from pathlib import Path

from signalforge.data.models import TradeAction, TradeTarget
from signalforge.paper.executor import compute_position_size, execute_signals
from signalforge.paper.portfolio import PortfolioManager


@pytest.fixture
def manager(tmp_path: Path) -> PortfolioManager:
    mgr = PortfolioManager(tmp_path / "portfolio.json")
    mgr.init(balance=5000.0)
    return mgr


def _make_target(
    symbol: str = "AAPL",
    action: TradeAction = TradeAction.BUY,
    entry: float = 200.0,
    target: float = 220.0,
    stop: float = 190.0,
    confidence: float = 0.75,
) -> TradeTarget:
    rr = abs(target - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0
    return TradeTarget(
        symbol=symbol, action=action, entry_price=entry,
        target_price=target, stop_loss=stop,
        risk_reward_ratio=rr, confidence=confidence,
        horizon_days=5, rationale="test signal",
    )


def test_position_size_20pct():
    qty = compute_position_size(5000.0, 200.0, 0.20)
    assert qty == 5  # 5000*0.20/200 = 5


def test_position_size_fractional():
    qty = compute_position_size(5000.0, 50000.0, 0.20)
    assert qty == pytest.approx(0.02)  # 1000/50000


def test_execute_buy_signal(manager: PortfolioManager):
    targets = [_make_target("AAPL", TradeAction.BUY, 200.0, 220.0, 190.0)]
    opened = execute_signals(targets, manager)
    assert len(opened) == 1
    assert opened[0].symbol == "AAPL"
    assert opened[0].side == "long"
    p = manager.load()
    assert len(p.positions) == 1
    assert p.cash < 5000.0


def test_execute_sell_signal(manager: PortfolioManager):
    targets = [_make_target("TSLA", TradeAction.SELL, 280.0, 255.0, 295.0)]
    opened = execute_signals(targets, manager)
    assert len(opened) == 1
    assert opened[0].side == "short"


def test_skip_hold_signal(manager: PortfolioManager):
    targets = [_make_target("SPY", TradeAction.HOLD, 500.0, 500.0, 500.0)]
    opened = execute_signals(targets, manager)
    assert len(opened) == 0


def test_skip_low_confidence(manager: PortfolioManager):
    targets = [_make_target("AAPL", confidence=0.10)]
    opened = execute_signals(targets, manager)
    assert len(opened) == 0


def test_skip_duplicate_symbol(manager: PortfolioManager):
    targets = [
        _make_target("AAPL", confidence=0.80),
        _make_target("AAPL", confidence=0.70),
    ]
    opened = execute_signals(targets, manager)
    assert len(opened) == 1


def test_multiple_signals(manager: PortfolioManager):
    targets = [
        _make_target("AAPL", entry=200.0, confidence=0.82),
        _make_target("MSFT", entry=425.0, confidence=0.71),
        _make_target("NVDA", entry=142.0, confidence=0.76),
    ]
    opened = execute_signals(targets, manager)
    assert len(opened) == 3
    p = manager.load()
    assert len(p.positions) == 3
    assert p.cash < 5000.0
