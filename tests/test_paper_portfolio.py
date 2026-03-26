"""Tests for PortfolioManager."""

import pytest
from pathlib import Path

from signalforge.paper.portfolio import PortfolioManager


@pytest.fixture
def portfolio_path(tmp_path: Path) -> Path:
    return tmp_path / "portfolio.json"


def test_init_creates_portfolio(portfolio_path: Path):
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    p = mgr.load()
    assert p.cash == 5000.0
    assert p.initial_balance == 5000.0
    assert p.positions == []
    assert p.trades == []


def test_open_position_deducts_cash(portfolio_path: Path):
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="AAPL", side="long", qty=5, entry_price=200.0,
        stop_loss=190.0, target_price=220.0,
    )
    p = mgr.load()
    assert p.cash == pytest.approx(4000.0)
    assert len(p.positions) == 1
    assert p.positions[0].symbol == "AAPL"


def test_close_position_adds_cash(portfolio_path: Path):
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="AAPL", side="long", qty=5, entry_price=200.0,
        stop_loss=190.0, target_price=220.0,
    )
    mgr.close_position("AAPL", exit_price=210.0, reason="manual")
    p = mgr.load()
    assert p.cash == pytest.approx(5050.0)
    assert len(p.positions) == 0
    assert len(p.trades) == 1
    assert p.trades[0].pnl == pytest.approx(50.0)


def test_close_short_position(portfolio_path: Path):
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="NVDA", side="short", qty=3, entry_price=100.0,
        stop_loss=110.0, target_price=85.0,
    )
    mgr.close_position("NVDA", exit_price=90.0, reason="target_hit")
    p = mgr.load()
    assert len(p.trades) == 1
    assert p.trades[0].pnl == pytest.approx(30.0)


def test_cannot_open_duplicate_position(portfolio_path: Path):
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


def test_cannot_close_nonexistent_position(portfolio_path: Path):
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    with pytest.raises(ValueError, match="No open position"):
        mgr.close_position("AAPL", exit_price=210.0, reason="manual")


def test_insufficient_cash(portfolio_path: Path):
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=100.0)
    with pytest.raises(ValueError, match="Insufficient cash"):
        mgr.open_position(
            symbol="AAPL", side="long", qty=5, entry_price=200.0,
            stop_loss=190.0, target_price=220.0,
        )


def test_save_load_roundtrip(portfolio_path: Path):
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="AAPL", side="long", qty=5, entry_price=200.0,
        stop_loss=190.0, target_price=220.0,
    )
    mgr2 = PortfolioManager(portfolio_path)
    p = mgr2.load()
    assert p.cash == pytest.approx(4000.0)
    assert len(p.positions) == 1
    assert p.positions[0].symbol == "AAPL"


def test_update_prices(portfolio_path: Path):
    mgr = PortfolioManager(portfolio_path)
    mgr.init(balance=5000.0)
    mgr.open_position(
        symbol="AAPL", side="long", qty=5, entry_price=200.0,
        stop_loss=190.0, target_price=220.0,
    )
    mgr.update_prices({"AAPL": 215.0})
    p = mgr.load()
    assert p.positions[0].current_price == pytest.approx(215.0)
    assert p.positions[0].unrealized_pnl == pytest.approx(75.0)
