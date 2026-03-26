"""Tests for paper trading models."""

import pytest
from datetime import datetime

from signalforge.paper.models import (
    Portfolio,
    Position,
    Trade,
    position_from_dict,
    trade_from_dict,
)


def test_position_unrealized_pnl_long():
    pos = Position(
        symbol="AAPL", side="long", qty=10, entry_price=150.0,
        current_price=160.0, stop_loss=140.0, target_price=170.0,
        opened_at=datetime(2026, 3, 26),
    )
    assert pos.unrealized_pnl == pytest.approx(100.0)


def test_position_unrealized_pnl_short():
    pos = Position(
        symbol="NVDA", side="short", qty=5, entry_price=200.0,
        current_price=190.0, stop_loss=210.0, target_price=180.0,
        opened_at=datetime(2026, 3, 26),
    )
    assert pos.unrealized_pnl == pytest.approx(50.0)


def test_position_market_value():
    pos = Position(
        symbol="AAPL", side="long", qty=10, entry_price=150.0,
        current_price=160.0, stop_loss=140.0, target_price=170.0,
        opened_at=datetime(2026, 3, 26),
    )
    assert pos.market_value == pytest.approx(1600.0)


def test_trade_pnl_long():
    trade = Trade(
        symbol="AAPL", side="long", qty=10,
        entry_price=150.0, exit_price=165.0,
        opened_at=datetime(2026, 3, 25), closed_at=datetime(2026, 3, 26),
        reason="target_hit",
    )
    assert trade.pnl == pytest.approx(150.0)


def test_trade_pnl_short():
    trade = Trade(
        symbol="NVDA", side="short", qty=5,
        entry_price=200.0, exit_price=180.0,
        opened_at=datetime(2026, 3, 25), closed_at=datetime(2026, 3, 26),
        reason="target_hit",
    )
    assert trade.pnl == pytest.approx(100.0)


def test_portfolio_total_value():
    pos = Position(
        symbol="AAPL", side="long", qty=10, entry_price=150.0,
        current_price=160.0, stop_loss=140.0, target_price=170.0,
        opened_at=datetime(2026, 3, 26),
    )
    portfolio = Portfolio(
        cash=3400.0, positions=[pos], trades=[], initial_balance=5000.0,
    )
    assert portfolio.total_value == pytest.approx(5000.0)


def test_portfolio_total_pnl():
    portfolio = Portfolio(
        cash=5200.0, positions=[], trades=[], initial_balance=5000.0,
    )
    assert portfolio.total_pnl == pytest.approx(200.0)
    assert portfolio.total_pnl_pct == pytest.approx(4.0)


def test_position_to_dict_roundtrip():
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
    assert restored.current_price == pos.current_price


def test_trade_to_dict_roundtrip():
    trade = Trade(
        symbol="AAPL", side="long", qty=10,
        entry_price=150.0, exit_price=165.0,
        opened_at=datetime(2026, 3, 25), closed_at=datetime(2026, 3, 26),
        reason="target_hit",
    )
    d = trade.to_dict()
    restored = trade_from_dict(d)
    assert restored.symbol == trade.symbol
    assert restored.pnl == pytest.approx(trade.pnl)
