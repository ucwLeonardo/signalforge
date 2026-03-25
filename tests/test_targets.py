"""Tests for the target calculator (BUY/SELL/HOLD logic)."""

from __future__ import annotations

import pytest

from signalforge.data.models import CombinedSignal, SupportResistance, TradeAction
from signalforge.ensemble.targets import TargetCalculator, _classify_action, _risk_reward


class TestClassifyAction:
    def test_buy_threshold(self) -> None:
        assert _classify_action(0.15) == TradeAction.BUY
        assert _classify_action(0.5) == TradeAction.BUY
        assert _classify_action(1.0) == TradeAction.BUY

    def test_sell_threshold(self) -> None:
        assert _classify_action(-0.15) == TradeAction.SELL
        assert _classify_action(-0.5) == TradeAction.SELL
        assert _classify_action(-1.0) == TradeAction.SELL

    def test_hold_zone(self) -> None:
        assert _classify_action(0.0) == TradeAction.HOLD
        assert _classify_action(0.14) == TradeAction.HOLD
        assert _classify_action(-0.14) == TradeAction.HOLD


class TestRiskReward:
    def test_positive_buy(self) -> None:
        rr = _risk_reward(entry=100.0, target=120.0, stop=90.0)
        assert rr == 2.0

    def test_positive_sell(self) -> None:
        rr = _risk_reward(entry=100.0, target=80.0, stop=110.0)
        assert rr == 2.0

    def test_zero_risk(self) -> None:
        rr = _risk_reward(entry=100.0, target=110.0, stop=100.0)
        assert rr == 0.0


class TestTargetCalculator:
    def setup_method(self) -> None:
        self.calc = TargetCalculator()

    def test_buy_target_stock(self) -> None:
        signal = CombinedSignal(
            direction=0.6,
            confidence=0.78,
            predicted_high=200.0,
            predicted_low=175.0,
            predicted_close=195.0,
        )
        levels = SupportResistance(support=170.0, resistance=205.0)
        target = self.calc.calculate(
            symbol="AAPL",
            signal=signal,
            current_price=185.0,
            levels=levels,
        )

        assert target.action == TradeAction.BUY
        assert target.entry_price <= 185.0  # entry at predicted_low or current
        assert target.target_price >= 185.0  # target above current
        assert target.stop_loss < target.entry_price
        assert target.risk_reward_ratio >= 0.0
        assert target.confidence == 0.78
        assert "AAPL" == target.symbol

    def test_sell_target_crypto(self) -> None:
        signal = CombinedSignal(
            direction=-0.5,
            confidence=0.65,
            predicted_high=70000.0,
            predicted_low=58000.0,
            predicted_close=60000.0,
        )
        levels = SupportResistance(support=55000.0, resistance=72000.0)
        target = self.calc.calculate(
            symbol="BTC/USDT",
            signal=signal,
            current_price=67000.0,
            levels=levels,
        )

        assert target.action == TradeAction.SELL
        assert target.entry_price == 67000.0
        assert target.target_price < 67000.0
        assert target.stop_loss > target.entry_price

    def test_hold_target_futures(self) -> None:
        signal = CombinedSignal(
            direction=0.05,
            confidence=0.45,
            predicted_close=5420.0,
        )
        target = self.calc.calculate(
            symbol="ES=F",
            signal=signal,
            current_price=5400.0,
        )

        assert target.action == TradeAction.HOLD
        assert target.entry_price == 5400.0
        assert target.target_price == 5400.0
        assert target.risk_reward_ratio == 0.0

    def test_buy_without_predicted_prices(self) -> None:
        """Buy signal with no price predictions should still work."""
        signal = CombinedSignal(direction=0.5, confidence=0.7)
        levels = SupportResistance(support=90.0, resistance=110.0)
        target = self.calc.calculate(
            symbol="MSFT",
            signal=signal,
            current_price=100.0,
            levels=levels,
        )
        assert target.action == TradeAction.BUY
        assert target.target_price >= 100.0

    def test_sell_without_levels(self) -> None:
        """Sell signal without support/resistance should use defaults."""
        signal = CombinedSignal(
            direction=-0.4,
            confidence=0.6,
            predicted_low=95.0,
            predicted_high=105.0,
        )
        target = self.calc.calculate(
            symbol="NVDA",
            signal=signal,
            current_price=100.0,
        )
        assert target.action == TradeAction.SELL

    def test_custom_horizon(self) -> None:
        signal = CombinedSignal(direction=0.3, confidence=0.5)
        target = self.calc.calculate(
            symbol="AAPL",
            signal=signal,
            current_price=180.0,
            horizon_days=10,
        )
        assert target.horizon_days == 10

    def test_rationale_contains_direction(self) -> None:
        signal = CombinedSignal(direction=0.7, confidence=0.85)
        target = self.calc.calculate(
            symbol="TSLA",
            signal=signal,
            current_price=250.0,
            levels=SupportResistance(support=230.0, resistance=270.0),
        )
        assert "Bullish" in target.rationale
        assert "0.70" in target.rationale or "+0.70" in target.rationale
