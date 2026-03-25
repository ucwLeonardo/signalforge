"""Tests for data models and symbol classification."""

from __future__ import annotations

from datetime import datetime

import pytest

from signalforge.data.models import (
    Action,
    Asset,
    AssetType,
    Bar,
    CombinedSignal,
    Signal,
    SupportResistance,
    TradeAction,
    TradeTarget,
    asset_from_symbol,
    classify_symbol,
)


class TestClassifySymbol:
    """Test symbol classification into asset types."""

    @pytest.mark.parametrize(
        "symbol,expected",
        [
            ("AAPL", AssetType.STOCK),
            ("MSFT", AssetType.STOCK),
            ("NVDA", AssetType.STOCK),
            ("TSLA", AssetType.STOCK),
            ("TSM", AssetType.STOCK),
            ("BTC/USDT", AssetType.CRYPTO),
            ("ETH/USDT", AssetType.CRYPTO),
            ("SOL/USDT", AssetType.CRYPTO),
            ("DOGE/USDT", AssetType.CRYPTO),
            ("ES=F", AssetType.FUTURES),
            ("NQ=F", AssetType.FUTURES),
            ("GC=F", AssetType.FUTURES),
            ("CL=F", AssetType.FUTURES),
        ],
    )
    def test_classify_symbol(self, symbol: str, expected: AssetType) -> None:
        assert classify_symbol(symbol) == expected

    def test_asset_from_symbol_stock(self) -> None:
        asset = asset_from_symbol("AAPL")
        assert asset.symbol == "AAPL"
        assert asset.asset_type == AssetType.STOCK

    def test_asset_from_symbol_crypto(self) -> None:
        asset = asset_from_symbol("BTC/USDT")
        assert asset.symbol == "BTC/USDT"
        assert asset.asset_type == AssetType.CRYPTO

    def test_asset_from_symbol_futures(self) -> None:
        asset = asset_from_symbol("ES=F")
        assert asset.symbol == "ES=F"
        assert asset.asset_type == AssetType.FUTURES


class TestBar:
    def test_bar_creation(self) -> None:
        bar = Bar(
            timestamp=datetime(2025, 1, 1),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1_000_000,
        )
        assert bar.close == 105.0
        assert bar.amount is None

    def test_bar_is_frozen(self) -> None:
        bar = Bar(
            timestamp=datetime(2025, 1, 1),
            open=100.0, high=110.0, low=95.0, close=105.0, volume=1_000_000,
        )
        with pytest.raises(AttributeError):
            bar.close = 200.0  # type: ignore[misc]


class TestSignal:
    def test_valid_confidence(self) -> None:
        sig = Signal(
            asset=Asset(symbol="AAPL", asset_type=AssetType.STOCK),
            timestamp=datetime(2025, 1, 1),
            action=Action.BUY,
            entry_price=150.0,
            confidence=0.75,
        )
        assert sig.confidence == 0.75

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            Signal(
                asset=Asset(symbol="AAPL", asset_type=AssetType.STOCK),
                timestamp=datetime(2025, 1, 1),
                action=Action.BUY,
                entry_price=150.0,
                confidence=1.5,
            )


class TestCombinedSignal:
    def test_combined_signal_defaults(self) -> None:
        sig = CombinedSignal(direction=0.5, confidence=0.8)
        assert sig.predicted_high is None
        assert sig.predicted_low is None
        assert sig.predicted_close is None


class TestTradeTarget:
    def test_trade_target_buy(self) -> None:
        target = TradeTarget(
            symbol="AAPL",
            action=TradeAction.BUY,
            entry_price=180.0,
            target_price=200.0,
            stop_loss=170.0,
            risk_reward_ratio=2.0,
            confidence=0.78,
            horizon_days=5,
            rationale="Bullish signal",
        )
        assert target.action == TradeAction.BUY
        assert target.risk_reward_ratio == 2.0

    def test_trade_action_values(self) -> None:
        assert TradeAction.BUY.value == "BUY"
        assert TradeAction.SELL.value == "SELL"
        assert TradeAction.HOLD.value == "HOLD"
