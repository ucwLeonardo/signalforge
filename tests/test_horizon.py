"""Tests for per-asset-type time horizon auto-switching."""

from __future__ import annotations

import pytest

from signalforge.config import DataConfig
from signalforge.data.models import TradeAction, TradeTarget, classify_symbol, AssetType


class TestDataConfigHorizonSettings:
    """DataConfig.get_horizon_settings returns correct (interval, pred_len)."""

    def test_stock_defaults(self) -> None:
        cfg = DataConfig()
        interval, pred_len = cfg.get_horizon_settings("stock")
        assert interval == "1d"
        assert pred_len == 5

    def test_crypto_defaults(self) -> None:
        cfg = DataConfig()
        interval, pred_len = cfg.get_horizon_settings("crypto")
        assert interval == "4h"
        assert pred_len == 30

    def test_futures_defaults(self) -> None:
        cfg = DataConfig()
        interval, pred_len = cfg.get_horizon_settings("futures")
        assert interval == "1d"
        assert pred_len == 5

    def test_options_defaults(self) -> None:
        cfg = DataConfig()
        interval, pred_len = cfg.get_horizon_settings("options")
        assert interval == "1d"
        assert pred_len == 5

    def test_unknown_type_falls_back_to_stock(self) -> None:
        cfg = DataConfig()
        interval, pred_len = cfg.get_horizon_settings("unknown")
        assert interval == "1d"
        assert pred_len == 5

    def test_custom_crypto_settings(self) -> None:
        cfg = DataConfig(crypto_interval="1h", crypto_pred_len=120)
        interval, pred_len = cfg.get_horizon_settings("crypto")
        assert interval == "1h"
        assert pred_len == 120


class TestTradeTargetHorizonDisplay:
    """TradeTarget.horizon_display formats correctly per interval."""

    def _make_target(self, horizon_bars: int, interval: str) -> TradeTarget:
        return TradeTarget(
            symbol="TEST",
            action=TradeAction.BUY,
            entry_price=100.0,
            target_price=110.0,
            stop_loss=95.0,
            risk_reward_ratio=2.0,
            confidence=0.8,
            horizon_bars=horizon_bars,
            interval=interval,
            rationale="test",
        )

    def test_daily_display(self) -> None:
        t = self._make_target(5, "1d")
        assert t.horizon_display == "5d"

    def test_4h_display(self) -> None:
        t = self._make_target(30, "4h")
        assert t.horizon_display == "5d (30x4h)"

    def test_1h_display(self) -> None:
        t = self._make_target(24, "1h")
        assert t.horizon_display == "1d (24x1h)"

    def test_12h_display(self) -> None:
        t = self._make_target(10, "12h")
        assert t.horizon_display == "5d (10x12h)"

    def test_unknown_interval_display(self) -> None:
        t = self._make_target(10, "3m")
        assert t.horizon_display == "10 bars (3m)"

    def test_default_interval_is_1d(self) -> None:
        t = TradeTarget(
            symbol="TEST",
            action=TradeAction.HOLD,
            entry_price=100.0,
            target_price=100.0,
            stop_loss=100.0,
            risk_reward_ratio=0.0,
            confidence=0.5,
            horizon_bars=5,
            rationale="test",
        )
        assert t.interval == "1d"
        assert t.horizon_display == "5d"


class TestClassifySymbol:
    """classify_symbol correctly identifies asset types."""

    def test_stock(self) -> None:
        assert classify_symbol("AAPL") == AssetType.STOCK

    def test_crypto(self) -> None:
        assert classify_symbol("BTC/USDT") == AssetType.CRYPTO

    def test_futures(self) -> None:
        assert classify_symbol("ES=F") == AssetType.FUTURES

    def test_option_human(self) -> None:
        assert classify_symbol("AAPL 2026-06-19 200 C") == AssetType.OPTIONS
