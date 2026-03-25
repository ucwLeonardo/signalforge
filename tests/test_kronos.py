"""Tests for the Kronos prediction engine (fallback mode)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.engines.kronos_engine import (
    KronosConfig,
    KronosEngine,
    _linear_regression_baseline,
)


class TestLinearRegressionBaseline:
    def test_basic_trend(self) -> None:
        """Linear data should produce linear predictions."""
        series = np.arange(100, dtype=np.float64)
        preds = _linear_regression_baseline(series, pred_len=5)
        assert len(preds) == 5
        # Should continue the linear trend
        np.testing.assert_allclose(preds, np.arange(100, 105, dtype=np.float64), atol=0.1)

    def test_constant_series(self) -> None:
        series = np.full(50, 42.0)
        preds = _linear_regression_baseline(series, pred_len=3)
        np.testing.assert_allclose(preds, [42.0, 42.0, 42.0], atol=0.01)

    def test_short_series(self) -> None:
        series = np.array([10.0, 20.0, 30.0])
        preds = _linear_regression_baseline(series, pred_len=2)
        assert len(preds) == 2
        # Should extrapolate upward
        assert preds[0] > 30.0


class TestKronosConfig:
    def test_default_config(self) -> None:
        cfg = KronosConfig()
        assert cfg.pred_len == 24
        assert cfg.temperature == 0.7
        assert cfg.sample_count == 20

    def test_frozen(self) -> None:
        cfg = KronosConfig()
        with pytest.raises(AttributeError):
            cfg.pred_len = 10  # type: ignore[misc]


class TestKronosEngine:
    def test_engine_name(self) -> None:
        engine = KronosEngine()
        assert engine.name == "kronos"

    def test_predict_fallback_stock(self, ohlcv_stock_df: pd.DataFrame) -> None:
        """Kronos engine should work in fallback mode for stock data."""
        engine = KronosEngine(KronosConfig(pred_len=5))
        result = engine.predict(ohlcv_stock_df, pred_len=5)

        assert len(result) == 5
        assert set(result.columns) >= {"open", "high", "low", "close", "volume"}

        # OHLC consistency: high >= max(open, close), low <= min(open, close)
        assert (result["high"] >= result["open"]).all()
        assert (result["high"] >= result["close"]).all()
        assert (result["low"] <= result["open"]).all()
        assert (result["low"] <= result["close"]).all()

        # Volume non-negative
        assert (result["volume"] >= 0).all()

    def test_predict_fallback_crypto(self, ohlcv_crypto_df: pd.DataFrame) -> None:
        """Kronos engine should work for crypto data (larger prices)."""
        engine = KronosEngine(KronosConfig(pred_len=5))
        result = engine.predict(ohlcv_crypto_df, pred_len=5)

        assert len(result) == 5
        assert (result["high"] >= result["low"]).all()
        # Crypto prices should be in a reasonable range
        last_close = ohlcv_crypto_df["close"].iloc[-1]
        # Predictions should be within 50% of last close (reasonable for 5 bars)
        assert result["close"].iloc[-1] > last_close * 0.5
        assert result["close"].iloc[-1] < last_close * 1.5

    def test_predict_fallback_futures(self, ohlcv_futures_df: pd.DataFrame) -> None:
        """Kronos engine should work for futures data."""
        engine = KronosEngine(KronosConfig(pred_len=5))
        result = engine.predict(ohlcv_futures_df, pred_len=5)
        assert len(result) == 5
        assert (result["high"] >= result["low"]).all()

    def test_predict_custom_horizon(self, ohlcv_stock_df: pd.DataFrame) -> None:
        engine = KronosEngine()
        result = engine.predict(ohlcv_stock_df, pred_len=10)
        assert len(result) == 10

    def test_predict_missing_columns(self) -> None:
        df = pd.DataFrame({"close": [1, 2, 3], "timestamp": pd.date_range("2025-01-01", periods=3)})
        engine = KronosEngine()
        with pytest.raises(ValueError, match="missing required columns"):
            engine.predict(df)

    def test_config_overrides(self) -> None:
        engine = KronosEngine(pred_len=10, temperature=0.5)
        assert engine.config.pred_len == 10
        assert engine.config.temperature == 0.5
