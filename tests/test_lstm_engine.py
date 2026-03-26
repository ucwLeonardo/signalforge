"""Tests for the LSTM prediction engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.engines.lstm_engine import (
    FEATURE_COLS,
    LSTMConfig,
    LSTMEngine,
    _create_sequences,
    _ewma_baseline,
    _prepare_features,
)


class TestLSTMConfig:
    def test_default_config(self) -> None:
        cfg = LSTMConfig()
        assert cfg.hidden_size == 64
        assert cfg.num_layers == 2
        assert cfg.dropout == 0.2
        assert cfg.lookback == 60
        assert cfg.epochs == 50

    def test_frozen(self) -> None:
        cfg = LSTMConfig()
        with pytest.raises(AttributeError):
            cfg.hidden_size = 128  # type: ignore[misc]


class TestPrepareFeatures:
    def test_output_shape(self, ohlcv_stock_df: pd.DataFrame) -> None:
        data = _prepare_features(ohlcv_stock_df)
        assert data.shape == (len(ohlcv_stock_df), len(FEATURE_COLS))
        assert data.dtype == np.float64

    def test_no_nan_in_raw_data(self, ohlcv_stock_df: pd.DataFrame) -> None:
        data = _prepare_features(ohlcv_stock_df)
        assert not np.isnan(data).any()


class TestCreateSequences:
    def test_output_shapes(self) -> None:
        data = np.random.randn(100, 5)
        X, Y = _create_sequences(data, lookback=10, pred_len=5)
        assert X.shape == (86, 10, 5)  # 100 - 10 - 5 + 1 = 86
        assert Y.shape == (86, 5, 5)

    def test_continuity(self) -> None:
        """Last element of X[i] should be followed by first element of Y[i]."""
        data = np.arange(50).reshape(50, 1).astype(float)
        X, Y = _create_sequences(data, lookback=5, pred_len=3)
        # X[0] = [0..4], Y[0] = [5..7]
        assert X[0, -1, 0] == 4.0
        assert Y[0, 0, 0] == 5.0


class TestEWMABaseline:
    def test_output_keys(self, ohlcv_stock_df: pd.DataFrame) -> None:
        result = _ewma_baseline(ohlcv_stock_df, pred_len=5)
        for col in FEATURE_COLS:
            assert col in result
            assert len(result[col]) == 5

    def test_volume_non_negative(self, ohlcv_stock_df: pd.DataFrame) -> None:
        result = _ewma_baseline(ohlcv_stock_df, pred_len=5)
        assert (result["volume"] >= 0).all()

    def test_ohlc_consistency(self, ohlcv_stock_df: pd.DataFrame) -> None:
        result = _ewma_baseline(ohlcv_stock_df, pred_len=5)
        assert (result["high"] >= result["open"]).all()
        assert (result["high"] >= result["close"]).all()
        assert (result["low"] <= result["open"]).all()
        assert (result["low"] <= result["close"]).all()


class TestLSTMEngine:
    def test_engine_name(self) -> None:
        engine = LSTMEngine()
        assert engine.name == "lstm"

    def test_predict_stock(self, ohlcv_stock_df: pd.DataFrame) -> None:
        """LSTM engine should produce valid predictions on stock data."""
        engine = LSTMEngine(LSTMConfig(
            epochs=5,  # fast for tests
            lookback=20,
            hidden_size=16,
            mc_samples=3,
        ))
        result = engine.predict(ohlcv_stock_df, pred_len=5)

        assert len(result) == 5
        assert set(result.columns) >= {"open", "high", "low", "close", "volume", "confidence"}

        # OHLC consistency
        assert (result["high"] >= result["open"]).all()
        assert (result["high"] >= result["close"]).all()
        assert (result["low"] <= result["open"]).all()
        assert (result["low"] <= result["close"]).all()
        assert (result["volume"] >= 0).all()

    def test_predict_crypto(self, ohlcv_crypto_df: pd.DataFrame) -> None:
        engine = LSTMEngine(LSTMConfig(epochs=3, lookback=20, hidden_size=16, mc_samples=2))
        result = engine.predict(ohlcv_crypto_df, pred_len=5)
        assert len(result) == 5

    def test_predict_with_insufficient_data(self) -> None:
        """Should fall back to EWMA when data is too short."""
        np.random.seed(42)
        short_df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=30, freq="1D"),
            "open": np.random.uniform(100, 110, 30),
            "high": np.random.uniform(110, 120, 30),
            "low": np.random.uniform(90, 100, 30),
            "close": np.random.uniform(100, 110, 30),
            "volume": np.random.uniform(1e6, 5e6, 30),
        })
        engine = LSTMEngine(LSTMConfig(lookback=60))
        result = engine.predict(short_df, pred_len=5)
        assert len(result) == 5  # Should still produce results via fallback

    def test_predict_missing_columns(self) -> None:
        df = pd.DataFrame({
            "close": [1, 2, 3],
            "timestamp": pd.date_range("2025-01-01", periods=3),
        })
        engine = LSTMEngine()
        with pytest.raises(ValueError, match="missing required columns"):
            engine.predict(df)

    def test_config_overrides(self) -> None:
        engine = LSTMEngine(hidden_size=128, epochs=10)
        assert engine.config.hidden_size == 128
        assert engine.config.epochs == 10

    def test_confidence_range(self, ohlcv_stock_df: pd.DataFrame) -> None:
        engine = LSTMEngine(LSTMConfig(epochs=3, lookback=20, hidden_size=16, mc_samples=3))
        result = engine.predict(ohlcv_stock_df, pred_len=5)
        conf = result["confidence"].iloc[0]
        assert 0.0 <= conf <= 1.0
