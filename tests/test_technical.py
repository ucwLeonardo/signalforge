"""Tests for the technical analysis engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.engines.technical import (
    TechnicalEngine,
    compute_signals,
    compute_support_resistance,
    _rsi,
    _macd,
    _bollinger_bands,
    _atr,
)


class TestIndicators:
    """Test individual indicator computations."""

    def test_rsi_range(self, ohlcv_stock_df: pd.DataFrame) -> None:
        rsi = _rsi(ohlcv_stock_df["close"])
        valid = rsi.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_rsi_default_length(self, ohlcv_stock_df: pd.DataFrame) -> None:
        rsi = _rsi(ohlcv_stock_df["close"], length=14)
        # First 13 values should be NaN-ish (EWM with min_periods=14)
        assert rsi.iloc[:13].isna().sum() >= 10

    def test_macd_returns_three_series(self, ohlcv_stock_df: pd.DataFrame) -> None:
        line, signal, hist = _macd(ohlcv_stock_df["close"])
        assert len(line) == len(ohlcv_stock_df)
        assert len(signal) == len(ohlcv_stock_df)
        assert len(hist) == len(ohlcv_stock_df)
        # Histogram = line - signal
        np.testing.assert_allclose(hist.dropna().values, (line - signal).dropna().values, atol=1e-10)

    def test_bollinger_bands_ordering(self, ohlcv_stock_df: pd.DataFrame) -> None:
        lower, mid, upper = _bollinger_bands(ohlcv_stock_df["close"])
        valid_mask = ~(lower.isna() | upper.isna())
        assert (lower[valid_mask] <= mid[valid_mask]).all()
        assert (mid[valid_mask] <= upper[valid_mask]).all()

    def test_atr_positive(self, ohlcv_stock_df: pd.DataFrame) -> None:
        df = ohlcv_stock_df
        atr = _atr(df["high"], df["low"], df["close"])
        valid = atr.dropna()
        assert (valid > 0).all()


class TestSupportResistance:
    """Test support/resistance level detection."""

    def test_returns_supports_and_resistances(self, ohlcv_stock_df: pd.DataFrame) -> None:
        supports, resistances = compute_support_resistance(ohlcv_stock_df)
        assert len(supports) >= 1
        assert len(resistances) >= 1

    def test_support_below_close(self, ohlcv_stock_df: pd.DataFrame) -> None:
        last_close = ohlcv_stock_df["close"].iloc[-1]
        supports, _ = compute_support_resistance(ohlcv_stock_df)
        assert all(s < last_close for s in supports)

    def test_resistance_above_or_at_close(self, ohlcv_stock_df: pd.DataFrame) -> None:
        last_close = ohlcv_stock_df["close"].iloc[-1]
        _, resistances = compute_support_resistance(ohlcv_stock_df)
        assert all(r >= last_close for r in resistances)

    def test_crypto_support_resistance(self, ohlcv_crypto_df: pd.DataFrame) -> None:
        supports, resistances = compute_support_resistance(ohlcv_crypto_df)
        assert len(supports) >= 1
        assert len(resistances) >= 1

    def test_futures_support_resistance(self, ohlcv_futures_df: pd.DataFrame) -> None:
        supports, resistances = compute_support_resistance(ohlcv_futures_df)
        assert len(supports) >= 1
        assert len(resistances) >= 1


class TestComputeSignals:
    """Test composite signal computation."""

    def test_signal_strength_range(self, ohlcv_stock_df: pd.DataFrame) -> None:
        result = compute_signals(ohlcv_stock_df)
        assert "signal_strength" in result.columns
        assert result["signal_strength"].min() >= -1.0
        assert result["signal_strength"].max() <= 1.0

    def test_output_columns(self, ohlcv_stock_df: pd.DataFrame) -> None:
        result = compute_signals(ohlcv_stock_df)
        expected_cols = {"timestamp", "signal_strength", "support", "resistance"}
        assert expected_cols.issubset(set(result.columns))

    def test_output_length_matches_input(self, ohlcv_stock_df: pd.DataFrame) -> None:
        result = compute_signals(ohlcv_stock_df)
        assert len(result) == len(ohlcv_stock_df)

    def test_crypto_signals(self, ohlcv_crypto_df: pd.DataFrame) -> None:
        result = compute_signals(ohlcv_crypto_df)
        assert len(result) == len(ohlcv_crypto_df)
        assert result["signal_strength"].min() >= -1.0

    def test_futures_signals(self, ohlcv_futures_df: pd.DataFrame) -> None:
        result = compute_signals(ohlcv_futures_df)
        assert len(result) == len(ohlcv_futures_df)


class TestTechnicalEngine:
    """Test the TechnicalEngine class."""

    def test_engine_name(self) -> None:
        engine = TechnicalEngine()
        assert engine.name == "technical"

    def test_predict_returns_signals(self, ohlcv_stock_df: pd.DataFrame) -> None:
        engine = TechnicalEngine()
        result = engine.predict(ohlcv_stock_df)
        assert "signal_strength" in result.columns
        assert len(result) > 0

    def test_predict_missing_columns(self) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        engine = TechnicalEngine()
        with pytest.raises(ValueError, match="missing required columns"):
            engine.predict(df)
