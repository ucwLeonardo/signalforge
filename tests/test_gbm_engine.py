"""Tests for the GBM ensemble prediction engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.engines.gbm_engine import (
    GBMConfig,
    GBMEnsembleEngine,
    _compute_features,
)


class TestGBMConfig:
    def test_default_config(self) -> None:
        cfg = GBMConfig()
        assert cfg.n_estimators == 200
        assert cfg.max_depth == 6
        assert cfg.learning_rate == 0.05
        assert cfg.label_horizon == 5

    def test_frozen(self) -> None:
        cfg = GBMConfig()
        with pytest.raises(AttributeError):
            cfg.n_estimators = 500  # type: ignore[misc]


class TestComputeFeatures:
    def test_output_has_features(self, ohlcv_stock_df: pd.DataFrame) -> None:
        features = _compute_features(ohlcv_stock_df)
        assert len(features) == len(ohlcv_stock_df)
        # Should have momentum, trend, volatility, volume, price action features
        assert "ret_5" in features.columns
        assert "ret_20" in features.columns
        assert "ma_ratio_5" in features.columns
        assert "vol_5" in features.columns
        assert "rsi_14" in features.columns
        assert "macd_hist" in features.columns
        assert "body_ratio" in features.columns

    def test_feature_count(self, ohlcv_stock_df: pd.DataFrame) -> None:
        features = _compute_features(ohlcv_stock_df)
        # Should produce a rich set of features (at least 20)
        assert features.shape[1] >= 20

    def test_custom_windows(self, ohlcv_stock_df: pd.DataFrame) -> None:
        features = _compute_features(ohlcv_stock_df, windows=(3, 7, 14))
        assert "ret_3" in features.columns
        assert "ret_7" in features.columns
        assert "ret_14" in features.columns


class TestGBMEnsembleEngine:
    def test_engine_name(self) -> None:
        engine = GBMEnsembleEngine()
        assert engine.name == "gbm"

    def test_predict_stock(self, ohlcv_stock_df: pd.DataFrame) -> None:
        """GBM engine should produce valid predictions on stock data."""
        engine = GBMEnsembleEngine(GBMConfig(n_estimators=20))
        result = engine.predict(ohlcv_stock_df, pred_len=5)

        assert len(result) == 5
        expected_cols = {"open", "high", "low", "close", "volume", "predicted_return", "confidence"}
        assert expected_cols <= set(result.columns)

        # OHLC consistency
        assert (result["high"] >= result["open"]).all()
        assert (result["high"] >= result["close"]).all()
        assert (result["low"] <= result["open"]).all()
        assert (result["low"] <= result["close"]).all()
        assert (result["volume"] >= 0).all()
        assert (result["low"] >= 0).all()

    def test_predict_crypto(self, ohlcv_crypto_df: pd.DataFrame) -> None:
        engine = GBMEnsembleEngine(GBMConfig(n_estimators=20))
        result = engine.predict(ohlcv_crypto_df, pred_len=5)
        assert len(result) == 5

        # Crypto predictions should be in a reasonable range
        last_close = ohlcv_crypto_df["close"].iloc[-1]
        pred_close = result["close"].iloc[-1]
        assert pred_close > last_close * 0.5
        assert pred_close < last_close * 1.5

    def test_predict_futures(self, ohlcv_futures_df: pd.DataFrame) -> None:
        engine = GBMEnsembleEngine(GBMConfig(n_estimators=20))
        result = engine.predict(ohlcv_futures_df, pred_len=5)
        assert len(result) == 5

    def test_predict_with_short_data(self) -> None:
        """Should handle insufficient data gracefully."""
        np.random.seed(42)
        short_df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=30, freq="1D"),
            "open": np.random.uniform(100, 110, 30),
            "high": np.random.uniform(110, 120, 30),
            "low": np.random.uniform(90, 100, 30),
            "close": np.random.uniform(100, 110, 30),
            "volume": np.random.uniform(1e6, 5e6, 30),
        })
        engine = GBMEnsembleEngine(GBMConfig(min_train_rows=60))
        result = engine.predict(short_df, pred_len=5)
        assert len(result) == 5  # Should still return zero predictions

    def test_predict_missing_columns(self) -> None:
        df = pd.DataFrame({
            "close": [1, 2, 3],
            "timestamp": pd.date_range("2025-01-01", periods=3),
        })
        engine = GBMEnsembleEngine()
        with pytest.raises(ValueError, match="missing required columns"):
            engine.predict(df)

    def test_config_overrides(self) -> None:
        engine = GBMEnsembleEngine(n_estimators=500, max_depth=4)
        assert engine.config.n_estimators == 500
        assert engine.config.max_depth == 4

    def test_confidence_range(self, ohlcv_stock_df: pd.DataFrame) -> None:
        engine = GBMEnsembleEngine(GBMConfig(n_estimators=20))
        result = engine.predict(ohlcv_stock_df, pred_len=5)
        for conf in result["confidence"]:
            assert 0.0 <= conf <= 1.0

    def test_predicted_returns_present(self, ohlcv_stock_df: pd.DataFrame) -> None:
        engine = GBMEnsembleEngine(GBMConfig(n_estimators=20))
        result = engine.predict(ohlcv_stock_df, pred_len=5)
        assert "predicted_return" in result.columns
        assert len(result["predicted_return"]) == 5

    def test_lgbm_vs_sklearn_fallback(self, ohlcv_stock_df: pd.DataFrame) -> None:
        """Both paths should produce valid results."""
        engine = GBMEnsembleEngine(GBMConfig(n_estimators=20))
        result = engine.predict(ohlcv_stock_df, pred_len=5)
        # Whether lgbm or sklearn, the result shape should be consistent
        assert result.shape[0] == 5
        assert result.shape[1] >= 7
