"""Shared fixtures for SignalForge tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def ohlcv_stock_df() -> pd.DataFrame:
    """Simulate 200 days of AAPL-like stock data."""
    np.random.seed(42)
    n = 200
    base = 180.0
    # Random walk with slight uptrend
    returns = np.random.normal(0.001, 0.015, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0.005, 0.003, n)))
    low = close * (1 - np.abs(np.random.normal(0.005, 0.003, n)))
    opn = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.uniform(50_000_000, 150_000_000, n)

    dates = pd.date_range(
        end=datetime.now(tz=timezone.utc),
        periods=n,
        freq="1D",
    )
    return pd.DataFrame({
        "timestamp": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture()
def ohlcv_crypto_df() -> pd.DataFrame:
    """Simulate 200 days of BTC/USDT-like crypto data."""
    np.random.seed(123)
    n = 200
    base = 65000.0
    returns = np.random.normal(0.002, 0.03, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0.01, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0.01, 0.005, n)))
    opn = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.uniform(10_000, 100_000, n)

    dates = pd.date_range(
        end=datetime.now(tz=timezone.utc),
        periods=n,
        freq="1D",
    )
    return pd.DataFrame({
        "timestamp": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture()
def ohlcv_futures_df() -> pd.DataFrame:
    """Simulate 200 days of ES=F-like futures data."""
    np.random.seed(99)
    n = 200
    base = 5400.0
    returns = np.random.normal(0.0005, 0.01, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0.003, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0.003, 0.002, n)))
    opn = close * (1 + np.random.normal(0, 0.002, n))
    volume = np.random.uniform(1_000_000, 5_000_000, n)

    dates = pd.date_range(
        end=datetime.now(tz=timezone.utc),
        periods=n,
        freq="1D",
    )
    return pd.DataFrame({
        "timestamp": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
