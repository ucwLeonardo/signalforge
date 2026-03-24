"""Technical analysis engine -- indicator-based signal generation.

Uses ``pandas_ta`` to compute common technical indicators and derives
composite buy/sell signals from RSI, MACD, Bollinger Bands, and
support/resistance proximity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from signalforge.engines.base import PredictionEngine

# ---------------------------------------------------------------------------
# Optional pandas_ta import
# ---------------------------------------------------------------------------

_PANDAS_TA_AVAILABLE: bool = False

try:
    import pandas_ta as ta  # type: ignore[import-untyped]

    _PANDAS_TA_AVAILABLE = True
except ImportError:
    ta = None  # type: ignore[assignment]
    logger.warning(
        "pandas_ta not installed. Install with:\n"
        "  pip install pandas_ta\n"
        "TechnicalEngine will use built-in fallback indicators."
    )


# ---------------------------------------------------------------------------
# Fallback indicator implementations (pure numpy/pandas)
# ---------------------------------------------------------------------------


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger_bands(
    series: pd.Series,
    length: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Lower band, mid band, upper band."""
    mid = series.rolling(window=length).mean()
    rolling_std = series.rolling(window=length).std()
    upper = mid + std_dev * rolling_std
    lower = mid - std_dev * rolling_std
    return lower, mid, upper


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=length).mean()


# ---------------------------------------------------------------------------
# Support / Resistance
# ---------------------------------------------------------------------------


def compute_support_resistance(
    df: pd.DataFrame,
    lookback: int = 60,
    num_levels: int = 3,
    cluster_pct: float = 0.015,
) -> tuple[list[float], list[float]]:
    """Find support and resistance levels via pivot-point clustering.

    Parameters
    ----------
    df:
        OHLCV DataFrame (must have ``high``, ``low``, ``close`` columns).
    lookback:
        Number of recent candles to scan for pivots.
    num_levels:
        Maximum number of S/R levels to return on each side.
    cluster_pct:
        Two pivots within this percentage of each other are merged.

    Returns
    -------
    tuple of (support_levels, resistance_levels)
        Each is a sorted list of price floats.
    """
    window = df.tail(lookback).copy()
    highs = window["high"].values
    lows = window["low"].values
    last_close = float(window["close"].iloc[-1])

    # Collect pivot highs and pivot lows (local extremes over 5-bar window)
    pivots: list[float] = []
    for i in range(2, len(highs) - 2):
        if highs[i] == max(highs[i - 2 : i + 3]):
            pivots.append(float(highs[i]))
        if lows[i] == min(lows[i - 2 : i + 3]):
            pivots.append(float(lows[i]))

    if not pivots:
        return [last_close * 0.98], [last_close * 1.02]

    # Cluster nearby pivots
    pivots.sort()
    clusters: list[list[float]] = [[pivots[0]]]
    for p in pivots[1:]:
        if (p - clusters[-1][-1]) / (clusters[-1][-1] + 1e-12) < cluster_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    # Weighted average per cluster, weighted by cluster size
    levels = sorted(
        [(np.mean(c), len(c)) for c in clusters],
        key=lambda x: x[1],
        reverse=True,
    )
    price_levels = [lv[0] for lv in levels]

    supports = sorted([p for p in price_levels if p < last_close], reverse=True)[:num_levels]
    resistances = sorted([p for p in price_levels if p >= last_close])[:num_levels]

    # Guarantee at least one level on each side
    if not supports:
        supports = [last_close * 0.98]
    if not resistances:
        resistances = [last_close * 1.02]

    return supports, resistances


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite buy/sell signals from technical indicators.

    Parameters
    ----------
    df:
        OHLCV DataFrame with ``open, high, low, close, volume`` columns and
        a datetime-like index (or ``timestamp`` column).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp, signal_strength, support, resistance``.
        ``signal_strength`` ranges from -1 (strong sell) to +1 (strong buy).
    """
    work = df.copy()
    if "timestamp" in work.columns and not isinstance(work.index, pd.DatetimeIndex):
        work = work.set_index("timestamp")

    close = work["close"]

    # --- Indicators --------------------------------------------------------
    if _PANDAS_TA_AVAILABLE:
        rsi_vals = ta.rsi(close, length=14)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_line = macd_df.iloc[:, 0]
        macd_signal = macd_df.iloc[:, 1]
        macd_hist = macd_df.iloc[:, 2]
        bb = ta.bbands(close, length=20, std=2.0)
        bb_lower = bb.iloc[:, 0]
        bb_upper = bb.iloc[:, 2]
        atr_vals = ta.atr(work["high"], work["low"], close, length=14)
    else:
        rsi_vals = _rsi(close)
        macd_line, macd_signal, macd_hist = _macd(close)
        bb_lower, _bb_mid, bb_upper = _bollinger_bands(close)
        atr_vals = _atr(work["high"], work["low"], close)

    # --- Per-bar signal components -----------------------------------------
    n = len(work)
    signals = np.zeros(n, dtype=np.float64)

    rsi_arr = rsi_vals.values if rsi_vals is not None else np.full(n, 50.0)
    macd_h = macd_hist.values if macd_hist is not None else np.zeros(n)
    bbl = bb_lower.values if bb_lower is not None else np.full(n, np.nan)
    bbu = bb_upper.values if bb_upper is not None else np.full(n, np.nan)
    close_arr = close.values

    for i in range(n):
        score = 0.0

        # RSI component (weight 0.30)
        r = rsi_arr[i] if not np.isnan(rsi_arr[i]) else 50.0
        if r < 30:
            score += 0.30 * ((30.0 - r) / 30.0)  # oversold -> buy
        elif r > 70:
            score -= 0.30 * ((r - 70.0) / 30.0)  # overbought -> sell

        # MACD histogram component (weight 0.30)
        h = macd_h[i] if not np.isnan(macd_h[i]) else 0.0
        # Normalise loosely -- cap contribution at +/- 0.30
        score += _clamp(h / (abs(close_arr[i]) * 0.01 + 1e-12), -1.0, 1.0) * 0.30

        # Bollinger Band component (weight 0.25)
        if not (np.isnan(bbl[i]) or np.isnan(bbu[i])):
            band_width = bbu[i] - bbl[i] + 1e-12
            position = (close_arr[i] - bbl[i]) / band_width  # 0 = lower, 1 = upper
            # Near lower band -> buy; near upper band -> sell
            score += (0.5 - position) * 0.50  # ranges roughly -0.25 to +0.25

        signals[i] = _clamp(score)

    # --- S/R levels (last bar context) -------------------------------------
    supports, resistances = compute_support_resistance(work)
    nearest_support = supports[0] if supports else float("nan")
    nearest_resistance = resistances[0] if resistances else float("nan")

    # S/R proximity bonus (weight 0.15 of total, applied to last 5 bars)
    for i in range(max(0, n - 5), n):
        price = close_arr[i]
        if not np.isnan(nearest_support):
            dist_s = (price - nearest_support) / (nearest_support + 1e-12)
            if 0 < dist_s < 0.01:
                signals[i] = _clamp(signals[i] + 0.15)  # near support -> buy
        if not np.isnan(nearest_resistance):
            dist_r = (nearest_resistance - price) / (nearest_resistance + 1e-12)
            if 0 < dist_r < 0.01:
                signals[i] = _clamp(signals[i] - 0.15)  # near resistance -> sell

    result = pd.DataFrame(
        {
            "timestamp": work.index,
            "signal_strength": signals,
            "support": nearest_support,
            "resistance": nearest_resistance,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------


class TechnicalEngine(PredictionEngine):
    """Technical-analysis engine that produces directional signals rather
    than explicit price predictions.

    Implements :class:`PredictionEngine` loosely -- :meth:`predict` returns a
    signal DataFrame (``timestamp, signal_strength, support, resistance``)
    instead of OHLCV candles.
    """

    @property
    def name(self) -> str:
        return "technical"

    def predict(self, df: pd.DataFrame, pred_len: int = 0) -> pd.DataFrame:
        """Compute technical signals over the provided OHLCV data.

        Parameters
        ----------
        df:
            Historical OHLCV DataFrame.
        pred_len:
            Ignored -- included for interface compatibility.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp, signal_strength, support, resistance``.
        """
        required = {"open", "high", "low", "close", "volume"}
        present = {c.lower() for c in df.columns} | (
            {df.index.name.lower()} if df.index.name else set()
        )
        missing = required - present
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        return compute_signals(df)
