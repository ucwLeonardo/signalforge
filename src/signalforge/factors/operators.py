"""WorldQuant-style time-series and cross-sectional operators.

These operators form the DSL (domain-specific language) for expressing
alpha factors.  They are designed to work on :class:`pd.Series` (single
asset) or :class:`pd.DataFrame` (multi-asset, columns = symbols).

Time-series operators act on one asset across time (rolling windows).
Cross-sectional operators act across all assets at a single timestamp.

Reference: Kakushadze (2016) "101 Formulaic Alphas", arXiv:1601.00991.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Time-series operators (operate on one asset across time)
# -----------------------------------------------------------------------

def delay(x: pd.Series | pd.DataFrame, d: int) -> pd.Series | pd.DataFrame:
    """Value of *x* from *d* periods ago.  ``delay(x, 1)`` = yesterday's value."""
    return x.shift(d)


def delta(x: pd.Series | pd.DataFrame, d: int) -> pd.Series | pd.DataFrame:
    """Change in *x* over *d* periods: ``x_t - x_{t-d}``."""
    return x - x.shift(d)


def ts_sum(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """Rolling sum over the past *w* periods."""
    return x.rolling(w, min_periods=1).sum()


def ts_mean(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """Rolling mean over the past *w* periods."""
    return x.rolling(w, min_periods=1).mean()


def ts_std(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """Rolling standard deviation over the past *w* periods."""
    return x.rolling(w, min_periods=max(2, w // 2)).std()


def ts_rank(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """Rank of current value within its own past *w*-period window.

    Returns a value in [0, 1] where 1 means the current value is the
    highest in the window.
    """
    def _rank_in_window(arr: np.ndarray) -> float:
        if len(arr) < 2 or np.all(np.isnan(arr)):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return np.nan
        current = valid[-1]
        return float(np.sum(valid <= current) / len(valid))

    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: col.rolling(w, min_periods=2).apply(_rank_in_window, raw=True))
    return x.rolling(w, min_periods=2).apply(_rank_in_window, raw=True)


def ts_corr(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    w: int,
) -> pd.Series | pd.DataFrame:
    """Rolling Pearson correlation between *x* and *y* over *w* periods."""
    return x.rolling(w, min_periods=max(3, w // 2)).corr(y)


def ts_cov(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    w: int,
) -> pd.Series | pd.DataFrame:
    """Rolling covariance between *x* and *y* over *w* periods."""
    return x.rolling(w, min_periods=max(3, w // 2)).cov(y)


def ts_min(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """Rolling minimum over the past *w* periods."""
    return x.rolling(w, min_periods=1).min()


def ts_max(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """Rolling maximum over the past *w* periods."""
    return x.rolling(w, min_periods=1).max()


def ts_argmax(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """How many periods ago the max occurred within the past *w* periods.

    Returns 0 if the max is at the current bar, *w-1* if at the oldest bar.
    """
    def _argmax(arr: np.ndarray) -> float:
        if np.all(np.isnan(arr)):
            return np.nan
        return float(len(arr) - 1 - np.nanargmax(arr))

    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: col.rolling(w, min_periods=1).apply(_argmax, raw=True))
    return x.rolling(w, min_periods=1).apply(_argmax, raw=True)


def ts_argmin(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """How many periods ago the min occurred within the past *w* periods."""
    def _argmin(arr: np.ndarray) -> float:
        if np.all(np.isnan(arr)):
            return np.nan
        return float(len(arr) - 1 - np.nanargmin(arr))

    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: col.rolling(w, min_periods=1).apply(_argmin, raw=True))
    return x.rolling(w, min_periods=1).apply(_argmin, raw=True)


def ts_zscore(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """Z-score of current value relative to the past *w* periods.

    ``(x - ts_mean(x, w)) / ts_std(x, w)``
    """
    mean = ts_mean(x, w)
    std = ts_std(x, w)
    return (x - mean) / (std + 1e-12)


def decay_linear(x: pd.Series | pd.DataFrame, w: int) -> pd.Series | pd.DataFrame:
    """Linearly weighted moving average with weights ``[w, w-1, ..., 1]``.

    Most recent observation gets highest weight.
    """
    weights = np.arange(1, w + 1, dtype=np.float64)
    weights = weights / weights.sum()

    def _weighted_avg(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        # Use the tail of weights matching the valid length
        w_slice = weights[-len(valid):]
        w_slice = w_slice / w_slice.sum()
        return float(np.dot(valid, w_slice))

    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: col.rolling(w, min_periods=1).apply(_weighted_avg, raw=True))
    return x.rolling(w, min_periods=1).apply(_weighted_avg, raw=True)


# -----------------------------------------------------------------------
# Cross-sectional operators (operate across all assets at one timestamp)
# -----------------------------------------------------------------------

def cs_rank(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Cross-sectional percentile rank across assets.

    For a :class:`pd.Series` (one timestamp, index = symbols), returns
    ranks in [0, 1].  For a :class:`pd.DataFrame` (index = timestamps,
    columns = symbols), ranks each row independently.
    """
    if isinstance(x, pd.DataFrame):
        return x.rank(axis=1, pct=True)
    return x.rank(pct=True)


def cs_zscore(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Cross-sectional z-score: ``(x - mean) / std`` across assets."""
    if isinstance(x, pd.DataFrame):
        mean = x.mean(axis=1)
        std = x.std(axis=1) + 1e-12
        return x.sub(mean, axis=0).div(std, axis=0)
    mean = x.mean()
    std = x.std() + 1e-12
    return (x - mean) / std


def cs_scale(
    x: pd.Series | pd.DataFrame,
    target: float = 1.0,
) -> pd.Series | pd.DataFrame:
    """Scale so that ``sum(abs(x)) = target`` across assets."""
    if isinstance(x, pd.DataFrame):
        abs_sum = x.abs().sum(axis=1) + 1e-12
        return x.div(abs_sum, axis=0) * target
    abs_sum = x.abs().sum() + 1e-12
    return x / abs_sum * target


def cs_demean(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Subtract the cross-sectional mean (market-neutralize)."""
    if isinstance(x, pd.DataFrame):
        return x.sub(x.mean(axis=1), axis=0)
    return x - x.mean()


# -----------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------

def sign(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Element-wise sign: -1, 0, or +1."""
    return np.sign(x)


def signedpower(
    x: pd.Series | pd.DataFrame,
    e: float,
) -> pd.Series | pd.DataFrame:
    """``sign(x) * abs(x)^e`` — preserves sign while applying power."""
    return np.sign(x) * np.abs(x) ** e
