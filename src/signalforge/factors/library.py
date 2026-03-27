"""Built-in alpha factor definitions.

Each factor is a :class:`FactorDef` with a name, category, computational
function, window size, applicable asset types, and description.

Factor categories:
    - momentum: price trend continuation / reversal
    - volatility: price dispersion and regime
    - volume: trading activity signals
    - trend: directional indicators
    - mean_reversion: overbought / oversold
    - price_action: candle-level patterns
    - options: Greeks and IV-based (options only)

Reference: WorldQuant 101 Alphas, Udacity AI Trading, Hull's Derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from signalforge.factors import operators as op


@dataclass(frozen=True)
class FactorDef:
    """Definition of a single alpha factor.

    Attributes:
        name: Unique factor identifier.
        category: Factor family (momentum, volatility, etc.).
        compute_fn: Callable that takes an OHLCV DataFrame and returns a Series.
        window: Primary rolling window used by this factor.
        asset_types: Tuple of asset types this factor applies to.
        description: Human-readable explanation.
    """

    name: str
    category: str
    compute_fn: Callable[[pd.DataFrame], pd.Series]
    window: int
    asset_types: tuple[str, ...] = ("stock", "crypto", "futures")
    description: str = ""


# -----------------------------------------------------------------------
# Helper: safe column access
# -----------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Get column as float64, case-insensitive."""
    for c in df.columns:
        if c.lower() == name.lower():
            return df[c].astype(np.float64)
    raise KeyError(f"Column '{name}' not found in DataFrame")


def _log_ret(df: pd.DataFrame) -> pd.Series:
    """Log returns from close prices."""
    close = _col(df, "close")
    return np.log(close / close.shift(1))


# -----------------------------------------------------------------------
# Momentum factors (6)
# -----------------------------------------------------------------------

def _ret_5d(df: pd.DataFrame) -> pd.Series:
    return _col(df, "close").pct_change(5)

def _ret_10d(df: pd.DataFrame) -> pd.Series:
    return _col(df, "close").pct_change(10)

def _ret_20d(df: pd.DataFrame) -> pd.Series:
    return _col(df, "close").pct_change(20)

def _ret_60d(df: pd.DataFrame) -> pd.Series:
    return _col(df, "close").pct_change(60)

def _momentum_12_1(df: pd.DataFrame) -> pd.Series:
    """12-month return minus 1-month return (skip recent month)."""
    close = _col(df, "close")
    ret_252 = close.pct_change(252)
    ret_21 = close.pct_change(21)
    return ret_252 - ret_21

def _reversal_5d(df: pd.DataFrame) -> pd.Series:
    """Short-term reversal: negative 5-day return."""
    return -_col(df, "close").pct_change(5)


# -----------------------------------------------------------------------
# Volatility factors (5)
# -----------------------------------------------------------------------

def _realized_vol_20d(df: pd.DataFrame) -> pd.Series:
    return _log_ret(df).rolling(20, min_periods=10).std()

def _vol_ratio_5_20(df: pd.DataFrame) -> pd.Series:
    lr = _log_ret(df)
    vol5 = lr.rolling(5, min_periods=3).std()
    vol20 = lr.rolling(20, min_periods=10).std()
    return vol5 / (vol20 + 1e-12)

def _atr_pct_14(df: pd.DataFrame) -> pd.Series:
    """Average True Range as percentage of close."""
    high = _col(df, "high")
    low = _col(df, "low")
    close = _col(df, "close")
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=7).mean() / (close + 1e-12)

def _bb_width_20(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band width: 2*std(20) / ma(20)."""
    close = _col(df, "close")
    ma = close.rolling(20, min_periods=10).mean()
    std = close.rolling(20, min_periods=10).std()
    return (2 * std) / (ma + 1e-12)

def _vol_regime(df: pd.DataFrame) -> pd.Series:
    """Binary volatility regime: 1 if current vol > 60d median, else 0."""
    vol20 = _log_ret(df).rolling(20, min_periods=10).std()
    median_60 = vol20.rolling(60, min_periods=30).median()
    return (vol20 > median_60).astype(np.float64)


# -----------------------------------------------------------------------
# Volume factors (4)
# -----------------------------------------------------------------------

def _volume_ratio_5_20(df: pd.DataFrame) -> pd.Series:
    vol = _col(df, "volume")
    avg5 = vol.rolling(5, min_periods=3).mean()
    avg20 = vol.rolling(20, min_periods=10).mean()
    return avg5 / (avg20 + 1e-12)

def _volume_momentum_5d(df: pd.DataFrame) -> pd.Series:
    return _col(df, "volume").pct_change(5)

def _vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """Close / VWAP ratio.  Approximates VWAP from typical price * volume."""
    close = _col(df, "close")
    high = _col(df, "high")
    low = _col(df, "low")
    volume = _col(df, "volume")
    typical = (high + low + close) / 3.0
    cum_tp_vol = (typical * volume).rolling(20, min_periods=5).sum()
    cum_vol = volume.rolling(20, min_periods=5).sum()
    vwap = cum_tp_vol / (cum_vol + 1e-12)
    return close / (vwap + 1e-12)

def _obv_slope(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume linear regression slope over 10 periods."""
    close = _col(df, "close")
    volume = _col(df, "volume")
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    # Slope via rolling linear regression
    x = np.arange(10, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(arr: np.ndarray) -> float:
        if np.all(np.isnan(arr)):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) < 5:
            return np.nan
        xx = np.arange(len(valid), dtype=np.float64)
        xx_mean = xx.mean()
        return float(np.sum((xx - xx_mean) * (valid - np.nanmean(valid))) / (np.sum((xx - xx_mean) ** 2) + 1e-12))

    return obv.rolling(10, min_periods=5).apply(_slope, raw=True)


# -----------------------------------------------------------------------
# Trend factors (5)
# -----------------------------------------------------------------------

def _ma_cross_5_20(df: pd.DataFrame) -> pd.Series:
    close = _col(df, "close")
    ma5 = close.rolling(5, min_periods=3).mean()
    ma20 = close.rolling(20, min_periods=10).mean()
    return (ma5 - ma20) / (ma20 + 1e-12)

def _ma_cross_10_40(df: pd.DataFrame) -> pd.Series:
    close = _col(df, "close")
    ma10 = close.rolling(10, min_periods=5).mean()
    ma40 = close.rolling(40, min_periods=20).mean()
    return (ma10 - ma40) / (ma40 + 1e-12)

def _ema_cross_12_26(df: pd.DataFrame) -> pd.Series:
    close = _col(df, "close")
    ema12 = close.ewm(span=12, min_periods=6).mean()
    ema26 = close.ewm(span=26, min_periods=13).mean()
    return (ema12 - ema26) / (ema26 + 1e-12)

def _price_position_52w(df: pd.DataFrame) -> pd.Series:
    """Position within 52-week (252-day) range: [0, 1]."""
    close = _col(df, "close")
    high_252 = close.rolling(252, min_periods=60).max()
    low_252 = close.rolling(252, min_periods=60).min()
    return (close - low_252) / (high_252 - low_252 + 1e-12)

def _adx_14(df: pd.DataFrame) -> pd.Series:
    """Average Directional Index (14-period)."""
    high = _col(df, "high")
    low = _col(df, "low")
    close = _col(df, "close")

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr14 = tr.ewm(span=14, min_periods=7).mean()
    plus_di = 100 * plus_dm.ewm(span=14, min_periods=7).mean() / (atr14 + 1e-12)
    minus_di = 100 * minus_dm.ewm(span=14, min_periods=7).mean() / (atr14 + 1e-12)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.ewm(span=14, min_periods=7).mean() / 100.0  # Normalize to [0, 1]


# -----------------------------------------------------------------------
# Mean-reversion factors (4)
# -----------------------------------------------------------------------

def _rsi_14(df: pd.DataFrame) -> pd.Series:
    """RSI (14-period), normalized to [-1, 1] from [0, 100]."""
    close = _col(df, "close")
    d = close.diff()
    gain = d.where(d > 0, 0.0).rolling(14, min_periods=7).mean()
    loss = (-d.where(d < 0, 0.0)).rolling(14, min_periods=7).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return (rsi - 50.0) / 50.0  # Center at 0, range [-1, 1]

def _bb_pctb_20(df: pd.DataFrame) -> pd.Series:
    """Bollinger %B: (close - lower) / (upper - lower)."""
    close = _col(df, "close")
    ma = close.rolling(20, min_periods=10).mean()
    std = close.rolling(20, min_periods=10).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (close - lower) / (upper - lower + 1e-12)

def _distance_from_ma20(df: pd.DataFrame) -> pd.Series:
    close = _col(df, "close")
    ma20 = close.rolling(20, min_periods=10).mean()
    return (close - ma20) / (ma20 + 1e-12)

def _distance_from_ma50(df: pd.DataFrame) -> pd.Series:
    close = _col(df, "close")
    ma50 = close.rolling(50, min_periods=25).mean()
    return (close - ma50) / (ma50 + 1e-12)


# -----------------------------------------------------------------------
# Price-action factors (3)
# -----------------------------------------------------------------------

def _body_ratio(df: pd.DataFrame) -> pd.Series:
    """Candle body as fraction of full range."""
    close = _col(df, "close")
    opn = _col(df, "open")
    high = _col(df, "high")
    low = _col(df, "low")
    return (close - opn).abs() / (high - low + 1e-12)

def _gap_pct(df: pd.DataFrame) -> pd.Series:
    """Overnight gap: (open - prev_close) / prev_close."""
    close = _col(df, "close")
    opn = _col(df, "open")
    return (opn - close.shift(1)) / (close.shift(1) + 1e-12)

def _candle_pattern_score(df: pd.DataFrame) -> pd.Series:
    """Composite candle pattern score: doji, hammer, engulfing.

    Returns a score in [-1, 1] where positive = bullish patterns,
    negative = bearish patterns.
    """
    close = _col(df, "close")
    opn = _col(df, "open")
    high = _col(df, "high")
    low = _col(df, "low")
    prev_close = close.shift(1)
    prev_open = opn.shift(1)

    body = (close - opn).abs()
    full_range = high - low + 1e-12
    upper_shadow = high - pd.concat([close, opn], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, opn], axis=1).min(axis=1) - low

    score = pd.Series(0.0, index=df.index)

    # Doji: tiny body relative to range (indecision)
    is_doji = body / full_range < 0.1
    score = score + is_doji.astype(np.float64) * 0.0  # Neutral

    # Hammer (bullish): small body at top, long lower shadow
    is_hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
    score = score + is_hammer.astype(np.float64) * 0.3

    # Inverted hammer / shooting star (bearish)
    is_shooting = (upper_shadow > 2 * body) & (lower_shadow < body)
    score = score - is_shooting.astype(np.float64) * 0.3

    # Bullish engulfing
    is_bull_engulf = (close > opn) & (prev_close < prev_open) & (close > prev_open) & (opn < prev_close)
    score = score + is_bull_engulf.astype(np.float64) * 0.5

    # Bearish engulfing
    is_bear_engulf = (close < opn) & (prev_close > prev_open) & (close < prev_open) & (opn > prev_close)
    score = score - is_bear_engulf.astype(np.float64) * 0.5

    return score


# -----------------------------------------------------------------------
# Crypto-specific factors (3)
# -----------------------------------------------------------------------

def _correlation_to_btc(df: pd.DataFrame) -> pd.Series:
    """Placeholder: 30d rolling autocorrelation as proxy.

    In production, this would correlate with BTC returns passed via
    the cross-sectional compute path.  As a single-asset fallback,
    we use return autocorrelation (trend persistence).
    """
    ret = _col(df, "close").pct_change()
    return ret.rolling(30, min_periods=15).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan,
        raw=True,
    )

def _volume_spike(df: pd.DataFrame) -> pd.Series:
    """Volume spike: volume / 20d average > 2.0 → 1, else 0."""
    vol = _col(df, "volume")
    avg20 = vol.rolling(20, min_periods=10).mean()
    ratio = vol / (avg20 + 1e-12)
    return (ratio > 2.0).astype(np.float64)

def _funding_rate_proxy(df: pd.DataFrame) -> pd.Series:
    """Proxy for perpetual futures funding rate.

    Estimated from intraday price dynamics: large positive overnight
    returns + high volume suggest positive funding (longs pay shorts).
    This is an approximation — real funding rate requires exchange API.
    """
    close = _col(df, "close")
    opn = _col(df, "open")
    vol = _col(df, "volume")
    # Overnight return proxy
    overnight_ret = (opn - close.shift(1)) / (close.shift(1) + 1e-12)
    # Volume-weighted overnight return as funding proxy
    vol_z = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-12)
    return overnight_ret * (1 + vol_z.clip(-2, 2))


# -----------------------------------------------------------------------
# Options factors (5) — requires options chain data
# -----------------------------------------------------------------------

def _iv_percentile(df: pd.DataFrame) -> pd.Series:
    """IV percentile: where current IV sits vs past 252 days.

    Uses ATR-based realized vol as proxy if no IV column present.
    If the DataFrame has an 'iv' or 'implied_volatility' column, uses that.
    """
    # Try to find IV column
    for col_name in ("iv", "implied_volatility", "IV"):
        if col_name in df.columns:
            iv = df[col_name].astype(np.float64)
            return iv.rolling(252, min_periods=60).apply(
                lambda x: float(np.sum(x <= x[-1]) / len(x)) if len(x) > 0 else np.nan,
                raw=True,
            )
    # Fallback: use realized vol as proxy
    rv = _log_ret(df).rolling(20, min_periods=10).std() * np.sqrt(252)
    return rv.rolling(252, min_periods=60).apply(
        lambda x: float(np.sum(x <= x[-1]) / len(x)) if len(x) > 0 else np.nan,
        raw=True,
    )

def _iv_skew(df: pd.DataFrame) -> pd.Series:
    """IV skew proxy: ratio of short-term vol to long-term vol.

    In a full options implementation, this would be OTM put IV / ATM IV.
    Here we approximate with realized vol term structure.
    """
    lr = _log_ret(df)
    vol_5 = lr.rolling(5, min_periods=3).std() * np.sqrt(252)
    vol_60 = lr.rolling(60, min_periods=30).std() * np.sqrt(252)
    return vol_5 / (vol_60 + 1e-12) - 1.0  # 0 = flat, positive = inverted (fear)

def _term_structure_slope(df: pd.DataFrame) -> pd.Series:
    """Volatility term structure slope.

    Positive = contango (normal), negative = backwardation (fear).
    Measured as difference between long-term and short-term realized vol.
    """
    lr = _log_ret(df)
    vol_10 = lr.rolling(10, min_periods=5).std() * np.sqrt(252)
    vol_60 = lr.rolling(60, min_periods=30).std() * np.sqrt(252)
    return vol_60 - vol_10

def _gamma_exposure_proxy(df: pd.DataFrame) -> pd.Series:
    """Gamma exposure proxy from price-volume dynamics.

    High gamma environments show mean-reversion (dealers hedging).
    Detected via negative autocorrelation of returns with high volume.
    """
    ret = _col(df, "close").pct_change()
    vol = _col(df, "volume")
    vol_z = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-12)
    # Negative autocorr + high volume = gamma hedging
    autocorr = ret.rolling(10, min_periods=5).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan,
        raw=True,
    )
    return -autocorr * vol_z.clip(0, 3)

def _theta_decay_rate(df: pd.DataFrame) -> pd.Series:
    """Theta decay proxy: rate of variance reduction over time.

    Measures how quickly uncertainty resolves — faster resolution
    suggests options are expensive (high theta).
    """
    lr = _log_ret(df)
    vol_5 = lr.rolling(5, min_periods=3).var()
    vol_20 = lr.rolling(20, min_periods=10).var()
    # Ratio of short/long variance; < 1 means variance is decaying
    return vol_5 / (vol_20 + 1e-12)


# -----------------------------------------------------------------------
# Factor Registry — all built-in factors
# -----------------------------------------------------------------------

BUILTIN_FACTORS: tuple[FactorDef, ...] = (
    # Momentum
    FactorDef("ret_5d", "momentum", _ret_5d, 5, ("stock", "crypto", "futures"), "5-day simple return"),
    FactorDef("ret_10d", "momentum", _ret_10d, 10, ("stock", "crypto", "futures"), "10-day simple return"),
    FactorDef("ret_20d", "momentum", _ret_20d, 20, ("stock", "crypto", "futures"), "20-day simple return"),
    FactorDef("ret_60d", "momentum", _ret_60d, 60, ("stock", "crypto", "futures"), "60-day simple return"),
    FactorDef("momentum_12_1", "momentum", _momentum_12_1, 252, ("stock",), "12-month minus 1-month return (skip recent month)"),
    FactorDef("reversal_5d", "momentum", _reversal_5d, 5, ("stock", "crypto", "futures"), "Short-term reversal: negative 5d return"),

    # Volatility
    FactorDef("realized_vol_20d", "volatility", _realized_vol_20d, 20, ("stock", "crypto", "futures"), "20-day realized volatility"),
    FactorDef("vol_ratio_5_20", "volatility", _vol_ratio_5_20, 20, ("stock", "crypto", "futures"), "5d/20d volatility ratio"),
    FactorDef("atr_pct_14", "volatility", _atr_pct_14, 14, ("stock", "crypto", "futures"), "ATR as % of close"),
    FactorDef("bb_width_20", "volatility", _bb_width_20, 20, ("stock", "crypto", "futures"), "Bollinger Band width"),
    FactorDef("vol_regime", "volatility", _vol_regime, 60, ("stock", "crypto", "futures"), "Binary vol regime (high/low)"),

    # Volume
    FactorDef("volume_ratio_5_20", "volume", _volume_ratio_5_20, 20, ("stock", "crypto", "futures"), "5d/20d volume ratio"),
    FactorDef("volume_momentum_5d", "volume", _volume_momentum_5d, 5, ("stock", "crypto", "futures"), "5-day volume change"),
    FactorDef("vwap_deviation", "volume", _vwap_deviation, 20, ("stock", "crypto", "futures"), "Close / VWAP ratio"),
    FactorDef("obv_slope", "volume", _obv_slope, 10, ("stock", "crypto", "futures"), "OBV linear regression slope"),

    # Trend
    FactorDef("ma_cross_5_20", "trend", _ma_cross_5_20, 20, ("stock", "crypto", "futures"), "5/20 MA crossover"),
    FactorDef("ma_cross_10_40", "trend", _ma_cross_10_40, 40, ("stock", "crypto", "futures"), "10/40 MA crossover"),
    FactorDef("ema_cross_12_26", "trend", _ema_cross_12_26, 26, ("stock", "crypto", "futures"), "12/26 EMA crossover (MACD-like)"),
    FactorDef("price_position_52w", "trend", _price_position_52w, 252, ("stock",), "Position in 52-week range [0,1]"),
    FactorDef("adx_14", "trend", _adx_14, 14, ("stock", "crypto", "futures"), "Average Directional Index (trend strength)"),

    # Mean reversion
    FactorDef("rsi_14", "mean_reversion", _rsi_14, 14, ("stock", "crypto", "futures"), "RSI(14) centered at 0"),
    FactorDef("bb_pctb_20", "mean_reversion", _bb_pctb_20, 20, ("stock", "crypto", "futures"), "Bollinger %B"),
    FactorDef("distance_from_ma20", "mean_reversion", _distance_from_ma20, 20, ("stock", "crypto", "futures"), "Close distance from 20-day MA"),
    FactorDef("distance_from_ma50", "mean_reversion", _distance_from_ma50, 50, ("stock", "crypto", "futures"), "Close distance from 50-day MA"),

    # Price action
    FactorDef("body_ratio", "price_action", _body_ratio, 1, ("stock", "crypto", "futures"), "Candle body / full range"),
    FactorDef("gap_pct", "price_action", _gap_pct, 1, ("stock", "crypto", "futures"), "Overnight gap as % of close"),
    FactorDef("candle_pattern_score", "price_action", _candle_pattern_score, 2, ("stock", "crypto", "futures"), "Composite candle pattern score [-1,1]"),

    # Crypto-specific
    FactorDef("correlation_to_btc", "crypto", _correlation_to_btc, 30, ("crypto",), "Return autocorrelation (BTC corr proxy)"),
    FactorDef("volume_spike", "crypto", _volume_spike, 20, ("crypto",), "Volume spike indicator (>2x avg)"),
    FactorDef("funding_rate_proxy", "crypto", _funding_rate_proxy, 20, ("crypto",), "Perpetual funding rate proxy"),

    # Options
    FactorDef("iv_percentile", "options", _iv_percentile, 252, ("stock", "options"), "IV percentile (realized vol proxy)"),
    FactorDef("iv_skew", "options", _iv_skew, 60, ("stock", "options"), "IV skew: short/long vol ratio"),
    FactorDef("term_structure_slope", "options", _term_structure_slope, 60, ("stock", "options"), "Vol term structure slope"),
    FactorDef("gamma_exposure_proxy", "options", _gamma_exposure_proxy, 20, ("stock", "options"), "Gamma exposure from price-volume dynamics"),
    FactorDef("theta_decay_rate", "options", _theta_decay_rate, 20, ("stock", "options"), "Theta decay: variance resolution rate"),
)

# Index for fast lookup
_FACTOR_INDEX: dict[str, FactorDef] = {f.name: f for f in BUILTIN_FACTORS}
_CATEGORY_INDEX: dict[str, list[FactorDef]] = {}
for _f in BUILTIN_FACTORS:
    _CATEGORY_INDEX.setdefault(_f.category, []).append(_f)


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def get_factor_by_name(name: str) -> FactorDef | None:
    """Look up a built-in factor by name.  Returns ``None`` if not found."""
    return _FACTOR_INDEX.get(name)


def get_applicable_factors(
    asset_type: str,
    categories: list[str] | None = None,
) -> list[FactorDef]:
    """Return all built-in factors applicable to *asset_type*.

    Parameters
    ----------
    asset_type:
        One of ``"stock"``, ``"crypto"``, ``"futures"``, ``"options"``.
    categories:
        Optional filter by category names.  If ``None``, return all
        applicable factors.
    """
    result = [f for f in BUILTIN_FACTORS if asset_type in f.asset_types]
    if categories:
        result = [f for f in result if f.category in categories]
    return result


def get_categories() -> list[str]:
    """Return all available factor categories."""
    return sorted(_CATEGORY_INDEX.keys())


def list_factors_by_category(category: str) -> list[FactorDef]:
    """Return all built-in factors in a category."""
    return list(_CATEGORY_INDEX.get(category, []))
