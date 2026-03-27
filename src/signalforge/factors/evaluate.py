"""Factor quality evaluation metrics.

Implements the standard quantitative finance factor evaluation framework:
    - IC (Information Coefficient): Spearman rank correlation with forward returns
    - IR (Information Ratio): IC consistency = mean(IC) / std(IC)
    - Turnover: how much the factor ranking changes between periods
    - Decay: IC at different forward horizons
    - Fitness: WorldQuant composite metric

Reference: WorldQuant Brain, Alphalens (Quantopian).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def information_coefficient(
    factor: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """Spearman rank correlation between factor values and forward returns.

    Parameters
    ----------
    factor:
        Factor values at time *t* (one value per asset, or time series).
    forward_returns:
        Realized returns over the factor's prediction horizon.
        Must be aligned with *factor*.

    Returns
    -------
    IC value in [-1, 1].  Values > 0.03 are considered meaningful.
    """
    # Align and drop NaN
    combined = pd.DataFrame({"factor": factor, "returns": forward_returns}).dropna()
    if len(combined) < 10:
        return 0.0

    corr, _ = stats.spearmanr(combined["factor"], combined["returns"])
    return float(corr) if not np.isnan(corr) else 0.0


def ic_series(
    factor_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    factor_col: str | None = None,
) -> pd.Series:
    """Compute IC per timestamp for a factor across multiple assets.

    Parameters
    ----------
    factor_df:
        DataFrame with MultiIndex ``(timestamp, symbol)`` or
        regular index (timestamps) with asset columns.
    forward_returns_df:
        Same structure as *factor_df*, containing forward returns.
    factor_col:
        If factor_df has multiple columns, which one to evaluate.

    Returns
    -------
    pd.Series indexed by timestamp with IC values.
    """
    if isinstance(factor_df.index, pd.MultiIndex):
        return _ic_series_multiindex(factor_df, forward_returns_df, factor_col)
    return _ic_series_wide(factor_df, forward_returns_df, factor_col)


def _ic_series_multiindex(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    factor_col: str | None,
) -> pd.Series:
    timestamps = factor_df.index.get_level_values("timestamp").unique()
    ics = {}
    for ts in timestamps:
        try:
            f_slice = factor_df.loc[ts]
            r_slice = returns_df.loc[ts]
            if factor_col and factor_col in f_slice.columns:
                f_vals = f_slice[factor_col]
            elif factor_col is None and len(f_slice.columns) == 1:
                f_vals = f_slice.iloc[:, 0]
            else:
                continue
            r_vals = r_slice.iloc[:, 0] if isinstance(r_slice, pd.DataFrame) else r_slice
            ics[ts] = information_coefficient(f_vals, r_vals)
        except Exception:
            continue
    return pd.Series(ics, dtype=np.float64)


def _ic_series_wide(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    factor_col: str | None,
) -> pd.Series:
    """Wide format: index = timestamps, columns = symbols."""
    if factor_col and factor_col in factor_df.columns:
        # Single-asset time series IC (rolling)
        f = factor_df[factor_col]
        r = returns_df.iloc[:, 0] if isinstance(returns_df, pd.DataFrame) else returns_df
        # Rolling IC over 60-period windows
        ics = {}
        for i in range(60, len(f)):
            window_f = f.iloc[i - 60 : i]
            window_r = r.iloc[i - 60 : i]
            ics[f.index[i]] = information_coefficient(window_f, window_r)
        return pd.Series(ics, dtype=np.float64)

    # Multi-symbol wide format: each row is a cross-section
    common_idx = factor_df.index.intersection(returns_df.index)
    ics = {}
    for ts in common_idx:
        f_row = factor_df.loc[ts].dropna()
        r_row = returns_df.loc[ts].dropna()
        common_syms = f_row.index.intersection(r_row.index)
        if len(common_syms) >= 5:
            ics[ts] = information_coefficient(f_row[common_syms], r_row[common_syms])
    return pd.Series(ics, dtype=np.float64)


def information_ratio(ic_ts: pd.Series) -> float:
    """Information Ratio: ``mean(IC) / std(IC)``.

    Measures signal consistency.  IR > 0.5 indicates a robust factor.

    Parameters
    ----------
    ic_ts:
        Time series of IC values (from :func:`ic_series`).
    """
    clean = ic_ts.dropna()
    if len(clean) < 5:
        return 0.0
    std = clean.std()
    if std < 1e-10:
        return 0.0
    return float(clean.mean() / std)


def factor_turnover(
    factor_today: pd.Series,
    factor_yesterday: pd.Series,
) -> float:
    """Portfolio turnover between two cross-sections.

    ``sum(|w_t - w_{t-1}|) / 2`` where weights are normalized factor ranks.

    Parameters
    ----------
    factor_today:
        Factor values at time *t* (index = symbols).
    factor_yesterday:
        Factor values at time *t-1* (index = symbols).

    Returns
    -------
    Turnover in [0, 1].  0 = no change, 1 = complete reversal.
    """
    common = factor_today.index.intersection(factor_yesterday.index)
    if len(common) < 2:
        return 0.0

    # Normalize to weights (rank then scale to sum=1)
    w_today = factor_today[common].rank(pct=True)
    w_yesterday = factor_yesterday[common].rank(pct=True)

    return float((w_today - w_yesterday).abs().sum() / 2.0)


def turnover_series(
    factor_panel: pd.DataFrame,
) -> pd.Series:
    """Compute turnover for each timestamp in a wide factor panel.

    Parameters
    ----------
    factor_panel:
        DataFrame with index = timestamps, columns = symbols.

    Returns
    -------
    pd.Series of turnover values indexed by timestamp.
    """
    turnovers = {}
    for i in range(1, len(factor_panel)):
        ts = factor_panel.index[i]
        turnovers[ts] = factor_turnover(
            factor_panel.iloc[i],
            factor_panel.iloc[i - 1],
        )
    return pd.Series(turnovers, dtype=np.float64)


def ic_decay(
    factor_values: pd.Series | pd.DataFrame,
    price_series: pd.Series,
    lags: tuple[int, ...] = (1, 2, 5, 10, 20),
) -> dict[int, float]:
    """IC at different forward return horizons — measures signal persistence.

    Parameters
    ----------
    factor_values:
        Factor values (time series for a single asset).
    price_series:
        Close prices aligned with *factor_values*.
    lags:
        Forward return horizons to evaluate.

    Returns
    -------
    Dict mapping lag → IC value.  Rapidly decaying IC means the factor
    is short-lived; slow decay means it persists.
    """
    if isinstance(factor_values, pd.DataFrame):
        factor_values = factor_values.iloc[:, 0]

    result = {}
    for lag in lags:
        fwd_ret = price_series.pct_change(lag).shift(-lag)
        combined = pd.DataFrame({"f": factor_values, "r": fwd_ret}).dropna()
        if len(combined) < 20:
            result[lag] = 0.0
            continue
        result[lag] = information_coefficient(combined["f"], combined["r"])
    return result


def factor_fitness(
    sharpe: float,
    total_return: float,
    turnover: float,
) -> float:
    """WorldQuant fitness metric.

    ``fitness = sharpe * sqrt(|returns| / max(turnover, 0.125))``

    Higher is better.  Penalizes high turnover.
    """
    effective_turnover = max(abs(turnover), 0.125)
    return sharpe * np.sqrt(abs(total_return) / effective_turnover)


def evaluate_single_factor(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    price_series: pd.Series | None = None,
) -> dict[str, Any]:
    """Comprehensive single-factor evaluation.

    Parameters
    ----------
    factor_values:
        Time series of factor values for one asset.
    forward_returns:
        Forward returns aligned with factor values.
    price_series:
        Close prices for IC decay calculation.

    Returns
    -------
    Dict with keys: ic, ir, mean_return, sharpe, turnover, decay, fitness.
    """
    # IC
    ic = information_coefficient(factor_values, forward_returns)

    # Rolling IC for IR
    rolling_ics = []
    window = min(60, len(factor_values) // 3)
    if window >= 20:
        for i in range(window, len(factor_values)):
            f_win = factor_values.iloc[i - window : i]
            r_win = forward_returns.iloc[i - window : i]
            rolling_ics.append(information_coefficient(f_win, r_win))
    ic_ts = pd.Series(rolling_ics, dtype=np.float64)
    ir = information_ratio(ic_ts)

    # Factor return: top quintile - bottom quintile
    combined = pd.DataFrame({"f": factor_values, "r": forward_returns}).dropna()
    if len(combined) >= 20:
        q80 = combined["f"].quantile(0.8)
        q20 = combined["f"].quantile(0.2)
        long_ret = combined.loc[combined["f"] >= q80, "r"].mean()
        short_ret = combined.loc[combined["f"] <= q20, "r"].mean()
        spread_return = (long_ret - short_ret) if not np.isnan(long_ret) and not np.isnan(short_ret) else 0.0
    else:
        spread_return = 0.0

    # Sharpe of spread return (annualized, assuming daily)
    if len(combined) >= 20:
        daily_returns = []
        q80_mask = combined["f"] >= combined["f"].quantile(0.8)
        q20_mask = combined["f"] <= combined["f"].quantile(0.2)
        factor_ret = combined.loc[q80_mask, "r"].mean() - combined.loc[q20_mask, "r"].mean()
        sharpe = float(spread_return / (combined["r"].std() + 1e-12)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Turnover estimate
    f_shifted = factor_values.shift(1)
    common = factor_values.dropna().index.intersection(f_shifted.dropna().index)
    if len(common) > 1:
        rank_changes = (factor_values[common].rank() - f_shifted[common].rank()).abs()
        avg_turnover = float(rank_changes.mean() / len(common))
    else:
        avg_turnover = 0.0

    # IC decay
    decay = {}
    if price_series is not None:
        decay = ic_decay(factor_values, price_series)

    # Fitness
    fitness = factor_fitness(sharpe, spread_return, avg_turnover)

    return {
        "ic": round(ic, 4),
        "ir": round(ir, 4),
        "spread_return": round(spread_return, 6),
        "sharpe": round(sharpe, 4),
        "turnover": round(avg_turnover, 4),
        "decay": {k: round(v, 4) for k, v in decay.items()},
        "fitness": round(fitness, 4),
    }


def factor_report(
    name: str,
    metrics: dict[str, Any],
) -> str:
    """Format a factor evaluation report as a string.

    Parameters
    ----------
    name:
        Factor name.
    metrics:
        Dict from :func:`evaluate_single_factor`.

    Returns
    -------
    Formatted report string.
    """
    lines = [
        f"Factor: {name}",
        f"  IC:            {metrics.get('ic', 0):.4f}  {'✓' if abs(metrics.get('ic', 0)) > 0.03 else '✗'} (threshold: |IC| > 0.03)",
        f"  IR:            {metrics.get('ir', 0):.4f}  {'✓' if abs(metrics.get('ir', 0)) > 0.5 else '✗'} (threshold: |IR| > 0.5)",
        f"  Spread Return: {metrics.get('spread_return', 0):.6f}",
        f"  Sharpe:        {metrics.get('sharpe', 0):.4f}",
        f"  Turnover:      {metrics.get('turnover', 0):.4f}  {'✓' if metrics.get('turnover', 0) < 0.3 else '✗'} (threshold: < 0.3)",
        f"  Fitness:       {metrics.get('fitness', 0):.4f}",
    ]
    decay = metrics.get("decay", {})
    if decay:
        decay_str = ", ".join(f"{k}d={v:.4f}" for k, v in sorted(decay.items()))
        lines.append(f"  IC Decay:      {decay_str}")

    # Overall assessment
    ic_ok = abs(metrics.get("ic", 0)) > 0.03
    ir_ok = abs(metrics.get("ir", 0)) > 0.5
    to_ok = metrics.get("turnover", 0) < 0.3
    if ic_ok and ir_ok and to_ok:
        lines.append("  Status:        ACCEPT ✓")
    elif ic_ok:
        lines.append("  Status:        CANDIDATE (IC ok, needs more validation)")
    else:
        lines.append("  Status:        REJECT ✗")

    return "\n".join(lines)
