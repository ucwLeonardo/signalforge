"""Factor preprocessing: winsorize, z-score, neutralization.

These transforms are applied cross-sectionally (across all assets at
each timestamp) to standardize factor values before evaluation or
use in a model.

Pipeline order (Udacity AI Trading methodology):
    1. Winsorize — clip outliers
    2. Neutralize — remove market/sector exposure
    3. Z-score — standardize to mean=0, std=1

Reference: Udacity AI for Trading Nanodegree, Project 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def winsorize(
    data: pd.Series | pd.DataFrame,
    lower: float = 0.025,
    upper: float = 0.975,
) -> pd.Series | pd.DataFrame:
    """Clip values to the [lower, upper] percentile range.

    Parameters
    ----------
    data:
        Factor values.  For a DataFrame, clips each column independently.
    lower:
        Lower percentile (e.g., 0.025 = 2.5th percentile).
    upper:
        Upper percentile (e.g., 0.975 = 97.5th percentile).

    Returns
    -------
    Clipped data with the same shape and index.
    """
    if isinstance(data, pd.DataFrame):
        return data.apply(lambda col: _winsorize_series(col, lower, upper))
    return _winsorize_series(data, lower, upper)


def _winsorize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore_normalize(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Cross-sectional z-score: ``(x - mean) / std``.

    For a DataFrame, normalizes each column independently.
    For use in cross-sectional context (all assets at one time),
    pass a Series indexed by symbol.
    """
    if isinstance(data, pd.DataFrame):
        mean = data.mean()
        std = data.std() + 1e-12
        return (data - mean) / std
    mean = data.mean()
    std = data.std() + 1e-12
    return (data - mean) / std


def market_neutralize(
    factor_df: pd.DataFrame,
    market_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Remove market beta exposure from each factor.

    Parameters
    ----------
    factor_df:
        DataFrame with factor columns.  Index = timestamps.
    market_returns:
        Market return series aligned with factor_df index.
        If ``None``, uses the cross-sectional mean of each factor
        column as the market component (simple demeaning).

    Returns
    -------
    Market-neutralized factor DataFrame.
    """
    result = factor_df.copy()
    if market_returns is None:
        # Simple demeaning: subtract column mean per timestamp
        for col in result.columns:
            result[col] = result[col] - result[col].mean()
        return result

    # Beta-adjusted neutralization per column
    for col in result.columns:
        factor_col = result[col].dropna()
        aligned_mkt = market_returns.reindex(factor_col.index).dropna()
        common_idx = factor_col.index.intersection(aligned_mkt.index)
        if len(common_idx) < 10:
            logger.debug("Insufficient data for market neutralization of '{}'", col)
            continue

        f = factor_col.loc[common_idx].values
        m = aligned_mkt.loc[common_idx].values
        # OLS beta
        m_dm = m - m.mean()
        beta = np.dot(m_dm, f - f.mean()) / (np.dot(m_dm, m_dm) + 1e-12)
        result.loc[common_idx, col] = f - beta * m

    return result


def sector_neutralize(
    factor_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Remove sector exposure by demeaning within each sector.

    Parameters
    ----------
    factor_df:
        DataFrame with MultiIndex ``(timestamp, symbol)`` and factor
        columns.  If flat index, assumes index = symbol.
    sector_map:
        Mapping of ``symbol -> sector_name``.

    Returns
    -------
    Sector-neutralized factor DataFrame.
    """
    result = factor_df.copy()

    if isinstance(result.index, pd.MultiIndex):
        # MultiIndex: group by (timestamp, sector)
        symbols = result.index.get_level_values("symbol")
        sectors = symbols.map(lambda s: sector_map.get(s, "unknown"))
        result["_sector"] = sectors.values
        for col in [c for c in result.columns if c != "_sector"]:
            result[col] = result.groupby(
                [result.index.get_level_values("timestamp"), "_sector"]
            )[col].transform(lambda x: x - x.mean())
        result = result.drop(columns=["_sector"])
    else:
        # Flat index = symbol
        sectors = result.index.map(lambda s: sector_map.get(s, "unknown"))
        for col in result.columns:
            group_mean = result.groupby(sectors)[col].transform("mean")
            result[col] = result[col] - group_mean

    return result


def preprocess_pipeline(
    factor_df: pd.DataFrame,
    steps: list[str] | tuple[str, ...] = ("winsorize", "zscore"),
    market_returns: pd.Series | None = None,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply a chain of preprocessing steps to factor data.

    Parameters
    ----------
    factor_df:
        Factor values (index = timestamps or MultiIndex, columns = factors).
    steps:
        Ordered list of preprocessing steps to apply.  Valid steps:
        ``"winsorize"``, ``"zscore"``, ``"market_neutralize"``,
        ``"sector_neutralize"``.
    market_returns:
        Required if ``"market_neutralize"`` is in *steps*.
    sector_map:
        Required if ``"sector_neutralize"`` is in *steps*.

    Returns
    -------
    Preprocessed factor DataFrame.
    """
    result = factor_df.copy()

    for step in steps:
        if step == "winsorize":
            result = winsorize(result)
        elif step == "zscore":
            result = zscore_normalize(result)
        elif step == "market_neutralize":
            result = market_neutralize(result, market_returns)
        elif step == "sector_neutralize":
            if sector_map is None:
                logger.warning("sector_neutralize requested but no sector_map provided — skipping")
                continue
            result = sector_neutralize(result, sector_map)
        else:
            logger.warning("Unknown preprocessing step: '{}' — skipping", step)

    return result
