"""Factor computation engine.

Computes factor values for single assets or cross-sectionally across
multiple assets.  Integrates built-in factors from :mod:`library` with
discovered factors from :class:`~signalforge.evolution.factor_registry.FactorRegistry`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from signalforge.factors import operators as op
from signalforge.factors.library import FactorDef, get_applicable_factors


def compute_factors(
    df: pd.DataFrame,
    asset_type: str = "stock",
    factor_names: list[str] | None = None,
    categories: list[str] | None = None,
    extra_factors: list[dict[str, Any]] | None = None,
    min_rows: int = 30,
) -> pd.DataFrame:
    """Compute factor values for a single asset.

    Parameters
    ----------
    df:
        OHLCV DataFrame with columns ``open, high, low, close, volume``.
    asset_type:
        Asset class: ``"stock"``, ``"crypto"``, ``"futures"``, ``"options"``.
    factor_names:
        Specific factor names to compute.  If ``None``, computes all
        applicable factors for the asset type.
    categories:
        Filter by category.  Ignored if *factor_names* is specified.
    extra_factors:
        Discovered factors from :class:`FactorRegistry` — each a dict
        with ``name`` and ``expression`` keys.
    min_rows:
        Minimum rows required in *df*.  Returns empty DataFrame if
        insufficient data.

    Returns
    -------
    pd.DataFrame
        Same index as *df*, one column per computed factor.
    """
    if len(df) < min_rows:
        logger.warning(
            "Insufficient data for factor computation ({} rows < {} min)",
            len(df),
            min_rows,
        )
        return pd.DataFrame(index=df.index)

    # Resolve which factors to compute
    if factor_names:
        from signalforge.factors.library import get_factor_by_name
        factor_defs = []
        for name in factor_names:
            fdef = get_factor_by_name(name)
            if fdef is not None:
                factor_defs.append(fdef)
            else:
                logger.debug("Factor '{}' not found in built-in library", name)
    else:
        factor_defs = get_applicable_factors(asset_type, categories)

    result = pd.DataFrame(index=df.index)

    # Compute built-in factors
    for fdef in factor_defs:
        try:
            values = fdef.compute_fn(df)
            result[fdef.name] = values
        except Exception as exc:
            logger.warning("Factor '{}' computation failed: {}", fdef.name, exc)
            result[fdef.name] = np.nan

    # Compute extra (discovered) factors via eval
    if extra_factors:
        for factor in extra_factors:
            name = factor.get("name", "unknown")
            expression = factor.get("expression", "")
            if not expression:
                continue
            try:
                values = eval(  # noqa: S307 — sandboxed to df/np/pd
                    expression,
                    {"__builtins__": {}},
                    {"df": df, "np": np, "pd": pd, "op": op},
                )
                if isinstance(values, (pd.Series, np.ndarray)):
                    result[name] = values
                else:
                    result[name] = float(values)
            except Exception as exc:
                logger.warning("Discovered factor '{}' failed: {}", name, exc)

    return result


def compute_cross_sectional(
    multi_df: dict[str, pd.DataFrame],
    asset_type: str = "stock",
    factor_names: list[str] | None = None,
    categories: list[str] | None = None,
    extra_factors: list[dict[str, Any]] | None = None,
    apply_cs_rank: bool = True,
) -> pd.DataFrame:
    """Compute factors then apply cross-sectional operators across symbols.

    Parameters
    ----------
    multi_df:
        Mapping of ``symbol -> OHLCV DataFrame``.  All DataFrames should
        share the same date index (or close to it).
    asset_type:
        Asset class for factor selection.
    factor_names:
        Specific factors to compute.  ``None`` = all applicable.
    categories:
        Filter by category.
    extra_factors:
        Discovered factors from registry.
    apply_cs_rank:
        If ``True``, apply ``cs_rank`` to each factor across symbols
        at each timestamp.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(timestamp, symbol)`` with factor columns.
        If *apply_cs_rank* is ``True``, values are percentile ranks [0, 1].
    """
    if not multi_df:
        return pd.DataFrame()

    # Step 1: Compute per-asset factors
    per_asset: dict[str, pd.DataFrame] = {}
    for symbol, df in multi_df.items():
        per_asset[symbol] = compute_factors(
            df,
            asset_type=asset_type,
            factor_names=factor_names,
            categories=categories,
            extra_factors=extra_factors,
        )

    if not per_asset:
        return pd.DataFrame()

    # Step 2: Align to common date index
    factor_cols = set()
    for fdf in per_asset.values():
        factor_cols.update(fdf.columns)
    factor_cols = sorted(factor_cols)

    if not factor_cols:
        return pd.DataFrame()

    # Build a panel: for each factor, create a DataFrame (dates × symbols)
    all_dates = sorted(
        set().union(*(fdf.index for fdf in per_asset.values()))
    )
    symbols = sorted(per_asset.keys())

    panels: dict[str, pd.DataFrame] = {}
    for col in factor_cols:
        panel = pd.DataFrame(index=all_dates, columns=symbols, dtype=np.float64)
        for sym in symbols:
            if col in per_asset[sym].columns:
                panel[sym] = per_asset[sym][col].reindex(all_dates)
        panels[col] = panel

    # Step 3: Apply cross-sectional rank if requested
    if apply_cs_rank:
        for col in factor_cols:
            panels[col] = op.cs_rank(panels[col])

    # Step 4: Reshape to MultiIndex (timestamp, symbol)
    records = []
    for date in all_dates:
        for sym in symbols:
            row = {"timestamp": date, "symbol": sym}
            for col in factor_cols:
                val = panels[col].loc[date, sym]
                row[col] = val if not pd.isna(val) else np.nan
            records.append(row)

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.set_index(["timestamp", "symbol"])

    return result
