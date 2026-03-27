"""Alpha factor computation, evaluation, and preprocessing.

This package provides a WorldQuant-style factor infrastructure for
SignalForge, including time-series and cross-sectional operators,
a library of built-in factors, preprocessing (winsorize, z-score,
neutralization), and evaluation metrics (IC, IR, turnover, decay).

Typical usage::

    from signalforge.factors import compute_factors, preprocess_pipeline, evaluate_factor

    # Compute factors for a single asset
    factor_df = compute_factors(ohlcv_df, asset_type="stock")

    # Preprocess: winsorize + z-score
    cleaned = preprocess_pipeline(factor_df, steps=["winsorize", "zscore"])

    # Evaluate factor quality
    report = evaluate_factor("momentum_12_1", factor_df, forward_returns)
"""

from signalforge.factors.compute import compute_cross_sectional, compute_factors
from signalforge.factors.evaluate import (
    factor_fitness,
    factor_report,
    factor_turnover,
    ic_decay,
    ic_series,
    information_coefficient,
    information_ratio,
)
from signalforge.factors.library import FactorDef, get_applicable_factors, get_factor_by_name
from signalforge.factors.operators import (
    cs_demean,
    cs_rank,
    cs_scale,
    cs_zscore,
    decay_linear,
    delay,
    delta,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std,
    ts_sum,
    ts_zscore,
)
from signalforge.factors.preprocess import (
    market_neutralize,
    preprocess_pipeline,
    sector_neutralize,
    winsorize,
    zscore_normalize,
)

__all__ = [
    # compute
    "compute_factors",
    "compute_cross_sectional",
    # evaluate
    "information_coefficient",
    "information_ratio",
    "ic_series",
    "ic_decay",
    "factor_turnover",
    "factor_fitness",
    "factor_report",
    # library
    "FactorDef",
    "get_applicable_factors",
    "get_factor_by_name",
    # operators
    "delay",
    "delta",
    "ts_sum",
    "ts_mean",
    "ts_std",
    "ts_rank",
    "ts_corr",
    "ts_cov",
    "ts_min",
    "ts_max",
    "ts_argmax",
    "ts_argmin",
    "ts_zscore",
    "decay_linear",
    "cs_rank",
    "cs_zscore",
    "cs_scale",
    "cs_demean",
    # preprocess
    "winsorize",
    "zscore_normalize",
    "market_neutralize",
    "sector_neutralize",
    "preprocess_pipeline",
]
