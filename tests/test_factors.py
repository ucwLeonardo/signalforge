"""Tests for the alpha factor module."""

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a realistic 500-row OHLCV DataFrame."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    opn = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.lognormal(15, 0.5, n)

    return pd.DataFrame({
        "timestamp": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }).set_index("timestamp")


@pytest.fixture
def multi_ohlcv(sample_ohlcv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Generate OHLCV for 5 symbols."""
    np.random.seed(42)
    result = {"SYM_A": sample_ohlcv}
    for sym in ["SYM_B", "SYM_C", "SYM_D", "SYM_E"]:
        noise = 1 + np.random.normal(0, 0.01, len(sample_ohlcv))
        df = sample_ohlcv.copy()
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * noise
        df["volume"] = df["volume"] * (1 + np.random.normal(0, 0.1, len(df)))
        result[sym] = df
    return result


# -----------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------

class TestOperators:
    def test_delay(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.operators import delay
        close = sample_ohlcv["close"]
        delayed = delay(close, 5)
        assert delayed.iloc[5] == pytest.approx(close.iloc[0], rel=1e-10)
        assert pd.isna(delayed.iloc[0])

    def test_delta(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.operators import delta
        close = sample_ohlcv["close"]
        d = delta(close, 5)
        expected = close.iloc[10] - close.iloc[5]
        assert d.iloc[10] == pytest.approx(expected, rel=1e-10)

    def test_ts_mean(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.operators import ts_mean
        close = sample_ohlcv["close"]
        ma = ts_mean(close, 20)
        expected = close.iloc[:20].mean()
        assert ma.iloc[19] == pytest.approx(expected, rel=1e-6)

    def test_ts_rank(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.operators import ts_rank
        close = sample_ohlcv["close"]
        ranked = ts_rank(close, 20)
        # ts_rank should return values in [0, 1]
        valid = ranked.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_ts_corr(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.operators import ts_corr
        close = sample_ohlcv["close"]
        volume = sample_ohlcv["volume"]
        corr = ts_corr(close, volume, 20)
        valid = corr.dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_decay_linear(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.operators import decay_linear
        close = sample_ohlcv["close"]
        dl = decay_linear(close, 10)
        # Should not be all NaN
        assert dl.dropna().shape[0] > 400

    def test_cs_rank(self) -> None:
        from signalforge.factors.operators import cs_rank
        # DataFrame: ranks each row
        df = pd.DataFrame({
            "A": [10.0, 20.0, 30.0],
            "B": [30.0, 10.0, 20.0],
            "C": [20.0, 30.0, 10.0],
        })
        ranked = cs_rank(df)
        # Each row should have ranks summing to ~2.0 (0.333 + 0.667 + 1.0)
        for i in range(3):
            row_sum = ranked.iloc[i].sum()
            assert row_sum == pytest.approx(2.0, abs=0.01)

    def test_cs_zscore(self) -> None:
        from signalforge.factors.operators import cs_zscore
        df = pd.DataFrame({
            "A": [10.0, 20.0, 30.0],
            "B": [30.0, 10.0, 20.0],
            "C": [20.0, 30.0, 10.0],
        })
        zscored = cs_zscore(df)
        # Each row should have mean ≈ 0
        for i in range(3):
            assert zscored.iloc[i].mean() == pytest.approx(0.0, abs=1e-6)

    def test_cs_demean(self) -> None:
        from signalforge.factors.operators import cs_demean
        df = pd.DataFrame({
            "A": [10.0, 20.0],
            "B": [30.0, 40.0],
        })
        demeaned = cs_demean(df)
        for i in range(2):
            assert demeaned.iloc[i].mean() == pytest.approx(0.0, abs=1e-6)


# -----------------------------------------------------------------------
# Library
# -----------------------------------------------------------------------

class TestLibrary:
    def test_get_applicable_factors_stock(self) -> None:
        from signalforge.factors.library import get_applicable_factors
        factors = get_applicable_factors("stock")
        # Should include momentum, volatility, etc.
        names = {f.name for f in factors}
        assert "ret_5d" in names
        assert "rsi_14" in names
        assert "adx_14" in names
        # Should NOT include crypto-only factors
        assert "funding_rate_proxy" not in names

    def test_get_applicable_factors_crypto(self) -> None:
        from signalforge.factors.library import get_applicable_factors
        factors = get_applicable_factors("crypto")
        names = {f.name for f in factors}
        assert "ret_5d" in names
        assert "funding_rate_proxy" in names
        # Stock-only factors excluded
        assert "momentum_12_1" not in names

    def test_get_applicable_factors_options(self) -> None:
        from signalforge.factors.library import get_applicable_factors
        factors = get_applicable_factors("options")
        names = {f.name for f in factors}
        assert "iv_percentile" in names
        assert "gamma_exposure_proxy" in names

    def test_get_factor_by_name(self) -> None:
        from signalforge.factors.library import get_factor_by_name
        f = get_factor_by_name("rsi_14")
        assert f is not None
        assert f.category == "mean_reversion"
        assert f.window == 14

    def test_get_factor_by_name_missing(self) -> None:
        from signalforge.factors.library import get_factor_by_name
        assert get_factor_by_name("nonexistent_factor") is None

    def test_get_categories(self) -> None:
        from signalforge.factors.library import get_categories
        cats = get_categories()
        assert "momentum" in cats
        assert "volatility" in cats
        assert "options" in cats

    def test_factor_computation(self, sample_ohlcv: pd.DataFrame) -> None:
        """Each built-in factor should compute without error on valid data."""
        from signalforge.factors.library import BUILTIN_FACTORS
        for fdef in BUILTIN_FACTORS:
            result = fdef.compute_fn(sample_ohlcv)
            assert isinstance(result, pd.Series), f"{fdef.name} did not return Series"
            assert len(result) == len(sample_ohlcv), f"{fdef.name} length mismatch"
            # Should have some non-NaN values
            non_nan = result.dropna()
            assert len(non_nan) > 0, f"{fdef.name} returned all NaN"


# -----------------------------------------------------------------------
# Compute
# -----------------------------------------------------------------------

class TestCompute:
    def test_compute_factors_stock(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.compute import compute_factors
        result = compute_factors(sample_ohlcv, asset_type="stock")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)
        # Should have many columns
        assert result.shape[1] > 20

    def test_compute_factors_crypto(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.compute import compute_factors
        result = compute_factors(sample_ohlcv, asset_type="crypto")
        assert "funding_rate_proxy" in result.columns
        assert "momentum_12_1" not in result.columns  # Stock-only

    def test_compute_factors_specific_names(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.compute import compute_factors
        result = compute_factors(
            sample_ohlcv,
            factor_names=["ret_5d", "rsi_14", "bb_width_20"],
        )
        assert set(result.columns) == {"ret_5d", "rsi_14", "bb_width_20"}

    def test_compute_factors_by_category(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.compute import compute_factors
        result = compute_factors(
            sample_ohlcv,
            categories=["momentum"],
        )
        from signalforge.factors.library import get_applicable_factors
        expected = {f.name for f in get_applicable_factors("stock", ["momentum"])}
        assert set(result.columns) == expected

    def test_compute_factors_insufficient_data(self) -> None:
        from signalforge.factors.compute import compute_factors
        tiny = pd.DataFrame({
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [100, 200],
        })
        result = compute_factors(tiny, min_rows=30)
        assert result.empty

    def test_compute_factors_with_extra(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.compute import compute_factors
        extra = [
            {"name": "custom_mom", "expression": "df['close'].pct_change(3)"},
        ]
        result = compute_factors(
            sample_ohlcv,
            factor_names=["ret_5d"],
            extra_factors=extra,
        )
        assert "ret_5d" in result.columns
        assert "custom_mom" in result.columns

    def test_compute_cross_sectional(self, multi_ohlcv: dict[str, pd.DataFrame]) -> None:
        from signalforge.factors.compute import compute_cross_sectional
        result = compute_cross_sectional(
            multi_ohlcv,
            factor_names=["ret_5d", "rsi_14"],
        )
        assert isinstance(result.index, pd.MultiIndex)
        assert "ret_5d" in result.columns
        assert "rsi_14" in result.columns
        # Values should be ranks in [0, 1]
        valid = result.dropna()
        assert valid["ret_5d"].min() >= 0.0
        assert valid["ret_5d"].max() <= 1.0


# -----------------------------------------------------------------------
# Preprocess
# -----------------------------------------------------------------------

class TestPreprocess:
    def test_winsorize(self) -> None:
        from signalforge.factors.preprocess import winsorize
        data = pd.Series([1, 2, 3, 4, 5, 100, -50])
        result = winsorize(data, lower=0.1, upper=0.9)
        assert result.max() < 100
        assert result.min() > -50

    def test_zscore_normalize(self) -> None:
        from signalforge.factors.preprocess import zscore_normalize
        data = pd.Series([10, 20, 30, 40, 50])
        result = zscore_normalize(data)
        assert result.mean() == pytest.approx(0.0, abs=1e-10)
        assert result.std() == pytest.approx(1.0, abs=0.2)

    def test_market_neutralize(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.compute import compute_factors
        from signalforge.factors.preprocess import market_neutralize
        factors = compute_factors(sample_ohlcv, factor_names=["ret_5d", "ret_20d"])
        result = market_neutralize(factors)
        # Neutralized factors should have ~zero mean
        assert result["ret_5d"].mean() == pytest.approx(0.0, abs=0.01)

    def test_preprocess_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.compute import compute_factors
        from signalforge.factors.preprocess import preprocess_pipeline
        factors = compute_factors(sample_ohlcv, factor_names=["ret_5d", "rsi_14"])
        result = preprocess_pipeline(factors, steps=["winsorize", "zscore"])
        # Should have same shape
        assert result.shape == factors.shape
        # Z-scored values should have reasonable range
        for col in result.columns:
            valid = result[col].dropna()
            if len(valid) > 10:
                assert valid.std() < 5.0  # Not wildly off


# -----------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------

class TestEvaluate:
    def test_information_coefficient(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.evaluate import information_coefficient
        close = sample_ohlcv["close"]
        factor = close.pct_change(5)
        fwd_ret = close.pct_change(5).shift(-5)
        ic = information_coefficient(factor, fwd_ret)
        assert -1.0 <= ic <= 1.0

    def test_information_ratio(self) -> None:
        from signalforge.factors.evaluate import information_ratio
        # Consistent positive IC → high IR
        ic_ts = pd.Series([0.05, 0.06, 0.04, 0.05, 0.07, 0.03, 0.05, 0.06, 0.04, 0.05])
        ir = information_ratio(ic_ts)
        assert ir > 0.0

    def test_factor_turnover(self) -> None:
        from signalforge.factors.evaluate import factor_turnover
        today = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=["A", "B", "C", "D", "E"])
        yesterday = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=["A", "B", "C", "D", "E"])
        turnover = factor_turnover(today, yesterday)
        assert turnover >= 0.0
        # Complete reversal should have high turnover
        assert turnover > 0.3

    def test_ic_decay(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.evaluate import ic_decay
        close = sample_ohlcv["close"]
        factor = close.pct_change(5)
        decay = ic_decay(factor, close, lags=(1, 5, 10))
        assert set(decay.keys()) == {1, 5, 10}
        for v in decay.values():
            assert -1.0 <= v <= 1.0

    def test_factor_fitness(self) -> None:
        from signalforge.factors.evaluate import factor_fitness
        fitness = factor_fitness(sharpe=1.5, total_return=0.1, turnover=0.2)
        assert fitness > 0.0

    def test_evaluate_single_factor(self, sample_ohlcv: pd.DataFrame) -> None:
        from signalforge.factors.evaluate import evaluate_single_factor
        close = sample_ohlcv["close"]
        factor = close.pct_change(5)
        fwd_ret = close.pct_change(5).shift(-5)
        metrics = evaluate_single_factor(factor, fwd_ret, close)
        assert "ic" in metrics
        assert "ir" in metrics
        assert "sharpe" in metrics
        assert "turnover" in metrics
        assert "fitness" in metrics

    def test_factor_report(self) -> None:
        from signalforge.factors.evaluate import factor_report
        metrics = {
            "ic": 0.05,
            "ir": 0.8,
            "spread_return": 0.001,
            "sharpe": 1.2,
            "turnover": 0.15,
            "decay": {1: 0.05, 5: 0.03, 10: 0.01},
            "fitness": 0.9,
        }
        report = factor_report("test_factor", metrics)
        assert "test_factor" in report
        assert "ACCEPT" in report
