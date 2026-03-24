"""Amazon Chronos-2 foundation-model prediction engine.

Wraps Amazon's Chronos-2 probabilistic time-series forecasting models from
HuggingFace.  If the ``chronos-forecasting`` package is not available on the
Python path the engine falls back to Holt's linear trend exponential smoothing
with historical-volatility-based prediction intervals, so that SignalForge
remains usable without a GPU or the Chronos library installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from signalforge.engines.base import PredictionEngine

# ---------------------------------------------------------------------------
# Chronos lazy import
# ---------------------------------------------------------------------------

_CHRONOS_AVAILABLE: bool = False

try:
    from chronos import ChronosPipeline  # type: ignore[import-untyped]

    _CHRONOS_AVAILABLE = True
    logger.info("Chronos library detected -- probabilistic forecasting available.")
except ImportError:
    logger.warning(
        "chronos-forecasting library not found. Install it with:\n"
        "  pip install chronos-forecasting\n"
        "Falling back to Holt's linear trend exponential smoothing baseline."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChronosConfig:
    """All tuneable knobs for :class:`ChronosEngine`.

    Attributes:
        enabled: Whether the engine is active.
        model: HuggingFace model identifier.  Supported families are
            ``chronos-bolt-{tiny,mini,small,base}`` and
            ``chronos-t5-{tiny,small,base,large}``.
        pred_len: Default prediction horizon (number of candles).
        num_samples: Number of sample paths for probabilistic forecasting.
        device: PyTorch device string (``"cuda"``, ``"cpu"``, etc.).
        quantiles: Quantile levels for prediction intervals.
    """

    enabled: bool = False
    model: str = "amazon/chronos-bolt-base"
    pred_len: int = 5
    num_samples: int = 20
    device: str = "cuda"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    """Return an explicit device string, falling back to cpu when CUDA is
    unavailable."""
    if device != "cuda":
        return device
    try:
        import torch  # type: ignore[import-untyped]

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _generate_future_timestamps(
    last_ts: pd.Timestamp,
    freq: str | pd.DateOffset,
    count: int,
) -> pd.DatetimeIndex:
    """Create *count* timestamps starting right after *last_ts*."""
    return pd.date_range(start=last_ts, periods=count + 1, freq=freq)[1:]


def _infer_freq(df: pd.DataFrame) -> str:
    """Best-effort frequency inference from an OHLCV dataframe."""
    idx = (
        df.index
        if isinstance(df.index, pd.DatetimeIndex)
        else pd.DatetimeIndex(df["timestamp"])
    )
    inferred = pd.infer_freq(idx)
    if inferred is not None:
        return inferred
    # Fallback: median delta
    deltas = idx.to_series().diff().dropna()
    median_delta = deltas.median()
    if median_delta <= pd.Timedelta(minutes=5):
        return "5min"
    if median_delta <= pd.Timedelta(hours=1):
        return "1h"
    if median_delta <= pd.Timedelta(days=1):
        return "1D"
    return "1W"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose index is a proper :class:`DatetimeIndex`."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out = out.set_index("timestamp")
        out.index = pd.DatetimeIndex(out.index)
    return out


# ---------------------------------------------------------------------------
# Holt's linear trend exponential smoothing fallback (numpy-only)
# ---------------------------------------------------------------------------


def _holts_linear_trend(
    series: np.ndarray,
    pred_len: int,
    alpha: float = 0.3,
    beta: float = 0.1,
) -> np.ndarray:
    """Forecast *pred_len* steps using Holt's additive linear trend method.

    Parameters
    ----------
    series:
        Historical 1-D values (at least 2 observations).
    pred_len:
        Number of future steps to forecast.
    alpha:
        Level smoothing parameter in (0, 1).
    beta:
        Trend smoothing parameter in (0, 1).

    Returns
    -------
    np.ndarray
        Array of *pred_len* forecasted values.
    """
    y = series.astype(np.float64)
    n = len(y)
    if n < 2:
        # Degenerate case: repeat the single value
        return np.full(pred_len, y[-1] if n == 1 else 0.0)

    # Initialise level and trend from first two observations
    level = y[0]
    trend = y[1] - y[0]

    for t in range(1, n):
        prev_level = level
        level = alpha * y[t] + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend

    # Forecast
    forecasts = np.array(
        [level + (k + 1) * trend for k in range(pred_len)],
        dtype=np.float64,
    )
    return forecasts


def _historical_volatility(series: np.ndarray, window: int = 20) -> float:
    """Compute annualised-style standard deviation of returns over the recent
    *window* observations.  Returns a raw standard deviation suitable for
    building prediction intervals."""
    if len(series) < 3:
        return 0.0
    recent = series[-window:].astype(np.float64)
    returns = np.diff(recent) / (np.abs(recent[:-1]) + 1e-12)
    return float(np.std(returns))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ChronosEngine(PredictionEngine):
    """Prediction engine backed by Amazon Chronos-2 foundation models.

    When the ``chronos-forecasting`` library is importable the engine loads
    the pipeline from HuggingFace and runs probabilistic sample-based
    forecasting.  Otherwise it transparently falls back to Holt's linear
    trend exponential smoothing with volatility-based prediction intervals.
    """

    def __init__(
        self,
        config: ChronosConfig | None = None,
        **overrides: Any,
    ) -> None:
        cfg = config or ChronosConfig()
        # Allow per-field overrides via kwargs
        if overrides:
            cfg = ChronosConfig(
                **{
                    fld.name: overrides.get(fld.name, getattr(cfg, fld.name))
                    for fld in cfg.__dataclass_fields__.values()
                }
            )
        self._config = cfg
        self._device = _resolve_device(cfg.device)
        self._pipeline: Any | None = None  # lazy-loaded

        logger.debug(
            "ChronosEngine initialised (chronos_available={}, device={}, model={})",
            _CHRONOS_AVAILABLE,
            self._device,
            cfg.model,
        )

    # -- public API ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "chronos"

    @property
    def config(self) -> ChronosConfig:
        return self._config

    @property
    def is_chronos_available(self) -> bool:
        return _CHRONOS_AVAILABLE

    def predict(
        self, df: pd.DataFrame, pred_len: int | None = None
    ) -> pd.DataFrame:
        """Generate probabilistic close-price forecasts with prediction
        intervals, plus predicted high/low.

        Parameters
        ----------
        df:
            Historical OHLCV dataframe.  Must contain at least ``close``; if
            ``high``, ``low``, and ``volume`` are present they are also
            forecasted.
        pred_len:
            Prediction horizon; defaults to ``config.pred_len``.

        Returns
        -------
        pd.DataFrame
            Predicted candles with columns ``timestamp, predicted_close,
            predicted_high, predicted_low, close_p10, close_p50, close_p90``.
        """
        horizon = pred_len if pred_len is not None else self._config.pred_len

        if "close" not in df.columns:
            raise ValueError(
                "Input DataFrame must contain a 'close' column."
            )

        try:
            prepared = _ensure_datetime_index(df)
            freq = _infer_freq(prepared)
            future_ts = _generate_future_timestamps(
                prepared.index[-1], freq, horizon
            )

            if _CHRONOS_AVAILABLE:
                predictions = self._predict_chronos(prepared, horizon)
            else:
                predictions = self._predict_fallback(prepared, horizon)

            result = pd.DataFrame(
                {
                    "timestamp": future_ts,
                    "predicted_close": predictions["predicted_close"],
                    "predicted_high": predictions["predicted_high"],
                    "predicted_low": predictions["predicted_low"],
                    "close_p10": predictions["close_p10"],
                    "close_p50": predictions["close_p50"],
                    "close_p90": predictions["close_p90"],
                }
            )
            return result

        except Exception:
            logger.exception("ChronosEngine.predict failed")
            raise

    # -- private helpers ----------------------------------------------------

    def _load_pipeline(self) -> Any:
        """Lazy-load the Chronos pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info(
            "Loading Chronos pipeline '{}' on {}",
            self._config.model,
            self._device,
        )
        self._pipeline = ChronosPipeline.from_pretrained(
            self._config.model,
            device_map=self._device,
        )
        logger.info("Chronos pipeline loaded successfully.")
        return self._pipeline

    def _predict_chronos(
        self,
        df: pd.DataFrame,
        pred_len: int,
    ) -> dict[str, np.ndarray]:
        """Run Chronos-2 probabilistic forecasting."""
        import torch  # type: ignore[import-untyped]

        pipeline = self._load_pipeline()
        quantiles = self._config.quantiles
        results: dict[str, np.ndarray] = {}

        # --- Close price (primary forecast) --------------------------------
        close_series = df["close"].values.astype(np.float64)
        context = torch.tensor(close_series, dtype=torch.float32)
        logger.debug(
            "Chronos predicting close ({} history -> {} ahead, {} samples)",
            len(close_series),
            pred_len,
            self._config.num_samples,
        )

        # pipeline.predict returns shape (1, num_samples, pred_len)
        samples = pipeline.predict(
            context,
            prediction_length=pred_len,
            num_samples=self._config.num_samples,
        )
        # Squeeze batch dimension -> (num_samples, pred_len)
        if samples.dim() == 3:
            samples = samples.squeeze(0)
        samples_np = samples.numpy()

        # Compute quantiles along the sample axis
        for q in quantiles:
            q_label = f"close_p{int(q * 100)}"
            results[q_label] = np.quantile(samples_np, q, axis=0)

        results["predicted_close"] = np.median(samples_np, axis=0)

        # --- High / Low / Volume (separate univariate forecasts) -----------
        for col, out_key in (
            ("high", "predicted_high"),
            ("low", "predicted_low"),
        ):
            if col in df.columns:
                col_series = df[col].values.astype(np.float64)
                col_context = torch.tensor(col_series, dtype=torch.float32)
                col_samples = pipeline.predict(
                    col_context,
                    prediction_length=pred_len,
                    num_samples=self._config.num_samples,
                )
                if col_samples.dim() == 3:
                    col_samples = col_samples.squeeze(0)
                results[out_key] = np.median(col_samples.numpy(), axis=0)
            else:
                # Scale from close forecast as fallback
                results[out_key] = results["predicted_close"].copy()

        # Enforce OHLC consistency: high >= close >= low
        results["predicted_high"] = np.maximum(
            results["predicted_high"], results["predicted_close"]
        )
        results["predicted_low"] = np.minimum(
            results["predicted_low"], results["predicted_close"]
        )

        return results

    def _predict_fallback(
        self,
        df: pd.DataFrame,
        pred_len: int,
    ) -> dict[str, np.ndarray]:
        """Fallback: Holt's linear trend exponential smoothing with
        volatility-based probabilistic intervals."""
        logger.info(
            "Using Holt's linear trend fallback (pred_len={})", pred_len
        )
        results: dict[str, np.ndarray] = {}

        # --- Close price ---------------------------------------------------
        close_series = df["close"].values.astype(np.float64)
        close_forecast = _holts_linear_trend(close_series, pred_len)
        results["predicted_close"] = close_forecast

        # --- Probabilistic intervals from historical volatility ------------
        vol = _historical_volatility(close_series)
        # Scale uncertainty with forecast horizon: sigma * sqrt(h)
        horizon_scale = np.sqrt(np.arange(1, pred_len + 1, dtype=np.float64))
        # Use last close value as the magnitude reference for intervals
        magnitude = np.abs(close_forecast) + 1e-12

        for q in self._config.quantiles:
            q_label = f"close_p{int(q * 100)}"
            # Convert quantile to z-score (approximate via inverse normal)
            z = _approx_norm_ppf(q)
            results[q_label] = close_forecast + z * vol * magnitude * horizon_scale

        # --- High / Low ----------------------------------------------------
        for col, out_key in (
            ("high", "predicted_high"),
            ("low", "predicted_low"),
        ):
            if col in df.columns:
                col_series = df[col].values.astype(np.float64)
                results[out_key] = _holts_linear_trend(col_series, pred_len)
            else:
                results[out_key] = close_forecast.copy()

        # Enforce OHLC consistency
        results["predicted_high"] = np.maximum(
            results["predicted_high"], results["predicted_close"]
        )
        results["predicted_low"] = np.minimum(
            results["predicted_low"], results["predicted_close"]
        )

        return results


# ---------------------------------------------------------------------------
# Numeric utilities (numpy-only, no scipy required)
# ---------------------------------------------------------------------------


def _approx_norm_ppf(p: float) -> float:
    """Approximate inverse of the standard normal CDF using the rational
    approximation from Abramowitz & Stegun (formula 26.2.23).

    Accurate to ~4.5 * 10^-4 for 0 < p < 1.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Symmetry: work with the lower half
    if p < 0.5:
        return -_approx_norm_ppf(1.0 - p)

    # Rational approximation for 0.5 <= p < 1.0
    t = np.sqrt(-2.0 * np.log(1.0 - p))
    # Coefficients from Abramowitz & Stegun
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t**2) / (
        1.0 + d1 * t + d2 * t**2 + d3 * t**3
    )
    return float(z)
