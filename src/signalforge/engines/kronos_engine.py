"""Kronos foundation-model prediction engine.

Wraps the Kronos time-series foundation model from HuggingFace.  If the
``kronos`` package is not available on the Python path the engine falls back
to a simple linear-regression baseline so that SignalForge remains usable
without a GPU or the Kronos repo cloned locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from signalforge.engines.base import PredictionEngine

# ---------------------------------------------------------------------------
# Kronos lazy import
# ---------------------------------------------------------------------------

_KRONOS_AVAILABLE: bool = False

try:
    from kronos import Kronos, KronosPredictor, KronosTokenizer  # type: ignore[import-untyped]

    _KRONOS_AVAILABLE = True
    logger.info("Kronos library detected -- GPU-accelerated predictions available.")
except ImportError:
    logger.warning(
        "Kronos library not found. Install it by cloning the repository:\n"
        "  git clone https://github.com/NeoQuasar/Kronos.git\n"
        "  cd Kronos && pip install -e .\n"
        "Falling back to linear-regression baseline."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KronosConfig:
    """All tuneable knobs for :class:`KronosEngine`.

    Attributes:
        model_name: HuggingFace model identifier.
        tokenizer_name: HuggingFace tokenizer identifier.
        pred_len: Default prediction horizon (number of candles).
        temperature: Sampling temperature for the generative model.
        top_p: Nucleus sampling threshold.
        sample_count: Number of samples to draw (median is returned).
        device: ``"cuda"`` or ``"cpu"``; ``"auto"`` picks automatically.
    """

    model_name: str = "NeoQuasar/Kronos-base"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    pred_len: int = 24
    temperature: float = 0.7
    top_p: float = 0.9
    sample_count: int = 20
    device: str = "auto"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    """Return an explicit ``'cuda'`` or ``'cpu'`` string."""
    if device != "auto":
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
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df["timestamp"])
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
# Fallback baseline
# ---------------------------------------------------------------------------


def _linear_regression_baseline(
    series: np.ndarray,
    pred_len: int,
) -> np.ndarray:
    """Predict *pred_len* future values via OLS on the last ``4 * pred_len``
    observations (or all available data if shorter).
    """
    lookback = min(len(series), pred_len * 4)
    y = series[-lookback:].astype(np.float64)
    x = np.arange(len(y), dtype=np.float64)

    # OLS: y = a + b*x
    x_mean = x.mean()
    y_mean = y.mean()
    b = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-12)
    a = y_mean - b * x_mean

    future_x = np.arange(len(y), len(y) + pred_len, dtype=np.float64)
    return a + b * future_x


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class KronosEngine(PredictionEngine):
    """Prediction engine backed by the Kronos foundation model.

    When the ``kronos`` library is importable the engine loads the model and
    tokenizer from HuggingFace and runs probabilistic generation.  Otherwise
    it transparently falls back to a simple linear-regression baseline.
    """

    def __init__(
        self,
        config: KronosConfig | None = None,
        **overrides: Any,
    ) -> None:
        cfg = config or KronosConfig()
        # Allow per-field overrides via kwargs
        if overrides:
            cfg = KronosConfig(
                **{
                    fld.name: overrides.get(fld.name, getattr(cfg, fld.name))
                    for fld in cfg.__dataclass_fields__.values()
                }
            )
        self._config = cfg
        self._device = _resolve_device(cfg.device)
        self._predictor: Any | None = None  # lazy-loaded

        logger.debug(
            "KronosEngine initialised (kronos_available={}, device={})",
            _KRONOS_AVAILABLE,
            self._device,
        )

    # -- public API ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "kronos"

    @property
    def config(self) -> KronosConfig:
        return self._config

    @property
    def is_kronos_available(self) -> bool:
        return _KRONOS_AVAILABLE

    def predict(self, df: pd.DataFrame, pred_len: int | None = None) -> pd.DataFrame:
        """Generate predicted OHLCV candles.

        Parameters
        ----------
        df:
            Historical OHLCV dataframe.  Must contain ``open, high, low,
            close, volume`` columns.
        pred_len:
            Prediction horizon; defaults to ``config.pred_len``.

        Returns
        -------
        pd.DataFrame
            Predicted candles with columns
            ``timestamp, open, high, low, close, volume``.
        """
        horizon = pred_len if pred_len is not None else self._config.pred_len

        required_cols = {"open", "high", "low", "close", "volume"}
        col_names = set(df.columns)
        if df.index.name is not None:
            col_names.add(df.index.name)
        missing = required_cols - {c.lower() for c in col_names}
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        prepared = _ensure_datetime_index(df)
        freq = _infer_freq(prepared)
        future_ts = _generate_future_timestamps(prepared.index[-1], freq, horizon)

        if _KRONOS_AVAILABLE:
            predictions = self._predict_kronos(prepared, horizon)
        else:
            predictions = self._predict_baseline(prepared, horizon)

        result = pd.DataFrame(
            {
                "timestamp": future_ts,
                "open": predictions["open"],
                "high": predictions["high"],
                "low": predictions["low"],
                "close": predictions["close"],
                "volume": predictions["volume"],
            }
        )
        return result

    # -- private helpers ----------------------------------------------------

    def _load_predictor(self) -> Any:
        """Lazy-load the Kronos model and tokenizer."""
        if self._predictor is not None:
            return self._predictor

        logger.info("Loading Kronos model '{}' on {}", self._config.model_name, self._device)
        model = Kronos.from_pretrained(self._config.model_name)
        tokenizer = KronosTokenizer.from_pretrained(self._config.tokenizer_name)
        self._predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=self._device,
        )
        logger.info("Kronos model loaded successfully.")
        return self._predictor

    def _predict_kronos(
        self,
        df: pd.DataFrame,
        pred_len: int,
    ) -> dict[str, np.ndarray]:
        """Run Kronos generative prediction for each OHLCV column."""
        predictor = self._load_predictor()
        results: dict[str, np.ndarray] = {}

        for col in ("open", "high", "low", "close", "volume"):
            series = df[col].values.astype(np.float64)
            logger.debug("Kronos predicting {} ({} history -> {} ahead)", col, len(series), pred_len)

            samples = predictor.predict(
                context=series,
                prediction_length=pred_len,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                num_samples=self._config.sample_count,
            )
            # samples shape: (sample_count, pred_len) -- take the median
            median_pred = np.median(samples, axis=0)

            # Volume must be non-negative
            if col == "volume":
                median_pred = np.maximum(median_pred, 0.0)

            results[col] = median_pred

        # Enforce OHLC consistency: high >= max(open, close), low <= min(open, close)
        results["high"] = np.maximum(results["high"], np.maximum(results["open"], results["close"]))
        results["low"] = np.minimum(results["low"], np.minimum(results["open"], results["close"]))

        return results

    def _predict_baseline(
        self,
        df: pd.DataFrame,
        pred_len: int,
    ) -> dict[str, np.ndarray]:
        """Fallback: simple linear-regression on each OHLCV column."""
        logger.info("Using linear-regression baseline (pred_len={})", pred_len)
        results: dict[str, np.ndarray] = {}

        for col in ("open", "high", "low", "close", "volume"):
            series = df[col].values
            results[col] = _linear_regression_baseline(series, pred_len)

        # Volume must be non-negative
        results["volume"] = np.maximum(results["volume"], 0.0)

        # Enforce OHLC consistency
        results["high"] = np.maximum(results["high"], np.maximum(results["open"], results["close"]))
        results["low"] = np.minimum(results["low"], np.minimum(results["open"], results["close"]))

        return results
