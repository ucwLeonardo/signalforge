"""Qlib factor prediction engine.

Wraps Microsoft Qlib's ML pipeline for alpha factor prediction.  When the
``qlib`` package is not installed the engine falls back to a lightweight
momentum/mean-reversion factor model built on top of pandas, numpy and
scikit-learn so that SignalForge remains fully usable without Qlib or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from signalforge.engines.base import PredictionEngine

# ---------------------------------------------------------------------------
# Qlib lazy import
# ---------------------------------------------------------------------------

_QLIB_AVAILABLE: bool = False

try:
    import qlib  # type: ignore[import-untyped]
    from qlib.contrib.model.gbdt import LGBModel  # type: ignore[import-untyped]

    _QLIB_AVAILABLE = True
    logger.info("Qlib library detected -- ML factor predictions available.")
except ImportError:
    logger.warning(
        "Qlib library not found. Install it with:\n"
        "  pip install pyqlib\n"
        "Falling back to momentum/mean-reversion factor model."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QlibEngineConfig:
    """All tuneable knobs for :class:`QlibEngine`.

    Attributes:
        enabled: Whether the engine is active.
        model_type: ML model backend -- ``"lgbm"``, ``"lstm"``, or
            ``"transformer"``.
        features: Qlib feature set name (``"alpha158"`` or ``"alpha360"``).
        label_horizon: Number of trading days used to compute forward return
            labels.
        device: ``"cuda"`` or ``"cpu"`` (relevant for neural backends).
        qlib_data_dir: Path to the local Qlib data bundle.
        region: Market region passed to ``qlib.init`` (``"us"`` or ``"cn"``).
    """

    enabled: bool = False
    model_type: str = "lgbm"
    features: str = "alpha158"
    label_horizon: int = 5
    device: str = "cuda"
    qlib_data_dir: str = "~/.qlib/qlib_data/us_data"
    region: str = "us"


# ---------------------------------------------------------------------------
# Helpers -- shared
# ---------------------------------------------------------------------------


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose index is a proper :class:`DatetimeIndex`."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out = out.set_index("timestamp")
        out.index = pd.DatetimeIndex(out.index)
    return out


def _infer_freq(df: pd.DataFrame) -> str:
    """Best-effort frequency inference from an OHLCV DataFrame."""
    idx = (
        df.index
        if isinstance(df.index, pd.DatetimeIndex)
        else pd.DatetimeIndex(df["timestamp"])
    )
    inferred = pd.infer_freq(idx)
    if inferred is not None:
        return inferred
    deltas = idx.to_series().diff().dropna()
    median_delta = deltas.median()
    if median_delta <= pd.Timedelta(minutes=5):
        return "5min"
    if median_delta <= pd.Timedelta(hours=1):
        return "1h"
    if median_delta <= pd.Timedelta(days=1):
        return "1D"
    return "1W"


def _generate_future_timestamps(
    last_ts: pd.Timestamp,
    freq: str | pd.DateOffset,
    count: int,
) -> pd.DatetimeIndex:
    """Create *count* timestamps starting right after *last_ts*."""
    return pd.date_range(start=last_ts, periods=count + 1, freq=freq)[1:]


# ---------------------------------------------------------------------------
# Fallback: momentum / mean-reversion factor model
# ---------------------------------------------------------------------------


def _compute_factor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive momentum and mean-reversion factors from OHLCV data.

    Features produced (all computed from the *close* and *volume* series):
        - ret_5: 5-day simple return.
        - ret_20: 20-day simple return.
        - ma_ratio_5_20: Ratio of 5-day moving average to 20-day MA.
        - vol_ratio: Ratio of 5-day realised volatility to 20-day volatility.
        - volume_ratio: Ratio of 5-day average volume to 20-day average volume.
    """
    close = df["close"].astype(np.float64)
    volume = df["volume"].astype(np.float64)

    log_ret = np.log(close / close.shift(1))

    features = pd.DataFrame(index=df.index)
    features["ret_5"] = close.pct_change(5)
    features["ret_20"] = close.pct_change(20)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    features["ma_ratio_5_20"] = ma5 / ma20

    vol5 = log_ret.rolling(5).std()
    vol20 = log_ret.rolling(20).std()
    features["vol_ratio"] = vol5 / (vol20 + 1e-12)

    avg_vol5 = volume.rolling(5).mean()
    avg_vol20 = volume.rolling(20).mean()
    features["volume_ratio"] = avg_vol5 / (avg_vol20 + 1e-12)

    return features


def _fallback_predict(
    df: pd.DataFrame,
    pred_len: int,
    label_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a Ridge regression on momentum/mean-reversion factors and return
    predicted returns together with a confidence array.

    Returns
    -------
    predicted_returns : ndarray of shape ``(pred_len,)``
    confidences : ndarray of shape ``(pred_len,)`` in [0, 1]
    """
    from sklearn.linear_model import Ridge  # noqa: E402 -- deferred import

    features = _compute_factor_features(df)
    close = df["close"].astype(np.float64)

    # Label: forward return over *label_horizon* days
    forward_return = close.shift(-label_horizon) / close - 1.0

    combined = features.copy()
    combined["label"] = forward_return

    # Drop rows where features or label are NaN
    combined = combined.dropna()

    if len(combined) < 30:
        logger.warning(
            "Insufficient data for fallback model ({} rows after dropping NaN). "
            "Returning zero predictions.",
            len(combined),
        )
        return np.zeros(pred_len), np.full(pred_len, 0.1)

    feature_cols = [c for c in combined.columns if c != "label"]
    X = combined[feature_cols].values
    y = combined["label"].values

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Use the most recent available features to forecast iteratively.  Since
    # we only have the latest row of features and the horizon may exceed one
    # step, we repeat the last observation with small decay for each step.
    latest_features = features.iloc[-1:].values  # shape (1, n_features)

    predictions: list[float] = []
    for step in range(pred_len):
        # Apply a gentle temporal decay so that farther-ahead predictions
        # regress toward zero (reflecting increased uncertainty).
        decay = 0.95 ** step
        pred_return: float = float(model.predict(latest_features)[0]) * decay
        predictions.append(pred_return)

    predicted_returns = np.array(predictions, dtype=np.float64)

    # Confidence: based on training R^2, decayed by step distance
    train_pred = model.predict(X)
    ss_res = np.sum((y - train_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
    base_confidence = 0.3 + 0.5 * r2  # map R^2 [0,1] -> confidence [0.3, 0.8]

    confidences = np.array(
        [float(np.clip(base_confidence * (0.95 ** step), 0.05, 0.95)) for step in range(pred_len)],
        dtype=np.float64,
    )

    return predicted_returns, confidences


# ---------------------------------------------------------------------------
# Qlib-backed prediction
# ---------------------------------------------------------------------------


def _qlib_predict(
    df: pd.DataFrame,
    config: QlibEngineConfig,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Qlib's Alpha158 + LGBModel pipeline and return predicted returns
    with confidence scores.

    Returns
    -------
    predicted_returns : ndarray of shape ``(pred_len,)``
    confidences : ndarray of shape ``(pred_len,)`` in [0, 1]
    """
    import qlib  # type: ignore[import-untyped]
    from qlib.contrib.data.handler import Alpha158 as Alpha158Handler  # type: ignore[import-untyped]
    from qlib.contrib.model.gbdt import LGBModel  # type: ignore[import-untyped]
    from qlib.data.dataset import DatasetH, TSDatasetH  # type: ignore[import-untyped]
    from qlib.data.dataset.handler import DataHandlerLP  # type: ignore[import-untyped]

    # Ensure qlib is initialised
    try:
        qlib.init(
            provider_uri=config.qlib_data_dir,
            region=config.region,
        )
        logger.info(
            "Qlib initialised (data_dir={}, region={}).",
            config.qlib_data_dir,
            config.region,
        )
    except Exception:
        # qlib.init raises if called twice; safe to ignore
        logger.debug("Qlib already initialised -- skipping init.")

    horizon = config.label_horizon

    # Build label expression: actual forward return (NOT rank-normalised)
    label_expr = [f"Ref($close, -{horizon}) / $close - 1"]
    label_name = ["LABEL0"]

    # Determine date range from the input DataFrame
    prepared = _ensure_datetime_index(df)
    start_date = str(prepared.index.min().date())
    end_date = str(prepared.index.max().date())

    # Alpha158 DataHandler WITHOUT CSRankNorm to preserve absolute returns
    handler_config = {
        "start_time": start_date,
        "end_time": end_date,
        "instruments": "all",
        "label": (label_expr, label_name),
        "infer_processors": [],  # no rank normalisation
        "learn_processors": [],
    }

    handler = Alpha158Handler(**handler_config)

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (start_date, end_date),
            "test": (start_date, end_date),
        },
    )

    # Select model backend
    if config.model_type == "lgbm":
        model = LGBModel(loss="mse")
    elif config.model_type == "lstm":
        from qlib.contrib.model.pytorch_lstm import LSTM  # type: ignore[import-untyped]

        model = LSTM(d_feat=158 if config.features == "alpha158" else 360, device=config.device)
    elif config.model_type == "transformer":
        from qlib.contrib.model.pytorch_transformer import Transformer  # type: ignore[import-untyped]

        model = Transformer(d_feat=158 if config.features == "alpha158" else 360, device=config.device)
    else:
        logger.warning(
            "Unknown model_type '{}', falling back to lgbm.",
            config.model_type,
        )
        model = LGBModel(loss="mse")

    model.fit(dataset)
    raw_predictions: pd.Series = model.predict(dataset)

    # raw_predictions is a Series indexed by (instrument, datetime).  We take
    # the last *pred_len* values as our forward-looking predictions.
    if isinstance(raw_predictions.index, pd.MultiIndex):
        # Pick latest instrument group
        last_instrument = raw_predictions.index.get_level_values(0)[-1]
        instrument_preds = raw_predictions.xs(last_instrument, level=0)
    else:
        instrument_preds = raw_predictions

    instrument_preds = instrument_preds.sort_index()

    # Take the tail -- these are the model's most recent predictions
    tail = instrument_preds.tail(pred_len).values.astype(np.float64)

    # Pad if Qlib returned fewer predictions than requested
    if len(tail) < pred_len:
        pad = np.full(pred_len - len(tail), tail[-1] if len(tail) > 0 else 0.0)
        tail = np.concatenate([tail, pad])

    predicted_returns = tail[:pred_len]

    # Confidence heuristic: inverse of prediction variance across recent window
    recent_std = float(np.std(predicted_returns) + 1e-12)
    base_conf = float(np.clip(1.0 / (1.0 + 10.0 * recent_std), 0.2, 0.9))
    confidences = np.array(
        [float(np.clip(base_conf * (0.97 ** step), 0.05, 0.95)) for step in range(pred_len)],
        dtype=np.float64,
    )

    return predicted_returns, confidences


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class QlibEngine(PredictionEngine):
    """Prediction engine backed by Microsoft Qlib's alpha factor pipeline.

    When the ``qlib`` package is importable the engine trains a gradient-boosted
    tree (or neural) model on Alpha158 features and predicts forward returns.
    Otherwise it transparently falls back to a momentum / mean-reversion factor
    model using scikit-learn Ridge regression.
    """

    def __init__(
        self,
        config: QlibEngineConfig | None = None,
        **overrides: Any,
    ) -> None:
        cfg = config or QlibEngineConfig()
        if overrides:
            cfg = QlibEngineConfig(
                **{
                    fld.name: overrides.get(fld.name, getattr(cfg, fld.name))
                    for fld in cfg.__dataclass_fields__.values()
                }
            )
        self._config = cfg
        self._qlib_initialised = False

        logger.debug(
            "QlibEngine initialised (qlib_available={}, model_type={}, features={}).",
            _QLIB_AVAILABLE,
            cfg.model_type,
            cfg.features,
        )

    # -- public API ---------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return "qlib"

    @property
    def config(self) -> QlibEngineConfig:
        return self._config

    @property
    def is_qlib_available(self) -> bool:  # noqa: D401
        return _QLIB_AVAILABLE

    def predict(self, df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        """Generate predicted OHLCV candles using alpha-factor predictions.

        Parameters
        ----------
        df:
            Historical OHLCV DataFrame.  Must contain ``open, high, low,
            close, volume`` columns and a datetime-like index or
            ``timestamp`` column.
        pred_len:
            Number of future candles to predict.

        Returns
        -------
        pd.DataFrame
            Predicted candles with columns ``timestamp, open, high, low,
            close, volume, predicted_return, confidence``.
        """
        required_cols = {"open", "high", "low", "close", "volume"}
        col_names = {c.lower() for c in df.columns}
        if df.index.name is not None:
            col_names.add(df.index.name.lower())
        missing = required_cols - col_names
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        prepared = _ensure_datetime_index(df)

        # Dispatch to Qlib or fallback
        try:
            if _QLIB_AVAILABLE and self._config.enabled:
                logger.info("Running Qlib {} model with {} features.", self._config.model_type, self._config.features)
                predicted_returns, confidences = _qlib_predict(
                    prepared, self._config, pred_len
                )
            else:
                if not _QLIB_AVAILABLE:
                    logger.info("Qlib not available -- using fallback factor model.")
                elif not self._config.enabled:
                    logger.info("QlibEngine disabled in config -- using fallback factor model.")
                predicted_returns, confidences = _fallback_predict(
                    prepared, pred_len, self._config.label_horizon
                )
        except Exception:
            logger.exception("Qlib prediction failed -- falling back to factor model.")
            predicted_returns, confidences = _fallback_predict(
                prepared, pred_len, self._config.label_horizon
            )

        # Convert predicted returns into price forecasts
        last_close = float(prepared["close"].iloc[-1])
        last_open = float(prepared["open"].iloc[-1])
        last_high = float(prepared["high"].iloc[-1])
        last_low = float(prepared["low"].iloc[-1])
        last_volume = float(prepared["volume"].iloc[-1])

        # Historical volatility for high/low estimation
        log_returns = np.log(
            prepared["close"].astype(np.float64)
            / prepared["close"].astype(np.float64).shift(1)
        ).dropna()
        hist_volatility = float(log_returns.std()) if len(log_returns) > 1 else 0.02

        # Historical high-low spread as fraction of close
        hl_spread = (
            (prepared["high"] - prepared["low"]) / prepared["close"]
        ).rolling(20).mean()
        avg_hl_spread = float(hl_spread.iloc[-1]) if not np.isnan(hl_spread.iloc[-1]) else 0.03

        # Build predicted OHLCV candles step-by-step
        timestamps_list: list[pd.Timestamp] = []
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[float] = []

        freq = _infer_freq(prepared)
        future_ts = _generate_future_timestamps(prepared.index[-1], freq, pred_len)

        current_close = last_close
        for step in range(pred_len):
            step_return = float(predicted_returns[step])
            predicted_close = current_close * (1.0 + step_return)

            # Open: previous close (standard candle assumption)
            predicted_open = current_close

            # High/Low: use historical spread scaled by volatility
            step_spread = avg_hl_spread * (1.0 + hist_volatility * np.sqrt(step + 1))
            half_spread = predicted_close * step_spread / 2.0
            predicted_high = max(predicted_open, predicted_close) + half_spread
            predicted_low = min(predicted_open, predicted_close) - half_spread

            # Ensure low is non-negative for price data
            predicted_low = max(predicted_low, 0.0)

            # Volume: decay toward recent average
            vol_decay = 0.98 ** step
            predicted_volume = max(last_volume * vol_decay, 0.0)

            opens.append(predicted_open)
            highs.append(predicted_high)
            lows.append(predicted_low)
            closes.append(predicted_close)
            volumes.append(predicted_volume)

            # Next step starts from this close
            current_close = predicted_close

        result = pd.DataFrame(
            {
                "timestamp": future_ts,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "predicted_return": predicted_returns.tolist(),
                "confidence": confidences.tolist(),
            }
        )
        return result
