"""LSTM sequence-to-sequence price prediction engine.

Uses a lightweight PyTorch LSTM trained on-the-fly on the asset's own history
to produce multi-step OHLCV forecasts.  Confidence intervals are derived via
MC Dropout (multiple stochastic forward passes at inference time).

No external dependencies beyond PyTorch and scikit-learn (both already required
by SignalForge core).  If PyTorch is somehow unavailable the engine falls back
to a simple exponential-weighted moving average baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from signalforge.engines.base import PredictionEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LSTMConfig:
    """Tuneable knobs for :class:`LSTMEngine`.

    Attributes:
        enabled: Whether the engine is active.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate (used for both training regularisation and
            MC Dropout at inference).
        lookback: Number of historical bars used as input sequence.
        epochs: Training epochs per prediction call.
        lr: Learning rate for Adam optimiser.
        mc_samples: Number of MC Dropout forward passes for confidence.
        device: PyTorch device string.
        model_dir: Directory for persisted model checkpoints.
        finetune_epochs: Epochs for incremental fine-tuning of a saved model.
    """

    enabled: bool = False
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lookback: int = 60
    epochs: int = 50
    lr: float = 1e-3
    mc_samples: int = 20
    device: str = "cuda"
    model_dir: str = "~/.signalforge/models/lstm"
    finetune_epochs: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    if device not in ("cuda", "auto"):
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out = out.set_index("timestamp")
        out.index = pd.DatetimeIndex(out.index)
    return out


def _infer_freq(df: pd.DataFrame) -> str:
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
    return pd.date_range(start=last_ts, periods=count + 1, freq=freq)[1:]


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------


def _build_model(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    output_size: int,
    pred_len: int,
) -> Any:
    """Build an LSTM seq2seq model.  Returns a ``torch.nn.Module``."""
    import torch
    import torch.nn as nn

    class _LSTMForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size * pred_len)
            self.output_size = output_size
            self.pred_len = pred_len

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Take last time-step hidden state
            last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
            last_hidden = self.dropout(last_hidden)
            out = self.fc(last_hidden)  # (batch, output_size * pred_len)
            return out.view(-1, self.pred_len, self.output_size)

    return _LSTMForecaster()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

FEATURE_COLS = ("open", "high", "low", "close", "volume")


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Extract and normalise OHLCV features.

    Returns an (N, 5) array of z-score normalised features.  The normalisation
    parameters (mean, std) are computed per-column.
    """
    data = np.column_stack([df[c].values.astype(np.float64) for c in FEATURE_COLS])
    return data


def _create_sequences(
    data: np.ndarray,
    lookback: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over *data* to create (input, target) pairs.

    Parameters
    ----------
    data : (N, features)
    lookback : input sequence length
    pred_len : target sequence length

    Returns
    -------
    X : (samples, lookback, features)
    Y : (samples, pred_len, features)
    """
    n = len(data)
    xs, ys = [], []
    for i in range(n - lookback - pred_len + 1):
        xs.append(data[i : i + lookback])
        ys.append(data[i + lookback : i + lookback + pred_len])
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Fallback: EWMA baseline
# ---------------------------------------------------------------------------


def _ewma_baseline(
    df: pd.DataFrame, pred_len: int
) -> dict[str, np.ndarray]:
    """Simple exponential-weighted moving average baseline."""
    results: dict[str, np.ndarray] = {}
    for col in FEATURE_COLS:
        series = df[col].values.astype(np.float64)
        # Use last EMA value as flat forecast, with gentle trend
        span = min(20, len(series))
        weights = np.exp(np.linspace(-1, 0, span))
        weights /= weights.sum()
        ema = np.dot(series[-span:], weights)

        # Add small trend from last 5 bars
        if len(series) >= 5:
            trend = (series[-1] - series[-5]) / 5.0
        else:
            trend = 0.0

        forecasts = np.array(
            [ema + trend * (k + 1) for k in range(pred_len)],
            dtype=np.float64,
        )
        if col == "volume":
            forecasts = np.maximum(forecasts, 0.0)
        results[col] = forecasts

    # Enforce OHLC consistency
    results["high"] = np.maximum(
        results["high"], np.maximum(results["open"], results["close"])
    )
    results["low"] = np.minimum(
        results["low"], np.minimum(results["open"], results["close"])
    )
    results["low"] = np.maximum(results["low"], 0.0)
    return results


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LSTMEngine(PredictionEngine):
    """LSTM-based price prediction engine.

    Trains a lightweight seq2seq LSTM on the asset's own history at prediction
    time.  Uses MC Dropout for uncertainty quantification.  Falls back to an
    EWMA baseline if PyTorch is unavailable or training data is insufficient.
    """

    def __init__(
        self,
        config: LSTMConfig | None = None,
        **overrides: Any,
    ) -> None:
        cfg = config or LSTMConfig()
        if overrides:
            cfg = LSTMConfig(
                **{
                    fld.name: overrides.get(fld.name, getattr(cfg, fld.name))
                    for fld in cfg.__dataclass_fields__.values()
                }
            )
        self._config = cfg
        self._device = _resolve_device(cfg.device)

        logger.debug(
            "LSTMEngine initialised (device={}, hidden={}, layers={}, lookback={})",
            self._device,
            cfg.hidden_size,
            cfg.num_layers,
            cfg.lookback,
        )

    @property
    def name(self) -> str:
        return "lstm"

    @property
    def config(self) -> LSTMConfig:
        return self._config

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        """Sanitize a symbol string for use as a filename."""
        return symbol.replace("/", "-")

    def save_model(
        self,
        symbol: str,
        model: Any,
        means: np.ndarray,
        stds: np.ndarray,
        config_meta: dict[str, Any],
    ) -> Path:
        """Persist model state_dict, normalization params, and metadata."""
        import torch

        model_dir = Path(self._config.model_dir).expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{self._safe_symbol(symbol)}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "means": means,
                "stds": stds,
                "config_meta": config_meta,
            },
            path,
        )
        logger.info("Saved LSTM model for {} to {}", symbol, path)
        return path

    def load_model(self, symbol: str) -> dict[str, Any] | None:
        """Load a previously saved model checkpoint, or return None."""
        import torch

        model_dir = Path(self._config.model_dir).expanduser()
        path = model_dir / f"{self._safe_symbol(symbol)}.pt"
        if not path.exists():
            return None
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        logger.info("Loaded LSTM model for {} from {}", symbol, path)
        return checkpoint

    def predict(self, df: pd.DataFrame, pred_len: int | None = None, symbol: str | None = None) -> pd.DataFrame:
        """Train on *df* and forecast *pred_len* future OHLCV candles.

        Parameters
        ----------
        symbol : optional symbol identifier used for model persistence.
            When provided, the engine will load/save models and use
            incremental fine-tuning if a checkpoint already exists.

        Returns a DataFrame with columns:
        ``timestamp, open, high, low, close, volume, confidence``.
        """
        horizon = pred_len if pred_len is not None else 5

        required_cols = {"open", "high", "low", "close", "volume"}
        col_names = {c.lower() for c in df.columns}
        if df.index.name is not None:
            col_names.add(df.index.name.lower())
        missing = required_cols - col_names
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        prepared = _ensure_datetime_index(df)
        freq = _infer_freq(prepared)
        future_ts = _generate_future_timestamps(prepared.index[-1], freq, horizon)

        min_data = self._config.lookback + horizon + 10
        if len(prepared) < min_data:
            logger.warning(
                "Insufficient data for LSTM ({} bars, need {}). Using EWMA fallback.",
                len(prepared),
                min_data,
            )
            preds = _ewma_baseline(prepared, horizon)
            return self._build_result(future_ts, preds, confidence=0.3)

        try:
            preds, confidence = self._predict_lstm(prepared, horizon, symbol=symbol)
        except Exception:
            logger.exception("LSTM prediction failed -- falling back to EWMA.")
            preds = _ewma_baseline(prepared, horizon)
            confidence = 0.3

        return self._build_result(future_ts, preds, confidence)

    def _predict_lstm(
        self,
        df: pd.DataFrame,
        pred_len: int,
        *,
        symbol: str | None = None,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Train LSTM and predict with MC Dropout confidence."""
        import torch
        import torch.nn as nn

        device = torch.device(self._device)
        cfg = self._config

        # Prepare data
        raw = _prepare_features(df)

        # Per-column normalisation
        means = raw.mean(axis=0)
        stds = raw.std(axis=0) + 1e-8
        normalised = (raw - means) / stds

        X, Y = _create_sequences(normalised, cfg.lookback, pred_len)

        if len(X) < 10:
            raise ValueError("Not enough sequences for training")

        # Train/val split: last 20% for validation
        split = max(1, int(len(X) * 0.8))
        X_train, Y_train = X[:split], Y[:split]

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)

        # Build model
        n_features = len(FEATURE_COLS)
        model = _build_model(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            output_size=n_features,
            pred_len=pred_len,
        ).to(device)

        # Determine training epochs: load saved model for fine-tuning or train from scratch
        checkpoint = self.load_model(symbol) if symbol is not None else None
        if checkpoint is not None:
            model.load_state_dict(checkpoint["state_dict"])
            train_epochs = cfg.finetune_epochs
            logger.info(
                "Fine-tuning saved model for {} ({} epochs)",
                symbol,
                train_epochs,
            )
        else:
            train_epochs = cfg.epochs
            if symbol is not None:
                logger.info(
                    "No saved model for {} -- training from scratch ({} epochs)",
                    symbol,
                    train_epochs,
                )

        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()

        # Training loop
        model.train()
        for epoch in range(train_epochs):
            optimiser.zero_grad()
            output = model(X_train_t)
            loss = loss_fn(output, Y_train_t)
            loss.backward()
            optimiser.step()

            if epoch % 20 == 0:
                logger.debug("LSTM epoch {}/{}: loss={:.6f}", epoch, train_epochs, loss.item())

        # Persist model if symbol provided
        if symbol is not None:
            config_meta = {
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "lookback": cfg.lookback,
                "pred_len": pred_len,
            }
            self.save_model(symbol, model, means, stds, config_meta)

        # Inference with MC Dropout
        # Keep dropout active for uncertainty estimation
        model.train()  # keeps dropout active
        last_seq = torch.tensor(
            normalised[-cfg.lookback:][np.newaxis, ...],
            dtype=torch.float32,
            device=device,
        )

        mc_predictions = []
        with torch.no_grad():
            for _ in range(cfg.mc_samples):
                pred = model(last_seq)  # (1, pred_len, n_features)
                mc_predictions.append(pred.cpu().numpy()[0])

        mc_array = np.array(mc_predictions)  # (mc_samples, pred_len, n_features)

        # Median prediction (denormalised)
        median_pred = np.median(mc_array, axis=0)  # (pred_len, n_features)
        denorm_pred = median_pred * stds + means

        # Confidence from MC Dropout variance
        mc_std = np.mean(np.std(mc_array, axis=0))
        confidence = float(np.clip(1.0 / (1.0 + 5.0 * mc_std), 0.15, 0.90))

        results: dict[str, np.ndarray] = {}
        for i, col in enumerate(FEATURE_COLS):
            results[col] = denorm_pred[:, i]

        # Enforce constraints
        results["volume"] = np.maximum(results["volume"], 0.0)
        results["high"] = np.maximum(
            results["high"], np.maximum(results["open"], results["close"])
        )
        results["low"] = np.minimum(
            results["low"], np.minimum(results["open"], results["close"])
        )
        results["low"] = np.maximum(results["low"], 0.0)

        return results, confidence

    @staticmethod
    def _build_result(
        timestamps: pd.DatetimeIndex,
        preds: dict[str, np.ndarray],
        confidence: float,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": preds["open"],
                "high": preds["high"],
                "low": preds["low"],
                "close": preds["close"],
                "volume": preds["volume"],
                "confidence": confidence,
            }
        )
