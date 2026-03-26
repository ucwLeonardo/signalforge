"""Gradient-boosted tree ensemble prediction engine.

Uses LightGBM (preferred) or scikit-learn's GradientBoostingRegressor as a
fallback to train a feature-rich model on the asset's own history and predict
forward returns.  Features include momentum, volatility, volume profile, and
various technical factors.

Dependencies:
    - LightGBM (optional; ``pip install lightgbm``)
    - scikit-learn (required; already a core SignalForge dependency)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from signalforge.engines.base import PredictionEngine

# ---------------------------------------------------------------------------
# LightGBM lazy import
# ---------------------------------------------------------------------------

_LGBM_AVAILABLE: bool = False

try:
    import lightgbm as lgb  # type: ignore[import-untyped]

    _LGBM_AVAILABLE = True
    logger.info("LightGBM detected -- gradient-boosted predictions available.")
except ImportError:
    logger.warning(
        "LightGBM not found. Install with: pip install lightgbm\n"
        "Falling back to sklearn GradientBoostingRegressor."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GBMConfig:
    """Tuneable knobs for :class:`GBMEnsembleEngine`.

    Attributes:
        enabled: Whether the engine is active.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        label_horizon: Forward return prediction period (days).
        feature_windows: Rolling window sizes for feature engineering.
        min_train_rows: Minimum rows required after feature computation.
    """

    enabled: bool = False
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    label_horizon: int = 5
    feature_windows: tuple[int, ...] = (5, 10, 20, 40)
    min_train_rows: int = 60


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _compute_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 20, 40),
) -> pd.DataFrame:
    """Derive a rich set of trading features from OHLCV data.

    Categories:
        - Momentum: returns over multiple horizons
        - Trend: moving average crossovers and ratios
        - Volatility: realised vol, ATR, Bollinger width
        - Volume: relative volume, volume momentum
        - Price action: candle body ratio, upper/lower shadows
    """
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    opn = df["open"].astype(np.float64)
    volume = df["volume"].astype(np.float64)

    log_ret = np.log(close / close.shift(1))
    features = pd.DataFrame(index=df.index)

    # --- Momentum features ---
    for w in windows:
        features[f"ret_{w}"] = close.pct_change(w)
        features[f"log_ret_{w}"] = log_ret.rolling(w).sum()

    # --- Trend features ---
    for w in windows:
        ma = close.rolling(w).mean()
        features[f"ma_ratio_{w}"] = close / (ma + 1e-12)

    # MA crossovers
    if len(windows) >= 2:
        short_w, long_w = windows[0], windows[-1]
        ma_short = close.rolling(short_w).mean()
        ma_long = close.rolling(long_w).mean()
        features["ma_cross"] = (ma_short - ma_long) / (ma_long + 1e-12)

    # --- Volatility features ---
    for w in windows:
        features[f"vol_{w}"] = log_ret.rolling(w).std()

    # ATR
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    for w in windows[:2]:
        features[f"atr_{w}"] = tr.rolling(w).mean() / (close + 1e-12)

    # Bollinger width
    for w in (20,):
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        features[f"bb_width_{w}"] = (2 * std) / (ma + 1e-12)
        features[f"bb_pctb_{w}"] = (close - (ma - 2 * std)) / (4 * std + 1e-12)

    # --- Volume features ---
    for w in windows[:2]:
        avg_vol = volume.rolling(w).mean()
        features[f"vol_ratio_{w}"] = volume / (avg_vol + 1e-12)

    features["volume_momentum"] = volume.pct_change(5)

    # --- Price action features ---
    body = (close - opn).abs()
    full_range = high - low + 1e-12
    features["body_ratio"] = body / full_range
    features["upper_shadow"] = (high - pd.concat([close, opn], axis=1).max(axis=1)) / full_range
    features["lower_shadow"] = (pd.concat([close, opn], axis=1).min(axis=1) - low) / full_range

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    features["rsi_14"] = 100 - 100 / (1 + rs)

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    features["macd_hist"] = (macd - macd_signal) / (close + 1e-12)

    return features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Engine
# ---------------------------------------------------------------------------


class GBMEnsembleEngine(PredictionEngine):
    """Gradient-boosted tree engine for forward return prediction.

    When LightGBM is available, uses it for fast, accurate gradient boosting.
    Otherwise falls back to scikit-learn's GradientBoostingRegressor.

    The engine:
      1. Computes ~30 hand-crafted features from OHLCV data
      2. Trains a GBM to predict forward returns over ``label_horizon`` days
      3. Iteratively forecasts multi-step prices from predicted returns
      4. Derives confidence from out-of-bag / validation error
    """

    def __init__(
        self,
        config: GBMConfig | None = None,
        **overrides: Any,
    ) -> None:
        cfg = config or GBMConfig()
        if overrides:
            cfg = GBMConfig(
                **{
                    fld.name: overrides.get(fld.name, getattr(cfg, fld.name))
                    for fld in cfg.__dataclass_fields__.values()
                }
            )
        self._config = cfg

        logger.debug(
            "GBMEnsembleEngine initialised (lgbm_available={}, n_estimators={}, max_depth={})",
            _LGBM_AVAILABLE,
            cfg.n_estimators,
            cfg.max_depth,
        )

    @property
    def name(self) -> str:
        return "gbm"

    @property
    def config(self) -> GBMConfig:
        return self._config

    def predict(self, df: pd.DataFrame, pred_len: int | None = None) -> pd.DataFrame:
        """Train on *df* and forecast *pred_len* future OHLCV candles.

        Returns a DataFrame with columns:
        ``timestamp, open, high, low, close, volume, predicted_return, confidence``.
        """
        horizon = pred_len if pred_len is not None else self._config.label_horizon

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

        try:
            predicted_returns, confidences = self._fit_predict(prepared, horizon)
        except Exception:
            logger.exception("GBM prediction failed -- returning zero predictions.")
            predicted_returns = np.zeros(horizon)
            confidences = np.full(horizon, 0.1)

        # Convert returns to price forecasts
        result = self._returns_to_ohlcv(
            prepared, predicted_returns, confidences, future_ts, horizon
        )
        return result

    def _fit_predict(
        self,
        df: pd.DataFrame,
        pred_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute features, train GBM, and predict forward returns."""
        cfg = self._config
        features = _compute_features(df, cfg.feature_windows)
        close = df["close"].astype(np.float64)

        # Label: forward return over label_horizon days
        forward_return = close.shift(-cfg.label_horizon) / close - 1.0
        combined = features.copy()
        combined["label"] = forward_return
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

        if len(combined) < cfg.min_train_rows:
            logger.warning(
                "Insufficient data for GBM ({} rows, need {}). Returning zero.",
                len(combined),
                cfg.min_train_rows,
            )
            return np.zeros(pred_len), np.full(pred_len, 0.1)

        feature_cols = [c for c in combined.columns if c != "label"]
        X = combined[feature_cols].values
        y = combined["label"].values

        # Train/val split: last 20%
        split = max(1, int(len(X) * 0.8))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        if _LGBM_AVAILABLE:
            model = self._train_lgbm(X_train, y_train, X_val, y_val)
        else:
            model = self._train_sklearn(X_train, y_train)

        # Validation R^2 for confidence
        val_pred = model.predict(X_val) if len(X_val) > 0 else np.array([])
        if len(val_pred) > 1:
            ss_res = np.sum((y_val - val_pred) ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2) + 1e-12
            r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
        else:
            r2 = 0.3

        # Predict using latest features (pass as DataFrame to preserve feature names)
        latest = features.replace([np.inf, -np.inf], np.nan).ffill().iloc[-1:]
        latest_x = latest[feature_cols]

        predictions: list[float] = []
        for step in range(pred_len):
            decay = 0.95 ** step
            pred_return = float(model.predict(latest_x)[0]) * decay
            predictions.append(pred_return)

        predicted_returns = np.array(predictions, dtype=np.float64)

        base_confidence = 0.3 + 0.5 * r2
        confidences = np.array(
            [float(np.clip(base_confidence * (0.95 ** s), 0.05, 0.95)) for s in range(pred_len)],
            dtype=np.float64,
        )

        return predicted_returns, confidences

    def _train_lgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """Train a LightGBM model."""
        cfg = self._config
        params = {
            "objective": "regression",
            "metric": "mse",
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": -1,
        }

        model = lgb.LGBMRegressor(**params)

        callbacks = []
        if len(X_val) > 0:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
            )
        else:
            model.fit(X_train, y_train)

        logger.debug(
            "LightGBM trained: {} rounds, best_iteration={}",
            cfg.n_estimators,
            getattr(model, "best_iteration_", cfg.n_estimators),
        )
        return model

    def _train_sklearn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """Fallback: train sklearn GradientBoostingRegressor."""
        from sklearn.ensemble import GradientBoostingRegressor

        cfg = self._config
        model = GradientBoostingRegressor(
            n_estimators=min(cfg.n_estimators, 100),  # cap for speed
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=0.8,
        )
        model.fit(X_train, y_train)
        logger.debug("sklearn GBR trained: {} estimators", model.n_estimators)
        return model

    @staticmethod
    def _returns_to_ohlcv(
        df: pd.DataFrame,
        predicted_returns: np.ndarray,
        confidences: np.ndarray,
        future_ts: pd.DatetimeIndex,
        pred_len: int,
    ) -> pd.DataFrame:
        """Convert predicted returns into OHLCV price forecasts."""
        last_close = float(df["close"].iloc[-1])
        last_volume = float(df["volume"].iloc[-1])

        # Historical high-low spread
        hl_spread = (
            (df["high"] - df["low"]) / df["close"]
        ).rolling(20).mean()
        avg_hl = float(hl_spread.iloc[-1]) if not np.isnan(hl_spread.iloc[-1]) else 0.03

        log_returns = np.log(
            df["close"].astype(np.float64) / df["close"].astype(np.float64).shift(1)
        ).dropna()
        hist_vol = float(log_returns.std()) if len(log_returns) > 1 else 0.02

        opens, highs, lows, closes, volumes = [], [], [], [], []
        current_close = last_close

        for step in range(pred_len):
            ret = float(predicted_returns[step])
            pred_close = current_close * (1.0 + ret)
            pred_open = current_close

            spread = avg_hl * (1.0 + hist_vol * np.sqrt(step + 1))
            half = pred_close * spread / 2.0
            pred_high = max(pred_open, pred_close) + half
            pred_low = max(min(pred_open, pred_close) - half, 0.0)

            pred_vol = max(last_volume * (0.98 ** step), 0.0)

            opens.append(pred_open)
            highs.append(pred_high)
            lows.append(pred_low)
            closes.append(pred_close)
            volumes.append(pred_vol)
            current_close = pred_close

        return pd.DataFrame(
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
