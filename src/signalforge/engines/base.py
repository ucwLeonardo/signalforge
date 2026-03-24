"""Base prediction engine interface and shared result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    """Container for prediction engine output.

    Attributes:
        asset_symbol: Ticker or asset identifier (e.g. ``"BTCUSDT"``).
        predictions: DataFrame of predicted OHLCV candles with a
            ``timestamp`` index and columns ``open, high, low, close, volume``.
        confidence: Overall confidence score in [0, 1].
        metadata: Arbitrary engine-specific metadata (model name, params, etc.).
    """

    asset_symbol: str
    predictions: pd.DataFrame
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )


class PredictionEngine(ABC):
    """Abstract base class that every prediction engine must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name (e.g. ``'kronos'``, ``'technical'``)."""

    @abstractmethod
    def predict(self, df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        """Generate predicted OHLCV candles.

        Parameters
        ----------
        df:
            Historical DataFrame with at least ``open, high, low, close, volume``
            columns and a datetime-like index or ``timestamp`` column.
        pred_len:
            Number of future candles to predict.

        Returns
        -------
        pd.DataFrame
            A DataFrame with ``pred_len`` rows and columns
            ``timestamp, open, high, low, close, volume``.
        """
