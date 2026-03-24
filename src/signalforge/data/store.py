"""Parquet-based data storage for OHLCV bars."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from signalforge.data.models import Asset, AssetType, Bar, asset_from_symbol
from signalforge.data.providers import OHLCV_COLUMNS, _df_to_bars


def _resolve_asset(symbol_or_asset: str | Asset) -> Asset:
    """Accept either a string symbol or an Asset, always return an Asset."""
    if isinstance(symbol_or_asset, Asset):
        return symbol_or_asset
    return asset_from_symbol(symbol_or_asset)


class DataStore:
    """Save and load OHLCV bar data as Parquet files.

    Directory layout::

        {root}/
          {asset_type}/
            {symbol}_{interval}.parquet
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _asset_dir(self, asset: Asset) -> Path:
        return self._root / asset.asset_type.value

    def _parquet_path(self, asset: Asset, interval: str) -> Path:
        safe_symbol = asset.symbol.replace("/", "-")
        return self._asset_dir(asset) / f"{safe_symbol}_{interval}.parquet"

    @staticmethod
    def bars_to_df(bars: list[Bar]) -> pd.DataFrame:
        """Convert a list of :class:`Bar` to a DataFrame."""
        if not bars:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        records = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save(
        self,
        symbol_or_asset: str | Asset,
        interval: str,
        df: pd.DataFrame,
        *,
        append: bool = True,
    ) -> Path:
        """Persist a DataFrame to Parquet.

        Parameters
        ----------
        symbol_or_asset:
            A symbol string (e.g. ``"AAPL"``) or :class:`Asset` instance.
        interval:
            Bar interval string (e.g. ``"1d"``, ``"1h"``).
        df:
            DataFrame with at least ``OHLCV_COLUMNS``.
        append:
            If ``True`` and a file already exists, merge new rows by
            timestamp and drop duplicates.  If ``False``, overwrite.

        Returns
        -------
        Path to the written Parquet file.
        """
        asset = _resolve_asset(symbol_or_asset)
        path = self._parquet_path(asset, interval)
        path.parent.mkdir(parents=True, exist_ok=True)

        incoming = df[OHLCV_COLUMNS].copy()
        incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], utc=True)

        if append and path.exists():
            existing = pd.read_parquet(path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            merged = pd.concat([existing, incoming], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
            merged = merged.sort_values("timestamp").reset_index(drop=True)
        else:
            merged = incoming.sort_values("timestamp").reset_index(drop=True)

        merged.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved {} bars for {} ({}) -> {}", len(merged), asset.symbol, interval, path)
        return path

    def save_bars(
        self,
        symbol_or_asset: str | Asset,
        interval: str,
        bars: list[Bar],
        *,
        append: bool = True,
    ) -> Path:
        """Convenience: convert bars to DataFrame then save."""
        asset = _resolve_asset(symbol_or_asset)
        return self.save(asset, interval, self.bars_to_df(bars), append=append)

    def load(
        self,
        symbol_or_asset: str | Asset,
        interval: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from Parquet, optionally filtering by date range.

        Returns an empty DataFrame (with correct columns) when no data is found.
        """
        asset = _resolve_asset(symbol_or_asset)
        path = self._parquet_path(asset, interval)
        if not path.exists():
            logger.debug("No parquet file at {}", path)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if start_date is not None:
            start_ts = pd.Timestamp(start_date, tz="UTC")
            df = df[df["timestamp"] >= start_ts]
        if end_date is not None:
            end_ts = pd.Timestamp(end_date, tz="UTC")
            df = df[df["timestamp"] <= end_ts]

        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.debug("Loaded {} bars for {} ({}) from {}", len(df), asset.symbol, interval, path)
        return df

    def load_bars(
        self,
        symbol_or_asset: str | Asset,
        interval: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Bar]:
        """Load data as a list of :class:`Bar`."""
        return _df_to_bars(self.load(symbol_or_asset, interval, start_date, end_date))

    def exists(self, symbol_or_asset: str | Asset, interval: str) -> bool:
        """Check whether stored data exists for the given asset/interval."""
        asset = _resolve_asset(symbol_or_asset)
        return self._parquet_path(asset, interval).exists()

    def delete(self, symbol_or_asset: str | Asset, interval: str) -> bool:
        """Delete stored data. Returns ``True`` if a file was removed."""
        asset = _resolve_asset(symbol_or_asset)
        path = self._parquet_path(asset, interval)
        if path.exists():
            path.unlink()
            logger.info("Deleted {}", path)
            return True
        return False
