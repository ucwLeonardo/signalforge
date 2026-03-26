"""Incremental data fetcher — downloads only new bars, caches in parquet store."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from signalforge.data.providers import get_provider
from signalforge.data.store import DataStore


class IncrementalFetcher:
    """Wraps DataStore + providers to implement incremental data fetching.

    First call for a symbol downloads the full lookback history.
    Subsequent calls only fetch bars after the last cached timestamp.
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store

    def fetch(
        self,
        symbol: str,
        interval: str,
        lookback_days: int,
        *,
        exchange_id: str | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data, using cache + incremental update.

        Returns the full DataFrame (cached + newly fetched).
        """
        provider_kwargs = {}
        if exchange_id:
            provider_kwargs["exchange_id"] = exchange_id
        provider = get_provider(symbol, **provider_kwargs)

        now = datetime.now()
        cached = self._store.load(symbol, interval)

        if cached.empty:
            # First run: download full history
            start = now - timedelta(days=lookback_days)
            logger.info(
                "First fetch for {} ({}): downloading {} days of history",
                symbol, interval, lookback_days,
            )
            df = provider.fetch(symbol, interval, start, now)
            if not df.empty:
                self._store.save(symbol, interval, df, append=False)
            return df

        # Incremental: find last cached timestamp, fetch from there
        last_ts = pd.Timestamp(cached["timestamp"].max())
        # Overlap by 1 day to catch any intraday gaps
        fetch_start = (last_ts - timedelta(days=1)).to_pydatetime()

        if (now - last_ts.to_pydatetime().replace(tzinfo=None)).days < 1:
            logger.debug("{} ({}) cache is fresh, skipping fetch", symbol, interval)
            return cached

        logger.info(
            "Incremental fetch for {} ({}): {} -> now ({} cached bars)",
            symbol, interval, fetch_start.strftime("%Y-%m-%d"), len(cached),
        )
        new_df = provider.fetch(symbol, interval, fetch_start, now)
        if not new_df.empty:
            self._store.save(symbol, interval, new_df, append=True)

        # Return the full dataset from store
        return self._store.load(symbol, interval)
