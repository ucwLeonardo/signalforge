"""Incremental data fetcher — downloads only new bars, caches in parquet store."""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from signalforge.data.providers import RateLimitError, get_provider
from signalforge.data.store import DataStore

_FETCH_TIMEOUT = 90  # seconds — must exceed rate limit window (60s) + network time
_CONSECUTIVE_FAIL_THRESHOLD = 10  # skip network after N consecutive failures


class IncrementalFetcher:
    """Wraps DataStore + providers to implement incremental data fetching.

    First call for a symbol downloads the full lookback history.
    Subsequent calls only fetch bars after the last cached timestamp.
    Tracks consecutive failures and stops attempting network fetches
    when the network appears unavailable.
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store
        self._consecutive_failures = 0
        self._network_disabled = False

    def fetch(
        self,
        symbol: str,
        interval: str,
        lookback_days: int,
    ) -> pd.DataFrame:
        """Fetch OHLCV data, using cache + incremental update.

        Returns the full DataFrame (cached + newly fetched).
        """
        now = datetime.now()
        cached = self._store.load(symbol, interval)

        # Fast path: cache is fresh (< 1 day old) — no network needed
        if not cached.empty:
            last_ts = pd.Timestamp(cached["timestamp"].max())
            age_days = (now - last_ts.to_pydatetime().replace(tzinfo=None)).days
            if age_days < 1:
                logger.debug("{} ({}) cache is fresh, skipping fetch", symbol, interval)
                return cached

        # If network is known-down, skip fetch and return cache only
        if self._network_disabled:
            if not cached.empty:
                return cached
            logger.debug("No cache for {} and network disabled, skipping", symbol)
            return pd.DataFrame()

        provider = get_provider(symbol)

        if cached.empty:
            # First run: download full history
            start = now - timedelta(days=lookback_days)
            logger.info(
                "First fetch for {} ({}): downloading {} days of history",
                symbol, interval, lookback_days,
            )
            df, is_rate_limit = self._timed_fetch(provider, symbol, interval, start, now)
            if df is not None and not df.empty:
                self._consecutive_failures = 0
                self._store.save(symbol, interval, df, append=False)
                return df
            if not is_rate_limit:
                self._record_failure()
            return pd.DataFrame()

        # Incremental: fetch only new bars since last cached timestamp
        last_ts = pd.Timestamp(cached["timestamp"].max())
        fetch_start = (last_ts - timedelta(days=1)).to_pydatetime()

        logger.info(
            "Incremental fetch for {} ({}): {} -> now ({} cached bars)",
            symbol, interval, fetch_start.strftime("%Y-%m-%d"), len(cached),
        )
        new_df, is_rate_limit = self._timed_fetch(provider, symbol, interval, fetch_start, now)
        if new_df is not None and not new_df.empty:
            self._consecutive_failures = 0
            self._store.save(symbol, interval, new_df, append=True)
            return self._store.load(symbol, interval)

        # Fetch failed or empty — return cached data
        if new_df is None and not is_rate_limit:
            self._record_failure()
            logger.warning(
                "Incremental fetch failed/timed out for {} ({}), using {} cached bars",
                symbol, interval, len(cached),
            )
        return cached

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _CONSECUTIVE_FAIL_THRESHOLD:
            self._network_disabled = True
            logger.warning(
                "Network appears unavailable after {} consecutive failures — "
                "skipping remaining fetches, using cached data only",
                self._consecutive_failures,
            )

    @staticmethod
    def _timed_fetch(
        provider, symbol, interval, start, end,
    ) -> tuple[pd.DataFrame | None, bool]:
        """Run provider.fetch with a timeout.

        Returns ``(df_or_none, is_rate_limit)``.  When the failure is due to
        a rate-limit (HTTP 429), the second element is ``True`` so that the
        caller can avoid counting it as a real network failure.
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(provider.fetch, symbol, interval, start, end)
        try:
            return future.result(timeout=_FETCH_TIMEOUT), False
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Fetch timed out after {}s for {} ({})",
                _FETCH_TIMEOUT, symbol, interval,
            )
            return None, False
        except RateLimitError:
            logger.warning(
                "Rate limited for {} ({}), will not count as failure",
                symbol, interval,
            )
            return None, True
        except Exception as exc:
            logger.warning(
                "Fetch failed for {} ({}): {}",
                symbol, interval, exc,
            )
            return None, False
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
