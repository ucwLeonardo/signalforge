"""Incremental data fetcher — downloads only new bars, caches in parquet store."""

from __future__ import annotations

import concurrent.futures
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from signalforge.data.providers import RateLimitError, get_provider
from signalforge.data.store import DataStore

_FETCH_TIMEOUT = 90  # seconds — must exceed rate limit window (60s) + network time
_CONSECUTIVE_FAIL_THRESHOLD = 10  # skip network after N consecutive failures
_EMPTY_SYMBOL_CACHE_DAYS = 7  # skip symbols that returned 0 bars for this many days


class IncrementalFetcher:
    """Wraps DataStore + providers to implement incremental data fetching.

    First call for a symbol downloads the full lookback history.
    Subsequent calls only fetch bars after the last cached timestamp.
    Tracks consecutive failures and stops attempting network fetches
    when the network appears unavailable.
    Remembers symbols that returned 0 bars to avoid wasting API calls.
    """

    def __init__(
        self,
        store: DataStore,
        cancel_flag: "Callable[[], bool] | None" = None,
    ) -> None:
        self._store = store
        self._consecutive_failures = 0
        self._network_disabled = False
        self._cancel_flag = cancel_flag
        self.last_fetch_source: str = ""  # "cache", "incremental", "full", "cache_fallback"
        self.last_new_bars: int = 0  # number of new bars added in last incremental fetch
        self._empty_cache_path = Path(store._root) / "empty_symbols.json"
        self._empty_symbols: dict[str, str] = self._load_empty_cache()

    def _load_empty_cache(self) -> dict[str, str]:
        """Load the empty-symbol cache from disk."""
        if self._empty_cache_path.exists():
            try:
                return json.loads(self._empty_cache_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_empty_cache(self) -> None:
        """Persist the empty-symbol cache to disk."""
        try:
            self._empty_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._empty_cache_path.write_text(json.dumps(self._empty_symbols))
        except Exception as exc:
            logger.warning("Failed to save empty-symbol cache: {}", exc)

    def _is_known_empty(self, symbol: str) -> bool:
        """Check if a symbol is known to return 0 bars recently."""
        recorded = self._empty_symbols.get(symbol)
        if not recorded:
            return False
        try:
            recorded_date = datetime.fromisoformat(recorded)
            return (datetime.now() - recorded_date).days < _EMPTY_SYMBOL_CACHE_DAYS
        except Exception:
            return False

    def _mark_empty(self, symbol: str) -> None:
        """Record that a symbol returned 0 bars."""
        self._empty_symbols[symbol] = datetime.now().isoformat()
        self._save_empty_cache()

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

        # Fast path: known-empty symbol — skip network call entirely
        if cached.empty and self._is_known_empty(symbol):
            logger.debug("{} ({}) known empty, skipping network call", symbol, interval)
            self.last_fetch_source = "empty_skip"
            return pd.DataFrame()

        # Fast path: cache is fresh — no network needed
        # For daily bars: stocks must have the most recent trading day's bar;
        # crypto is more relaxed (live prices handle freshness in the pipeline).
        if not cached.empty:
            last_ts = pd.Timestamp(cached["timestamp"].max())
            if interval in ("1d", "1D", "daily"):
                from signalforge.data.calendar import last_trading_day
                last_bar_date = last_ts.to_pydatetime().date()
                if last_ts.tzinfo is not None:
                    last_bar_date = last_ts.tz_convert("UTC").date()
                is_crypto = "/" in symbol
                if is_crypto:
                    # Crypto: daily bars are for model training; allow up to 2 days
                    age_days = (now - last_ts.to_pydatetime().replace(tzinfo=None)).days
                    cache_fresh = age_days <= 2
                else:
                    # Stocks/futures: must have the most recent trading day's bar
                    expected = last_trading_day()
                    cache_fresh = last_bar_date >= expected
            else:
                age_days = (now - last_ts.to_pydatetime().replace(tzinfo=None)).days
                cache_fresh = age_days <= 1

            if cache_fresh:
                logger.debug("{} ({}) cache is fresh, skipping fetch", symbol, interval)
                self.last_fetch_source = "cache"
                return cached

        # If network is known-down, skip fetch and return cache only
        if self._network_disabled:
            if not cached.empty:
                self.last_fetch_source = "cache"
                return cached
            logger.debug("No cache for {} and network disabled, skipping", symbol)
            self.last_fetch_source = ""
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
                self.last_fetch_source = "full"
                return df
            if not is_rate_limit:
                self._record_failure()
                if df is not None and df.empty:
                    # API returned successfully but with 0 bars — remember this
                    self._mark_empty(symbol)
            self.last_fetch_source = ""
            return pd.DataFrame()

        # Incremental: fetch only new bars since last cached timestamp
        last_ts = pd.Timestamp(cached["timestamp"].max())
        fetch_start = (last_ts - timedelta(days=1)).to_pydatetime()

        logger.info(
            "Incremental fetch for {} ({}): {} -> now ({} cached bars)",
            symbol, interval, fetch_start.strftime("%Y-%m-%d"), len(cached),
        )
        cached_count = len(cached)
        new_df, is_rate_limit = self._timed_fetch(provider, symbol, interval, fetch_start, now)
        if new_df is not None and not new_df.empty:
            self._consecutive_failures = 0
            self._store.save(symbol, interval, new_df, append=True)
            result = self._store.load(symbol, interval)
            self.last_new_bars = max(0, len(result) - cached_count)
            self.last_fetch_source = "incremental" if self.last_new_bars > 0 else "cache"
            return result

        # Fetch failed or empty — return cached data
        if new_df is None and not is_rate_limit:
            self._record_failure()
            logger.warning(
                "Incremental fetch failed/timed out for {} ({}), using {} cached bars",
                symbol, interval, len(cached),
            )
        self.last_fetch_source = "cache_fallback"
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

    def _timed_fetch(
        self, provider, symbol, interval, start, end,
    ) -> tuple[pd.DataFrame | None, bool]:
        """Run provider.fetch with a timeout, interruptible by cancel_flag.

        Returns ``(df_or_none, is_rate_limit)``.  When the failure is due to
        a rate-limit (HTTP 429), the second element is ``True`` so that the
        caller can avoid counting it as a real network failure.
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(provider.fetch, symbol, interval, start, end)
        try:
            # Poll in 0.5s intervals so cancel_flag is checked frequently
            elapsed = 0.0
            while elapsed < _FETCH_TIMEOUT:
                try:
                    return future.result(timeout=0.5), False
                except concurrent.futures.TimeoutError:
                    elapsed += 0.5
                    if self._cancel_flag is not None and self._cancel_flag():
                        logger.info("Fetch cancelled for {} ({})", symbol, interval)
                        future.cancel()
                        return None, False
            # Overall timeout
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
