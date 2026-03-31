"""Data providers for stocks, crypto, and futures via Massive (Polygon) API."""

from __future__ import annotations

import abc
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from loguru import logger

from signalforge.data.models import Asset, AssetType, Bar, asset_from_symbol

# ---------------------------------------------------------------------------
# Column contract shared by every provider
# ---------------------------------------------------------------------------
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# ---------------------------------------------------------------------------
# Futures → ETF mapping (Massive doesn't have raw futures contracts)
# ---------------------------------------------------------------------------
_FUTURES_TO_ETF: dict[str, str] = {
    "ES=F": "SPY",   # S&P 500
    "NQ=F": "QQQ",   # Nasdaq 100
    "YM=F": "DIA",   # Dow Jones
    "GC=F": "GLD",   # Gold
    "SI=F": "SLV",   # Silver
    "CL=F": "USO",   # Crude Oil
    "NG=F": "UNG",   # Natural Gas
    "ZB=F": "TLT",   # 30-Year Treasury
    "ZN=F": "IEF",   # 10-Year Treasury
    "ZC=F": "CORN",  # Corn
}

# Interval mapping: SignalForge → Polygon multiplier/timespan
_INTERVAL_MAP: dict[str, tuple[int, str]] = {
    "1m": (1, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "1h": (1, "hour"),
    "4h": (4, "hour"),
    "1d": (1, "day"),
    "1w": (1, "week"),
    "1M": (1, "month"),
}


def _df_to_bars(df: pd.DataFrame) -> list[Bar]:
    """Convert a DataFrame with OHLCV_COLUMNS into a list of Bar objects."""
    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        bars.append(
            Bar(
                timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                amount=(
                    float(row.amount)
                    if hasattr(row, "amount") and row.amount is not None
                    else None
                ),
            )
        )
    return bars


def _resolve_asset(symbol_or_asset: str | Asset) -> Asset:
    """Accept either a string symbol or an Asset, always return an Asset."""
    if isinstance(symbol_or_asset, Asset):
        return symbol_or_asset
    return asset_from_symbol(symbol_or_asset)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseProvider(abc.ABC):
    """Common interface for all data providers."""

    @abc.abstractmethod
    def fetch(
        self,
        symbol_or_asset: str | Asset,
        interval: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data and return a DataFrame with ``OHLCV_COLUMNS``."""

    def fetch_bars(
        self,
        symbol_or_asset: str | Asset,
        interval: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Bar]:
        """Convenience wrapper that returns a list of :class:`Bar`."""
        df = self.fetch(symbol_or_asset, interval, start_date, end_date)
        return _df_to_bars(df)


# ---------------------------------------------------------------------------
# Massive (Polygon) unified provider
# ---------------------------------------------------------------------------
_MASSIVE_BASE_URL = "https://api.polygon.io"
_RATE_LIMIT_CALLS = 5
_RATE_LIMIT_WINDOW = 60  # seconds

# Module-level shared rate limiter — all MassiveProvider instances share this
# so that multiple instances created during a scan don't exceed the API limit.
_shared_call_timestamps: list[float] = []


class RateLimitError(Exception):
    """Raised when the API returns HTTP 429 and retry also fails."""


class MassiveProvider(BaseProvider):
    """Unified provider for stocks, crypto, and futures via Massive (Polygon) API.

    Handles ticker format conversion internally:
    - Stocks: ``AAPL`` → ``AAPL``
    - Crypto: ``BTC/USDT`` → ``X:BTCUSD``
    - Futures: ``ES=F`` → ``SPY`` (ETF proxy)
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("MASSIVE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Massive API key required. Set MASSIVE_API_KEY env var "
                "or pass api_key parameter."
            )
        self._session = requests.Session()
        self._session.params = {"apiKey": self._api_key}  # type: ignore[assignment]

    @staticmethod
    def _wait_for_rate_limit() -> None:
        """Sliding-window rate limiter: at most _RATE_LIMIT_CALLS per _RATE_LIMIT_WINDOW.

        Uses a module-level timestamp list so all provider instances share
        the same budget.
        """
        global _shared_call_timestamps
        now = time.monotonic()
        _shared_call_timestamps = [
            ts for ts in _shared_call_timestamps
            if now - ts < _RATE_LIMIT_WINDOW
        ]
        if len(_shared_call_timestamps) >= _RATE_LIMIT_CALLS:
            oldest = _shared_call_timestamps[0]
            sleep_time = oldest + _RATE_LIMIT_WINDOW - now + 1.0
            if sleep_time > 0:
                logger.info("Rate limited, waiting {:.0f}s...", sleep_time)
                time.sleep(sleep_time)
            now = time.monotonic()
            _shared_call_timestamps = [
                ts for ts in _shared_call_timestamps
                if now - ts < _RATE_LIMIT_WINDOW
            ]
        _shared_call_timestamps.append(time.monotonic())

    def _handle_429(
        self,
        resp: requests.Response,
        url: str,
        params: dict | None = None,
    ) -> requests.Response:
        """Handle HTTP 429: wait and retry once, otherwise raise."""
        if resp.status_code == 429:
            logger.warning("HTTP 429 received, waiting {}s before retry...", _RATE_LIMIT_WINDOW)
            time.sleep(_RATE_LIMIT_WINDOW)
            self._wait_for_rate_limit()
            retry_resp = self._session.get(url, params=params, timeout=30)
            if retry_resp.status_code == 429:
                raise RateLimitError(
                    f"Rate limited twice for {url}; giving up."
                )
            retry_resp.raise_for_status()
            return retry_resp
        resp.raise_for_status()
        return resp

    @staticmethod
    def to_polygon_ticker(symbol: str) -> str:
        """Convert a SignalForge symbol to a Polygon ticker.

        ``BTC/USDT`` → ``X:BTCUSD``
        ``ES=F`` → ``SPY``
        ``AAPL`` → ``AAPL``
        """
        # Futures → ETF
        if symbol.endswith("=F"):
            etf = _FUTURES_TO_ETF.get(symbol)
            if etf is None:
                raise ValueError(
                    f"No ETF mapping for futures symbol {symbol}. "
                    f"Known: {', '.join(_FUTURES_TO_ETF)}"
                )
            return etf

        # Crypto: BTC/USDT → X:BTCUSD
        if "/" in symbol:
            base = symbol.split("/")[0]
            return f"X:{base}USD"

        # Stocks: pass-through
        return symbol

    def fetch(
        self,
        symbol_or_asset: str | Asset,
        interval: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        asset = _resolve_asset(symbol_or_asset)
        now = datetime.now(tz=timezone.utc)
        start = start_date or (now - timedelta(days=730))
        end = end_date or now

        polygon_ticker = self.to_polygon_ticker(asset.symbol)
        multiplier, timespan = _INTERVAL_MAP.get(interval, (1, "day"))

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        logger.info(
            "MassiveProvider fetching {} → {} interval={} {} -> {}",
            asset.symbol, polygon_ticker, interval, start_str, end_str,
        )

        all_results: list[dict] = []
        url = (
            f"{_MASSIVE_BASE_URL}/v2/aggs/ticker/{polygon_ticker}"
            f"/range/{multiplier}/{timespan}/{start_str}/{end_str}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": "50000"}

        self._wait_for_rate_limit()
        resp = self._session.get(url, params=params, timeout=30)
        resp = self._handle_429(resp, url, params=params)
        data = resp.json()

        if data.get("resultsCount", 0) == 0:
            logger.warning("No data returned for {} ({})", asset.symbol, polygon_ticker)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        all_results.extend(data.get("results", []))

        # Pagination: Polygon uses next_url for large result sets
        while data.get("next_url"):
            self._wait_for_rate_limit()
            resp = self._session.get(data["next_url"], timeout=30)
            resp = self._handle_429(resp, data["next_url"])
            data = resp.json()
            all_results.extend(data.get("results", []))

        result = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [r["t"] for r in all_results], unit="ms", utc=True,
                ),
                "open": [float(r["o"]) for r in all_results],
                "high": [float(r["h"]) for r in all_results],
                "low": [float(r["l"]) for r in all_results],
                "close": [float(r["c"]) for r in all_results],
                "volume": [float(r["v"]) for r in all_results],
            }
        )
        result = result.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "MassiveProvider returned {} bars for {} ({})",
            len(result), asset.symbol, polygon_ticker,
        )
        return result


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
_shared_provider: MassiveProvider | None = None


def get_provider(symbol_or_asset: str | Asset | AssetType, **kwargs: object) -> BaseProvider:
    """Return a shared MassiveProvider instance for any asset type.

    All asset types (stocks, crypto, futures) are handled by MassiveProvider
    with internal ticker format conversion.  The shared instance reuses its
    ``requests.Session`` for TCP/TLS connection pooling.
    """
    global _shared_provider
    if kwargs:
        # Custom kwargs → create a fresh one-off instance
        return MassiveProvider(**kwargs)  # type: ignore[arg-type]
    if _shared_provider is None:
        _shared_provider = MassiveProvider()
    return _shared_provider
