"""Data providers for stocks, crypto, and futures."""

from __future__ import annotations

import abc
from datetime import datetime, timedelta, timezone

import pandas as pd
from loguru import logger

from signalforge.data.models import Asset, AssetType, Bar, OptionContract, asset_from_symbol, parse_option_symbol

# ---------------------------------------------------------------------------
# Column contract shared by every provider
# ---------------------------------------------------------------------------
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


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
# Stock provider (yfinance)
# ---------------------------------------------------------------------------
class StockProvider(BaseProvider):
    """Fetches stock OHLCV data via *yfinance*."""

    def fetch(
        self,
        symbol_or_asset: str | Asset,
        interval: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        asset = _resolve_asset(symbol_or_asset)
        now = datetime.now(tz=timezone.utc)
        start = start_date or (now - timedelta(days=730))
        end = end_date or now

        logger.info(
            "StockProvider fetching {} interval={} {} -> {}",
            asset.symbol, interval, start.date(), end.date(),
        )

        ticker = yf.Ticker(asset.symbol)
        hist: pd.DataFrame = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )

        if hist.empty:
            logger.warning("No data returned for {}", asset.symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        hist = hist.reset_index()
        date_col = "Date" if "Date" in hist.columns else "Datetime"
        result = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hist[date_col], utc=True),
                "open": hist["Open"].astype(float),
                "high": hist["High"].astype(float),
                "low": hist["Low"].astype(float),
                "close": hist["Close"].astype(float),
                "volume": hist["Volume"].astype(float),
            }
        )
        result = result.sort_values("timestamp").reset_index(drop=True)
        logger.info("StockProvider returned {} bars for {}", len(result), asset.symbol)
        return result


# ---------------------------------------------------------------------------
# Crypto provider (ccxt / Binance)
# ---------------------------------------------------------------------------
class CryptoProvider(BaseProvider):
    """Fetches crypto OHLCV data via *ccxt* (Binance by default)."""

    def __init__(self, exchange_id: str = "binance") -> None:
        self._exchange_id = exchange_id

    def _create_exchange(self) -> object:
        import ccxt

        exchange_cls = getattr(ccxt, self._exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown ccxt exchange: {self._exchange_id}")
        return exchange_cls({"enableRateLimit": True})

    def fetch(
        self,
        symbol_or_asset: str | Asset,
        interval: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        asset = _resolve_asset(symbol_or_asset)
        exchange = self._create_exchange()

        now = datetime.now(tz=timezone.utc)
        start = start_date or (now - timedelta(days=730))
        end = end_date or now
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        logger.info(
            "CryptoProvider fetching {} on {} interval={} {} -> {}",
            asset.symbol, self._exchange_id, interval, start.date(), end.date(),
        )

        all_ohlcv: list[list[float]] = []
        while since_ms < end_ms:
            batch: list[list[float]] = exchange.fetch_ohlcv(  # type: ignore[attr-defined]
                asset.symbol,
                timeframe=interval,
                since=since_ms,
                limit=1000,
            )
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = int(batch[-1][0])
            if last_ts <= since_ms:
                break
            since_ms = last_ts + 1

        if not all_ohlcv:
            logger.warning("No data returned for {}", asset.symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.DataFrame(all_ohlcv, columns=["ts_ms", "open", "high", "low", "close", "volume"])
        df = df[df["ts_ms"] <= end_ms].copy()
        result = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["ts_ms"], unit="ms", utc=True),
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "volume": df["volume"].astype(float),
            }
        )
        result = result.sort_values("timestamp").reset_index(drop=True)
        logger.info("CryptoProvider returned {} bars for {}", len(result), asset.symbol)
        return result


# ---------------------------------------------------------------------------
# Futures provider (yfinance)
# ---------------------------------------------------------------------------
class FuturesProvider(BaseProvider):
    """Fetches futures OHLCV data via *yfinance*.

    Futures tickers on Yahoo Finance use the ``=F`` suffix (e.g. ``ES=F``).
    """

    def fetch(
        self,
        symbol_or_asset: str | Asset,
        interval: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        asset = _resolve_asset(symbol_or_asset)
        now = datetime.now(tz=timezone.utc)
        start = start_date or (now - timedelta(days=365))
        end = end_date or now

        logger.info(
            "FuturesProvider fetching {} interval={} {} -> {}",
            asset.symbol, interval, start.date(), end.date(),
        )

        ticker = yf.Ticker(asset.symbol)
        hist: pd.DataFrame = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )

        if hist.empty:
            logger.warning("No data returned for {}", asset.symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        hist = hist.reset_index()
        date_col = "Date" if "Date" in hist.columns else "Datetime"
        result = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hist[date_col], utc=True),
                "open": hist["Open"].astype(float),
                "high": hist["High"].astype(float),
                "low": hist["Low"].astype(float),
                "close": hist["Close"].astype(float),
                "volume": hist["Volume"].astype(float),
            }
        )
        result = result.sort_values("timestamp").reset_index(drop=True)
        logger.info("FuturesProvider returned {} bars for {}", len(result), asset.symbol)
        return result


# ---------------------------------------------------------------------------
# Options provider (yfinance chain + underlying OHLCV)
# ---------------------------------------------------------------------------
class OptionsProvider(BaseProvider):
    """Fetches option data using the underlying stock's OHLCV via *yfinance*,
    enriched with current option chain Greeks/IV when available.

    For historical signal generation, we use the underlying's price action
    because free historical OHLCV for individual option contracts is not
    available. The option chain snapshot provides current IV, volume, OI,
    bid/ask, and Greeks for the specific contract.
    """

    def fetch(
        self,
        symbol_or_asset: str | Asset,
        interval: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch underlying OHLCV and enrich with option chain data.

        The returned DataFrame has standard OHLCV columns for the underlying
        plus additional option-specific columns when available:
        ``implied_volatility``, ``option_volume``, ``open_interest``,
        ``bid``, ``ask``, ``delta``, ``gamma``, ``theta``.
        """
        import yfinance as yf

        asset = _resolve_asset(symbol_or_asset)
        contract = parse_option_symbol(asset.symbol)
        if contract is None:
            raise ValueError(f"Cannot parse option symbol: {asset.symbol}")

        now = datetime.now(tz=timezone.utc)
        start = start_date or (now - timedelta(days=365))
        end = end_date or now

        logger.info(
            "OptionsProvider fetching underlying {} for option {} interval={} {} -> {}",
            contract.underlying, asset.symbol, interval, start.date(), end.date(),
        )

        # 1. Fetch underlying stock OHLCV
        ticker = yf.Ticker(contract.underlying)
        hist: pd.DataFrame = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )

        if hist.empty:
            logger.warning("No data returned for underlying {}", contract.underlying)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        hist = hist.reset_index()
        date_col = "Date" if "Date" in hist.columns else "Datetime"
        result = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hist[date_col], utc=True),
                "open": hist["Open"].astype(float),
                "high": hist["High"].astype(float),
                "low": hist["Low"].astype(float),
                "close": hist["Close"].astype(float),
                "volume": hist["Volume"].astype(float),
            }
        )
        result = result.sort_values("timestamp").reset_index(drop=True)

        # 2. Enrich with current option chain snapshot
        try:
            chain = ticker.option_chain(contract.expiration)
            options_df = chain.calls if contract.option_type == "C" else chain.puts
            match = options_df[options_df["strike"] == contract.strike]
            if not match.empty:
                row = match.iloc[0]
                # Add option metadata to the last bar
                result["implied_volatility"] = float(row.get("impliedVolatility", 0.0))
                result["option_volume"] = float(row.get("volume", 0) or 0)
                result["open_interest"] = float(row.get("openInterest", 0) or 0)
                result["bid"] = float(row.get("bid", 0.0))
                result["ask"] = float(row.get("ask", 0.0))
                logger.info(
                    "Option chain enriched: IV={:.2%} Vol={} OI={}",
                    result["implied_volatility"].iloc[-1],
                    result["option_volume"].iloc[-1],
                    result["open_interest"].iloc[-1],
                )
            else:
                logger.warning(
                    "Strike {} not found in chain for {} exp {}",
                    contract.strike, contract.underlying, contract.expiration,
                )
        except Exception as e:
            logger.warning("Option chain fetch failed (non-fatal): {}", e)

        logger.info("OptionsProvider returned {} bars for {}", len(result), asset.symbol)
        return result

    def fetch_chain(self, underlying: str, expiration: str) -> dict[str, pd.DataFrame]:
        """Fetch the full option chain for an underlying + expiration.

        Returns ``{"calls": DataFrame, "puts": DataFrame}``.
        """
        import yfinance as yf

        ticker = yf.Ticker(underlying)
        chain = ticker.option_chain(expiration)
        return {"calls": chain.calls, "puts": chain.puts}

    def fetch_expirations(self, underlying: str) -> tuple[str, ...]:
        """List available expiration dates for an underlying."""
        import yfinance as yf

        ticker = yf.Ticker(underlying)
        return ticker.options


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
def get_provider(symbol_or_asset: str | Asset | AssetType, **kwargs: object) -> BaseProvider:
    """Return the appropriate provider for the given symbol, Asset, or AssetType.

    Accepts:
    - A plain symbol string (e.g. ``"AAPL"``, ``"BTC/USDT"``, ``"ES=F"``)
    - An :class:`Asset` instance
    - An :class:`AssetType` enum value
    """
    if isinstance(symbol_or_asset, AssetType):
        asset_type = symbol_or_asset
    elif isinstance(symbol_or_asset, Asset):
        asset_type = symbol_or_asset.asset_type
    else:
        asset = asset_from_symbol(symbol_or_asset)
        asset_type = asset.asset_type

    providers: dict[AssetType, type[BaseProvider]] = {
        AssetType.STOCK: StockProvider,
        AssetType.CRYPTO: CryptoProvider,
        AssetType.FUTURES: FuturesProvider,
        AssetType.OPTIONS: OptionsProvider,
    }
    cls = providers.get(asset_type)
    if cls is None:
        raise ValueError(f"No provider for asset type {asset_type}")
    return cls(**kwargs)  # type: ignore[arg-type]
