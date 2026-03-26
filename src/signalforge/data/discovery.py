"""Dynamic asset discovery for SignalForge.

Automatically discovers tradeable assets from the market instead of
relying solely on a fixed configuration list.  Every discovery function
has a graceful fallback so the system works even without network access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from signalforge.config import Config


# ---------------------------------------------------------------------------
# Hardcoded fallback lists (used when live discovery fails)
# ---------------------------------------------------------------------------

_FALLBACK_US_STOCKS: list[str] = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AEP",
    "AIG", "AMAT", "AMD", "AMGN", "AMZN", "ANET", "AVGO", "AXP", "BA",
    "BAC", "BDX", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK-B", "BSX",
    "C", "CAT", "CHTR", "CI", "CL", "CMCSA", "COF", "COP", "COST",
    "CRM", "CSCO", "CVS", "CVX", "D", "DD", "DE", "DHR", "DIS",
    "DUK", "ECL", "EL", "EMR", "EW", "EXC", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM",
    "ICE", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KHC", "KLAC", "KO",
    "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD", "MCHP", "MCO",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "MU", "NEE", "NFLX", "NKE", "NOW", "NSC", "NVDA", "ORCL", "PEP",
    "PFE", "PG", "PM", "PYPL", "QCOM", "REGN", "RTX", "SBUX", "SCHW",
    "SHW", "SO", "SPG", "SYK", "T", "TGT", "TMO", "TMUS", "TXN",
    "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT",
    "XOM", "ZTS",
]

_FALLBACK_CRYPTO: list[str] = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "SHIB/USDT", "LTC/USDT", "ATOM/USDT",
    "ETC/USDT", "XLM/USDT", "NEAR/USDT", "FIL/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "AAVE/USDT", "MKR/USDT", "GRT/USDT",
    "INJ/USDT", "SUI/USDT", "TIA/USDT", "SEI/USDT", "FTM/USDT",
]

_FUTURES: list[str] = [
    "ES=F", "NQ=F", "YM=F", "GC=F", "SI=F",
    "CL=F", "NG=F", "ZB=F", "ZN=F", "ZC=F",
]


# ---------------------------------------------------------------------------
# Stock discovery
# ---------------------------------------------------------------------------

def discover_stocks(
    max_symbols: int = 500,
    min_market_cap: float = 1e9,
) -> list[str]:
    """Discover US stock tickers from the S&P 500 Wikipedia table.

    Falls back to a hardcoded list of top US stocks by market cap when
    the network request fails.

    Parameters
    ----------
    max_symbols:
        Maximum number of tickers to return.
    min_market_cap:
        Minimum market cap filter (reserved for future use; the Wikipedia
        table does not expose market cap directly).

    Returns
    -------
    list[str]
        Up to *max_symbols* ticker strings, sorted alphabetically.
    """
    try:
        import pandas as pd

        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        if not tables:
            raise ValueError("No tables found on S&P 500 Wikipedia page")

        sp500_table = tables[0]
        # The ticker column is typically named "Symbol"
        col = sp500_table.columns[0]
        tickers = (
            sp500_table[col]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        tickers = sorted(set(tickers))[:max_symbols]
        logger.info(
            "Discovered {} S&P 500 stocks from Wikipedia", len(tickers)
        )
        return tickers

    except Exception as exc:
        logger.warning(
            "S&P 500 discovery failed ({}), using fallback stock list", exc
        )
        fallback = sorted(_FALLBACK_US_STOCKS)[:max_symbols]
        logger.info(
            "Using fallback list of {} US stocks", len(fallback)
        )
        return fallback


# ---------------------------------------------------------------------------
# Crypto discovery
# ---------------------------------------------------------------------------

def discover_crypto(
    exchange_id: str = "gate",
    quote: str = "USDT",
    max_symbols: int = 200,
    min_volume_usd: float = 1e6,
) -> list[str]:
    """Discover actively-traded crypto pairs from a ccxt exchange.

    Filters by quote currency, active status, and 24-hour volume.
    Falls back to a hardcoded list of popular pairs on failure.

    Parameters
    ----------
    exchange_id:
        ccxt exchange identifier (e.g. ``"gate"``, ``"binance"``).
    quote:
        Quote currency to filter by (e.g. ``"USDT"``).
    max_symbols:
        Maximum number of pairs to return.
    min_volume_usd:
        Minimum 24h volume in USD to include a pair.

    Returns
    -------
    list[str]
        Up to *max_symbols* ``"BASE/QUOTE"`` strings, sorted by volume
        descending.
    """
    try:
        import ccxt  # type: ignore[import-untyped]

        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown ccxt exchange: {exchange_id}")

        exchange = exchange_cls()
        exchange.load_markets()

        # Collect active markets matching the requested quote currency
        candidate_symbols: list[str] = []
        for symbol, market in exchange.markets.items():
            if (
                market.get("quote") == quote
                and market.get("active", True)
                and market.get("spot", True)
            ):
                candidate_symbols.append(symbol)

        if not candidate_symbols:
            raise ValueError(
                f"No active {quote} spot markets found on {exchange_id}"
            )

        logger.info(
            "Found {} candidate {}/{} pairs on {}, fetching volumes...",
            len(candidate_symbols), "*", quote, exchange_id,
        )

        # Fetch tickers for volume ranking (batch to avoid timeout)
        try:
            # Only fetch top 100 to avoid Gate.io timeout
            batch = candidate_symbols[:100]
            exchange.timeout = 15000  # 15s timeout
            tickers = exchange.fetch_tickers(batch)
            volume_pairs: list[tuple[str, float]] = []
            for sym in batch:
                ticker = tickers.get(sym, {})
                vol_quote = ticker.get("quoteVolume") or 0.0
                if vol_quote >= min_volume_usd:
                    volume_pairs.append((sym, float(vol_quote)))

            volume_pairs.sort(key=lambda p: p[1], reverse=True)
            result = [p[0] for p in volume_pairs[:max_symbols]]
        except Exception as ticker_exc:
            logger.warning("Ticker fetch failed ({}), using alphabetical", ticker_exc)
            result = sorted(candidate_symbols)[:max_symbols]

        logger.info(
            "Discovered {} crypto pairs on {} (quote={})",
            len(result),
            exchange_id,
            quote,
        )
        return result

    except Exception as exc:
        logger.warning(
            "Crypto discovery failed ({}), using fallback crypto list", exc
        )
        fallback = [
            p for p in _FALLBACK_CRYPTO if p.endswith(f"/{quote}")
        ][:max_symbols]
        logger.info("Using fallback list of {} crypto pairs", len(fallback))
        return fallback


# ---------------------------------------------------------------------------
# Futures discovery
# ---------------------------------------------------------------------------

def discover_futures() -> list[str]:
    """Return a curated list of main US futures symbols for yfinance.

    Returns
    -------
    list[str]
        Futures symbols such as ``"ES=F"``, ``"GC=F"``, etc.
    """
    logger.info("Using curated list of {} US futures", len(_FUTURES))
    return list(_FUTURES)


# ---------------------------------------------------------------------------
# Master discovery
# ---------------------------------------------------------------------------

def discover_all(
    categories: list[str],
    config: "Config | None" = None,
) -> list[str]:
    """Discover assets for the requested categories, merged with config.

    Config-defined symbols are always included.  Discovered symbols are
    added on top (deduplicated).  If discovery fails for a category the
    config symbols alone are returned for that category.

    Parameters
    ----------
    categories:
        List of category names: ``"us_stocks"``, ``"crypto"``,
        ``"futures"``, ``"options"``.
    config:
        Optional :class:`Config` instance.  When ``None`` the default
        config is loaded.

    Returns
    -------
    list[str]
        Combined, deduplicated symbol list.
    """
    if config is None:
        from signalforge.config import load_config

        config = load_config()

    # Start with config-defined symbols (always included)
    config_symbols: list[str] = []
    if "us_stocks" in categories:
        config_symbols.extend(config.us_stocks)
    if "crypto" in categories:
        config_symbols.extend(config.crypto)
    if "futures" in categories:
        config_symbols.extend(config.futures)
    if "options" in categories:
        config_symbols.extend(config.options)

    # Discover additional symbols per category
    discovered: list[str] = []

    if "us_stocks" in categories:
        try:
            discovered.extend(discover_stocks())
        except Exception as exc:
            logger.warning("Stock discovery error: {}", exc)

    if "crypto" in categories:
        try:
            exchange_id = config.data.crypto_exchange
            discovered.extend(discover_crypto(exchange_id=exchange_id))
        except Exception as exc:
            logger.warning("Crypto discovery error: {}", exc)

    if "futures" in categories:
        try:
            discovered.extend(discover_futures())
        except Exception as exc:
            logger.warning("Futures discovery error: {}", exc)

    # Merge: config symbols first, then discovered (preserving order)
    seen: set[str] = set()
    merged: list[str] = []
    for sym in config_symbols + discovered:
        if sym not in seen:
            seen.add(sym)
            merged.append(sym)

    config_count = len(config_symbols)
    discovered_new = len(merged) - config_count
    logger.info(
        "Asset discovery: {} from config + {} newly discovered = {} total",
        config_count,
        discovered_new,
        len(merged),
    )
    return merged
