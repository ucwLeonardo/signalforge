"""Dynamic asset discovery for SignalForge.

Automatically discovers tradeable assets from the market instead of
relying solely on a fixed configuration list.  Every discovery function
has a graceful fallback so the system works even without network access.
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from signalforge.config import Config

_DISCOVERY_TIMEOUT = 15  # seconds — abort discovery and use fallback


def _run_with_timeout(fn, timeout: int = _DISCOVERY_TIMEOUT):
    """Run *fn* in a thread with a hard timeout. Returns result or raises."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


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
    "POL/USDT", "UNI/USDT", "SHIB/USDT", "LTC/USDT", "ATOM/USDT",
    "ETC/USDT", "XLM/USDT", "NEAR/USDT", "FIL/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "AAVE/USDT", "MKR/USDT", "GRT/USDT",
    "INJ/USDT", "SUI/USDT", "TIA/USDT", "SEI/USDT", "S/USDT",
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
    the network request fails or times out.
    """
    def _fetch_sp500() -> list[str]:
        import pandas as pd

        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        if not tables:
            raise ValueError("No tables found on S&P 500 Wikipedia page")

        sp500_table = tables[0]
        col = sp500_table.columns[0]
        tickers = (
            sp500_table[col]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        return sorted(set(tickers))[:max_symbols]

    try:
        tickers = _run_with_timeout(_fetch_sp500)
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
    quote: str = "USDT",
    max_symbols: int = 200,
    **_kwargs: object,
) -> list[str]:
    """Return curated list of popular crypto pairs.

    Uses the Massive (Polygon) API to discover crypto tickers when
    available, falling back to a hardcoded list of top pairs.
    """
    def _fetch_crypto() -> list[str]:
        import os
        import requests

        api_key = os.environ.get("MASSIVE_API_KEY", "")
        if not api_key:
            raise ValueError("MASSIVE_API_KEY not set")

        resp = requests.get(
            "https://api.polygon.io/v3/reference/tickers",
            params={
                "market": "crypto",
                "active": "true",
                "limit": "1000",
                "apiKey": api_key,
            },
            timeout=_DISCOVERY_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        # Convert Polygon tickers (X:BTCUSD) → SignalForge format (BTC/USDT)
        pairs: list[str] = []
        for r in results:
            ticker = r.get("ticker", "")
            if ticker.startswith("X:") and ticker.endswith("USD"):
                base = ticker[2:-3]  # X:BTCUSD → BTC
                pairs.append(f"{base}/{quote}")

        logger.info("Discovered {} crypto pairs from Massive API", len(pairs))
        return pairs[:max_symbols]

    try:
        result = _run_with_timeout(_fetch_crypto)
        if result:
            return result
        raise ValueError("Empty result from Massive crypto discovery")
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
    """Return a curated list of main US futures symbols for yfinance."""
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
    """
    if config is None:
        from signalforge.config import load_config

        config = load_config()

    # Build per-category lists: config symbols first, then discovered
    # This ensures all stocks come before all crypto, etc.
    category_order = ["us_stocks", "crypto", "futures", "options"]
    config_map: dict[str, list[str]] = {
        "us_stocks": list(config.us_stocks),
        "crypto": list(config.crypto),
        "futures": list(config.futures),
        "options": list(config.options),
    }
    discovered_map: dict[str, list[str]] = {}

    if "us_stocks" in categories:
        try:
            discovered_map["us_stocks"] = discover_stocks()
        except Exception as exc:
            logger.warning("Stock discovery error: {}", exc)

    if "crypto" in categories:
        try:
            discovered_map["crypto"] = discover_crypto()
        except Exception as exc:
            logger.warning("Crypto discovery error: {}", exc)

    if "futures" in categories:
        try:
            discovered_map["futures"] = discover_futures()
        except Exception as exc:
            logger.warning("Futures discovery error: {}", exc)

    # Merge by category: all stocks → all crypto → all futures → all options
    seen: set[str] = set()
    merged: list[str] = []
    config_count = 0
    for cat in category_order:
        if cat not in categories:
            continue
        for sym in config_map.get(cat, []) + discovered_map.get(cat, []):
            if sym not in seen:
                seen.add(sym)
                merged.append(sym)
        config_count += len(config_map.get(cat, []))

    discovered_new = len(merged) - config_count
    logger.info(
        "Asset discovery: {} from config + {} newly discovered = {} total",
        config_count,
        discovered_new,
        len(merged),
    )
    return merged
