"""Unified price fetching for paper trading.

Sources (with fallback chain):
  - Crypto: CoinGecko (primary, no key, no geo-block) → Binance → Polygon prev close
  - Stocks: Polygon API prev close (same key as MassiveProvider)
  - Futures: Polygon via ETF proxy (ES=F → SPY)
  - Options: Polygon snapshot API (O:AAPL260619C00200000)

Proxy-aware: auto-detects HTTP_PROXY/HTTPS_PROXY env vars and Clash proxy.
No silent fallbacks. Errors are logged with symbol-level detail.
"""

from __future__ import annotations

import os
import sys

import requests


class PriceFetchError(Exception):
    """Raised when price fetching fails for one or more symbols."""

    def __init__(self, message: str, failed: list[str] | None = None) -> None:
        super().__init__(message)
        self.failed = failed or []


# ---------------------------------------------------------------------------
# Shared session with proxy auto-detection
# ---------------------------------------------------------------------------

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return a shared requests session with proxy auto-detection."""
    global _session
    if _session is not None:
        return _session

    _session = requests.Session()

    # Auto-detect proxy: env vars → common Clash ports → no proxy
    proxy_url = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )

    if not proxy_url:
        # Try common Clash proxy ports
        for port in (7897, 7890):
            try:
                test = requests.get(
                    f"http://127.0.0.1:{port}",
                    timeout=1,
                    allow_redirects=False,
                )
                proxy_url = f"http://127.0.0.1:{port}"
                break
            except requests.RequestException:
                continue

    if proxy_url:
        _session.proxies = {"http": proxy_url, "https": proxy_url}
        sys.stderr.write(f"[Prices] Using proxy: {proxy_url}\n")

    return _session


# ---------------------------------------------------------------------------
# Symbol classification
# ---------------------------------------------------------------------------

def _classify(symbol: str) -> str:
    """Classify symbol into: crypto, stock, futures, options."""
    if "/" in symbol:
        return "crypto"
    if symbol.endswith("=F"):
        return "futures"
    # Options: "AAPL 2026-06-19 200 C" (space-separated) or OCC format
    if " " in symbol:
        return "options"
    # OCC format: e.g. AAPL260619C00200000 (letters + 6 digits + C/P + 8 digits)
    import re
    if re.match(r"^[A-Z]+\d{6}[CP]\d{8}$", symbol):
        return "options"
    return "stock"


# ---------------------------------------------------------------------------
# CoinGecko (crypto) — free, no API key, no geo-restriction
# ---------------------------------------------------------------------------

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Map common trading pair base symbols to CoinGecko IDs
_COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "LINK": "chainlink",
    "DOGE": "dogecoin",
    "RNDR": "render-token",
    "MKR": "maker",
    "TAO": "bittensor",
    "FET": "fetch-ai",
    "SEI": "sei-network",
    "ARB": "arbitrum",
    "OP": "optimism",
    "AAVE": "aave",
    "POL": "matic-network",
    "SUI": "sui",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "SHIB": "shiba-inu",
    "PEPE": "pepe",
    "BONK": "bonk",
    "TIA": "celestia",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "ADA": "cardano",
    "XRP": "ripple",
    "NEAR": "near",
    "APT": "aptos",
    "WIF": "dogwifcoin",
    "INJ": "injective-protocol",
    "TRX": "tron",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "FIL": "filecoin",
    "ALGO": "algorand",
    "HBAR": "hedera-hashgraph",
    "ICP": "internet-computer",
    "VET": "vechain",
    "SAND": "the-sandbox",
    "MANA": "decentraland",
    "GRT": "the-graph",
    "CRV": "curve-dao-token",
    "RUNE": "thorchain",
    "STX": "blockstack",
    "IMX": "immutable-x",
    "GALA": "gala",
    "ENS": "ethereum-name-service",
    "COMP": "compound-governance-token",
    "SNX": "havven",
    "1INCH": "1inch",
    "SUSHI": "sushi",
    "YFI": "yearn-finance",
    "BAL": "balancer",
    "ZRX": "0x",
    "LDO": "lido-dao",
    "RPL": "rocket-pool",
    "PENDLE": "pendle",
    "JUP": "jupiter-exchange-solana",
    "JTO": "jito-governance-token",
    "W": "wormhole",
    "ENA": "ethena",
    "STRK": "starknet",
    "S": "fantom",
    "BNB": "binancecoin",
    "ETC": "ethereum-classic",
    "XLM": "stellar",
}


def _to_coingecko_id(symbol: str) -> str | None:
    """BTC/USDT → bitcoin. Returns None if unmapped."""
    base = symbol.split("/")[0].upper()
    return _COINGECKO_IDS.get(base)


def _fetch_coingecko_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch real-time prices from CoinGecko. Free, no API key needed."""
    if not symbols:
        return {}

    session = _get_session()
    sf_to_cg: dict[str, str] = {}
    unmapped: list[str] = []

    for sym in symbols:
        cg_id = _to_coingecko_id(sym)
        if cg_id:
            sf_to_cg[sym] = cg_id
        else:
            unmapped.append(sym)

    if unmapped:
        sys.stderr.write(
            f"[Prices] CoinGecko: no mapping for: {', '.join(unmapped)}\n"
        )

    if not sf_to_cg:
        return {}

    prices: dict[str, float] = {}
    errors: list[str] = []

    # CoinGecko supports batch — one request for all
    cg_ids = list(set(sf_to_cg.values()))
    try:
        resp = session.get(
            f"{_COINGECKO_BASE}/simple/price",
            params={"ids": ",".join(cg_ids), "vs_currencies": "usd"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        for sf_sym, cg_id in sf_to_cg.items():
            price_data = data.get(cg_id, {})
            price = price_data.get("usd", 0)
            if price and price > 0:
                prices[sf_sym] = float(price)
            else:
                errors.append(f"{sf_sym}: CoinGecko returned no price for {cg_id}")

    except requests.RequestException as exc:
        errors.append(f"CoinGecko batch fetch failed: {exc}")

    if errors:
        sys.stderr.write(f"[Prices] CoinGecko warnings: {'; '.join(errors)}\n")

    return prices


# ---------------------------------------------------------------------------
# Binance (crypto) — real-time, free, no API key (geo-restricted in China)
# ---------------------------------------------------------------------------

_BINANCE_BASE = "https://api.binance.com/api/v3"


def _to_binance_symbol(symbol: str) -> str:
    """BTC/USDT → BTCUSDT."""
    parts = symbol.split("/")
    return (parts[0] + parts[1]).upper()


def _fetch_binance_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch real-time prices from Binance. Used as fallback."""
    if not symbols:
        return {}

    session = _get_session()
    prices: dict[str, float] = {}
    errors: list[str] = []

    sf_to_binance = {sym: _to_binance_symbol(sym) for sym in symbols}

    if len(symbols) <= 10:
        for sf_sym, b_sym in sf_to_binance.items():
            try:
                resp = session.get(
                    f"{_BINANCE_BASE}/ticker/price",
                    params={"symbol": b_sym},
                    timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()
                # Check for geo-restriction response
                if "code" in data and data.get("msg", "").startswith("Service unavailable"):
                    errors.append(f"{sf_sym}: Binance geo-restricted")
                    break  # All requests will fail, skip rest
                price = float(data.get("price", 0))
                if price > 0:
                    prices[sf_sym] = price
                else:
                    errors.append(f"{sf_sym}: Binance returned price=0")
            except requests.RequestException as exc:
                errors.append(f"{sf_sym}: Binance error: {exc}")
    else:
        try:
            resp = session.get(f"{_BINANCE_BASE}/ticker/price", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # Check for geo-restriction
            if isinstance(data, dict) and data.get("msg", "").startswith("Service unavailable"):
                errors.append("Binance geo-restricted")
            elif isinstance(data, list):
                all_tickers = {t["symbol"]: float(t["price"]) for t in data}
                for sf_sym, b_sym in sf_to_binance.items():
                    price = all_tickers.get(b_sym, 0)
                    if price > 0:
                        prices[sf_sym] = price
                    else:
                        errors.append(f"{sf_sym}: not found on Binance ({b_sym})")
        except requests.RequestException as exc:
            errors.append(f"Binance batch fetch failed: {exc}")

    if errors:
        sys.stderr.write(f"[Prices] Binance warnings: {'; '.join(errors)}\n")

    return prices


# ---------------------------------------------------------------------------
# Polygon (crypto fallback) — prev close via API key
# ---------------------------------------------------------------------------

def _fetch_polygon_crypto_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch prev close from Polygon for crypto. Last resort fallback."""
    if not symbols:
        return {}

    provider = _get_polygon_provider()
    prices: dict[str, float] = {}
    errors: list[str] = []

    for sym in symbols:
        base = sym.split("/")[0].upper()
        ticker = f"X:{base}USD"
        try:
            provider._wait_for_rate_limit()
            resp = provider._session.get(
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                prices[sym] = float(results[0]["c"])
            else:
                errors.append(f"{sym}: Polygon returned no results for {ticker}")
        except requests.RequestException as exc:
            errors.append(f"{sym}: Polygon crypto error: {exc}")

    if errors:
        sys.stderr.write(f"[Prices] Polygon crypto warnings: {'; '.join(errors)}\n")

    return prices


# ---------------------------------------------------------------------------
# Polygon (stocks, futures) — prev close, requires API key
# ---------------------------------------------------------------------------

_polygon_provider = None


def _get_polygon_provider():
    """Get or create a MassiveProvider with proxy-aware session."""
    global _polygon_provider
    if _polygon_provider is not None:
        return _polygon_provider

    from signalforge.data.providers import MassiveProvider
    _polygon_provider = MassiveProvider()
    # Inject proxy settings into MassiveProvider's session
    session = _get_session()
    if session.proxies:
        _polygon_provider._session.proxies.update(session.proxies)
    return _polygon_provider


def _fetch_polygon_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch previous close from Polygon for stock/futures symbols."""
    if not symbols:
        return {}

    provider = _get_polygon_provider()
    prices: dict[str, float] = {}
    errors: list[str] = []

    for sym in symbols:
        try:
            ticker = provider.to_polygon_ticker(sym)
        except ValueError as exc:
            errors.append(f"{sym}: {exc}")
            continue

        try:
            provider._wait_for_rate_limit()
            resp = provider._session.get(
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                prices[sym] = float(results[0]["c"])
            else:
                errors.append(f"{sym}: Polygon returned no results for {ticker}")
        except requests.RequestException as exc:
            errors.append(f"{sym}: Polygon error: {exc}")

    if errors:
        sys.stderr.write(f"[Prices] Polygon warnings: {'; '.join(errors)}\n")

    return prices


# ---------------------------------------------------------------------------
# Polygon (options) — snapshot API
# ---------------------------------------------------------------------------

def _to_polygon_option_ticker(symbol: str) -> str:
    """Convert option symbol to Polygon format: O:AAPL260619C00200000.

    Accepts both human format ("AAPL 2026-06-19 200 C") and OCC format.
    """
    from signalforge.data.models import parse_option_symbol

    contract = parse_option_symbol(symbol)
    if contract is None:
        raise ValueError(f"Cannot parse option symbol: {symbol}")
    return f"O:{contract.occ_symbol}"


def _fetch_polygon_option_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch option prices from Polygon snapshot API.

    Uses /v3/snapshot/options/{underlying} filtered by contract,
    or /v3/snapshot?ticker=O:... for individual contracts.
    """
    if not symbols:
        return {}

    provider = _get_polygon_provider()
    prices: dict[str, float] = {}
    errors: list[str] = []

    for sym in symbols:
        try:
            ticker = _to_polygon_option_ticker(sym)
        except ValueError as exc:
            errors.append(f"{sym}: {exc}")
            continue

        try:
            provider._wait_for_rate_limit()
            # Use the universal snapshot endpoint for a single option contract
            resp = provider._session.get(
                "https://api.polygon.io/v3/snapshot",
                params={"ticker": ticker},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                result = results[0]
                # Try last trade price, then mid of bid/ask, then prev day close
                price = None
                last_trade = result.get("last_trade", {})
                if last_trade and last_trade.get("price"):
                    price = float(last_trade["price"])
                if price is None:
                    last_quote = result.get("last_quote", {})
                    bid = last_quote.get("bid", 0)
                    ask = last_quote.get("ask", 0)
                    if bid > 0 and ask > 0:
                        price = (bid + ask) / 2
                if price is None:
                    day = result.get("day", {})
                    if day and day.get("close"):
                        price = float(day["close"])
                if price and price > 0:
                    prices[sym] = price
                else:
                    errors.append(f"{sym}: Polygon snapshot returned no usable price for {ticker}")
            else:
                # Might not have options access — check status
                status = data.get("status", "")
                errors.append(
                    f"{sym}: Polygon returned no results for {ticker} "
                    f"(status={status}). Options may require a paid Polygon plan."
                )
        except requests.RequestException as exc:
            errors.append(f"{sym}: Polygon options error: {exc}")

    if errors:
        sys.stderr.write(f"[Prices] Options warnings: {'; '.join(errors)}\n")

    return prices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for a list of symbols.

    Routes each symbol to the correct source with fallback chain:
      - Crypto: CoinGecko (primary) → Binance → Polygon prev close
      - Stocks: Polygon prev close
      - Futures (ends with '=F'): Polygon via ETF proxy
      - Options: Polygon snapshot

    Logs warnings for individual failures but returns what it can.
    """
    if not symbols:
        return {}

    crypto_syms: list[str] = []
    stock_syms: list[str] = []  # includes futures
    option_syms: list[str] = []

    for sym in symbols:
        cat = _classify(sym)
        if cat == "crypto":
            crypto_syms.append(sym)
        elif cat in ("stock", "futures"):
            stock_syms.append(sym)
        elif cat == "options":
            option_syms.append(sym)

    prices: dict[str, float] = {}

    # Fetch crypto and stock/option prices in parallel
    import concurrent.futures

    def _fetch_crypto_chain() -> dict[str, float]:
        """CoinGecko → Binance → Polygon fallback chain."""
        result: dict[str, float] = {}
        if not crypto_syms:
            return result
        cg_prices = _fetch_coingecko_prices(crypto_syms)
        result.update(cg_prices)

        missing = [s for s in crypto_syms if s not in result]
        if missing:
            sys.stderr.write(
                f"[Prices] CoinGecko missed {len(missing)}, trying Binance...\n"
            )
            result.update(_fetch_binance_prices(missing))

        missing = [s for s in crypto_syms if s not in result]
        if missing:
            sys.stderr.write(
                f"[Prices] Binance missed {len(missing)}, trying Polygon...\n"
            )
            result.update(_fetch_polygon_crypto_prices(missing))
        return result

    def _fetch_stock_prices() -> dict[str, float]:
        if not stock_syms:
            return {}
        return _fetch_polygon_prices(stock_syms)

    def _fetch_option_prices() -> dict[str, float]:
        if not option_syms:
            return {}
        return _fetch_polygon_option_prices(option_syms)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        crypto_fut = pool.submit(_fetch_crypto_chain)
        stock_fut = pool.submit(_fetch_stock_prices)
        option_fut = pool.submit(_fetch_option_prices)

        for fut in concurrent.futures.as_completed([crypto_fut, stock_fut, option_fut]):
            try:
                prices.update(fut.result())
            except Exception as exc:
                sys.stderr.write(f"[Prices] Parallel fetch error: {exc}\n")

    # Report missing
    missing = [s for s in symbols if s not in prices]
    if missing:
        sys.stderr.write(
            f"[Prices] WARNING: No price for: {', '.join(missing)}\n"
        )

    fetched = [s for s in symbols if s in prices]
    if fetched:
        sys.stderr.write(
            f"[Prices] Fetched {len(fetched)}/{len(symbols)} prices\n"
        )

    return prices
