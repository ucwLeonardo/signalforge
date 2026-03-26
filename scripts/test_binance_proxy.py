"""Test Binance connectivity with various proxy/endpoint configurations."""

import os
import ccxt

proxy = os.environ.get("https_proxy", os.environ.get("HTTPS_PROXY", ""))
print(f"Current https_proxy: {proxy}")
print()

# Test 1: Default Binance (api.binance.com)
print("=== Test 1: binance (default endpoint) ===")
try:
    ex = ccxt.binance()
    ohlcv = ex.fetch_ohlcv("BTC/USDT", "1d", limit=2)
    print(f"  PASS  close=${ohlcv[-1][4]:,.2f}")
except Exception as e:
    print(f"  FAIL  {str(e).split(chr(10))[0][:100]}")

# Test 2: Binance with explicit proxy passthrough
print("\n=== Test 2: binance with explicit proxy ===")
try:
    ex = ccxt.binance({"proxies": {"https": proxy, "http": proxy}} if proxy else {})
    ohlcv = ex.fetch_ohlcv("BTC/USDT", "1d", limit=2)
    print(f"  PASS  close=${ohlcv[-1][4]:,.2f}")
except Exception as e:
    print(f"  FAIL  {str(e).split(chr(10))[0][:100]}")

# Test 3: Binance US (binance.us) - different domain
print("\n=== Test 3: binanceus ===")
try:
    ex = ccxt.binanceus()
    ohlcv = ex.fetch_ohlcv("BTC/USDT", "1d", limit=2)
    print(f"  PASS  close=${ohlcv[-1][4]:,.2f}")
except Exception as e:
    print(f"  FAIL  {str(e).split(chr(10))[0][:100]}")

# Test 4: Binance with alternative hostnames
print("\n=== Test 4: binance with hostname overrides ===")
ALT_HOSTS = [
    "api1.binance.com",
    "api2.binance.com",
    "api3.binance.com",
    "api4.binance.com",
]
for host in ALT_HOSTS:
    try:
        ex = ccxt.binance({"hostname": host})
        ohlcv = ex.fetch_ohlcv("BTC/USDT", "1d", limit=2)
        print(f"  PASS  {host}  close=${ohlcv[-1][4]:,.2f}")
        break
    except Exception as e:
        print(f"  FAIL  {host}  {str(e).split(chr(10))[0][:80]}")

# Test 5: Direct HTTPS request to check if it's a DNS/routing issue
print("\n=== Test 5: raw HTTPS to api.binance.com ===")
try:
    import urllib.request
    req = urllib.request.Request(
        "https://api.binance.com/api/v3/ping",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp = urllib.request.urlopen(req, timeout=10)
    print(f"  PASS  status={resp.status}  body={resp.read().decode()}")
except Exception as e:
    print(f"  FAIL  {e}")
