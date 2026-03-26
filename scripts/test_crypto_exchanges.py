"""Test which ccxt exchanges work for BTC/USDT OHLCV from China."""

import ccxt

EXCHANGES = ["okx", "bybit", "gate", "kucoin", "bitget", "htx", "mexc"]

for name in EXCHANGES:
    try:
        ex = getattr(ccxt, name)()
        ohlcv = ex.fetch_ohlcv("BTC/USDT", "1d", limit=2)
        if ohlcv:
            print(f"  PASS  {name:10s}  close=${ohlcv[-1][4]:,.2f}")
        else:
            print(f"  FAIL  {name:10s}  empty response")
    except Exception as e:
        msg = str(e).split("\n")[0][:80]
        print(f"  FAIL  {name:10s}  {msg}")
