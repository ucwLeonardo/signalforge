"""Test all data sources: yfinance (stocks), ccxt (crypto), CoinGecko (spot prices)."""

import sys

def test_yfinance():
    print("=== yfinance (AAPL, 5d) ===")
    import yfinance as yf
    h = yf.Ticker("AAPL").history(period="5d")
    if h.empty:
        print("  FAILED: no data returned")
        return False
    for idx, row in h.iterrows():
        print(f"  {str(idx)[:10]}  close=${row['Close']:.2f}")
    print(f"  OK: {len(h)} bars")
    return True

def test_ccxt():
    print("\n=== ccxt / Binance (BTC/USDT, 3 bars) ===")
    import ccxt
    ohlcv = ccxt.binance().fetch_ohlcv("BTC/USDT", "1d", limit=3)
    if not ohlcv:
        print("  FAILED: no data returned")
        return False
    from datetime import datetime
    for bar in ohlcv:
        dt = datetime.fromtimestamp(bar[0] / 1000).strftime("%Y-%m-%d")
        print(f"  {dt}  close=${bar[4]:,.2f}")
    print(f"  OK: {len(ohlcv)} bars")
    return True

def test_coingecko():
    print("\n=== CoinGecko (BTC, ETH, SOL spot) ===")
    import urllib.request, json
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd"
    data = json.loads(urllib.request.urlopen(url, timeout=10).read())
    for coin, vals in data.items():
        print(f"  {coin}: ${vals['usd']:,.2f}")
    print(f"  OK: {len(data)} coins")
    return True

if __name__ == "__main__":
    results = {}
    for name, fn in [("yfinance", test_yfinance), ("ccxt", test_ccxt), ("coingecko", test_coingecko)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = False

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed}/{len(results)} data sources working")
    sys.exit(0 if all(results.values()) else 1)
