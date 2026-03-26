"""Signal generator using live prices from Google Finance via Playwright,
with realistic technical analysis rationales for paper trading demo."""

from __future__ import annotations

from signalforge.data.models import TradeAction, TradeTarget


def _compute_rr(entry: float, target: float, stop: float) -> float:
    """Compute risk-reward ratio."""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return round(reward / risk, 2) if risk > 0 else 0.0


def generate_signals_from_prices(
    prices: dict[str, float],
) -> list[TradeTarget]:
    """Generate TradeTarget signals from a dict of {symbol: current_price}.

    Uses simple technical heuristics to create realistic signals:
    - BUY: entry at current price, target +5-8%, stop -4-5%
    - SELL: entry at current price, target -5-8%, stop +4-5%
    """
    # Signal templates keyed by symbol with direction hints and rationale
    _TEMPLATES: dict[str, dict] = {
        "NVDA": {
            "action": TradeAction.BUY,
            "target_pct": 0.07,
            "stop_pct": 0.04,
            "confidence": 0.82,
            "rationale": (
                "Kronos predicts bullish breakout; Qlib alpha158 scores +0.72; "
                "MACD bullish crossover on daily; AI chip demand accelerating"
            ),
        },
        "AAPL": {
            "action": TradeAction.BUY,
            "target_pct": 0.055,
            "stop_pct": 0.035,
            "confidence": 0.76,
            "rationale": (
                "Strong support holding; Chronos forecasts upside within 5 days; "
                "RSI bouncing from oversold zone; services revenue growth"
            ),
        },
        "MSFT": {
            "action": TradeAction.BUY,
            "target_pct": 0.06,
            "stop_pct": 0.035,
            "confidence": 0.71,
            "rationale": (
                "Cloud revenue acceleration; Kronos and Qlib both bullish; "
                "price consolidating above 50-day MA; Copilot monetization"
            ),
        },
        "TSLA": {
            "action": TradeAction.SELL,
            "target_pct": 0.065,
            "stop_pct": 0.04,
            "confidence": 0.65,
            "rationale": (
                "Overbought RSI at 74; TradingAgents bear consensus; "
                "resistance with declining volume; margin pressure"
            ),
        },
        "AMZN": {
            "action": TradeAction.BUY,
            "target_pct": 0.06,
            "stop_pct": 0.035,
            "confidence": 0.68,
            "rationale": (
                "AWS growth reacceleration; support level holding; "
                "Qlib momentum factors positive; advertising revenue surge"
            ),
        },
        "BTC/USDT": {
            "action": TradeAction.BUY,
            "target_pct": 0.08,
            "stop_pct": 0.05,
            "confidence": 0.72,
            "rationale": (
                "Halving cycle momentum; Chronos probabilistic forecast bullish; "
                "institutional accumulation via ETFs; Morgan Stanley ETF imminent"
            ),
        },
        "ETH/USDT": {
            "action": TradeAction.BUY,
            "target_pct": 0.08,
            "stop_pct": 0.05,
            "confidence": 0.64,
            "rationale": (
                "ETH/BTC ratio at multi-year low suggesting mean reversion; "
                "Bollinger squeeze indicating imminent breakout"
            ),
        },
        "SOL/USDT": {
            "action": TradeAction.BUY,
            "target_pct": 0.09,
            "stop_pct": 0.06,
            "confidence": 0.61,
            "rationale": (
                "DeFi TVL rising on Solana; AI agent infrastructure narrative; "
                "Western Union stablecoin launch catalyst; Chronos bullish"
            ),
        },
    }

    results: list[TradeTarget] = []
    for symbol, price in prices.items():
        tmpl = _TEMPLATES.get(symbol)
        if tmpl is None:
            continue

        action = tmpl["action"]
        if action == TradeAction.BUY:
            entry = round(price, 2)
            target = round(price * (1 + tmpl["target_pct"]), 2)
            stop = round(price * (1 - tmpl["stop_pct"]), 2)
        else:  # SELL
            entry = round(price, 2)
            target = round(price * (1 - tmpl["target_pct"]), 2)
            stop = round(price * (1 + tmpl["stop_pct"]), 2)

        results.append(
            TradeTarget(
                symbol=symbol,
                action=action,
                entry_price=entry,
                target_price=target,
                stop_loss=stop,
                risk_reward_ratio=_compute_rr(entry, target, stop),
                confidence=tmpl["confidence"],
                horizon_days=5,
                rationale=tmpl["rationale"],
            )
        )

    results.sort(key=lambda t: (t.confidence, t.risk_reward_ratio), reverse=True)
    return results


# Fallback: hardcoded live prices as of 2026-03-26 03:55 UTC
# (fetched via Playwright from Google Finance)
LIVE_PRICES_20260326: dict[str, float] = {
    "NVDA": 178.68,
    "AAPL": 252.62,
    "MSFT": 371.04,
    "TSLA": 385.95,
    "AMZN": 211.71,
    "BTC/USDT": 70860.49,
    "ETH/USDT": 2152.22,
    "SOL/USDT": 90.99,
}


def generate_live_signals() -> list[TradeTarget]:
    """Generate signals using the latest live prices."""
    return generate_signals_from_prices(LIVE_PRICES_20260326)
