"""Convert a CombinedSignal into actionable buy/sell/hold trade targets."""

from __future__ import annotations

from signalforge.config import TradingParams, get_trading_params
from signalforge.data.models import (
    CombinedSignal,
    SupportResistance,
    TradeAction,
    TradeTarget,
)

# Direction thresholds to decide BUY / SELL / HOLD.
_BUY_THRESHOLD = 0.15
_SELL_THRESHOLD = -0.15


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class TargetCalculator:
    """Translate a ``CombinedSignal`` into a concrete ``TradeTarget``.

    Parameters
    ----------
    default_atr:
        Average True Range used as a buffer for stop-loss calculations when no
        explicit ATR is supplied.  Expressed in the same unit as price.
    horizon_days:
        Default forecast horizon in calendar days.
    """

    def __init__(
        self,
        default_atr: float = 0.0,
        horizon_days: int = 5,
    ) -> None:
        self._default_atr = default_atr
        self._horizon_days = horizon_days

    def calculate(
        self,
        symbol: str,
        signal: CombinedSignal,
        current_price: float,
        levels: SupportResistance | None = None,
        atr: float | None = None,
        horizon_days: int | None = None,
        asset_type: str | None = None,
        trading_params: TradingParams | None = None,
    ) -> TradeTarget:
        """Produce a ``TradeTarget`` from a combined signal.

        Parameters
        ----------
        symbol:
            Ticker or asset identifier.
        signal:
            Unified signal from :class:`SignalCombiner`.
        current_price:
            Latest market price for the asset.
        levels:
            Optional support/resistance levels.
        atr:
            Average True Range for stop-loss buffer.
        horizon_days:
            Override for forecast horizon.
        asset_type:
            One of "stock", "crypto", "futures", "options".
        trading_params:
            Explicit trading params; if None, derived from asset_type.
        """
        effective_atr = atr if atr is not None else self._default_atr
        params = trading_params or get_trading_params(asset_type or "stock")
        effective_horizon = horizon_days if horizon_days is not None else params.horizon_days
        action = _classify_action(signal.direction)

        # Discard low-confidence signals — not actionable
        if signal.confidence < 0.70 and action != TradeAction.HOLD:
            return self._hold_target(symbol, signal, current_price, effective_horizon)

        if action == TradeAction.BUY:
            return self._buy_target(
                symbol, signal, current_price, levels, effective_atr, effective_horizon, params,
            )

        if action == TradeAction.SELL:
            return self._sell_target(
                symbol, signal, current_price, levels, effective_atr, effective_horizon, params,
            )

        return self._hold_target(symbol, signal, current_price, effective_horizon)

    # ------------------------------------------------------------------
    # Private target builders
    # ------------------------------------------------------------------

    @staticmethod
    def _buy_target(
        symbol: str,
        signal: CombinedSignal,
        current_price: float,
        levels: SupportResistance | None,
        atr: float,
        horizon_days: int,
        params: TradingParams,
    ) -> TradeTarget:
        # Entry: always use current market price (signal provides direction,
        # not a precise entry point — actual entry happens at market price).
        entry = current_price

        # --- Target: ATR-based with clamping ---
        raw_target_dist = atr * params.atr_target_multiplier if atr > 0 else entry * params.min_target_pct
        target_dist = _clamp(
            raw_target_dist,
            entry * params.min_target_pct,
            entry * params.max_target_pct,
        )

        # Also consider predicted_high and resistance as candidates
        candidates: list[float] = [entry + target_dist]
        if signal.predicted_high is not None:
            candidates.append(signal.predicted_high)
        if levels is not None:
            candidates.append(levels.resistance)
        target = max(candidates)

        # Clamp final target distance
        final_target_dist = _clamp(
            target - entry,
            entry * params.min_target_pct,
            entry * params.max_target_pct,
        )
        target = entry + final_target_dist

        # --- Stop: ATR-based with clamping ---
        raw_stop_dist = atr * params.atr_stop_multiplier if atr > 0 else entry * params.min_stop_pct
        stop_dist = _clamp(
            raw_stop_dist,
            entry * params.min_stop_pct,
            entry * params.max_stop_pct,
        )
        stop = entry - stop_dist

        rr = _risk_reward(entry, target, stop)

        rationale = (
            f"Bullish signal (direction={signal.direction:+.2f}, "
            f"confidence={signal.confidence:.0%}). "
            f"Entry near {entry:.2f}, targeting {target:.2f} with stop at {stop:.2f}."
        )

        return TradeTarget(
            symbol=symbol,
            action=TradeAction.BUY,
            entry_price=round(entry, 4),
            target_price=round(target, 4),
            stop_loss=round(stop, 4),
            risk_reward_ratio=round(rr, 2),
            confidence=signal.confidence,
            horizon_days=horizon_days,
            rationale=rationale,
        )

    @staticmethod
    def _sell_target(
        symbol: str,
        signal: CombinedSignal,
        current_price: float,
        levels: SupportResistance | None,
        atr: float,
        horizon_days: int,
        params: TradingParams,
    ) -> TradeTarget:
        entry = current_price

        # --- Target (below entry): ATR-based with clamping ---
        raw_target_dist = atr * params.atr_target_multiplier if atr > 0 else entry * params.min_target_pct
        target_dist = _clamp(
            raw_target_dist,
            entry * params.min_target_pct,
            entry * params.max_target_pct,
        )

        # Also consider predicted_low and support
        candidates: list[float] = [entry - target_dist]
        if signal.predicted_low is not None:
            candidates.append(signal.predicted_low)
        if levels is not None:
            candidates.append(levels.support)
        target = min(candidates)

        # Clamp final target distance
        final_target_dist = _clamp(
            entry - target,
            entry * params.min_target_pct,
            entry * params.max_target_pct,
        )
        target = entry - final_target_dist

        # --- Stop (above entry): ATR-based with clamping ---
        raw_stop_dist = atr * params.atr_stop_multiplier if atr > 0 else entry * params.min_stop_pct
        stop_dist = _clamp(
            raw_stop_dist,
            entry * params.min_stop_pct,
            entry * params.max_stop_pct,
        )
        stop = entry + stop_dist

        rr = _risk_reward(entry, target, stop)

        rationale = (
            f"Bearish signal (direction={signal.direction:+.2f}, "
            f"confidence={signal.confidence:.0%}). "
            f"Entry at {entry:.2f}, targeting {target:.2f} with stop at {stop:.2f}."
        )

        return TradeTarget(
            symbol=symbol,
            action=TradeAction.SELL,
            entry_price=round(entry, 4),
            target_price=round(target, 4),
            stop_loss=round(stop, 4),
            risk_reward_ratio=round(rr, 2),
            confidence=signal.confidence,
            horizon_days=horizon_days,
            rationale=rationale,
        )

    @staticmethod
    def _hold_target(
        symbol: str,
        signal: CombinedSignal,
        current_price: float,
        horizon_days: int,
    ) -> TradeTarget:
        return TradeTarget(
            symbol=symbol,
            action=TradeAction.HOLD,
            entry_price=round(current_price, 4),
            target_price=round(current_price, 4),
            stop_loss=round(current_price, 4),
            risk_reward_ratio=0.0,
            confidence=signal.confidence,
            horizon_days=horizon_days,
            rationale=(
                f"Neutral signal (direction={signal.direction:+.2f}, "
                f"confidence={signal.confidence:.0%}). No actionable trade."
            ),
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _classify_action(direction: float) -> TradeAction:
    if direction >= _BUY_THRESHOLD:
        return TradeAction.BUY
    if direction <= _SELL_THRESHOLD:
        return TradeAction.SELL
    return TradeAction.HOLD


def _risk_reward(entry: float, target: float, stop: float) -> float:
    """Compute reward-to-risk ratio.  Returns 0.0 when risk is zero."""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk == 0.0:
        return 0.0
    return reward / risk
