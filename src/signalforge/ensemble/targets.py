"""Convert a CombinedSignal into actionable buy/sell/hold trade targets."""

from __future__ import annotations

from signalforge.data.models import (
    CombinedSignal,
    SupportResistance,
    TradeAction,
    TradeTarget,
)

# Direction thresholds to decide BUY / SELL / HOLD.
_BUY_THRESHOLD = 0.15
_SELL_THRESHOLD = -0.15


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
            Optional support/resistance levels.  When *None*, the calculator
            relies solely on the predicted prices inside *signal*.
        atr:
            Average True Range for stop-loss buffer.  Falls back to
            ``default_atr`` when *None*.
        horizon_days:
            Override for forecast horizon.
        """
        effective_atr = atr if atr is not None else self._default_atr
        effective_horizon = horizon_days if horizon_days is not None else self._horizon_days
        action = _classify_action(signal.direction)

        if action == TradeAction.BUY:
            return self._buy_target(
                symbol, signal, current_price, levels, effective_atr, effective_horizon,
            )

        if action == TradeAction.SELL:
            return self._sell_target(
                symbol, signal, current_price, levels, effective_atr, effective_horizon,
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
    ) -> TradeTarget:
        # Entry: predicted pullback (predicted_low) if available, else current.
        entry = (
            signal.predicted_low
            if signal.predicted_low is not None and signal.predicted_low < current_price
            else current_price
        )

        # Target: predicted high or resistance, whichever is higher.
        candidates: list[float] = []
        if signal.predicted_high is not None:
            candidates.append(signal.predicted_high)
        if levels is not None:
            candidates.append(levels.resistance)
        target = max(candidates) if candidates else current_price * 1.02

        # Stop: support or predicted_low - ATR, whichever is lower.
        stop_candidates: list[float] = []
        if levels is not None:
            stop_candidates.append(levels.support)
        if signal.predicted_low is not None:
            stop_candidates.append(signal.predicted_low - atr)
        stop = min(stop_candidates) if stop_candidates else entry - atr if atr > 0.0 else entry * 0.98

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
    ) -> TradeTarget:
        entry = current_price

        # Target: predicted low or support, whichever is lower.
        candidates: list[float] = []
        if signal.predicted_low is not None:
            candidates.append(signal.predicted_low)
        if levels is not None:
            candidates.append(levels.support)
        target = min(candidates) if candidates else current_price * 0.98

        # Stop: resistance or predicted_high + ATR, whichever is higher.
        stop_candidates: list[float] = []
        if levels is not None:
            stop_candidates.append(levels.resistance)
        if signal.predicted_high is not None:
            stop_candidates.append(signal.predicted_high + atr)
        stop = max(stop_candidates) if stop_candidates else entry + atr if atr > 0.0 else entry * 1.02

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
