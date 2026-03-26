"""Signal combination logic: weighted ensemble of multiple engine predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from signalforge.data.models import CombinedSignal

# ---------------------------------------------------------------------------
# Default engine weights (must sum to 1.0 when all engines are present).
# If only a subset of engines report, the combiner re-normalises on the fly.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {
    "lstm": 0.25,
    "gbm": 0.20,
    "technical": 0.15,
    "qlib": 0.15,
    "chronos": 0.15,
    "agents": 0.10,
    "kronos": 0.00,
}


@dataclass(frozen=True)
class _EngineContribution:
    """Internal: one engine's normalised contribution."""

    direction: float  # -1 .. +1
    weight: float
    predicted_high: float | None
    predicted_low: float | None
    predicted_close: float | None


class SignalCombiner:
    """Combine predictions from heterogeneous engines into a single signal.

    Parameters
    ----------
    weights:
        Mapping of engine name -> weight.  If *None*, ``DEFAULT_WEIGHTS`` is
        used.  Weights are re-normalised at combination time so only the
        *relative* magnitudes matter.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights: dict[str, float] = dict(weights or DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine(self, predictions: dict[str, dict[str, Any]]) -> CombinedSignal:
        """Produce a unified ``CombinedSignal`` from engine prediction dicts.

        Parameters
        ----------
        predictions:
            ``{engine_name: prediction_dict}``.  Each *prediction_dict* is
            engine-specific but may contain:

            * **price-based** (e.g. Kronos/Chronos):
              ``predicted_high``, ``predicted_low``, ``predicted_close``,
              ``current_price``.
            * **signal-based** (e.g. technical, sentiment, qlib):
              ``signal`` (float in -1..+1) **or** ``direction`` (float).

        Returns
        -------
        CombinedSignal
        """
        if not predictions:
            return CombinedSignal(direction=0.0, confidence=0.0)

        contributions = self._extract_contributions(predictions)

        if not contributions:
            return CombinedSignal(direction=0.0, confidence=0.0)

        direction = self._weighted_direction(contributions)
        confidence = self._compute_confidence(contributions)
        pred_high = self._weighted_price(contributions, "predicted_high")
        pred_low = self._weighted_price(contributions, "predicted_low")
        pred_close = self._weighted_price(contributions, "predicted_close")

        return CombinedSignal(
            direction=_clamp(direction, -1.0, 1.0),
            confidence=_clamp(confidence, 0.0, 1.0),
            predicted_high=pred_high,
            predicted_low=pred_low,
            predicted_close=pred_close,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_contributions(
        self,
        predictions: dict[str, dict[str, Any]],
    ) -> list[_EngineContribution]:
        """Convert raw engine dicts into normalised contributions."""
        raw: list[_EngineContribution] = []

        for engine_name, pred in predictions.items():
            weight = self._weights.get(engine_name, 0.0)
            if weight <= 0.0:
                continue

            direction = self._direction_from(pred)
            if direction is None:
                continue

            raw.append(
                _EngineContribution(
                    direction=_clamp(direction, -1.0, 1.0),
                    weight=weight,
                    predicted_high=_to_float(pred.get("predicted_high")),
                    predicted_low=_to_float(pred.get("predicted_low")),
                    predicted_close=_to_float(pred.get("predicted_close")),
                ),
            )

        # Re-normalise weights so they sum to 1.0 for the *active* engines.
        total_w = sum(c.weight for c in raw)
        if total_w <= 0.0:
            return []

        return [
            _EngineContribution(
                direction=c.direction,
                weight=c.weight / total_w,
                predicted_high=c.predicted_high,
                predicted_low=c.predicted_low,
                predicted_close=c.predicted_close,
            )
            for c in raw
        ]

    @staticmethod
    def _direction_from(pred: dict[str, Any]) -> float | None:
        """Derive a -1..+1 direction from an engine prediction dict.

        Price-based engines: direction = sign(predicted_close - current_price),
        scaled by the percentage move (capped at 1.0).

        Signal-based engines: read ``signal`` or ``direction`` directly.
        """
        # Try signal-based first (explicit direction value).
        for key in ("signal", "direction"):
            val = pred.get(key)
            if val is not None:
                f = _to_float(val)
                if f is not None:
                    return _clamp(f, -1.0, 1.0)

        # Fall back to price-based derivation.
        predicted_close = _to_float(pred.get("predicted_close"))
        current_price = _to_float(pred.get("current_price"))
        if predicted_close is not None and current_price is not None and current_price > 0.0:
            pct_change = (predicted_close - current_price) / current_price
            # Scale so a 5 % move maps to ~1.0 direction.
            direction = _clamp(pct_change / 0.05, -1.0, 1.0)
            return direction

        return None

    @staticmethod
    def _weighted_direction(contributions: list[_EngineContribution]) -> float:
        return sum(c.direction * c.weight for c in contributions)

    @staticmethod
    def _weighted_price(
        contributions: list[_EngineContribution],
        attr: str,
    ) -> float | None:
        """Weighted average of a price field, ignoring engines that lack it."""
        values = [
            (getattr(c, attr), c.weight)
            for c in contributions
            if getattr(c, attr) is not None
        ]
        if not values:
            return None
        total_w = sum(w for _, w in values)
        if total_w <= 0.0:
            return None
        return sum(v * w for v, w in values) / total_w

    @staticmethod
    def _compute_confidence(contributions: list[_EngineContribution]) -> float:
        """Confidence measures agreement among engines.

        1.0 when all engines point the same direction with maximum strength,
        0.0 when they perfectly cancel out.

        Formula: 1 - normalised_std_of_directions.
        """
        if len(contributions) < 2:
            return abs(contributions[0].direction) if contributions else 0.0

        directions = [c.direction for c in contributions]
        mean = sum(directions) / len(directions)
        variance = sum((d - mean) ** 2 for d in directions) / len(directions)
        # Max possible std for values in [-1, 1] is 1.0 (all at extremes).
        std = math.sqrt(variance)
        return _clamp(1.0 - std, 0.0, 1.0)


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
