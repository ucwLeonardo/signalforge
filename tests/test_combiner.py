"""Tests for the signal combiner (ensemble logic)."""

from __future__ import annotations

import pytest

from signalforge.ensemble.combiner import SignalCombiner


class TestSignalCombiner:
    """Test the weighted ensemble signal combiner."""

    def test_empty_predictions(self) -> None:
        combiner = SignalCombiner()
        result = combiner.combine({})
        assert result.direction == 0.0
        assert result.confidence == 0.0

    def test_single_signal_engine(self) -> None:
        combiner = SignalCombiner({"technical": 1.0})
        result = combiner.combine({
            "technical": {"signal": 0.6},
        })
        assert abs(result.direction - 0.6) < 0.01
        assert result.confidence > 0.5

    def test_single_price_engine(self) -> None:
        combiner = SignalCombiner({"kronos": 1.0})
        result = combiner.combine({
            "kronos": {
                "predicted_close": 210.0,
                "predicted_high": 220.0,
                "predicted_low": 200.0,
                "current_price": 200.0,
            },
        })
        # 5% up -> direction should be +1.0 (capped)
        assert result.direction > 0.0
        assert result.predicted_close == 210.0

    def test_multi_engine_agreement_bullish(self) -> None:
        """All engines agree on bullish -> high confidence."""
        combiner = SignalCombiner({
            "kronos": 0.4,
            "technical": 0.3,
            "qlib": 0.3,
        })
        result = combiner.combine({
            "kronos": {"predicted_close": 105.0, "current_price": 100.0},
            "technical": {"signal": 0.8},
            "qlib": {"signal": 0.7},
        })
        assert result.direction > 0.5
        assert result.confidence > 0.6

    def test_multi_engine_disagreement(self) -> None:
        """Engines disagree -> lower confidence."""
        combiner = SignalCombiner({
            "kronos": 0.5,
            "technical": 0.5,
        })
        result = combiner.combine({
            "kronos": {"predicted_close": 110.0, "current_price": 100.0},  # bullish
            "technical": {"signal": -0.8},  # bearish
        })
        # Direction should be moderate (mixed)
        assert abs(result.direction) < 0.8
        # Confidence should be lower due to disagreement
        assert result.confidence < 0.9

    def test_bearish_signal(self) -> None:
        combiner = SignalCombiner({"kronos": 0.5, "technical": 0.5})
        result = combiner.combine({
            "kronos": {"predicted_close": 90.0, "current_price": 100.0},
            "technical": {"signal": -0.7},
        })
        assert result.direction < 0.0

    def test_weight_renormalization(self) -> None:
        """Weights re-normalize when only subset of engines report."""
        combiner = SignalCombiner({
            "kronos": 0.4,
            "technical": 0.15,
            "qlib": 0.15,
            "chronos": 0.15,
            "agents": 0.15,
        })
        # Only kronos and technical report
        result = combiner.combine({
            "kronos": {"predicted_close": 105.0, "current_price": 100.0},
            "technical": {"signal": 0.5},
        })
        # Should still produce a valid signal
        assert -1.0 <= result.direction <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_zero_weight_engine_ignored(self) -> None:
        combiner = SignalCombiner({"kronos": 1.0, "technical": 0.0})
        result = combiner.combine({
            "kronos": {"predicted_close": 105.0, "current_price": 100.0},
            "technical": {"signal": -1.0},  # this should be ignored
        })
        assert result.direction > 0.0

    def test_price_predictions_propagated(self) -> None:
        combiner = SignalCombiner({"kronos": 1.0})
        result = combiner.combine({
            "kronos": {
                "predicted_close": 210.0,
                "predicted_high": 220.0,
                "predicted_low": 195.0,
                "current_price": 200.0,
            },
        })
        assert result.predicted_close == 210.0
        assert result.predicted_high == 220.0
        assert result.predicted_low == 195.0

    def test_direction_clamped(self) -> None:
        """Extreme predictions should be clamped to [-1, 1]."""
        combiner = SignalCombiner({"kronos": 1.0})
        result = combiner.combine({
            "kronos": {
                "predicted_close": 200.0,  # 100% gain
                "current_price": 100.0,
            },
        })
        assert result.direction <= 1.0
        assert result.direction >= -1.0
