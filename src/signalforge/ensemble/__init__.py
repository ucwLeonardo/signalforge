"""Ensemble layer: combine predictions from multiple engines into unified signals."""

from signalforge.ensemble.combiner import SignalCombiner
from signalforge.ensemble.targets import TargetCalculator

__all__ = [
    "SignalCombiner",
    "TargetCalculator",
]
