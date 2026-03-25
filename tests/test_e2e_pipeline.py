"""End-to-end integration tests for the full signal pipeline.

These tests exercise the complete flow: data → engines → ensemble → targets,
using synthetic data to avoid network dependencies.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from signalforge.config import load_config
from signalforge.data.models import CombinedSignal, SupportResistance, TradeAction, TradeTarget
from signalforge.engines.kronos_engine import KronosConfig, KronosEngine
from signalforge.engines.technical import TechnicalEngine, compute_signals, compute_support_resistance
from signalforge.ensemble.combiner import SignalCombiner
from signalforge.ensemble.targets import TargetCalculator
from signalforge.output.report import ReportGenerator


class TestFullPipelineStock:
    """End-to-end test: stock (AAPL-like) through all enabled engines."""

    def test_stock_full_pipeline(self, ohlcv_stock_df: pd.DataFrame) -> None:
        """Run Kronos + Technical → Combine → Targets → Report for a stock."""
        symbol = "AAPL"
        current_price = float(ohlcv_stock_df["close"].iloc[-1])

        # Step 1: Kronos prediction (fallback mode)
        kronos = KronosEngine(KronosConfig(pred_len=5))
        kronos_pred = kronos.predict(ohlcv_stock_df, pred_len=5)
        assert len(kronos_pred) == 5

        # Step 2: Technical analysis
        tech_engine = TechnicalEngine()
        signals_df = compute_signals(ohlcv_stock_df)
        supports, resistances = compute_support_resistance(ohlcv_stock_df)

        # Step 3: Combine
        engine_results = {
            "kronos": {
                "type": "price",
                "predicted_close": float(kronos_pred["close"].iloc[-1]),
                "predicted_high": float(kronos_pred["high"].max()),
                "predicted_low": float(kronos_pred["low"].min()),
                "current_price": current_price,
            },
            "technical": {
                "type": "signal",
                "signal": float(signals_df["signal_strength"].iloc[-1]),
                "support": supports[0],
                "resistance": resistances[0],
            },
        }

        weights = {"kronos": 0.6, "technical": 0.4}
        combiner = SignalCombiner(weights)
        combined = combiner.combine(engine_results)

        assert -1.0 <= combined.direction <= 1.0
        assert 0.0 <= combined.confidence <= 1.0

        # Step 4: Calculate targets
        levels = SupportResistance(support=supports[0], resistance=resistances[0])
        calc = TargetCalculator()
        target = calc.calculate(
            symbol=symbol,
            signal=combined,
            current_price=current_price,
            levels=levels,
        )

        assert target.symbol == "AAPL"
        assert target.action in (TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD)
        assert target.confidence >= 0.0

        # Step 5: Generate report
        gen = ReportGenerator()
        table_out = gen.generate_report([target], fmt="table")
        assert "AAPL" in table_out

        json_out = gen.generate_report([target], fmt="json")
        assert "AAPL" in json_out


class TestFullPipelineCrypto:
    """End-to-end test: crypto (BTC/USDT-like) through all enabled engines."""

    def test_crypto_full_pipeline(self, ohlcv_crypto_df: pd.DataFrame) -> None:
        symbol = "BTC/USDT"
        current_price = float(ohlcv_crypto_df["close"].iloc[-1])

        # Kronos
        kronos = KronosEngine(KronosConfig(pred_len=5))
        kronos_pred = kronos.predict(ohlcv_crypto_df, pred_len=5)

        # Technical
        signals_df = compute_signals(ohlcv_crypto_df)
        supports, resistances = compute_support_resistance(ohlcv_crypto_df)

        # Combine
        engine_results = {
            "kronos": {
                "predicted_close": float(kronos_pred["close"].iloc[-1]),
                "predicted_high": float(kronos_pred["high"].max()),
                "predicted_low": float(kronos_pred["low"].min()),
                "current_price": current_price,
            },
            "technical": {
                "signal": float(signals_df["signal_strength"].iloc[-1]),
                "support": supports[0],
                "resistance": resistances[0],
            },
        }

        combiner = SignalCombiner({"kronos": 0.6, "technical": 0.4})
        combined = combiner.combine(engine_results)

        # Targets
        calc = TargetCalculator()
        target = calc.calculate(
            symbol=symbol,
            signal=combined,
            current_price=current_price,
            levels=SupportResistance(support=supports[0], resistance=resistances[0]),
        )

        assert target.symbol == "BTC/USDT"
        assert target.action in (TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD)

        # Report
        gen = ReportGenerator()
        output = gen.generate_report([target], fmt="json")
        assert "BTC/USDT" in output


class TestFullPipelineFutures:
    """End-to-end test: futures (ES=F-like)."""

    def test_futures_full_pipeline(self, ohlcv_futures_df: pd.DataFrame) -> None:
        symbol = "ES=F"
        current_price = float(ohlcv_futures_df["close"].iloc[-1])

        kronos = KronosEngine(KronosConfig(pred_len=5))
        kronos_pred = kronos.predict(ohlcv_futures_df, pred_len=5)

        signals_df = compute_signals(ohlcv_futures_df)
        supports, resistances = compute_support_resistance(ohlcv_futures_df)

        engine_results = {
            "kronos": {
                "predicted_close": float(kronos_pred["close"].iloc[-1]),
                "predicted_high": float(kronos_pred["high"].max()),
                "predicted_low": float(kronos_pred["low"].min()),
                "current_price": current_price,
            },
            "technical": {
                "signal": float(signals_df["signal_strength"].iloc[-1]),
                "support": supports[0],
                "resistance": resistances[0],
            },
        }

        combiner = SignalCombiner({"kronos": 0.6, "technical": 0.4})
        combined = combiner.combine(engine_results)

        calc = TargetCalculator()
        target = calc.calculate(
            symbol=symbol,
            signal=combined,
            current_price=current_price,
            levels=SupportResistance(support=supports[0], resistance=resistances[0]),
        )

        assert target.symbol == "ES=F"
        assert target.action in (TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD)


class TestMultiAssetScan:
    """Test scanning multiple asset types together (simulating `signalforge scan`)."""

    def test_multi_asset_scan(
        self,
        ohlcv_stock_df: pd.DataFrame,
        ohlcv_crypto_df: pd.DataFrame,
        ohlcv_futures_df: pd.DataFrame,
    ) -> None:
        """Simulate scanning stocks, crypto, and futures together."""
        assets = [
            ("AAPL", ohlcv_stock_df),
            ("BTC/USDT", ohlcv_crypto_df),
            ("ES=F", ohlcv_futures_df),
        ]

        combiner = SignalCombiner({"kronos": 0.6, "technical": 0.4})
        calc = TargetCalculator()
        all_targets: list[TradeTarget] = []

        for symbol, df in assets:
            current_price = float(df["close"].iloc[-1])

            kronos = KronosEngine(KronosConfig(pred_len=5))
            kronos_pred = kronos.predict(df, pred_len=5)
            signals_df = compute_signals(df)
            supports, resistances = compute_support_resistance(df)

            engine_results = {
                "kronos": {
                    "predicted_close": float(kronos_pred["close"].iloc[-1]),
                    "predicted_high": float(kronos_pred["high"].max()),
                    "predicted_low": float(kronos_pred["low"].min()),
                    "current_price": current_price,
                },
                "technical": {
                    "signal": float(signals_df["signal_strength"].iloc[-1]),
                    "support": supports[0],
                    "resistance": resistances[0],
                },
            }

            combined = combiner.combine(engine_results)
            target = calc.calculate(
                symbol=symbol,
                signal=combined,
                current_price=current_price,
                levels=SupportResistance(support=supports[0], resistance=resistances[0]),
            )
            all_targets.append(target)

        assert len(all_targets) == 3
        symbols = [t.symbol for t in all_targets]
        assert "AAPL" in symbols
        assert "BTC/USDT" in symbols
        assert "ES=F" in symbols

        # Generate combined report
        gen = ReportGenerator()
        table = gen.generate_report(all_targets, fmt="table")
        assert "AAPL" in table
        assert "BTC/USDT" in table
        assert "ES=F" in table

    def test_sort_by_confidence(
        self,
        ohlcv_stock_df: pd.DataFrame,
        ohlcv_crypto_df: pd.DataFrame,
        ohlcv_futures_df: pd.DataFrame,
    ) -> None:
        """Verify we can sort results by confidence to find top signals."""
        assets = [
            ("AAPL", ohlcv_stock_df),
            ("BTC/USDT", ohlcv_crypto_df),
            ("ES=F", ohlcv_futures_df),
        ]

        combiner = SignalCombiner({"kronos": 0.6, "technical": 0.4})
        calc = TargetCalculator()
        all_targets: list[TradeTarget] = []

        for symbol, df in assets:
            current_price = float(df["close"].iloc[-1])
            kronos = KronosEngine(KronosConfig(pred_len=5))
            kronos_pred = kronos.predict(df, pred_len=5)
            signals_df = compute_signals(df)
            supports, resistances = compute_support_resistance(df)

            engine_results = {
                "kronos": {
                    "predicted_close": float(kronos_pred["close"].iloc[-1]),
                    "predicted_high": float(kronos_pred["high"].max()),
                    "predicted_low": float(kronos_pred["low"].min()),
                    "current_price": current_price,
                },
                "technical": {
                    "signal": float(signals_df["signal_strength"].iloc[-1]),
                    "support": supports[0],
                    "resistance": resistances[0],
                },
            }

            combined = combiner.combine(engine_results)
            target = calc.calculate(
                symbol=symbol, signal=combined,
                current_price=current_price,
                levels=SupportResistance(support=supports[0], resistance=resistances[0]),
            )
            all_targets.append(target)

        # Sort by confidence descending
        sorted_targets = sorted(all_targets, key=lambda t: t.confidence, reverse=True)
        assert sorted_targets[0].confidence >= sorted_targets[-1].confidence

        # Filter to BUY signals only
        buy_signals = [t for t in sorted_targets if t.action == TradeAction.BUY]
        # All buy signals should have positive entry < target
        for t in buy_signals:
            assert t.target_price >= t.entry_price
