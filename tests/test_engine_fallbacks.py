"""Tests for Phase 2/3 engine fallbacks (Qlib, Chronos, Agents).

All engines fall back to simpler algorithms when their dependencies
(pyqlib, chronos-forecasting, tradingagents) are not installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from signalforge.engines.kronos_engine import KronosConfig, KronosEngine


class TestQlibEngineFallback:
    """Test QlibEngine fallback (ridge regression on momentum factors)."""

    def test_qlib_fallback_produces_predictions(self, ohlcv_stock_df: pd.DataFrame) -> None:
        from signalforge.engines.qlib_engine import QlibEngine
        from signalforge.config import QlibConfig

        engine = QlibEngine(QlibConfig(enabled=True))
        result = engine.predict(ohlcv_stock_df, pred_len=5)

        assert not result.empty
        assert "predicted_return" in result.columns
        assert len(result) == 5

    def test_qlib_fallback_crypto(self, ohlcv_crypto_df: pd.DataFrame) -> None:
        from signalforge.engines.qlib_engine import QlibEngine
        from signalforge.config import QlibConfig

        engine = QlibEngine(QlibConfig(enabled=True))
        result = engine.predict(ohlcv_crypto_df, pred_len=5)
        assert not result.empty
        assert "predicted_return" in result.columns

    def test_qlib_fallback_futures(self, ohlcv_futures_df: pd.DataFrame) -> None:
        from signalforge.engines.qlib_engine import QlibEngine
        from signalforge.config import QlibConfig

        engine = QlibEngine(QlibConfig(enabled=True))
        result = engine.predict(ohlcv_futures_df, pred_len=5)
        assert not result.empty

    def test_qlib_engine_name(self) -> None:
        from signalforge.engines.qlib_engine import QlibEngine
        from signalforge.config import QlibConfig

        engine = QlibEngine(QlibConfig())
        assert engine.name == "qlib"


class TestChronosEngineFallback:
    """Test ChronosEngine fallback (Holt's linear trend exponential smoothing)."""

    def test_chronos_fallback_produces_predictions(self, ohlcv_stock_df: pd.DataFrame) -> None:
        from signalforge.engines.chronos_engine import ChronosEngine
        from signalforge.config import ChronosConfig

        engine = ChronosEngine(ChronosConfig(enabled=True))
        result = engine.predict(ohlcv_stock_df, pred_len=5)

        assert not result.empty
        assert "predicted_close" in result.columns
        assert len(result) == 5

    def test_chronos_fallback_has_quantiles(self, ohlcv_stock_df: pd.DataFrame) -> None:
        from signalforge.engines.chronos_engine import ChronosEngine
        from signalforge.config import ChronosConfig

        engine = ChronosEngine(ChronosConfig(enabled=True))
        result = engine.predict(ohlcv_stock_df, pred_len=5)

        # Fallback should produce high/low predictions
        assert "predicted_high" in result.columns
        assert "predicted_low" in result.columns

    def test_chronos_fallback_crypto(self, ohlcv_crypto_df: pd.DataFrame) -> None:
        from signalforge.engines.chronos_engine import ChronosEngine
        from signalforge.config import ChronosConfig

        engine = ChronosEngine(ChronosConfig(enabled=True))
        result = engine.predict(ohlcv_crypto_df, pred_len=5)
        assert not result.empty

    def test_chronos_engine_name(self) -> None:
        from signalforge.engines.chronos_engine import ChronosEngine
        from signalforge.config import ChronosConfig

        engine = ChronosEngine(ChronosConfig())
        assert engine.name == "chronos"


class TestAgentsEngineFallback:
    """Test AgentsEngine fallback (rule-based price-action sentiment)."""

    def test_agents_fallback_produces_direction(self, ohlcv_stock_df: pd.DataFrame) -> None:
        from signalforge.engines.agents_engine import AgentsEngine
        from signalforge.config import AgentsConfig

        engine = AgentsEngine(AgentsConfig(enabled=True))
        result = engine.predict(ohlcv_stock_df, pred_len=5)

        assert not result.empty
        assert "direction" in result.columns
        # Direction should be in [-1, 1]
        assert result["direction"].min() >= -1.0
        assert result["direction"].max() <= 1.0

    def test_agents_fallback_crypto(self, ohlcv_crypto_df: pd.DataFrame) -> None:
        from signalforge.engines.agents_engine import AgentsEngine
        from signalforge.config import AgentsConfig

        engine = AgentsEngine(AgentsConfig(enabled=True))
        result = engine.predict(ohlcv_crypto_df, pred_len=5)
        assert not result.empty
        assert "direction" in result.columns

    def test_agents_engine_name(self) -> None:
        from signalforge.engines.agents_engine import AgentsEngine
        from signalforge.config import AgentsConfig

        engine = AgentsEngine(AgentsConfig())
        assert engine.name == "agents"


class TestAllEnginesPipeline:
    """Test the full pipeline with all engines enabled (fallback mode)."""

    def test_all_engines_stock(self, ohlcv_stock_df: pd.DataFrame) -> None:
        """Run all 5 engines on stock data and combine."""
        from signalforge.config import QlibConfig, ChronosConfig, AgentsConfig
        from signalforge.engines.kronos_engine import KronosConfig, KronosEngine
        from signalforge.engines.qlib_engine import QlibEngine
        from signalforge.engines.chronos_engine import ChronosEngine
        from signalforge.engines.agents_engine import AgentsEngine
        from signalforge.engines.technical import compute_signals, compute_support_resistance
        from signalforge.ensemble.combiner import SignalCombiner
        from signalforge.ensemble.targets import TargetCalculator
        from signalforge.data.models import SupportResistance, TradeAction

        current_price = float(ohlcv_stock_df["close"].iloc[-1])

        # Run all 5 engines
        kronos = KronosEngine(KronosConfig(pred_len=5))
        kronos_pred = kronos.predict(ohlcv_stock_df, pred_len=5)

        qlib = QlibEngine(QlibConfig(enabled=True))
        qlib_pred = qlib.predict(ohlcv_stock_df, pred_len=5)

        chronos = ChronosEngine(ChronosConfig(enabled=True))
        chronos_pred = chronos.predict(ohlcv_stock_df, pred_len=5)

        agents = AgentsEngine(AgentsConfig(enabled=True))
        agents_pred = agents.predict(ohlcv_stock_df, pred_len=5)

        signals_df = compute_signals(ohlcv_stock_df)
        supports, resistances = compute_support_resistance(ohlcv_stock_df)

        # Combine all results
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

        if not qlib_pred.empty and "predicted_return" in qlib_pred.columns:
            pred_ret = float(qlib_pred["predicted_return"].iloc[-1])
            pred_close = current_price * (1 + pred_ret)
            engine_results["qlib"] = {
                "predicted_close": pred_close,
                "predicted_high": pred_close * 1.02,
                "predicted_low": pred_close * 0.98,
                "current_price": current_price,
            }

        if not chronos_pred.empty and "predicted_close" in chronos_pred.columns:
            engine_results["chronos"] = {
                "predicted_close": float(chronos_pred["predicted_close"].iloc[-1]),
                "predicted_high": float(chronos_pred["predicted_high"].max()),
                "predicted_low": float(chronos_pred["predicted_low"].min()),
                "current_price": current_price,
            }

        if not agents_pred.empty and "direction" in agents_pred.columns:
            engine_results["agents"] = {
                "signal": float(agents_pred["direction"].iloc[-1]),
            }

        # Must have at least 3 engines reporting
        assert len(engine_results) >= 3, f"Only {len(engine_results)} engines reported"

        weights = {
            "kronos": 0.40,
            "qlib": 0.20,
            "chronos": 0.15,
            "agents": 0.10,
            "technical": 0.15,
        }
        combiner = SignalCombiner(weights)
        combined = combiner.combine(engine_results)

        assert -1.0 <= combined.direction <= 1.0
        assert 0.0 <= combined.confidence <= 1.0

        calc = TargetCalculator()
        target = calc.calculate(
            symbol="AAPL",
            signal=combined,
            current_price=current_price,
            levels=SupportResistance(support=supports[0], resistance=resistances[0]),
        )

        assert target.action in (TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD)
        assert target.confidence > 0.0
