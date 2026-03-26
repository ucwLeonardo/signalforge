"""Main pipeline: fetch data → run engines → ensemble → generate targets."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
from loguru import logger

from signalforge.config import Config


def _classify_symbol(symbol: str) -> str:
    """Classify a symbol as stock, crypto, futures, or options."""
    from signalforge.data.models import parse_option_symbol

    if parse_option_symbol(symbol) is not None:
        return "options"
    if "/" in symbol:
        return "crypto"
    if symbol.endswith("=F"):
        return "futures"
    return "stock"


def _get_lookback_days(symbol_type: str, config: Config) -> int:
    if symbol_type == "crypto":
        return config.data.crypto_lookback_days
    if symbol_type == "futures":
        return config.data.futures_lookback_days
    if symbol_type == "options":
        return config.data.options_lookback_days
    return config.data.stocks_lookback_days


def run_pipeline(
    symbols: list[str],
    config: Config,
    interval: str = "1d",
    pred_len: int = 5,
    engines: Sequence[str] | None = None,
    use_store: bool = False,
    progress_cb: "Callable[[dict], None] | None" = None,
) -> list[dict]:
    """Run the full signal generation pipeline.

    Parameters
    ----------
    use_store:
        If True, use incremental data fetching via DataStore (parquet cache).
        First run downloads full history; subsequent runs fetch only new bars.
    progress_cb:
        Optional callback called with a progress dict at each stage.
        Dict keys: total, completed, symbol, stage, detail.

    Returns list of TradeTarget-like dicts for each symbol.
    """
    from signalforge.data.models import SupportResistance
    from signalforge.data.providers import get_provider
    from signalforge.ensemble.combiner import SignalCombiner
    from signalforge.ensemble.targets import TargetCalculator

    total = len(symbols)

    def _report(completed: int, symbol: str, stage: str, detail: str = "") -> None:
        if progress_cb is not None:
            progress_cb({
                "total": total,
                "completed": completed,
                "symbol": symbol,
                "stage": stage,
                "detail": detail,
            })

    # Set up incremental fetcher if requested
    fetcher = None
    if use_store:
        from signalforge.data.incremental import IncrementalFetcher
        from signalforge.data.store import DataStore

        fetcher = IncrementalFetcher(DataStore(config.data_dir))

    weights = {
        "kronos": config.ensemble.kronos_weight,
        "qlib": config.ensemble.qlib_weight,
        "chronos": config.ensemble.chronos_weight,
        "agents": config.ensemble.agents_weight,
        "technical": config.ensemble.technical_weight,
        "lstm": config.ensemble.lstm_weight,
        "gbm": config.ensemble.gbm_weight,
    }
    combiner = SignalCombiner(weights)
    calculator = TargetCalculator()
    all_targets = []

    for idx, symbol in enumerate(symbols):
        logger.info(f"Processing {symbol}")
        sym_type = _classify_symbol(symbol)
        _report(idx, symbol, "data", f"Downloading {sym_type} data...")

        # 1. Fetch data (incremental if store enabled, direct otherwise)
        try:
            lookback = _get_lookback_days(sym_type, config)
            exchange_id = config.data.crypto_exchange if sym_type == "crypto" else None

            if fetcher is not None:
                df = fetcher.fetch(
                    symbol, interval, lookback, exchange_id=exchange_id,
                )
            else:
                provider_kwargs = {}
                if exchange_id:
                    provider_kwargs["exchange_id"] = exchange_id
                provider = get_provider(symbol, **provider_kwargs)
                end = datetime.now()
                start = end - timedelta(days=lookback)
                df = provider.fetch(symbol, interval, start, end)

            if df.empty or len(df) < 30:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
                _report(idx, symbol, "data", f"Skipped: only {len(df)} bars")
                continue
        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}: {e}")
            _report(idx, symbol, "data", f"Failed: {e}")
            continue

        current_price = float(df["close"].iloc[-1])
        _report(idx, symbol, "data", f"Ready: {len(df)} bars, last=${current_price:,.2f}")
        engine_results: dict[str, dict] = {}

        # 2. Run prediction engines

        if engines is None or "kronos" in engines or "all" in engines:
            if config.kronos.enabled:
                try:
                    from signalforge.engines.kronos_engine import KronosEngine

                    kronos = KronosEngine(config.kronos)
                    predictions = kronos.predict(df, pred_len=pred_len)
                    if not predictions.empty:
                        engine_results["kronos"] = {
                            "type": "price",
                            "predicted_close": float(predictions["close"].iloc[-1]),
                            "predicted_high": float(predictions["high"].max()),
                            "predicted_low": float(predictions["low"].min()),
                            "predictions": predictions,
                        }
                except Exception as e:
                    logger.error(f"Kronos failed for {symbol}: {e}")

        # --- Qlib factor engine ---
        if engines is None or "qlib" in engines or "all" in engines:
            if config.qlib.enabled:
                try:
                    from signalforge.engines.qlib_engine import QlibEngine

                    qlib_eng = QlibEngine(config.qlib)
                    qlib_pred = qlib_eng.predict(df, pred_len=pred_len)
                    if not qlib_pred.empty and "predicted_return" in qlib_pred.columns:
                        pred_ret = float(qlib_pred["predicted_return"].iloc[-1])
                        pred_close = current_price * (1 + pred_ret)
                        engine_results["qlib"] = {
                            "type": "price",
                            "predicted_close": pred_close,
                            "predicted_high": pred_close * 1.02,
                            "predicted_low": pred_close * 0.98,
                        }
                except Exception as e:
                    logger.error(f"Qlib failed for {symbol}: {e}")

        # --- Chronos forecasting engine ---
        if engines is None or "chronos" in engines or "all" in engines:
            if config.chronos.enabled:
                try:
                    from signalforge.engines.chronos_engine import ChronosEngine

                    chronos_eng = ChronosEngine(config.chronos)
                    chronos_pred = chronos_eng.predict(df, pred_len=pred_len)
                    if not chronos_pred.empty and "predicted_close" in chronos_pred.columns:
                        engine_results["chronos"] = {
                            "type": "price",
                            "predicted_close": float(chronos_pred["predicted_close"].iloc[-1]),
                            "predicted_high": float(chronos_pred["predicted_high"].max()),
                            "predicted_low": float(chronos_pred["predicted_low"].min()),
                        }
                except Exception as e:
                    logger.error(f"Chronos failed for {symbol}: {e}")

        # NOTE: TradingAgents (Gemini) runs ONCE for top N after all assets are scored.
        # See post-loop section below.

        # --- LSTM engine ---
        if engines is None or "lstm" in engines or "all" in engines:
            if config.lstm.enabled:
                _report(idx, symbol, "lstm", "Training/inference LSTM seq2seq")
                try:
                    from signalforge.engines.lstm_engine import LSTMEngine

                    lstm_eng = LSTMEngine(config.lstm)
                    lstm_pred = lstm_eng.predict(df, pred_len=pred_len, symbol=symbol if use_store else None)
                    if not lstm_pred.empty:
                        engine_results["lstm"] = {
                            "type": "price",
                            "predicted_close": float(lstm_pred["close"].iloc[-1]),
                            "predicted_high": float(lstm_pred["high"].max()),
                            "predicted_low": float(lstm_pred["low"].min()),
                        }
                except Exception as e:
                    logger.error(f"LSTM failed for {symbol}: {e}")

        # --- GBM ensemble engine ---
        if engines is None or "gbm" in engines or "all" in engines:
            if config.gbm.enabled:
                _report(idx, symbol, "gbm", "Training/inference GBM (LightGBM)")
                try:
                    from signalforge.engines.gbm_engine import GBMEnsembleEngine

                    gbm_eng = GBMEnsembleEngine(config.gbm)
                    gbm_pred = gbm_eng.predict(df, pred_len=pred_len, symbol=symbol if use_store else None)
                    if not gbm_pred.empty and "predicted_return" in gbm_pred.columns:
                        pred_ret = float(gbm_pred["predicted_return"].iloc[-1])
                        pred_close = current_price * (1 + pred_ret)
                        engine_results["gbm"] = {
                            "type": "price",
                            "predicted_close": pred_close,
                            "predicted_high": pred_close * 1.02,
                            "predicted_low": pred_close * 0.98,
                        }
                except Exception as e:
                    logger.error(f"GBM failed for {symbol}: {e}")

        # --- Technical analysis ---
        if engines is None or "technical" in engines or "all" in engines:
            _report(idx, symbol, "technical", "RSI, MACD, BBands, S/R levels")
            try:
                from signalforge.engines.technical import TechnicalEngine, compute_signals, compute_support_resistance

                signals_df = compute_signals(df)
                supports, resistances = compute_support_resistance(df)

                support = supports[0] if supports else current_price * 0.95
                resistance = resistances[0] if resistances else current_price * 1.05

                if not signals_df.empty:
                    last_signal = float(signals_df["signal_strength"].iloc[-1])
                    engine_results["technical"] = {
                        "type": "signal",
                        "signal_strength": last_signal,
                        "signal": last_signal,
                        "support": support,
                        "resistance": resistance,
                    }
            except Exception as e:
                logger.error(f"Technical analysis failed for {symbol}: {e}")

        if not engine_results:
            logger.warning(f"No engine produced results for {symbol}")
            continue

        # 3. Combine signals - inject current_price into price-type engines
        active_engines = sorted(engine_results.keys())
        _report(idx, symbol, "ensemble", f"Combining: {', '.join(active_engines)}")
        for eng_name, eng_result in engine_results.items():
            if eng_result.get("type") == "price":
                eng_result["current_price"] = current_price
        combined = combiner.combine(engine_results)

        # 4. Calculate targets
        support = engine_results.get("technical", {}).get("support", current_price * 0.95)
        resistance = engine_results.get("technical", {}).get("resistance", current_price * 1.05)
        levels = SupportResistance(support=support, resistance=resistance)

        target = calculator.calculate(
            symbol=symbol,
            signal=combined,
            current_price=current_price,
            levels=levels,
            horizon_days=pred_len,
        )

        all_targets.append(target)
        _report(idx + 1, symbol, "done",
                f"{target.action.value} conf={target.confidence:.0%} "
                f"entry=${target.entry_price:,.2f}")
        logger.info(
            f"{symbol}: {target.action} | "
            f"Entry: {target.entry_price:.2f} | "
            f"Target: {target.target_price:.2f} | "
            f"Stop: {target.stop_loss:.2f} | "
            f"Conf: {target.confidence:.0%}"
        )

    # --- Post-loop: TradingAgents (Gemini) for top N candidates ---
    if (engines is None or "agents" in engines or "all" in engines) and config.agents.enabled:
        if all_targets:
            # Sort by confidence, take top candidates for LLM review
            sorted_targets = sorted(all_targets, key=lambda t: t.confidence, reverse=True)
            top_n = min(10, len(sorted_targets))
            top_symbols = [t.symbol for t in sorted_targets[:top_n]]

            _report(total, "", "agents",
                    f"Gemini LLM reviewing top {top_n}: {', '.join(top_symbols)}")

            try:
                from signalforge.engines.agents_engine import AgentsEngine

                agents_eng = AgentsEngine(config.agents)

                # Build a summary prompt with all top candidates
                summary_lines = []
                for t in sorted_targets[:top_n]:
                    summary_lines.append(
                        f"{t.symbol}: {t.action.value} conf={t.confidence:.0%} "
                        f"entry=${t.entry_price:,.2f} target=${t.target_price:,.2f} "
                        f"stop=${t.stop_loss:,.2f}"
                    )
                # Use agents engine's predict with a synthetic summary DataFrame
                # The engine will format this into a Gemini prompt
                import pandas as _pd
                summary_df = _pd.DataFrame({
                    "close": [t.entry_price for t in sorted_targets[:top_n]],
                    "open": [t.entry_price for t in sorted_targets[:top_n]],
                    "high": [t.target_price for t in sorted_targets[:top_n]],
                    "low": [t.stop_loss for t in sorted_targets[:top_n]],
                    "volume": [1e6] * top_n,
                }, index=[t.symbol for t in sorted_targets[:top_n]])

                agents_pred = agents_eng.predict(summary_df, pred_len=pred_len)
                if not agents_pred.empty and "direction" in agents_pred.columns:
                    llm_direction = float(agents_pred["direction"].iloc[-1])
                    logger.info(f"TradingAgents LLM verdict: direction={llm_direction:+.2f}")
                    # Adjust confidence of top targets based on LLM agreement
                    for t in sorted_targets[:top_n]:
                        if (llm_direction > 0 and t.action.value == "BUY") or \
                           (llm_direction < 0 and t.action.value == "SELL"):
                            # LLM agrees — boost confidence slightly
                            idx_t = all_targets.index(t)
                            from dataclasses import replace as _replace
                            all_targets[idx_t] = _replace(t,
                                confidence=min(t.confidence * 1.1, 0.95),
                                rationale=t.rationale + " | LLM: agrees with direction"
                            )
                        else:
                            idx_t = all_targets.index(t)
                            from dataclasses import replace as _replace
                            all_targets[idx_t] = _replace(t,
                                confidence=t.confidence * 0.9,
                                rationale=t.rationale + " | LLM: disagrees with direction"
                            )
            except Exception as e:
                logger.error(f"TradingAgents post-analysis failed: {e}")

    _report(total, "", "complete", f"{len(all_targets)} signals from {total} assets")
    return all_targets
