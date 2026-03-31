"""Main pipeline: fetch data → run engines → ensemble → generate targets."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
from loguru import logger

from signalforge.config import Config, get_trading_params


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


def _check_data_quality(df: pd.DataFrame, symbol: str, sym_type: str) -> str | None:
    """Return a reason string if data quality is too low, else None.

    Freshness rules:
      - Stocks/Futures: last bar must be from the most recent US trading day.
      - Crypto: bars only feed models; allow up to 7 days stale.
        Live price (fetched separately) provides minute-level freshness.
    """
    if len(df) < 30:
        return None  # Already handled by min-bars check

    # --- Freshness check ---
    if "timestamp" in df.columns:
        last_ts = pd.Timestamp(df["timestamp"].iloc[-1])
        last_bar_date = last_ts.date()
        if last_ts.tzinfo is not None:
            last_bar_date = last_ts.tz_convert("UTC").date()

        if sym_type in ("stock", "futures"):
            from signalforge.data.calendar import last_trading_day
            expected = last_trading_day()
            if last_bar_date < expected:
                return f"Stale (last: {last_bar_date}, expected: {expected})"
        elif sym_type == "crypto":
            # Crypto bars are for model training only; live price handles
            # current pricing. Allow up to 7 days stale.
            if last_ts.tzinfo is not None:
                now = pd.Timestamp.now(tz="UTC")
            else:
                now = pd.Timestamp.now()
            staleness_days = (now - last_ts).days
            if staleness_days > 7:
                return f"Stale ({staleness_days}d, last: {last_bar_date})"

    # --- Sparsity check ---
    if "timestamp" in df.columns and len(df) >= 30:
        recent = df.tail(30)
        ts = pd.to_datetime(recent["timestamp"])
        date_range = (ts.iloc[-1] - ts.iloc[0]).days
        if date_range > 0:
            fill_rate = len(recent) / date_range
            if sym_type == "crypto" and fill_rate < 0.4:
                return f"Sparse ({len(recent)} bars over {date_range}d, fill={fill_rate:.0%})"
            if sym_type in ("stock", "futures") and fill_rate < 0.3:
                return f"Sparse ({len(recent)} bars over {date_range}d, fill={fill_rate:.0%})"

    # --- Flat/illiquid candle check ---
    recent = df.tail(30)
    flat_count = ((recent["open"] == recent["close"]) &
                  (recent["high"] == recent["low"])).sum()
    if flat_count > 15:
        return f"Illiquid ({flat_count}/30 flat candles)"

    return None


def _get_lookback_days(symbol_type: str, config: Config) -> int:
    if symbol_type == "crypto":
        return config.data.crypto_lookback_days
    if symbol_type == "futures":
        return config.data.futures_lookback_days
    if symbol_type == "options":
        return config.data.options_lookback_days
    return config.data.stocks_lookback_days


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Average True Range from OHLC data."""
    if len(df) < 2:
        return 0.0
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    tr_values = []
    for i in range(1, len(df)):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        tr_values.append(tr)
    if not tr_values:
        return 0.0
    return float(sum(tr_values[-period:]) / min(len(tr_values), period))


def run_pipeline(
    symbols: list[str],
    config: Config,
    interval: str = "1d",
    pred_len: int = 5,
    engines: Sequence[str] | None = None,
    use_store: bool = False,
    progress_cb: "Callable[[dict], None] | None" = None,
    cancel_flag: "Callable[[], bool] | None" = None,
) -> list[dict]:
    """Run the full signal generation pipeline in sequential phases.

    Phase 1: Download all data (incremental)
    Phase 2: Train/predict with all engines per symbol
    Phase 3: TradingAgents LLM review for top N

    Parameters
    ----------
    use_store:
        If True, use incremental data fetching via DataStore (parquet cache).
    progress_cb:
        Optional callback called with a progress dict at each stage.
    cancel_flag:
        Optional callable returning True if the scan should be cancelled.

    Returns list of TradeTarget objects.
    """
    from signalforge.data.models import SupportResistance
    from signalforge.data.providers import get_provider
    from signalforge.ensemble.combiner import SignalCombiner
    from signalforge.ensemble.targets import TargetCalculator

    total = len(symbols)
    # Phase weights for progress: data=40%, prices=10%, engines=50%
    _phase1_weight = 40  # data download
    _phase15_weight = 10  # live price fetch
    _phase2_weight = 50  # engine processing

    def _report(completed: int, symbol: str, stage: str, detail: str = "",
                phase_pct: float = 0.0,
                step: int = 0, step_total: int = 0) -> None:
        """Report progress with fine-grained phase percentage.

        phase_pct: 0.0-100.0 overall progress across all phases.
        step/step_total: current operation progress (e.g. 5/32 in engine phase).
        """
        if progress_cb is not None:
            progress_cb({
                "total": total,
                "completed": completed,
                "symbol": symbol,
                "stage": stage,
                "detail": detail,
                "phase_pct": phase_pct,
                "step": step,
                "step_total": step_total,
            })

    def _cancelled() -> bool:
        return cancel_flag is not None and cancel_flag()

    # Set up incremental fetcher if requested
    fetcher = None
    if use_store:
        from signalforge.data.incremental import IncrementalFetcher
        from signalforge.data.store import DataStore

        fetcher = IncrementalFetcher(DataStore(config.data_dir), cancel_flag=cancel_flag)

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

    # ====================================================================
    # PHASE 1: Download all data (two-pass: cache first, then API)
    # ====================================================================
    _report(0, "", "discovery", f"Phase 1: Loading data for {total} assets...",
            phase_pct=0.0)
    symbol_data: dict[str, "pd.DataFrame"] = {}  # symbol -> DataFrame

    # --- Pass A: serve fresh cache instantly (no network) ---------------
    stale_symbols: list[str] = []  # symbols needing API fetch
    cached_count = 0

    if fetcher is not None:
        for idx, symbol in enumerate(symbols):
            if _cancelled():
                _report(idx, "", "cancelled", "Scan cancelled by user")
                return []

            sym_type = _classify_symbol(symbol)
            sym_label = sym_type.capitalize()
            df, source = fetcher.check_cache(symbol, interval)

            if df is not None:
                # Cache hit (fresh or known-empty)
                p1_pct = ((idx + 1) / max(total, 1)) * _phase1_weight
                if df.empty or len(df) < 30:
                    _report(idx, symbol, "data_skipped",
                            f"{sym_label} · Skipped (known empty)" if source == "empty_skip"
                            else f"{sym_label} · Insufficient data ({len(df)} bars)",
                            phase_pct=p1_pct)
                else:
                    quality_issue = _check_data_quality(df, symbol, sym_type)
                    if quality_issue:
                        _report(idx, symbol, "data_skipped",
                                f"{sym_label} · {quality_issue}",
                                phase_pct=p1_pct)
                    else:
                        current_price = float(df["close"].iloc[-1])
                        symbol_data[symbol] = df
                        cached_count += 1
                        _report(idx, symbol, "data_cached",
                                f"{sym_label} · {len(df)} bars, ${current_price:,.2f}",
                                phase_pct=p1_pct)
            else:
                stale_symbols.append(symbol)

        if stale_symbols:
            stale_pct = (cached_count / max(total, 1)) * _phase1_weight
            _report(cached_count, "", "data",
                    f"Cache: {cached_count}/{total} ready · Fetching {len(stale_symbols)} stale...",
                    phase_pct=stale_pct)

    else:
        # No fetcher — all symbols need full download
        stale_symbols = list(symbols)

    # --- Pass B: fetch only stale/missing symbols via API ---------------
    for fetch_idx, symbol in enumerate(stale_symbols):
        if _cancelled():
            _report(0, "", "cancelled", "Scan cancelled by user")
            return []

        # Map back to the original index for consistent progress
        orig_idx = symbols.index(symbol) if symbol in symbols else fetch_idx
        sym_type = _classify_symbol(symbol)
        sym_label = sym_type.capitalize()

        overall_done = cached_count + fetch_idx
        p1_pct = (overall_done / max(total, 1)) * _phase1_weight
        _report(overall_done, symbol, "data",
                f"{sym_label} · Fetching ({fetch_idx+1}/{len(stale_symbols)})...",
                phase_pct=p1_pct, step=overall_done+1, step_total=total)

        try:
            lookback = _get_lookback_days(sym_type, config)

            if fetcher is not None:
                df = fetcher.fetch(symbol, interval, lookback)
                fetch_source = fetcher.last_fetch_source
            else:
                provider = get_provider(symbol)
                end = datetime.now()
                start = end - timedelta(days=lookback)
                df = provider.fetch(symbol, interval, start, end)
                fetch_source = "full"

            overall_after = cached_count + fetch_idx + 1
            p1_done_pct = (overall_after / max(total, 1)) * _phase1_weight
            if df.empty or len(df) < 30:
                if fetcher is not None and fetcher.last_fetch_source == "empty_skip":
                    _report(overall_after, symbol, "data_skipped",
                            f"{sym_label} · Skipped (known empty)",
                            phase_pct=p1_done_pct)
                else:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
                    _report(overall_after, symbol, "data_skipped",
                            f"{sym_label} · Insufficient data ({len(df)} bars)",
                            phase_pct=p1_done_pct)
                continue

            quality_issue = _check_data_quality(df, symbol, sym_type)
            if quality_issue:
                logger.warning(f"Low quality data for {symbol}: {quality_issue}")
                _report(overall_after, symbol, "data_skipped",
                        f"{sym_label} · {quality_issue}",
                        phase_pct=p1_done_pct)
                continue

            current_price = float(df["close"].iloc[-1])
            symbol_data[symbol] = df
            if fetch_source in ("cache", "cache_fallback"):
                _report(overall_after, symbol, "data_cached",
                        f"{sym_label} · {len(df)} bars, ${current_price:,.2f}",
                        phase_pct=p1_done_pct)
            elif fetch_source == "incremental":
                new_bars = fetcher.last_new_bars if fetcher else 0
                _report(overall_after, symbol, "data_done",
                        f"{sym_label} · Fetched +{new_bars} new bars ({len(df)} total, ${current_price:,.2f})",
                        phase_pct=p1_done_pct)
            elif fetch_source == "full":
                _report(overall_after, symbol, "data_done",
                        f"{sym_label} · Downloaded {len(df)} bars (${current_price:,.2f})",
                        phase_pct=p1_done_pct)

        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}: {e}")
            overall_after = cached_count + fetch_idx + 1
            p1_done_pct = (overall_after / max(total, 1)) * _phase1_weight
            _report(overall_after, symbol, "data_error", f"{sym_label} · {e}",
                    phase_pct=p1_done_pct)

    ready_count = len(symbol_data)
    _report(total, "", "discovery", f"Phase 1 complete: {ready_count}/{total} assets ready",
            phase_pct=_phase1_weight)
    logger.info(f"Data download complete: {ready_count}/{total} assets")

    # ====================================================================
    # PHASE 1.5: Fetch live prices for all ready symbols
    # ====================================================================
    live_prices: dict[str, float] = {}
    if symbol_data:
        _report(total, "", "prices", f"Fetching live prices...",
                phase_pct=_phase1_weight, step=0, step_total=ready_count)

        def _on_price_progress(fetched: int, price_total: int) -> None:
            pct = _phase1_weight + (fetched / max(price_total, 1)) * _phase15_weight
            _report(total, "", "prices",
                    f"Live prices: {fetched}/{price_total} fetched...",
                    phase_pct=pct, step=fetched, step_total=price_total)

        try:
            from signalforge.paper.prices import fetch_prices
            live_prices = fetch_prices(list(symbol_data.keys()),
                                       progress_cb=_on_price_progress)
            logger.info(f"Live prices fetched: {len(live_prices)}/{ready_count}")
        except Exception as e:
            logger.warning(f"Live price fetch failed, using bar close: {e}")

        _report(total, "", "prices",
                f"Live prices: {len(live_prices)}/{ready_count} fetched",
                phase_pct=_phase1_weight + _phase15_weight)

        # Report missing crypto live prices (crypto REQUIRES live price)
        crypto_missing = [
            s for s in symbol_data
            if _classify_symbol(s) == "crypto" and s not in live_prices
        ]
        if crypto_missing:
            _report(total, "", "prices",
                    f"WARNING: No live price for {len(crypto_missing)} crypto: "
                    f"{', '.join(crypto_missing[:5])}"
                    + (f" +{len(crypto_missing)-5} more" if len(crypto_missing) > 5 else ""),
                    phase_pct=_phase1_weight + _phase15_weight)

    # ====================================================================
    # PHASE 2: Train/predict engines for each symbol
    # ====================================================================
    all_targets = []
    p2_base = _phase1_weight + _phase15_weight  # 50% already done
    p2_total = max(ready_count, 1)

    for idx, (symbol, df) in enumerate(symbol_data.items()):
        if _cancelled():
            _report(total, "", "cancelled", "Scan cancelled by user")
            return all_targets

        sym_type = _classify_symbol(symbol)
        bar_close = float(df["close"].iloc[-1])
        symbol_atr = _compute_atr(df)
        trading_params = get_trading_params(sym_type)

        # Crypto REQUIRES live price — never use stale bar close
        if sym_type == "crypto" and symbol not in live_prices:
            logger.warning(f"No live price for crypto {symbol}, skipping")
            p2_pct = p2_base + ((idx + 1) / p2_total) * _phase2_weight
            _report(total, symbol, "data_skipped",
                    f"Crypto · No live price available",
                    phase_pct=p2_pct)
            continue

        # Prefer live price over bar close
        current_price = live_prices.get(symbol, bar_close)
        price_source = "live" if symbol in live_prices else "bar"

        # Warn when live price diverges >5% from bar close (bad Polygon data)
        if symbol in live_prices and bar_close > 0:
            divergence = abs(current_price - bar_close) / bar_close
            if divergence > 0.05:
                logger.warning(
                    f"{symbol}: live ${current_price:,.2f} diverges "
                    f"{divergence:.1%} from bar close ${bar_close:,.2f} — "
                    f"Polygon daily data may be from illiquid source")
            elif divergence > 0.01:
                logger.info(
                    f"Processing {symbol}: live=${current_price:,.2f} "
                    f"(bar=${bar_close:,.2f}, diff={((current_price - bar_close) / bar_close):+.1%})")
            else:
                logger.info(f"Processing {symbol}: ${current_price:,.2f} ({price_source})")
        else:
            logger.info(f"Processing {symbol}: ${current_price:,.2f} ({price_source})")

        engine_results: dict[str, dict] = {}
        # Phase 2 progress for this symbol
        p2_sym_start = p2_base + (idx / p2_total) * _phase2_weight
        p2_sym_end = p2_base + ((idx + 1) / p2_total) * _phase2_weight

        # --- Kronos ---
        if engines is None or "kronos" in engines or "all" in engines:
            if config.kronos.enabled:
                if _cancelled():
                    _report(total, "", "cancelled", "Scan cancelled by user")
                    return all_targets
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

        # --- Qlib ---
        if engines is None or "qlib" in engines or "all" in engines:
            if config.qlib.enabled:
                if _cancelled():
                    _report(total, "", "cancelled", "Scan cancelled by user")
                    return all_targets
                try:
                    from signalforge.engines.qlib_engine import QlibEngine
                    qlib_eng = QlibEngine(config.qlib)
                    qlib_pred = qlib_eng.predict(df, pred_len=pred_len)
                    if not qlib_pred.empty and "predicted_return" in qlib_pred.columns:
                        pred_ret = float(qlib_pred["predicted_return"].iloc[-1])
                        pred_close = current_price * (1 + pred_ret)
                        # Use ATR-based range instead of hardcoded +/-2%
                        qlib_range = symbol_atr * 1.5 if symbol_atr > 0 else pred_close * 0.02
                        engine_results["qlib"] = {
                            "type": "price",
                            "predicted_close": pred_close,
                            "predicted_high": pred_close + qlib_range,
                            "predicted_low": pred_close - qlib_range,
                        }
                except Exception as e:
                    logger.error(f"Qlib failed for {symbol}: {e}")

        # --- Chronos ---
        if engines is None or "chronos" in engines or "all" in engines:
            if config.chronos.enabled:
                if _cancelled():
                    _report(total, "", "cancelled", "Scan cancelled by user")
                    return all_targets
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

        # --- LSTM ---
        if engines is None or "lstm" in engines or "all" in engines:
            if config.lstm.enabled:
                if _cancelled():
                    _report(total, "", "cancelled", "Scan cancelled by user")
                    return all_targets
                _report(idx, symbol, "lstm", f"LSTM",
                        phase_pct=p2_sym_start + (p2_sym_end - p2_sym_start) * 0.2,
                        step=idx+1, step_total=ready_count)
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

        # --- GBM ---
        if engines is None or "gbm" in engines or "all" in engines:
            if config.gbm.enabled:
                if _cancelled():
                    _report(total, "", "cancelled", "Scan cancelled by user")
                    return all_targets
                _report(idx, symbol, "gbm", f"GBM",
                        phase_pct=p2_sym_start + (p2_sym_end - p2_sym_start) * 0.5,
                        step=idx+1, step_total=ready_count)
                try:
                    from signalforge.engines.gbm_engine import GBMEnsembleEngine
                    gbm_eng = GBMEnsembleEngine(config.gbm)
                    gbm_pred = gbm_eng.predict(df, pred_len=pred_len, symbol=symbol if use_store else None)
                    if not gbm_pred.empty and "predicted_return" in gbm_pred.columns:
                        pred_ret = float(gbm_pred["predicted_return"].iloc[-1])
                        pred_close = current_price * (1 + pred_ret)
                        # Use ATR-based range instead of hardcoded +/-2%
                        gbm_range = symbol_atr * 1.5 if symbol_atr > 0 else pred_close * 0.02
                        engine_results["gbm"] = {
                            "type": "price",
                            "predicted_close": pred_close,
                            "predicted_high": pred_close + gbm_range,
                            "predicted_low": pred_close - gbm_range,
                        }
                except Exception as e:
                    logger.error(f"GBM failed for {symbol}: {e}")

        # --- Technical ---
        if engines is None or "technical" in engines or "all" in engines:
            if _cancelled():
                _report(total, "", "cancelled", "Scan cancelled by user")
                return all_targets
            _report(idx, symbol, "technical", "RSI, MACD, BBands, S/R",
                    phase_pct=p2_sym_start + (p2_sym_end - p2_sym_start) * 0.7,
                    step=idx+1, step_total=ready_count)
            try:
                from signalforge.engines.technical import TechnicalEngine, compute_signals, compute_support_resistance
                signals_df = compute_signals(df)
                supports, resistances = compute_support_resistance(df)
                sr_fallback = symbol_atr * 2 if symbol_atr > 0 else current_price * 0.03
                support = supports[0] if supports else current_price - sr_fallback
                resistance = resistances[0] if resistances else current_price + sr_fallback
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

        # Combine signals
        active_engines = sorted(engine_results.keys())
        _report(idx, symbol, "ensemble", f"Combining: {', '.join(active_engines)}",
                phase_pct=p2_sym_start + (p2_sym_end - p2_sym_start) * 0.9,
                step=idx+1, step_total=ready_count)
        for eng_name, eng_result in engine_results.items():
            if eng_result.get("type") == "price":
                eng_result["current_price"] = current_price
        combined = combiner.combine(engine_results)

        # Calculate targets
        sr_fb = symbol_atr * 2 if symbol_atr > 0 else current_price * 0.03
        support = engine_results.get("technical", {}).get("support", current_price - sr_fb)
        resistance = engine_results.get("technical", {}).get("resistance", current_price + sr_fb)
        levels = SupportResistance(support=support, resistance=resistance)

        target = calculator.calculate(
            symbol=symbol,
            signal=combined,
            current_price=current_price,
            levels=levels,
            atr=symbol_atr,
            horizon_days=trading_params.horizon_days,
            asset_type=sym_type,
            trading_params=trading_params,
        )

        all_targets.append(target)
        _report(idx + 1, symbol, "done",
                f"{target.action.value} conf={target.confidence:.0%} "
                f"entry=${target.entry_price:,.2f}",
                phase_pct=p2_sym_end,
                step=idx+1, step_total=ready_count)
        logger.info(
            f"{symbol}: {target.action} | "
            f"Entry: {target.entry_price:.2f} | "
            f"Target: {target.target_price:.2f} | "
            f"Stop: {target.stop_loss:.2f} | "
            f"Conf: {target.confidence:.0%}"
        )

    # --- Post-loop: TradingAgents (Gemini) for top N candidates ---
    if _cancelled():
        _report(len(symbol_data), "", "cancelled", "Scan cancelled by user")
        return all_targets
    if (engines is None or "agents" in engines or "all" in engines) and config.agents.enabled:
        if all_targets:
            # Sort by confidence, take top candidates for LLM review
            sorted_targets = sorted(all_targets, key=lambda t: t.confidence, reverse=True)
            top_n = min(10, len(sorted_targets))
            top_symbols = [t.symbol for t in sorted_targets[:top_n]]

            _report(total, "", "agents",
                    f"Gemini LLM reviewing top {top_n}: {', '.join(top_symbols)}",
                    phase_pct=95.0)

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

    _report(total, "", "complete", f"{len(all_targets)} signals from {total} assets",
            phase_pct=100.0)
    return all_targets
