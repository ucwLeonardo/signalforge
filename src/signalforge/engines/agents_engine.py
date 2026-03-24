"""TradingAgents multi-agent LLM analysis engine.

Wraps TauricResearch/TradingAgents for qualitative multi-agent analysis.
If the ``tradingagents`` package is not available on the Python path the
engine falls back to a simple rule-based sentiment derived from price action
so that SignalForge remains usable without LLM API keys configured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from signalforge.engines.base import PredictionEngine

# ---------------------------------------------------------------------------
# TradingAgents lazy import
# ---------------------------------------------------------------------------

_TRADING_AGENTS_AVAILABLE: bool = False

try:
    from tradingagents.graph.trading_graph import TradingAgentsGraph  # type: ignore[import-untyped]

    _TRADING_AGENTS_AVAILABLE = True
    logger.info(
        "TradingAgents library detected -- multi-agent LLM analysis available."
    )
except ImportError:
    logger.warning(
        "TradingAgents library not found. Install it with:\n"
        "  pip install tradingagents\n"
        "Falling back to rule-based price-action sentiment."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentsConfig:
    """All tuneable knobs for :class:`AgentsEngine`.

    Attributes:
        enabled: Whether the agents engine is active.
        llm_provider: LLM backend -- ``"openai"``, ``"anthropic"``,
            ``"google"``, or ``"ollama"``.
        deep_think_model: Model identifier for complex reasoning tasks
            (debate synthesis, risk assessment).
        quick_think_model: Model identifier for lighter tasks (individual
            analyst reports, quick summaries).
        max_debate_rounds: Maximum rounds of bull/bear debate.
        max_risk_rounds: Maximum rounds of risk-assessment refinement.
    """

    enabled: bool = False
    llm_provider: str = "anthropic"
    deep_think_model: str = "claude-sonnet-4-6"
    quick_think_model: str = "claude-haiku-4-5-20251001"
    max_debate_rounds: int = 1
    max_risk_rounds: int = 1


# ---------------------------------------------------------------------------
# Decision mapping
# ---------------------------------------------------------------------------

# Ordered longest-first so "STRONG BUY" matches before "BUY", etc.
_DECISION_SCORE: tuple[tuple[str, float], ...] = (
    ("STRONG BUY", +0.9),
    ("STRONG SELL", -0.9),
    ("OVERWEIGHT", +0.4),
    ("UNDERWEIGHT", -0.4),
    ("BUY", +0.8),
    ("SELL", -0.8),
    ("HOLD", 0.0),
)


def _parse_decision_score(decision: str) -> float:
    """Map a TradingAgents decision string to a numeric direction score."""
    normalised = decision.strip().upper()
    for key, score in _DECISION_SCORE:
        if key in normalised:
            return score
    logger.warning(
        "Unrecognised TradingAgents decision '{}'; defaulting to HOLD (0.0)",
        decision,
    )
    return 0.0


# ---------------------------------------------------------------------------
# Fallback indicators (pure numpy / pandas)
# ---------------------------------------------------------------------------


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(
        alpha=1.0 / length, min_periods=length, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / length, min_periods=length, adjust=False
    ).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _price_action_sentiment(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Derive a simple sentiment signal from recent price action.

    Rules
    -----
    * 5-day return > +3 % **and** RSI < 70  =>  bullish  (+0.6)
    * 5-day return < -3 % **and** RSI > 30  =>  bearish  (-0.6)
    * Otherwise                              =>  neutral  ( 0.0)

    Returns a dict with ``direction``, ``confidence``, and ``rationale``.
    """
    close = df["close"] if "close" in df.columns else df["Close"]
    close = close.astype(np.float64)

    if len(close) < 15:
        return {
            "direction": 0.0,
            "confidence": 0.1,
            "rationale": "Insufficient history for price-action sentiment.",
        }

    ret_5d = (close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0
    rsi_series = _rsi(close)
    current_rsi = float(rsi_series.iloc[-1])

    if ret_5d > 3.0 and current_rsi < 70.0:
        direction = 0.6
        label = "bullish"
    elif ret_5d < -3.0 and current_rsi > 30.0:
        direction = -0.6
        label = "bearish"
    else:
        direction = 0.0
        label = "neutral"

    confidence = min(abs(ret_5d) / 10.0, 1.0) * 0.5 + 0.1

    rationale = (
        f"Price-action fallback: {label}. "
        f"5d return={ret_5d:+.2f}%, RSI(14)={current_rsi:.1f}. "
        f"(TradingAgents unavailable -- using rule-based heuristic.)"
    )

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_REPORT_KEYS: tuple[str, ...] = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
)


def _extract_rationale(final_state: dict[str, Any]) -> str:
    """Combine key analyst findings from the TradingAgents final state."""
    parts: list[str] = []

    for key in _REPORT_KEYS:
        report = final_state.get(key)
        if report and isinstance(report, str):
            # Take the first meaningful paragraph (up to ~300 chars)
            snippet = report.strip()[:300]
            if snippet:
                parts.append(f"[{key}] {snippet}")

    # Include debate judge decision if present
    debate = final_state.get("investment_debate_state")
    if isinstance(debate, dict):
        judge = debate.get("judge_decision")
        if judge and isinstance(judge, str):
            parts.append(f"[debate_judge] {judge.strip()[:300]}")

    # Include final trade decision text
    decision_text = final_state.get("final_trade_decision")
    if decision_text and isinstance(decision_text, str):
        parts.append(f"[final_decision] {decision_text.strip()[:300]}")

    if not parts:
        return "No detailed analyst reports available."
    return "\n".join(parts)


def _extract_analyst_reports(final_state: dict[str, Any]) -> dict[str, str]:
    """Pull individual analyst reports out of the TradingAgents state."""
    reports: dict[str, str] = {}
    for key in _REPORT_KEYS:
        value = final_state.get(key)
        if value and isinstance(value, str):
            reports[key] = value

    # Include structured sub-reports
    for nested_key, sub_keys in (
        ("investment_debate_state", ("judge_decision", "current_response")),
        ("risk_debate_state", ("judge_decision",)),
    ):
        nested = final_state.get(nested_key)
        if isinstance(nested, dict):
            for sk in sub_keys:
                val = nested.get(sk)
                if val and isinstance(val, str):
                    reports[f"{nested_key}.{sk}"] = val

    plan = final_state.get("investment_plan")
    if plan and isinstance(plan, str):
        reports["investment_plan"] = plan

    return reports


def _call_with_retries(
    fn: Any,
    *args: Any,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Call *fn* with exponential-backoff retries on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            # Only retry on likely-transient errors (rate limits, timeouts)
            exc_str = str(exc).lower()
            transient = any(
                tok in exc_str
                for tok in ("rate", "limit", "timeout", "429", "503", "overloaded")
            )
            if not transient or attempt == max_retries:
                logger.error(
                    "TradingAgents call failed (attempt {}/{}): {}",
                    attempt,
                    max_retries,
                    exc,
                )
                raise
            wait = backoff_base**attempt
            logger.warning(
                "Transient error on attempt {}/{}: {}. Retrying in {:.1f}s ...",
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
    # Unreachable, but satisfies the type checker
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AgentsEngine(PredictionEngine):
    """Prediction engine backed by TauricResearch/TradingAgents.

    When the ``tradingagents`` library is importable the engine spins up a
    multi-agent LLM pipeline (fundamental, sentiment, news, technical
    analysts + risk manager + debate) and converts the qualitative verdict
    into a numeric signal.  Otherwise it transparently falls back to a
    simple rule-based price-action heuristic.
    """

    def __init__(
        self,
        config: AgentsConfig | None = None,
        **overrides: Any,
    ) -> None:
        cfg = config or AgentsConfig()
        if overrides:
            cfg = AgentsConfig(
                **{
                    fld.name: overrides.get(fld.name, getattr(cfg, fld.name))
                    for fld in cfg.__dataclass_fields__.values()
                }
            )
        self._config = cfg
        self._graph: Any | None = None  # lazy-loaded

        logger.debug(
            "AgentsEngine initialised (tradingagents_available={}, provider={})",
            _TRADING_AGENTS_AVAILABLE,
            cfg.llm_provider,
        )

    # -- public API ---------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return "agents"

    @property
    def config(self) -> AgentsConfig:
        return self._config

    @property
    def is_agents_available(self) -> bool:
        return _TRADING_AGENTS_AVAILABLE and self._config.enabled

    # ---- predict ----------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        pred_len: int = 1,
    ) -> pd.DataFrame:
        """Generate a qualitative directional signal from price history.

        Parameters
        ----------
        df:
            Historical OHLCV dataframe (must contain ``close`` at minimum).
        pred_len:
            Number of forward periods the signal applies to (used only for
            labelling; the analysis itself is a single-shot assessment).

        Returns
        -------
        pd.DataFrame
            A single-row (or *pred_len*-row) DataFrame with columns:
            ``direction`` (-1..1), ``confidence`` (0..1), ``rationale`` (str).
        """
        if self.is_agents_available:
            # Attempt to infer symbol from dataframe metadata
            symbol = getattr(df, "name", None) or df.attrs.get("symbol", "UNKNOWN")
            date_str = str(pd.Timestamp.now().date())
            if hasattr(df.index, "max") and isinstance(
                df.index, pd.DatetimeIndex
            ):
                date_str = str(df.index.max().date())

            try:
                result = self.analyze(str(symbol), date_str)
                signal = {
                    "direction": result["direction"],
                    "confidence": result["confidence"],
                    "rationale": result["rationale"],
                }
            except Exception as exc:
                logger.error(
                    "TradingAgents analysis failed, falling back: {}", exc
                )
                signal = _price_action_sentiment(df)
        else:
            signal = _price_action_sentiment(df)

        rows = [signal] * pred_len
        return pd.DataFrame(rows)

    # ---- analyze ----------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        date: str,
    ) -> dict[str, Any]:
        """Run the full TradingAgents multi-agent pipeline.

        Parameters
        ----------
        symbol:
            Ticker or asset identifier (e.g. ``"AAPL"``, ``"BTCUSDT"``).
        date:
            Trade date in ISO format (``"YYYY-MM-DD"``).

        Returns
        -------
        dict
            Keys: ``decision``, ``direction``, ``confidence``,
            ``rationale``, ``analyst_reports``.

        Raises
        ------
        RuntimeError
            If TradingAgents is not installed or not enabled.
        """
        if not _TRADING_AGENTS_AVAILABLE:
            raise RuntimeError(
                "TradingAgents is not installed. "
                "Install with: pip install tradingagents"
            )
        if not self._config.enabled:
            raise RuntimeError(
                "AgentsEngine is disabled. Set enabled=True in AgentsConfig."
            )

        graph = self._load_graph()

        logger.info(
            "Running TradingAgents analysis for {} on {} ...", symbol, date
        )
        final_state, decision = _call_with_retries(
            graph.propagate, symbol, date
        )

        direction = _parse_decision_score(decision)
        confidence = min(abs(direction) + 0.2, 1.0)
        rationale = _extract_rationale(final_state)
        analyst_reports = _extract_analyst_reports(final_state)

        logger.info(
            "TradingAgents verdict for {}: {} (direction={:+.2f})",
            symbol,
            decision,
            direction,
        )

        return {
            "decision": decision,
            "direction": direction,
            "confidence": round(confidence, 4),
            "rationale": rationale,
            "analyst_reports": analyst_reports,
        }

    # -- private helpers ----------------------------------------------------

    def _load_graph(self) -> Any:
        """Lazy-load the TradingAgentsGraph with configured models."""
        if self._graph is not None:
            return self._graph

        logger.info(
            "Initialising TradingAgentsGraph (provider={}, deep={}, quick={})",
            self._config.llm_provider,
            self._config.deep_think_model,
            self._config.quick_think_model,
        )

        config = {
            "llm_provider": self._config.llm_provider,
            "deep_think_llm": self._config.deep_think_model,
            "quick_think_llm": self._config.quick_think_model,
            "max_debate_rounds": self._config.max_debate_rounds,
            "max_risk_discuss_rounds": self._config.max_risk_rounds,
        }

        self._graph = TradingAgentsGraph(config=config)

        logger.info("TradingAgentsGraph ready.")
        return self._graph
