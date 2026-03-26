"""LLM-powered multi-analyst trading signal engine.

Uses Google Gemini to run a structured multi-perspective analysis:
  1. Technical Analyst — reads indicators and chart patterns
  2. Fundamental Analyst — evaluates valuation and macro context
  3. Sentiment Analyst — gauges market mood from price action
  4. Risk Manager — identifies downside risks
  5. Decision Synthesiser — combines all views into a final verdict

Falls back to a rule-based price-action heuristic when no LLM API key
is available (``GEMINI_API_KEY`` not set).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from signalforge.engines.base import PredictionEngine

# ---------------------------------------------------------------------------
# Gemini lazy import (new google-genai SDK)
# ---------------------------------------------------------------------------

_GEMINI_AVAILABLE: bool = False
_genai_client: Any = None

try:
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types as genai_types  # type: ignore[import-untyped]

    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if _api_key:
        _genai_client = genai.Client(api_key=_api_key)
        _GEMINI_AVAILABLE = True
        logger.info("Gemini API (google-genai) configured -- LLM-powered analysis available.")
    else:
        logger.warning(
            "GEMINI_API_KEY not set. Falling back to rule-based sentiment."
        )
except ImportError:
    try:
        # Fallback to old deprecated SDK
        import google.generativeai as genai_old  # type: ignore[import-untyped]

        _api_key = os.environ.get("GEMINI_API_KEY", "")
        if _api_key:
            genai_old.configure(api_key=_api_key)
            _GEMINI_AVAILABLE = True
            logger.info("Gemini API (legacy SDK) configured.")
        else:
            logger.warning("GEMINI_API_KEY not set. Falling back to rule-based sentiment.")
    except ImportError:
        logger.warning(
            "google-genai not installed. Install with:\n"
            "  pip install google-genai\n"
            "Falling back to rule-based price-action sentiment."
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentsConfig:
    """Tuneable knobs for :class:`AgentsEngine`.

    Attributes:
        enabled: Whether the agents engine is active.
        llm_provider: LLM backend (currently ``"gemini"``).
        model: Gemini model identifier.
        temperature: Sampling temperature for generation.
        max_debate_rounds: Number of rounds for the synthesiser to refine.
        max_risk_rounds: Number of risk-assessment refinement rounds.
    """

    enabled: bool = False
    llm_provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_debate_rounds: int = 1
    max_risk_rounds: int = 1


# ---------------------------------------------------------------------------
# Decision mapping
# ---------------------------------------------------------------------------

_DECISION_SCORE: tuple[tuple[str, float], ...] = (
    ("STRONG BUY", +0.9),
    ("STRONG SELL", -0.9),
    ("BUY", +0.7),
    ("SELL", -0.7),
    ("HOLD", 0.0),
)


def _parse_decision_score(decision: str) -> float:
    """Map a decision string to a numeric direction score."""
    normalised = decision.strip().upper()
    for key, score in _DECISION_SCORE:
        if key in normalised:
            return score
    logger.warning("Unrecognised decision '{}'; defaulting to HOLD (0.0)", decision)
    return 0.0


# ---------------------------------------------------------------------------
# Market data summary builder
# ---------------------------------------------------------------------------


def _build_market_summary(df: pd.DataFrame) -> str:
    """Build a concise market data summary for the LLM prompt."""
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    volume = df["volume"].astype(np.float64)

    current = float(close.iloc[-1])
    ret_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0
    ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0
    ret_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0

    high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
    low_52w = float(low.tail(252).min()) if len(low) >= 252 else float(low.min())
    avg_vol_20 = float(volume.tail(20).mean())

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-12)))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = float((ema12.iloc[-1] - ema26.iloc[-1]))
    macd_signal = float((ema12 - ema26).ewm(span=9).mean().iloc[-1])

    # Bollinger
    ma20 = float(close.rolling(20).mean().iloc[-1])
    std20 = float(close.rolling(20).std().iloc[-1])
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20

    # Volatility
    log_ret = np.log(close / close.shift(1)).dropna()
    vol_20d = float(log_ret.tail(20).std() * np.sqrt(252) * 100)

    return f"""MARKET DATA SUMMARY:
- Current Price: ${current:.2f}
- 1-Day Return: {ret_1d:+.2f}%
- 5-Day Return: {ret_5d:+.2f}%
- 20-Day Return: {ret_20d:+.2f}%
- 52-Week High: ${high_52w:.2f} ({(current/high_52w-1)*100:+.1f}% from high)
- 52-Week Low: ${low_52w:.2f} ({(current/low_52w-1)*100:+.1f}% from low)
- 20-Day Avg Volume: {avg_vol_20:,.0f}
- RSI(14): {rsi:.1f}
- MACD: {macd:.3f} (Signal: {macd_signal:.3f}, Histogram: {macd-macd_signal:.3f})
- Bollinger Bands: Lower=${bb_lower:.2f} | MA20=${ma20:.2f} | Upper=${bb_upper:.2f}
- 20-Day Annualised Volatility: {vol_20d:.1f}%

RECENT PRICE ACTION (last 10 days):"""


def _recent_prices(df: pd.DataFrame, n: int = 10) -> str:
    """Format last N daily candles."""
    tail = df.tail(n)
    lines = []
    for _, row in tail.iterrows():
        date = str(row.get("timestamp", row.name))[:10]
        lines.append(
            f"  {date}: O=${row['open']:.2f} H=${row['high']:.2f} "
            f"L=${row['low']:.2f} C=${row['close']:.2f} V={row['volume']:,.0f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM analysis via Gemini
# ---------------------------------------------------------------------------


_SINGLE_ANALYST_PROMPT = """You are a multi-perspective trading analyst. Analyse the following market data from four perspectives, then synthesise a final trading decision.

{market_data}

Analyse from these four perspectives:

1. TECHNICAL: Trend direction, support/resistance, momentum, chart patterns.
2. FUNDAMENTAL: Price vs historical range, valuation context, macro environment.
3. SENTIMENT: Volume patterns, RSI extremes, volatility, fear/greed.
4. RISK: Downside risk, volatility regime, position sizing.

Then combine all perspectives into one final decision.

Respond in this EXACT format:
TECHNICAL: [1 sentence]
FUNDAMENTAL: [1 sentence]
SENTIMENT: [1 sentence]
RISK: [1 sentence]
DECISION: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
CONFIDENCE: [0.0 to 1.0]
RATIONALE: [1-2 sentence rationale combining key factors]"""


def _call_gemini(
    prompt: str,
    model_name: str,
    temperature: float,
    max_retries: int = 3,
) -> str:
    """Call Gemini API with retries.  Uses the new google-genai SDK when
    available, falling back to the legacy google.generativeai SDK."""
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if _genai_client is not None:
                # New google-genai SDK
                response = _genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=500,
                    ),
                )
                return response.text or ""
            else:
                # Legacy google.generativeai SDK
                import google.generativeai as genai_old  # type: ignore[import-untyped]

                model = genai_old.GenerativeModel(
                    model_name,
                    safety_settings={
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                    },
                )
                response = model.generate_content(
                    prompt,
                    generation_config=genai_old.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=500,
                    ),
                )
                return response.text or ""
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            transient = any(
                tok in exc_str for tok in ("rate", "limit", "429", "503", "overloaded", "timeout")
            )
            if not transient or attempt == max_retries:
                raise
            wait = 2.0 ** attempt
            logger.warning("Gemini call failed (attempt {}/{}): {}. Retrying in {:.0f}s", attempt, max_retries, exc, wait)
            time.sleep(wait)

    raise last_exc  # type: ignore[misc]


def _parse_response(text: str, field: str) -> str:
    """Extract a field value from structured LLM response."""
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith(field.upper() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _run_gemini_analysis(
    df: pd.DataFrame,
    config: AgentsConfig,
) -> dict[str, Any]:
    """Run consolidated single-call Gemini analysis.

    Uses one prompt with all 4 analyst perspectives + synthesis to stay
    within free-tier rate limits (5 req/min).
    """
    market_data = _build_market_summary(df) + "\n" + _recent_prices(df)
    prompt = _SINGLE_ANALYST_PROMPT.format(market_data=market_data)

    try:
        response_text = _call_gemini(prompt, config.model, config.temperature)
        logger.debug("Gemini analysis responded ({} chars)", len(response_text))
    except Exception as exc:
        logger.error("Gemini analysis failed: {}", exc)
        return {
            "direction": 0.0,
            "confidence": 0.3,
            "rationale": f"Gemini analysis unavailable: {exc}",
            "raw_response": "",
        }

    decision = _parse_response(response_text, "DECISION") or "HOLD"
    confidence_str = _parse_response(response_text, "CONFIDENCE")
    rationale = _parse_response(response_text, "RATIONALE") or "No rationale."

    try:
        confidence = float(confidence_str)
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5

    direction = _parse_decision_score(decision)

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "rationale": f"[Gemini {config.model}] {decision}: {rationale}",
        "raw_response": response_text,
    }


# ---------------------------------------------------------------------------
# Fallback: rule-based price-action sentiment
# ---------------------------------------------------------------------------


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _price_action_sentiment(df: pd.DataFrame) -> dict[str, Any]:
    """Simple rule-based sentiment fallback."""
    close = df["close"].astype(np.float64)

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

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "rationale": (
            f"Price-action fallback: {label}. "
            f"5d return={ret_5d:+.2f}%, RSI(14)={current_rsi:.1f}. "
            f"(Gemini API unavailable -- using rule-based heuristic.)"
        ),
    }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AgentsEngine(PredictionEngine):
    """Multi-analyst LLM trading signal engine powered by Google Gemini.

    Runs a structured multi-perspective analysis using Gemini, with four
    specialised analyst prompts and a decision synthesiser.  Falls back to
    rule-based price-action sentiment when no API key is available.
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

        logger.debug(
            "AgentsEngine initialised (gemini_available={}, model={})",
            _GEMINI_AVAILABLE,
            cfg.model,
        )

    @property
    def name(self) -> str:
        return "agents"

    @property
    def config(self) -> AgentsConfig:
        return self._config

    @property
    def is_gemini_available(self) -> bool:
        return _GEMINI_AVAILABLE

    def predict(
        self,
        df: pd.DataFrame,
        pred_len: int = 1,
    ) -> pd.DataFrame:
        """Generate a directional signal via Gemini multi-analyst pipeline.

        Returns a DataFrame with columns:
        ``direction`` (-1..1), ``confidence`` (0..1), ``rationale`` (str).
        """
        if _GEMINI_AVAILABLE and self._config.enabled:
            try:
                result = _run_gemini_analysis(df, self._config)
                signal = {
                    "direction": result["direction"],
                    "confidence": result["confidence"],
                    "rationale": result["rationale"],
                }
            except Exception as exc:
                logger.error("Gemini analysis failed, falling back: {}", exc)
                signal = _price_action_sentiment(df)
        else:
            signal = _price_action_sentiment(df)

        rows = [signal] * pred_len
        return pd.DataFrame(rows)
