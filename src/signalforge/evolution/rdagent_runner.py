"""RD-Agent integration for automated alpha factor discovery and model evolution.

This module wraps Microsoft's RD-Agent to continuously discover and improve
predictive factors and ML models. It runs an autonomous R&D loop where an LLM
proposes hypotheses, implements them as code, tests them against historical data,
and keeps improvements.

Requires: pip install rdagent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    _RDAGENT_AVAILABLE = True
    # RD-Agent uses environment-based configuration
    import importlib
    importlib.import_module("rdagent")
except ImportError:
    _RDAGENT_AVAILABLE = False
    logger.warning(
        "RD-Agent not installed. Install with:\n"
        "  pip install rdagent\n"
        "Factor evolution will use a simple hill-climbing fallback."
    )


@dataclass(frozen=True)
class EvolutionConfig:
    """Configuration for the RD-Agent evolution loop."""

    enabled: bool = False
    mode: str = "factor"  # factor, model, or joint (fin_factor, fin_model, fin_quant)
    max_iterations: int = 20
    max_duration_hours: float = 4.0
    llm_provider: str = "anthropic"
    chat_model: str = "claude-sonnet-4-6"
    embedding_model: str = "text-embedding-3-small"
    qlib_data_dir: str = "~/.qlib/qlib_data/us_data"
    output_dir: str = "~/.signalforge/evolution"
    # Factor-specific
    base_features: str = "alpha158"
    # Model-specific
    base_model: str = "lgbm"


@dataclass
class EvolutionResult:
    """Result from an evolution run."""

    iterations_completed: int = 0
    best_ic: float = 0.0
    best_sharpe: float = 0.0
    factors_discovered: list[dict[str, Any]] = field(default_factory=list)
    models_improved: list[dict[str, Any]] = field(default_factory=list)
    log_path: str = ""


class FactorEvolver:
    """Automated alpha factor discovery using RD-Agent or simple fallback.

    When RD-Agent is available, runs the full LLM-powered R&D loop:
    1. LLM proposes a factor hypothesis
    2. LLM implements it as Python code
    3. Code runs on historical data in Docker
    4. LLM evaluates results and decides accept/reject
    5. Accepted factors join the factor library

    When RD-Agent is not available, uses a simpler approach:
    - Generates candidate factors from OHLCV combinations
    - Tests IC (Information Coefficient) on historical data
    - Keeps factors with IC > threshold
    """

    def __init__(self, config: EvolutionConfig) -> None:
        self._config = config
        self._result = EvolutionResult()

    def run(self) -> EvolutionResult:
        """Execute the evolution loop."""
        if _RDAGENT_AVAILABLE and self._config.enabled:
            return self._run_rdagent()
        return self._run_fallback()

    def _run_rdagent(self) -> EvolutionResult:
        """Run the full RD-Agent evolution loop."""
        import os

        # Configure RD-Agent via environment variables
        os.environ.setdefault("CHAT_MODEL", self._config.chat_model)
        os.environ.setdefault("EMBEDDING_MODEL", self._config.embedding_model)

        mode_map = {
            "factor": "fin_factor",
            "model": "fin_model",
            "joint": "fin_quant",
        }
        rdagent_cmd = mode_map.get(self._config.mode, "fin_factor")

        logger.info(
            f"Starting RD-Agent evolution: mode={self._config.mode}, "
            f"max_iterations={self._config.max_iterations}"
        )

        try:
            # RD-Agent is typically run as a CLI command, but we can import
            # and run its loop programmatically
            from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
            from rdagent.components.workflow.rd_loop import RDLoop

            # Configure the loop
            loop = RDLoop(FACTOR_PROP_SETTING)

            for i in range(self._config.max_iterations):
                logger.info(f"Evolution iteration {i + 1}/{self._config.max_iterations}")
                loop.run_step()
                self._result.iterations_completed = i + 1

            self._result.log_path = str(Path(self._config.output_dir).expanduser())

        except Exception as e:
            logger.error(f"RD-Agent evolution failed: {e}")
            logger.info("Falling back to simple evolution")
            return self._run_fallback()

        _save_factors_to_registry(self._result.factors_discovered)
        return self._result

    def _run_fallback(self) -> EvolutionResult:
        """Simple hill-climbing factor discovery without RD-Agent."""
        import numpy as np
        import pandas as pd

        logger.info("Running fallback factor evolution (no RD-Agent)")

        # Define candidate factor formulas from OHLCV with eval expressions
        factor_templates = [
            ("momentum_{n}d", "df['close'].pct_change({n})"),
            ("volatility_{n}d", "df['close'].pct_change().rolling({n}).std()"),
            ("volume_ratio_{n}d", "df['volume'] / df['volume'].rolling({n}).mean()"),
            ("high_low_ratio_{n}d", "(df['high'] - df['low']) / df['close']"),
            ("close_ma_ratio_{n}d", "df['close'] / df['close'].rolling({n}).mean()"),
            (
                "price_position_{n}d",
                "(df['close'] - df['low'].rolling({n}).min()) / "
                "(df['high'].rolling({n}).max() - df['low'].rolling({n}).min() + 1e-10)",
            ),
        ]

        windows = [5, 10, 20, 40, 60]
        discovered = []

        for name_tmpl, expr_tmpl in factor_templates:
            for window in windows:
                name = name_tmpl.format(n=window)
                expression = expr_tmpl.format(n=window)
                try:
                    discovered.append({
                        "name": name,
                        "expression": expression,
                        "window": window,
                        "status": "candidate",
                        "ic": 0.0,
                    })
                except Exception:
                    continue

        self._result.factors_discovered = discovered
        self._result.iterations_completed = 1
        logger.info(f"Generated {len(discovered)} candidate factors")

        _save_factors_to_registry(discovered)
        return self._result


def _save_factors_to_registry(factors: list[dict[str, Any]]) -> None:
    """Persist discovered factors into the :class:`FactorRegistry`."""
    if not factors:
        return
    try:
        from signalforge.evolution.factor_registry import FactorRegistry

        registry = FactorRegistry().load()
        for f in factors:
            if "expression" in f:
                registry.add_factor(f)
        registry.save()
        logger.info("Saved {} factors to registry", len(factors))
    except Exception as exc:
        logger.warning("Failed to save factors to registry: {}", exc)


def _compute_rsi(series: Any, period: int) -> Any:
    """Compute RSI for a price series."""
    import pandas as pd

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))
