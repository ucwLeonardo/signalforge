"""Factor registry for storing and retrieving discovered alpha factors.

Factors are persisted as JSON at ``~/.signalforge/evolution/factor_registry.json``.
Each factor has an expression that can be ``eval``'d with a pandas DataFrame ``df``
in scope, along with ``numpy`` (``np``) and ``pandas`` (``pd``).
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

_DEFAULT_REGISTRY_PATH = Path("~/.signalforge/evolution/factor_registry.json")


@dataclass(frozen=True)
class Factor:
    """A single discovered factor."""

    name: str
    expression: str
    window: int
    status: str = "candidate"  # candidate | accepted | rejected
    ic: float = 0.0
    ir: float = 0.0
    turnover: float = 0.0
    decay_5d: float = 0.0
    fitness: float = 0.0
    discovered_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat())


class FactorRegistry:
    """Persistent registry of discovered alpha factors.

    Parameters
    ----------
    path:
        Location of the JSON file.  Defaults to
        ``~/.signalforge/evolution/factor_registry.json``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path or _DEFAULT_REGISTRY_PATH).expanduser()
        self._factors: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> FactorRegistry:
        """Load factors from disk.  Returns *self* for chaining."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._factors = data if isinstance(data, list) else []
                logger.debug("Loaded {} factors from {}", len(self._factors), self._path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load factor registry: {}", exc)
                self._factors = []
        else:
            self._factors = []
        return self

    def save(self) -> None:
        """Persist current factors to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._factors, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("Saved {} factors to {}", len(self._factors), self._path)

    # ------------------------------------------------------------------
    # Mutation helpers (return new state, but mutate internal list for
    # convenience -- the list itself is the mutable container)
    # ------------------------------------------------------------------

    def add_factor(self, factor: Factor | dict[str, Any]) -> None:
        """Add a factor to the registry.

        Duplicate names are silently skipped.
        """
        rec = asdict(factor) if isinstance(factor, Factor) else dict(factor)
        # Ensure required keys
        rec.setdefault("status", "candidate")
        rec.setdefault("ic", 0.0)
        rec.setdefault("discovered_at", _dt.datetime.utcnow().isoformat())

        if any(f["name"] == rec["name"] for f in self._factors):
            logger.debug("Factor '{}' already in registry -- skipping", rec["name"])
            return

        self._factors.append(rec)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_accepted_factors(self) -> list[dict[str, Any]]:
        """Return all factors with status ``accepted``."""
        return [f for f in self._factors if f.get("status") == "accepted"]

    def get_candidates(self) -> list[dict[str, Any]]:
        """Return all factors with status ``candidate``."""
        return [f for f in self._factors if f.get("status") == "candidate"]

    def list_all(self) -> list[dict[str, Any]]:
        """Return a copy of every factor in the registry."""
        return list(self._factors)

    # ------------------------------------------------------------------
    # Factor evaluation and promotion
    # ------------------------------------------------------------------

    def evaluate_factor(
        self,
        name: str,
        df: "pd.DataFrame",
        forward_returns: "pd.Series | None" = None,
        label_horizon: int = 5,
    ) -> dict[str, Any] | None:
        """Evaluate a factor and update its metrics in the registry.

        Parameters
        ----------
        name:
            Factor name in the registry.
        df:
            OHLCV DataFrame for the factor to compute on.
        forward_returns:
            Pre-computed forward returns.  If ``None``, computed from
            close prices using *label_horizon*.
        label_horizon:
            Days ahead for forward return computation.

        Returns
        -------
        Updated metrics dict, or ``None`` if factor not found.
        """
        import numpy as np
        import pandas as pd

        from signalforge.factors.evaluate import evaluate_single_factor

        # Find the factor
        target = None
        for f in self._factors:
            if f["name"] == name:
                target = f
                break
        if target is None:
            logger.warning("Factor '{}' not found in registry", name)
            return None

        # Compute factor values
        expression = target.get("expression", "")
        if not expression:
            return None

        try:
            factor_values = eval(  # noqa: S307
                expression,
                {"__builtins__": {}},
                {"df": df, "np": np, "pd": pd},
            )
            if not isinstance(factor_values, pd.Series):
                factor_values = pd.Series(factor_values, index=df.index)
        except Exception as exc:
            logger.warning("Factor '{}' computation failed: {}", name, exc)
            return None

        # Compute forward returns if not provided
        close = df["close"].astype(np.float64)
        if forward_returns is None:
            forward_returns = close.pct_change(label_horizon).shift(-label_horizon)

        # Evaluate
        metrics = evaluate_single_factor(
            factor_values, forward_returns, close,
        )

        # Update the registry record
        target["ic"] = metrics["ic"]
        target["ir"] = metrics.get("ir", 0.0)
        target["turnover"] = metrics.get("turnover", 0.0)
        target["fitness"] = metrics.get("fitness", 0.0)
        decay = metrics.get("decay", {})
        target["decay_5d"] = decay.get(5, 0.0)

        logger.info(
            "Factor '{}' evaluated: IC={:.4f}, IR={:.4f}, fitness={:.4f}",
            name,
            metrics["ic"],
            metrics.get("ir", 0.0),
            metrics.get("fitness", 0.0),
        )
        return metrics

    def auto_promote(
        self,
        min_ic: float = 0.03,
        min_ir: float = 0.5,
        max_turnover: float = 0.3,
    ) -> list[str]:
        """Promote candidates that pass quality thresholds.

        Returns list of promoted factor names.
        """
        promoted = []
        for f in self._factors:
            if f.get("status") != "candidate":
                continue
            ic = abs(f.get("ic", 0.0))
            ir = abs(f.get("ir", 0.0))
            turnover = f.get("turnover", 1.0)

            if ic >= min_ic and ir >= min_ir and turnover <= max_turnover:
                f["status"] = "accepted"
                promoted.append(f["name"])
                logger.info(
                    "Factor '{}' promoted to accepted (IC={:.4f}, IR={:.4f})",
                    f["name"],
                    f.get("ic", 0),
                    f.get("ir", 0),
                )
        return promoted

    def reject_stale(self, max_age_days: int = 90) -> list[str]:
        """Reject candidate factors older than *max_age_days* that were never promoted.

        Returns list of rejected factor names.
        """
        rejected = []
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=max_age_days)

        for f in self._factors:
            if f.get("status") != "candidate":
                continue
            discovered = f.get("discovered_at", "")
            try:
                disc_dt = _dt.datetime.fromisoformat(discovered)
                if disc_dt < cutoff:
                    f["status"] = "rejected"
                    rejected.append(f["name"])
            except (ValueError, TypeError):
                continue

        if rejected:
            logger.info("Rejected {} stale candidate factors", len(rejected))
        return rejected

    def promote_factor(self, name: str) -> bool:
        """Manually promote a factor to ``accepted``.  Returns ``True`` on success."""
        for f in self._factors:
            if f["name"] == name:
                f["status"] = "accepted"
                return True
        return False

    def reject_factor(self, name: str) -> bool:
        """Manually reject a factor.  Returns ``True`` on success."""
        for f in self._factors:
            if f["name"] == name:
                f["status"] = "rejected"
                return True
        return False
