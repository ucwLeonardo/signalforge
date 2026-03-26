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

    def list_all(self) -> list[dict[str, Any]]:
        """Return a copy of every factor in the registry."""
        return list(self._factors)
