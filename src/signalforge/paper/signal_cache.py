"""Per-account signal cache with JSON persistence.

Signals are stored in each account's directory:
  ~/.signalforge/accounts/{name}/signals_full.json
  ~/.signalforge/accounts/{name}/signals_watchlist.json

Each file contains a JSON object:
  {
    "scanned_at": "2026-03-28T15:30:00",
    "scan_type": "full" | "watchlist",
    "signals": [ ... serialized TradeTarget objects ... ]
  }
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from signalforge.data.models import TradeAction, TradeTarget


def _signal_to_dict(sig: TradeTarget) -> dict:
    return {
        "symbol": sig.symbol,
        "action": sig.action.value,
        "entry_price": sig.entry_price,
        "target_price": sig.target_price,
        "stop_loss": sig.stop_loss,
        "risk_reward_ratio": sig.risk_reward_ratio,
        "confidence": sig.confidence,
        "horizon_days": sig.horizon_days,
        "rationale": sig.rationale,
    }


def _signal_from_dict(d: dict) -> TradeTarget:
    return TradeTarget(
        symbol=d["symbol"],
        action=TradeAction(d["action"]),
        entry_price=d["entry_price"],
        target_price=d["target_price"],
        stop_loss=d["stop_loss"],
        risk_reward_ratio=d["risk_reward_ratio"],
        confidence=d["confidence"],
        horizon_days=d["horizon_days"],
        rationale=d.get("rationale", ""),
    )


class SignalCache:
    """Per-account signal persistence.

    Each account stores two signal sets:
      - full: from a Scan All operation (discovery + config symbols)
      - watchlist: from a Watchlist scan (config symbols only)
    """

    def __init__(self, account_dir: Path) -> None:
        self._dir = account_dir

    def _path(self, scan_type: str) -> Path:
        return self._dir / f"signals_{scan_type}.json"

    def save(
        self,
        signals: list[TradeTarget],
        scan_type: str = "full",
    ) -> None:
        """Persist signals to disk for the given scan type."""
        self._dir.mkdir(parents=True, exist_ok=True)
        data = {
            "scanned_at": datetime.now().isoformat(),
            "scan_type": scan_type,
            "count": len(signals),
            "signals": [_signal_to_dict(s) for s in signals],
        }
        with open(self._path(scan_type), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, scan_type: str = "full") -> list[TradeTarget]:
        """Load cached signals from disk. Returns empty list if no cache."""
        path = self._path(scan_type)
        if not path.exists():
            return []
        with open(path) as f:
            data = json.load(f)
        return [_signal_from_dict(d) for d in data.get("signals", [])]

    def metadata(self, scan_type: str = "full") -> dict | None:
        """Return cache metadata (scanned_at, count) without loading signals."""
        path = self._path(scan_type)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return {
            "scanned_at": data.get("scanned_at"),
            "scan_type": data.get("scan_type"),
            "count": data.get("count", 0),
        }

    def clear(self, scan_type: str | None = None) -> None:
        """Remove cached signal files. If scan_type is None, clear both."""
        types = [scan_type] if scan_type else ["full", "watchlist"]
        for st in types:
            path = self._path(st)
            if path.exists():
                path.unlink()
