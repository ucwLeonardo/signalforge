"""Signal report generation with Rich tables, JSON, and CSV output."""

from __future__ import annotations

import csv
import io
import json
from typing import Sequence

from rich.console import Console
from rich.table import Table

from signalforge.data.models import TradeAction, TradeTarget

# Colour map for trade actions (Rich markup style names).
_ACTION_STYLES: dict[TradeAction, str] = {
    TradeAction.BUY: "bold green",
    TradeAction.SELL: "bold red",
    TradeAction.HOLD: "dim",
}


class ReportGenerator:
    """Format a list of :class:`TradeTarget` into human- or machine-readable output.

    Supported formats:

    * ``"table"`` -- Rich CLI table (default).
    * ``"json"``  -- Pretty-printed JSON array.
    * ``"csv"``   -- RFC-4180 CSV string (with header row).
    """

    def generate_report(
        self,
        targets: Sequence[TradeTarget],
        fmt: str = "table",
    ) -> str:
        """Render *targets* in the requested format.

        Parameters
        ----------
        targets:
            Trade targets to include in the report.
        fmt:
            One of ``"table"``, ``"json"``, or ``"csv"``.

        Returns
        -------
        str
            The rendered report as a string.

        Raises
        ------
        ValueError
            If *fmt* is not a recognised format.
        """
        fmt_lower = fmt.lower()
        if fmt_lower == "table":
            return self._render_table(targets)
        if fmt_lower == "json":
            return self._render_json(targets)
        if fmt_lower == "csv":
            return self._render_csv(targets)
        raise ValueError(
            f"Unknown report format {fmt!r}. Supported: table, json, csv."
        )

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_table(targets: Sequence[TradeTarget]) -> str:
        table = Table(
            title="SignalForge Trade Targets",
            show_lines=True,
            header_style="bold cyan",
        )

        table.add_column("Symbol", style="bold")
        table.add_column("Action", justify="center")
        table.add_column("Entry", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Stop", justify="right")
        table.add_column("R:R", justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Rationale", max_width=60)

        for t in targets:
            style = _ACTION_STYLES.get(t.action, "")
            table.add_row(
                t.symbol,
                f"[{style}]{t.action.value}[/{style}]",
                f"{t.entry_price:.2f}",
                f"{t.target_price:.2f}",
                f"{t.stop_loss:.2f}",
                f"{t.risk_reward_ratio:.2f}",
                f"{t.confidence:.0%}",
                t.rationale,
            )

        console = Console(file=io.StringIO(), force_terminal=True, width=140)
        console.print(table)
        output = console.file.getvalue()  # type: ignore[attr-defined]
        return output

    @staticmethod
    def _render_json(targets: Sequence[TradeTarget]) -> str:
        rows = [
            {
                "symbol": t.symbol,
                "action": t.action.value,
                "entry_price": t.entry_price,
                "target_price": t.target_price,
                "stop_loss": t.stop_loss,
                "risk_reward_ratio": t.risk_reward_ratio,
                "confidence": t.confidence,
                "horizon_days": t.horizon_days,
                "rationale": t.rationale,
            }
            for t in targets
        ]
        return json.dumps(rows, indent=2)

    @staticmethod
    def _render_csv(targets: Sequence[TradeTarget]) -> str:
        buf = io.StringIO()
        fieldnames = [
            "symbol",
            "action",
            "entry_price",
            "target_price",
            "stop_loss",
            "risk_reward_ratio",
            "confidence",
            "horizon_days",
            "rationale",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for t in targets:
            writer.writerow(
                {
                    "symbol": t.symbol,
                    "action": t.action.value,
                    "entry_price": t.entry_price,
                    "target_price": t.target_price,
                    "stop_loss": t.stop_loss,
                    "risk_reward_ratio": t.risk_reward_ratio,
                    "confidence": t.confidence,
                    "horizon_days": t.horizon_days,
                    "rationale": t.rationale,
                },
            )
        return buf.getvalue()
