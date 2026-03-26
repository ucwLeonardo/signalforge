"""HTTP server for the paper trading dashboard.

Serves a JSON API and the static dashboard HTML using only Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any

from signalforge.paper.models import position_from_dict
from signalforge.paper.portfolio import PortfolioManager
from signalforge.paper.simulator import generate_live_signals, LIVE_PRICES_20260326

# Live prices cache — updated by fetch_live_prices()
_live_prices: dict[str, float] = dict(LIVE_PRICES_20260326)
_prices_last_fetched: datetime | None = None
_PRICE_CACHE_SECONDS = 60  # refetch every 60s

_DASHBOARD_DIR = Path(__file__).resolve().parent
_JSON_CONTENT = "application/json"


# ---------------------------------------------------------------------------
# Value-history tracker
# ---------------------------------------------------------------------------

def _history_path(portfolio_path: Path) -> Path:
    """Return the value-history JSON path next to the portfolio file."""
    return portfolio_path.parent / "paper_value_history.json"


def _load_history(portfolio_path: Path) -> list[dict[str, Any]]:
    hp = _history_path(portfolio_path)
    if not hp.exists():
        return []
    with open(hp) as f:
        return json.load(f)


def _save_history(portfolio_path: Path, history: list[dict[str, Any]]) -> None:
    hp = _history_path(portfolio_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    with open(hp, "w") as f:
        json.dump(history, f, indent=2)


def _fetch_live_prices_google() -> dict[str, float]:
    """Fetch live prices from Google Finance using urllib (stdlib).

    Returns a dict of {symbol: price} for symbols we know how to look up.
    Falls back to cached prices on any error.
    """
    import re
    import urllib.request

    symbol_urls = {
        "NVDA": "https://www.google.com/finance/quote/NVDA:NASDAQ",
        "AAPL": "https://www.google.com/finance/quote/AAPL:NASDAQ",
        "MSFT": "https://www.google.com/finance/quote/MSFT:NASDAQ",
        "TSLA": "https://www.google.com/finance/quote/TSLA:NASDAQ",
        "AMZN": "https://www.google.com/finance/quote/AMZN:NASDAQ",
        "BTC/USDT": "https://www.google.com/finance/quote/BTC-USD",
        "ETH/USDT": "https://www.google.com/finance/quote/ETH-USD",
        "SOL/USDT": "https://www.google.com/finance/quote/SOL-USD",
    }

    prices: dict[str, float] = {}
    for symbol, url in symbol_urls.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            # Parse price from <title> tag: "NVDA $178.68 ..." or "BTC/USD 70,860.49 ..."
            title_match = re.search(r"<title>([^<]+)</title>", html)
            if title_match:
                title = title_match.group(1)
                # Try dollar format first: $178.68
                price_match = re.search(r"\$([0-9,]+\.?\d*)", title)
                if price_match:
                    prices[symbol] = float(price_match.group(1).replace(",", ""))
                else:
                    # Crypto format: BTC/USD 70,860.49
                    price_match = re.search(r"\s([0-9,]+\.\d+)\s", title)
                    if price_match:
                        prices[symbol] = float(price_match.group(1).replace(",", ""))
        except Exception:
            continue  # Skip this symbol, use cached price

    return prices


def _get_live_prices() -> dict[str, float]:
    """Get live prices, using cache to avoid hammering Google Finance."""
    global _live_prices, _prices_last_fetched

    now = datetime.now()
    if (
        _prices_last_fetched is None
        or (now - _prices_last_fetched).total_seconds() > _PRICE_CACHE_SECONDS
    ):
        try:
            fetched = _fetch_live_prices_google()
            if fetched:
                _live_prices.update(fetched)
                _prices_last_fetched = now
                sys.stderr.write(
                    f"[{now:%H:%M:%S}] Updated {len(fetched)} live prices\n"
                )
        except Exception as exc:
            sys.stderr.write(f"[{now:%H:%M:%S}] Price fetch failed: {exc}\n")

    return _live_prices


def _update_portfolio_prices(manager: PortfolioManager) -> None:
    """Update open positions with latest live prices."""
    prices = _get_live_prices()
    portfolio = manager.load()
    held_symbols = {p.symbol for p in portfolio.positions}
    price_updates = {s: p for s, p in prices.items() if s in held_symbols}
    if price_updates:
        manager.update_prices(price_updates)


def _append_snapshot(manager: PortfolioManager) -> None:
    """Append a value snapshot to the history file."""
    portfolio = manager.load()
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "total_value": round(portfolio.total_value, 2),
        "cash": round(portfolio.cash, 2),
        "positions_value": round(portfolio.positions_value, 2),
    }
    history = _load_history(manager.path)
    history.append(snapshot)
    _save_history(manager.path, history)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class PaperTradingHandler(BaseHTTPRequestHandler):
    """Handle API and static file requests for the paper trading dashboard."""

    manager: PortfolioManager  # set on the class before server starts

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(
        self, data: Any, status: int = HTTPStatus.OK
    ) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", _JSON_CONTENT)
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    # -- routing --------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        routes: dict[str, Any] = {
            "/api/portfolio": self._handle_portfolio,
            "/api/signals": self._handle_signals,
            "/api/history": self._handle_history,
            "/api/prices": self._handle_prices_proxy,
        }
        handler = routes.get(self.path)
        if handler is not None:
            handler()
            return

        # Serve static dashboard
        if self.path in ("/", "/index.html", "/dashboard.html"):
            self._serve_dashboard()
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        routes: dict[str, Any] = {
            "/api/open": self._handle_open,
            "/api/close": self._handle_close,
            "/api/update-stops": self._handle_update_stops,
            "/api/update-prices": self._handle_update_prices,
        }
        handler = routes.get(self.path)
        if handler is not None:
            handler()
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    # -- GET handlers ---------------------------------------------------

    def _handle_portfolio(self) -> None:
        manager = self.__class__.manager
        self._ensure_portfolio(manager)
        _update_portfolio_prices(manager)
        portfolio = manager.load()
        _append_snapshot(manager)
        history = _load_history(manager.path)
        self._send_json({
            "cash": round(portfolio.cash, 2),
            "initial_balance": round(portfolio.initial_balance, 2),
            "total_value": round(portfolio.total_value, 2),
            "total_pnl": round(portfolio.total_pnl, 2),
            "total_pnl_pct": round(portfolio.total_pnl_pct, 2),
            "positions_value": round(portfolio.positions_value, 2),
            "realized_pnl": round(portfolio.realized_pnl, 2),
            "unrealized_pnl": round(portfolio.unrealized_pnl, 2),
            "positions": [p.to_dict() for p in portfolio.positions],
            "created_at": portfolio.created_at.isoformat(),
            "value_history": history,
        })

    def _handle_signals(self) -> None:
        signals = generate_live_signals()
        self._send_json([
            {
                "symbol": s.symbol,
                "action": s.action.value,
                "entry_price": s.entry_price,
                "target_price": s.target_price,
                "stop_loss": s.stop_loss,
                "risk_reward_ratio": s.risk_reward_ratio,
                "confidence": s.confidence,
                "horizon_days": s.horizon_days,
                "rationale": s.rationale,
            }
            for s in signals
        ])

    def _handle_history(self) -> None:
        manager = self.__class__.manager
        self._ensure_portfolio(manager)
        portfolio = manager.load()
        self._send_json([t.to_dict() for t in portfolio.trades])

    def _handle_prices_proxy(self) -> None:
        """Proxy endpoint: server fetches live prices and returns them.

        The browser can't fetch Google Finance directly (CORS), so it
        calls this endpoint instead. The server uses urllib which may
        also be blocked by proxy, but falls back to cached prices.
        """
        manager = self.__class__.manager
        self._ensure_portfolio(manager)

        # Try to fetch fresh prices
        prices = _get_live_prices()

        # Also update positions while we're at it
        portfolio = manager.load()
        held = {p.symbol for p in portfolio.positions}
        updates = {s: p for s, p in prices.items() if s in held}
        if updates:
            manager.update_prices(updates)

        self._send_json({"prices": prices})

    # -- POST handlers --------------------------------------------------

    def _handle_open(self) -> None:
        manager = self.__class__.manager
        self._ensure_portfolio(manager)
        try:
            body = self._read_body()
            required = ("symbol", "side", "qty", "entry_price", "stop_loss", "target_price")
            missing = [k for k in required if k not in body]
            if missing:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    f"Missing fields: {', '.join(missing)}",
                )
                return
            position = manager.open_position(
                symbol=body["symbol"],
                side=body["side"],
                qty=float(body["qty"]),
                entry_price=float(body["entry_price"]),
                stop_loss=float(body["stop_loss"]),
                target_price=float(body["target_price"]),
            )
            self._send_json(position.to_dict(), status=HTTPStatus.CREATED)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    def _handle_close(self) -> None:
        manager = self.__class__.manager
        self._ensure_portfolio(manager)
        try:
            body = self._read_body()
            if "symbol" not in body or "exit_price" not in body:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "Missing fields: symbol, exit_price required",
                )
                return
            trade = manager.close_position(
                symbol=body["symbol"],
                exit_price=float(body["exit_price"]),
                reason=body.get("reason", "manual"),
            )
            self._send_json(trade.to_dict())
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    def _handle_update_stops(self) -> None:
        """Update stop_loss and/or target_price on a position.

        Since Position is frozen, we rebuild the positions list with
        dataclasses.replace() for the matching symbol.
        """
        manager = self.__class__.manager
        self._ensure_portfolio(manager)
        try:
            body = self._read_body()
            symbol = body.get("symbol")
            if not symbol:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST, "Missing field: symbol"
                )
                return

            new_stop = body.get("stop_loss")
            new_target = body.get("target_price")
            if new_stop is None and new_target is None:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "Provide at least one of: stop_loss, target_price",
                )
                return

            portfolio = manager.load()
            found = False
            updated_positions = []
            for pos in portfolio.positions:
                if pos.symbol == symbol:
                    found = True
                    replacements: dict[str, float] = {}
                    if new_stop is not None:
                        replacements["stop_loss"] = float(new_stop)
                    if new_target is not None:
                        replacements["target_price"] = float(new_target)
                    updated_positions.append(replace(pos, **replacements))
                else:
                    updated_positions.append(pos)

            if not found:
                self._send_error_json(
                    HTTPStatus.NOT_FOUND,
                    f"No open position for {symbol}",
                )
                return

            portfolio.positions = updated_positions
            # Save via the manager's internal method
            manager._save(portfolio)

            # Return the updated position
            updated = next(p for p in updated_positions if p.symbol == symbol)
            self._send_json(updated.to_dict())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    def _handle_update_prices(self) -> None:
        """Accept price updates from the browser (which can fetch from Google Finance)."""
        manager = self.__class__.manager
        self._ensure_portfolio(manager)
        try:
            body = self._read_body()
            prices = body.get("prices", {})
            if not prices or not isinstance(prices, dict):
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "Provide {prices: {symbol: price, ...}}",
                )
                return
            float_prices = {k: float(v) for k, v in prices.items()}
            # Update global cache too
            global _live_prices
            _live_prices.update(float_prices)
            manager.update_prices(float_prices)
            portfolio = manager.load()
            self._send_json({
                "updated": len(float_prices),
                "total_value": round(portfolio.total_value, 2),
                "unrealized_pnl": round(portfolio.unrealized_pnl, 2),
            })
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    # -- static file serving --------------------------------------------

    def _serve_dashboard(self) -> None:
        dashboard_file = _DASHBOARD_DIR / "dashboard.html"
        if not dashboard_file.exists():
            self._send_error_json(
                HTTPStatus.NOT_FOUND,
                "dashboard.html not found. Place it in the paper/ directory.",
            )
            return
        body = dashboard_file.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _ensure_portfolio(manager: PortfolioManager) -> None:
        """Initialize portfolio if it does not exist yet."""
        if not manager.exists():
            manager.init()

    def log_message(self, format: str, *args: Any) -> None:
        """Override to include timestamp in log output."""
        sys.stderr.write(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {self.address_string()} "
            f"- {format % args}\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper trading dashboard HTTP server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port to listen on (default: 8787)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to portfolio JSON file (default: ~/.signalforge/paper_portfolio.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manager = PortfolioManager(path=args.path)

    # Attach manager to the handler class so all requests can access it
    PaperTradingHandler.manager = manager

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", args.port), PaperTradingHandler)
    print(f"Paper trading server running on http://localhost:{args.port}")
    print(f"Portfolio path: {manager.path}")
    print(f"Dashboard: http://localhost:{args.port}/")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
