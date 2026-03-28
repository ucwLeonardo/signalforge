"""HTTP server for the paper trading dashboard.

Serves a JSON API and the static dashboard HTML using only Python stdlib.
Supports multiple named accounts via ?account=NAME query parameter.
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
from urllib.parse import urlparse, parse_qs

from signalforge.paper.models import position_from_dict
from signalforge.paper.portfolio import AccountManager, PortfolioManager
from signalforge.paper.simulator import (
    auto_build_portfolio,
    generate_real_signals,
)

import threading

# Live prices cache — updated by browser via /api/update-prices
_live_prices: dict[str, float] = {}

# Cached signals — only regenerated on explicit trigger (auto-build or manual scan)
_cached_signals: list = []
_scan_cancel: bool = False  # set to True to cancel running scan
_scan_progress: dict = {
    "running": False,
    "total": 0,
    "completed": 0,
    "symbol": "",
    "stage": "",
    "detail": "",
    "error": None,
    "log": [],
}

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


def _fetch_live_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for symbols via yfinance (stocks) and ccxt (crypto).

    This runs on the server (WSL2 with network) — not affected by sandbox.
    """
    prices: dict[str, float] = {}

    stock_syms = [s for s in symbols if "/" not in s and not s.endswith("=F")]
    crypto_syms = [s for s in symbols if "/" in s]
    futures_syms = [s for s in symbols if s.endswith("=F")]

    # Stocks + Futures via yfinance
    yf_syms = stock_syms + futures_syms
    if yf_syms:
        try:
            import yfinance as yf
            tickers = yf.Tickers(" ".join(yf_syms))
            for sym in yf_syms:
                try:
                    info = tickers.tickers[sym].fast_info
                    price = info.get("lastPrice") or info.get("last_price")
                    if price and price > 0:
                        prices[sym] = float(price)
                except Exception:
                    pass
        except Exception as exc:
            sys.stderr.write(f"yfinance price fetch failed: {exc}\n")

    # Crypto via ccxt
    if crypto_syms:
        try:
            import ccxt
            from signalforge.config import load_config
            cfg = load_config()
            exchange_id = cfg.data.crypto_exchange
            exchange = getattr(ccxt, exchange_id)()
            for sym in crypto_syms:
                try:
                    ticker = exchange.fetch_ticker(sym)
                    if ticker and ticker.get("last"):
                        prices[sym] = float(ticker["last"])
                except Exception:
                    pass
        except Exception as exc:
            sys.stderr.write(f"ccxt price fetch failed: {exc}\n")

    return prices


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

    account_manager: AccountManager  # set on the class before server starts
    legacy_manager: PortfolioManager | None = None  # set when --path is used

    def _parse_request_path(self) -> tuple[str, dict[str, str]]:
        """Split self.path into route and query params."""
        parsed = urlparse(self.path)
        params = {}
        for k, v in parse_qs(parsed.query).items():
            params[k] = v[0] if v else ""
        return parsed.path, params

    def _get_manager(self, params: dict[str, str] | None = None) -> PortfolioManager:
        """Get PortfolioManager for the requested account."""
        if self.__class__.legacy_manager is not None:
            return self.__class__.legacy_manager
        account = (params or {}).get("account", "default")
        return self.__class__.account_manager.get_manager(account)

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
        route, params = self._parse_request_path()
        routes: dict[str, Any] = {
            "/api/portfolio": self._handle_portfolio,
            "/api/signals": self._handle_signals,
            "/api/history": self._handle_history,
            "/api/prices": self._handle_prices_proxy,
            "/api/accounts": self._handle_accounts_list,
            "/api/scan/status": self._handle_scan_status,
        }
        handler = routes.get(route)
        if handler is not None:
            handler(params)
            return

        # Serve static dashboard
        if route in ("/", "/index.html", "/dashboard.html"):
            self._serve_dashboard()
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        route, params = self._parse_request_path()
        routes: dict[str, Any] = {
            "/api/open": self._handle_open,
            "/api/close": self._handle_close,
            "/api/update-stops": self._handle_update_stops,
            "/api/update-prices": self._handle_update_prices,
            "/api/accounts/create": self._handle_account_create,
            "/api/accounts/reset": self._handle_account_reset,
            "/api/accounts/delete": self._handle_account_delete,
            "/api/auto-build": self._handle_auto_build,
            "/api/scan": self._handle_scan_start,
            "/api/scan/cancel": self._handle_scan_cancel,
        }
        handler = routes.get(route)
        if handler is not None:
            handler(params)
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    # -- GET handlers ---------------------------------------------------

    def _handle_portfolio(self, params: dict[str, str]) -> None:
        manager = self._get_manager(params)
        try:
            self._ensure_portfolio(manager)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
        # Return data immediately — price updates happen via /api/update-prices
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
            "asset_categories": portfolio.asset_categories,
            "value_history": history,
        })

    @staticmethod
    def _classify_symbol(symbol: str) -> str:
        """Classify a symbol into a category name."""
        if "/" in symbol:
            return "crypto"
        if symbol.endswith("=F"):
            return "futures"
        return "us_stocks"

    def _handle_signals(self, params: dict[str, str]) -> None:
        """Return cached signals, filtered by account's asset_categories."""
        global _cached_signals
        # Load account's allowed categories
        manager = self._get_manager(params)
        allowed_categories: set[str] | None = None
        if manager.exists():
            portfolio = manager.load()
            allowed_categories = set(portfolio.asset_categories)

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
            for s in _cached_signals
            if allowed_categories is None or self._classify_symbol(s.symbol) in allowed_categories
        ])

    def _handle_history(self, params: dict[str, str]) -> None:
        manager = self._get_manager(params)
        try:
            self._ensure_portfolio(manager)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
        portfolio = manager.load()
        self._send_json([t.to_dict() for t in portfolio.trades])

    def _handle_prices_proxy(self, params: dict[str, str]) -> None:
        """Fetch current prices for held positions via yfinance/ccxt."""
        manager = self._get_manager(params)
        if not manager.exists():
            self._send_json({"prices": {}})
            return

        portfolio = manager.load()
        held = [p.symbol for p in portfolio.positions]
        if not held:
            self._send_json({"prices": dict(_live_prices)})
            return

        # Fetch real prices for held symbols
        prices = _fetch_live_prices(held)
        if prices:
            _live_prices.update(prices)
            manager.update_prices(prices)

        self._send_json({"prices": dict(_live_prices)})

    def _handle_accounts_list(self, params: dict[str, str]) -> None:
        """List all accounts with summaries."""
        am = self.__class__.account_manager
        names = am.list_accounts()
        accounts = []
        for name in names:
            try:
                accounts.append(am.get_account_summary(name))
            except Exception:
                accounts.append({"name": name, "error": "failed to load"})
        self._send_json({"accounts": accounts})

    # -- POST handlers --------------------------------------------------

    def _handle_open(self, params: dict[str, str]) -> None:
        manager = self._get_manager(params)
        try:
            self._ensure_portfolio(manager)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
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

    def _handle_close(self, params: dict[str, str]) -> None:
        manager = self._get_manager(params)
        try:
            self._ensure_portfolio(manager)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
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

    def _handle_update_stops(self, params: dict[str, str]) -> None:
        """Update stop_loss and/or target_price on a position."""
        manager = self._get_manager(params)
        try:
            self._ensure_portfolio(manager)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
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
            manager._save(portfolio)

            updated = next(p for p in updated_positions if p.symbol == symbol)
            self._send_json(updated.to_dict())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    def _handle_update_prices(self, params: dict[str, str]) -> None:
        """Accept price updates from the browser."""
        manager = self._get_manager(params)
        try:
            self._ensure_portfolio(manager)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
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

    # -- Account management endpoints -----------------------------------

    def _handle_account_create(self, params: dict[str, str]) -> None:
        """Create a new account. Body: {name, balance?, asset_categories?}"""
        try:
            body = self._read_body()
            name = body.get("name", "").strip()
            if not name:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing field: name")
                return
            balance = float(body.get("balance", 5000))
            force = body.get("force", False)
            asset_categories = body.get("asset_categories")
            am = self.__class__.account_manager
            am.create_account(name, balance=balance, force=force, asset_categories=asset_categories)
            self._send_json(am.get_account_summary(name), status=HTTPStatus.CREATED)
        except (ValueError, FileExistsError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except (json.JSONDecodeError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    def _handle_account_reset(self, params: dict[str, str]) -> None:
        """Reset an account. Body: {name, balance?}"""
        try:
            body = self._read_body()
            name = body.get("name", "").strip()
            if not name:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing field: name")
                return
            balance = body.get("balance")
            if balance is not None:
                balance = float(balance)
            am = self.__class__.account_manager
            am.reset_account(name, balance=balance)
            self._send_json(am.get_account_summary(name))
        except (ValueError, FileExistsError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except (json.JSONDecodeError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    def _handle_account_delete(self, params: dict[str, str]) -> None:
        """Delete an account. Body: {name}"""
        try:
            body = self._read_body()
            name = body.get("name", "").strip()
            if not name:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing field: name")
                return
            am = self.__class__.account_manager
            am.delete_account(name)
            self._send_json({"deleted": name})
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except (json.JSONDecodeError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    # -- scan & auto-build endpoints -------------------------------------

    def _handle_scan_start(self, params: dict[str, str]) -> None:
        """Start pipeline scan in background thread. Poll /api/scan/status for progress.

        Body: {categories: ["us_stocks", "crypto"]}
        """
        global _scan_progress
        if _scan_progress["running"]:
            self._send_json({
                "status": "already_running",
                **{k: _scan_progress[k] for k in ("total", "completed", "symbol", "stage")},
            })
            return
        try:
            body = self._read_body()
            categories = body.get("categories", ["us_stocks", "crypto"])

            global _scan_cancel
            _scan_cancel = False
            _scan_progress = {
                "running": True, "total": 0, "completed": 0,
                "symbol": "", "stage": "starting", "detail": "Initializing pipeline...",
                "error": None, "log": [],
            }

            def _on_progress(p: dict) -> None:
                global _scan_progress
                _scan_progress.update(p)
                _scan_progress["running"] = True
                _scan_progress.setdefault("log", []).append({
                    "symbol": p.get("symbol", ""),
                    "stage": p.get("stage", ""),
                    "detail": p.get("detail", ""),
                    "ts": datetime.now().strftime("%H:%M:%S"),
                })

            def _run_scan() -> None:
                global _cached_signals, _scan_progress
                try:
                    sys.stderr.write(f"[Scan] Background scan started: {categories}\n")
                    signals = generate_real_signals(
                        categories=categories,
                        progress_cb=_on_progress,
                        cancel_flag=lambda: _scan_cancel,
                    )
                    _cached_signals = signals
                    _scan_progress = {
                        "running": False, "total": _scan_progress.get("total", 0),
                        "completed": _scan_progress.get("total", 0),
                        "symbol": "", "stage": "complete",
                        "detail": f"Done: {len(signals)} signals generated",
                        "error": None,
                    }
                    sys.stderr.write(f"[Scan] Done: {len(signals)} signals\n")
                except Exception as exc:
                    _scan_progress = {
                        "running": False, "total": 0, "completed": 0,
                        "symbol": "", "stage": "error", "detail": "",
                        "error": str(exc),
                    }
                    sys.stderr.write(f"[Scan] Failed: {exc}\n")

            thread = threading.Thread(target=_run_scan, daemon=True)
            thread.start()
            self._send_json({"status": "started", "categories": categories})
        except (json.JSONDecodeError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    def _handle_scan_status(self, params: dict[str, str]) -> None:
        """Return detailed scan progress with log history."""
        # Only send last 50 log entries to avoid huge responses
        log = _scan_progress.get("log", [])[-50:]
        self._send_json({
            "running": _scan_progress["running"],
            "total": _scan_progress.get("total", 0),
            "completed": _scan_progress.get("completed", 0),
            "symbol": _scan_progress.get("symbol", ""),
            "stage": _scan_progress.get("stage", ""),
            "detail": _scan_progress.get("detail", ""),
            "error": _scan_progress.get("error"),
            "count": len(_cached_signals) if not _scan_progress["running"] else None,
            "log": log,
        })

    def _handle_scan_cancel(self, params: dict[str, str]) -> None:
        """Cancel a running scan."""
        global _scan_cancel
        if _scan_progress.get("running"):
            _scan_cancel = True
            self._send_json({"status": "cancelling"})
        else:
            self._send_json({"status": "not_running"})

    def _handle_auto_build(self, params: dict[str, str]) -> None:
        """Auto-build portfolio: run pipeline → top N → Kelly allocation → open positions.

        Body: {categories: ["us_stocks", "crypto"], top_n?: 5}
        """
        global _cached_signals
        manager = self._get_manager(params)
        try:
            self._ensure_portfolio(manager)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
        try:
            body = self._read_body()
            categories = body.get("categories", ["us_stocks", "crypto"])
            top_n = int(body.get("top_n", 5))

            result = auto_build_portfolio(
                manager=manager,
                categories=categories,
                top_n=top_n,
            )

            # Cache the signals generated during auto-build for the signals endpoint
            if "all_signals" in result:
                _cached_signals = result.pop("all_signals")

            if "error" in result and not result.get("positions_opened"):
                self._send_error_json(HTTPStatus.BAD_REQUEST, result["error"])
                return
            self._send_json(result)
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
        """Initialize portfolio if it does not exist yet (default account only)."""
        if not manager.exists():
            # Only auto-create for the default account path
            if manager.path.parent.name == "default":
                manager.init()
            else:
                raise ValueError(
                    f"Account '{manager.path.parent.name}' does not exist. "
                    "Create it first via the accounts API."
                )

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

    am = AccountManager()
    PaperTradingHandler.account_manager = am

    # If --path is given, use legacy single-file mode
    if args.path:
        PaperTradingHandler.legacy_manager = PortfolioManager(path=args.path)
    else:
        PaperTradingHandler.legacy_manager = None

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", args.port), PaperTradingHandler)
    accounts = am.list_accounts()
    print(f"Paper trading server running on http://localhost:{args.port}")
    print(f"Accounts: {', '.join(accounts) if accounts else '(none yet, will auto-create default)'}")
    print(f"Dashboard: http://localhost:{args.port}/")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
