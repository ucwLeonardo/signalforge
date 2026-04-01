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
from signalforge.paper.signal_cache import SignalCache
from signalforge.paper.simulator import (
    auto_build_portfolio,
    build_from_cached_signals,
    generate_real_signals,
)

import threading

from signalforge.paper.prices import _classify, _is_us_market_hours


# Live prices cache — updated by background thread for ALL accounts
_live_prices: dict[str, float] = {}
_PRICE_UPDATE_INTERVAL = 30  # seconds
_last_stock_trading_day: str | None = None  # ISO date of last fetched close
_price_status: dict[str, Any] = {
    "last_update": None,
    "symbols_updated": 0,
    "accounts_updated": 0,
    "last_error": None,
    "source": None,
}

_scan_cancel: bool = False  # set to True to cancel running scan
_scan_account: str = ""  # which account is currently scanning
_scan_type: str = "full"  # "full" or "watchlist"
_scan_progress: dict = {
    "running": False,
    "total": 0,
    "completed": 0,
    "symbol": "",
    "stage": "",
    "detail": "",
    "phase_pct": 0,
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
    """Fetch current prices: Binance for crypto (real-time), Polygon for stocks.

    Logs errors but does not crash the server — returns partial results.
    """
    from signalforge.paper.prices import PriceFetchError, fetch_prices

    try:
        return fetch_prices(symbols)
    except PriceFetchError as exc:
        sys.stderr.write(f"[Prices] {exc}\n")
        return {}
    except Exception as exc:
        sys.stderr.write(f"[Prices] Unexpected error: {exc}\n")
        return {}


def _background_price_updater(account_manager: AccountManager) -> None:
    """Background thread: update prices for ALL accounts with open positions.

    Runs every _PRICE_UPDATE_INTERVAL seconds. Collects all held symbols
    across all accounts, fetches prices once, then updates each account.
    """
    import time

    while True:
        time.sleep(_PRICE_UPDATE_INTERVAL)
        try:
            names = account_manager.list_accounts()
            if not names:
                continue

            # Collect all held symbols across all accounts
            account_positions: dict[str, list[str]] = {}
            all_symbols: set[str] = set()
            for name in names:
                mgr = account_manager.get_manager(name)
                if not mgr.exists():
                    continue
                portfolio = mgr.load()
                held = [p.symbol for p in portfolio.positions]
                if held:
                    account_positions[name] = held
                    all_symbols.update(held)

            if not all_symbols:
                continue

            market_open = _is_us_market_hours()

            # Outside US market hours: stock prev-close only changes once per
            # trading day, so fetch stocks exactly once when last_trading_day
            # advances (new close data available). Crypto trades 24/7 — always.
            stock_refresh_td: str | None = None
            if not market_open:
                from signalforge.data.calendar import last_trading_day
                global _last_stock_trading_day
                latest_td = last_trading_day().isoformat()
                if _last_stock_trading_day == latest_td:
                    # Already have this trading day's close — skip stocks
                    all_symbols = {s for s in all_symbols if _classify(s) == "crypto"}
                    if not all_symbols:
                        continue
                else:
                    stock_refresh_td = latest_td

            # Fetch prices once for all symbols
            prices = _fetch_live_prices(list(all_symbols))
            if not prices:
                continue

            global _live_prices, _price_status
            _live_prices.update(prices)

            # Mark stock refresh complete only when all held stocks got prices
            if stock_refresh_td:
                stock_held = {s for s in all_symbols if _classify(s) != "crypto"}
                if not stock_held or stock_held.issubset(prices.keys()):
                    _last_stock_trading_day = stock_refresh_td

            # Update each account's positions
            updated_accounts = 0
            for name, held in account_positions.items():
                account_prices = {s: prices[s] for s in held if s in prices}
                if account_prices:
                    mgr = account_manager.get_manager(name)
                    mgr.update_prices(account_prices)
                    _append_snapshot(mgr)
                    updated_accounts += 1

            _price_status = {
                "last_update": datetime.now().isoformat(),
                "symbols_updated": len(prices),
                "accounts_updated": updated_accounts,
                "last_error": None,
                "source": "coingecko+yahoo" if market_open else "coingecko+polygon_prevclose",
            }
            sys.stderr.write(
                f"[Prices] Updated {len(prices)} prices for {updated_accounts} accounts\n"
            )
        except Exception as exc:
            _price_status["last_error"] = str(exc)
            _price_status["last_update"] = datetime.now().isoformat()
            sys.stderr.write(f"[Prices] Background update error: {exc}\n")


def _append_snapshot(manager: PortfolioManager) -> None:
    """Append a value snapshot to the history file.

    Deduplicates: skips if total_value and positions_value are identical to the
    last recorded snapshot (avoids flat duplicate lines on the chart).
    """
    portfolio = manager.load()
    total_val = round(portfolio.total_value, 2)
    pos_val = round(portfolio.positions_value, 2)
    cash_val = round(portfolio.cash, 2)

    history = _load_history(manager.path)

    history.append({
        "timestamp": datetime.now().isoformat(),
        "total_value": total_val,
        "cash": cash_val,
        "positions_value": pos_val,
    })
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

    def _get_signal_cache(self, params: dict[str, str] | None = None) -> SignalCache:
        """Get SignalCache for the requested account."""
        manager = self._get_manager(params)
        return SignalCache(manager.path.parent)

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
            "/api/signals/meta": self._handle_signals_meta,
            "/api/history": self._handle_history,
            # /api/prices removed — background thread updates all accounts
            "/api/accounts": self._handle_accounts_list,
            "/api/scan/status": self._handle_scan_status,
            "/api/watchlist": self._handle_watchlist_get,
            "/api/price-status": self._handle_price_status,
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
            # /api/update-prices removed — background thread handles this
            "/api/accounts/create": self._handle_account_create,
            "/api/accounts/reset": self._handle_account_reset,
            "/api/accounts/deposit": self._handle_account_deposit,
            "/api/accounts/delete": self._handle_account_delete,
            "/api/accounts/categories": self._handle_account_categories,
            "/api/auto-build": self._handle_auto_build,
            "/api/scan": self._handle_scan_start,
            "/api/scan/cancel": self._handle_scan_cancel,
            "/api/watchlist": self._handle_watchlist_save,
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
        history = sorted(_load_history(manager.path), key=lambda h: h.get("timestamp", ""))
        # Fees: open_fees from actual recorded values; close_fees estimated per asset type
        from signalforge.paper.portfolio import _fee_for_symbol
        open_fees = sum(p.open_fee for p in portfolio.positions)
        close_fees = sum(_fee_for_symbol(p.symbol, p.qty, p.current_price)
                         for p in portfolio.positions)
        total_fees = round(open_fees + close_fees, 2)

        self._send_json({
            "cash": round(portfolio.cash, 2),
            "initial_balance": round(portfolio.initial_balance, 2),
            "total_value": round(portfolio.total_value, 2),
            "total_pnl": round(portfolio.total_pnl, 2),
            "total_pnl_pct": round(portfolio.total_pnl_pct, 2),
            "positions_value": round(portfolio.positions_value, 2),
            "realized_pnl": round(portfolio.realized_pnl, 2),
            "unrealized_pnl": round(portfolio.unrealized_pnl, 2),
            "total_fees": total_fees,
            "open_fees": round(open_fees, 2),
            "close_fees": round(close_fees, 2),
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
        """Return cached signals for this account.

        Query params:
          - scan_type: "full" or "watchlist" (default: "watchlist" if exists, else "full")
        """
        cache = self._get_signal_cache(params)

        # Prefer watchlist cache, fall back to full
        scan_type = params.get("scan_type", "")
        if scan_type:
            signals = cache.load(scan_type)
        else:
            signals = cache.load("watchlist") or cache.load("full")

        # Filter by account's allowed categories
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
            for s in signals
            if allowed_categories is None or self._classify_symbol(s.symbol) in allowed_categories
        ])

    def _handle_signals_meta(self, params: dict[str, str]) -> None:
        """Return signal cache metadata for this account (timestamps, counts)."""
        cache = self._get_signal_cache(params)
        self._send_json({
            "full": cache.metadata("full"),
            "watchlist": cache.metadata("watchlist"),
        })

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
        """Return cached live prices. Background thread keeps them updated."""
        manager = self._get_manager(params)
        if not manager.exists():
            self._send_json({"prices": {}})
            return

        portfolio = manager.load()
        held = {p.symbol for p in portfolio.positions}
        # Return only prices relevant to this account's positions
        account_prices = {s: p for s, p in _live_prices.items() if s in held}
        self._send_json({"prices": account_prices})

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

    def _handle_price_status(self, params: dict[str, str]) -> None:
        """Return background price updater status."""
        self._send_json(_price_status)

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

    def _handle_account_deposit(self, params: dict[str, str]) -> None:
        """Deposit funds into an account. Body: {name, amount}"""
        try:
            body = self._read_body()
            name = body.get("name", "").strip()
            if not name:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing field: name")
                return
            amount = body.get("amount")
            if amount is None:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing field: amount")
                return
            amount = float(amount)
            am = self.__class__.account_manager
            am.deposit(name, amount)
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

    def _handle_account_categories(self, params: dict[str, str]) -> None:
        """Update asset categories for an account. Body: {categories: [...]}"""
        manager = self._get_manager(params)
        try:
            self._ensure_portfolio(manager)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
        try:
            body = self._read_body()
            categories = body.get("categories")
            if not categories or not isinstance(categories, list):
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "Provide {categories: [\"us_stocks\", \"crypto\", ...]}",
                )
                return
            valid = {"us_stocks", "crypto", "futures", "options"}
            categories = [c for c in categories if c in valid]
            if not categories:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "At least one valid category required")
                return
            portfolio = manager.load()
            portfolio.asset_categories = categories
            manager._save(portfolio)
            self._send_json({"categories": categories})
        except (json.JSONDecodeError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    # -- scan & auto-build endpoints -------------------------------------

    def _handle_scan_start(self, params: dict[str, str]) -> None:
        """Start pipeline scan in background thread. Poll /api/scan/status for progress.

        Body: {categories: ["us_stocks", "crypto"], config_only: bool}
        Signals are persisted per-account as full or watchlist cache.
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
            config_only = body.get("config_only", False)
            scan_type = "watchlist" if config_only else "full"

            # Capture the account's signal cache for the background thread
            cache = self._get_signal_cache(params)
            account = (params or {}).get("account", "default")

            global _scan_cancel, _scan_account, _scan_type
            _scan_cancel = False
            _scan_account = account
            _scan_type = scan_type
            _scan_progress = {
                "running": True, "total": 0, "completed": 0,
                "symbol": "", "stage": "starting", "detail": "Initializing pipeline...",
                "phase_pct": 0, "error": None, "log": [],
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
                global _scan_progress, _scan_cancel
                try:
                    sys.stderr.write(f"[Scan] Background scan started: {categories} ({scan_type}) for account={account}\n")
                    signals = generate_real_signals(
                        categories=categories,
                        progress_cb=_on_progress,
                        cancel_flag=lambda: _scan_cancel,
                        config_only=config_only,
                    )
                    # Persist to disk even if partially cancelled
                    if signals:
                        cache.save(signals, scan_type)
                        sys.stderr.write(f"[Scan] Saved {len(signals)} signals to {scan_type} cache for account={account}\n")

                    if _scan_cancel:
                        completed = _scan_progress.get("completed", 0)
                        total = _scan_progress.get("total", 0)
                        log = _scan_progress.get("log", [])
                        _scan_progress = {
                            "running": False, "total": total,
                            "completed": completed,
                            "symbol": "", "stage": "cancelled",
                            "detail": f"Stopped at {completed}/{total} assets",
                            "phase_pct": _scan_progress.get("phase_pct", 0),
                            "error": None, "log": log,
                        }
                        sys.stderr.write(f"[Scan] Cancelled: {len(signals)} signals\n")
                    else:
                        log = _scan_progress.get("log", [])
                        _scan_progress = {
                            "running": False, "total": _scan_progress.get("total", 0),
                            "completed": _scan_progress.get("total", 0),
                            "symbol": "", "stage": "complete",
                            "detail": f"Done: {len(signals)} signals generated",
                            "phase_pct": 100, "error": None, "log": log,
                        }
                        sys.stderr.write(f"[Scan] Done: {len(signals)} signals\n")
                except Exception as exc:
                    log = _scan_progress.get("log", [])
                    _scan_progress = {
                        "running": False, "total": 0, "completed": 0,
                        "symbol": "", "stage": "error", "detail": "",
                        "phase_pct": _scan_progress.get("phase_pct", 0),
                        "error": str(exc), "log": log,
                    }
                    sys.stderr.write(f"[Scan] Failed: {exc}\n")

            thread = threading.Thread(target=_run_scan, daemon=True)
            thread.start()
            self._send_json({"status": "started", "categories": categories, "scan_type": scan_type})
        except (json.JSONDecodeError, TypeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}")

    def _handle_scan_status(self, params: dict[str, str]) -> None:
        """Return detailed scan progress with log history."""
        log = _scan_progress.get("log", [])
        # Get cached signal count from account's cache when scan is not running
        count = None
        if not _scan_progress["running"]:
            cache = self._get_signal_cache(params)
            signals = cache.load("watchlist") or cache.load("full")
            count = len(signals) if signals else 0
        self._send_json({
            "running": _scan_progress["running"],
            "total": _scan_progress.get("total", 0),
            "completed": _scan_progress.get("completed", 0),
            "symbol": _scan_progress.get("symbol", ""),
            "stage": _scan_progress.get("stage", ""),
            "detail": _scan_progress.get("detail", ""),
            "phase_pct": _scan_progress.get("phase_pct", 0),
            "step": _scan_progress.get("step", 0),
            "step_total": _scan_progress.get("step_total", 0),
            "error": _scan_progress.get("error"),
            "count": count,
            "scan_account": _scan_account,
            "scan_type": _scan_type,
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

    # ---- Watchlist (config symbols) API ----

    @staticmethod
    def _get_config_path() -> Path:
        candidates = [
            Path.cwd() / "config" / "default.yaml",
            Path(__file__).parent.parent.parent.parent / "config" / "default.yaml",
            Path.home() / ".signalforge" / "config.yaml",
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]  # default location for new file

    def _handle_watchlist_get(self, params: dict[str, str]) -> None:
        """Return config symbols grouped by category."""
        from signalforge.config import load_config
        cfg = load_config()
        self._send_json({
            "us_stocks": list(cfg.us_stocks),
            "crypto": list(cfg.crypto),
            "futures": list(cfg.futures),
            "options": list(cfg.options),
        })

    def _handle_watchlist_save(self, params: dict[str, str]) -> None:
        """Save watchlist symbols back to config YAML.

        Body: {us_stocks: [...], crypto: [...], futures: [...], options: [...]}
        """
        import yaml
        body = self._read_body()
        config_path = self._get_config_path()

        # Load existing YAML to preserve non-asset settings
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = {}

        # Update only the assets section
        raw.setdefault("assets", {})
        for cat in ("us_stocks", "crypto", "futures", "options"):
            if cat in body:
                raw["assets"][cat] = body[cat]

        with open(config_path, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        self._send_json({"status": "saved", "path": str(config_path)})

    def _handle_auto_build(self, params: dict[str, str]) -> None:
        """Auto-build portfolio from cached scan signals.

        Uses persisted signals from the last scan — no re-scanning.
        Selects top N by confidence, allocates via half-Kelly, opens positions.

        Body: {top_n?: 5, scan_type?: "watchlist"|"full"}
        """
        manager = self._get_manager(params)
        try:
            self._ensure_portfolio(manager)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
        try:
            body = self._read_body()
            top_n = int(body.get("top_n", 5))
            scan_type = body.get("scan_type", "")

            # Load from per-account cache
            cache = self._get_signal_cache(params)
            if scan_type:
                cached_signals = cache.load(scan_type)
            else:
                cached_signals = cache.load("watchlist") or cache.load("full")

            if not cached_signals:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "No cached signals. Run a scan first.",
                )
                return

            # Use account's asset categories for filtering
            portfolio = manager.load()
            categories = portfolio.asset_categories or ["us_stocks", "crypto"]

            # Clear value history before building — fresh chart from build moment
            _save_history(manager.path, [])

            result = build_from_cached_signals(
                manager=manager,
                cached_signals=cached_signals,
                categories=categories,
                top_n=top_n,
            )

            if "error" in result and not result.get("positions_opened"):
                self._send_error_json(HTTPStatus.BAD_REQUEST, result["error"])
                return

            # Force-refresh prices for all held positions after build.
            # Auto Build uses Polygon prev-close which may lag for some
            # symbols (ADRs, late-publishing tickers). A second fetch
            # corrects any stale entry prices immediately.
            portfolio = manager.load()
            held = [p.symbol for p in portfolio.positions]
            if held:
                from signalforge.paper.prices import fetch_prices
                fresh = fetch_prices(held)
                if fresh:
                    manager.update_prices(fresh)
                    _live_prices.update(fresh)

            # Record initial snapshot right after build
            _append_snapshot(manager)
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
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
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

    # Start background price updater for ALL accounts
    price_thread = threading.Thread(
        target=_background_price_updater, args=(am,), daemon=True
    )
    price_thread.start()

    print(f"Paper trading server running on http://localhost:{args.port}")
    print(f"Accounts: {', '.join(accounts) if accounts else '(none yet, will auto-create default)'}")
    print(f"Dashboard: http://localhost:{args.port}/")
    print(f"Background price updates: every {_PRICE_UPDATE_INTERVAL}s for all accounts")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
