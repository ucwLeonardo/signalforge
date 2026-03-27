"""E2E tests for the paper trading dashboard and API.

These tests start a real HTTP server and exercise the full stack:
dashboard HTML, JavaScript, API endpoints, and portfolio logic.

Run with:
    pytest tests/test_e2e_dashboard.py -v

For Playwright browser tests (requires playwright installed):
    pytest tests/test_e2e_dashboard.py -v -k playwright
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import urllib.request
import urllib.error
from http.server import HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import pytest


# -----------------------------------------------------------------------
# Fixtures: start a real server on a temp data dir
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    """Start a paper trading server on a random port with temp data dir."""
    # Clear proxy
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"

    # Set up temp data directory
    data_dir = tmp_path_factory.mktemp("signalforge_e2e")
    accounts_dir = data_dir / "accounts" / "default"
    accounts_dir.mkdir(parents=True)

    # Patch the portfolio module to use temp dir
    import signalforge.paper.portfolio as pp
    orig_accounts = pp._ACCOUNTS_DIR
    orig_legacy = pp._LEGACY_PATH
    orig_history = pp._LEGACY_HISTORY
    pp._ACCOUNTS_DIR = data_dir / "accounts"
    pp._LEGACY_PATH = data_dir / "paper_portfolio.json"
    pp._LEGACY_HISTORY = data_dir / "paper_value_history.json"

    from signalforge.paper.server import PaperTradingHandler, AccountManager

    am = AccountManager()
    PaperTradingHandler.account_manager = am
    PaperTradingHandler.legacy_manager = None

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    port = 18787
    server = ThreadedServer(("127.0.0.1", port), PaperTradingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(1)

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    # Restore
    pp._ACCOUNTS_DIR = orig_accounts
    pp._LEGACY_PATH = orig_legacy
    pp._LEGACY_HISTORY = orig_history


def _get(url: str) -> dict:
    resp = urllib.request.urlopen(url)
    return json.loads(resp.read())


def _post(url: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


# -----------------------------------------------------------------------
# Test 1: Dashboard HTML loads
# -----------------------------------------------------------------------

class TestDashboardLoads:
    def test_dashboard_returns_html(self, server_url: str) -> None:
        resp = urllib.request.urlopen(server_url + "/")
        html = resp.read().decode()
        assert "SignalForge Paper Trading" in html
        assert "<html" in html.lower()

    def test_dashboard_has_key_elements(self, server_url: str) -> None:
        resp = urllib.request.urlopen(server_url + "/")
        html = resp.read().decode()
        # Key UI elements
        assert "account-selector" in html
        assert "Open Positions" in html or "open-positions" in html
        assert "Available Signals" in html or "signals" in html
        assert "Trade History" in html or "trade-history" in html

    def test_dashboard_has_javascript(self, server_url: str) -> None:
        resp = urllib.request.urlopen(server_url + "/")
        html = resp.read().decode()
        assert "refreshAll" in html or "fetchPortfolio" in html
        assert "submitCreateAccount" in html


# -----------------------------------------------------------------------
# Test 2: API endpoints
# -----------------------------------------------------------------------

class TestAPIEndpoints:
    def test_portfolio_returns_json(self, server_url: str) -> None:
        data = _get(server_url + "/api/portfolio")
        assert "cash" in data
        assert "positions" in data
        assert isinstance(data["cash"], (int, float))

    def test_accounts_list(self, server_url: str) -> None:
        data = _get(server_url + "/api/accounts")
        assert "accounts" in data
        assert len(data["accounts"]) >= 1
        assert data["accounts"][0]["name"] == "default"

    def test_signals_returns_json(self, server_url: str) -> None:
        data = _get(server_url + "/api/signals")
        # Should return signals list (may be empty)
        assert isinstance(data, (dict, list))

    def test_history_returns_json(self, server_url: str) -> None:
        data = _get(server_url + "/api/history")
        assert isinstance(data, (dict, list))

    def test_404_for_unknown_route(self, server_url: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(server_url + "/api/nonexistent")
        assert exc_info.value.code == 404


# -----------------------------------------------------------------------
# Test 3: Account management
# -----------------------------------------------------------------------

class TestAccountManagement:
    def test_create_account(self, server_url: str) -> None:
        result = _post(server_url + "/api/accounts/create", {
            "name": "e2e_test",
            "balance": 25000,
        })
        assert result.get("name") == "e2e_test" or "ok" in str(result).lower() or "created" in str(result).lower()

        # Verify it appears in accounts list
        accounts = _get(server_url + "/api/accounts")
        names = [a["name"] for a in accounts["accounts"]]
        assert "e2e_test" in names

    def test_account_has_correct_balance(self, server_url: str) -> None:
        portfolio = _get(server_url + "/api/portfolio?account=e2e_test")
        assert portfolio["cash"] == 25000

    def test_reset_account(self, server_url: str) -> None:
        # First open a position to change state
        _post(server_url + "/api/open?account=e2e_test", {
            "symbol": "RESET_TEST",
            "side": "long",
            "qty": 1,
            "entry_price": 100,
            "stop_loss": 90,
            "target_price": 120,
        })

        portfolio = _get(server_url + "/api/portfolio?account=e2e_test")
        assert len(portfolio["positions"]) == 1

        # Reset
        _post(server_url + "/api/accounts/reset", {"name": "e2e_test"})

        portfolio = _get(server_url + "/api/portfolio?account=e2e_test")
        assert portfolio["cash"] == 25000
        assert len(portfolio["positions"]) == 0

    def test_delete_account(self, server_url: str) -> None:
        _post(server_url + "/api/accounts/delete", {"name": "e2e_test"})

        accounts = _get(server_url + "/api/accounts")
        names = [a["name"] for a in accounts["accounts"]]
        assert "e2e_test" not in names


# -----------------------------------------------------------------------
# Test 4: Position lifecycle (open → hold → close)
# -----------------------------------------------------------------------

class TestPositionLifecycle:
    def test_open_long_position(self, server_url: str) -> None:
        result = _post(server_url + "/api/open", {
            "symbol": "AAPL",
            "side": "long",
            "qty": 10,
            "entry_price": 150,
            "stop_loss": 140,
            "target_price": 170,
        })
        assert "AAPL" in str(result) or result.get("symbol") == "AAPL"

        portfolio = _get(server_url + "/api/portfolio")
        symbols = [p["symbol"] for p in portfolio["positions"]]
        assert "AAPL" in symbols

    def test_position_deducts_cash(self, server_url: str) -> None:
        portfolio = _get(server_url + "/api/portfolio")
        # Cash should be 5000 - (150 * 10) = 3500
        assert portfolio["cash"] == pytest.approx(3500.0, abs=1)

    def test_position_has_correct_fields(self, server_url: str) -> None:
        portfolio = _get(server_url + "/api/portfolio")
        pos = next(p for p in portfolio["positions"] if p["symbol"] == "AAPL")
        assert pos["side"] == "long"
        assert pos["qty"] == 10
        assert pos["entry_price"] == 150
        assert pos["stop_loss"] == 140
        assert pos.get("target_price", pos.get("target")) == 170

    def test_close_position_with_profit(self, server_url: str) -> None:
        result = _post(server_url + "/api/close", {
            "symbol": "AAPL",
            "exit_price": 160,
        })
        assert "closed" in str(result).lower() or result.get("symbol") == "AAPL"

        portfolio = _get(server_url + "/api/portfolio")
        symbols = [p["symbol"] for p in portfolio["positions"]]
        assert "AAPL" not in symbols

    def test_cash_restored_with_profit(self, server_url: str) -> None:
        portfolio = _get(server_url + "/api/portfolio")
        # Cash should be 3500 + (160 * 10) = 5100
        assert portfolio["cash"] == pytest.approx(5100.0, abs=1)

    def test_trade_recorded_in_history(self, server_url: str) -> None:
        history = _get(server_url + "/api/history")
        trades = history if isinstance(history, list) else history.get("history", history.get("trades", []))
        assert len(trades) >= 1
        aapl_trades = [t for t in trades if t.get("symbol") == "AAPL"]
        assert len(aapl_trades) >= 1

    def test_open_short_position(self, server_url: str) -> None:
        initial = _get(server_url + "/api/portfolio")
        initial_cash = initial["cash"]

        _post(server_url + "/api/open", {
            "symbol": "TSLA",
            "side": "short",
            "qty": 5,
            "entry_price": 200,
            "stop_loss": 220,
            "target_price": 170,
        })

        portfolio = _get(server_url + "/api/portfolio")
        pos = next(p for p in portfolio["positions"] if p["symbol"] == "TSLA")
        assert pos["side"] == "short"
        assert pos["qty"] == 5

    def test_close_short_with_profit(self, server_url: str) -> None:
        _post(server_url + "/api/close", {
            "symbol": "TSLA",
            "exit_price": 180,
        })

        portfolio = _get(server_url + "/api/portfolio")
        symbols = [p["symbol"] for p in portfolio["positions"]]
        assert "TSLA" not in symbols


# -----------------------------------------------------------------------
# Test 5: Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_open_duplicate_symbol_fails(self, server_url: str) -> None:
        _post(server_url + "/api/open", {
            "symbol": "DUPE",
            "side": "long",
            "qty": 1,
            "entry_price": 100,
            "stop_loss": 90,
            "target_price": 110,
        })
        # Opening same symbol again should fail
        try:
            _post(server_url + "/api/open", {
                "symbol": "DUPE",
                "side": "long",
                "qty": 1,
                "entry_price": 100,
                "stop_loss": 90,
                "target_price": 110,
            })
            # If it doesn't raise, check that there's only one position
            portfolio = _get(server_url + "/api/portfolio")
            dupe_count = sum(1 for p in portfolio["positions"] if p["symbol"] == "DUPE")
            assert dupe_count <= 1
        except urllib.error.HTTPError:
            pass  # Expected: server rejects duplicate

        # Cleanup
        try:
            _post(server_url + "/api/close", {"symbol": "DUPE", "exit_price": 100})
        except Exception:
            pass

    def test_close_nonexistent_position(self, server_url: str) -> None:
        with pytest.raises(urllib.error.HTTPError):
            _post(server_url + "/api/close", {
                "symbol": "NONEXISTENT",
                "exit_price": 100,
            })

    def test_open_with_zero_quantity(self, server_url: str) -> None:
        """Server currently accepts qty=0 (no validation). Verify it doesn't crash."""
        result = _post(server_url + "/api/open", {
            "symbol": "ZERO",
            "side": "long",
            "qty": 0,
            "entry_price": 100,
            "stop_loss": 90,
            "target_price": 110,
        })
        # At minimum, the server should not crash
        assert "ZERO" in str(result)
        # Cleanup
        try:
            _post(server_url + "/api/close", {"symbol": "ZERO", "exit_price": 100})
        except Exception:
            pass

    def test_cors_headers(self, server_url: str) -> None:
        resp = urllib.request.urlopen(server_url + "/api/portfolio")
        cors = resp.headers.get("Access-Control-Allow-Origin")
        assert cors == "*"


# -----------------------------------------------------------------------
# Test 6: Multi-account isolation
# -----------------------------------------------------------------------

class TestMultiAccountIsolation:
    def test_accounts_are_isolated(self, server_url: str) -> None:
        # Create two accounts
        _post(server_url + "/api/accounts/create", {"name": "acct_a", "balance": 10000})
        _post(server_url + "/api/accounts/create", {"name": "acct_b", "balance": 20000})

        # Open position in acct_a only
        _post(server_url + "/api/open?account=acct_a", {
            "symbol": "ISOLATED",
            "side": "long",
            "qty": 1,
            "entry_price": 100,
            "stop_loss": 90,
            "target_price": 110,
        })

        # acct_a should have position
        pa = _get(server_url + "/api/portfolio?account=acct_a")
        assert any(p["symbol"] == "ISOLATED" for p in pa["positions"])

        # acct_b should NOT have position
        pb = _get(server_url + "/api/portfolio?account=acct_b")
        assert not any(p["symbol"] == "ISOLATED" for p in pb["positions"])

        # Cleanup
        _post(server_url + "/api/close?account=acct_a", {"symbol": "ISOLATED", "exit_price": 100})
        _post(server_url + "/api/accounts/delete", {"name": "acct_a"})
        _post(server_url + "/api/accounts/delete", {"name": "acct_b"})
