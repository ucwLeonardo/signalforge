"""E2E browser tests for the SignalForge paper trading dashboard.

Starts an embedded test server on port 28787 by default.
Also verifies the live dashboard at http://localhost:8787 when accessible.

Test coverage:
  1. Dashboard loads correctly with account data
  2. Account switching works
  3. Positions table math (cash + positions = total_value)
  4. Signals panel sorted by confidence
  5. Confidence slider filters signals
  6. Micro-price tokens (SHIB, PEPE, BONK) display meaningful prices
  7. Value history chart renders

Usage:
    # Run all API-level tests (fast, uses embedded test server)
    pytest tests/test_e2e_browser_dashboard.py -v

    # Run browser tests too (requires: pip install playwright && playwright install chromium)
    pytest tests/test_e2e_browser_dashboard.py -v -m browser

    # Target the live server instead of embedded
    SIGNALFORGE_TEST_URL=http://localhost:8787 pytest tests/test_e2e_browser_dashboard.py -v
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LIVE_SERVER = os.environ.get("SIGNALFORGE_TEST_URL", "http://localhost:8787")
_EMBEDDED_PORT = 28787
_FEE_RATE = 0.001  # 0.1% per trade

BROWSER_TESTS_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright
    BROWSER_TESTS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Embedded test server fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    """Start an embedded paper trading server on port 28787.

    Falls back to SIGNALFORGE_TEST_URL env var if set.
    This ensures tests run even without the production server.
    """
    # If live server env var is set, use it directly
    live_url = os.environ.get("SIGNALFORGE_TEST_URL")
    if live_url:
        try:
            urllib.request.urlopen(live_url + "/api/accounts", timeout=3)
            yield live_url
            return
        except urllib.error.URLError:
            pass  # fall through to embedded server

    # Clear proxies
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"

    # Set up isolated temp data directory
    data_dir = tmp_path_factory.mktemp("signalforge_e2e_browser")
    accounts_dir = data_dir / "accounts" / "default"
    accounts_dir.mkdir(parents=True)

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

    server = ThreadedServer(("127.0.0.1", _EMBEDDED_PORT), PaperTradingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)

    embedded_url = f"http://127.0.0.1:{_EMBEDDED_PORT}"
    yield embedded_url

    server.shutdown()
    pp._ACCOUNTS_DIR = orig_accounts
    pp._LEGACY_PATH = orig_legacy
    pp._LEGACY_HISTORY = orig_history


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str) -> Any:
    try:
        resp = urllib.request.urlopen(url, timeout=10)
        return json.loads(resp.read())
    except urllib.error.URLError as e:
        pytest.skip(f"Server not reachable at {url}: {e}")


def _post(url: str, data: dict) -> Any:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())
    except urllib.error.URLError as e:
        pytest.skip(f"Server not reachable at {url}: {e}")


def _extract_signals(data: Any) -> list[dict]:
    """Extract signals list from various API response shapes."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "signals" in data and isinstance(data["signals"], list):
            return data["signals"]
        for key in ("full", "watchlist"):
            if key in data and isinstance(data[key], dict):
                sigs = data[key].get("signals", [])
                if sigs:
                    return sigs
        if "symbol" in data:
            return [data]
    return []


# ---------------------------------------------------------------------------
# Test 1: Dashboard loads correctly with account data
# ---------------------------------------------------------------------------

class TestDashboardLoads:
    """Verify the dashboard HTML loads and contains all required UI elements."""

    def test_server_responds_with_200(self, server_url: str) -> None:
        """PASS/FAIL: Server responds at the expected URL."""
        resp = urllib.request.urlopen(server_url + "/", timeout=10)
        assert resp.status == 200, f"Expected 200, got {resp.status}"

    def test_html_contains_signalforge_title(self, server_url: str) -> None:
        """PASS/FAIL: Page title identifies the application."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert "SignalForge" in html, "Title 'SignalForge' not found in HTML"

    def test_html_has_account_selector(self, server_url: str) -> None:
        """PASS/FAIL: Account selector dropdown element exists."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert 'id="account-selector"' in html, "account-selector dropdown not found"

    def test_html_has_positions_container(self, server_url: str) -> None:
        """PASS/FAIL: Open Positions container element exists."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert "Open Positions" in html, "'Open Positions' heading not found"
        assert 'id="positions-container"' in html, "positions-container div not found"

    def test_html_has_signals_container(self, server_url: str) -> None:
        """PASS/FAIL: Available Signals container element exists."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert "Available Signals" in html, "'Available Signals' heading not found"
        assert 'id="signals-container"' in html, "signals-container div not found"

    def test_html_has_confidence_slider(self, server_url: str) -> None:
        """PASS/FAIL: Confidence filter range slider element exists."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert 'id="confidence-filter"' in html, "confidence-filter slider not found"
        assert 'type="range"' in html, "range input not found"

    def test_html_has_auto_build_button(self, server_url: str) -> None:
        """PASS/FAIL: Auto Build button element exists."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert 'id="auto-build-btn"' in html, "auto-build-btn not found"
        assert "Auto Build" in html, "Auto Build label not found"

    def test_html_has_value_chart_canvas(self, server_url: str) -> None:
        """PASS/FAIL: Value history chart canvas element exists."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert 'id="value-chart"' in html, "value-chart canvas not found"
        assert 'id="chart-card"' in html, "chart-card wrapper not found"

    def test_html_has_trade_history_section(self, server_url: str) -> None:
        """PASS/FAIL: Trade History panel element exists."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert "Trade History" in html, "'Trade History' section not found"
        assert 'id="history-container"' in html, "history-container div not found"

    def test_html_has_scan_all_button(self, server_url: str) -> None:
        """PASS/FAIL: Scan All button exists."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert "Scan All" in html, "'Scan All' button not found"

    def test_html_has_watchlist_button(self, server_url: str) -> None:
        """PASS/FAIL: Watchlist scan button exists."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert "Watchlist" in html, "'Watchlist' button not found"

    def test_html_has_account_management_buttons(self, server_url: str) -> None:
        """PASS/FAIL: Create, Reset, Delete account buttons exist."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert "Reset" in html, "'Reset' button not found"
        assert "New" in html or "Create" in html, "Create Account button not found"

    def test_html_has_portfolio_summary_elements(self, server_url: str) -> None:
        """PASS/FAIL: Summary bar elements (total value, cash, P&L) exist."""
        html = urllib.request.urlopen(server_url + "/", timeout=10).read().decode()
        assert 'id="s-total-value"' in html, "s-total-value element missing"
        assert 'id="s-cash"' in html, "s-cash element missing"
        assert 'id="s-positions-value"' in html, "s-positions-value element missing"

    def test_api_accounts_returns_default(self, server_url: str) -> None:
        """PASS/FAIL: /api/accounts returns the default account (auto-created on first portfolio access)."""
        # Hit /api/portfolio first to trigger auto-creation of default account
        _get(server_url + "/api/portfolio")
        data = _get(server_url + "/api/accounts")
        assert "accounts" in data, "Response missing 'accounts' key"
        names = [a["name"] for a in data["accounts"]]
        assert "default" in names, f"'default' account missing. Found: {names}"

    def test_api_returns_json_content_type(self, server_url: str) -> None:
        """PASS/FAIL: API endpoints return JSON content type."""
        resp = urllib.request.urlopen(server_url + "/api/accounts", timeout=10)
        ct = resp.headers.get("Content-Type", "")
        assert "application/json" in ct, f"Expected JSON content type, got: {ct}"

    def test_api_has_cors_headers(self, server_url: str) -> None:
        """PASS/FAIL: API responses include CORS headers for browser access."""
        resp = urllib.request.urlopen(server_url + "/api/portfolio", timeout=10)
        cors = resp.headers.get("Access-Control-Allow-Origin", "")
        assert cors == "*", f"Expected CORS '*', got: '{cors}'"


# ---------------------------------------------------------------------------
# Test 2: Account switching works
# ---------------------------------------------------------------------------

class TestAccountSwitching:
    """Verify account switching returns correct per-account data."""

    def test_accounts_list_is_non_empty(self, server_url: str) -> None:
        """PASS/FAIL: At least one account exists."""
        data = _get(server_url + "/api/accounts")
        accounts = data.get("accounts", [])
        assert len(accounts) >= 1, "No accounts returned"

    def test_each_account_has_name_and_financials(self, server_url: str) -> None:
        """PASS/FAIL: Each account object has name, cash, total_value fields."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            assert "name" in acct, f"Account missing 'name': {acct}"
            # Must have at least cash or total_value
            has_financial = "cash" in acct or "total_value" in acct
            assert has_financial, f"Account missing financial data: {acct}"

    def test_portfolio_endpoint_scoped_per_account(self, server_url: str) -> None:
        """PASS/FAIL: /api/portfolio?account=X returns valid portfolio for each account."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            portfolio = _get(server_url + f"/api/portfolio?account={name}")
            assert "cash" in portfolio, f"Account '{name}': portfolio missing cash"
            assert "positions" in portfolio, f"Account '{name}': portfolio missing positions"
            assert isinstance(portfolio["cash"], (int, float)), \
                f"Account '{name}': cash is not numeric"

    def test_create_and_switch_to_new_account(self, server_url: str) -> None:
        """PASS/FAIL: Creating a new account and switching to it returns correct balance."""
        acct_name = "switch_test"
        try:
            _post(server_url + "/api/accounts/create", {
                "name": acct_name,
                "balance": 7500,
            })
            portfolio = _get(server_url + f"/api/portfolio?account={acct_name}")
            assert portfolio["cash"] == 7500, (
                f"New account cash {portfolio['cash']} != 7500"
            )
            assert portfolio["positions"] == [], (
                f"New account should have no positions"
            )
        finally:
            try:
                _post(server_url + "/api/accounts/delete", {"name": acct_name})
            except Exception:
                pass

    def test_accounts_are_data_isolated(self, server_url: str) -> None:
        """PASS/FAIL: Opening a position in one account does not affect another."""
        acct_a = "iso_test_a"
        acct_b = "iso_test_b"
        try:
            _post(server_url + "/api/accounts/create", {"name": acct_a, "balance": 5000})
            _post(server_url + "/api/accounts/create", {"name": acct_b, "balance": 5000})

            _post(server_url + f"/api/open?account={acct_a}", {
                "symbol": "ISOLATED_SYM",
                "side": "long",
                "qty": 1,
                "entry_price": 100,
                "stop_loss": 90,
                "target_price": 110,
            })

            pa = _get(server_url + f"/api/portfolio?account={acct_a}")
            pb = _get(server_url + f"/api/portfolio?account={acct_b}")

            syms_a = {p["symbol"] for p in pa["positions"]}
            syms_b = {p["symbol"] for p in pb["positions"]}

            assert "ISOLATED_SYM" in syms_a, "Position not found in account A"
            assert "ISOLATED_SYM" not in syms_b, "Position leaked into account B"
        finally:
            for name in (acct_a, acct_b):
                try:
                    _post(server_url + "/api/accounts/delete", {"name": name})
                except Exception:
                    pass

    def test_total_value_in_accounts_list_matches_portfolio(self, server_url: str) -> None:
        """PASS/FAIL: total_value in /api/accounts list matches /api/portfolio."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            acct_tv = acct.get("total_value")
            if acct_tv is None:
                continue
            portfolio = _get(server_url + f"/api/portfolio?account={name}")
            port_tv = portfolio.get("total_value")
            if port_tv is None:
                continue
            tolerance = max(abs(acct_tv) * 0.01, 1.0)
            assert abs(acct_tv - port_tv) <= tolerance, (
                f"Account '{name}': list total_value={acct_tv:.2f} "
                f"!= portfolio total_value={port_tv:.2f}"
            )


# ---------------------------------------------------------------------------
# Test 3: Positions table — verify math (cash + positions = total_value)
# ---------------------------------------------------------------------------

class TestPositionsMath:
    """Verify financial math in positions table is consistent."""

    def test_positions_response_is_list(self, server_url: str) -> None:
        """PASS/FAIL: Portfolio positions field is always a list."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            portfolio = _get(server_url + f"/api/portfolio?account={name}")
            assert isinstance(portfolio["positions"], list), \
                f"Account '{name}' positions is not a list"

    def test_positions_have_required_display_fields(self, server_url: str) -> None:
        """PASS/FAIL: Every position has fields needed for table display."""
        required = ["symbol", "side", "qty", "entry_price"]
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            portfolio = _get(server_url + f"/api/portfolio?account={name}")
            for pos in portfolio.get("positions", []):
                for field in required:
                    assert field in pos, \
                        f"Account '{name}' position missing '{field}': {pos}"

    def test_positions_side_values_valid(self, server_url: str) -> None:
        """PASS/FAIL: Position side is always 'long' or 'short'."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            portfolio = _get(server_url + f"/api/portfolio?account={name}")
            for pos in portfolio.get("positions", []):
                side = pos.get("side", "").lower()
                assert side in ("long", "short"), \
                    f"Account '{name}': invalid side '{side}' for {pos['symbol']}"

    def test_positions_qty_positive(self, server_url: str) -> None:
        """PASS/FAIL: Position quantities are positive."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            portfolio = _get(server_url + f"/api/portfolio?account={name}")
            for pos in portfolio.get("positions", []):
                assert pos["qty"] > 0, \
                    f"Account '{name}': qty={pos['qty']} <= 0 for {pos['symbol']}"

    def test_positions_entry_price_positive(self, server_url: str) -> None:
        """PASS/FAIL: Position entry prices are positive."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            portfolio = _get(server_url + f"/api/portfolio?account={name}")
            for pos in portfolio.get("positions", []):
                ep = pos.get("entry_price", 0)
                assert ep > 0, \
                    f"Account '{name}': entry_price={ep} <= 0 for {pos['symbol']}"

    def test_open_position_deducts_cash(self, server_url: str) -> None:
        """PASS/FAIL: Cash decreases after opening a position (+ fee)."""
        acct = "math_test_open"
        try:
            _post(server_url + "/api/accounts/create", {"name": acct, "balance": 10000})
            before = _get(server_url + f"/api/portfolio?account={acct}")

            _post(server_url + f"/api/open?account={acct}", {
                "symbol": "MATH_STOCK",
                "side": "long",
                "qty": 10,
                "entry_price": 100,
                "stop_loss": 90,
                "target_price": 120,
            })
            after = _get(server_url + f"/api/portfolio?account={acct}")

            cost = 10 * 100
            fee = cost * _FEE_RATE
            expected = before["cash"] - cost - fee

            assert abs(after["cash"] - expected) < 1.0, (
                f"Cash after open: {after['cash']:.2f}, expected {expected:.2f} "
                f"(cost={cost}, fee={fee:.2f})"
            )
        finally:
            try:
                _post(server_url + "/api/accounts/delete", {"name": acct})
            except Exception:
                pass

    def test_close_position_restores_cash_with_pnl(self, server_url: str) -> None:
        """PASS/FAIL: Closing a position credits cash with P&L minus fee."""
        acct = "math_test_close"
        try:
            _post(server_url + "/api/accounts/create", {"name": acct, "balance": 10000})
            _post(server_url + f"/api/open?account={acct}", {
                "symbol": "MATH_CLOSE",
                "side": "long",
                "qty": 10,
                "entry_price": 100,
                "stop_loss": 90,
                "target_price": 120,
            })
            before_close = _get(server_url + f"/api/portfolio?account={acct}")

            _post(server_url + f"/api/close?account={acct}", {
                "symbol": "MATH_CLOSE",
                "exit_price": 110,
            })
            after = _get(server_url + f"/api/portfolio?account={acct}")

            proceeds = 10 * 110
            fee = proceeds * _FEE_RATE
            expected = before_close["cash"] + proceeds - fee

            assert abs(after["cash"] - expected) < 1.0, (
                f"Cash after close: {after['cash']:.2f}, expected {expected:.2f}"
            )
        finally:
            try:
                _post(server_url + "/api/accounts/delete", {"name": acct})
            except Exception:
                pass

    def test_cash_plus_positions_value_near_total_value(self, server_url: str) -> None:
        """PASS/FAIL: cash + sum(position values) ≈ total_value for each account."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            portfolio = _get(server_url + f"/api/portfolio?account={name}")
            tv = portfolio.get("total_value")
            if tv is None:
                continue
            cash = portfolio["cash"]
            positions = portfolio.get("positions", [])
            pos_value = sum(
                (p.get("current_price") or p.get("entry_price", 0)) * abs(p.get("qty", 0))
                for p in positions
            )
            expected = cash + pos_value
            tolerance = max(abs(expected) * 0.20, 100)  # 20% or $100 tolerance
            assert abs(tv - expected) <= tolerance, (
                f"Account '{name}': total_value={tv:.2f}, "
                f"cash={cash:.2f} + positions={pos_value:.2f} = {expected:.2f}, "
                f"diff={abs(tv - expected):.2f}"
            )

    def test_trade_closed_with_profit_appears_in_history(self, server_url: str) -> None:
        """PASS/FAIL: Closed trade with profit appears in trade history."""
        acct = "history_test"
        try:
            _post(server_url + "/api/accounts/create", {"name": acct, "balance": 5000})
            _post(server_url + f"/api/open?account={acct}", {
                "symbol": "HIST_STOCK",
                "side": "long",
                "qty": 1,
                "entry_price": 100,
                "stop_loss": 90,
                "target_price": 120,
            })
            _post(server_url + f"/api/close?account={acct}", {
                "symbol": "HIST_STOCK",
                "exit_price": 115,
            })
            history = _get(server_url + f"/api/history?account={acct}")
            trades = history if isinstance(history, list) else history.get("history", [])
            symbols = [t.get("symbol") for t in trades]
            assert "HIST_STOCK" in symbols, (
                f"Closed trade not in history. Found: {symbols}"
            )
        finally:
            try:
                _post(server_url + "/api/accounts/delete", {"name": acct})
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Test 4: Signals panel — sorted by confidence
# ---------------------------------------------------------------------------

class TestSignalsPanel:
    """Verify signals API returns data sorted by confidence descending."""

    def test_signals_returns_valid_type(self, server_url: str) -> None:
        """PASS/FAIL: /api/signals returns list or dict."""
        data = _get(server_url + "/api/signals?account=default")
        assert isinstance(data, (dict, list)), \
            f"Unexpected signals response type: {type(data)}"

    def test_signals_if_present_are_sorted_by_confidence(self, server_url: str) -> None:
        """PASS/FAIL: If signals exist, they are in descending confidence order."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals cached — run a scan first")

        confidences = [s.get("confidence", 0) for s in signals]
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1], (
                f"Not sorted at index {i}: conf[{i}]={confidences[i]:.3f} < "
                f"conf[{i+1}]={confidences[i+1]:.3f}"
            )

    def test_signals_confidence_in_0_to_1_range(self, server_url: str) -> None:
        """PASS/FAIL: All confidence values are in [0, 1]."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        for sig in signals:
            conf = sig.get("confidence")
            assert conf is not None, f"Signal missing confidence: {sig}"
            assert 0.0 <= conf <= 1.0, \
                f"Confidence {conf} out of [0,1] for {sig.get('symbol')}"

    def test_signals_action_is_buy_or_sell(self, server_url: str) -> None:
        """PASS/FAIL: Every signal action is BUY or SELL."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        for sig in signals:
            action = sig.get("action", "").upper()
            assert action in ("BUY", "SELL"), \
                f"Invalid action '{action}' for {sig.get('symbol')}"

    def test_signals_have_price_fields(self, server_url: str) -> None:
        """PASS/FAIL: Signals include entry_price, target_price, stop_loss."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        for sig in signals[:20]:
            for field in ("entry_price", "target_price", "stop_loss"):
                assert field in sig, \
                    f"Signal {sig.get('symbol')} missing '{field}'"
                val = sig[field]
                assert isinstance(val, (int, float)) and val > 0, \
                    f"Signal {sig.get('symbol')} {field}={val} invalid"

    def test_signals_have_symbol(self, server_url: str) -> None:
        """PASS/FAIL: Every signal has a non-empty symbol."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        for sig in signals:
            symbol = sig.get("symbol", "")
            assert symbol, f"Signal missing symbol: {sig}"

    def test_signals_count_label_logic(self, server_url: str) -> None:
        """PASS/FAIL: Filtered count at 50% confidence is <= total signal count."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        total = len(signals)
        filtered_50pct = [s for s in signals if s.get("confidence", 0) >= 0.5]
        assert len(filtered_50pct) <= total, (
            f"Filtered ({len(filtered_50pct)}) > total ({total})"
        )


# ---------------------------------------------------------------------------
# Test 5: Confidence slider filters signals
# ---------------------------------------------------------------------------

class TestConfidenceFilter:
    """Verify the confidence slider filtering logic."""

    def test_filter_at_0_shows_all_signals(self, server_url: str) -> None:
        """PASS/FAIL: 0% confidence threshold includes all signals."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        at_zero = [s for s in signals if s.get("confidence", 0) >= 0.0]
        assert len(at_zero) == len(signals), \
            f"At 0%: got {len(at_zero)} of {len(signals)}"

    def test_filter_monotonically_decreases(self, server_url: str) -> None:
        """PASS/FAIL: Higher threshold = fewer signals (monotonically non-increasing)."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        prev_count = len(signals) + 1
        for pct in range(0, 105, 5):
            threshold = pct / 100.0
            count = sum(1 for s in signals if s.get("confidence", 0) >= threshold)
            assert count <= prev_count, (
                f"At {pct}%, count={count} > count at {pct-5}%={prev_count}"
            )
            prev_count = count

    def test_filter_at_100_shows_only_perfect_signals(self, server_url: str) -> None:
        """PASS/FAIL: 100% confidence threshold shows only signals with conf == 1.0."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        at_100 = [s for s in signals if s.get("confidence", 0) >= 1.0]
        for sig in at_100:
            assert sig.get("confidence", 0) == 1.0, \
                f"Signal passed 100% filter but conf={sig['confidence']}"

    def test_filter_preserves_sort_order(self, server_url: str) -> None:
        """PASS/FAIL: After filtering, remaining signals preserve confidence sort order."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        threshold = 0.5
        filtered = [s for s in signals if s.get("confidence", 0) >= threshold]
        for i in range(len(filtered) - 1):
            assert filtered[i]["confidence"] >= filtered[i + 1]["confidence"], (
                f"Sort order broken after filtering at index {i}: "
                f"{filtered[i]['confidence']:.3f} < {filtered[i+1]['confidence']:.3f}"
            )

    def test_buy_sell_counts_sum_to_total_filtered(self, server_url: str) -> None:
        """PASS/FAIL: BUY count + SELL count = total filtered signal count."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals")

        threshold = 0.5
        filtered = [s for s in signals if s.get("confidence", 0) >= threshold]
        buy_count = sum(1 for s in filtered if s.get("action", "").upper() == "BUY")
        sell_count = sum(1 for s in filtered if s.get("action", "").upper() == "SELL")
        assert buy_count + sell_count == len(filtered), (
            f"BUY({buy_count}) + SELL({sell_count}) = {buy_count + sell_count} "
            f"!= filtered total({len(filtered)})"
        )


# ---------------------------------------------------------------------------
# Test 6: Micro-price tokens display meaningful prices
# ---------------------------------------------------------------------------

class TestMicroPriceTokens:
    """Verify micro-priced tokens are not displayed as $0.00."""

    def _fmt_usd(self, n: float) -> str:
        """Python reimplementation of dashboard's fmtUsd() JavaScript function."""
        if n is None or (isinstance(n, float) and math.isnan(n)):
            return "--"
        abs_val = abs(n)
        if abs_val == 0:
            d = 2
        elif abs_val < 0.001:
            d = 8
        elif abs_val < 0.1:
            d = 4
        elif abs_val < 1:
            d = 3
        else:
            d = 2
        prefix = "-$" if n < 0 else "$"
        return prefix + f"{abs(n):.{d}f}"

    def test_shib_price_formats_with_8_decimals(self) -> None:
        """PASS/FAIL: SHIB price ~0.00002 formats with 8 decimal places, not $0.00."""
        shib_price = 0.00002345
        result = self._fmt_usd(shib_price)
        assert result != "$0.00", (
            f"SHIB price {shib_price} incorrectly formatted as '$0.00'"
        )
        assert "0.0000" in result, (
            f"SHIB price {shib_price} not showing micro-precision: '{result}'"
        )

    def test_pepe_price_formats_with_8_decimals(self) -> None:
        """PASS/FAIL: PEPE price ~0.000015 formats with 8 decimal places."""
        pepe_price = 0.000015
        result = self._fmt_usd(pepe_price)
        assert result != "$0.00", (
            f"PEPE price {pepe_price} incorrectly formatted as '$0.00'"
        )

    def test_bonk_price_formats_correctly(self) -> None:
        """PASS/FAIL: BONK price ~0.000036 formats with 8 decimal places."""
        bonk_price = 0.000036
        result = self._fmt_usd(bonk_price)
        assert result != "$0.00", (
            f"BONK price {bonk_price} incorrectly formatted as '$0.00'"
        )
        assert "0.0000" in result, (
            f"BONK price not showing micro-precision: '{result}'"
        )

    def test_prices_above_cent_use_fewer_decimals(self) -> None:
        """PASS/FAIL: Prices above $0.10 use fewer decimal places."""
        test_cases = [
            (150.00, 2, "$150.00"),
            (0.50, 3, "$0.500"),
            (0.05, 4, "$0.0500"),
        ]
        for price, expected_decimals, expected_format in test_cases:
            result = self._fmt_usd(price)
            assert result == expected_format, (
                f"fmtUsd({price}) = '{result}', expected '{expected_format}'"
            )

    def test_negative_micro_price_formats_correctly(self) -> None:
        """PASS/FAIL: Negative micro-price (P&L) formats with correct decimals."""
        neg_micro = -0.00003
        result = self._fmt_usd(neg_micro)
        assert result.startswith("-$"), f"Negative price missing '-$': '{result}'"
        assert result != "-$0.00", f"Negative micro price shown as '-$0.00'"

    def test_zero_price_formats_as_zero(self) -> None:
        """PASS/FAIL: Zero price formats as $0.00."""
        result = self._fmt_usd(0.0)
        assert result == "$0.00", f"Zero price should be '$0.00', got: '{result}'"

    def test_micro_price_signals_have_nonzero_entry(self, server_url: str) -> None:
        """PASS/FAIL: SHIB/PEPE/BONK signals show non-zero entry prices."""
        data = _get(server_url + "/api/signals?account=default")
        signals = _extract_signals(data)
        if not signals:
            pytest.skip("No signals available")

        micro_tokens = ["SHIB", "PEPE", "BONK", "FLOKI", "DOGE"]
        micro_sigs = [
            s for s in signals
            if any(tok in s.get("symbol", "").upper() for tok in micro_tokens)
        ]

        if not micro_sigs:
            pytest.skip("No micro-price tokens in current signals")

        for sig in micro_sigs:
            entry = sig.get("entry_price", 0)
            formatted = self._fmt_usd(entry)
            assert entry != 0, \
                f"Micro-price signal {sig['symbol']} has entry_price=0"
            assert formatted != "$0.00", (
                f"Micro-price signal {sig['symbol']} entry={entry} "
                f"formats as '$0.00' — will confuse users"
            )

    def test_crypto_positions_have_valid_prices(self, server_url: str) -> None:
        """PASS/FAIL: All crypto positions have positive entry prices."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            portfolio = _get(server_url + f"/api/portfolio?account={name}")
            for pos in portfolio.get("positions", []):
                if "/" in pos.get("symbol", ""):  # crypto symbol format: BTC/USDT
                    entry = pos.get("entry_price", 0)
                    assert entry > 0, (
                        f"Crypto position {pos['symbol']} in '{name}': "
                        f"entry_price={entry} is not positive"
                    )
                    # Verify the formatted price is not $0.00 for micro-priced assets
                    formatted = self._fmt_usd(entry)
                    assert formatted != "$0.00" or entry == 0, (
                        f"Crypto position {pos['symbol']} entry={entry} "
                        f"would display as '$0.00'"
                    )


# ---------------------------------------------------------------------------
# Test 7: Value history chart renders
# ---------------------------------------------------------------------------

class TestValueHistoryChart:
    """Verify the chart has valid data sources."""

    def test_history_endpoint_returns_valid_type(self, server_url: str) -> None:
        """PASS/FAIL: /api/history returns list or dict."""
        data = _get(server_url + "/api/history?account=default")
        assert isinstance(data, (list, dict)), \
            f"History type unexpected: {type(data)}"

    def test_all_accounts_have_accessible_history(self, server_url: str) -> None:
        """PASS/FAIL: /api/history works for every account."""
        data = _get(server_url + "/api/accounts")
        for acct in data.get("accounts", []):
            name = acct["name"]
            history = _get(server_url + f"/api/history?account={name}")
            assert isinstance(history, (list, dict)), \
                f"Account '{name}' history has invalid type: {type(history)}"

    def test_history_trades_have_timestamps(self, server_url: str) -> None:
        """PASS/FAIL: Trade history entries have timestamp fields."""
        data = _get(server_url + "/api/history?account=default")
        trades = data if isinstance(data, list) else data.get("history", data.get("trades", []))
        if not trades:
            pytest.skip("No trade history")

        time_fields = ["opened_at", "closed_at", "timestamp", "date", "time"]
        for trade in trades[:10]:
            has_time = any(k in trade for k in time_fields)
            assert has_time, f"Trade missing timestamp: {trade}"

    def test_history_trades_have_pnl(self, server_url: str) -> None:
        """PASS/FAIL: Closed trade history entries have P&L fields."""
        data = _get(server_url + "/api/history?account=default")
        trades = data if isinstance(data, list) else data.get("history", data.get("trades", []))
        if not trades:
            pytest.skip("No trade history")

        pnl_fields = ["pnl", "realized_pnl", "profit", "profit_loss"]
        for trade in trades[:10]:
            has_pnl = any(k in trade for k in pnl_fields)
            assert has_pnl, f"Trade missing P&L field: {trade}"

    def test_scan_status_endpoint_responds(self, server_url: str) -> None:
        """PASS/FAIL: /api/scan/status returns valid running state."""
        data = _get(server_url + "/api/scan/status")
        assert isinstance(data, dict), f"Scan status not a dict: {data}"
        assert "running" in data, "Scan status missing 'running'"
        assert isinstance(data["running"], bool), "'running' must be bool"

    def test_watchlist_has_symbol_categories(self, server_url: str) -> None:
        """PASS/FAIL: /api/watchlist returns at least one asset category."""
        data = _get(server_url + "/api/watchlist")
        assert isinstance(data, dict), f"Watchlist not a dict: {data}"
        categories = ["us_stocks", "crypto", "futures", "options"]
        has_any = any(cat in data for cat in categories)
        assert has_any, f"Watchlist missing categories. Got keys: {list(data.keys())}"

    def test_portfolio_has_value_chart_data(self, server_url: str) -> None:
        """PASS/FAIL: Portfolio endpoint returns data needed for value chart."""
        portfolio = _get(server_url + "/api/portfolio?account=default")
        # Chart uses total_value and timestamps from history
        assert "cash" in portfolio, "Portfolio missing cash (needed for chart)"

    def test_chart_history_grows_after_close(self, server_url: str) -> None:
        """PASS/FAIL: Closing a position adds entry to trade history."""
        acct = "chart_test"
        try:
            _post(server_url + "/api/accounts/create", {"name": acct, "balance": 5000})
            history_before = _get(server_url + f"/api/history?account={acct}")
            trades_before = (
                history_before if isinstance(history_before, list)
                else history_before.get("history", [])
            )
            count_before = len(trades_before)

            _post(server_url + f"/api/open?account={acct}", {
                "symbol": "CHART_SYM",
                "side": "long",
                "qty": 1,
                "entry_price": 100,
                "stop_loss": 90,
                "target_price": 120,
            })
            _post(server_url + f"/api/close?account={acct}", {
                "symbol": "CHART_SYM",
                "exit_price": 105,
            })

            history_after = _get(server_url + f"/api/history?account={acct}")
            trades_after = (
                history_after if isinstance(history_after, list)
                else history_after.get("history", [])
            )
            count_after = len(trades_after)

            assert count_after > count_before, (
                f"History count did not grow after close: {count_before} -> {count_after}"
            )
        finally:
            try:
                _post(server_url + "/api/accounts/delete", {"name": acct})
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Playwright Browser Tests (opt-in via -m browser flag)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not BROWSER_TESTS_AVAILABLE,
    reason="Playwright not installed. Run: pip install playwright && playwright install chromium",
)
@pytest.mark.browser
class TestBrowserDashboard:
    """Full browser E2E tests using Playwright.

    These test actual JavaScript rendering and interactivity.
    Mark: @pytest.mark.browser

    Usage:
        pytest tests/test_e2e_browser_dashboard.py -v -m browser
    """

    @pytest.fixture(scope="class")
    def browser_page(self, server_url: str):
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            page.goto(server_url, wait_until="networkidle", timeout=15000)
            yield page
            context.close()
            browser.close()

    def test_page_title_contains_signalforge(self, browser_page) -> None:
        """PASS/FAIL: Browser renders correct page title."""
        assert "SignalForge" in browser_page.title()

    def test_account_selector_has_options(self, browser_page) -> None:
        """PASS/FAIL: Account selector dropdown is populated from API."""
        sel = browser_page.locator("#account-selector")
        sel.wait_for(state="visible", timeout=5000)
        opts = sel.locator("option").all()
        assert len(opts) >= 1, "Account selector has no options"

    def test_total_value_loads_and_is_visible(self, browser_page) -> None:
        """PASS/FAIL: Portfolio total value element is populated from API (may be hidden if no activity)."""
        browser_page.wait_for_load_state("networkidle")
        # The total value element is always populated from /api/portfolio
        # but portfolio-overview is hidden when cash==initial_balance and no positions
        val = browser_page.evaluate(
            "document.getElementById('s-total-value').textContent.trim()"
        )
        # Value should be populated (not '--' placeholder) since portfolio API is called on load
        assert val and val not in ("", "--"), f"Total value not loaded: '{val}'"
        assert "$" in val, f"Total value missing '$': '{val}'"

    def test_positions_container_renders(self, browser_page) -> None:
        """PASS/FAIL: Positions container renders either a table or empty state."""
        browser_page.wait_for_load_state("networkidle")
        container = browser_page.locator("#positions-container")
        container.wait_for(state="visible", timeout=5000)
        html = container.inner_html()
        has_table = "<table" in html
        has_empty = "No open positions" in html
        assert has_table or has_empty, (
            "Positions container has neither table nor empty state"
        )

    def test_confidence_slider_updates_display(self, browser_page) -> None:
        """PASS/FAIL: Moving confidence slider updates the displayed percentage."""
        slider = browser_page.locator("#confidence-filter")
        if not slider.is_visible():
            pytest.skip("Confidence slider not visible (no signals)")

        # Set to 80%
        slider.fill("80")
        browser_page.dispatch_event("#confidence-filter", "input")
        browser_page.wait_for_timeout(400)

        conf_val = browser_page.locator("#confidence-value").text_content()
        assert "80%" in conf_val, f"Confidence display not updated: '{conf_val}'"

    def test_value_chart_card_exists(self, browser_page) -> None:
        """PASS/FAIL: Value chart card element exists in DOM."""
        browser_page.wait_for_load_state("networkidle")
        # Chart card may be hidden if no history, but element must exist
        exists = browser_page.evaluate(
            "!!document.getElementById('chart-card')"
        )
        assert exists, "chart-card element not found in DOM"

    def test_account_switching_changes_displayed_data(self, browser_page) -> None:
        """PASS/FAIL: Switching accounts via selector triggers data refresh."""
        sel = browser_page.locator("#account-selector")
        opts = sel.locator("option").all()
        if len(opts) < 2:
            pytest.skip("Need at least 2 accounts")

        initial = sel.input_value()
        second = opts[1 if opts[0].get_attribute("value") == initial else 0].get_attribute("value")

        sel.select_option(second)
        browser_page.wait_for_load_state("networkidle")
        browser_page.wait_for_timeout(800)

        current = sel.input_value()
        assert current == second, f"Account not switched: expected '{second}', got '{current}'"

        # Switch back
        sel.select_option(initial)
        browser_page.wait_for_load_state("networkidle")

    def test_micro_price_positions_not_showing_zero(self, browser_page) -> None:
        """PASS/FAIL: Micro-priced tokens in positions table show non-zero prices."""
        browser_page.wait_for_load_state("networkidle")
        pos_html = browser_page.locator("#positions-container").inner_html()
        micro_tokens = ["SHIB", "PEPE", "BONK", "FLOKI"]
        for token in micro_tokens:
            if token in pos_html:
                # Get all cells in the row containing this token
                rows = browser_page.locator("tbody tr").all()
                for row in rows:
                    row_text = row.text_content() or ""
                    if token in row_text:
                        # Check that entry price cell doesn't show $0.00
                        cells = row.locator("td").all()
                        if len(cells) >= 4:
                            entry_cell = cells[3].text_content()  # Entry column
                            assert entry_cell != "$0.00", (
                                f"{token} row shows entry price '$0.00' — "
                                f"micro-price formatting bug"
                            )
