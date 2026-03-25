"""Tests for options support: symbol parsing, classification, and provider."""

from __future__ import annotations

import pytest

from signalforge.data.models import (
    AssetType,
    OptionContract,
    asset_from_symbol,
    classify_symbol,
    parse_option_symbol,
)


class TestParseOptionSymbol:
    """Test parsing of option symbols in both human and OCC formats."""

    def test_human_format_call(self) -> None:
        result = parse_option_symbol("AAPL 2026-06-19 200 C")
        assert result is not None
        assert result.underlying == "AAPL"
        assert result.expiration == "2026-06-19"
        assert result.strike == 200.0
        assert result.option_type == "C"
        assert result.occ_symbol == "AAPL260619C00200000"

    def test_human_format_put(self) -> None:
        result = parse_option_symbol("NVDA 2026-03-21 150 P")
        assert result is not None
        assert result.underlying == "NVDA"
        assert result.expiration == "2026-03-21"
        assert result.strike == 150.0
        assert result.option_type == "P"

    def test_human_format_decimal_strike(self) -> None:
        result = parse_option_symbol("SPY 2026-04-17 520.5 C")
        assert result is not None
        assert result.strike == 520.5
        assert result.option_type == "C"

    def test_occ_format_call(self) -> None:
        result = parse_option_symbol("AAPL260619C00200000")
        assert result is not None
        assert result.underlying == "AAPL"
        assert result.expiration == "2026-06-19"
        assert result.strike == 200.0
        assert result.option_type == "C"
        assert result.occ_symbol == "AAPL260619C00200000"

    def test_occ_format_put(self) -> None:
        result = parse_option_symbol("TSLA260320P00250000")
        assert result is not None
        assert result.underlying == "TSLA"
        assert result.expiration == "2026-03-20"
        assert result.strike == 250.0
        assert result.option_type == "P"

    def test_occ_format_fractional_strike(self) -> None:
        # Strike 152.5 encoded as 00152500
        result = parse_option_symbol("SPY260417C00152500")
        assert result is not None
        assert result.strike == 152.5

    def test_not_an_option_stock(self) -> None:
        assert parse_option_symbol("AAPL") is None

    def test_not_an_option_crypto(self) -> None:
        assert parse_option_symbol("BTC/USDT") is None

    def test_not_an_option_futures(self) -> None:
        assert parse_option_symbol("ES=F") is None

    def test_not_an_option_random(self) -> None:
        assert parse_option_symbol("hello world") is None

    def test_case_insensitive_human(self) -> None:
        result = parse_option_symbol("aapl 2026-06-19 200 c")
        assert result is not None
        assert result.underlying == "AAPL"
        assert result.option_type == "C"


class TestClassifySymbolOptions:
    """Test that option symbols are classified correctly."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "AAPL 2026-06-19 200 C",
            "NVDA 2026-03-21 150 P",
            "AAPL260619C00200000",
            "TSLA260320P00250000",
        ],
    )
    def test_options_classified(self, symbol: str) -> None:
        assert classify_symbol(symbol) == AssetType.OPTIONS

    def test_stock_not_confused_with_option(self) -> None:
        assert classify_symbol("AAPL") == AssetType.STOCK

    def test_asset_from_option_symbol(self) -> None:
        asset = asset_from_symbol("AAPL 2026-06-19 200 C")
        assert asset.asset_type == AssetType.OPTIONS


class TestOptionContract:
    def test_frozen(self) -> None:
        contract = OptionContract(
            underlying="AAPL",
            expiration="2026-06-19",
            strike=200.0,
            option_type="C",
        )
        with pytest.raises(AttributeError):
            contract.strike = 210.0  # type: ignore[misc]

    def test_roundtrip_human_to_occ(self) -> None:
        """Parse human format, then parse the OCC output — should match."""
        human = parse_option_symbol("MSFT 2026-09-18 450 C")
        assert human is not None
        occ = parse_option_symbol(human.occ_symbol)
        assert occ is not None
        assert occ.underlying == human.underlying
        assert occ.expiration == human.expiration
        assert occ.strike == human.strike
        assert occ.option_type == human.option_type
