"""Tests for the report generator."""

from __future__ import annotations

import json

import pytest

from signalforge.data.models import TradeAction, TradeTarget
from signalforge.output.report import ReportGenerator


@pytest.fixture()
def sample_targets() -> list[TradeTarget]:
    return [
        TradeTarget(
            symbol="AAPL",
            action=TradeAction.BUY,
            entry_price=180.50,
            target_price=200.00,
            stop_loss=170.00,
            risk_reward_ratio=1.86,
            confidence=0.78,
            horizon_days=5,
            rationale="Bullish signal",
        ),
        TradeTarget(
            symbol="BTC/USDT",
            action=TradeAction.SELL,
            entry_price=67000.0,
            target_price=60000.0,
            stop_loss=72000.0,
            risk_reward_ratio=1.4,
            confidence=0.65,
            horizon_days=5,
            rationale="Bearish signal",
        ),
        TradeTarget(
            symbol="ES=F",
            action=TradeAction.HOLD,
            entry_price=5400.0,
            target_price=5400.0,
            stop_loss=5400.0,
            risk_reward_ratio=0.0,
            confidence=0.92,
            horizon_days=5,
            rationale="Neutral signal",
        ),
    ]


class TestReportGenerator:
    def test_table_format(self, sample_targets: list[TradeTarget]) -> None:
        gen = ReportGenerator()
        output = gen.generate_report(sample_targets, fmt="table")
        assert "AAPL" in output
        assert "BTC/USDT" in output
        assert "ES=F" in output
        assert "BUY" in output
        assert "SELL" in output
        assert "HOLD" in output

    def test_json_format(self, sample_targets: list[TradeTarget]) -> None:
        gen = ReportGenerator()
        output = gen.generate_report(sample_targets, fmt="json")
        data = json.loads(output)
        assert len(data) == 3
        assert data[0]["symbol"] == "AAPL"
        assert data[0]["action"] == "BUY"
        assert data[0]["entry_price"] == 180.50
        assert data[1]["symbol"] == "BTC/USDT"
        assert data[1]["action"] == "SELL"
        assert data[2]["symbol"] == "ES=F"
        assert data[2]["action"] == "HOLD"

    def test_csv_format(self, sample_targets: list[TradeTarget]) -> None:
        gen = ReportGenerator()
        output = gen.generate_report(sample_targets, fmt="csv")
        lines = output.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "symbol" in lines[0]
        assert "AAPL" in lines[1]

    def test_unknown_format(self, sample_targets: list[TradeTarget]) -> None:
        gen = ReportGenerator()
        with pytest.raises(ValueError, match="Unknown report format"):
            gen.generate_report(sample_targets, fmt="xml")

    def test_empty_targets(self) -> None:
        gen = ReportGenerator()
        output = gen.generate_report([], fmt="json")
        assert json.loads(output) == []

    def test_json_fields_complete(self, sample_targets: list[TradeTarget]) -> None:
        gen = ReportGenerator()
        output = gen.generate_report(sample_targets, fmt="json")
        data = json.loads(output)
        expected_keys = {
            "symbol", "action", "entry_price", "target_price",
            "stop_loss", "risk_reward_ratio", "confidence",
            "horizon_days", "rationale",
        }
        for row in data:
            assert set(row.keys()) == expected_keys
