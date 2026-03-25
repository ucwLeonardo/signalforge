"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from signalforge.config import load_config


class TestConfig:
    def test_load_default_config(self) -> None:
        cfg = load_config()
        assert len(cfg.us_stocks) > 0
        assert "AAPL" in cfg.us_stocks
        assert len(cfg.crypto) > 0
        assert "BTC/USDT" in cfg.crypto
        assert len(cfg.futures) > 0
        assert "ES=F" in cfg.futures

    def test_kronos_config(self) -> None:
        cfg = load_config()
        assert cfg.kronos.enabled is True
        assert cfg.kronos.pred_len > 0

    def test_ensemble_weights(self) -> None:
        cfg = load_config()
        total = (
            cfg.ensemble.kronos_weight
            + cfg.ensemble.qlib_weight
            + cfg.ensemble.chronos_weight
            + cfg.ensemble.agents_weight
            + cfg.ensemble.technical_weight
        )
        # Weights should sum to approximately 1.0
        assert 0.9 <= total <= 1.1

    def test_qlib_enabled(self) -> None:
        cfg = load_config()
        assert cfg.qlib.enabled is True

    def test_chronos_enabled(self) -> None:
        cfg = load_config()
        assert cfg.chronos.enabled is True

    def test_agents_enabled(self) -> None:
        cfg = load_config()
        assert cfg.agents.enabled is True

    def test_options_list(self) -> None:
        cfg = load_config()
        assert isinstance(cfg.options, list)

    def test_expanded_crypto_list(self) -> None:
        cfg = load_config()
        assert len(cfg.crypto) >= 20
        assert "PEPE/USDT" in cfg.crypto
        assert "FET/USDT" in cfg.crypto

    def test_options_data_config(self) -> None:
        cfg = load_config()
        assert cfg.data.options_provider == "yfinance"
        assert cfg.data.options_lookback_days == 365
