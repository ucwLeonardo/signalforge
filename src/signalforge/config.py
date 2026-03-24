"""Configuration loader for SignalForge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:default} in strings."""
    if isinstance(obj, str):
        import re

        def _replace(match: re.Match[str]) -> str:
            var = match.group(1)
            if ":" in var:
                name, default = var.split(":", 1)
                return os.environ.get(name, default)
            return os.environ.get(var, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", _replace, obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


@dataclass(frozen=True)
class KronosConfig:
    enabled: bool = True
    model: str = "NeoQuasar/Kronos-base"
    tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    pred_len: int = 5
    lookback: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    sample_count: int = 20
    device: str = "cuda"


@dataclass(frozen=True)
class EnsembleConfig:
    kronos_weight: float = 0.40
    qlib_weight: float = 0.20
    chronos_weight: float = 0.15
    agents_weight: float = 0.10
    technical_weight: float = 0.15


@dataclass(frozen=True)
class DataConfig:
    stocks_provider: str = "yfinance"
    stocks_interval: str = "1d"
    stocks_lookback_days: int = 730
    crypto_provider: str = "ccxt"
    crypto_exchange: str = "binance"
    crypto_interval: str = "1d"
    crypto_lookback_days: int = 730
    futures_provider: str = "yfinance"
    futures_interval: str = "1d"
    futures_lookback_days: int = 365


@dataclass(frozen=True)
class Config:
    data_dir: str = ""
    cache_dir: str = ""
    results_dir: str = ""
    us_stocks: list[str] = field(default_factory=list)
    crypto: list[str] = field(default_factory=list)
    futures: list[str] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    kronos: KronosConfig = field(default_factory=KronosConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    output_format: str = "table"
    confidence_threshold: float = 0.3


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file with env var expansion."""
    if config_path is None:
        # Look for config in standard locations
        candidates = [
            Path.cwd() / "config" / "default.yaml",
            Path(__file__).parent.parent.parent.parent / "config" / "default.yaml",
            Path.home() / ".signalforge" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not Path(config_path).exists():
        logger.warning("No config file found, using defaults")
        home = str(Path.home())
        return Config(
            data_dir=f"{home}/.signalforge/data",
            cache_dir=f"{home}/.signalforge/cache",
            results_dir=f"{home}/.signalforge/results",
            us_stocks=["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"],
            crypto=["BTC/USDT", "ETH/USDT"],
            futures=["ES=F", "NQ=F", "GC=F"],
        )

    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    raw = _expand_env_vars(raw)

    project = raw.get("project", {})
    assets = raw.get("assets", {})
    data_raw = raw.get("data", {})
    engines = raw.get("engines", {})
    ensemble_raw = raw.get("ensemble", {})
    output_raw = raw.get("output", {})

    kronos_raw = engines.get("kronos", {})
    kronos = KronosConfig(
        enabled=kronos_raw.get("enabled", True),
        model=kronos_raw.get("model", "NeoQuasar/Kronos-base"),
        tokenizer=kronos_raw.get("tokenizer", "NeoQuasar/Kronos-Tokenizer-base"),
        pred_len=kronos_raw.get("pred_len", 5),
        lookback=kronos_raw.get("lookback", 512),
        temperature=kronos_raw.get("temperature", 0.8),
        top_p=kronos_raw.get("top_p", 0.9),
        sample_count=kronos_raw.get("sample_count", 20),
        device=kronos_raw.get("device", "cuda"),
    )

    stocks_data = data_raw.get("stocks", {})
    crypto_data = data_raw.get("crypto", {})
    futures_data = data_raw.get("futures", {})

    data = DataConfig(
        stocks_provider=stocks_data.get("provider", "yfinance"),
        stocks_interval=stocks_data.get("interval", "1d"),
        stocks_lookback_days=stocks_data.get("lookback_days", 730),
        crypto_provider=crypto_data.get("provider", "ccxt"),
        crypto_exchange=crypto_data.get("exchange", "binance"),
        crypto_interval=crypto_data.get("interval", "1d"),
        crypto_lookback_days=crypto_data.get("lookback_days", 730),
        futures_provider=futures_data.get("provider", "yfinance"),
        futures_interval=futures_data.get("interval", "1d"),
        futures_lookback_days=futures_data.get("lookback_days", 365),
    )

    ensemble = EnsembleConfig(
        kronos_weight=ensemble_raw.get("kronos_weight", 0.40),
        qlib_weight=ensemble_raw.get("qlib_weight", 0.20),
        chronos_weight=ensemble_raw.get("chronos_weight", 0.15),
        agents_weight=ensemble_raw.get("agents_weight", 0.10),
        technical_weight=ensemble_raw.get("technical_weight", 0.15),
    )

    home = str(Path.home())
    return Config(
        data_dir=project.get("data_dir", f"{home}/.signalforge/data"),
        cache_dir=project.get("cache_dir", f"{home}/.signalforge/cache"),
        results_dir=project.get("results_dir", f"{home}/.signalforge/results"),
        us_stocks=assets.get("us_stocks", []),
        crypto=assets.get("crypto", []),
        futures=assets.get("futures", []),
        data=data,
        kronos=kronos,
        ensemble=ensemble,
        output_format=output_raw.get("format", "table"),
        confidence_threshold=output_raw.get("confidence_threshold", 0.3),
    )
