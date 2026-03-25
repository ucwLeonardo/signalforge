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
class QlibConfig:
    enabled: bool = False
    model_type: str = "lgbm"
    features: str = "alpha158"
    label_horizon: int = 5
    device: str = "cuda"
    qlib_data_dir: str = "~/.qlib/qlib_data/us_data"
    region: str = "us"


@dataclass(frozen=True)
class ChronosConfig:
    enabled: bool = False
    model: str = "amazon/chronos-bolt-base"
    pred_len: int = 5
    num_samples: int = 20
    device: str = "cuda"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class AgentsConfig:
    enabled: bool = False
    llm_provider: str = "anthropic"
    deep_think_model: str = "claude-sonnet-4-6"
    quick_think_model: str = "claude-haiku-4-5-20251001"
    max_debate_rounds: int = 1
    max_risk_rounds: int = 1


@dataclass(frozen=True)
class EnsembleConfig:
    kronos_weight: float = 0.35
    qlib_weight: float = 0.20
    chronos_weight: float = 0.15
    agents_weight: float = 0.10
    technical_weight: float = 0.20


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
    options_provider: str = "yfinance"
    options_interval: str = "1d"
    options_lookback_days: int = 365
    polygon_api_key: str = ""


@dataclass(frozen=True)
class Config:
    data_dir: str = ""
    cache_dir: str = ""
    results_dir: str = ""
    us_stocks: list[str] = field(default_factory=list)
    crypto: list[str] = field(default_factory=list)
    futures: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    kronos: KronosConfig = field(default_factory=KronosConfig)
    qlib: QlibConfig = field(default_factory=QlibConfig)
    chronos: ChronosConfig = field(default_factory=ChronosConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
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
            options=[],
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

    qlib_raw = engines.get("qlib", {})
    qlib_cfg = QlibConfig(
        enabled=qlib_raw.get("enabled", False),
        model_type=qlib_raw.get("model_type", "lgbm"),
        features=qlib_raw.get("features", "alpha158"),
        label_horizon=qlib_raw.get("label_horizon", 5),
        device=qlib_raw.get("device", "cuda"),
        qlib_data_dir=qlib_raw.get("qlib_data_dir", "~/.qlib/qlib_data/us_data"),
        region=qlib_raw.get("region", "us"),
    )

    chronos_raw = engines.get("chronos", {})
    chronos_cfg = ChronosConfig(
        enabled=chronos_raw.get("enabled", False),
        model=chronos_raw.get("model", "amazon/chronos-bolt-base"),
        pred_len=chronos_raw.get("pred_len", 5),
        num_samples=chronos_raw.get("num_samples", 20),
        device=chronos_raw.get("device", "cuda"),
    )

    agents_raw = engines.get("trading_agents", {})
    agents_cfg = AgentsConfig(
        enabled=agents_raw.get("enabled", False),
        llm_provider=agents_raw.get("llm_provider", "anthropic"),
        deep_think_model=agents_raw.get("deep_think_model", "claude-sonnet-4-6"),
        quick_think_model=agents_raw.get("quick_think_model", "claude-haiku-4-5-20251001"),
        max_debate_rounds=agents_raw.get("max_debate_rounds", 1),
        max_risk_rounds=agents_raw.get("max_risk_discuss_rounds", 1),
    )

    stocks_data = data_raw.get("stocks", {})
    crypto_data = data_raw.get("crypto", {})
    futures_data = data_raw.get("futures", {})

    options_data = data_raw.get("options", {})

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
        options_provider=options_data.get("provider", "yfinance"),
        options_interval=options_data.get("interval", "1d"),
        options_lookback_days=options_data.get("lookback_days", 365),
        polygon_api_key=options_data.get("polygon_api_key", ""),
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
        options=assets.get("options", []),
        data=data,
        kronos=kronos,
        qlib=qlib_cfg,
        chronos=chronos_cfg,
        agents=agents_cfg,
        ensemble=ensemble,
        output_format=output_raw.get("format", "table"),
        confidence_threshold=output_raw.get("confidence_threshold", 0.3),
    )
