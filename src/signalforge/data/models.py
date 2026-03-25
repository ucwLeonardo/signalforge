"""Shared domain models used across SignalForge modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AssetType(str, Enum):
    """Supported asset types."""

    STOCK = "stock"
    CRYPTO = "crypto"
    FUTURES = "futures"
    OPTIONS = "options"


class Action(str, Enum):
    """Signal action types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeAction(str, Enum):
    """Possible trade actions (legacy, uppercase variant)."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Bar:
    """Single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float | None = None


@dataclass(frozen=True)
class Asset:
    """Tradeable asset identifier."""

    symbol: str
    asset_type: AssetType
    exchange: str | None = None


@dataclass(frozen=True)
class Signal:
    """Trading signal produced by an engine or ensemble."""

    asset: Asset
    timestamp: datetime
    action: Action
    entry_price: float
    exit_price: float | None = None
    stop_loss: float | None = None
    confidence: float = 0.0
    rationale: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


# ---------------------------------------------------------------------------
# Ensemble / target models (used by combiner & output layers)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CombinedSignal:
    """Unified signal produced by the ensemble combiner.

    Attributes:
        direction: Ranges from -1 (strong sell) to +1 (strong buy).
        confidence: Ranges from 0 (no confidence) to 1 (full agreement).
        predicted_high: Ensemble-predicted high price (or None).
        predicted_low: Ensemble-predicted low price (or None).
        predicted_close: Ensemble-predicted close price (or None).
    """

    direction: float
    confidence: float
    predicted_high: float | None = None
    predicted_low: float | None = None
    predicted_close: float | None = None


@dataclass(frozen=True)
class SupportResistance:
    """Support and resistance price levels for a symbol.

    Attributes:
        support: Nearest support price level.
        resistance: Nearest resistance price level.
    """

    support: float
    resistance: float


@dataclass(frozen=True)
class TradeTarget:
    """Actionable trade target derived from a combined signal.

    Attributes:
        symbol: Ticker / asset symbol.
        action: BUY, SELL, or HOLD.
        entry_price: Recommended entry price.
        target_price: Recommended profit target price.
        stop_loss: Recommended stop-loss price.
        risk_reward_ratio: Reward-to-risk ratio (target delta / stop delta).
        confidence: Signal confidence from 0 to 1.
        horizon_days: Forecast horizon in calendar days.
        rationale: Human-readable reasoning for the recommendation.
    """

    symbol: str
    action: TradeAction
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    confidence: float
    horizon_days: int
    rationale: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptionContract:
    """Parsed option contract details.

    Attributes:
        underlying: Underlying stock ticker (e.g. ``AAPL``).
        expiration: Expiration date string (e.g. ``2026-06-19``).
        strike: Strike price.
        option_type: ``"C"`` for call, ``"P"`` for put.
        occ_symbol: OCC-format symbol (e.g. ``AAPL260619C00200000``).
    """

    underlying: str
    expiration: str
    strike: float
    option_type: str  # "C" or "P"
    occ_symbol: str = ""


def parse_option_symbol(symbol: str) -> OptionContract | None:
    """Parse an option symbol in human or OCC format.

    Supported formats:
      - Human: ``"AAPL 2026-06-19 200 C"``
      - OCC:   ``"AAPL260619C00200000"``

    Returns None if the symbol is not a recognised option format.
    """
    import re

    # Human format: "AAPL 2026-06-19 200 C" or "AAPL 2026-06-19 200.5 P"
    human = re.match(
        r"^([A-Z]+)\s+(\d{4}-\d{2}-\d{2})\s+([\d.]+)\s+([CP])$",
        symbol.strip().upper(),
    )
    if human:
        underlying = human.group(1)
        expiration = human.group(2)
        strike = float(human.group(3))
        opt_type = human.group(4)
        # Build OCC symbol: AAPL260619C00200000
        exp_compact = expiration.replace("-", "")[2:]  # 260619
        strike_occ = f"{int(strike * 1000):08d}"
        occ = f"{underlying}{exp_compact}{opt_type}{strike_occ}"
        return OptionContract(
            underlying=underlying,
            expiration=expiration,
            strike=strike,
            option_type=opt_type,
            occ_symbol=occ,
        )

    # OCC format: AAPL260619C00200000 (ticker + YYMMDD + C/P + 8-digit strike*1000)
    occ = re.match(
        r"^([A-Z]+)(\d{6})([CP])(\d{8})$",
        symbol.strip().upper(),
    )
    if occ:
        underlying = occ.group(1)
        exp_raw = occ.group(2)  # YYMMDD
        opt_type = occ.group(3)
        strike = int(occ.group(4)) / 1000.0
        expiration = f"20{exp_raw[:2]}-{exp_raw[2:4]}-{exp_raw[4:6]}"
        return OptionContract(
            underlying=underlying,
            expiration=expiration,
            strike=strike,
            option_type=opt_type,
            occ_symbol=symbol.strip().upper(),
        )

    return None


def classify_symbol(symbol: str) -> AssetType:
    """Infer asset type from a symbol string.

    - Matches option format (human or OCC) -> options
    - Contains ``/`` -> crypto  (e.g. ``BTC/USDT``)
    - Ends with ``=F`` -> futures  (e.g. ``ES=F``)
    - Otherwise -> stock
    """
    if parse_option_symbol(symbol) is not None:
        return AssetType.OPTIONS
    if "/" in symbol:
        return AssetType.CRYPTO
    if symbol.endswith("=F"):
        return AssetType.FUTURES
    return AssetType.STOCK


def asset_from_symbol(symbol: str) -> Asset:
    """Build an :class:`Asset` from a plain symbol string."""
    return Asset(symbol=symbol, asset_type=classify_symbol(symbol))
