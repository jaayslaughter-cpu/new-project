"""
Data contracts for the options bot pipeline.

Defines the exact schema at every module boundary:
  OptionChainRow      — raw output from market data ingestion
  EnrichedOptionRow   — after Greeks layer adds IV, delta, gamma, theta, vega
  ApprovedOrder       — after risk manager validates and sizes the trade
  FilledOrder         — after broker confirms execution

Zero-hallucination policy: all Optional fields default to None.
Never fill a None with an estimate — raise DataValidationError instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional


OptionType = Literal["call", "put"]
OrderSide = Literal["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"]


@dataclass
class OptionChainRow:
    """
    Raw contract data as returned by the market data layer.
    No Greeks computed yet — those come from the enrichment layer.

    Fields marked Optional are None if the data source did not provide them.
    The pipeline must never estimate these — raise DataValidationError if a
    required downstream field is None.
    """
    # --- Identity ---
    symbol: str                        # OCC contract symbol e.g. "SPY260620C00580000"
    underlying: str                    # e.g. "SPY"
    option_type: OptionType
    strike: float
    expiry: date
    dte: int                           # days to expiration (required, reject if 0)

    # --- Market quote ---
    bid: Optional[float]               # None if stale / market closed
    ask: Optional[float]
    last_price: Optional[float]
    mid_price: Optional[float]         # (bid+ask)/2, None if either is None
    volume: Optional[int]
    open_interest: Optional[int]

    # --- Underlying ---
    underlying_price: float            # required — cannot be None

    # --- Staleness tracking ---
    data_timestamp: datetime           # when this quote was fetched
    source: str = "yfinance"           # data source identifier

    # --- Derived quality flags ---
    spread_pct: Optional[float] = None  # (ask-bid)/mid, computed on ingestion
    in_the_money: bool = False

    def __post_init__(self):
        """Compute derived fields after construction."""
        if self.bid is not None and self.ask is not None and self.bid + self.ask > 0:
            self.mid_price = (self.bid + self.ask) / 2.0
            self.spread_pct = (self.ask - self.bid) / self.mid_price if self.mid_price > 0 else None
        if self.option_type == "call":
            self.in_the_money = self.underlying_price > self.strike
        else:
            self.in_the_money = self.underlying_price < self.strike


@dataclass
class EnrichedOptionRow:
    """
    Contract data after the Greeks enrichment layer.
    Inherits all OptionChainRow fields plus computed Greeks and IV.

    iv=None means the IV solve failed — do NOT use this contract for
    strategies that require IV. The pipeline raises IVSolveError upstream.
    """
    # --- All OptionChainRow fields ---
    raw: OptionChainRow

    # --- Greeks (None = failed to compute) ---
    iv: Optional[float] = None         # implied volatility as decimal (0.25 = 25%)
    delta: Optional[float] = None      # range [-1, 1]
    gamma: Optional[float] = None      # always >= 0
    theta: Optional[float] = None      # daily decay, negative for long positions
    vega: Optional[float] = None       # per 1% IV move
    rho: Optional[float] = None        # per 1% rate move
    vanna: Optional[float] = None       # dDelta/dSigma per 1% vol
    volga: Optional[float] = None       # dVega/dSigma per (1% vol)^2
    charm: Optional[float] = None       # dDelta/dt per day (0DTE decay)

    # --- Pricing model used ---
    pricing_model: str = "black_scholes"   # or "binomial_crr" for American options
    risk_free_rate: Optional[float] = None  # rate used in pricing

    @property
    def symbol(self) -> str:
        return self.raw.symbol

    @property
    def underlying(self) -> str:
        return self.raw.underlying

    @property
    def option_type(self) -> OptionType:
        return self.raw.option_type

    @property
    def strike(self) -> float:
        return self.raw.strike

    @property
    def expiry(self) -> date:
        return self.raw.expiry

    @property
    def dte(self) -> int:
        return self.raw.dte

    @property
    def bid(self) -> Optional[float]:
        return self.raw.bid

    @property
    def ask(self) -> Optional[float]:
        return self.raw.ask

    @property
    def mid_price(self) -> Optional[float]:
        return self.raw.mid_price

    @property
    def open_interest(self) -> Optional[int]:
        return self.raw.open_interest

    @property
    def spread_pct(self) -> Optional[float]:
        return self.raw.spread_pct

    @property
    def underlying_price(self) -> float:
        return self.raw.underlying_price


@dataclass
class OrderLeg:
    """One leg of a single or multi-leg options order."""
    symbol: str
    option_type: OptionType
    strike: float
    expiry: date
    side: OrderSide
    quantity: int                      # number of contracts


@dataclass
class ApprovedOrder:
    """
    Order approved by the risk manager, ready for broker submission.

    hard_stop_price is MANDATORY. The execution guard raises RiskVetoError
    if this is None. No naked positions — ever.
    """
    legs: list[OrderLeg]

    # --- Pricing ---
    net_debit_credit: float            # positive = debit (we pay), negative = credit (we receive)
    estimated_fill_price: float        # mid-price or limit price

    # --- Risk parameters (all required) ---
    hard_stop_price: float             # MANDATORY — execution guard rejects None
    max_loss_dollars: float            # hard-capped by risk manager
    position_size_contracts: int       # floor(equity * risk_pct / max_loss)
    risk_approved: bool                # must be True before execution layer touches it

    # --- Optional profit target ---
    profit_target_price: Optional[float] = None
    profit_target_pct: Optional[float] = None  # e.g. 0.50 = close at 50% of max profit

    # --- Slippage budget ---
    slippage_budget_pct: float = 0.02  # 2% of mid — passed to broker as limit tolerance

    # --- Metadata ---
    strategy_name: str = ""
    underlying: str = ""
    signal_timestamp: Optional[datetime] = None
    notes: str = ""


@dataclass
class FilledOrder:
    """
    Confirmed execution returned by the broker adapter.
    Stored in the database as the position record.
    """
    order_id: str                      # broker-assigned order ID
    approved_order: ApprovedOrder

    # --- Actual fill details ---
    fill_price: float
    fill_timestamp: datetime
    slippage_actual: float             # fill_price - estimated_fill_price

    # --- Position state ---
    status: Literal["open", "closed", "partial"] = "open"
    close_price: Optional[float] = None
    close_timestamp: Optional[datetime] = None
    realized_pnl: Optional[float] = None

    # --- Broker metadata ---
    broker: str = "alpaca"
    account_id: str = ""
