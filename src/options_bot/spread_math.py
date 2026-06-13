"""
Spread Math — vertical options spread pricing and P&L calculator.

Computes max loss, max profit, mid price, natural price, and break-even
for all four vertical spread types at both entry and exit.

Design principles:
  - Zero external dependencies (stdlib only)
  - No dataclass inheritance chains — pure functions returning plain dicts
  - All math documented inline before the code that implements it
  - Works standalone; imported by strategy.py and risk.py

Supported spread types:
  - Bull call spread  (debit): buy lower call, sell higher call
  - Bear put spread   (debit): buy higher put, sell lower put
  - Bull put spread   (credit): sell higher put, buy lower put  ← our primary strategy
  - Bear call spread  (credit): sell lower call, buy higher call

Mathematical basis
------------------
For a vertical spread with two legs (short + long):

  mid_low  = (low_bid  + low_ask)  / 2
  mid_high = (high_bid + high_ask) / 2

  For a CREDIT spread (short higher, long lower — bull put spread):
    net_credit  = mid_high - mid_low          (premium received)
    spread_width = high_strike - low_strike
    max_profit   = net_credit * 100 * contracts
    max_loss     = (spread_width - net_credit) * 100 * contracts
    break_even   = high_strike - net_credit   (for put spreads)

  For a DEBIT spread (long lower, short higher — bull call spread):
    net_debit    = mid_high - mid_low          (premium paid)
    spread_width = high_strike - low_strike
    max_loss     = net_debit * 100 * contracts
    max_profit   = (spread_width - net_debit) * 100 * contracts
    break_even   = low_strike + net_debit      (for call spreads)

  At EXIT:
    cost_to_close  = current mid price of the spread
    realized_pnl   = (entry_credit - cost_to_close) * 100 * contracts  [credit spread]
                   = (cost_to_close - entry_debit) * 100 * contracts    [debit spread]

All prices in dollars per share (option premiums are per-share; multiply by 100 for per-contract).
"""

from __future__ import annotations

import math
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SpreadType = Literal[
    "bull_call",   # debit spread: buy lower call, sell higher call
    "bear_put",    # debit spread: buy higher put, sell lower put
    "bull_put",    # credit spread: sell higher put, buy lower put
    "bear_call",   # credit spread: sell lower call, buy higher call
]

TradeAction = Literal["entry", "exit"]


# ---------------------------------------------------------------------------
# Core calculator
# ---------------------------------------------------------------------------

def calc_spread(
    spread_type: SpreadType,
    action: TradeAction,
    low_strike: float,
    low_bid: float,
    low_ask: float,
    high_strike: float,
    high_bid: float,
    high_ask: float,
    num_contracts: int = 1,
    underlying_price: float | None = None,
) -> dict:
    """
    Calculate pricing for a vertical spread at entry or exit.

    Parameters
    ----------
    spread_type : SpreadType
        One of: "bull_call", "bear_put", "bull_put", "bear_call"
    action : TradeAction
        "entry" (opening the position) or "exit" (closing it)
    low_strike : float
        Strike price of the lower leg
    low_bid, low_ask : float
        Bid and ask of the lower strike option
    high_strike : float
        Strike price of the upper leg
    high_bid, high_ask : float
        Bid and ask of the upper strike option
    num_contracts : int
        Number of contracts (default 1)
    underlying_price : float or None
        Current underlying price; used for distance-from-money metadata

    Returns
    -------
    dict with keys:
        spread_type, action, num_contracts,
        low_strike, high_strike, spread_width,
        low_mid, high_mid,
        net_price,           — net credit (positive) or debit (positive)
        is_credit,           — True for bull_put and bear_call
        max_profit,          — dollars, total for num_contracts
        max_loss,            — dollars, total for num_contracts (always positive)
        break_even,          — price level at break-even
        cost_basis,          — total cash flow (negative = paid, positive = received)
        spread_id,           — unique string identifier
        low_distance,        — low_strike distance from underlying (if provided)
        high_distance        — high_strike distance from underlying (if provided)
    """
    # --- Input validation ---
    if low_strike >= high_strike:
        raise ValueError(
            f"low_strike ({low_strike}) must be less than high_strike ({high_strike})"
        )
    if num_contracts < 1:
        raise ValueError(f"num_contracts must be >= 1, got {num_contracts}")
    for name, val in [
        ("low_bid", low_bid), ("low_ask", low_ask),
        ("high_bid", high_bid), ("high_ask", high_ask),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")

    # --- Mid prices ---
    # Formula: mid = (bid + ask) / 2
    low_mid  = (low_bid  + low_ask)  / 2.0
    high_mid = (high_bid + high_ask) / 2.0

    spread_width = high_strike - low_strike

    # --- Determine spread economics ---
    # Credit spreads: bull_put (short higher put, long lower put)
    #                 bear_call (short lower call, long higher call)
    # Debit spreads:  bull_call (long lower call, short higher call)
    #                 bear_put  (long higher put, short lower put)
    is_credit = spread_type in ("bull_put", "bear_call")

    if is_credit:
        # Net credit = premium of short leg - cost of long leg
        # bull_put:   short = high strike put  → high_mid
        #             long  = low strike put   → low_mid
        # bear_call:  short = low strike call  → low_mid
        #             long  = high strike call → high_mid
        if spread_type == "bull_put":
            net_price = high_mid - low_mid     # positive = credit received
        else:  # bear_call
            net_price = low_mid - high_mid     # short lower call
    else:
        # Net debit = cost of long leg - premium of short leg
        # bull_call:  long = low strike call   → low_mid
        #             short= high strike call  → high_mid
        # bear_put:   long = high strike put   → high_mid
        #             short= low strike put    → low_mid
        if spread_type == "bull_call":
            net_price = high_mid - low_mid     # positive = debit paid
        else:  # bear_put
            net_price = high_mid - low_mid     # positive = debit paid

    # --- Max profit and max loss ---
    # Credit spread:
    #   max_profit = net_credit * 100 * contracts       (keep the premium)
    #   max_loss   = (spread_width - net_credit) * 100  (spread goes against us)
    # Debit spread:
    #   max_loss   = net_debit * 100 * contracts         (paid out)
    #   max_profit = (spread_width - net_debit) * 100    (spread moves our way fully)
    if is_credit:
        max_profit_per_contract = net_price
        max_loss_per_contract   = spread_width - net_price
    else:
        max_loss_per_contract   = net_price
        max_profit_per_contract = spread_width - net_price

    max_profit = round(max_profit_per_contract * 100 * num_contracts, 2)
    max_loss   = round(max_loss_per_contract   * 100 * num_contracts, 2)

    # Ensure max_loss is always positive (it represents a loss amount)
    max_loss = abs(max_loss)

    # --- Break-even ---
    # bull_put:  break_even = high_strike - net_credit   (underlying must stay above this)
    # bear_call: break_even = low_strike  + net_credit   (underlying must stay below this)
    # bull_call: break_even = low_strike  + net_debit    (underlying must rise above this)
    # bear_put:  break_even = high_strike - net_debit    (underlying must fall below this)
    if spread_type == "bull_put":
        break_even = high_strike - net_price
    elif spread_type == "bear_call":
        break_even = low_strike + net_price
    elif spread_type == "bull_call":
        break_even = low_strike + net_price
    else:  # bear_put
        break_even = high_strike - net_price

    # --- Cost basis (cash flow) ---
    # Positive = received cash (credit); negative = paid cash (debit)
    cost_basis = round(net_price * 100 * num_contracts * (1 if is_credit else -1), 2)

    # --- Distance metadata ---
    low_distance  = None
    high_distance = None
    if underlying_price is not None:
        low_distance  = round(underlying_price - low_strike,  2)
        high_distance = round(high_strike - underlying_price, 2)

    # --- Spread ID for logging ---
    spread_id = (
        f"{spread_type}_{action}"
        f"_low{low_strike:.0f}_high{high_strike:.0f}"
        f"_w{spread_width:.0f}"
        f"_c{num_contracts}"
    )

    return {
        "spread_type":    spread_type,
        "action":         action,
        "num_contracts":  num_contracts,
        "low_strike":     low_strike,
        "high_strike":    high_strike,
        "spread_width":   round(spread_width, 2),
        "low_mid":        round(low_mid,  4),
        "high_mid":       round(high_mid, 4),
        "net_price":      round(net_price, 4),
        "is_credit":      is_credit,
        "max_profit":     max_profit,
        "max_loss":       max_loss,
        "break_even":     round(break_even, 2),
        "cost_basis":     cost_basis,
        "spread_id":      spread_id,
        "low_distance":   low_distance,
        "high_distance":  high_distance,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers — match the strategy layer's call sites
# ---------------------------------------------------------------------------

def bull_put_entry(
    low_strike: float,
    low_bid: float,
    low_ask: float,
    high_strike: float,
    high_bid: float,
    high_ask: float,
    num_contracts: int = 1,
    underlying_price: float | None = None,
) -> dict:
    """
    Entry pricing for a bull put spread (our primary credit spread).

    Short the higher put, long the lower put.
    Net credit = high_mid - low_mid
    Max loss = (spread_width - credit) * 100 * contracts
    Break-even = high_strike - credit
    """
    return calc_spread(
        spread_type="bull_put",
        action="entry",
        low_strike=low_strike,
        low_bid=low_bid,
        low_ask=low_ask,
        high_strike=high_strike,
        high_bid=high_bid,
        high_ask=high_ask,
        num_contracts=num_contracts,
        underlying_price=underlying_price,
    )


def bull_put_exit(
    low_strike: float,
    low_bid: float,
    low_ask: float,
    high_strike: float,
    high_bid: float,
    high_ask: float,
    num_contracts: int = 1,
    entry_credit: float | None = None,
) -> dict:
    """
    Exit pricing for a bull put spread (closing/buying back the spread).

    At exit:
      cost_to_close = current spread mid price
      realized_pnl  = (entry_credit - cost_to_close) * 100 * contracts
                      (positive = profit, negative = loss)
    """
    result = calc_spread(
        spread_type="bull_put",
        action="exit",
        low_strike=low_strike,
        low_bid=low_bid,
        low_ask=low_ask,
        high_strike=high_strike,
        high_bid=high_bid,
        high_ask=high_ask,
        num_contracts=num_contracts,
    )

    if entry_credit is not None:
        cost_to_close = result["net_price"]
        realized_pnl = round(
            (entry_credit - cost_to_close) * 100 * num_contracts, 2
        )
        result["entry_credit"]  = round(entry_credit, 4)
        result["cost_to_close"] = round(cost_to_close, 4)
        result["realized_pnl"]  = realized_pnl

    return result


def strangle_entry(
    put_strike: float,
    put_bid: float,
    put_ask: float,
    call_strike: float,
    call_bid: float,
    call_ask: float,
    num_contracts: int = 1,
    underlying_price: float | None = None,
) -> dict:
    """
    Pricing for a short strangle entry (sell OTM put + sell OTM call).

    For strangles, both legs are short so this is not a traditional spread.
    Returns combined credit and profit zone metrics.

    Net credit   = put_mid + call_mid
    Profit zone  = [put_strike, call_strike]
    Upper BE     = call_strike + net_credit
    Lower BE     = put_strike  - net_credit
    """
    put_mid  = (put_bid  + put_ask)  / 2.0
    call_mid = (call_bid + call_ask) / 2.0
    net_credit = put_mid + call_mid

    upper_be = call_strike + net_credit
    lower_be = put_strike  - net_credit

    profit_zone_width = call_strike - put_strike

    low_distance  = None
    high_distance = None
    if underlying_price is not None:
        low_distance  = round(underlying_price - put_strike,  2)
        high_distance = round(call_strike - underlying_price, 2)

    # For risk sizing, we use 3x credit as practical max loss estimate
    # (actual max loss is theoretically large; the hard stop enforces this)
    practical_max_loss = round(net_credit * 3.0 * 100 * num_contracts, 2)
    cost_basis         = round(net_credit * 100 * num_contracts, 2)

    return {
        "spread_type":         "strangle",
        "action":              "entry",
        "num_contracts":       num_contracts,
        "put_strike":          put_strike,
        "call_strike":         call_strike,
        "put_mid":             round(put_mid, 4),
        "call_mid":            round(call_mid, 4),
        "net_credit":          round(net_credit, 4),
        "is_credit":           True,
        "profit_zone_width":   round(profit_zone_width, 2),
        "upper_break_even":    round(upper_be, 2),
        "lower_break_even":    round(lower_be, 2),
        "practical_max_loss":  practical_max_loss,
        "cost_basis":          cost_basis,
        "low_distance":        low_distance,
        "high_distance":       high_distance,
        "spread_id": (
            f"strangle_entry"
            f"_p{put_strike:.0f}_c{call_strike:.0f}"
            f"_c{num_contracts}"
        ),
    }


def profit_target_price(entry_credit: float, target_pct: float = 0.50) -> float:
    """
    Calculate the spread mid-price at which to take profit.

    For a credit spread: close when the spread can be bought back at
    (1 - target_pct) * entry_credit.

    Parameters
    ----------
    entry_credit : float
        Premium received when opening the position (per share)
    target_pct : float
        Fraction of max profit to capture before closing (default 0.50 = 50%)

    Returns
    -------
    float — the spread mid-price trigger for the closing order
    """
    if not 0 < target_pct < 1:
        raise ValueError(f"target_pct must be in (0, 1), got {target_pct}")
    return round(entry_credit * (1.0 - target_pct), 4)


def stop_price(entry_credit: float, stop_multiplier: float = 2.0) -> float:
    """
    Calculate the spread mid-price at which to trigger the stop-loss.

    For a credit spread: close when the spread costs more than
    stop_multiplier * entry_credit to buy back.

    Parameters
    ----------
    entry_credit : float
        Premium received when opening the position (per share)
    stop_multiplier : float
        Multiple of original credit at which to stop out (default 2.0)

    Returns
    -------
    float — the spread mid-price trigger for the stop order
    """
    if stop_multiplier <= 1.0:
        raise ValueError(f"stop_multiplier must be > 1.0, got {stop_multiplier}")
    return round(entry_credit * stop_multiplier, 4)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_spread_inputs(
    low_bid: float,
    low_ask: float,
    high_bid: float,
    high_ask: float,
    low_strike: float,
    high_strike: float,
) -> list[str]:
    """
    Return a list of validation error messages (empty if all inputs are valid).
    Use before calling calc_spread to give clear error messages upstream.
    """
    errors = []
    if low_strike >= high_strike:
        errors.append(f"low_strike ({low_strike}) >= high_strike ({high_strike})")
    if low_bid > low_ask:
        errors.append(f"low_bid ({low_bid}) > low_ask ({low_ask})")
    if high_bid > high_ask:
        errors.append(f"high_bid ({high_bid}) > high_ask ({high_ask})")
    if low_bid < 0 or low_ask < 0 or high_bid < 0 or high_ask < 0:
        errors.append("bid/ask prices must be non-negative")
    if not math.isfinite(low_bid + low_ask + high_bid + high_ask):
        errors.append("bid/ask prices contain NaN or Inf")
    return errors
