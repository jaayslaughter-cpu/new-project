"""
parity_check.py — Put-call parity data-quality gate.

This is NOT an arbitrage-profit module. True put-call-parity arbitrage is
captured by HFT in milliseconds; a daily-scan retail bot will never beat
them to it. The purpose here is narrower and more useful for this bot:
**detect unreliable option-chain quotes before we price a spread on them.**

European put-call parity (the relationship every liquid option chain must
approximately satisfy for the same strike + expiry):

    C - P = S - K * e^(-rT)

where C = call mid, P = put mid, S = spot, K = strike, r = risk-free rate,
T = years to expiry. Rearranged, the "parity residual" is:

    residual = (C - P) - (S - K * e^(-rT))

If |residual| is small relative to the underlying price, the chain's calls
and puts are internally consistent and we can trust the quotes. If it's
large, something is wrong with the data — stale quotes, a crossed/locked
market, a bad mid from a one-sided quote, a wrong underlying price, or a
mismatched expiry — and pricing a spread on those quotes would mean sizing
risk off garbage. In that case the safe action (capital preservation first)
is to SKIP the ticker this scan, not trade on numbers we don't trust.

IMPORTANT CAVEATS (why the threshold is deliberately loose):
  - US equity/ETF options are AMERICAN, not European. Parity holds only as
    an inequality for American options; early-exercise premium (mainly on
    deep ITM puts, and around dividends for calls) creates a legitimate
    residual that is NOT a data error. The gate must tolerate this — it uses
    a wide threshold and only fires on GROSS violations that indicate bad
    data, not the small, expected American/dividend deviations.
  - Dividends shift parity (C - P = S - D - K*e^(-rT) with dividends). We do
    not model the dividend term, which widens the acceptable band further —
    another reason the threshold is loose and this is a "gross garbage"
    detector, not a precise arbitrage screen.
  - ATM pairs are the most reliable test (minimal early-exercise premium,
    tightest quotes). The gate checks pairs nearest the money and ignores
    deep ITM/OTM pairs where American premium and wide spreads dominate.

Net: this fires only when the data is clearly broken, which is exactly when
we want to skip. It is intentionally hard to trip on legitimate quotes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


# Default: residual must exceed this fraction of spot to count as a violation.
# 3% of underlying is far wider than any legitimate American-exercise /
# dividend deviation for near-the-money pairs, so tripping it means the
# quotes are genuinely unreliable, not just American-style.
_DEFAULT_MAX_RESIDUAL_PCT = 0.03

# Only test pairs within this fraction of spot (near-the-money), where parity
# is most reliable and early-exercise premium is smallest.
_NEAR_THE_MONEY_PCT = 0.10

# Need at least this many valid call/put pairs to form a judgment. With fewer
# than this, we don't have enough evidence to declare the chain bad — fail
# open (allow), since a separate liquidity gate already handles thin chains.
_MIN_PAIRS = 3

# Fraction of tested pairs that must violate for the whole chain to be
# flagged. A single bad pair could be one stale contract; a majority of
# near-the-money pairs violating means the chain itself is untrustworthy.
_VIOLATION_FRACTION = 0.5


@dataclass
class ParityResult:
    """Outcome of the parity data-quality check for one chain."""
    ok: bool                       # True = quotes trustworthy (or insufficient data to judge)
    pairs_tested: int
    pairs_violating: int
    worst_residual_pct: float      # largest |residual|/spot seen, for logging
    reason: str


def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    """Mid price from a two-sided quote; None if either side missing/non-positive."""
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    if ask < bid:  # crossed/locked market — itself a data-quality red flag
        return None
    return (bid + ask) / 2.0


def check_parity(
    enriched_rows: Sequence,
    rate: float = 0.045,
    max_residual_pct: float = _DEFAULT_MAX_RESIDUAL_PCT,
    near_the_money_pct: float = _NEAR_THE_MONEY_PCT,
    min_pairs: int = _MIN_PAIRS,
    violation_fraction: float = _VIOLATION_FRACTION,
) -> ParityResult:
    """
    Check put-call parity across near-the-money strike/expiry pairs to gauge
    whether the chain's quotes are internally consistent (trustworthy) or
    grossly violated (likely bad data → skip the trade).

    Accepts a sequence of EnrichedOptionRow (or anything exposing the same
    .strike/.expiry/.option_type/.bid/.ask/.underlying_price/.dte fields).

    Returns ParityResult. `ok=True` means trustworthy OR insufficient data to
    judge (fail-open — a separate liquidity gate handles thin chains). `ok=False`
    means a majority of near-the-money pairs grossly violate parity, signalling
    unreliable quotes the bot should not size risk against.
    """
    if not enriched_rows:
        return ParityResult(True, 0, 0, 0.0, "empty chain — nothing to check (fail-open)")

    spot = enriched_rows[0].underlying_price
    if spot is None or spot <= 0:
        return ParityResult(True, 0, 0, 0.0, "no valid underlying price (fail-open)")

    # Index calls and puts by (strike, expiry) so we can pair them up.
    calls: dict[tuple, object] = {}
    puts: dict[tuple, object] = {}
    for row in enriched_rows:
        key = (round(row.strike, 4), row.expiry)
        if row.option_type == "call":
            calls[key] = row
        elif row.option_type == "put":
            puts[key] = row

    pairs_tested = 0
    pairs_violating = 0
    worst_residual_pct = 0.0

    for key in calls.keys() & puts.keys():
        strike, expiry = key

        # Near-the-money only — parity is most reliable here.
        if abs(strike - spot) / spot > near_the_money_pct:
            continue

        call_row = calls[key]
        put_row = puts[key]

        c_mid = _mid(call_row.bid, call_row.ask)
        p_mid = _mid(put_row.bid, put_row.ask)
        if c_mid is None or p_mid is None:
            continue  # can't test a pair without two valid mids

        dte = max(call_row.dte, 0)
        years = dte / 365.0

        # Theoretical (European) parity value of (C - P).
        theoretical = spot - strike * math.exp(-rate * years)
        actual = c_mid - p_mid
        residual = actual - theoretical
        residual_pct = abs(residual) / spot

        pairs_tested += 1
        worst_residual_pct = max(worst_residual_pct, residual_pct)
        if residual_pct > max_residual_pct:
            pairs_violating += 1
            logger.debug(
                "[Parity] %s K=%.2f exp=%s: residual=%.2f (%.1f%% of spot) "
                "C=%.2f P=%.2f theo(C-P)=%.2f",
                getattr(call_row, "underlying", "?"), strike, expiry,
                residual, residual_pct * 100, c_mid, p_mid, theoretical,
            )

    if pairs_tested < min_pairs:
        return ParityResult(
            True, pairs_tested, pairs_violating, worst_residual_pct,
            f"only {pairs_tested} near-the-money pair(s) testable "
            f"(< {min_pairs} needed) — insufficient evidence, fail-open",
        )

    violating_frac = pairs_violating / pairs_tested
    if violating_frac >= violation_fraction:
        return ParityResult(
            False, pairs_tested, pairs_violating, worst_residual_pct,
            f"{pairs_violating}/{pairs_tested} near-the-money pairs violate "
            f"put-call parity by >{max_residual_pct*100:.0f}% of spot "
            f"(worst {worst_residual_pct*100:.1f}%) — chain quotes unreliable, skipping",
        )

    return ParityResult(
        True, pairs_tested, pairs_violating, worst_residual_pct,
        f"parity OK — {pairs_violating}/{pairs_tested} pairs violated "
        f"(worst {worst_residual_pct*100:.1f}% of spot, within tolerance)",
    )
