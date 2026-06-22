"""Vol-risk-premium (VRP) gate — is implied vol actually rich vs realized?

GATED FEATURE. Built + tested but dormant. Activation mirrors Iron Condor:
blocked until (1) >=30 closed trades AND (2) an explicit enable flag.

This is the most on-thesis addition in the batch: the entire short-premium book
profits from the vol risk premium (IV > subsequent realized vol). Today entries
gate on IV-quality but never confirm IV is rich relative to what the underlying
is *realizing*. This module computes RV from OHLC (realized_vol.py, no new data)
and turns VRP = IV - RV into an entry gate / sizing nudge.

Intended wiring: a *gate*, not a standalone signal. When VRP is thin or negative,
it should veto or shrink premium-selling entries; it never initiates a trade.
Not applicable to the 0DTE GEX scalper (separate fast path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .realized_vol import rv_yang_zhang, ESTIMATORS

# --- PROVISIONAL_WEIGHTS (require paper-trading data to calibrate) ---
PROVISIONAL_RV_WINDOW = 21              # ~1 month realized-vol lookback
PROVISIONAL_MIN_VRP = 0.02              # IV must exceed RV by >=2 vol pts to pass
PROVISIONAL_RICH_VRP = 0.06             # >=6 vol pts = clearly rich (full size)
PROVISIONAL_MIN_IV_RV_RATIO = 1.10      # IV at least 10% above RV
PROVISIONAL_ESTIMATOR = "yang_zhang"    # drift-free, gap-aware; best for ETFs


@dataclass(frozen=True)
class VRPGateResult:
    iv: float                  # implied vol used (annualized, e.g. 0.22)
    rv: float                  # realized vol estimate (annualized)
    vrp: float                 # iv - rv, in vol points
    iv_rv_ratio: float
    passes: bool               # True => premium selling permitted by this gate
    size_factor: float         # 0..1 sizing multiplier (0 when blocked)
    estimator: str
    window: int


def evaluate_vrp_gate(
    iv: float,
    open_: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    window: int = PROVISIONAL_RV_WINDOW,
    estimator: str = PROVISIONAL_ESTIMATOR,
) -> "VRPGateResult | None":
    """Compare current IV to realized vol; decide if premium selling is favored.

    iv: annualized implied vol of the structure (e.g. ATM IV).
    OHLC sequences: recent daily bars for the underlying.

    Returns None if RV can't be computed (insufficient bars) -> caller treats as
    'no read'. Per the core book's fail-open posture, a None read should not by
    itself block trading; an explicit thin/negative VRP (passes=False) should.
    """
    if iv is None or iv <= 0:
        return None
    fn = ESTIMATORS.get(estimator, rv_yang_zhang)
    if estimator in ("close_to_close",):
        rv = fn(close, window)
    elif estimator in ("parkinson",):
        rv = fn(high, low, window)
    else:
        rv = fn(open_, high, low, close, window)
    if rv is None or rv <= 0:
        return None
    vrp = iv - rv
    ratio = iv / rv
    passes = (vrp >= PROVISIONAL_MIN_VRP) and (ratio >= PROVISIONAL_MIN_IV_RV_RATIO)
    if not passes:
        size_factor = 0.0
    else:
        # Ramp size from MIN_VRP (->0) to RICH_VRP (->1).
        span = PROVISIONAL_RICH_VRP - PROVISIONAL_MIN_VRP
        size_factor = max(0.0, min(1.0, (vrp - PROVISIONAL_MIN_VRP) / span)) if span > 0 else 1.0
    return VRPGateResult(
        iv=round(iv, 4),
        rv=round(rv, 4),
        vrp=round(vrp, 4),
        iv_rv_ratio=round(ratio, 3),
        passes=passes,
        size_factor=round(size_factor, 3),
        estimator=estimator,
        window=window,
    )
