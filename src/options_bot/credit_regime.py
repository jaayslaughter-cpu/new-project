"""Credit-spread regime input — HYG/IEF ratio as a high-yield stress proxy.

GATED FEATURE. Built + tested but DORMANT. Activation mirrors the Iron Condor
pattern: blocked until (1) >=30 closed trades AND (2) an explicit enable flag
(credit_regime_enabled in OrchestratorConfig).

Reframed onto data already in the stack: needs only HYG and IEF daily closes,
both available via yfinance / Alpha Vantage. No new subscription.

New edge vs current stack: regime detection currently uses VIX, yield curve,
breadth, ADX, plus SPY/TLT divergence and dollar stress. It has no *credit*
read. HYG (high-yield corporate) relative to IEF (7-10y Treasury) is a clean
credit-spread proxy: when the ratio falls and its z-score goes negative, credit
is widening — risk-off — which should down-weight premium selling. This is the
'credit spreads' cross-asset category flagged as genuinely additive, and it
serves the capital-preservation mandate: it pulls the bot defensive precisely
when credit markets signal stress, before that stress shows up in equity vol.

Returns a regime *adjustment*, not a standalone trade signal — it nudges the
existing regime score; wired in (dormant) at integration time.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Sequence

# --- PROVISIONAL_WEIGHTS (require paper-trading data to calibrate) ---
PROVISIONAL_LOOKBACK = 60           # trading days for the z-score baseline
PROVISIONAL_STRESS_Z = -1.0         # ratio z below this = credit stress
PROVISIONAL_CALM_Z = 1.0            # ratio z above this = credit calm
# How much a full stress reading shifts the regime score (small — one input
# among several). Sign convention: negative = more defensive.
PROVISIONAL_MAX_ADJUSTMENT = 0.10


@dataclass(frozen=True)
class CreditRegimeSignal:
    ratio: float
    z_score: float
    state: str                  # 'stress' / 'calm' / 'neutral'
    regime_adjustment: float    # additive nudge to existing regime score
    lookback: int


def compute_credit_regime(hyg_closes: Sequence[float],
                          ief_closes: Sequence[float]) -> "CreditRegimeSignal | None":
    """Compute the HYG/IEF credit-spread proxy z-score and a regime nudge.

    Returns None if there isn't enough history (caller treats as 'no read').
    """
    n = min(len(hyg_closes), len(ief_closes))
    if n < PROVISIONAL_LOOKBACK + 1:
        return None

    hyg = list(hyg_closes)[-PROVISIONAL_LOOKBACK:]
    ief = list(ief_closes)[-PROVISIONAL_LOOKBACK:]
    ratios = [h / i for h, i in zip(hyg, ief) if i]
    if len(ratios) < PROVISIONAL_LOOKBACK:
        return None

    mu = mean(ratios)
    sd = pstdev(ratios)
    current = ratios[-1]
    z = 0.0 if sd == 0 else (current - mu) / sd

    if z <= PROVISIONAL_STRESS_Z:
        state = "stress"
    elif z >= PROVISIONAL_CALM_Z:
        state = "calm"
    else:
        state = "neutral"

    # Linear in z, clamped. Negative z (stress) -> negative (defensive) nudge.
    adj = max(-PROVISIONAL_MAX_ADJUSTMENT,
              min(PROVISIONAL_MAX_ADJUSTMENT,
                  (z / abs(PROVISIONAL_STRESS_Z)) * PROVISIONAL_MAX_ADJUSTMENT))

    return CreditRegimeSignal(
        ratio=round(current, 4),
        z_score=round(z, 3),
        state=state,
        regime_adjustment=round(adj, 4),
        lookback=PROVISIONAL_LOOKBACK,
    )


def fetch_hyg_ief_closes(lookback_days: int = PROVISIONAL_LOOKBACK + 5):
    """Network fetch via yfinance. Guarded — returns (None, None) on any
    failure so the bot degrades gracefully rather than raising into the
    regime-detection path. Kept isolated from the pure scoring core above."""
    try:
        import yfinance as yf
        hyg = yf.Ticker("HYG").history(period=f"{lookback_days + 10}d")
        ief = yf.Ticker("IEF").history(period=f"{lookback_days + 10}d")
        if hyg.empty or ief.empty:
            return None, None
        return list(hyg["Close"].values), list(ief["Close"].values)
    except Exception:
        return None, None
