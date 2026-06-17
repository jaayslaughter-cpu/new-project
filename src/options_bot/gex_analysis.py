"""
gex_analysis.py — Gamma Exposure (GEX) strike map analysis.

Adapted from volcon-strike-map (calculator.py + models.py) by volcon contributors.
Original: https://github.com/volcon/volcon-strike-map

Computes gamma-notional exposure per strike from an enriched option chain,
identifies put wall, call wall, and pin strike by gamma concentration,
classifies dealer gamma regime (positive/negative/mixed), and scores each
strike 0-100 by a weighted combination of OI, volume, gamma notional, and
side imbalance.

Used by the strategy layer to add a GEX-weighted layer on top of the
volume profile check — avoids selecting short strikes that sit near
high-gamma walls where dealers will aggressively hedge.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GEXStrike:
    """Gamma exposure analysis for a single strike."""
    strike: float
    call_oi: int
    put_oi: int
    call_volume: int
    put_volume: int
    call_gamma_notional: float   # $ gamma: gamma × OI × 100 × spot² × 0.01
    put_gamma_notional: float
    abs_gamma_notional: float    # |call_gex| + |put_gex|
    net_gamma_proxy: float       # call_gex - put_gex (positive = dealer long gamma)
    gex_score: float             # 0-100 composite score
    cdf_below: float             # probability underlying is below this strike at expiry
    tags: list[str] = field(default_factory=list)


@dataclass
class GEXAnalysis:
    """Full GEX analysis for one ticker / expiry."""
    ticker: str
    expiry: date
    spot: float
    dte_days: int
    atm_iv: float
    expected_move: float         # spot × atm_iv × sqrt(dte/365)
    gamma_regime: str            # "positive" | "negative" | "mixed"
    put_wall: Optional[GEXStrike]      # highest put-gamma strike below spot
    call_wall: Optional[GEXStrike]     # highest call-gamma strike above spot
    pin_strike: Optional[GEXStrike]    # highest total-gamma strike (expected pin)
    levels: list[GEXStrike]      # top-N strikes by gex_score
    negative_gamma: bool         # True = negative gamma regime (breakout risk)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _cdf_below(spot: float, strike: float, iv: float, dte_days: int) -> float:
    """Probability that underlying closes below `strike` at expiry (log-normal)."""
    if spot <= 0 or strike <= 0:
        return 0.5
    sigma = max(iv, 0.0001)
    years = max(dte_days, 1) / 365.0
    d2 = (math.log(spot / strike) - 0.5 * sigma * sigma * years) / (sigma * math.sqrt(years))
    return min(max(_normal_cdf(-d2), 0.0), 1.0)


def _gamma_notional(gamma: Optional[float], oi: Optional[int], spot: float) -> float:
    """Dollar gamma: gamma × OI × 100 × spot² × 0.01"""
    if not gamma or not oi or gamma <= 0 or oi <= 0:
        return 0.0
    return gamma * oi * 100 * spot * spot * 0.01


def _normalize(value: float, max_value: float) -> float:
    return value / max_value if max_value > 0 else 0.0


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_gex(
    ticker: str,
    enriched_rows: list,        # list[EnrichedOptionRow] from greeks.py
    expiry: date,
    spot: float,
    dte_days: int,
    atm_iv: float = 0.20,
    max_levels: int = 15,
) -> Optional[GEXAnalysis]:
    """
    Build a GEX strike map from an enriched option chain.

    Parameters
    ----------
    ticker : str
        Underlying symbol (e.g. "SPY")
    enriched_rows : list[EnrichedOptionRow]
        Already-enriched rows from GreeksEnricher — must have gamma, OI,
        volume, option_type, and strike populated.
    expiry : date
        The expiry date to analyse (filter rows to this expiry).
    spot : float
        Current underlying price.
    dte_days : int
        Days to expiration.
    atm_iv : float
        ATM implied volatility (used for expected move and CDF).
    max_levels : int
        Maximum number of strike levels to return in `levels`.

    Returns
    -------
    GEXAnalysis or None if there are insufficient rows.
    """
    # Filter to target expiry and group by strike
    chain = [r for r in enriched_rows if r.expiry == expiry]
    if not chain:
        logger.debug("[GEX] %s: no rows for expiry %s", ticker, expiry)
        return None

    # Build per-strike aggregates
    strikes: dict[float, dict] = {}
    for row in chain:
        k = round(float(row.strike), 2)
        if k not in strikes:
            strikes[k] = {
                "call_gamma": 0.0, "put_gamma": 0.0,
                "call_oi": 0, "put_oi": 0,
                "call_vol": 0, "put_vol": 0,
                "ivs": [],
            }
        s = strikes[k]
        oi  = row.open_interest or 0
        vol = getattr(row.raw, "volume", None) or 0
        gma = row.gamma or 0.0
        iv  = row.iv or 0.0
        if iv > 0:
            s["ivs"].append(iv)
        from .contracts import OptionType
        if row.option_type == OptionType.CALL:
            s["call_oi"]    += oi
            s["call_vol"]   += vol
            s["call_gamma"] += _gamma_notional(gma, oi, spot)
        else:
            s["put_oi"]    += oi
            s["put_vol"]   += vol
            s["put_gamma"] += _gamma_notional(gma, oi, spot)

    if not strikes:
        return None

    # Determine ATM IV from chain if not supplied
    all_ivs = [iv for d in strikes.values() for iv in d["ivs"]]
    if all_ivs:
        atm_iv = float(median(all_ivs))

    # Compute composite scores
    max_oi      = max(d["call_oi"] + d["put_oi"]         for d in strikes.values()) or 1
    max_vol     = max(d["call_vol"] + d["put_vol"]        for d in strikes.values()) or 1
    max_abs_gex = max(abs(d["call_gamma"]) + abs(d["put_gamma"]) for d in strikes.values()) or 1

    gex_levels: list[GEXStrike] = []
    for k, d in strikes.items():
        total_oi  = d["call_oi"]  + d["put_oi"]
        total_vol = d["call_vol"] + d["put_vol"]
        abs_gex   = abs(d["call_gamma"]) + abs(d["put_gamma"])
        net_gex   = d["call_gamma"] - d["put_gamma"]
        side_oi_imb  = abs(d["call_oi"]  - d["put_oi"])  / max(total_oi,  1)
        side_vol_imb = abs(d["call_vol"] - d["put_vol"]) / max(total_vol, 1)
        score = 100.0 * (
            0.30 * _normalize(total_oi,  max_oi)
            + 0.15 * _normalize(total_vol, max_vol)
            + 0.35 * _normalize(abs_gex,   max_abs_gex)
            + 0.10 * side_oi_imb
            + 0.10 * side_vol_imb
        )
        tags = []
        cdf  = _cdf_below(spot, k, atm_iv, dte_days)
        if cdf <= 0.25:
            tags.append("lower_tail")
        if cdf >= 0.75:
            tags.append("upper_tail")
        gex_levels.append(GEXStrike(
            strike=k,
            call_oi=d["call_oi"],   put_oi=d["put_oi"],
            call_volume=d["call_vol"], put_volume=d["put_vol"],
            call_gamma_notional=round(d["call_gamma"], 2),
            put_gamma_notional=round(d["put_gamma"],  2),
            abs_gamma_notional=round(abs_gex, 2),
            net_gamma_proxy=round(net_gex, 2),
            gex_score=round(score, 2),
            cdf_below=round(cdf, 4),
            tags=tags,
        ))

    # Identify walls and pin strike
    below = [l for l in gex_levels if l.strike < spot and l.put_oi  > 0]
    above = [l for l in gex_levels if l.strike > spot and l.call_oi > 0]
    put_wall_lvl  = max(below or gex_levels, key=lambda l: (l.put_gamma_notional,  l.put_oi))
    call_wall_lvl = max(above or gex_levels, key=lambda l: (l.call_gamma_notional, l.call_oi))
    pin_lvl       = max(gex_levels, key=lambda l: l.abs_gamma_notional)

    put_wall_lvl.tags  = sorted(set(put_wall_lvl.tags  + ["put_wall",  "support_candidate"]))
    call_wall_lvl.tags = sorted(set(call_wall_lvl.tags + ["call_wall", "resistance_candidate"]))
    pin_lvl.tags       = sorted(set(pin_lvl.tags       + ["pin_strike"]))

    # Gamma regime
    net_total = sum(l.net_gamma_proxy    for l in gex_levels)
    abs_total = sum(l.abs_gamma_notional for l in gex_levels)
    ratio = net_total / abs_total if abs_total else 0.0
    if ratio > 0.15:
        gamma_regime = "positive"
    elif ratio < -0.15:
        gamma_regime = "negative"
    else:
        gamma_regime = "mixed"

    expected_move = spot * atm_iv * math.sqrt(max(dte_days, 1) / 365.0)
    top_levels = sorted(gex_levels, key=lambda l: l.gex_score, reverse=True)[:max_levels]

    logger.debug(
        "[GEX] %s %s: regime=%s put_wall=%.1f pin=%.1f call_wall=%.1f "
        "expected_move=±$%.2f",
        ticker, expiry, gamma_regime,
        put_wall_lvl.strike, pin_lvl.strike, call_wall_lvl.strike, expected_move,
    )

    return GEXAnalysis(
        ticker=ticker,
        expiry=expiry,
        spot=spot,
        dte_days=dte_days,
        atm_iv=round(atm_iv, 4),
        expected_move=round(expected_move, 4),
        gamma_regime=gamma_regime,
        put_wall=put_wall_lvl,
        call_wall=call_wall_lvl,
        pin_strike=pin_lvl,
        levels=top_levels,
        negative_gamma=gamma_regime == "negative",
    )


# ---------------------------------------------------------------------------
# Strategy integration helper
# ---------------------------------------------------------------------------

def check_strike_gex_safety(
    analysis: Optional[GEXAnalysis],
    short_strike: float,
    min_distance_pct: float = 1.5,
) -> tuple[bool, str]:
    """
    Return (safe, reason) for placing a short strike near GEX walls.

    Rules (non-fatal: if analysis is None, returns safe=True):
      1. Short strike must not be within min_distance_pct% of the put wall.
         Put walls are strong support — dealers buy aggressively near them,
         which can temporarily hold the underlying up but also creates sharp
         reversals when the wall is breached.
      2. Short strike must not be above the pin strike.
         The pin strike is where the most gamma is concentrated — the
         underlying is magnetically attracted to it at expiry.
      3. Negative gamma regime warning: in negative gamma regimes, moves
         accelerate rather than mean-revert. Allowed but logged.
    """
    if analysis is None:
        return True, "GEX data unavailable — skipping check"

    # Rule 1: distance from put wall
    if analysis.put_wall is not None:
        wall = analysis.put_wall.strike
        dist_pct = abs(short_strike - wall) / wall * 100
        if dist_pct < min_distance_pct:
            return False, (
                f"Short strike ${short_strike:.1f} is within {dist_pct:.1f}% "
                f"of put wall ${wall:.1f} (min {min_distance_pct}%)"
            )

    # Rule 2: don't short above the pin strike
    if analysis.pin_strike is not None and short_strike > analysis.pin_strike.strike:
        return False, (
            f"Short strike ${short_strike:.1f} is above pin strike "
            f"${analysis.pin_strike.strike:.1f} — elevated assignment risk"
        )

    # Rule 3: regime warning
    regime_note = ""
    if analysis.negative_gamma:
        regime_note = " [WARN: negative gamma regime — moves can accelerate]"

    return True, (
        f"GEX OK: put_wall=${analysis.put_wall.strike if analysis.put_wall else 'N/A':.1f} "
        f"pin=${analysis.pin_strike.strike if analysis.pin_strike else 'N/A':.1f} "
        f"regime={analysis.gamma_regime}{regime_note}"
    )


# ---------------------------------------------------------------------------
# CBOE Free GEX Fallback
# ---------------------------------------------------------------------------
# Uses CBOE's public delayed options JSON endpoint (no API key required) to
# compute GEX, put/call ratios, and gamma walls as a fallback when the bot's
# primary enriched chain is unavailable (e.g. pre-market, Alpaca data outage).
#
# Source: adapted from OpenTrading/tools/options/opt.py
# Key changes: integrated with circuit_breaker, returns GEXAnalysis-compatible
# dict rather than printing, stdlib only (no requests).
#
# Usage in GEXEngine:
#   if primary_analysis is None:
#       fallback = fetch_cboe_gex(ticker, dte_max=7)
#       if fallback:
#           # use fallback["pin_strike"], fallback["put_wall"], etc.
# ---------------------------------------------------------------------------

_CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
_CBOE_UA  = "Mozilla/5.0 (OptionsBot gex-fallback/1.0)"


def fetch_cboe_gex(
    ticker: str,
    dte_max: int = 7,
) -> Optional[dict]:
    """
    Fetch GEX, put/call ratios, and gamma walls from CBOE's free delayed chain.

    No API key required. Data is delayed ~15 minutes — suitable as a fallback
    when live Alpaca chain data is unavailable, not as a primary signal.

    Returns a dict with keys:
        spot          float   — current underlying price
        net_gex_usd   float   — net $ GEX per 1% move (positive = vol-suppressing)
        gex_sign      str     — 'positive' | 'negative' | 'flat'
        pc_oi         float   — put/call open interest ratio
        pc_vol        float   — put/call volume ratio
        call_wall     float   — strike with highest positive gamma (resistance)
        put_wall      float   — strike with most negative gamma (support)
        negative_gamma bool   — True if net GEX is negative (vol-amplifying)
        source        str     — 'cboe_delayed'

    Returns None if the fetch fails or the chain is empty.
    """
    from .circuit_breaker import data_circuit_breaker as _cb_gex
    cb_key = "cboe_gex_fallback"

    if not _cb_gex.is_available(cb_key):
        logger.debug("[GEX-CBOE] Circuit breaker OPEN — skipping fallback")
        return None

    try:
        import urllib.request as _ur
        import ssl as _ssl
        import json as _json
        from datetime import date as _date, datetime as _dt

        ctx = _ssl.create_default_context()
        url = _CBOE_URL.format(sym=ticker.upper().strip())
        req = _ur.Request(url, headers={"User-Agent": _CBOE_UA})

        with _ur.urlopen(req, timeout=20, context=ctx) as r:
            raw = _json.loads(r.read().decode("utf-8", errors="replace"))

        data = raw.get("data", {})
        spot = data.get("current_price") or data.get("close")
        rows = data.get("options", [])

        if not spot or not rows:
            _cb_gex.record_failure(cb_key, "empty chain")
            return None

        today = _date.today()
        call_oi = put_oi = call_vol = put_vol = 0
        net_gamma = 0.0
        by_strike: dict[float, float] = {}

        for o in rows:
            try:
                sym = o.get("option", "")
                if len(sym) < 15:
                    continue
                # Parse OCC symbol: SPY260615C00500000
                strike = int(sym[-8:]) / 1000.0
                cp     = sym[-9]
                exp    = f"20{sym[-15:-13]}-{sym[-13:-11]}-{sym[-11:-9]}"
                dte    = (_dt.strptime(exp, "%Y-%m-%d").date() - today).days
            except Exception:
                continue

            if dte < 0 or dte > dte_max:
                continue

            oi    = o.get("open_interest") or 0
            vol   = o.get("volume") or 0
            gamma = o.get("gamma") or 0.0
            # Dealer convention: long calls (positive gamma), short puts (negative)
            signed_g = gamma * oi * (1 if cp == "C" else -1)
            net_gamma += signed_g
            by_strike[strike] = by_strike.get(strike, 0.0) + signed_g

            if cp == "C":
                call_oi  += oi;  call_vol  += vol
            else:
                put_oi   += oi;  put_vol   += vol

        if not by_strike:
            _cb_gex.record_failure(cb_key, "no valid contracts after DTE filter")
            return None

        # $ GEX per 1% move = Σ(signed gamma * OI) * 100 * spot² * 0.01
        dollar_gex = net_gamma * 100 * spot * spot * 0.01
        call_wall = max(by_strike, key=by_strike.get)
        put_wall  = min(by_strike, key=by_strike.get)

        _cb_gex.record_success(cb_key)
        result = {
            "spot":          float(spot),
            "net_gex_usd":   round(dollar_gex, 2),
            "gex_sign":      "positive" if dollar_gex > 0 else "negative" if dollar_gex < 0 else "flat",
            "pc_oi":         round(put_oi  / call_oi,  3) if call_oi  else None,
            "pc_vol":        round(put_vol / call_vol, 3) if call_vol else None,
            "call_wall":     call_wall,
            "put_wall":      put_wall,
            "negative_gamma": dollar_gex < 0,
            "source":        "cboe_delayed",
        }
        logger.info(
            "[GEX-CBOE] %s: spot=%.2f net_gex=$%.1fbn sign=%s "
            "put_wall=%.1f call_wall=%.1f pc_oi=%.2f",
            ticker, spot, dollar_gex / 1e9, result["gex_sign"],
            put_wall, call_wall, result["pc_oi"] or 0,
        )
        return result

    except Exception as exc:
        from .circuit_breaker import data_circuit_breaker as _cb_gex2
        _cb_gex2.record_failure(cb_key, str(exc))
        logger.warning("[GEX-CBOE] Fallback fetch failed for %s: %s", ticker, exc)
        return None
