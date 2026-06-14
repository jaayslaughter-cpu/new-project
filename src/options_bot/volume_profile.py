"""
Volume Profile — Support & Resistance via Volume-at-Price.

Identifies statistically significant price levels where the most trading
activity has occurred. These levels act as gravitational attractors for
price — a strong HVN below the short put strike means the underlying
may gravitate down toward it, threatening the position.

Use in the options pipeline
---------------------------
Before entering a bull put spread, call check_strike_safety() to confirm
the short strike is not directly adjacent to a strong HVN or between spot
and the POC. A strike sitting above a strong HVN gives the underlying a
clear magnetic target to fall toward.

Key output
----------
  poc_price           highest-volume price in the range (strongest magnet)
  value_area_high     top of 70% volume concentration zone
  value_area_low      bottom of 70% volume concentration zone
  support_levels      HVNs below current price (closest first)
  resistance_levels   HVNs above current price (closest first)
  vwap_20d / vwap_50d rolling VWAP support/resistance
  lvn_levels          Low Volume Nodes (price moves through quickly)

Source
------
signal_engine_v1-main/volume_profile.py — MIT license.
Core algorithm unchanged; rewritten for our module interface:
  - Circuit breaker on yfinance calls
  - Cached per-ticker with configurable TTL
  - check_strike_safety() for strategy layer integration
  - No argparse/CLI (kept in original for standalone use)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .circuit_breaker import data_circuit_breaker as _cb

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_DEFAULT_PERIOD   = "1y"
_DEFAULT_BINS     = 60
_DEFAULT_N_LEVELS = 5
_MERGE_PCT        = 1.5    # merge HVNs within 1.5% of each other
_VALUE_AREA_PCT   = 0.70   # capture 70% of volume in value area
_PEAK_WINDOW      = 3      # bars each side for local maxima detection
_MIN_PEAK_PCT     = 0.08   # HVN must be ≥ 8% of max bin volume
_CACHE_TTL        = 3600   # 1-hour cache per ticker


# ── Volume distribution math ──────────────────────────────────────────────────

def _build_volume_profile(hist, n_bins: int):
    """
    Distribute each bar's volume proportionally across its High-Low range
    into price bins. Returns (volume_by_bin, bin_centers, price_min, bin_size).

    Unlike naive approaches that put all volume at the close price, this
    distributes volume based on actual High-Low overlap with each price bin,
    giving a realistic picture of where trading actually occurred.
    """
    import math as _math

    price_min = float(hist["Low"].min())
    price_max = float(hist["High"].max())
    if price_max <= price_min:
        return None, None, None, None

    bin_size      = (price_max - price_min) / n_bins
    volume_by_bin = np.zeros(n_bins)

    for _, row in hist.iterrows():
        bar_low  = float(row["Low"])
        bar_high = float(row["High"])
        bar_vol  = float(row["Volume"])

        if _math.isnan(bar_low) or _math.isnan(bar_high) or _math.isnan(bar_vol):
            continue

        bar_range = bar_high - bar_low
        if bar_range <= 0:
            # Doji-like bar — assign all volume to close bin
            b = int((float(row["Close"]) - price_min) / bin_size)
            b = max(0, min(n_bins - 1, b))
            volume_by_bin[b] += bar_vol
            continue

        first_bin = max(0, int((bar_low  - price_min) / bin_size))
        last_bin  = min(n_bins - 1, int((bar_high - price_min) / bin_size))

        for b in range(first_bin, last_bin + 1):
            bin_lo   = price_min + b * bin_size
            bin_hi   = bin_lo + bin_size
            overlap  = max(0.0, min(bar_high, bin_hi) - max(bar_low, bin_lo))
            proportion = overlap / bar_range
            volume_by_bin[b] += bar_vol * proportion

    bin_centers = [price_min + (b + 0.5) * bin_size for b in range(n_bins)]
    return volume_by_bin, bin_centers, price_min, bin_size


def _find_peaks(volume_arr: list, window: int = _PEAK_WINDOW, min_pct: float = _MIN_PEAK_PCT) -> list:
    """Local maxima (High Volume Nodes): highest in ±window bars and ≥ min_pct of max."""
    max_vol   = max(volume_arr) if volume_arr else 0
    threshold = max_vol * min_pct
    peaks     = []
    n         = len(volume_arr)
    for i in range(window, n - window):
        if volume_arr[i] < threshold:
            continue
        if (all(volume_arr[i] >= volume_arr[i - j] for j in range(1, window + 1)) and
                all(volume_arr[i] >= volume_arr[i + j] for j in range(1, window + 1))):
            peaks.append(i)
    return peaks


def _find_troughs(volume_arr: list, window: int = _PEAK_WINDOW) -> list:
    """Local minima (Low Volume Nodes) — price moves through these quickly."""
    troughs = []
    n = len(volume_arr)
    for i in range(window, n - window):
        if (all(volume_arr[i] <= volume_arr[i - j] for j in range(1, window + 1)) and
                all(volume_arr[i] <= volume_arr[i + j] for j in range(1, window + 1))):
            troughs.append(i)
    return troughs


def _compute_value_area(
    volume_by_bin: list,
    bin_centers: list,
    poc_bin: int,
    target_pct: float = _VALUE_AREA_PCT,
) -> tuple[float, float]:
    """
    Expand outward from POC until target_pct of total volume is captured.
    Returns (va_low_price, va_high_price).
    """
    total_vol  = sum(volume_by_bin)
    target_vol = total_vol * target_pct
    n          = len(volume_by_bin)

    lo = hi = poc_bin
    accumulated = volume_by_bin[poc_bin]

    while accumulated < target_vol:
        add_above = volume_by_bin[hi + 1] if hi + 1 < n else 0
        add_below = volume_by_bin[lo - 1] if lo - 1 >= 0 else 0
        if add_above == 0 and add_below == 0:
            break
        if add_above >= add_below:
            hi += 1
            accumulated += add_above
        else:
            lo -= 1
            accumulated += add_below

    return bin_centers[lo], bin_centers[hi]


def _merge_nearby(levels: list, merge_pct: float = _MERGE_PCT) -> list:
    """Merge HVNs within merge_pct% of each other, keeping the stronger one."""
    if not levels:
        return []
    levels = sorted(levels, key=lambda x: x["price"])
    merged = [levels[0]]
    for lvl in levels[1:]:
        gap_pct = (lvl["price"] - merged[-1]["price"]) / merged[-1]["price"] * 100
        if gap_pct < merge_pct:
            if lvl["volume"] > merged[-1]["volume"]:
                merged[-1] = lvl
        else:
            merged.append(lvl)
    return merged


def _compute_vwap(hist, period_days: int) -> Optional[float]:
    """Rolling VWAP over last N trading days."""
    recent = hist.tail(period_days)
    if recent.empty or recent["Volume"].sum() == 0:
        return None
    typical = (recent["High"] + recent["Low"] + recent["Close"]) / 3
    return round(float((typical * recent["Volume"]).sum() / recent["Volume"].sum()), 4)


# ── Main function ─────────────────────────────────────────────────────────────

def get_volume_profile(
    ticker: str,
    period: str = _DEFAULT_PERIOD,
    n_bins: int = _DEFAULT_BINS,
    n_levels: int = _DEFAULT_N_LEVELS,
) -> dict:
    """
    Compute volume profile support/resistance for a ticker.

    Parameters
    ----------
    ticker : str
    period : str
        yfinance period string (e.g. "1y", "6mo")
    n_bins : int
        Number of price buckets (higher = more granular)
    n_levels : int
        Max S/R levels to return per side

    Returns
    -------
    dict with keys:
        current_price, poc_price, poc_distance_pct,
        value_area_high, value_area_low,
        vwap_20d, vwap_50d,
        support_levels, resistance_levels,
        nearest_support, nearest_resistance,
        lvn_levels, interpretation
    Returns empty dict on failure (caller should allow trade through).
    """
    src = f"yfinance_vp_{ticker}"
    if not _cb.is_available(src):
        logger.debug("[VolumeProfile] %s skipped — circuit breaker OPEN", ticker)
        return {}

    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is None or len(hist) < 30:
            _cb.record_failure(src, "insufficient history")
            return {}
        _cb.record_success(src)
    except Exception as exc:
        _cb.record_failure(src, str(exc))
        logger.warning("[VolumeProfile] %s fetch failed: %s", ticker, exc)
        return {}

    volume_by_bin, bin_centers, price_min, bin_size = _build_volume_profile(hist, n_bins)
    if volume_by_bin is None:
        return {}

    current_price = float(hist["Close"].iloc[-1])
    max_vol       = float(max(volume_by_bin))

    poc_bin   = int(np.argmax(volume_by_bin))
    poc_price = round(bin_centers[poc_bin], 4)

    va_low, va_high = _compute_value_area(list(volume_by_bin), bin_centers, poc_bin)

    peak_bins  = _find_peaks(list(volume_by_bin))
    hvn_levels = []
    for b in peak_bins:
        price        = round(bin_centers[b], 4)
        vol          = float(volume_by_bin[b])
        strength_pct = round(vol / max_vol * 100, 1)
        hvn_levels.append({"price": price, "volume": vol, "strength_pct": strength_pct})
    hvn_levels = _merge_nearby(hvn_levels)

    trough_bins = _find_troughs(list(volume_by_bin))
    lvn_prices  = [round(bin_centers[b], 4) for b in trough_bins][:5]

    def _enrich(levels: list) -> list:
        enriched = []
        for l in levels:
            dist = round((l["price"] - current_price) / current_price * 100, 2)
            enriched.append({
                "price":        l["price"],
                "distance_pct": dist,
                "strength_pct": l.get("strength_pct", 0),
            })
        return enriched

    supports    = _enrich(sorted(
        [l for l in hvn_levels if l["price"] < current_price],
        key=lambda x: x["price"], reverse=True
    )[:n_levels])

    resistances = _enrich(sorted(
        [l for l in hvn_levels if l["price"] > current_price],
        key=lambda x: x["price"]
    )[:n_levels])

    nearest_support    = supports[0]    if supports    else None
    nearest_resistance = resistances[0] if resistances else None
    poc_dist_pct       = round((poc_price - current_price) / current_price * 100, 2)

    vwap_20d = _compute_vwap(hist, 20)
    vwap_50d = _compute_vwap(hist, 50)

    parts = []
    if nearest_support:
        parts.append(
            f"Nearest support ${nearest_support['price']} "
            f"({nearest_support['distance_pct']}%, "
            f"strength {nearest_support['strength_pct']}%)"
        )
    if nearest_resistance:
        parts.append(
            f"nearest resistance ${nearest_resistance['price']} "
            f"({nearest_resistance['distance_pct']:+.2f}%, "
            f"strength {nearest_resistance['strength_pct']}%)"
        )
    dir_word = "above (resistance)" if poc_dist_pct >= 0 else "below (support)"
    parts.append(f"POC ${poc_price} is {abs(poc_dist_pct)}% {dir_word}")
    if vwap_20d:
        label = "above" if current_price > vwap_20d else "below"
        parts.append(f"price is {label} 20d VWAP ${vwap_20d}")

    return {
        "ticker":             ticker,
        "current_price":      round(current_price, 4),
        "poc_price":          poc_price,
        "poc_distance_pct":   poc_dist_pct,
        "value_area_high":    round(va_high, 4),
        "value_area_low":     round(va_low,  4),
        "vwap_20d":           vwap_20d,
        "vwap_50d":           vwap_50d,
        "support_levels":     supports,
        "resistance_levels":  resistances,
        "nearest_support":    nearest_support,
        "nearest_resistance": nearest_resistance,
        "lvn_levels":         lvn_prices,
        "interpretation":     ". ".join(parts) + ".",
    }


# ── Strategy integration ──────────────────────────────────────────────────────

def check_strike_safety(
    ticker: str,
    short_strike: float,
    spot: float,
    spread_type: str = "bull_put",
    min_hvn_distance_pct: float = 1.5,
    profile: Optional[dict] = None,
) -> tuple[bool, str]:
    """
    Determine if a short strike is safely away from volume-based S/R levels.

    For a bull put spread (short put below spot):
      UNSAFE if the short strike sits between spot and a strong HVN —
      the stock has a magnetic pull toward that HVN and may trade right
      to the short strike.

      UNSAFE if the short strike is within min_hvn_distance_pct% of the POC
      and the POC is below spot.

    Parameters
    ----------
    ticker : str
    short_strike : float
    spot : float
    spread_type : str
        "bull_put" or "bear_call"
    min_hvn_distance_pct : float
        Minimum distance from short strike to any HVN (default 1.5%)
    profile : dict or None
        Pre-computed volume profile. If None, fetches fresh.

    Returns
    -------
    (safe: bool, reason: str)
    safe=False means don't trade this strike.
    """
    if profile is None:
        profile = get_volume_profile(ticker)

    if not profile:
        # Data unavailable — allow through (non-fatal)
        return True, "volume profile unavailable — allowing"

    poc   = profile.get("poc_price", 0)
    va_lo = profile.get("value_area_low", 0)
    va_hi = profile.get("value_area_high", float("inf"))

    if spread_type == "bull_put":
        # Short put strike should be below spot
        # Danger 1: strike is between spot and the POC (if POC is below spot)
        if poc < spot and short_strike > poc and short_strike < spot:
            poc_dist_pct = abs(short_strike - poc) / spot * 100
            if poc_dist_pct < min_hvn_distance_pct:
                return False, (
                    f"short strike ${short_strike} is within {poc_dist_pct:.1f}% "
                    f"of POC ${poc} — high magnetic pull risk"
                )

        # Danger 2: strike is too close to a strong HVN below spot
        for lvl in profile.get("support_levels", []):
            hvn_price = lvl["price"]
            if hvn_price >= short_strike:
                continue  # below the short strike, not a concern
            dist_pct = abs(short_strike - hvn_price) / spot * 100
            if dist_pct < min_hvn_distance_pct and lvl.get("strength_pct", 0) > 30:
                return False, (
                    f"short strike ${short_strike} is only {dist_pct:.1f}% "
                    f"above HVN ${hvn_price} (strength {lvl['strength_pct']}%)"
                )

        # Danger 3: strike is inside the value area — contested zone
        if va_lo < short_strike < va_hi:
            return False, (
                f"short strike ${short_strike} is inside value area "
                f"[${va_lo}–${va_hi}] — high contested zone"
            )

    else:  # bear_call — short call above spot
        if poc > spot and short_strike < poc and short_strike > spot:
            poc_dist_pct = abs(poc - short_strike) / spot * 100
            if poc_dist_pct < min_hvn_distance_pct:
                return False, (
                    f"short call ${short_strike} within {poc_dist_pct:.1f}% "
                    f"of POC ${poc} — resistance likely breached"
                )

        if va_lo < short_strike < va_hi:
            return False, (
                f"short call ${short_strike} inside value area "
                f"[${va_lo}–${va_hi}]"
            )

    return True, f"strike ${short_strike} is safely away from all HVNs"


# ── Module-level cache ────────────────────────────────────────────────────────

class VolumeProfileCache:
    """
    Per-session cache of volume profiles.
    Refreshes after cache_ttl_seconds (default 1 hour).
    """

    def __init__(self, ttl_seconds: int = _CACHE_TTL):
        self._ttl   = ttl_seconds
        self._store: dict[str, tuple[float, dict]] = {}

    def get(self, ticker: str) -> dict:
        now = time.monotonic()
        ts, cached = self._store.get(ticker, (0.0, {}))
        if cached and (now - ts) < self._ttl:
            return cached
        result = get_volume_profile(ticker)
        self._store[ticker] = (now, result)
        return result

    def invalidate(self, ticker: str) -> None:
        self._store.pop(ticker, None)


# Shared instance — import and reuse in strategy/orchestrator
volume_profile_cache = VolumeProfileCache()
