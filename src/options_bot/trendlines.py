"""
trendlines.py — automatic trendline detection from OHLCV data.

Adapted from stock-options-scanner/trendlines.py.
Pure Python — no numpy or pandas dependency.

Uses swing highs/lows to fit OLS support and resistance trendlines,
classifies the chart pattern, and flags proximity/breakout conditions.

Input:  list of OHLCV dicts [{date, open, high, low, close, volume}, ...]
Output: TrendlineResult dataclass with pattern, proximity flags, and score.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrendlineResult:
    """Result of trendline analysis for a single ticker."""
    pattern: str                     # e.g. "ascending_channel", "converging_wedge"
    support_level: Optional[float]   # current projected support trendline price
    support_slope: Optional[float]   # slope %/day (positive = rising)
    resist_level: Optional[float]    # current projected resistance trendline price
    resist_slope: Optional[float]    # slope %/day (negative = falling)
    near_support: bool               # price within 2.5% above support TL
    near_resistance: bool            # price within 2% below resistance TL
    broke_above: bool                # price > resistance TL by >0.5%
    broke_below: bool                # price < support TL
    trendline_score: float           # 0.0-1.5 contribution to scanner total score
    summary: str                     # one-liner for logs


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _find_swings(
    highs: list[float],
    lows: list[float],
    window: int = 2,
) -> tuple[list[int], list[int]]:
    """Return (swing_high_indices, swing_low_indices) using a rolling window."""
    n = len(highs)
    sh, sl = [], []
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            sh.append(i)
        if lows[i] == min(lows[i - window: i + window + 1]):
            sl.append(i)
    return sh, sl


def _fit_line(x: list[float], y: list[float]) -> tuple[float, float]:
    """Ordinary least-squares. Returns (slope, intercept). Pure Python."""
    n = len(x)
    if n < 2:
        return 0.0, y[0] if y else 0.0
    sx = sum(x); sy = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sx2 = sum(xi * xi for xi in x)
    d = n * sx2 - sx * sx
    if d == 0:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / d
    return slope, (sy - slope * sx) / n


def _direction(slope_pct: float) -> str:
    if slope_pct > 0.15:
        return "rising"
    if slope_pct < -0.15:
        return "falling"
    return "flat"


def analyze_trendlines(ohlcv: list[dict]) -> Optional[TrendlineResult]:
    """
    Detect support/resistance trendlines and score them for scanner use.

    Trendline score contribution (+0 to +1.5):
      +1.5  breakout above resistance on volume  (momentum confirmed)
      +1.0  near ascending support               (ideal short-put entry zone)
      +0.8  ascending_triangle / converging_wedge near support
      +0.6  ascending channel near support
      +0.3  any bullish pattern (ascending) detected, no proximity signal
      +0.0  descending/bearish pattern or insufficient data
      -0.0  breakdown below support (score floors at 0, not negative)

    Returns None if fewer than 8 bars of data.
    """
    if not ohlcv or len(ohlcv) < 8:
        return None

    highs  = [_safe_float(b.get("high",  b.get("close", 0))) or 0.0 for b in ohlcv]
    lows   = [_safe_float(b.get("low",   b.get("close", 0))) or 0.0 for b in ohlcv]
    closes = [_safe_float(b.get("close", 0)) or 0.0 for b in ohlcv]

    # Drop trailing zero bars
    while closes and closes[-1] == 0.0:
        closes.pop(); highs.pop(); lows.pop()

    if len(closes) < 8:
        return None

    current = closes[-1]
    if current == 0.0:
        return None

    n_bars   = len(closes)
    last_idx = float(n_bars - 1)
    sh, sl   = _find_swings(highs, lows, window=2)

    support_level = support_slope = resist_level = resist_slope = None
    rt_dir = st_dir = None

    # Resistance trendline
    if len(sh) >= 2:
        sh_idx = sh[-5:]
        slope, intercept = _fit_line([float(i) for i in sh_idx],
                                      [highs[i] for i in sh_idx])
        proj = _safe_float(slope * last_idx + intercept)
        if proj and proj > 0:
            resist_level = round(proj, 2)
            slope_pct    = round(slope / proj * 100, 3)
            resist_slope = slope_pct
            rt_dir       = _direction(slope_pct)

    # Support trendline
    if len(sl) >= 2:
        sl_idx = sl[-5:]
        slope, intercept = _fit_line([float(i) for i in sl_idx],
                                      [lows[i] for i in sl_idx])
        proj = _safe_float(slope * last_idx + intercept)
        if proj and proj > 0:
            support_level = round(proj, 2)
            slope_pct     = round(slope / proj * 100, 3)
            support_slope = slope_pct
            st_dir        = _direction(slope_pct)

    if support_level is None and resist_level is None:
        return TrendlineResult(
            pattern="insufficient_data",
            support_level=None, support_slope=None,
            resist_level=None,  resist_slope=None,
            near_support=False, near_resistance=False,
            broke_above=False,  broke_below=False,
            trendline_score=0.0,
            summary="insufficient swing data",
        )

    # Pattern classification
    if rt_dir and st_dir:
        pattern = {
            ("rising",  "rising"):  "ascending_channel",
            ("falling", "falling"): "descending_channel",
            ("flat",    "rising"):  "ascending_triangle",
            ("falling", "flat"):    "descending_triangle",
            ("falling", "rising"):  "converging_wedge",
            ("rising",  "falling"): "expanding_wedge",
            ("flat",    "flat"):    "horizontal_range",
            ("rising",  "flat"):    "rising_support_flat_resistance",
            ("flat",    "falling"): "falling_support_flat_resistance",
        }.get((rt_dir, st_dir), "mixed")
    elif rt_dir:
        pattern = f"resistance_{rt_dir}_only"
    elif st_dir:
        pattern = f"support_{st_dir}_only"
    else:
        pattern = "unknown"

    # Proximity / breakout flags
    near_resistance = broke_above = False
    near_support    = broke_below = False

    if resist_level:
        d = (current - resist_level) / resist_level * 100
        near_resistance = -2.0 <= d <= 0.5
        broke_above     = d > 0.5

    if support_level:
        d = (current - support_level) / support_level * 100
        near_support  = 0.0 <= d <= 2.5
        broke_below   = d < 0.0

    # Scoring — only reward bullish structural setups
    BULLISH_PATTERNS = {
        "ascending_channel", "ascending_triangle",
        "converging_wedge", "rising_support_flat_resistance",
    }
    is_bullish_pattern = pattern in BULLISH_PATTERNS

    if broke_above:
        score = 1.5    # confirmed breakout — strongest signal
    elif near_support and pattern in ("ascending_triangle", "converging_wedge"):
        score = 0.8    # ideal entry zone in bullish compression pattern
    elif near_support and pattern == "ascending_channel":
        score = 0.6    # channel support entry
    elif near_support and is_bullish_pattern:
        score = 1.0    # generic bullish pattern near support
    elif is_bullish_pattern:
        score = 0.3    # bullish structure, no proximity confirmation
    else:
        score = 0.0    # neutral/bearish — don't reward

    # Summary
    parts = [f"pattern={pattern}"]
    if support_level:
        dist = round((current - support_level) / support_level * 100, 1)
        tag = "NEAR" if near_support else ("BROKE_BELOW" if broke_below else "above")
        parts.append(f"support=${support_level:.2f}({dist:+.1f}%,{tag})")
    if resist_level:
        dist = round((current - resist_level) / resist_level * 100, 1)
        tag = "NEAR" if near_resistance else ("BROKE_ABOVE" if broke_above else "below")
        parts.append(f"resist=${resist_level:.2f}({dist:+.1f}%,{tag})")

    return TrendlineResult(
        pattern=pattern,
        support_level=support_level, support_slope=support_slope,
        resist_level=resist_level,   resist_slope=resist_slope,
        near_support=near_support,   near_resistance=near_resistance,
        broke_above=broke_above,     broke_below=broke_below,
        trendline_score=round(score, 2),
        summary=" | ".join(parts),
    )
