"""
IV Robustness & Quality Gate — options_bot port of volscope iv_robustness.py.

Solves the single-spike IV contamination bug: a single extreme IV event
(earnings crush, M&A, litigation) stretches the 52-week MIN-MAX range,
making standard IV Rank show a misleading CHEAP verdict even when IV
Percentile correctly signals elevated options prices.

Real example (FISV May-2026):
    Standard IVR  12.5 → signals CHEAP
    IV Percentile 78.6 → signals HIGH
    Divergence    66.1 → single-spike contamination confirmed

This module provides four functions that gate ticker entry in TickerGate:

1. robust_iv_rank()        — winsorized 5th/95th percentile IVR (spike-immune)
2. detect_contamination()  — |IVR − IVP| categorical severity
3. detect_structural_break() — Pelt or CUSUM change-point detection
4. assess_iv_quality()     — composite TRADE / CAUTION / BLOCK recommendation

Integration
-----------
TickerGate.filter() calls assess_iv_quality() after fetching 252d of IV
history via yfinance. BLOCK tickers are hard-excluded. CAUTION tickers
pass but are logged and the recommendation is attached to the order metadata
so the Discord alert shows it.

IVQualityGate is a lightweight wrapper used by TickerGate — it caches
results for _IV_TTL_SECONDS (4 hours) so repeated calls within a session
don't re-fetch history.

Sources
-------
Adapted from: volscope-main/volscope/analytics/iv_robustness.py (v0.6.1)
Original authors: volscope project contributors
Adaptation: rewritten imports, removed DB/UI dependencies, added yfinance
fetcher, integrated with our circuit_breaker and logging conventions.

References
----------
- López de Prado (2018), AFML Ch. 7 + Ch. 11 (regime detection)
- Killick, Fearnhead & Eckley (2012) Pelt algorithm
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from .circuit_breaker import data_circuit_breaker as _cb

logger = logging.getLogger(__name__)

# Cache TTL — IV quality rarely changes within a trading session
_IV_TTL_SECONDS = 4 * 60 * 60   # 4 hours


# ---------------------------------------------------------------------------
# Contamination level
# ---------------------------------------------------------------------------

class ContaminationLevel(str, Enum):
    """Categorical severity from |IVR − IVP| divergence."""
    CLEAN   = "clean"    # ≤ 15 points  → metrics agree
    MILD    = "mild"     # 15-30 points → prefer IVP
    SEVERE  = "severe"   # 30-50 points → use IVP only
    EXTREME = "extreme"  # > 50 points  → consider blocking


# ---------------------------------------------------------------------------
# A1: Robust IV Rank
# ---------------------------------------------------------------------------

def robust_iv_rank(
    iv_series: pd.Series,
    *,
    lookback: int = 252,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> Optional[float]:
    """
    Compute IV Rank using winsorized 5th/95th-percentile bounds.

    Standard IV Rank uses MIN and MAX of the 52-week window.  A single
    extreme IV spike stretches that range, making current IV look cheap
    even when it is elevated relative to normal conditions.

    This function uses the 5th and 95th percentile as bounds instead,
    producing a 'where is current IV in its TYPICAL range' measure.

    Returns: robust IVR in [-50, 150] (capped), or None on insufficient data.
    Values outside [0, 100] indicate below/above the typical range.
    """
    if iv_series is None or len(iv_series) < lookback:
        return None

    history = iv_series.dropna().iloc[-lookback:].copy()
    history = history[(history > 0) & (history < 500)]

    if len(history) < int(lookback * 0.8):
        return None

    try:
        current = float(iv_series.iloc[-1])
    except (IndexError, TypeError, ValueError):
        return None
    if pd.isna(current) or current <= 0:
        return None

    iv_low  = float(history.quantile(lower_quantile))
    iv_high = float(history.quantile(upper_quantile))
    if iv_high - iv_low < 1.0:
        return None

    rank = (current - iv_low) / (iv_high - iv_low) * 100.0
    return float(max(-50.0, min(150.0, rank)))


# ---------------------------------------------------------------------------
# A2: Contamination detection
# ---------------------------------------------------------------------------

def detect_contamination(
    iv_rank: Optional[float],
    iv_percentile: Optional[float],
) -> tuple[ContaminationLevel, float]:
    """
    Categorise the IVR/IVP divergence into a severity level.

    Returns (level, divergence) where divergence = |IVR - IVP|.
    Returns (CLEAN, 0.0) if either input is None/NaN.
    """
    if (
        iv_rank is None
        or iv_percentile is None
        or not math.isfinite(float(iv_rank))
        or not math.isfinite(float(iv_percentile))
    ):
        return ContaminationLevel.CLEAN, 0.0

    divergence = abs(float(iv_rank) - float(iv_percentile))
    if divergence <= 15.0:
        return ContaminationLevel.CLEAN, divergence
    if divergence <= 30.0:
        return ContaminationLevel.MILD, divergence
    if divergence <= 50.0:
        return ContaminationLevel.SEVERE, divergence
    return ContaminationLevel.EXTREME, divergence


# ---------------------------------------------------------------------------
# A3: Structural break detection
# ---------------------------------------------------------------------------

def detect_structural_break(
    iv_series: pd.Series,
    *,
    min_segment_length: int = 60,
    pelt_penalty: float = 10.0,
    magnitude_floor: float = 0.30,
) -> Optional[dict[str, Any]]:
    """
    Detect a permanent regime shift in the IV series.

    Uses the Pelt change-point algorithm (ruptures library) when available,
    falls back to CUSUM. Returns None if no significant break found.

    Returns dict with: break_date, days_since_break, pre_break_mean,
    post_break_mean, magnitude, direction.
    """
    if iv_series is None or len(iv_series) < 2 * min_segment_length:
        return None

    try:
        import ruptures as rpt  # type: ignore[import-not-found]
        history = iv_series.dropna()
        if len(history) < 2 * min_segment_length:
            return None
        arr = history.values.astype(float)
        try:
            algo = rpt.Pelt(model="rbf", min_size=min_segment_length).fit(arr)
            breakpoints = algo.predict(pen=pelt_penalty)
        except Exception as exc:
            logger.debug("[IVQuality] Pelt failed (%s); using CUSUM fallback", exc)
            return _detect_break_cusum(iv_series, min_segment_length, magnitude_floor)
        if len(breakpoints) <= 1:
            return None
        last_break_idx = int(breakpoints[-2])
        return _build_break_report(history, arr, last_break_idx, magnitude_floor)
    except ImportError:
        return _detect_break_cusum(iv_series, min_segment_length, magnitude_floor)


def _detect_break_cusum(
    iv_series: pd.Series,
    min_segment_length: int,
    magnitude_floor: float,
) -> Optional[dict[str, Any]]:
    """CUSUM fallback when ruptures is unavailable. O(n), correct for our use case."""
    history = iv_series.dropna()
    if len(history) < 2 * min_segment_length:
        return None
    arr = history.values.astype(float)
    n = len(arr)
    best_score, best_idx = 0.0, -1
    for split in range(min_segment_length, n - min_segment_length):
        pre, post = arr[:split], arr[split:]
        pre_mean = float(pre.mean())
        if pre_mean == 0:
            continue
        magnitude = abs(float(post.mean()) - pre_mean) / pre_mean
        std = float(np.sqrt(pre.var(ddof=1) + post.var(ddof=1) + 1e-9))
        score = abs(float(post.mean()) - pre_mean) / std
        if score > best_score and magnitude >= magnitude_floor:
            best_score, best_idx = score, split
    if best_idx < 0:
        return None
    return _build_break_report(history, arr, best_idx, magnitude_floor)


def _build_break_report(
    history: pd.Series,
    arr: np.ndarray,
    last_break_idx: int,
    magnitude_floor: float,
) -> Optional[dict[str, Any]]:
    pre, post = arr[:last_break_idx], arr[last_break_idx:]
    if len(pre) == 0 or len(post) == 0:
        return None
    pre_mean, post_mean = float(pre.mean()), float(post.mean())
    if pre_mean == 0:
        return None
    magnitude = abs(post_mean - pre_mean) / pre_mean
    if magnitude < magnitude_floor:
        return None
    days_since = int(len(arr) - last_break_idx)
    if isinstance(history.index, pd.DatetimeIndex):
        break_date = history.index[last_break_idx].strftime("%Y-%m-%d")
    else:
        break_date = f"day_{last_break_idx}_of_{len(arr)}"
    return {
        "break_date":      break_date,
        "days_since_break": days_since,
        "pre_break_mean":  pre_mean,
        "post_break_mean": post_mean,
        "magnitude":       magnitude,
        "direction":       "up" if post_mean > pre_mean else "down",
    }


# ---------------------------------------------------------------------------
# A4: Composite quality assessment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IVQualityReport:
    """
    Composite quality assessment for a single ticker.

    Quality score is a 0-100 deduction model:
        SEVERE contamination        → -25
        EXTREME contamination       → -50
        Structural break <90d ago   → -30
        Structural break 90-180d    → -15
        Robust IVR diverges >30pt   → -20
        Insufficient data           → forced to 0

    Recommendation:
        ≥ 70 → TRADE   — IV signals are reliable, proceed normally
        40-69 → CAUTION — trade with reduced size or skip
        < 40  → BLOCK  — do not trade until quality recovers
    """
    ticker:          str
    iv_rank:         Optional[float]
    iv_percentile:   Optional[float]
    robust_iv_rank:  Optional[float]
    contamination:   ContaminationLevel
    divergence:      float
    structural_break: Optional[dict[str, Any]]
    quality_score:   int           # 0-100
    tradable:        bool
    recommendation:  str           # 'TRADE' | 'CAUTION' | 'BLOCK'
    warnings:        list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker":           self.ticker,
            "iv_rank":          self.iv_rank,
            "iv_percentile":    self.iv_percentile,
            "robust_iv_rank":   self.robust_iv_rank,
            "contamination":    self.contamination.value,
            "divergence":       round(self.divergence, 2),
            "structural_break": self.structural_break,
            "quality_score":    int(self.quality_score),
            "tradable":         bool(self.tradable),
            "recommendation":   self.recommendation,
            "warnings":         list(self.warnings),
        }


def assess_iv_quality(
    ticker: str,
    iv_series: pd.Series,
    iv_rank_value: Optional[float],
    iv_percentile_value: Optional[float],
) -> IVQualityReport:
    """End-to-end IV quality assessment combining all four checks."""
    warnings: list[str] = []
    quality = 100

    robust = robust_iv_rank(iv_series)

    def _missing(v: object) -> bool:
        return v is None or (isinstance(v, float) and not math.isfinite(v))

    insufficient = (
        _missing(iv_rank_value)
        or _missing(iv_percentile_value)
        or iv_series is None
        or len(iv_series.dropna()) < 100
    )

    contamination, divergence = detect_contamination(iv_rank_value, iv_percentile_value)

    if contamination == ContaminationLevel.SEVERE:
        quality -= 25
        warnings.append(
            f"IVR ({iv_rank_value:.1f}) and IVP ({iv_percentile_value:.1f}) "
            f"diverge by {divergence:.1f}pt — spike contamination. Trust IVP over IVR."
        )
    elif contamination == ContaminationLevel.EXTREME:
        quality -= 50
        warnings.append(
            f"EXTREME divergence ({divergence:.1f}pt) IVR vs IVP. "
            f"52-week range contaminated by outlier — consider blocking."
        )

    if iv_rank_value is not None and robust is not None:
        robust_div = abs(float(iv_rank_value) - float(robust))
        if robust_div > 30.0:
            quality -= 20
            warnings.append(
                f"Robust IVR ({robust:.1f}) deviates from raw IVR "
                f"({iv_rank_value:.1f}) by {robust_div:.1f}pt — outliers in 52-week range."
            )

    break_info = detect_structural_break(iv_series) if iv_series is not None else None
    if break_info is not None:
        days_since = int(break_info["days_since_break"])
        magnitude_pct = float(break_info["magnitude"]) * 100.0
        direction = break_info["direction"]
        if days_since < 90:
            quality -= 30
            warnings.append(
                f"Recent structural break {break_info['break_date']} ({days_since}d ago). "
                f"IV mean shifted {direction} {magnitude_pct:.0f}%. "
                f"Pre-break history is not representative of current regime."
            )
        elif days_since < 180:
            quality -= 15
            warnings.append(
                f"Structural break {days_since}d ago ({break_info['break_date']}). "
                f"Treat IV metrics with caution."
            )

    if insufficient:
        quality = 0
        warnings.append("Insufficient IV history (< 100 valid observations).")

    quality = max(0, min(100, int(quality)))

    if quality >= 70:
        recommendation = "TRADE"
    elif quality >= 40:
        recommendation = "CAUTION"
    else:
        recommendation = "BLOCK"

    return IVQualityReport(
        ticker=ticker,
        iv_rank=iv_rank_value,
        iv_percentile=iv_percentile_value,
        robust_iv_rank=robust,
        contamination=contamination,
        divergence=divergence,
        structural_break=break_info,
        quality_score=quality,
        tradable=(recommendation != "BLOCK"),
        recommendation=recommendation,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# IVQualityGate — TickerGate integration layer
# ---------------------------------------------------------------------------

class IVQualityGate:
    """
    Fetches 252d of IV history via yfinance, runs assess_iv_quality(),
    and returns TRADE / CAUTION / BLOCK per ticker.

    Caches results for _IV_TTL_SECONDS to avoid redundant fetches
    within the same scan session. Fail-open: if yfinance is unavailable
    or the ticker has < 252d of history, returns CAUTION (not BLOCK)
    so the pipeline degrades gracefully rather than silently blocking
    all tickers on a data outage.

    Usage in TickerGate:
        gate = IVQualityGate(block_on_block=True, block_on_caution=False)
        ok, report = gate.check(ticker)
        if not ok:
            # ticker is BLOCK quality — skip
    """

    def __init__(
        self,
        *,
        block_on_block: bool = True,
        block_on_caution: bool = False,
        lookback_days: int = 252,
        cache_ttl: int = _IV_TTL_SECONDS,
    ):
        self.block_on_block   = block_on_block
        self.block_on_caution = block_on_caution
        self.lookback_days    = lookback_days
        self.cache_ttl        = cache_ttl
        self._cache: dict[str, tuple[float, IVQualityReport]] = {}

    def check(self, ticker: str) -> tuple[bool, Optional[IVQualityReport]]:
        """
        Returns (allowed: bool, report: IVQualityReport | None).

        allowed=True  → ticker passes the IV quality gate (TRADE or CAUTION)
        allowed=False → ticker is BLOCK quality, skip for this session
        report=None   → data unavailable, fail-open (allowed=True)
        """
        # Cache hit
        cached = self._cache.get(ticker)
        if cached is not None:
            ts, report = cached
            if time.time() - ts < self.cache_ttl:
                return self._decision(report), report

        if not _cb.is_available("yfinance_iv_quality"):
            logger.debug("[IVQuality] Circuit breaker OPEN for %s — fail-open", ticker)
            return True, None

        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            hist = t.history(period=f"{self.lookback_days + 30}d", interval="1d")
            if hist.empty or len(hist) < 100:
                logger.debug("[IVQuality] %s: insufficient price history — fail-open", ticker)
                _cb.record_success("yfinance_iv_quality")
                return True, None

            # Use realised vol as IV proxy (annualised 30d rolling)
            # yfinance doesn't provide IV directly — we use HV as the series
            # for structural break and robust rank computation.
            # When the orchestrator has actual IV from the options chain,
            # it can call assess_iv_quality() directly with real IV data.
            closes = hist["Close"].dropna()
            returns = closes.pct_change().dropna()
            hv_30d = returns.rolling(30).std() * (252 ** 0.5) * 100
            iv_series = hv_30d.dropna()

            if len(iv_series) < 100:
                _cb.record_success("yfinance_iv_quality")
                return True, None

            # Compute standard IVR and IVP from the HV series as proxies
            current_hv = float(iv_series.iloc[-1])
            window = iv_series.iloc[-self.lookback_days:]
            iv_rank_val = (
                (current_hv - float(window.min())) /
                (float(window.max()) - float(window.min())) * 100.0
                if float(window.max()) != float(window.min()) else None
            )
            iv_pct_val = float((window < current_hv).mean() * 100.0)

            _cb.record_success("yfinance_iv_quality")
            report = assess_iv_quality(ticker, iv_series, iv_rank_val, iv_pct_val)
            self._cache[ticker] = (time.time(), report)

            if report.recommendation != "TRADE":
                logger.info(
                    "[IVQuality] %s: %s (score=%d divergence=%.1f) — %s",
                    ticker, report.recommendation, report.quality_score,
                    report.divergence,
                    " | ".join(report.warnings) if report.warnings else "no warnings",
                )

            return self._decision(report), report

        except Exception as exc:
            _cb.record_failure("yfinance_iv_quality", str(exc))
            logger.warning("[IVQuality] %s: fetch failed (%s) — fail-open", ticker, exc)
            return True, None

    def _decision(self, report: IVQualityReport) -> bool:
        if report.recommendation == "BLOCK" and self.block_on_block:
            return False
        if report.recommendation == "CAUTION" and self.block_on_caution:
            return False
        return True


__all__ = [
    "ContaminationLevel",
    "IVQualityReport",
    "IVQualityGate",
    "robust_iv_rank",
    "detect_contamination",
    "detect_structural_break",
    "assess_iv_quality",
]
