"""
Market Regime Detection.

Ported from TradeX (backend/engine/trading/regime_detector.py) with:
  - async removed (our orchestrator is synchronous APScheduler)
  - fmp_client replaced with yfinance (SPY historical prices for breadth/ADX)
  - fred_client replaced with fredapi + our existing Treasury rate pattern
  - FRED API key read from FRED_API_KEY env var (optional — falls back to
    yfinance ^VIX and a hardcoded yield-curve proxy if key is absent)
  - Cache TTL: 15 minutes (configurable)

Classifies current market environment as one of:
  trending       — low VIX, strong directional movement, rising market
  mean_reverting — mid VIX, choppy/sideways, low trend strength
  high_volatility — high or rising VIX, inverted yield curve

Strategy weights per regime (options allocation):
  high_volatility  → options = 0.35  (most favorable for premium selling)
  mean_reverting   → options = 0.15
  trending         → options = 0.10  (directional moves hurt short premium)

Usage:
    detector = RegimeDetector()
    result = detector.detect()
    # result = {
    #   "regime": "high_volatility",
    #   "confidence": 0.72,
    #   "options_weight": 0.35,
    #   "indicators": {"vix_level": 28.4, ...},
    #   "should_trade_options": True,
    # }
    if result["should_trade_options"]:
        # proceed with scan
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import Optional

from .hurst import hurst_exponent, classify_regime as hurst_classify, hurst_options_weight
from .circuit_breaker import data_circuit_breaker as _cb
from .breadth import get_market_breadth, composite_to_regime_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime → Risk Policy lookup table
# Ported from investor-bot/risk/regime_policy.py + extended for options
#
# Maps each regime string to a frozen policy that controls execution:
#   block_new_entries      — set True in crisis to halt all new trades
#   max_trades_per_scan    — cap new entries per scan cycle
#   min_confidence_boost   — extra confidence points required before entry
#   size_multiplier        — fraction of normal Kelly position size
#   favored_strategy       — which strategy to prefer in this regime
#
# These replace the scattered if/elif regime checks in orchestrator.py.
# The orchestrator calls get_regime_policy(regime_name) once and applies it.
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc


@_dc(frozen=True)
class RegimePolicy:
    """Execution constraints for a given market regime."""
    block_new_entries:    bool
    max_trades_per_scan:  int
    min_confidence_boost: int    # added to the 45-point threshold
    size_multiplier:      float  # applied on top of Kelly sizing
    favored_strategy:     str    # "short_put_spread" | "short_strangle" | "csp" | "any"


REGIME_POLICY: dict[str, RegimePolicy] = {
    # Full-risk regimes — mean-reverting environments are ideal for short premium
    "mean_reverting": RegimePolicy(
        block_new_entries=False,
        max_trades_per_scan=5,
        min_confidence_boost=0,
        size_multiplier=1.0,
        favored_strategy="short_put_spread",
    ),
    # Trending market: price makes sustained directional moves
    # Risk is directional blow-through on short puts → reduce size, prefer strangles
    "trending": RegimePolicy(
        block_new_entries=False,
        max_trades_per_scan=3,
        min_confidence_boost=5,     # need higher conviction
        size_multiplier=0.75,
        favored_strategy="short_strangle",  # collect both sides to be direction-neutral
    ),
    # High volatility: wide bid/ask, large moves — reduce exposure significantly
    "high_volatility": RegimePolicy(
        block_new_entries=False,
        max_trades_per_scan=2,
        min_confidence_boost=10,
        size_multiplier=0.50,
        favored_strategy="csp",     # CSP only — defined risk, single leg
    ),
    # Crisis / extreme vol: full halt
    "crisis": RegimePolicy(
        block_new_entries=True,
        max_trades_per_scan=0,
        min_confidence_boost=99,
        size_multiplier=0.0,
        favored_strategy="any",
    ),
    # VANNA_DOMINANT (Fed days, OPEX): reduce to half size, 0DTE disabled
    "vanna_dominant": RegimePolicy(
        block_new_entries=False,
        max_trades_per_scan=2,
        min_confidence_boost=8,
        size_multiplier=0.50,
        favored_strategy="short_put_spread",
    ),
    # Neutral/unknown fallback
    "neutral": RegimePolicy(
        block_new_entries=False,
        max_trades_per_scan=3,
        min_confidence_boost=0,
        size_multiplier=0.75,
        favored_strategy="any",
    ),
}

_DEFAULT_POLICY = REGIME_POLICY["neutral"]


def get_regime_policy(regime_name: str) -> RegimePolicy:
    """
    Return the RegimePolicy for a given regime string.

    Case-insensitive. Falls back to neutral policy if regime is unknown.
    Used by the orchestrator to apply execution constraints without
    scattered if/elif regime checks.
    """
    key = regime_name.lower().replace("-", "_").replace(" ", "_")
    # Partial match: "mean_reverting_breadth" → "mean_reverting"
    for k in REGIME_POLICY:
        if key.startswith(k) or k in key:
            return REGIME_POLICY[k]
    return _DEFAULT_POLICY

# Cache TTL — 15 minutes, same as TradeX default
_CACHE_TTL_SECONDS = 15 * 60

# Strategy weights per regime — from TradeX exactly
_TRENDING_WEIGHTS      = {"momentum": 0.50, "event_driven": 0.20, "statistical": 0.20, "options": 0.10}
_MEAN_REVERTING_WEIGHTS = {"momentum": 0.15, "event_driven": 0.20, "statistical": 0.50, "options": 0.15}
_HIGH_VOLATILITY_WEIGHTS = {"momentum": 0.10, "event_driven": 0.40, "statistical": 0.15, "options": 0.35}

# Minimum options weight to proceed with trading
_MIN_OPTIONS_WEIGHT_TO_TRADE = 0.10


class RegimeDetector:
    """
    Synchronous market regime classifier.

    Combines VIX level, VIX trend, yield curve slope, SPY breadth,
    and ADX-style trend strength into a regime score.

    All data fetches fail silently — if a source is unavailable,
    sensible defaults are used and the regime is still returned.
    """

    def __init__(self, cache_ttl_seconds: int = _CACHE_TTL_SECONDS):
        self._cache_ttl = cache_ttl_seconds
        self._cached: Optional[dict] = None
        self._cached_at: float = 0.0
        logger.info("[RegimeDetector] Initialized (cache_ttl=%ds)", cache_ttl_seconds)

    def detect(self) -> dict:
        """
        Detect and return the current market regime.

        Returns a dict with:
          regime:               str   — "trending" | "mean_reverting" | "high_volatility"
          confidence:           float — 0.33 to 0.95
          options_weight:       float — fraction of portfolio for options (from regime weights)
          should_trade_options: bool  — True if options_weight > MIN threshold
          indicators:           dict  — raw indicator values used for classification
          from_cache:           bool  — True if this is a cached result

        Never raises — returns a safe default on any failure.
        """
        # Return cached result if still fresh
        if self._cached and (time.monotonic() - self._cached_at < self._cache_ttl):
            logger.debug("[RegimeDetector] Returning cached regime: %s", self._cached["regime"])
            return {**self._cached, "from_cache": True}

        try:
            indicators = self._gather_indicators()
            regime, confidence = self._classify(indicators)
            weights = _weights_for_regime(regime)
            options_weight = weights["options"]

            result = {
                "regime": regime,
                "confidence": round(confidence, 3),
                "options_weight": options_weight,
                "should_trade_options": options_weight >= _MIN_OPTIONS_WEIGHT_TO_TRADE,
                "indicators": indicators,
                "strategy_weights": weights,
                "from_cache": False,
            }

            self._cached = result
            self._cached_at = time.monotonic()

            logger.info(
                "[RegimeDetector] Regime: %s (confidence=%.2f, options_weight=%.2f) "
                "VIX=%.1f trend=%s strength=%.2f curve=%.2f",
                regime, confidence, options_weight,
                indicators.get("vix_level", 0),
                indicators.get("vix_trend", "?"),
                indicators.get("trend_strength", 0),
                indicators.get("yield_curve_slope", 0),
            )
            return result

        except Exception as exc:
            logger.error("[RegimeDetector] Detection failed: %s — using safe default", exc)
            return _safe_default()

    def invalidate_cache(self) -> None:
        """Force next call to detect() to re-fetch all indicators."""
        self._cached = None
        self._cached_at = 0.0

    # ------------------------------------------------------------------
    # Indicator collection
    # ------------------------------------------------------------------

    def _gather_indicators(self) -> dict:
        """Fetch all indicators. Each sub-fetch fails independently."""
        vix_level      = self._fetch_vix_level()
        vix_trend      = self._compute_vix_trend()
        trend_strength = self._compute_trend_strength()
        yield_slope    = self._fetch_yield_curve_slope()
        hurst_val      = self._compute_hurst()
        vix_pct        = self._compute_vix_percentile(vix_level)
        vix_term       = self._fetch_vix_term_structure()

        # AUDIT FIX: Real market breadth from 100-name constituent universe
        # Replaces: (1) SPY up/down day proxy (mislabeled as breadth)
        #           (2) Sector ETF dispersion (proxy for rotation, not breadth)
        # Source: breadth.py using Alpaca bars + yfinance fallback
        # Data quality flag is propagated to scoring to down-weight thin data.
        breadth_data   = get_market_breadth()
        breadth_scores = composite_to_regime_score(breadth_data)
        stock_bond     = self._compute_stock_bond_divergence()
        dollar_stress  = self._compute_dollar_stress()

        return {
            "vix_level":          round(vix_level,      2) if vix_level      is not None else 20.0,
            "vix_trend":          vix_trend or "stable",
            "trend_strength":     round(trend_strength,  4) if trend_strength is not None else 0.5,
            "yield_curve_slope":  round(yield_slope,     4) if yield_slope    is not None else 0.5,
            "hurst":              round(hurst_val,       4) if hurst_val      is not None else 0.5,
            "hurst_regime":       hurst_classify(hurst_val) if hurst_val is not None else "random_walk",
            "vix_percentile":     round(vix_pct,         1) if vix_pct        is not None else 50.0,
            "vix_term_structure": vix_term.get("state", "unknown"),
            "vix_term_ratio":     vix_term.get("ratio", 1.0),
            # Real breadth metrics (direct arithmetic, labeled)
            "breadth_composite":  round(breadth_data.get("composite_breadth", 0.5), 4),
            "pct_above_20ma":     round(breadth_data.get("pct_above_20ma",    0.5), 4),
            "pct_above_50ma":     round(breadth_data.get("pct_above_50ma",    0.5), 4),
            "adv_dec_ratio":      round(breadth_data.get("adv_dec_ratio",     1.0), 3),
            "up_vol_ratio":       round(breadth_data.get("up_vol_ratio",      0.5), 4),
            "breadth_quality":    breadth_data.get("data_quality", "unknown"),
            # Pre-scored contributions (PROVISIONAL_WEIGHTS — see breadth.py)
            "_breadth_scores":    breadth_scores,
            # Stock-bond (SPY/TLT) divergence — flight-to-safety signal.
            # signal_trusted is gated on realized 20d correlation being
            # negative (ECB FSR Nov 2022 finding: this can break down in
            # high-inflation regimes) — see _compute_stock_bond_divergence.
            "big_blue_day":               stock_bond["big_blue_day"],
            "capitulation":                stock_bond["capitulation"],
            "stock_bond_correlation_20d":  stock_bond["stock_bond_correlation_20d"],
            "stock_bond_signal_trusted":   stock_bond["signal_trusted"],
            # Dollar stress (UUP) divergence — global funding-stress signal.
            # Informational/contextual weight only — see _compute_dollar_stress
            # docstring for why this doesn't have the same correlation-gate
            # rigor as the SPY/TLT signal above.
            "dollar_stress_day":          dollar_stress["dollar_stress_day"],
            "uup_spy_correlation_20d":    dollar_stress["uup_spy_correlation_20d"],
        }

    def _fetch_vix_level(self) -> Optional[float]:
        """
        Fetch current VIX using a 3-source fallback chain:
          1. CBOE direct CSV (authoritative, no rate limit, no auth required)
          2. Stooq daily CSV  (free, no auth)
          3. yfinance ^VIX   (original single source — now last resort)

        Using a fallback chain means a yfinance outage during a real volatility
        spike — exactly when VIX data is most critical — does not silently default
        to VIX=20.0 and allow new trades through the regime gate.
        """
        # ── Source 1: CBOE direct (authoritative) ────────────────────────────
        src1 = "cboe_vix"
        if _cb.is_available(src1):
            try:
                import csv as _csv
                from io import StringIO as _StringIO
                import urllib.request as _ur
                url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
                req = _ur.Request(url, headers={"User-Agent": "OptionsBot/1.0"})
                with _ur.urlopen(req, timeout=8) as r:
                    text = r.read().decode("utf-8", errors="ignore")
                rows = list(_csv.DictReader(_StringIO(text)))
                # Walk backward for most-recent valid close
                for row in reversed(rows):
                    c = row.get("CLOSE") or row.get("Close")
                    if c:
                        v = float(str(c).strip().replace(",", ""))
                        if v > 0:
                            _cb.record_success(src1)
                            logger.debug("[RegimeDetector] VIX from CBOE CSV: %.2f", v)
                            return v
                _cb.record_failure(src1, "no valid close in CSV")
            except Exception as exc:
                _cb.record_failure(src1, str(exc))
                logger.debug("[RegimeDetector] CBOE VIX failed: %s", exc)

        # ── Source 2: Stooq daily CSV ─────────────────────────────────────────
        src2 = "stooq_vix"
        if _cb.is_available(src2):
            try:
                import csv as _csv
                from io import StringIO as _StringIO
                import urllib.request as _ur
                url = "https://stooq.com/q/d/l/?s=%5Evix&i=d"
                req = _ur.Request(url, headers={"User-Agent": "OptionsBot/1.0"})
                with _ur.urlopen(req, timeout=8) as r:
                    text = r.read().decode("utf-8", errors="ignore")
                rows = list(_csv.DictReader(_StringIO(text)))
                if rows:
                    last = rows[-1]
                    c = last.get("Close") or last.get("close")
                    if c:
                        v = float(str(c).strip())
                        if v > 0:
                            _cb.record_success(src2)
                            logger.debug("[RegimeDetector] VIX from Stooq: %.2f", v)
                            return v
                _cb.record_failure(src2, "empty or unparseable")
            except Exception as exc:
                _cb.record_failure(src2, str(exc))
                logger.debug("[RegimeDetector] Stooq VIX failed: %s", exc)

        # ── Source 3: yfinance (original, now last resort) ────────────────────
        src3 = "yfinance_vix"
        if not _cb.is_available(src3):
            logger.warning("[RegimeDetector] All VIX sources unavailable — returning None")
            return None
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            price = vix.fast_info.get("lastPrice")
            if price and float(price) > 0:
                _cb.record_success(src3)
                logger.debug("[RegimeDetector] VIX from yfinance: %.2f", float(price))
                return float(price)
            hist = vix.history(period="5d")
            if not hist.empty:
                _cb.record_success(src3)
                return float(hist["Close"].iloc[-1])
            _cb.record_failure(src3, "empty response")
        except Exception as exc:
            _cb.record_failure(src3, str(exc))
            logger.warning("[RegimeDetector] All 3 VIX sources failed: %s", exc)
        return None

    def _fetch_vix_term_structure(self) -> dict:
        """
        Fetch VIX term structure using ^VIX (30-day) and ^VIX3M (3-month) as proxies.
        (^VIX3M was formerly ^VXV — CBOE renamed the ticker on 2017-09-18.)
        Adapted from trading-main/src/options/vix_monitor.py.

        Contango  (VXV > VIX, slope > 0): Normal market — IV expected to decay.
                  Theta works in our favor. Mild boost to mean_reverting score.
        Backwardation (VXV < VIX, slope < 0): Fear mode — volatility spike expected.
                  Penalises mean_reverting, boosts high_volatility.

        VXV/VIX ratio interpretation:
          > 1.05  strong contango   (very favorable for short premium)
          > 1.00  mild contango     (neutral-to-favorable)
          < 1.00  backwardation     (caution — reduce size or skip)
          < 0.95  strong backwardation (high_volatility regime signal)

        Returns dict with keys: state, vix30, vxv3m, slope, ratio.
        Non-fatal — returns {"state": "unknown"} on any failure.
        """
        cb_key = "vxv_term_structure"
        if not _cb.is_available(cb_key):
            return {"state": "unknown"}
        try:
            import yfinance as yf
            vix_ticker = yf.Ticker("^VIX")
            # AUDIT FIX: was "^VXV", which Yahoo no longer serves — CBOE
            # renamed the 3-month vol index ticker from VXV to VIX3M on
            # 2017-09-18. The stale ticker caused "possibly delisted; no
            # price data found" on every regime detection, silently
            # neutralizing the term-structure signal. ^VIX3M is the current
            # ticker. (Variable names left as vxv* internally to minimize
            # churn — only the fetched ticker symbol was wrong.)
            vxv_ticker = yf.Ticker("^VIX3M")
            vix_data = vix_ticker.history(period="2d")
            vxv_data = vxv_ticker.history(period="2d")
            if vix_data.empty or vxv_data.empty:
                _cb.record_failure(cb_key, "empty data")
                return {"state": "unknown"}
            vix30 = float(vix_data["Close"].iloc[-1])
            vxv3m = float(vxv_data["Close"].iloc[-1])
            slope = vxv3m - vix30
            ratio = vxv3m / vix30 if vix30 > 0 else 1.0
            if ratio > 1.0:
                state = "contango"
            elif ratio < 1.0:
                state = "backwardation"
            else:
                state = "flat"
            _cb.record_success(cb_key)
            logger.debug(
                "[RegimeDetector] VIX term structure: %s (VIX=%.1f VXV=%.1f ratio=%.3f)",
                state.upper(), vix30, vxv3m, ratio,
            )
            return {"state": state, "vix30": vix30, "vxv3m": vxv3m,
                    "slope": round(slope, 2), "ratio": round(ratio, 4)}
        except Exception as exc:
            _cb.record_failure(cb_key, str(exc))
            logger.debug("[RegimeDetector] VIX term structure fetch failed: %s", exc)
            return {"state": "unknown"}

    def _compute_vix_trend(self) -> Optional[str]:
        """
        Classify VIX as 'rising', 'falling', or 'stable'.
        Compares latest VIX to its 5-day SMA.
        """
        try:
            import yfinance as yf
            hist = yf.Ticker("^VIX").history(period="20d")
            if hist.empty or len(hist) < 3:
                return "stable"
            closes = hist["Close"].tolist()
            latest = closes[-1]
            sma5 = sum(closes[-5:]) / min(len(closes[-5:]), 5)
            pct = ((latest - sma5) / sma5 * 100) if sma5 > 0 else 0
            if pct > 10:
                return "rising"
            elif pct < -10:
                return "falling"
            return "stable"
        except Exception as exc:
            logger.warning("[RegimeDetector] VIX trend failed: %s", exc)
        return "stable"

    def _compute_market_breadth(self) -> Optional[float]:
        """DEPRECATED: Replaced by breadth.get_market_breadth(). Kept for reference only."""
        return None  # breadth.py now handles all breadth computation
        """
        AUDIT FIX: Previous implementation used SPY up/down days as a breadth
        proxy. AUDIT FINDING: "SPY is a single instrument, not an advance/decline
        line. It can mislead the regime detector."

        REPLACEMENT: Sector ETF dispersion — measures how spread apart the
        11 SPDR sector ETFs are in their 20-day returns. High dispersion =
        rotation/trending market. Low dispersion = mean-reverting/range-bound.

        OUTPUT: Dispersion ratio (0–1):
          0.0 = all sectors moving together (low dispersion, mean-reverting)
          1.0 = sectors highly divergent (high dispersion, trending/rotating)

        LABEL: This is a SECTOR DISPERSION measure, not advance/decline breadth.
        It measures cross-sector return variance, not individual stock advances.
        It is a proxy for market rotation intensity, which correlates with
        trending vs mean-reverting regimes. Labeled correctly in indicators dict
        as 'sector_dispersion' (renamed from 'breadth').
        """
        sector_etfs = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLC",
                       "XLY", "XLP", "XLU", "XLRE", "XLB"]
        src = "yfinance_sectors"
        if not _cb.is_available(src):
            return None
        try:
            import yfinance as yf
            import numpy as np

            data = yf.download(sector_etfs, period="35d", progress=False, auto_adjust=True)
            closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data

            if closes.empty or len(closes) < 20:
                _cb.record_failure(src, "insufficient data")
                return None

            # 20-day returns per sector
            recent = closes.tail(20)
            returns_20d = []
            for etf in sector_etfs:
                if etf in recent.columns:
                    col = recent[etf].dropna()
                    if len(col) >= 2 and float(col.iloc[0]) != 0:
                        r = float(col.iloc[-1]) / float(col.iloc[0]) - 1.0
                        returns_20d.append(r)

            if len(returns_20d) < 5:
                _cb.record_failure(src, "insufficient sectors")
                return None

            _cb.record_success(src)
            # Dispersion = std of cross-sector returns, normalized to [0,1]
            # Typical range: std < 0.02 = low dispersion, > 0.08 = high
            dispersion = float(np.std(returns_20d))
            normalized  = min(1.0, dispersion / 0.08)
            logger.debug(
                "[RegimeDetector] Sector dispersion: std=%.4f normalized=%.3f "
                "(%d sectors)", dispersion, normalized, len(returns_20d)
            )
            return round(normalized, 4)

        except Exception as exc:
            _cb.record_failure(src, str(exc))
            logger.warning("[RegimeDetector] Sector dispersion failed: %s", exc)
        return None

    def _compute_trend_strength(self) -> Optional[float]:
        """
        ADX-style trend strength (0-1) from SPY price data.
        Computes DX = |+DI - -DI| / (+DI + -DI) over last 20 bars.
        From TradeX — uses yfinance instead of FMP.
        """
        try:
            import yfinance as yf
            hist = yf.Ticker("SPY").history(period="40d")
            if hist.empty or len(hist) < 10:
                return 0.5

            bars = hist.reset_index()
            bars = bars.tail(21)   # 21 bars → 20 day-over-day calculations
            if len(bars) < 10:
                return 0.5

            plus_dm_sum = minus_dm_sum = tr_sum = 0.0

            for i in range(1, len(bars)):
                high_i   = float(bars["High"].iloc[i])
                low_i    = float(bars["Low"].iloc[i])
                close_p  = float(bars["Close"].iloc[i-1])
                high_p   = float(bars["High"].iloc[i-1])
                low_p    = float(bars["Low"].iloc[i-1])

                tr = max(high_i - low_i, abs(high_i - close_p), abs(low_i - close_p))
                tr_sum += tr

                up_move   = high_i - high_p
                down_move = low_p  - low_i

                if up_move > down_move and up_move > 0:
                    plus_dm_sum += up_move
                if down_move > up_move and down_move > 0:
                    minus_dm_sum += down_move

            if tr_sum == 0:
                return 0.5

            plus_di  = (plus_dm_sum  / tr_sum) * 100
            minus_di = (minus_dm_sum / tr_sum) * 100
            di_sum   = plus_di + minus_di
            if di_sum == 0:
                return 0.5

            dx = abs(plus_di - minus_di) / di_sum
            return float(min(1.0, max(0.0, dx)))

        except Exception as exc:
            logger.warning("[RegimeDetector] Trend strength failed: %s", exc)
        return None

    def _compute_stock_bond_divergence(self) -> dict:
        """
        Stock-bond (SPY/TLT) divergence flags — flight-to-safety signal.

        Source: empirically validated by an independent backtest study
        (Clarion `backtests/spy_tlt_signals`, 2002-2026 incl. GFC) — Sharpe
        0.65 vs SPY buy-and-hold 0.43, max drawdown cut from -55% to -14%.
        Two Tier-1 signals:
          big_blue_day — SPY 1d return < -1% AND TLT 1d return > +1%
          capitulation — SPY 1d return < -1% AND TLT red AND
                          SPY volume > 1.5x its trailing 20d average

        IMPORTANT CAVEAT (ECB Financial Stability Review, Nov 2022,
        "Cross-asset correlations in a more inflationary environment"):
        the stock-bond correlation that this signal's premise depends on
        (bonds rally when stocks fall) is NOT stable — it has trended
        positive during high-inflation regimes via the discount-rate
        channel, meaning stocks and bonds can sell off together instead
        of bonds providing the expected offset. To guard against trusting
        a broken-down relationship, this method also computes the trailing
        20-day realized SPY/TLT return correlation and only marks the
        signal "trusted" when that correlation is negative — i.e. the
        flight-to-safety mechanism is actually currently functioning.
        When correlation is positive, the divergence flags are still
        reported (for visibility/logging) but flagged untrusted and are
        not added to the regime score.

        Returns dict with: big_blue_day, capitulation, spy_1d_return,
        tlt_1d_return, spy_volume_ratio, stock_bond_correlation_20d,
        signal_trusted (bool).
        """
        _safe = {
            "big_blue_day": False, "capitulation": False,
            "spy_1d_return": 0.0, "tlt_1d_return": 0.0,
            "spy_volume_ratio": 1.0, "stock_bond_correlation_20d": None,
            "signal_trusted": False,
        }
        try:
            import yfinance as yf
            import numpy as np

            spy_hist = yf.Ticker("SPY").history(period="35d")
            tlt_hist = yf.Ticker("TLT").history(period="35d")
            if spy_hist.empty or tlt_hist.empty or len(spy_hist) < 22 or len(tlt_hist) < 22:
                return _safe

            spy_close = spy_hist["Close"]
            tlt_close = tlt_hist["Close"]
            spy_vol   = spy_hist["Volume"]

            spy_1d_return = float(spy_close.iloc[-1] / spy_close.iloc[-2] - 1.0)
            tlt_1d_return = float(tlt_close.iloc[-1] / tlt_close.iloc[-2] - 1.0)

            spy_vol_avg20 = float(spy_vol.iloc[-21:-1].mean())
            spy_vol_today = float(spy_vol.iloc[-1])
            spy_volume_ratio = (spy_vol_today / spy_vol_avg20) if spy_vol_avg20 > 0 else 1.0

            big_blue_day = (spy_1d_return < -0.01) and (tlt_1d_return > 0.01)
            capitulation = (
                (spy_1d_return < -0.01)
                and (tlt_1d_return < 0.0)
                and (spy_volume_ratio > 1.5)
            )

            # Trailing 20-day realized correlation — the ECB-informed gate.
            # Negative correlation = flight-to-safety mechanism is working
            # (bonds and stocks moving opposite directions, as the signal
            # assumes). Positive correlation = the relationship has broken
            # down (both sell off together) and the signal should not be
            # trusted, per the ECB's inflationary-regime finding.
            n = min(21, len(spy_close), len(tlt_close))
            spy_rets = spy_close.iloc[-n:].pct_change().dropna()
            tlt_rets = tlt_close.iloc[-n:].pct_change().dropna()
            common_len = min(len(spy_rets), len(tlt_rets))
            correlation_20d = None
            if common_len >= 10:
                corr_matrix = np.corrcoef(
                    spy_rets.iloc[-common_len:].values,
                    tlt_rets.iloc[-common_len:].values,
                )
                correlation_20d = float(corr_matrix[0, 1])

            signal_trusted = (correlation_20d is not None) and (correlation_20d < 0.0)

            return {
                "big_blue_day":               big_blue_day,
                "capitulation":                capitulation,
                "spy_1d_return":               round(spy_1d_return, 4),
                "tlt_1d_return":                round(tlt_1d_return, 4),
                "spy_volume_ratio":             round(spy_volume_ratio, 2),
                "stock_bond_correlation_20d":   round(correlation_20d, 3) if correlation_20d is not None else None,
                "signal_trusted":               signal_trusted,
            }
        except Exception as exc:
            logger.warning("[RegimeDetector] Stock-bond divergence failed: %s", exc)
        return _safe

    def _compute_dollar_stress(self) -> dict:
        """
        Dollar strength (UUP) divergence signal — global funding-stress flag.

        Source: classic macro signal, not a single backtested study like the
        SPY/TLT signal above. The dollar tends to spike during episodes of
        acute risk aversion / global dollar funding stress (investors and
        institutions scrambling for USD liquidity), often coinciding with
        equity selloffs. This is well-documented macro behaviour (e.g. 2008,
        March 2020) rather than a specific empirically-backtested edge —
        treat this flag as informational context for the regime score, not
        a precisely calibrated signal the way the SPY/TLT one is.

        Uses UUP (Invesco DB US Dollar Index Bullish Fund) as the proxy —
        raw DXY futures aren't reliably available via yfinance, UUP is a
        liquid, tradeable ETF that tracks dollar strength closely.

        dollar_stress_day — UUP 1d return > +0.5% AND SPY 1d return < -1%
                             (dollar spiking while equities sell off)

        Also computes the realized 20d UUP/SPY correlation for visibility/
        logging (no hard trust-gate applied here, unlike the ECB-informed
        gate on the stock-bond signal — there isn't a specific study backing
        a correlation threshold for this one, so it's reported but not used
        to suppress the flag).

        Returns dict with: dollar_stress_day, uup_1d_return, spy_1d_return,
        uup_spy_correlation_20d.
        """
        _safe = {
            "dollar_stress_day": False, "uup_1d_return": 0.0,
            "spy_1d_return": 0.0, "uup_spy_correlation_20d": None,
        }
        try:
            import yfinance as yf
            import numpy as np

            uup_hist = yf.Ticker("UUP").history(period="35d")
            spy_hist = yf.Ticker("SPY").history(period="35d")
            if uup_hist.empty or spy_hist.empty or len(uup_hist) < 22 or len(spy_hist) < 22:
                return _safe

            uup_close = uup_hist["Close"]
            spy_close = spy_hist["Close"]

            uup_1d_return = float(uup_close.iloc[-1] / uup_close.iloc[-2] - 1.0)
            spy_1d_return = float(spy_close.iloc[-1] / spy_close.iloc[-2] - 1.0)

            dollar_stress_day = (uup_1d_return > 0.005) and (spy_1d_return < -0.01)

            n = min(21, len(uup_close), len(spy_close))
            uup_rets = uup_close.iloc[-n:].pct_change().dropna()
            spy_rets = spy_close.iloc[-n:].pct_change().dropna()
            common_len = min(len(uup_rets), len(spy_rets))
            correlation_20d = None
            if common_len >= 10:
                corr_matrix = np.corrcoef(
                    uup_rets.iloc[-common_len:].values,
                    spy_rets.iloc[-common_len:].values,
                )
                correlation_20d = float(corr_matrix[0, 1])

            return {
                "dollar_stress_day":         dollar_stress_day,
                "uup_1d_return":             round(uup_1d_return, 4),
                "spy_1d_return":             round(spy_1d_return, 4),
                "uup_spy_correlation_20d":   round(correlation_20d, 3) if correlation_20d is not None else None,
            }
        except Exception as exc:
            logger.warning("[RegimeDetector] Dollar stress signal failed: %s", exc)
        return _safe

    def _compute_vix_percentile(self, current_vix: Optional[float], window_days: int = 252) -> Optional[float]:
        """
        Compute where the current VIX level sits within its own N-day history.

        Returns a percentile 0–100:
          VIX at 90th percentile → historically elevated → strong short-premium signal
          VIX at 20th percentile → historically suppressed → avoid selling premium

        More robust than raw VIX level: VIX=22 means very different things if
        the 1-year range is 12–45 vs 18–25.
        """
        if current_vix is None:
            return None
        if not _cb.is_available("yfinance_vix_hist"):
            return None
        try:
            import yfinance as yf
            from scipy.stats import percentileofscore
            hist = yf.Ticker("^VIX").history(period="2y")
            if hist.empty or len(hist) < 20:
                _cb.record_failure("yfinance_vix_hist", "insufficient history")
                return None
            _cb.record_success("yfinance_vix_hist")
            closes = hist["Close"].tolist()
            window = closes[-window_days:] if len(closes) >= window_days else closes
            pct = percentileofscore(window, current_vix)
            logger.debug("[RegimeDetector] VIX percentile=%.1f (VIX=%.2f, window=%d days)",
                         pct, current_vix, len(window))
            return float(pct)
        except Exception as exc:
            _cb.record_failure("yfinance_vix_hist", str(exc))
            logger.warning("[RegimeDetector] VIX percentile failed: %s", exc)
        return None

    def _compute_hurst(self) -> Optional[float]:
        """
        Compute the Hurst exponent for SPY over the last 252 trading days.
        H < 0.48 → mean-reverting (best for short premium)
        H > 0.52 → trending (dangerous for short premium)
        """
        try:
            import yfinance as yf
            hist = yf.Ticker("SPY").history(period="2y")
            if hist.empty or len(hist) < 50:
                return None
            closes = hist["Close"].values
            h = hurst_exponent(closes[-252:] if len(closes) >= 252 else closes)
            logger.debug("[RegimeDetector] Hurst=%.4f (%s)", h, hurst_classify(h))
            return h
        except Exception as exc:
            logger.warning("[RegimeDetector] Hurst computation failed: %s", exc)
        return None

    def _fetch_yield_curve_slope(self) -> Optional[float]:
        """
        10Y - 2Y Treasury spread. Three-tier fallback chain:

          1. FRED API (most accurate, requires FRED_API_KEY env var)
          2. US Treasury par yield curve XML (free, no key, daily updates)
             Source: https://home.treasury.gov/resource-center/data-chart-center/
                     interest-rates/ — same data FRED publishes, direct from source.
             Adapted from: OpenTrading/tools/macro/macro.py (_yield_rows / get_yields)
          3. yfinance ^TNX / ^IRX (last resort — delayed, less precise)
        """
        from datetime import datetime as _dt, timezone as _tz

        fred_key = os.getenv("FRED_API_KEY", "")

        # ── Tier 1: FRED API ──────────────────────────────────────────
        if fred_key and _cb.is_available("fred"):
            try:
                from fredapi import Fred
                fred  = Fred(api_key=fred_key)
                dgs10 = fred.get_series("DGS10").dropna()
                dgs2  = fred.get_series("DGS2").dropna()
                if len(dgs10) > 0 and len(dgs2) > 0:
                    _cb.record_success("fred")
                    slope = float(dgs10.iloc[-1]) - float(dgs2.iloc[-1])
                    logger.debug("[RegimeDetector] Yield curve via FRED: %.3f", slope)
                    return slope
                _cb.record_failure("fred", "empty series")
            except Exception as exc:
                _cb.record_failure("fred", str(exc))
                logger.warning("[RegimeDetector] FRED yield curve failed: %s", exc)

        # ── Tier 2: Treasury XML (free, no key) ───────────────────────
        if _cb.is_available("treasury_yield_xml"):
            try:
                import urllib.request as _ur
                import ssl as _ssl
                import xml.etree.ElementTree as _ET

                year = _dt.now(_tz.utc).year
                _YIELD_URL = (
                    "https://home.treasury.gov/resource-center/data-chart-center/"
                    "interest-rates/pages/xml?data=daily_treasury_yield_curve"
                    "&field_tdr_date_value={year}"
                )
                ctx = _ssl.create_default_context()
                req = _ur.Request(
                    _YIELD_URL.format(year=year),
                    headers={"User-Agent": "OptionsBot yield-curve/1.0 research@localhost"}
                )
                with _ur.urlopen(req, timeout=15, context=ctx) as r:
                    body = r.read().decode("utf-8", errors="replace")

                root = _ET.fromstring(body)

                def _local(tag: str) -> str:
                    return tag.rsplit("}", 1)[-1]

                rows = []
                for props in root.iter():
                    if _local(props.tag) != "properties":
                        continue
                    d = {_local(c.tag): (c.text or "").strip() for c in props}
                    if d.get("NEW_DATE") and d.get("BC_2YEAR") and d.get("BC_10YEAR"):
                        try:
                            rows.append({
                                "date": d["NEW_DATE"],
                                "y2":  float(d["BC_2YEAR"]),
                                "y10": float(d["BC_10YEAR"]),
                            })
                        except (ValueError, KeyError):
                            continue

                if rows:
                    rows.sort(key=lambda r: r["date"])
                    latest = rows[-1]
                    slope = latest["y10"] - latest["y2"]
                    _cb.record_success("treasury_yield_xml")
                    logger.debug(
                        "[RegimeDetector] Yield curve via Treasury XML: "
                        "2Y=%.2f 10Y=%.2f slope=%.3f (%s)",
                        latest["y2"], latest["y10"], slope, latest["date"],
                    )
                    return slope

                _cb.record_failure("treasury_yield_xml", "no valid rows parsed")

            except Exception as exc:
                _cb.record_failure("treasury_yield_xml", str(exc))
                logger.warning("[RegimeDetector] Treasury XML yield curve failed: %s", exc)

        # ── Tier 3: yfinance ^TNX / ^IRX (last resort) ────────────────
        if not _cb.is_available("yfinance_rates"):
            return None
        try:
            import yfinance as yf
            t10 = yf.Ticker("^TNX").fast_info.get("lastPrice")
            t2  = yf.Ticker("^IRX").fast_info.get("lastPrice")
            if t10 and t2:
                _cb.record_success("yfinance_rates")
                slope = float(t10) - float(t2)
                logger.debug("[RegimeDetector] Yield curve via yfinance: %.3f", slope)
                return slope
            _cb.record_failure("yfinance_rates", "missing TNX/IRX")
        except Exception as exc:
            _cb.record_failure("yfinance_rates", str(exc))
            logger.warning("[RegimeDetector] Yield curve fallback failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Classification — ported verbatim from TradeX _classify()
    # ------------------------------------------------------------------

    def _classify(self, indicators: dict) -> tuple[str, float]:
        """
        Score each regime and return (regime_name, confidence).
        Logic ported from TradeX, extended with Hurst exponent as fourth signal.
        """
        vix            = indicators.get("vix_level", 20.0)
        vix_trend      = indicators.get("vix_trend", "stable")
        trend_strength = indicators.get("trend_strength", 0.5)
        # Real breadth scores from constituent universe (PROVISIONAL_WEIGHTS — see breadth.py)
        breadth_scores = indicators.get("_breadth_scores", {})
        slope          = indicators.get("yield_curve_slope", 0.5)
        hurst          = indicators.get("hurst", 0.5)
        vix_pct        = indicators.get("vix_percentile", 50.0)

        # AUDIT NOTE: Score weights below are PROVISIONAL heuristics.
        # They have not been calibrated against historical trade outcomes.
        # Label: treat regime classification as a directional opinion,
        # not a statistically validated rule. Confidence reflects this.

        scores: dict[str, float] = {
            "trending":        0.0,
            "mean_reverting":  0.0,
            "high_volatility": 0.0,
        }

        # High-volatility signals
        if vix > 30:           scores["high_volatility"] += 0.50
        elif vix > 25:         scores["high_volatility"] += 0.35
        elif vix > 20:         scores["high_volatility"] += 0.10

        if vix_trend == "rising":   scores["high_volatility"] += 0.30
        elif vix_trend == "stable": scores["high_volatility"] += 0.05

        # Trending signals
        if vix < 15:           scores["trending"] += 0.30
        elif vix < 20:         scores["trending"] += 0.20
        elif vix < 25:         scores["trending"] += 0.05

        if trend_strength > 0.7:    scores["trending"] += 0.40
        elif trend_strength > 0.6:  scores["trending"] += 0.30
        elif trend_strength > 0.4:  scores["trending"] += 0.10

        if vix_trend == "falling":  scores["trending"] += 0.10

        # Mean-reverting signals
        if 15 <= vix <= 25:         scores["mean_reverting"] += 0.20
        if trend_strength < 0.3:    scores["mean_reverting"] += 0.35
        elif trend_strength < 0.4:  scores["mean_reverting"] += 0.25
        elif trend_strength < 0.5:  scores["mean_reverting"] += 0.10

        if vix_trend == "stable":   scores["mean_reverting"] += 0.15

        # Real market breadth — FINAL AUDIT FIX
        # Replaces: (1) SPY up/down proxy (2) sector ETF dispersion proxy
        # Source: breadth.py — 100-name constituent universe, Alpaca + yfinance bars
        # Scores are from composite_to_regime_score() — PROVISIONAL_WEIGHTS labeled
        # Quality-weighted: thin data (< 40 tickers) receives 30% of stated weight
        for regime_key, contrib in breadth_scores.items():
            if regime_key in scores:
                scores[regime_key] += contrib

        # Yield curve: inversion → higher volatility signal
        if slope < 0:         scores["high_volatility"] += 0.15
        elif slope > 1.0:     scores["trending"]         += 0.05

        # VIX term structure — 6th signal (adapted from trading-main VIXMonitor)
        # Contango (VXV > VIX): IV term premium is normal — theta decay works.
        # Backwardation (VXV < VIX): near-term fear spike — don't sell premium.
        vix_term_state = indicators.get("vix_term_structure", "unknown")
        vix_term_ratio = indicators.get("vix_term_ratio", 1.0)
        if vix_term_state == "backwardation":
            # Strong backwardation = high_vol signal; penalise mean_reverting
            if vix_term_ratio < 0.95:
                scores["high_volatility"] += 0.25   # Strong backwardation
                scores["mean_reverting"]  -= 0.15   # Not safe to sell premium
            else:
                scores["high_volatility"] += 0.10   # Mild backwardation
        elif vix_term_state == "contango":
            # Contango = normal environment, mild mean-reverting boost
            if vix_term_ratio > 1.05:
                scores["mean_reverting"]  += 0.10   # Strong contango
            else:
                scores["mean_reverting"]  += 0.05   # Mild contango

        # VIX percentile — 7th signal (where VIX sits in its own history)
        # High percentile = historically elevated vol = ideal for short premium
        # Low percentile = historically suppressed vol = avoid selling premium
        if vix_pct >= 80:
            scores["high_volatility"] += 0.20
        elif vix_pct >= 65:
            scores["high_volatility"] += 0.12
            scores["mean_reverting"]  += 0.05
        elif vix_pct <= 25:
            scores["trending"]        += 0.10   # calm market, directional
        elif vix_pct <= 40:
            scores["mean_reverting"]  += 0.08   # mid-range, choppy
        else:
            scores["mean_reverting"]  += 0.04   # neutral zone

        # Hurst exponent — 7th signal (was previously 5th, now 7th with VIX pct added)
        # Mean-reverting Hurst boosts mean_reverting regime
        # Trending Hurst boosts trending regime
        # Random walk Hurst is neutral (small boost to high_volatility as a hedge)
        if hurst < 0.40:
            scores["mean_reverting"]  += 0.20
        elif hurst < 0.48:
            scores["mean_reverting"]  += 0.12
        elif hurst > 0.60:
            scores["trending"]        += 0.20
        elif hurst > 0.52:
            scores["trending"]        += 0.12
        else:
            scores["high_volatility"] += 0.05  # random walk — slight uncertainty bump

        # Stock-bond (SPY/TLT) divergence — 8th signal.
        # capitulation is the stronger of the two flags (adds volume
        # confirmation on top of the basic divergence) — weighted higher.
        # Only contributes to the score when signal_trusted is True (i.e.
        # the realized 20d SPY/TLT correlation is actually negative right
        # now — see _compute_stock_bond_divergence docstring for why this
        # gate exists). An untrusted signal is still logged/visible in
        # indicators for monitoring, but does not move the regime score.
        if indicators.get("stock_bond_signal_trusted"):
            if indicators.get("capitulation"):
                scores["high_volatility"] += 0.20
            elif indicators.get("big_blue_day"):
                scores["high_volatility"] += 0.12

        # Dollar stress (UUP) divergence — 9th signal, lighter weight than
        # the SPY/TLT signal since this is general macro behaviour rather
        # than a specifically backtested edge (see _compute_dollar_stress
        # docstring). No correlation trust-gate — always contributes when
        # the flag fires.
        if indicators.get("dollar_stress_day"):
            scores["high_volatility"] += 0.10

        # Pick winner
        regime = max(scores, key=scores.__getitem__)
        top_score = scores[regime]
        total = sum(scores.values())
        confidence = (top_score / total) if total > 0 else 0.33
        confidence = max(0.33, min(0.95, confidence))

        return regime, confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights_for_regime(regime: str) -> dict[str, float]:
    if regime == "trending":
        return dict(_TRENDING_WEIGHTS)
    elif regime == "mean_reverting":
        return dict(_MEAN_REVERTING_WEIGHTS)
    elif regime == "high_volatility":
        return dict(_HIGH_VOLATILITY_WEIGHTS)
    return {"momentum": 0.25, "event_driven": 0.25, "statistical": 0.25, "options": 0.25}


def _safe_default() -> dict:
    """Return a permissive default when detection fails entirely."""
    return {
        "regime": "mean_reverting",
        "confidence": 0.33,
        "options_weight": 0.15,
        "should_trade_options": True,
        "indicators": {
            "vix_level": 20.0, "vix_trend": "stable",
            "breadth": 1.0, "trend_strength": 0.5, "yield_curve_slope": 0.5,
        },
        "strategy_weights": dict(_MEAN_REVERTING_WEIGHTS),
        "from_cache": False,
    }
