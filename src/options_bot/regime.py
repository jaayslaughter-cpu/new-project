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
        Fetch VIX term structure using ^VIX (30-day) and ^VXV (3-month) as proxies.
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
            vxv_ticker = yf.Ticker("^VXV")
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
        """10Y - 2Y Treasury spread from FRED, falling back to yfinance TNX/IRX."""
        fred_key = os.getenv("FRED_API_KEY", "")

        if fred_key and _cb.is_available("fred"):
            try:
                from fredapi import Fred
                fred  = Fred(api_key=fred_key)
                dgs10 = fred.get_series("DGS10").dropna()
                dgs2  = fred.get_series("DGS2").dropna()
                if len(dgs10) > 0 and len(dgs2) > 0:
                    _cb.record_success("fred")
                    return float(dgs10.iloc[-1]) - float(dgs2.iloc[-1])
                _cb.record_failure("fred", "empty series")
            except Exception as exc:
                _cb.record_failure("fred", str(exc))
                logger.warning("[RegimeDetector] FRED yield curve failed: %s", exc)

        # Fallback: yfinance ^TNX / ^IRX
        if not _cb.is_available("yfinance_rates"):
            return None
        try:
            import yfinance as yf
            t10 = yf.Ticker("^TNX").fast_info.get("lastPrice")
            t2  = yf.Ticker("^IRX").fast_info.get("lastPrice")
            if t10 and t2:
                _cb.record_success("yfinance_rates")
                return float(t10) - float(t2)
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
