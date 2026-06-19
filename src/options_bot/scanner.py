"""
Ticker scanner — pre-entry quality filters for the options pipeline.

Two complementary screens run before the strategy layer evaluates
any options chain. Both return per-ticker scores that the orchestrator
uses to filter the ticker list for that day's scan.

1. BullishScanner — technical composite score (SMA, RSI, MACD, ADX)
   Answers: "Is this ticker in a bullish technical condition right now?"
   Used by: CSP and ShortPutSpread (don't sell puts on bearish tickers)

2. PiotroskiScorer — fundamental quality score (F-score, 0–9)
   Answers: "Is this company financially healthy?"
   Used by: Monthly scan (weekly reset) — filters out deteriorating companies
   Not used by: 0DTE (too slow, uses index ETFs)

Design
------
Both use yfinance — no additional API keys required. Both cache results
for their respective TTLs (technicals: 30 min, fundamentals: 7 days).
Both degrade gracefully — if data is unavailable, the ticker is allowed
through with a warning rather than blocking the scan.

Extracted from:
  trading_skills-main/src/trading_skills/scanner_bullish.py (technical)
  trading_skills-main/src/trading_skills/piotroski.py (fundamental)
Rewritten: no trading_skills dependency, integrated with our circuit breaker.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .volume_profile import volume_profile_cache, check_strike_safety
from .sec_signals import score_sec_signals, is_entry_confirmed

logger = logging.getLogger(__name__)

from .trendlines import analyze_trendlines, TrendlineResult
from .iv_quality import IVQualityGate, IVQualityReport
from .circuit_breaker import data_circuit_breaker as _cb_bb

import os
import json

# Cache TTLs
_TECH_TTL_SECONDS   = 30 * 60      # 30 minutes
_FUND_TTL_SECONDS   = 7 * 24 * 3600  # 7 days


# ---------------------------------------------------------------------------
# Bullish Technical Scanner
# ---------------------------------------------------------------------------

@dataclass
class TechnicalScore:
    ticker:        str
    score:         float          # 0.0 – 8.5 (higher = more bullish)
    sma_score:     float          # 0–2: above SMA20 (+1), above SMA50 (+1)
    rsi_score:     float          # 0–1: RSI 50–70 (+1), 30–50 (+0.5)
    macd_score:    float          # 0–1.8: above signal (+1.0), histogram rising (+0.5), magnitude (+0.3 max)
    adx_score:     float          # 0–1.5: ADX>25 with +DI>-DI (+1.5)
    trendline_score: float = 0.0  # 0–1.5: bullish pattern near support/breakout
    bb_score:      float = 0.0    # 0–1.5: Bollinger Band position (near lower band = best for puts)
    rsi:           Optional[float] = None
    adx:           Optional[float] = None
    bb_pct:        Optional[float] = None  # 0=at lower band, 1=at upper band
    trendline:     Optional[TrendlineResult] = None
    is_bullish:    bool = False   # score >= threshold
    computed_at:   datetime = None

    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now(tz=timezone.utc)
        self.is_bullish = self.score >= 4.5


class BullishScanner:
    """
    Computes a composite technical score for a ticker using SMA, RSI, MACD, ADX.

    Score components:
      Price above SMA20:           +1.0
      Price above SMA50:           +1.0
      RSI 50–70 (healthy bullish): +1.0
      RSI 30–50 (recovering):      +0.5
      MACD above signal:           +1.0
      MACD histogram rising:       +0.5
# Normalised hist magnitude: up to +0.3 (hist/price*100, capped at 1.0)
      ADX > 25 with +DI > -DI:     +1.5

    Max score: 6.0. Bullish threshold: 3.0 (configurable).

    Usage in the pipeline:
      Don't sell puts on a ticker with score < 3.0 — bearish technicals
      increase the chance the underlying drops through the strike.

    Source: trading_skills-main/scanner_bullish.py — rewritten without
    the trading_skills package dependency.
    """

    def __init__(self, bullish_threshold: float = 4.5, period: str = "3mo"):
        self.threshold = bullish_threshold
        self.period    = period
        self._cache: dict[str, tuple[float, TechnicalScore]] = {}  # ticker → (ts, score)

    def score(self, ticker: str) -> Optional[TechnicalScore]:
        """
        Compute bullish score for a ticker. Returns cached result if fresh.
        Returns None if data is unavailable (caller should allow the ticker through).
        """
        now = time.monotonic()
        cached_ts, cached_score = self._cache.get(ticker, (0.0, None))
        if cached_score is not None and (now - cached_ts) < _TECH_TTL_SECONDS:
            return cached_score

        result = self._compute(ticker)
        if result is not None:
            self._cache[ticker] = (now, result)
        return result

    def compute(self, ticker: str) -> "Optional[TechnicalScore]":
        """Public wrapper — returns full TechnicalScore or None."""
        return self.score(ticker)

    def is_entry_allowed(self, ticker: str) -> bool:
        """
        Returns True if the technical score allows a new short-put entry.
        Allows through if data is unavailable (non-fatal degradation).
        """
        s = self.score(ticker)
        if s is None:
            logger.debug("[BullishScanner] %s: no data — allowing entry", ticker)
            return True
        if not s.is_bullish:
            logger.info(
                "[BullishScanner] %s: score=%.1f < %.1f — blocking entry "
                "(RSI=%.0f ADX=%.0f SMA=%.0f)",
                ticker, s.score, self.threshold,
                s.rsi or 0, s.adx or 0, s.sma_score,
            )
            return False
        logger.debug("[BullishScanner] %s: score=%.1f — entry allowed", ticker, s.score)
        return True

    def _compute(self, ticker: str) -> Optional[TechnicalScore]:
        try:
            import yfinance as yf
            import numpy as np

            hist = yf.Ticker(ticker).history(period=self.period)
            if hist.empty or len(hist) < 50:
                logger.debug("[BullishScanner] %s: insufficient history", ticker)
                return None

            close = hist["Close"].values.flatten()
            high  = hist["High"].values.flatten()
            low   = hist["Low"].values.flatten()
            price = float(close[-1])

            # SMA
            sma20 = float(np.mean(close[-20:]))
            sma50 = float(np.mean(close[-50:]))
            sma_score = (1.0 if price > sma20 else 0.0) + (1.0 if price > sma50 else 0.0)

            # RSI (14-period)
            delta = np.diff(close[-30:])
            gain  = np.where(delta > 0, delta, 0.0)
            loss  = np.where(delta < 0, -delta, 0.0)
            ag = float(np.mean(gain[-14:]))
            al = float(np.mean(loss[-14:]))
            if al == 0:
                rsi = 100.0
            elif ag == 0:
                rsi = 0.0
            else:
                rsi = 100.0 - 100.0 / (1.0 + ag / al)

            if 50 <= rsi <= 70:
                rsi_score = 1.0
            elif 30 <= rsi < 50:
                rsi_score = 0.5
            else:
                rsi_score = 0.0

            # MACD (12/26/9) — proper EMA computation
            # EMA series helper: returns full array of EMA values
            def ema_series(arr: np.ndarray, n: int) -> np.ndarray:
                k = 2.0 / (n + 1)
                result = np.empty(len(arr))
                result[0] = arr[0]
                for i in range(1, len(arr)):
                    result[i] = arr[i] * k + result[i - 1] * (1 - k)
                return result

            # Need enough bars: 26 for EMA26 warmup + 9 for signal line + buffer
            if len(close) < 60:
                macd_score = 0.0
            else:
                ema12_arr  = ema_series(close[-60:], 12)
                ema26_arr  = ema_series(close[-60:], 26)
                macd_arr   = ema12_arr - ema26_arr

                # 9-period EMA of the MACD line = signal line (correct computation)
                signal_arr = ema_series(macd_arr, 9)

                # Use last two bars for current vs previous histogram
                macd_line  = macd_arr[-1]
                signal_now = signal_arr[-1]
                histogram  = macd_line - signal_now
                hist_prev  = macd_arr[-2] - signal_arr[-2]

                macd_score = 0.0
                if macd_line > signal_now:
                    macd_score += 1.0
                # Histogram rising: strictly > previous bar (not just > 0)
                if histogram > hist_prev:
                    macd_score += 0.5

                # Price-normalised histogram magnitude bonus (+0.3 max).
                # Adapted from Algo-Trader/strategy/scorer.py:
                # macd_hist / price * 100 makes histogram comparable across
                # tickers regardless of price level. A $1 hist on a $10 stock
                # (10%) is far more significant than on a $500 stock (0.2%).
                # Clip at 1.0 to prevent extreme values dominating the score.
                spot_price = float(close[-1]) if len(close) > 0 else 1.0
                if spot_price > 0:
                    hist_pct = abs(histogram) / spot_price * 100
                    macd_score += min(hist_pct, 1.0) * 0.3

            # ADX (14-period, Wilder-smoothed — true ADX not raw DX)
            # Raw DX is 2-3x more volatile than ADX and causes false positives.
            # True ADX = 14-period Wilder EMA of DX.
            adx_score = 0.0
            adx = None
            try:
                n = 14
                # Need 2*n bars minimum for meaningful ADX warmup
                if len(high) >= 2 * n + 1:
                    tr_arr = np.maximum(high[1:] - low[1:],
                             np.maximum(np.abs(high[1:] - close[:-1]),
                                        np.abs(low[1:]  - close[:-1])))
                    dm_plus_arr  = np.where(
                        (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                        np.maximum(high[1:] - high[:-1], 0.0), 0.0)
                    dm_minus_arr = np.where(
                        (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                        np.maximum(low[:-1] - low[1:], 0.0), 0.0)

                    # Wilder smoothing: seed with sum of first n bars,
                    # then apply: smoothed = prev * (n-1)/n + current
                    def wilder(arr, n):
                        s = float(np.sum(arr[:n]))
                        result = [s]
                        for v in arr[n:]:
                            s = s * (n - 1) / n + float(v)
                            result.append(s)
                        return result

                    atr_w  = wilder(tr_arr,       n)
                    dmp_w  = wilder(dm_plus_arr,  n)
                    dmm_w  = wilder(dm_minus_arr, n)

                    # Build DX series then Wilder-smooth into ADX
                    dx_series = []
                    di_plus_last = di_minus_last = 0.0
                    for i in range(len(atr_w)):
                        atr_i = atr_w[i]
                        if atr_i == 0:
                            dx_series.append(0.0)
                            continue
                        dip = 100 * dmp_w[i] / atr_i
                        dim = 100 * dmm_w[i] / atr_i
                        di_plus_last  = dip
                        di_minus_last = dim
                        di_sum = dip + dim
                        dx_series.append(100 * abs(dip - dim) / di_sum if di_sum > 0 else 0.0)

                    if len(dx_series) >= n:
                        adx_smooth = wilder(np.array(dx_series), n)
                        adx = float(adx_smooth[-1]) / n  # normalise back from Wilder sum

                        if adx > 25 and di_plus_last > di_minus_last:
                            adx_score = 1.5
            except Exception:
                pass

            # Trendline analysis — uses same OHLCV bars already in memory
            # Pure Python OLS on swing highs/lows — adds 0-1.5 to total score
            tl_result = None
            trendline_score = 0.0
            try:
                ohlcv = [
                    {"high": float(high[i]), "low": float(low[i]),
                     "close": float(close[i])}
                    for i in range(len(close))
                ]
                tl_result = analyze_trendlines(ohlcv)
                if tl_result is not None:
                    trendline_score = tl_result.trendline_score
                    logger.debug(
                        "[BullishScanner] %s trendline: %s (score=%.1f)",
                        ticker, tl_result.summary, trendline_score,
                    )
            except Exception as exc:
                logger.debug("[BullishScanner] %s trendline failed: %s", ticker, exc)

            # Bollinger Bands score — AV BBANDS with local numpy fallback
            # Near lower band = oversold + wide bands = elevated IV = best put premiums
            bb_score = 0.0
            bb_pct   = None
            try:
                av_key = os.getenv("ALPHA_VANTAGE_KEY", "")
                if av_key and _cb_bb.is_available("av_bbands"):
                    _bb_url = (
                        f"https://www.alphavantage.co/query"
                        f"?function=BBANDS&symbol={ticker}&interval=daily"
                        f"&time_period=20&series_type=close&nbdevup=2&nbdevdn=2"
                        f"&apikey={av_key}"
                    )
                    import urllib.request as _ur2, ssl as _ssl2
                    _ctx2 = _ssl2.create_default_context()
                    _req2 = _ur2.Request(_bb_url, headers={"User-Agent": "OptionsBot/1.0"})
                    with _ur2.urlopen(_req2, timeout=10, context=_ctx2) as _r2:
                        _bb_data = json.loads(_r2.read().decode("utf-8", errors="replace"))
                    _bb_series = _bb_data.get("Technical Analysis: BBANDS", {})
                    if _bb_series:
                        _latest = _bb_series[sorted(_bb_series.keys())[-1]]
                        _bb_upper = float(_latest.get("Real Upper Band", 0))
                        _bb_lower = float(_latest.get("Real Lower Band", 0))
                        _bb_width = _bb_upper - _bb_lower
                        if _bb_width > 0:
                            bb_pct = max(0.0, min(1.0, (price - _bb_lower) / _bb_width))
                        _cb_bb.record_success("av_bbands")
                else:
                    # Local fallback using last 20 closes already in memory
                    if len(close) >= 20:
                        import numpy as _np
                        _mid = float(_np.mean(close[-20:]))
                        _std = float(_np.std(close[-20:], ddof=1))
                        _w   = 4 * _std
                        if _w > 0:
                            bb_pct = max(0.0, min(1.0, (price - (_mid - 2*_std)) / _w))
                if bb_pct is not None:
                    if bb_pct <= 0.20:   bb_score = 1.5   # near lower band
                    elif bb_pct <= 0.40: bb_score = 0.8   # lower half
                    elif bb_pct <= 0.60: bb_score = 0.3   # near midline
                    logger.debug("[BullishScanner] %s BB: pct=%.2f score=%.1f", ticker, bb_pct, bb_score)
            except Exception as _bb_exc:
                _cb_bb.record_failure("av_bbands", str(_bb_exc))
                logger.debug("[BullishScanner] %s BB failed: %s", ticker, _bb_exc)

            total = sma_score + rsi_score + macd_score + adx_score + trendline_score + bb_score

            return TechnicalScore(
                ticker=ticker,
                score=round(total, 2),
                sma_score=sma_score,
                rsi_score=rsi_score,
                macd_score=macd_score,
                adx_score=adx_score,
                trendline_score=trendline_score,
                bb_score=bb_score,
                rsi=round(rsi, 1),
                adx=round(adx, 1) if adx is not None else None,
                bb_pct=round(bb_pct, 3) if bb_pct is not None else None,
                trendline=tl_result,
                is_bullish=total >= self.threshold,
            )

        except Exception as exc:
            logger.warning("[BullishScanner] %s: compute failed: %s", ticker, exc)
            return None


# ---------------------------------------------------------------------------
# Piotroski F-Score
# ---------------------------------------------------------------------------

@dataclass
class PiotroskiScore:
    ticker:      str
    f_score:     int     # 0–9 (higher = healthier fundamentals)
    # Profitability (0–4)
    roa_positive:   bool = False   # ROA > 0
    cfo_positive:   bool = False   # Cash from operations > 0
    roa_improving:  bool = False   # ROA improved year-over-year
    accrual_low:    bool = False    # CFO/assets > ROA (earnings quality)
    # Leverage / Liquidity (0–3)
    leverage_down:  bool = False   # Long-term debt ratio decreased
    current_up:     bool = False   # Current ratio improved
    no_dilution:    bool = False   # No new shares issued
    # Operating Efficiency (0–2)
    margin_up:      bool = False   # Gross margin improved
    turnover_up:    bool = False   # Asset turnover improved
    is_quality:     bool = False   # f_score >= 7
    computed_at:    datetime = None

    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now(tz=timezone.utc)
        self.is_quality = self.f_score >= 7


class PiotroskiScorer:
    """
    Computes the Piotroski F-score (0–9) from financial statement data.

    A score of 7–9 indicates a financially healthy company with improving
    fundamentals — good for selling puts (lower bankruptcy/blowup risk).
    A score of 0–2 is a red flag — avoid short puts on these.

    Scores update weekly (financial statements update quarterly but
    we recheck weekly to catch any data revisions).

    Source: trading_skills-main/piotroski.py — rewritten without the
    trading_skills package dependency.

    Note: For ETFs (SPY, QQQ, IWM) the F-score is not meaningful.
    The scorer returns None for ETFs and the caller should allow through.
    """

    # Tickers we know are ETFs — skip Piotroski for these
    _ETF_SKIP = {"SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLF",
                 "XLK", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU",
                 "XLRE", "XLB", "SMH", "ARKK", "VXX", "UVXY"}

    def __init__(self, quality_threshold: int = 7):
        self.threshold = quality_threshold
        self._cache: dict[str, tuple[float, Optional[PiotroskiScore]]] = {}

    def score(self, ticker: str) -> Optional[PiotroskiScore]:
        """
        Compute Piotroski F-score. Returns cached result if fresh (7 days).
        Returns None for ETFs or if data is unavailable.
        """
        if ticker.upper() in self._ETF_SKIP:
            return None

        now = time.monotonic()
        cached_ts, cached_score = self._cache.get(ticker, (0.0, None))
        if (now - cached_ts) < _FUND_TTL_SECONDS:
            return cached_score

        result = self._compute(ticker)
        self._cache[ticker] = (now, result)
        return result

    # Hard ETF bypass list — Piotroski requires income statements and
    # balance sheets that ETFs don't have. yfinance sometimes returns
    # partial data for ETFs that computes to a low F-score and blocks entry.
    # All ETFs in the universe plus common expansion tickers are listed here.
    _ETF_PASS = {
        "SPY","QQQ","IWM","DIA","MDY",
        "XLF","XLK","XLE","XLV","XLI","XLC","XLY","XLP","XLB","XLRE",
        "GLD","TLT","EEM","HYG","SMH",
        "VIXY","AGG","LQD","BND","VEA","VWO","EWJ",
        "XBI","XRT","XHB","USO","UNG","COPX","SLV","SOXX",
        "GDX","GDXJ","ARKK","ARKG","IAU","UVXY","VXX","TQQQ","SQQQ",
    }

    def is_entry_allowed(self, ticker: str) -> bool:
        """
        Returns True if Piotroski score allows a new position.

        ETFs are always allowed — they have no financial statements and
        any F-score computed from partial yfinance data is meaningless.
        """
        # Explicit ETF bypass — never block on Piotroski
        if ticker.upper().strip() in self._ETF_PASS:
            logger.debug("[Piotroski] %s: ETF — Piotroski bypassed", ticker)
            return True

        s = self.score(ticker)
        if s is None:
            return True   # no data — fail open
        if not s.is_quality:
            logger.info(
                "[Piotroski] %s: F-score=%d < %d — blocking entry "
                "(profitability=%d/4, leverage=%d/3, efficiency=%d/2)",
                ticker, s.f_score, self.threshold,
                sum([s.roa_positive, s.cfo_positive, s.roa_improving, s.accrual_low]),
                sum([s.leverage_down, s.current_up, s.no_dilution]),
                sum([s.margin_up, s.turnover_up]),
            )
            return False
        logger.debug("[Piotroski] %s: F-score=%d — entry allowed", ticker, s.f_score)
        return True

    def _compute(self, ticker: str) -> Optional[PiotroskiScore]:
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)

            # Balance sheet and income statement (annual)
            bs  = t.balance_sheet
            inc = t.income_stmt
            cf  = t.cashflow

            if bs is None or bs.empty or inc is None or inc.empty:
                return None
            if len(bs.columns) < 2:
                return None

            def _get(df, *keys) -> Optional[float]:
                for k in keys:
                    for col in df.index:
                        if k.lower() in str(col).lower():
                            try:
                                return float(df.loc[col].iloc[0])
                            except Exception:
                                pass
                return None

            def _get_prev(df, *keys) -> Optional[float]:
                for k in keys:
                    for col in df.index:
                        if k.lower() in str(col).lower():
                            try:
                                return float(df.loc[col].iloc[1])
                            except Exception:
                                pass
                return None

            # --- Profitability signals ---
            total_assets      = _get(bs, "Total Assets")
            total_assets_prev = _get_prev(bs, "Total Assets")
            net_income        = _get(inc, "Net Income")
            net_income_prev   = _get_prev(inc, "Net Income")
            cfo               = _get(cf, "Operating Cash Flow", "Cash From Operations")

            roa      = net_income / total_assets if total_assets and net_income else None
            roa_prev = net_income_prev / total_assets_prev \
                       if total_assets_prev and net_income_prev else None

            roa_positive  = roa > 0 if roa is not None else False
            cfo_positive  = cfo > 0 if cfo is not None else False
            roa_improving = (roa > roa_prev) if (roa is not None and roa_prev is not None) else False
            accrual_low   = ((cfo / total_assets) > roa) \
                            if (cfo and total_assets and roa is not None) else False

            # --- Leverage / Liquidity signals ---
            lt_debt      = _get(bs, "Long Term Debt")
            lt_debt_prev = _get_prev(bs, "Long Term Debt")
            curr_assets  = _get(bs, "Current Assets")
            curr_liab    = _get(bs, "Current Liabilities")
            curr_a_prev  = _get_prev(bs, "Current Assets")
            curr_l_prev  = _get_prev(bs, "Current Liabilities")
            shares       = _get(bs, "Common Stock", "Ordinary Shares Number")
            shares_prev  = _get_prev(bs, "Common Stock", "Ordinary Shares Number")

            leverage_now  = lt_debt / total_assets if (lt_debt and total_assets) else None
            leverage_prev = lt_debt_prev / total_assets_prev \
                            if (lt_debt_prev and total_assets_prev) else None
            current_now  = curr_assets / curr_liab if (curr_assets and curr_liab) else None
            current_prev = curr_a_prev / curr_l_prev if (curr_a_prev and curr_l_prev) else None

            leverage_down = (leverage_now < leverage_prev) \
                            if (leverage_now is not None and leverage_prev is not None) else False
            current_up    = (current_now > current_prev) \
                            if (current_now is not None and current_prev is not None) else False
            no_dilution   = (shares <= shares_prev) \
                            if (shares is not None and shares_prev is not None) else False

            # --- Operating Efficiency signals ---
            revenue      = _get(inc, "Total Revenue", "Revenue")
            revenue_prev = _get_prev(inc, "Total Revenue", "Revenue")
            cogs         = _get(inc, "Cost Of Revenue", "Cost Of Goods")
            cogs_prev    = _get_prev(inc, "Cost Of Revenue", "Cost Of Goods")

            gross_margin      = (revenue - cogs) / revenue \
                                if (revenue and cogs) else None
            gross_margin_prev = (revenue_prev - cogs_prev) / revenue_prev \
                                if (revenue_prev and cogs_prev) else None
            asset_turn        = revenue / total_assets if (revenue and total_assets) else None
            asset_turn_prev   = revenue_prev / total_assets_prev \
                                if (revenue_prev and total_assets_prev) else None

            margin_up   = (gross_margin > gross_margin_prev) \
                          if (gross_margin and gross_margin_prev) else False
            turnover_up = (asset_turn > asset_turn_prev) \
                          if (asset_turn and asset_turn_prev) else False

            f_score = sum([
                roa_positive, cfo_positive, roa_improving, accrual_low,
                leverage_down, current_up, no_dilution,
                margin_up, turnover_up,
            ])

            logger.info("[Piotroski] %s F-score=%d/9", ticker, f_score)

            return PiotroskiScore(
                ticker=ticker, f_score=f_score,
                roa_positive=roa_positive, cfo_positive=cfo_positive,
                roa_improving=roa_improving, accrual_low=accrual_low,
                leverage_down=leverage_down, current_up=current_up,
                no_dilution=no_dilution, margin_up=margin_up,
                turnover_up=turnover_up,
            )

        except Exception as exc:
            logger.warning("[Piotroski] %s: compute failed: %s", ticker, exc)
            return None


# ---------------------------------------------------------------------------
# Combined pre-entry gate
# ---------------------------------------------------------------------------

class TickerGate:
    """
    Combined technical + fundamental gate for the monthly options scan.

    Usage in orchestrator:
        gate = TickerGate()
        allowed_tickers = gate.filter(config.tickers)
        # Only scan options chain for allowed_tickers this cycle
    """

    def __init__(
        self,
        bullish_threshold: float = 2.0,  # lowered from 3.0 for ETF universe
        piotroski_threshold: int = 6,
        require_bullish: bool = True,
        require_piotroski: bool = True,
        require_iv_quality: bool = True,
        iv_block_on_caution: bool = False,
    ):
        # ETF universe uses 2.0 threshold — ETFs rarely score above 3.0
        # because broad-market funds have mixed sector technicals by design.
        # 2.0 means: price above at least one SMA + one bullish momentum signal.
        self.scanner   = BullishScanner(bullish_threshold)
        self.piotroski = PiotroskiScorer(piotroski_threshold)
        self.require_bullish    = require_bullish
        self.require_piotroski  = require_piotroski
        self.require_iv_quality = require_iv_quality
        self.iv_gate = IVQualityGate(
            block_on_block=require_iv_quality,
            block_on_caution=iv_block_on_caution,
        )

    def filter(self, tickers: list[str]) -> list[str]:
        """
        Returns the subset of tickers that pass all screens.
        Logs why each ticker was blocked.

        Screens (in order):
          1. IV quality gate — BLOCK tickers excluded (spike contamination,
             structural break, or insufficient history).
          2. BullishScanner  — bearish technicals excluded.
          3. PiotroskiScorer — weak fundamentals excluded.

        IV quality is checked first because it is the cheapest (cached 4h)
        and most likely to catch a data-integrity issue before we spend time
        on the options chain.
        """
        allowed = []
        for t in tickers:
            iv_ok, iv_report = self.iv_gate.check(t)
            if not iv_ok:
                logger.info(
                    "[TickerGate] %s blocked: IV quality BLOCK "
                    "(score=%d contamination=%s divergence=%.1f)",
                    t,
                    iv_report.quality_score if iv_report else 0,
                    iv_report.contamination.value if iv_report else "unknown",
                    iv_report.divergence if iv_report else 0.0,
                )
                continue

            tech_ok = (not self.require_bullish)   or self.scanner.is_entry_allowed(t)
            fund_ok = (not self.require_piotroski)  or self.piotroski.is_entry_allowed(t)

            if tech_ok and fund_ok:
                allowed.append(t)
                if iv_report and iv_report.recommendation == "CAUTION":
                    logger.warning(
                        "[TickerGate] %s passed but IV quality is CAUTION "
                        "(score=%d): %s",
                        t, iv_report.quality_score,
                        " | ".join(iv_report.warnings),
                    )
            else:
                reason = []
                if not tech_ok: reason.append("bearish technicals")
                if not fund_ok: reason.append("weak fundamentals")
                logger.info("[TickerGate] %s blocked: %s", t, " + ".join(reason))

        logger.info(
            "[TickerGate] %d/%d tickers passed pre-entry screens",
            len(allowed), len(tickers)
        )
        return allowed

    def filter_ranked(
        self,
        tickers: list[str],
        top_n: int = 10,
    ) -> list[str]:
        """
        Score all tickers, return top_n sorted by technical score (desc).
        Piotroski and IV quality are hard gates. Chain fetching only runs on winners.
        Logs: 'Shortlist: 10/20 tickers: SPY(5.8) QQQ(5.2)...'
        """
        scored: list[tuple[float, str]] = []
        for t in tickers:
            # IV quality hard gate
            iv_ok, iv_report = self.iv_gate.check(t)
            if not iv_ok:
                logger.info(
                    "[TickerGate] %s blocked: IV quality BLOCK (score=%d)",
                    t, iv_report.quality_score if iv_report else 0,
                )
                continue

            fund_ok = (not self.require_piotroski) or self.piotroski.is_entry_allowed(t)
            if not fund_ok:
                logger.info("[TickerGate] %s blocked: weak fundamentals", t)
                continue
            ts = self.scanner.compute(t) if self.require_bullish else None
            if ts is None:
                scored.append((0.0, t))
                continue
            if not ts.is_bullish:
                logger.info(
                    "[TickerGate] %s blocked: bearish technicals (score=%.1f)", t, ts.score
                )
                continue
            scored.append((ts.score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        shortlist = [t for _, t in scored[:top_n]]

        logger.info(
            "[TickerGate] Shortlist: %d/%d tickers (top %d by score): %s",
            len(shortlist), len(tickers), top_n,
            ", ".join(f"{t}({s:.1f})" for s, t in scored[:top_n]),
        )
        return shortlist

    def check_strike(
        self,
        ticker: str,
        short_strike: float,
        spot: float,
        spread_type: str = "bull_put",
    ) -> tuple[bool, str]:
        """
        Check if a short strike is safe relative to volume-based S/R levels.
        Wraps volume_profile.check_strike_safety() with per-ticker cache.

        Returns (safe: bool, reason: str).
        """
        profile = volume_profile_cache.get(ticker)
        return check_strike_safety(ticker, short_strike, spot, spread_type, profile=profile)

    def sec_confirmation(self, ticker: str) -> tuple[bool, str]:
        """
        Check SEC EDGAR for insider buying / activist confirmation.
        Non-blocking — returns (True, reason) when present, (False, reason) when absent.
        """
        return is_entry_confirmed(ticker)
