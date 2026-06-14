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

logger = logging.getLogger(__name__)

# Cache TTLs
_TECH_TTL_SECONDS   = 30 * 60      # 30 minutes
_FUND_TTL_SECONDS   = 7 * 24 * 3600  # 7 days


# ---------------------------------------------------------------------------
# Bullish Technical Scanner
# ---------------------------------------------------------------------------

@dataclass
class TechnicalScore:
    ticker:        str
    score:         float          # 0.0 – 5.5 (higher = more bullish)
    sma_score:     float          # 0–2: above SMA20 (+1), above SMA50 (+1)
    rsi_score:     float          # 0–1: RSI 50–70 (+1), 30–50 (+0.5)
    macd_score:    float          # 0–1.5: above signal (+1), histogram rising (+0.5)
    adx_score:     float          # 0–1.5: ADX>25 with +DI>-DI (+1.5)
    rsi:           Optional[float] = None
    adx:           Optional[float] = None
    is_bullish:    bool = False   # score >= threshold
    computed_at:   datetime = None

    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now(tz=timezone.utc)
        self.is_bullish = self.score >= 3.0


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
      ADX > 25 with +DI > -DI:     +1.5

    Max score: 6.0. Bullish threshold: 3.0 (configurable).

    Usage in the pipeline:
      Don't sell puts on a ticker with score < 3.0 — bearish technicals
      increase the chance the underlying drops through the strike.

    Source: trading_skills-main/scanner_bullish.py — rewritten without
    the trading_skills package dependency.
    """

    def __init__(self, bullish_threshold: float = 3.0, period: str = "3mo"):
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

            # MACD (12/26/9)
            def ema(arr: np.ndarray, n: int) -> float:
                k = 2.0 / (n + 1)
                e = float(arr[0])
                for v in arr[1:]:
                    e = float(v) * k + e * (1 - k)
                return e

            ema12 = ema(close[-40:], 12)
            ema26 = ema(close[-40:], 26)
            macd_line = ema12 - ema26

            # Signal = 9-period EMA of MACD (approximate from last two values)
            ema12_prev = ema(close[-41:-1], 12)
            ema26_prev = ema(close[-41:-1], 26)
            macd_prev  = ema12_prev - ema26_prev
            signal     = (macd_line * (2/10) + macd_prev * (8/10))
            histogram  = macd_line - signal
            hist_prev  = macd_prev - (macd_prev * (2/10) + macd_prev * (8/10))

            macd_score = 0.0
            if macd_line > signal:
                macd_score += 1.0
            if histogram > hist_prev:
                macd_score += 0.5

            # ADX (14-period, simplified)
            adx_score = 0.0
            adx = None
            try:
                n = 14
                if len(high) >= n + 1:
                    tr  = np.maximum(high[1:] - low[1:],
                          np.maximum(np.abs(high[1:] - close[:-1]),
                                     np.abs(low[1:]  - close[:-1])))
                    dm_plus  = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                                        np.maximum(high[1:] - high[:-1], 0), 0)
                    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                                        np.maximum(low[:-1] - low[1:], 0), 0)

                    atr14 = float(np.mean(tr[-n:]))
                    dmp14 = float(np.mean(dm_plus[-n:]))
                    dmm14 = float(np.mean(dm_minus[-n:]))

                    di_plus  = 100 * dmp14 / atr14 if atr14 > 0 else 0
                    di_minus = 100 * dmm14 / atr14 if atr14 > 0 else 0
                    di_sum   = di_plus + di_minus
                    dx       = 100 * abs(di_plus - di_minus) / di_sum if di_sum > 0 else 0
                    adx      = dx  # simplified (true ADX is smoothed DX)

                    if adx > 25 and di_plus > di_minus:
                        adx_score = 1.5
            except Exception:
                pass

            total = sma_score + rsi_score + macd_score + adx_score

            return TechnicalScore(
                ticker=ticker,
                score=round(total, 2),
                sma_score=sma_score,
                rsi_score=rsi_score,
                macd_score=macd_score,
                adx_score=adx_score,
                rsi=round(rsi, 1),
                adx=round(adx, 1) if adx is not None else None,
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

    def is_entry_allowed(self, ticker: str) -> bool:
        """
        Returns True if Piotroski score allows a new position.
        Allows through if ticker is an ETF or data is unavailable.
        """
        s = self.score(ticker)
        if s is None:
            return True   # ETF or unavailable — allow
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
        bullish_threshold: float = 3.0,
        piotroski_threshold: int = 6,
        require_bullish: bool = True,
        require_piotroski: bool = True,
    ):
        self.scanner   = BullishScanner(bullish_threshold)
        self.piotroski = PiotroskiScorer(piotroski_threshold)
        self.require_bullish   = require_bullish
        self.require_piotroski = require_piotroski

    def filter(self, tickers: list[str]) -> list[str]:
        """
        Returns the subset of tickers that pass both screens.
        Logs why each ticker was blocked.
        """
        allowed = []
        for t in tickers:
            tech_ok = (not self.require_bullish)  or self.scanner.is_entry_allowed(t)
            fund_ok = (not self.require_piotroski) or self.piotroski.is_entry_allowed(t)
            if tech_ok and fund_ok:
                allowed.append(t)
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
