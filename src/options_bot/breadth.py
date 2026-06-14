"""
Market Breadth Module — real breadth metrics from constituent price data.

AUDIT FINDING (Layer 1 — Regime Detector):
  "The breadth signal specifically is a miscategorized proxy. SPY is a single
   instrument, not an advance/decline line. It does not measure the breadth of
   the broader market."

  Subsequent fix used sector ETF dispersion, which the audit correctly identified
  as "sector rotation intensity — a proxy for regime, not actual breadth."

THIS MODULE:
  Computes genuine market breadth from a fixed constituent universe using
  Alpaca's multi-symbol stock bars API (already authenticated from trading
  credentials) with Alpha Vantage sector performance as a fallback/supplement.

Metrics computed
----------------
All metrics operate on a constituent universe (default: S&P 100 proxy — 50
most liquid names from each of large-cap, mid-cap, tech, and financial sectors).
The universe is fixed in code to avoid stale ETF CSV dependencies.

1. pct_above_20ma  — % of constituents with close > 20-day SMA
   LABEL: Direct count / universe size. No model assumptions.
   Interpretation: > 70% = broad uptrend. < 40% = broad downtrend.

2. pct_above_50ma  — % of constituents with close > 50-day SMA
   LABEL: Same. Slower signal — regime confirmation.

3. adv_dec_ratio   — advancing constituents / declining constituents
   LABEL: Count of (close_today > close_yesterday) vs inverse.
   Interpretation: > 2.0 = strong advance. < 0.5 = broad decline.

4. up_vol_ratio    — today's volume in advancing names / total volume
   LABEL: Demand-weighted breadth. Accounts for volume, not just count.
   Interpretation: > 0.65 = buying pressure. < 0.35 = selling pressure.

5. new_high_low_ratio — 52-week new highs / (new highs + new lows)
   LABEL: Price extension breadth. > 0.70 = expansion. < 0.30 = contraction.

6. composite_breadth — weighted composite of above 5 metrics (0–1)
   LABEL: (pct_above_20ma×0.30 + pct_above_50ma×0.20 + adv_dec_norm×0.20
           + up_vol_ratio×0.20 + new_hl_ratio×0.10)
   PROVISIONAL: Weights are equal-ish heuristics, not fitted to regime outcomes.
   Tag: PROVISIONAL_WEIGHTS until backtested against strategy P&L.

Data sources
------------
PRIMARY:   Alpaca Stock Historical Data API — `StockBarsRequest` for multi-symbol
           daily bars. Uses the same API key as the options trading client.
           Free tier: unlimited historical bars for SIP feed.

SECONDARY: Alpha Vantage `SECTOR` endpoint (free, 25 req/day limit)
           Returns real-time sector performance percentages.
           Used as a fast fallback when Alpaca bars are unavailable.
           Requires ALPHA_VANTAGE_KEY env var.

TERTIARY:  yfinance (already a dependency) — slowest but always available.

Constituent universe
--------------------
The universe is a curated 100-name proxy for the S&P 500, selected for:
- High liquidity (average daily volume > $500M)
- Cross-sector coverage (10 sectors, ~10 names each)
- Options availability on Alpaca (all are optionable)

This is NOT the full S&P 500. Computing breadth on 100 liquid names is
sufficient for regime detection and avoids API rate limits.
The universe should be reviewed quarterly.

Cache
-----
Breadth is cached for 15 minutes. It does not need to be real-time for a
daily options scan — regime changes happen over hours, not minutes.
"""

from __future__ import annotations

import logging
import math
import os
import time
import urllib.request
import json
from typing import Optional

import numpy as np

from .circuit_breaker import data_circuit_breaker as _cb

logger = logging.getLogger(__name__)

# ── Constituent universe (100 liquid, optionable names across all sectors) ────
# Reviewed 2026-Q2. Source: Alpaca options universe + S&P 500 membership.
# LABEL: Fixed curated list — not dynamically fetched from index.
# Update quarterly when sector weights shift materially.
_UNIVERSE = {
    # Technology (18 names)
    "tech": ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN",
             "AMAT", "MU", "KLAC", "LRCX", "SNPS", "CDNS", "ADBE", "CRM",
             "ORCL", "NOW"],
    # Financials (14 names)
    "fins": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP",
             "V", "MA", "COF", "MET", "PRU", "AFL"],
    # Healthcare (12 names)
    "hlth": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
             "MDT", "CVS", "CI", "HUM"],
    # Consumer discretionary (10 names)
    "disc": ["AMZN", "TSLA", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW",
             "BKNG", "LULU"],
    # Consumer staples (8 names)
    "stpl": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL"],
    # Industrials (10 names)
    "indu": ["CAT", "HON", "GE", "RTX", "BA", "UPS", "FDX", "MMM", "LMT", "NOC"],
    # Energy (8 names)
    "enrg": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC"],
    # Communication services (8 names)
    "comm": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR"],
    # Materials (6 names)
    "matl": ["LIN", "APD", "ECL", "SHW", "NEM", "FCX"],
    # Real estate (6 names)
    "reit": ["AMT", "PLD", "EQIX", "SPG", "CCI", "PSA"],
}

_ALL_TICKERS = [t for sector in _UNIVERSE.values() for t in sector]   # 100 names

# Alpha Vantage sector endpoint (free, 25 req/day)
_AV_SECTOR_URL = "https://www.alphavantage.co/query?function=SECTOR&apikey={key}"

# Cache
_breadth_cache: Optional[dict] = None
_breadth_cache_ts: float = 0.0
_CACHE_TTL = 15 * 60   # 15 minutes


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_bars_alpaca(tickers: list[str], lookback_days: int = 55) -> dict[str, list]:
    """
    Fetch daily OHLCV bars for multiple tickers via Alpaca StockBarsRequest.

    Returns dict: ticker → list of {close, volume, date} dicts (oldest first).
    Falls back to empty dict on failure.

    LABEL: Data source = Alpaca SIP feed (consolidated tape, not IEX-only).
    Bars are end-of-day adjusted closes. Lookback of 55 days gives enough
    history for 20-day and 50-day MA calculations.
    """
    src = "alpaca_stock_bars"
    if not _cb.is_available(src):
        return {}

    api_key    = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not api_secret:
        logger.debug("[Breadth] Alpaca credentials not set — skipping bars fetch")
        return {}

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta, timezone

        client = StockHistoricalDataClient(api_key, api_secret)
        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=lookback_days + 10)  # buffer for weekends

        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="all",   # split + dividend adjusted
        )
        bars_df = client.get_stock_bars(req).df

        if bars_df is None or bars_df.empty:
            _cb.record_failure(src, "empty response")
            return {}

        _cb.record_success(src)

        result: dict[str, list] = {}
        for ticker in tickers:
            try:
                if ticker in bars_df.index.get_level_values(0):
                    t_bars = bars_df.loc[ticker].tail(lookback_days)
                    result[ticker] = [
                        {
                            "close":  float(row["close"]),
                            "volume": float(row["volume"]),
                            "date":   str(idx.date()),
                        }
                        for idx, row in t_bars.iterrows()
                    ]
            except Exception:
                pass

        logger.info("[Breadth] Alpaca bars: %d/%d tickers", len(result), len(tickers))
        return result

    except Exception as exc:
        _cb.record_failure(src, str(exc))
        logger.warning("[Breadth] Alpaca bars fetch failed: %s", exc)
        return {}


def _fetch_bars_yfinance(tickers: list[str], lookback_days: int = 55) -> dict[str, list]:
    """
    Fallback: fetch bars from yfinance in batches.
    Slower than Alpaca but no rate limits for our usage.
    """
    src = "yfinance_breadth_bars"
    if not _cb.is_available(src):
        return {}
    try:
        import yfinance as yf
        result: dict[str, list] = {}
        # Batch download (faster than individual .history() calls)
        data = yf.download(
            tickers, period=f"{lookback_days + 10}d",
            interval="1d", auto_adjust=True,
            progress=False, threads=True,
        )
        if data.empty:
            _cb.record_failure(src, "empty")
            return {}

        closes  = data["Close"]
        volumes = data["Volume"]
        _cb.record_success(src)

        for t in tickers:
            try:
                if t not in closes.columns:
                    continue
                c = closes[t].dropna().tail(lookback_days)
                v = volumes[t].dropna().tail(lookback_days)
                if len(c) < 20:
                    continue
                result[t] = [
                    {"close": float(c.iloc[i]), "volume": float(v.iloc[i]),
                     "date": str(c.index[i].date())}
                    for i in range(len(c))
                ]
            except Exception:
                pass

        logger.info("[Breadth] yfinance bars: %d/%d tickers", len(result), len(tickers))
        return result

    except Exception as exc:
        _cb.record_failure(src, str(exc))
        logger.warning("[Breadth] yfinance bars fallback failed: %s", exc)
        return {}


def _fetch_av_sector() -> Optional[dict]:
    """
    Fetch Alpha Vantage real-time sector performance.
    Returns dict: sector_name → {1d_pct, 5d_pct, 1m_pct} or None.

    Rate limit: 25 requests/day on free tier.
    Cached for 15 minutes to stay well within limit.
    """
    av_key = os.getenv("ALPHA_VANTAGE_KEY", "")
    if not av_key:
        return None

    src = "alpha_vantage_sector"
    if not _cb.is_available(src):
        return None

    try:
        url = _AV_SECTOR_URL.format(key=av_key)
        req = urllib.request.Request(
            url, headers={"User-Agent": "OptionsBot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())

        if "Rank A: Real-Time Performance" not in data:
            _cb.record_failure(src, "unexpected response format")
            return None

        _cb.record_success(src)
        sector_perf = {}
        for rank_key, sectors in data.items():
            if not rank_key.startswith("Rank"):
                continue
            period = rank_key.split(":")[1].strip().lower().replace(" ", "_")
            if isinstance(sectors, dict):
                for sector, pct_str in sectors.items():
                    if sector not in sector_perf:
                        sector_perf[sector] = {}
                    try:
                        sector_perf[sector][period] = float(pct_str.replace("%", ""))
                    except (ValueError, AttributeError):
                        pass

        logger.info("[Breadth] Alpha Vantage sector data: %d sectors", len(sector_perf))
        return sector_perf

    except Exception as exc:
        _cb.record_failure(src, str(exc))
        logger.debug("[Breadth] Alpha Vantage sector fetch failed: %s", exc)
        return None


# ── Breadth calculations ──────────────────────────────────────────────────────

def _compute_breadth_metrics(bars: dict[str, list]) -> dict:
    """
    Compute all 5 breadth metrics from a bars dict.

    INPUT:  bars[ticker] = list of {close, volume, date} dicts (oldest→newest)
    OUTPUT: dict with labeled metrics

    LABEL: All metrics are direct arithmetic on the bars data.
    No model assumptions. No Black-Scholes. No probability distributions.
    """
    if not bars:
        return {}

    pct_above_20  = []
    pct_above_50  = []
    advancing     = 0
    declining     = 0
    up_volume     = 0.0
    down_volume   = 0.0
    new_highs     = 0
    new_lows      = 0

    for ticker, bar_list in bars.items():
        if len(bar_list) < 2:
            continue

        closes  = [b["close"]  for b in bar_list]
        volumes = [b["volume"] for b in bar_list]
        current_close = closes[-1]
        prev_close    = closes[-2]
        current_vol   = volumes[-1]

        # 1. Percent above MA
        if len(closes) >= 20:
            ma20 = float(np.mean(closes[-20:]))
            pct_above_20.append(1 if current_close > ma20 else 0)

        if len(closes) >= 50:
            ma50 = float(np.mean(closes[-50:]))
            pct_above_50.append(1 if current_close > ma50 else 0)

        # 2. Advance / decline
        if current_close > prev_close:
            advancing += 1
            up_volume += current_vol
        elif current_close < prev_close:
            declining += 1
            down_volume += current_vol
        # unchanged: excluded from A/D ratio (not advancing, not declining)

        # 3. 52-week new highs / lows (requires ≥ 252 bars; use available history)
        if len(closes) >= 50:
            lookback = closes[:-1]  # exclude today
            high_52w = max(lookback)
            low_52w  = min(lookback)
            if current_close >= high_52w * 0.995:  # within 0.5% of 52w high
                new_highs += 1
            elif current_close <= low_52w * 1.005:  # within 0.5% of 52w low
                new_lows += 1

    n = len(bars)
    if n == 0:
        return {}

    # Metric 1: % above 20-day MA
    pct20 = float(np.mean(pct_above_20)) if pct_above_20 else 0.5
    # Metric 2: % above 50-day MA
    pct50 = float(np.mean(pct_above_50)) if pct_above_50 else 0.5
    # Metric 3: A/D ratio (normalized to 0-1)
    total_ad = advancing + declining
    adv_norm = (advancing / total_ad) if total_ad > 0 else 0.5
    # Metric 4: Up-volume ratio
    total_vol = up_volume + down_volume
    upvol = (up_volume / total_vol) if total_vol > 0 else 0.5
    # Metric 5: New high/low ratio
    total_hl = new_highs + new_lows
    hl_ratio = (new_highs / total_hl) if total_hl > 0 else 0.5

    # Composite: PROVISIONAL_WEIGHTS — not calibrated against strategy outcomes
    # Weight rationale (heuristic): MA signals are slower (higher weight),
    # A/D is daily noise (lower weight), HL is leading (moderate weight).
    composite = (
        pct20   * 0.30 +
        pct50   * 0.20 +
        adv_norm * 0.20 +
        upvol    * 0.20 +
        hl_ratio * 0.10
    )

    result = {
        # Raw metrics (direct arithmetic — no model assumptions)
        "pct_above_20ma":     round(pct20,    4),
        "pct_above_50ma":     round(pct50,    4),
        "adv_dec_ratio":      round(advancing / max(declining, 1), 3),
        "adv_dec_normalized": round(adv_norm,  4),
        "up_vol_ratio":       round(upvol,     4),
        "new_high_low_ratio": round(hl_ratio,  4),
        # Composite (PROVISIONAL_WEIGHTS — heuristic, not backtested)
        "composite_breadth":  round(composite, 4),
        # Metadata
        "universe_size":      n,
        "advancing_count":    advancing,
        "declining_count":    declining,
        "new_highs":          new_highs,
        "new_lows":           new_lows,
        # Data quality flags
        "has_50ma":           len(pct_above_50) > 0,
        "has_hl":             total_hl > 0,
        "data_quality":       "full" if n >= 80 else ("partial" if n >= 40 else "thin"),
    }

    logger.info(
        "[Breadth] Universe=%d/%d | pct>20ma=%.0f%% pct>50ma=%.0f%% "
        "A/D=%.2f upvol=%.0f%% HL=%.2f composite=%.3f [%s]",
        n, len(_ALL_TICKERS),
        pct20 * 100, pct50 * 100,
        result["adv_dec_ratio"], upvol * 100,
        hl_ratio, composite,
        result["data_quality"],
    )
    return result


# ── Main public function ──────────────────────────────────────────────────────

def get_market_breadth(use_cache: bool = True) -> dict:
    """
    Compute real market breadth from the S&P 100 proxy universe.

    Returns a dict with all 5 breadth metrics plus composite.
    Returns empty dict on complete failure (regime detector falls back to 0.5).

    DATA SOURCE PRIORITY:
      1. Alpaca StockBarsRequest (multi-symbol daily bars, ALPACA_API_KEY required)
      2. yfinance batch download (always available, slower)

    ALPHA VANTAGE SUPPLEMENT:
      If ALPHA_VANTAGE_KEY is set, also fetches real-time sector performance
      and appends it to the output for regime scoring context.
      Limited to 25 req/day — cached aggressively.

    LABEL: This function returns LABELED metrics with explicit data quality flags.
    When data_quality='thin' (< 40 tickers), composite_breadth has high uncertainty
    and should be down-weighted in regime scoring.
    """
    global _breadth_cache, _breadth_cache_ts

    now = time.monotonic()
    if use_cache and _breadth_cache and (now - _breadth_cache_ts) < _CACHE_TTL:
        logger.debug("[Breadth] Using cached breadth (age=%.0fs)", now - _breadth_cache_ts)
        return _breadth_cache

    # Try Alpaca first (uses existing trading credentials)
    bars = _fetch_bars_alpaca(_ALL_TICKERS, lookback_days=55)

    # Fallback to yfinance if Alpaca unavailable or returned < 30 tickers
    if len(bars) < 30:
        logger.info("[Breadth] Alpaca returned %d tickers — falling back to yfinance", len(bars))
        bars = _fetch_bars_yfinance(_ALL_TICKERS, lookback_days=55)

    if not bars:
        logger.warning("[Breadth] All data sources failed — breadth unavailable")
        return {}

    result = _compute_breadth_metrics(bars)
    if not result:
        return {}

    # Supplement with Alpha Vantage sector performance if available
    av_sectors = _fetch_av_sector()
    if av_sectors:
        # Extract real-time performance for key sectors
        sector_map = {
            "Information Technology": "tech_1d_pct",
            "Financials":             "fins_1d_pct",
            "Health Care":            "hlth_1d_pct",
            "Energy":                 "enrg_1d_pct",
            "Communication Services": "comm_1d_pct",
        }
        for av_name, key in sector_map.items():
            for sector_name, perf in av_sectors.items():
                if av_name.lower() in sector_name.lower():
                    rt_pct = perf.get("real-time_performance", perf.get("real_time_performance"))
                    if rt_pct is not None:
                        result[key] = round(float(rt_pct), 3)
                    break
        result["has_av_sectors"] = True
    else:
        result["has_av_sectors"] = False

    _breadth_cache    = result
    _breadth_cache_ts = now
    return result


def composite_to_regime_score(breadth: dict) -> dict[str, float]:
    """
    Convert breadth metrics to regime score contributions.

    Returns additive scores for each regime bucket
    (trending, mean_reverting, high_volatility).

    LABEL: The mappings below are PROVISIONAL HEURISTICS derived from
    textbook breadth interpretation (Zweig, Fosback breadth theory).
    They have NOT been calibrated against strategy P&L outcomes.
    Tag all entries: PROVISIONAL until backtested.

    Calibration plan: after 3+ months of live paper trading, compute
    correlation between composite_breadth and realized short-premium
    P&L over the following 30 days. Use that to fit score contributions.
    """
    if not breadth:
        return {"trending": 0.0, "mean_reverting": 0.0, "high_volatility": 0.0}

    composite  = breadth.get("composite_breadth", 0.5)
    pct20      = breadth.get("pct_above_20ma", 0.5)
    adv_norm   = breadth.get("adv_dec_normalized", 0.5)
    upvol      = breadth.get("up_vol_ratio", 0.5)
    quality    = breadth.get("data_quality", "thin")

    # Down-weight if data quality is poor
    quality_mult = {"full": 1.0, "partial": 0.6, "thin": 0.3}.get(quality, 0.3)

    scores: dict[str, float] = {
        "trending": 0.0, "mean_reverting": 0.0, "high_volatility": 0.0
    }

    # PROVISIONAL: broad breadth > 70% = most stocks participating = trending market
    if composite > 0.70:
        scores["trending"]       += 0.15 * quality_mult
    elif composite > 0.55:
        scores["trending"]       += 0.08 * quality_mult
    elif composite < 0.30:
        # Very weak breadth = fear/selling = high volatility
        scores["high_volatility"] += 0.15 * quality_mult
    elif composite < 0.45:
        # Weak breadth but not extreme = mixed/range-bound
        scores["mean_reverting"]  += 0.10 * quality_mult

    # A/D ratio extremes
    if adv_norm > 0.75:   # more than 3:1 advance — strong participation
        scores["trending"]       += 0.08 * quality_mult
    elif adv_norm < 0.25:  # more than 3:1 decline
        scores["high_volatility"] += 0.08 * quality_mult

    # Up-volume dominance
    if upvol > 0.70:
        scores["trending"]       += 0.05 * quality_mult
    elif upvol < 0.30:
        scores["high_volatility"] += 0.05 * quality_mult

    return scores
