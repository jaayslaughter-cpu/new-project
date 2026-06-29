"""
options_bot/dividends.py
────────────────────────
Per-ticker dividend yield for Black-Scholes-Merton continuous dividend pricing.

Design
──────
The bot enriches every option row with a continuous dividend yield *q* so that
``bs_price`` / ``bs_greeks`` / ``solve_iv`` use the Merton formula rather than
the plain Black-Scholes formula where ``q = 0``.  For OTM short-premium
positions the effect is small on SPY/QQQ/IWM, but materially changes put
pricing for high-yielders like TLT (~3.5%) and HYG (~4.5%).

Two sources, in priority order:

1. **Live cache** (``refresh_dividend_yield``) – uses Alpaca's
   ``CorporateActionsClient`` to sum cash dividends paid over the trailing
   12 months, then divides by current mid-price.  Same pattern as
   ``gamma-scalping-main/market/dividends.py`` (MIT licence, adapted).
   Called explicitly; never called automatically at scan time.

2. **Static table** (``DIVIDEND_YIELDS``) – calibrated trailing-12m yields
   as of June 2026.  Used at scan time (zero API overhead).  Refreshed by
   running ``refresh_dividend_yield`` offline and updating the table.

``get_dividend_yield(ticker)`` returns from the live cache when available,
falling back to the static table, then 0.0 for unknown tickers.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Static calibrated table ────────────────────────────────────────────────
# Trailing-12-month yield as of 2026-06.
# VXX and GLD pay no dividends.  SMH/XLK/XBI are low-yield growth ETFs.
# TLT and HYG are the highest-impact names — delta and theta change notably.

DIVIDEND_YIELDS: Dict[str, float] = {
    "SPY":  0.013,   # S&P 500 — ~1.3% TTM
    "QQQ":  0.005,   # Nasdaq-100 — ~0.5% TTM
    "IWM":  0.012,   # Russell 2000 — ~1.2% TTM
    "TLT":  0.035,   # 20+ yr Treasury — ~3.5% TTM
    "XLF":  0.020,   # Financials — ~2.0% TTM
    "XLK":  0.007,   # Technology — ~0.7% TTM
    "XLE":  0.032,   # Energy — ~3.2% TTM
    "XLV":  0.015,   # Health Care — ~1.5% TTM
    "XLI":  0.018,   # Industrials — ~1.8% TTM
    "GLD":  0.000,   # Gold — no dividend
    "EEM":  0.025,   # EM equities — ~2.5% TTM
    "HYG":  0.045,   # High-yield bonds — ~4.5% TTM
    "SMH":  0.007,   # Semis — ~0.7% TTM
    "VXX":  0.000,   # VIX futures — no dividend
    "XBI":  0.003,   # Biotech — ~0.3% TTM
}

# ── Live cache (populated by refresh_dividend_yield) ─────────────────────
# Maps ticker → (yield_decimal, fetched_at).  TTL = 24 h.
_LIVE_CACHE: Dict[str, Tuple[float, datetime.datetime]] = {}
_CACHE_TTL_HOURS = 24


def get_dividend_yield(ticker: str) -> float:
    """
    Return the annualised continuous dividend yield for *ticker*.

    Priority:
      1. Live cache (if populated and not stale).
      2. Static ``DIVIDEND_YIELDS`` table.
      3. 0.0 (safe default — equivalent to plain Black-Scholes).
    """
    now = datetime.datetime.utcnow()
    if ticker in _LIVE_CACHE:
        q, fetched_at = _LIVE_CACHE[ticker]
        age_hours = (now - fetched_at).total_seconds() / 3600
        if age_hours < _CACHE_TTL_HOURS:
            return q

    return DIVIDEND_YIELDS.get(ticker, 0.0)


def refresh_dividend_yield(
    ticker: str,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> float:
    """
    Fetch a live trailing-12-month dividend yield from Alpaca and update
    the in-process cache.

    Adapted from gamma-scalping-main/market/dividends.py (MIT licence).
    Uses ``CorporateActionsClient`` to sum cash dividends, then divides by
    the current mid-price from ``StockHistoricalDataClient``.

    Parameters
    ----------
    ticker    : str   — e.g. "SPY"
    api_key   : str   — defaults to ALPACA_API_KEY env var
    secret_key: str   — defaults to ALPACA_SECRET_KEY env var

    Returns
    -------
    float — dividend yield as decimal (e.g. 0.013 = 1.3%).
    Falls back to the static table entry on any error.
    """
    api_key    = api_key    or os.getenv("ALPACA_API_KEY", "")
    secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        logger.warning(
            "[Dividends] No API keys available for live refresh of %s — "
            "using static table.",
            ticker,
        )
        return DIVIDEND_YIELDS.get(ticker, 0.0)

    try:
        from alpaca.data.historical.corporate_actions import CorporateActionsClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.enums import CorporateActionsType
        from alpaca.data.requests import (
            CorporateActionsRequest,
            StockLatestQuoteRequest,
        )

        actions_client = CorporateActionsClient(api_key, secret_key)
        stock_client   = StockHistoricalDataClient(api_key, secret_key)

        end_date   = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=365)

        actions_req = CorporateActionsRequest(
            symbols=[ticker],
            types=[CorporateActionsType.CASH_DIVIDEND],
            start=start_date,
            end=end_date,
        )
        actions = actions_client.get_corporate_actions(actions_req)
        cash_divs = actions.data.get("cash_dividends", [])

        if not cash_divs:
            logger.info(
                "[Dividends] No cash dividends found for %s — yield = 0.0", ticker
            )
            q = 0.0
        else:
            quote_req     = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            latest_quote  = stock_client.get_stock_latest_quote(quote_req)
            ask = latest_quote[ticker].ask_price
            bid = latest_quote[ticker].bid_price
            current_price = (ask + bid) / 2.0

            dividends_ttm = sum(d.rate for d in cash_divs)

            if current_price <= 0:
                logger.warning(
                    "[Dividends] Zero price for %s — cannot compute yield", ticker
                )
                q = DIVIDEND_YIELDS.get(ticker, 0.0)
            else:
                q = dividends_ttm / current_price
                logger.info(
                    "[Dividends] %s: TTM dividends=$%.4f, price=$%.2f → yield=%.4f",
                    ticker, dividends_ttm, current_price, q,
                )

    except Exception as exc:
        logger.warning(
            "[Dividends] Live refresh failed for %s (%s) — using static table.",
            ticker, exc,
        )
        q = DIVIDEND_YIELDS.get(ticker, 0.0)

    _LIVE_CACHE[ticker] = (q, datetime.datetime.utcnow())
    return q


def refresh_all(
    tickers: Optional[list] = None,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    Refresh live yields for all tickers in the universe (or a subset).
    Returns a dict of {ticker: yield}.  Safe to call from a script; never
    called automatically by the live bot scheduler.
    """
    tickers = tickers or list(DIVIDEND_YIELDS.keys())
    results: Dict[str, float] = {}
    for ticker in tickers:
        results[ticker] = refresh_dividend_yield(ticker, api_key, secret_key)
    return results
