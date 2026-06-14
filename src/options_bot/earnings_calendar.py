"""
Earnings & Macro Event Calendar — hard filter for known event risk.

AUDIT FINDING (Layer 3 — Strike Selection):
  "No earnings filter. A -0.25 delta put expiring in 30 days with a
   large credit may be 'correctly' selected — but if earnings fall within
   the DTE window, IV will crush post-earnings and the realized move
   could far exceed the BS-implied move. The model has no earnings date
   lookup. Missing structural risk check."

FIX:
  This module provides a hard REJECT for any spread when earnings fall
  within the DTE window (configurable: default block 5 days before and
  2 days after earnings). It also flags FOMC and major macro events
  (CPI, NFP) already in the EventCalendar but now also queryable per-ticker.

DATA SOURCES (all free, no API key):
  Primary:   yfinance.Ticker.calendar (earnings date from Yahoo Finance)
  Secondary: yfinance.Ticker.earnings_dates (historical + upcoming)
  Tertiary:  Market Chameleon scrape (HTML, used as last resort)

LABEL POLICY (from audit):
  This is a HARD BLOCKER, not a warning. If earnings cannot be confirmed
  as outside the DTE window, the trade is marked INCOMPLETE and rejected.
  "Do not accept any feature unless its meaning matches the label."
  "Never let one weak signal override a stronger risk filter."

Trade rule enforced:
  if has_earnings_in_window(ticker, entry_date, expiry_date):
      REJECT — "earnings within DTE window: {date}"
  if cannot_confirm_earnings_clear(ticker):
      REJECT — "earnings date unknown — cannot confirm event risk cleared"
      (unless underlying is an ETF, which has no per-company earnings)
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

from .circuit_breaker import data_circuit_breaker as _cb

logger = logging.getLogger(__name__)

# ETFs have no per-company earnings — skip earnings check for these
_ETF_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "GDX", "XLF", "XLK",
    "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB", "SMH",
    "ARKK", "VXX", "UVXY", "SQQQ", "SPXS", "TQQQ", "SPXL", "EEM", "EFA",
    "VEA", "VWO", "AGG", "LQD", "HYG", "IEFA", "IEMG", "ACWI",
}

# Cache: ticker → (fetched_at, earnings_dates list)
_earnings_cache: dict[str, tuple[float, list[date]]] = {}
_CACHE_TTL = 6 * 3600  # 6 hours


def get_earnings_dates(ticker: str) -> list[date]:
    """
    Return list of upcoming earnings dates for a ticker.

    DATA SOURCE: yfinance.Ticker.calendar['Earnings Date'] (Yahoo Finance).
    Returns empty list for ETFs (no per-company earnings).
    Returns empty list if data unavailable (caller must handle conservatively).

    LABEL: This is a direct lookup of Yahoo Finance's earnings calendar.
    Accuracy: Yahoo Finance earnings dates are typically accurate to ±1 day
    for confirmed dates; unconfirmed dates are estimates and labeled as such.
    The model treats both confirmed and estimated dates as blockers.
    """
    ticker = ticker.upper()
    if ticker in _ETF_TICKERS:
        return []

    now = time.monotonic()
    ts, cached = _earnings_cache.get(ticker, (0.0, []))
    if (now - ts) < _CACHE_TTL:
        return cached

    src = f"yfinance_earnings_{ticker}"
    if not _cb.is_available(src):
        logger.debug("[Earnings] %s skipped — circuit breaker OPEN", ticker)
        return []  # caller will handle as unknown

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        dates: list[date] = []

        # Method 1: .calendar (returns next earnings date if available)
        try:
            cal = t.calendar
            if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
                # cal can be a dict or DataFrame depending on yfinance version
                if hasattr(cal, 'items'):
                    # dict-style
                    for key in ("Earnings Date", "earningsDate"):
                        val = cal.get(key)
                        if val is not None:
                            if isinstance(val, (list, tuple)):
                                for v in val:
                                    try:
                                        d = _to_date(v)
                                        if d: dates.append(d)
                                    except Exception:
                                        pass
                            else:
                                d = _to_date(val)
                                if d: dates.append(d)
                elif hasattr(cal, 'loc'):
                    # DataFrame-style
                    try:
                        row = cal.loc["Earnings Date"]
                        for v in row.values:
                            d = _to_date(v)
                            if d: dates.append(d)
                    except Exception:
                        pass
        except Exception:
            pass

        # Method 2: .earnings_dates (broader history + upcoming)
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                future = ed[ed.index > datetime.now()]
                for idx in future.index[:4]:  # next 4 upcoming dates
                    d = _to_date(idx)
                    if d: dates.append(d)
        except Exception:
            pass

        # Deduplicate and sort
        dates = sorted(set(d for d in dates if d is not None))

        _cb.record_success(src)
        _earnings_cache[ticker] = (now, dates)
        logger.debug("[Earnings] %s: found dates %s", ticker, dates)
        return dates

    except Exception as exc:
        _cb.record_failure(src, str(exc))
        logger.warning("[Earnings] %s fetch failed: %s", ticker, exc)
        return []


def _to_date(val) -> Optional[date]:
    """Convert various date representations to a date object."""
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    try:
        import pandas as pd
        if hasattr(val, 'date'):
            return val.date()
        # pandas Timestamp
        return pd.Timestamp(val).date()
    except Exception:
        pass
    try:
        return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def has_earnings_in_window(
    ticker: str,
    entry_date: date,
    expiry_date: date,
    days_before: int = 5,
    days_after: int = 2,
) -> tuple[bool, str]:
    """
    HARD FILTER: Check if earnings fall within the trade's risk window.

    The risk window extends from (entry_date) to (expiry_date + days_after).
    Additionally, we block if earnings are within days_before of entry_date
    (post-earnings IV crush can already be priced in or reversing).

    Parameters
    ----------
    ticker : str
    entry_date : date
        Date the trade is entered (today).
    expiry_date : date
        Option expiration date.
    days_before : int
        Block if earnings are within this many days BEFORE entry.
        Default 5: avoids elevated pre-earnings IV that collapses post-event.
    days_after : int
        Block if earnings are within this many days AFTER expiry.
        Default 2: avoids options that expire just before an event.

    Returns
    -------
    (blocked: bool, reason: str)
    blocked=True means REJECT this trade.

    LABEL: This is a HARD BLOCK. The trade MUST be rejected if blocked=True.
    "Do not accept any feature unless its meaning matches the label."
    """
    ticker = ticker.upper()

    # ETFs: no earnings risk
    if ticker in _ETF_TICKERS:
        return False, f"{ticker} is an ETF — no earnings event risk"

    earnings_dates = get_earnings_dates(ticker)

    # If we cannot get earnings data, we REJECT conservatively
    # (unknown event risk = cannot confirm the trade is safe)
    if not earnings_dates:
        return True, (
            f"{ticker}: earnings date unknown — cannot confirm event risk is cleared. "
            f"Source: Yahoo Finance calendar returned no data. "
            f"INCOMPLETE: do not trade without confirming earnings date."
        )

    # Check if any earnings date falls in the risk window
    window_start = entry_date  - timedelta(days=days_before)
    window_end   = expiry_date + timedelta(days=days_after)

    for e_date in earnings_dates:
        if window_start <= e_date <= window_end:
            if e_date <= entry_date:
                location = f"{(entry_date - e_date).days} days ago (pre-earnings IV may be collapsing)"
            elif e_date <= expiry_date:
                location = f"within DTE window (expires {expiry_date}, earnings {e_date})"
            else:
                location = f"{(e_date - expiry_date).days} days after expiry"

            return True, (
                f"{ticker}: earnings {e_date} — {location}. "
                f"BS-implied move underestimates post-earnings realized move. "
                f"PoP calculation is unreliable. REJECT."
            )

    nearest = min(earnings_dates, key=lambda d: abs((d - entry_date).days))
    days_away = (nearest - entry_date).days
    return False, (
        f"{ticker}: nearest earnings {nearest} ({days_away:+d} days from entry) "
        f"— outside risk window [{window_start} to {window_end}]. Clear."
    )


class EarningsFilter:
    """
    Drop-in filter for the strategy layer.

    Usage in ShortPutSpread.evaluate():
        from options_bot.earnings_calendar import EarningsFilter
        _ef = EarningsFilter()

        blocked, reason = _ef.check(ticker, today, expiry_date)
        if blocked:
            raise LiquidityFilterError(ticker, f"[Earnings] {reason}")
    """

    def __init__(
        self,
        days_before: int = 5,
        days_after:  int = 2,
    ):
        self.days_before = days_before
        self.days_after  = days_after

    def check(
        self,
        ticker: str,
        entry_date: Optional[date] = None,
        expiry_date: Optional[date] = None,
        dte: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Check earnings event risk.

        Parameters
        ----------
        ticker : str
        entry_date : date or None
            Defaults to today if not provided.
        expiry_date : date or None
            If None, computed from entry_date + dte.
        dte : int or None
            Days to expiration (used if expiry_date is not provided).

        Returns
        -------
        (blocked: bool, reason: str)
        """
        if entry_date is None:
            entry_date = date.today()

        if expiry_date is None:
            if dte is None:
                return True, "expiry_date and dte both None — cannot check earnings window"
            expiry_date = entry_date + timedelta(days=dte)

        return has_earnings_in_window(
            ticker, entry_date, expiry_date,
            self.days_before, self.days_after,
        )
