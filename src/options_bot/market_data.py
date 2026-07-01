"""
Market data ingestion via yfinance.

This is the free data layer. It pulls live option chains from Yahoo Finance
and returns typed OptionChainRow objects ready for the Greeks enrichment layer.

Limitations vs ThetaData / ORATS:
  - Snapshot only (no historical chains)
  - No intraday bid/ask history
  - Greeks not provided (we calculate them in greeks.py)
  - Quote delay ~15 min during market hours
  - IV provided by Yahoo but we recalculate it ourselves for accuracy

Usage:
    from options_bot.market_data import YFinanceDataLoader

    loader = YFinanceDataLoader(ticker="SPY")
    chain = loader.get_chain(expiry="2026-07-18")          # one expiration
    full  = loader.get_full_chain()                        # all expirations
    filtered = loader.get_chain_filtered(                  # with liquidity pre-filter
        expiry="2026-07-18",
        min_open_interest=100,
        max_spread_pct=0.20,
        min_delta=0.10,
        max_delta=0.40,
    )
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from .contracts import OptionChainRow, OptionType
from .exceptions import (
    DataValidationError,
    LiquidityFilterError,
    PipelineConnectionError,
    StalenessError,
)

logger = logging.getLogger(__name__)

# Max age for a live quote before we raise StalenessError
LIVE_QUOTE_MAX_AGE_SECONDS = 300  # 5 minutes


class YFinanceDataLoader:
    """
    Loads option chain data from Yahoo Finance via yfinance.

    Implements defensive programming throughout:
      - validates all required fields
      - raises named exceptions on failure (never returns None silently)
      - logs entry, exit, and state transitions
      - applies liquidity pre-filtering before returning rows
    """

    def __init__(self, ticker: str, staleness_max_seconds: float = LIVE_QUOTE_MAX_AGE_SECONDS):
        """
        Parameters
        ----------
        ticker : str
            Underlying ticker symbol (e.g. "SPY", "AAPL", "SPX")
        staleness_max_seconds : float
            Maximum age of a quote before StalenessError is raised.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise PipelineConnectionError(
                "yfinance not installed. Run: pip install yfinance"
            )

        self.ticker_str = ticker.upper()
        self.staleness_max_seconds = staleness_max_seconds
        self._yf_ticker = yf.Ticker(self.ticker_str)
        self._fetch_time: Optional[datetime] = None
        self._underlying_price: Optional[float] = None

        logger.info("[YFinanceDataLoader] Initialized for %s", self.ticker_str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_expirations(self) -> list[str]:
        """
        Returns all available expiration dates as YYYY-MM-DD strings.

        Raises
        ------
        PipelineConnectionError
            If Yahoo Finance cannot be reached or returns no data.
        """
        logger.debug("[YFinanceDataLoader] Fetching expirations for %s", self.ticker_str)

        # yfinance rate-limits aggressively when multiple tickers are fetched
        # in rapid succession from the same IP (as happens during a scan with
        # 9+ tickers).  Retry up to 3 times with exponential backoff + jitter
        # so a transient 429 doesn't abort the entire ticker evaluation.
        import time as _time
        import random as _random

        _max_attempts = 3
        _last_exc: Exception = RuntimeError("unreachable")
        for _attempt in range(1, _max_attempts + 1):
            try:
                expirations = self._yf_ticker.options
                break
            except Exception as exc:
                _last_exc = exc
                _is_rate_limit = "rate limit" in str(exc).lower() or "too many" in str(exc).lower() or "429" in str(exc)
                if _attempt < _max_attempts and _is_rate_limit:
                    _delay = (2 ** _attempt) + _random.uniform(0.5, 2.0)
                    logger.warning(
                        "[YFinanceDataLoader] %s rate-limited (attempt %d/%d) — retrying in %.1fs",
                        self.ticker_str, _attempt, _max_attempts, _delay,
                    )
                    _time.sleep(_delay)
                else:
                    raise PipelineConnectionError(
                        f"Failed to fetch expirations for {self.ticker_str}: {exc}"
                    ) from exc

        if not expirations:
            raise PipelineConnectionError(
                f"No option expirations available for {self.ticker_str}. "
                "Ticker may be invalid or options not listed."
            )

        logger.info(
            "[YFinanceDataLoader] %s has %d expirations (%s → %s)",
            self.ticker_str, len(expirations), expirations[0], expirations[-1]
        )
        return list(expirations)

    def get_underlying_price(self) -> float:
        """
        Returns the current underlying price.

        Raises
        ------
        DataValidationError
            If price cannot be fetched or is zero/None.
        """
        try:
            info = self._yf_ticker.fast_info
            price = float(info["lastPrice"])
        except Exception as exc:
            raise DataValidationError(
                "underlying_price",
                f"Could not fetch price for {self.ticker_str}: {exc}"
            ) from exc

        if not price or price <= 0:
            raise DataValidationError(
                "underlying_price",
                f"Invalid price {price!r} for {self.ticker_str}"
            )

        self._underlying_price = price
        return price

    def get_chain(self, expiry: str) -> list[OptionChainRow]:
        """
        Fetches the full option chain for a single expiration.

        Parameters
        ----------
        expiry : str
            Expiration date as "YYYY-MM-DD"

        Returns
        -------
        list[OptionChainRow]
            All calls and puts for the expiration. Rows with missing bid AND
            ask are included but flagged — liquidity filter downstream rejects them.

        Raises
        ------
        PipelineConnectionError
            If the chain cannot be fetched.
        DataValidationError
            If the underlying price is unavailable.
        """
        logger.info(
            "[YFinanceDataLoader] Fetching chain %s %s", self.ticker_str, expiry
        )

        underlying_price = self.get_underlying_price()
        fetch_time = datetime.now(tz=timezone.utc)

        try:
            chain_data = self._yf_ticker.option_chain(expiry)
        except Exception as exc:
            raise PipelineConnectionError(
                f"Failed to fetch option chain {self.ticker_str} {expiry}: {exc}"
            ) from exc

        expiry_date = date.fromisoformat(expiry)
        today = date.today()
        dte = (expiry_date - today).days

        if dte <= 0:
            raise DataValidationError(
                "dte",
                f"Expiry {expiry} is today or in the past (dte={dte}). "
                "Refusing to process expired contracts."
            )

        rows: list[OptionChainRow] = []

        for option_type, df in [("call", chain_data.calls), ("put", chain_data.puts)]:
            if df is None or df.empty:
                logger.warning(
                    "[YFinanceDataLoader] No %s data for %s %s",
                    option_type, self.ticker_str, expiry
                )
                continue

            for _, row in df.iterrows():
                chain_row = self._parse_row(
                    row=row,
                    option_type=option_type,
                    expiry_date=expiry_date,
                    dte=dte,
                    underlying_price=underlying_price,
                    fetch_time=fetch_time,
                )
                if chain_row is not None:
                    rows.append(chain_row)

        self._fetch_time = fetch_time

        # Enrich open interest from Massive (real OI) when configured. yfinance
        # OI is unreliable (often 0) and Alpaca has no OI field; Massive's free
        # tier provides real end-of-day OI. Fails open: if Massive is not
        # configured or errors, rows keep their yfinance OI and the (softened)
        # OI gate defers to the spread gate exactly as before.
        # Overlay Alpaca's real-time bid/ask first (yfinance quotes are
        # frequently 0/0 — the 2026-06-23 zero_quote wall), then real OI from
        # Massive. Both are keyed by OCC symbol and both fail open.
        rows = self._enrich_quotes(rows, expiry)
        rows = self._enrich_open_interest(rows, expiry)

        logger.info(
            "[YFinanceDataLoader] Chain %s %s: %d rows fetched",
            self.ticker_str, expiry, len(rows)
        )
        return rows

    def _enrich_quotes(
        self, rows: list[OptionChainRow], expiry: str
    ) -> list[OptionChainRow]:
        """Overlay Alpaca's live bid/ask onto each row (matched by OCC symbol)
        and recompute mid_price/spread_pct. yfinance option quotes are
        unreliable — frequently 0/0, the 2026-06-23 zero_quote wall — while
        Alpaca is the execution venue and returns real quotes. A row is
        overwritten only when Alpaca returns a usable quote (present and not
        0/0); otherwise it keeps its yfinance quote (fail-open, per row). No-op
        if Alpaca is unavailable or returns nothing."""
        try:
            from .alpaca_quotes import get_quote_map
            qmap = get_quote_map(self.ticker_str, expiry)
        except Exception as exc:  # defensive — enrichment must never break a scan
            logger.debug("[YFinanceDataLoader] quote enrichment skipped: %s", exc)
            return rows

        if not qmap:
            return rows

        enriched = 0
        for row in rows:
            q = qmap.get(row.symbol)
            if q is None:
                continue
            bid, ask = q
            # Skip if Alpaca has no usable quote either (fail-open, per row).
            if (bid is None and ask is None) or ((bid or 0) == 0 and (ask or 0) == 0):
                continue
            row.bid = bid
            row.ask = ask
            # Recompute derived fields (mirrors OptionChainRow.__post_init__).
            if bid is not None and ask is not None and bid + ask > 0:
                row.mid_price = (bid + ask) / 2.0
                row.spread_pct = (
                    (ask - bid) / row.mid_price if row.mid_price > 0 else None
                )
            row.source = "yfinance+alpaca_quote"
            enriched += 1
        if enriched:
            logger.info(
                "[YFinanceDataLoader] Alpaca quotes applied to %d/%d %s contracts",
                enriched, len(rows), self.ticker_str
            )
        return rows

    def _enrich_open_interest(
        self, rows: list[OptionChainRow], expiry: str
    ) -> list[OptionChainRow]:
        """Override each row's open_interest with Massive's real OI when present.
        Matched by bare OCC symbol. No-op (returns rows unchanged) if Massive is
        unavailable or returns nothing for a contract."""
        try:
            from .massive_data import get_open_interest_map
            oi_map = get_open_interest_map(self.ticker_str, expiry)
        except Exception as exc:  # defensive — enrichment must never break a scan
            logger.debug("[YFinanceDataLoader] OI enrichment skipped: %s", exc)
            return rows

        if not oi_map:
            return rows

        enriched = 0
        for row in rows:
            real_oi = oi_map.get(row.symbol)
            if real_oi is not None and real_oi != row.open_interest:
                row.open_interest = real_oi
                enriched += 1
        if enriched:
            logger.info(
                "[YFinanceDataLoader] Massive OI applied to %d/%d %s contracts",
                enriched, len(rows), self.ticker_str
            )
        return rows

    def get_full_chain(self) -> dict[str, list[OptionChainRow]]:
        """
        Fetches chains for all available expirations.

        Returns
        -------
        dict mapping expiry string → list of OptionChainRow
        """
        expirations = self.get_expirations()
        result: dict[str, list[OptionChainRow]] = {}

        for expiry in expirations:
            try:
                rows = self.get_chain(expiry)
                result[expiry] = rows
            except (PipelineConnectionError, DataValidationError) as exc:
                logger.warning(
                    "[YFinanceDataLoader] Skipping expiry %s: %s", expiry, exc
                )
                continue

        if not result:
            raise PipelineConnectionError(
                f"Failed to fetch any chains for {self.ticker_str}"
            )

        total_rows = sum(len(v) for v in result.values())
        logger.info(
            "[YFinanceDataLoader] Full chain: %d expirations, %d total rows",
            len(result), total_rows
        )
        return result

    def get_chain_filtered(
        self,
        expiry: str,
        min_open_interest: int = 100,
        max_spread_pct: float = 0.25,
        min_dte: int = 1,
        option_type: Optional[str] = None,    # "call", "put", or None for both
        min_delta: Optional[float] = None,     # requires greeks layer; skipped here
        max_delta: Optional[float] = None,
    ) -> list[OptionChainRow]:
        """
        Fetches a chain and applies liquidity pre-filters.

        Raises LiquidityFilterError (logged) for each rejected contract but
        continues processing — only the accepted rows are returned.

        Parameters
        ----------
        min_open_interest : int
            Reject contracts with OI below this threshold.
        max_spread_pct : float
            Reject contracts where (ask-bid)/mid exceeds this fraction.
        min_dte : int
            Reject if DTE is below this (default 1 — no same-day expiry).
        option_type : str or None
            Filter to "call" or "put" only. None returns both.
        """
        raw_rows = self.get_chain(expiry)

        accepted: list[OptionChainRow] = []
        # Track WHY contracts are rejected. Previously the per-reason detail was
        # logged at debug (invisible at Railway's info level), so the
        # 2026-06-22 "all N contracts rejected" logs never said which gate fired.
        reasons: Counter = Counter()

        for row in raw_rows:
            # Option type filter
            if option_type and row.option_type != option_type:
                continue

            # DTE gate
            if row.dte < min_dte:
                reasons["dte"] += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: dte=%d < min_dte=%d",
                    row.symbol, row.dte, min_dte
                )
                continue

            # Bid/ask presence
            if row.bid is None or row.ask is None:
                reasons["no_quote"] += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: missing bid/ask", row.symbol
                )
                continue

            # Zero bid AND zero ask — not tradeable
            if row.bid == 0 and row.ask == 0:
                reasons["zero_quote"] += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: zero bid and ask", row.symbol
                )
                continue

            # Open interest gate.
            # NOTE: neither data source gives reliable OI. yfinance frequently
            # returns 0 (the root cause of the 2026-06-22 all-rejected scan on
            # SPY/IWM/SMH — impossibly illiquid only if the data is wrong), and
            # Alpaca's option snapshot carries NO open_interest field at all.
            # So OI of 0 or None is treated as "unknown" and we defer to the
            # spread gate (the real fill-quality guard). Only a POSITIVE OI
            # below the threshold is a genuine liquidity rejection.
            if row.open_interest and row.open_interest < min_open_interest:
                reasons["oi"] += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: OI=%d < min=%d",
                    row.symbol, row.open_interest, min_open_interest
                )
                continue

            # Spread gate — the primary, data-reliable liquidity guard.
            if row.spread_pct is not None and row.spread_pct > max_spread_pct:
                reasons["spread"] += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: spread=%.1f%% > max=%.1f%%",
                    row.symbol, row.spread_pct * 100, max_spread_pct * 100
                )
                continue

            accepted.append(row)

        rejected = sum(reasons.values())
        # INFO-level breakdown so Railway logs show exactly which gate fired.
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "none"
        logger.info(
            "[YFinanceDataLoader] Filter result %s %s: %d accepted, %d rejected "
            "(by reason: %s)",
            self.ticker_str, expiry, len(accepted), rejected, breakdown
        )

        if not accepted:
            raise LiquidityFilterError(
                f"{self.ticker_str} {expiry}",
                f"All {len(raw_rows)} contracts failed liquidity filters "
                f"(min_oi={min_open_interest}, max_spread={max_spread_pct:.0%}; "
                f"rejections by reason: {breakdown})"
            )

        return accepted

    def check_staleness(self) -> None:
        """
        Raises StalenessError if the last fetch was too long ago.
        Call before using cached chain data in a live trading loop.
        """
        if self._fetch_time is None:
            raise StalenessError("chain_data", float("inf"), self.staleness_max_seconds)

        now = datetime.now(tz=timezone.utc)
        age = (now - self._fetch_time).total_seconds()

        if age > self.staleness_max_seconds:
            raise StalenessError("chain_data", age, self.staleness_max_seconds)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_row(
        self,
        row: pd.Series,
        option_type: str,
        expiry_date: date,
        dte: int,
        underlying_price: float,
        fetch_time: datetime,
    ) -> Optional[OptionChainRow]:
        """
        Parses one row from a yfinance option chain DataFrame into an OptionChainRow.

        Returns None and logs a warning if required fields are missing,
        rather than raising — the caller collects valid rows.
        """
        try:
            symbol = str(row.get("contractSymbol", ""))
            if not symbol:
                logger.warning("[YFinanceDataLoader] Row missing contractSymbol, skipping")
                return None

            strike = float(row.get("strike", 0))
            if strike <= 0:
                logger.warning(
                    "[YFinanceDataLoader] Invalid strike %s for %s, skipping",
                    strike, symbol
                )
                return None

            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            last_price = _safe_float(row.get("lastPrice"))
            volume = _safe_int(row.get("volume"))
            open_interest = _safe_int(row.get("openInterest"))

            return OptionChainRow(
                symbol=symbol,
                underlying=self.ticker_str,
                option_type=option_type,
                strike=strike,
                expiry=expiry_date,
                dte=dte,
                bid=bid,
                ask=ask,
                last_price=last_price,
                mid_price=None,      # computed in __post_init__
                volume=volume,
                open_interest=open_interest,
                underlying_price=underlying_price,
                data_timestamp=fetch_time,
                source="yfinance",
            )

        except Exception as exc:
            logger.warning(
                "[YFinanceDataLoader] Failed to parse row: %s — %s",
                row.get("contractSymbol", "?"), exc
            )
            return None


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None if conversion fails or value is NaN."""
    if value is None:
        return None
    try:
        result = float(value)
        import math
        return None if math.isnan(result) else result
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> Optional[int]:
    """Convert a value to int, returning None if conversion fails."""
    if value is None:
        return None
    try:
        f = float(value)
        import math
        if math.isnan(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None
