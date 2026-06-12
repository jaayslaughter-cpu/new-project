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
        try:
            expirations = self._yf_ticker.options
        except Exception as exc:
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
        logger.info(
            "[YFinanceDataLoader] Chain %s %s: %d rows fetched",
            self.ticker_str, expiry, len(rows)
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
        rejected = 0

        for row in raw_rows:
            # Option type filter
            if option_type and row.option_type != option_type:
                continue

            # DTE gate
            if row.dte < min_dte:
                rejected += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: dte=%d < min_dte=%d",
                    row.symbol, row.dte, min_dte
                )
                continue

            # Bid/ask presence
            if row.bid is None or row.ask is None:
                rejected += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: missing bid/ask", row.symbol
                )
                continue

            # Zero bid AND zero ask — not tradeable
            if row.bid == 0 and row.ask == 0:
                rejected += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: zero bid and ask", row.symbol
                )
                continue

            # Open interest gate
            if row.open_interest is not None and row.open_interest < min_open_interest:
                rejected += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: OI=%d < min=%d",
                    row.symbol, row.open_interest, min_open_interest
                )
                continue

            # Spread gate
            if row.spread_pct is not None and row.spread_pct > max_spread_pct:
                rejected += 1
                logger.debug(
                    "[LiquidityFilter] Rejected %s: spread=%.1f%% > max=%.1f%%",
                    row.symbol, row.spread_pct * 100, max_spread_pct * 100
                )
                continue

            accepted.append(row)

        logger.info(
            "[YFinanceDataLoader] Filter result %s %s: %d accepted, %d rejected",
            self.ticker_str, expiry, len(accepted), rejected
        )

        if not accepted:
            raise LiquidityFilterError(
                f"{self.ticker_str} {expiry}",
                f"All {len(raw_rows)} contracts failed liquidity filters "
                f"(min_oi={min_open_interest}, max_spread={max_spread_pct:.0%})"
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
