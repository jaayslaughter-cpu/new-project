"""
Alpaca real-time option-quote source for the liquidity gate.

WHY THIS EXISTS
---------------
The chain is fetched from yfinance, whose option bid/ask is unreliable and
frequently returns 0/0 — the root cause of the 2026-06-23 "all contracts
rejected (zero_quote)" baseline: every shortlisted ticker, including SPY and
QQQ monthlies, came back with zero bid AND zero ask, so the liquidity gate
correctly rejected 100% of contracts. Alpaca — the execution venue — DOES
return real quotes (the boot probe returned SPY...P00400000 bid=1.28 ask=1.29).

This module overlays Alpaca's live bid/ask onto the yfinance chain rows, matched
by OCC symbol, so the liquidity gate (and downstream sizing) sees the same
quotes the bot will actually trade against. yfinance still provides the chain
*structure* (the contract list / strikes / expiries); Alpaca provides the price.

DESIGN CONSTRAINTS
------------------
- NO CACHING. Unlike Massive OI (an end-of-day, day-stable figure cached per
  trading day), quotes are intraday and must be fresh on every scan. One Alpaca
  call per (ticker, expiry) per request (~10-15 calls per scan).
- FAIL-OPEN, per row. No credentials, a disabled kill-switch, a tripped circuit
  breaker, or any error returns an empty map; the caller then keeps each row's
  existing yfinance quote and the gate behaves exactly as it does today. A row
  is overwritten only when Alpaca returns a *usable* quote for it; a missing or
  (0,0) Alpaca quote leaves the row untouched. This module can only ever REPLACE
  a bad quote with the venue's real quote — it can never blank a good one.
- Kill switch: ALPACA_QUOTE_ENRICH=false disables enrichment without a deploy.

Credentials are read from ALPACA_API_KEY / ALPACA_SECRET_KEY at call time (not
import time) so Railway env-injection timing can't blank them — same lesson as
the Discord webhook and Massive fixes.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .circuit_breaker import data_circuit_breaker as _cb

logger = logging.getLogger(__name__)

_CB_KEY = "alpaca_quotes"

# Lazily-constructed, process-cached Alpaca option data client.
_client = None
_warned_no_key = False


def _enabled() -> bool:
    return os.getenv("ALPACA_QUOTE_ENRICH", "true").strip().lower() != "false"


def _creds() -> tuple[Optional[str], Optional[str]]:
    k = os.getenv("ALPACA_API_KEY")
    s = os.getenv("ALPACA_SECRET_KEY")
    return (k.strip() if k else None, s.strip() if s else None)


def _get_client():
    """Lazily build (and process-cache) the Alpaca option data client.
    Returns None if credentials are missing or alpaca-py is unavailable."""
    global _client
    if _client is not None:
        return _client
    key, secret = _creds()
    if not key or not secret:
        return None
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        _client = OptionHistoricalDataClient(api_key=key, secret_key=secret)
        return _client
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.debug("[AlpacaQuotes] client init failed: %s", exc)
        return None


def _extract(raw) -> dict[str, tuple[Optional[float], Optional[float]]]:
    """Pull {OCC symbol: (bid, ask)} from an alpaca-py option-chain response
    (dict-like: symbol -> snapshot with .latest_quote.bid_price/.ask_price)."""
    out: dict[str, tuple[Optional[float], Optional[float]]] = {}
    items = raw.items() if hasattr(raw, "items") else []
    for symbol, snap in items:
        quote = getattr(snap, "latest_quote", None)
        if quote is None:
            continue
        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)
        try:
            bid = float(bid) if bid is not None else None
            ask = float(ask) if ask is not None else None
        except (TypeError, ValueError):
            continue
        out[str(symbol)] = (bid, ask)
    return out


def get_quote_map(
    ticker: str, expiry: str
) -> dict[str, tuple[Optional[float], Optional[float]]]:
    """
    Return {OCC_symbol: (bid, ask)} for one underlying + one expiry from Alpaca.

    expiry : 'YYYY-MM-DD'. NOT cached (quotes are intraday). Returns {} on any
    failure, when disabled, or when credentials are absent (fail-open — never
    raises into the caller).
    """
    global _warned_no_key

    if not _enabled():
        return {}

    key, secret = _creds()
    if not key or not secret:
        if not _warned_no_key:
            logger.debug(
                "[AlpacaQuotes] ALPACA creds not set — quote enrichment disabled"
            )
            _warned_no_key = True
        return {}

    if not _cb.is_available(_CB_KEY):
        logger.debug("[AlpacaQuotes] circuit breaker OPEN — skipping quote fetch")
        return {}

    try:
        client = _get_client()
        if client is None:
            return {}
        from alpaca.data.requests import OptionChainRequest
        req = OptionChainRequest(
            underlying_symbol=ticker.upper().strip(),
            expiration_date=expiry,
        )
        raw = client.get_option_chain(req)
        qmap = _extract(raw)
        _cb.record_success(_CB_KEY)
        if qmap:
            logger.info(
                "[AlpacaQuotes] quote map %s %s: %d contracts",
                ticker, expiry, len(qmap)
            )
        return qmap
    except Exception as exc:  # noqa: BLE001 — fail-open is the whole point
        _cb.record_failure(_CB_KEY, str(exc))
        logger.warning(
            "[AlpacaQuotes] quote fetch %s %s failed (%s) — fail-open",
            ticker, expiry, exc
        )
        return {}
