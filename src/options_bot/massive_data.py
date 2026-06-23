"""
Massive (formerly Polygon.io) options open-interest source.

WHY THIS EXISTS
---------------
The live liquidity gate reads open interest from yfinance, which frequently
returns 0 — the root cause of the 2026-06-22 "all contracts rejected" scans
(every ticker, incl. SPY/IWM). Alpaca's option snapshot carries no OI field at
all. Massive's options chain snapshot DOES return real open interest, even on
the free "Options Basic" tier (greeks/IV/OI are included; only real-time
quotes/trades are gated to paid tiers — which is fine, because OI is an
end-of-day daily figure and quotes still come from Alpaca).

DESIGN CONSTRAINTS (free tier)
------------------------------
- 5 requests/minute limit → we cache the OI map per (ticker, expiry) for the
  whole trading day and filter the request to a single expiry, so a daily scan
  makes ~1 call per shortlisted ticker (~10-15/day total).
- End-of-day delayed data → perfect for OI (a once-daily number); we never use
  Massive for prices/quotes (Alpaca supplies the real-time quote/spread).
- FAIL-OPEN: no API key, a rate-limit (429), or any error returns an empty map.
  The caller then behaves exactly as it does without Massive (OI treated as
  "unknown", spread gate decides). Massive can only ever ADD reliable OI; it
  can never block the trade path.

The API key is read from the MASSIVE_API_KEY env var at call time (not import
time) so Railway env injection timing can't blank it — same lesson as the
Discord webhook fix.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import date
from typing import Optional

from .circuit_breaker import data_circuit_breaker as _cb

logger = logging.getLogger(__name__)

_CB_KEY = "massive_oi"
_BASE_URL = "https://api.massive.com/v3/snapshot/options/{underlying}"
_UA = "OptionsBot massive-oi/1.0"
_TIMEOUT = 15
_MAX_PAGES = 10  # safety cap; a single-expiry chain is normally 1-2 pages

# Daily cache: {(ticker, expiry, iso_date): {occ_symbol: open_interest}}
_cache: dict[tuple[str, str, str], dict[str, int]] = {}

# Remember whether we've already logged "no key" so we don't spam the log.
_warned_no_key = False


def _api_key() -> Optional[str]:
    key = os.getenv("MASSIVE_API_KEY")
    return key.strip() if key else None


def _strip_occ_prefix(ticker: str) -> str:
    """Massive option tickers are 'O:SPY260731P00500000'; the bot's rows use the
    bare OCC symbol 'SPY260731P00500000'. Normalise to the bare form."""
    return ticker[2:] if ticker.startswith("O:") else ticker


def get_open_interest_map(ticker: str, expiry: str) -> dict[str, int]:
    """
    Return {bare_OCC_symbol: open_interest} for one underlying + one expiry.

    expiry : 'YYYY-MM-DD'. Cached per trading day. Returns {} on any failure or
    when no MASSIVE_API_KEY is set (fail-open — never raises into the caller).
    """
    global _warned_no_key

    key = _api_key()
    if not key:
        if not _warned_no_key:
            logger.debug("[Massive] MASSIVE_API_KEY not set — OI enrichment disabled")
            _warned_no_key = True
        return {}

    cache_key = (ticker, expiry, date.today().isoformat())
    if cache_key in _cache:
        return _cache[cache_key]

    if not _cb.is_available(_CB_KEY):
        logger.debug("[Massive] circuit breaker OPEN — skipping OI fetch")
        return {}

    oi_map: dict[str, int] = {}
    try:
        params = {
            "expiration_date": expiry,
            "limit": 250,
            "apiKey": key,
        }
        url = _BASE_URL.format(underlying=urllib.parse.quote(ticker.upper().strip()))
        pages = 0
        while url and pages < _MAX_PAGES:
            full = url + (("&" if "?" in url else "?") + urllib.parse.urlencode(params)) if pages == 0 else url
            req = urllib.request.Request(full, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                payload = json.loads(r.read().decode("utf-8", errors="replace"))

            for res in payload.get("results", []) or []:
                details = res.get("details", {}) or {}
                occ = details.get("ticker")
                oi = res.get("open_interest")
                if occ is not None and oi is not None:
                    oi_map[_strip_occ_prefix(occ)] = int(oi)

            # Pagination: next_url already carries the cursor; append apiKey only.
            next_url = payload.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": key}
            else:
                url = None
            pages += 1

        _cb.record_success(_CB_KEY)
        _cache[cache_key] = oi_map
        logger.info(
            "[Massive] OI map %s %s: %d contracts", ticker, expiry, len(oi_map)
        )
        return oi_map

    except urllib.error.HTTPError as exc:
        # 429 = rate limited on the free tier; treat as a soft failure.
        _cb.record_failure(_CB_KEY, f"HTTP {exc.code}")
        logger.warning("[Massive] OI fetch %s %s failed (HTTP %s) — fail-open",
                       ticker, expiry, exc.code)
        return {}
    except Exception as exc:  # noqa: BLE001 — fail-open is the whole point
        _cb.record_failure(_CB_KEY, str(exc))
        logger.warning("[Massive] OI fetch %s %s failed (%s) — fail-open",
                       ticker, expiry, exc)
        return {}
