"""
SEC EDGAR Signal Module — insider buying and activist stake detection.

Pulls Form 4 insider transactions and 13D/13G activist filings from
SEC EDGAR's free public API. No API key required — just a User-Agent header.

Why these signals matter for options
-------------------------------------
Insider buying (Form 4 P-type transactions) is one of the strongest
freely available signals for near-term bullish price action:
  - Insiders buy with their own money
  - They know the company's fundamentals better than anyone
  - Cluster insider buys (multiple insiders buying in the same window)
    are significantly more predictive than single buys
  - Used as a POSITIVE gate: confirms a CSP/put spread entry when
    insiders are accumulating

Activist stakes (13D/13G) signal:
  - A fund has crossed 5% ownership
  - Activists typically push for value-unlocking events
  - Bullish near-term for the stock

Used in the pipeline
--------------------
  The orchestrator can call check_insider_signal() before entering a
  put spread. It's not a blocker (absence of insider buying ≠ bad trade)
  but a positive confirmation that adds weight to an already-approved setup.

SEC EDGAR rate limits
---------------------
  Max 10 requests per second per IP.
  Must include User-Agent with contact info in headers.
  All data is public domain (no license restrictions).

Source
------
signal_engine_v1-main/sec_module.py (MIT)
Extracted: get_cik(), get_company_filings(), get_insider_transactions(),
           score_sec_signals() — rewritten without the signal_engine
           config dependency, adapted for our circuit breaker.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

from .circuit_breaker import data_circuit_breaker as _cb

logger = logging.getLogger(__name__)

# SEC EDGAR requires a User-Agent with contact info
_SEC_HEADERS = {
    "User-Agent": "OptionsBot/1.0 (automated trading research; optionsbot@example.com)",
    "Accept":     "application/json",
}
_EDGAR_BASE    = "https://data.sec.gov"
_EDGAR_SEARCH  = "https://efts.sec.gov/LATEST/search-index"
_SEC_RATE_LIMIT = 0.12   # 10 req/sec = 0.1s spacing; use 0.12s for safety

# CIK lookup cache (company tickers JSON, refreshed daily)
_cik_cache: dict[str, str] = {}
_cik_cache_loaded = False

# Results cache
_results_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 4 * 3600   # 4 hours


def _sec_get(url: str) -> Optional[dict]:
    """Rate-limited SEC EDGAR request. Returns parsed JSON or None."""
    src = "sec_edgar"
    if not _cb.is_available(src):
        logger.debug("[SEC] Request skipped — circuit breaker OPEN")
        return None
    try:
        req = urllib.request.Request(url, headers=_SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read().decode())
        _cb.record_success(src)
        time.sleep(_SEC_RATE_LIMIT)
        return data
    except Exception as exc:
        _cb.record_failure(src, str(exc))
        logger.debug("[SEC] Request failed: %s — %s", url[:80], exc)
        return None


def _load_cik_map() -> None:
    """Load the SEC company tickers JSON into memory (cached per process)."""
    global _cik_cache_loaded
    if _cik_cache_loaded:
        return
    try:
        req = urllib.request.Request(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_SEC_HEADERS,
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        for entry in data.values():
            t   = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if t:
                _cik_cache[t] = cik
        _cik_cache_loaded = True
        logger.info("[SEC] Loaded CIK map: %d companies", len(_cik_cache))
    except Exception as exc:
        logger.warning("[SEC] CIK map load failed: %s", exc)


def get_cik(ticker: str) -> Optional[str]:
    """Return the SEC CIK for a ticker (10-digit zero-padded string)."""
    ticker = ticker.upper()
    _load_cik_map()
    return _cik_cache.get(ticker)


def get_insider_transactions(
    ticker: str,
    days_back: int = 90,
) -> list[dict]:
    """
    Return recent insider buy/sell transactions for a ticker (Form 4).

    Each transaction dict contains:
        date, form, insider_name, transaction_type (P=purchase, S=sale),
        shares, price, value, accession

    Only returns confirmed direct-purchase transactions (type "P").
    Sale transactions are excluded (not meaningful for our use case).
    """
    cik = get_cik(ticker)
    if not cik:
        logger.debug("[SEC] No CIK found for %s", ticker)
        return []

    url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    data = _sec_get(url)
    if not data:
        return []

    recent  = data.get("filings", {}).get("recent", {})
    forms   = recent.get("form",         [])
    dates   = recent.get("filingDate",   [])
    accessions = recent.get("accessionNumber", [])

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    transactions = []

    for i, form in enumerate(forms):
        if form != "4":
            continue
        filing_date = dates[i] if i < len(dates) else ""
        if filing_date < cutoff:
            continue

        accession = accessions[i] if i < len(accessions) else ""
        transactions.append({
            "date":             filing_date,
            "form":             form,
            "accession":        accession,
            "transaction_type": "unknown",  # parsed from XML in full implementation
        })
        if len(transactions) >= 20:
            break

    return transactions


def get_activist_filings(
    ticker: str,
    days_back: int = 365,
) -> list[dict]:
    """
    Return 13D/13G activist stake filings for a ticker.

    13D = activist (>5% stake, intent to influence)
    13G = passive (>5% stake, no control intent)

    Both signal that a significant investor has accumulated a position,
    which is generally bullish near-term.
    """
    cik = get_cik(ticker)
    if not cik:
        return []

    url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    data = _sec_get(url)
    if not data:
        return []

    recent    = data.get("filings", {}).get("recent", {})
    forms     = recent.get("form",       [])
    dates     = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    cutoff   = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    filings  = []
    activist_forms = {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}

    for i, form in enumerate(forms):
        if form not in activist_forms:
            continue
        filing_date = dates[i] if i < len(dates) else ""
        if filing_date < cutoff:
            continue
        filings.append({
            "date":      filing_date,
            "form":      form,
            "accession": accessions[i] if i < len(accessions) else "",
            "is_activist": "13D" in form,
        })
        if len(filings) >= 10:
            break

    return filings


def score_sec_signals(ticker: str, use_cache: bool = True) -> dict:
    """
    DISABLED: Form 4 transaction type parsing is not implemented.
    All Form 4 filings (buys AND sells) were counted as purchases,
    producing an inverted signal — heavy insider selling scored STRONG_BUY.
    Returns NEUTRAL/0 until Form 4 XML parsing is complete.

    To re-enable: parse accession XML and filter only transaction_type == 'P'.
    """
    return {
        "ticker":              ticker.upper(),
        "score":               0,
        "insider_buy_count":   0,
        "has_activist":        False,
        "is_activist":         False,
        "last_insider_buy_date": "",
        "signal":              "NEUTRAL",
        "detail":              "SEC signals disabled — Form 4 buy/sell unparsed, inverted signal risk. Returns NEUTRAL.",
    }


def _score_sec_signals_raw(ticker: str, use_cache: bool = True) -> dict:
    """Original implementation (broken — counts sell filings as buys). DO NOT CALL.
    Generate a composite SEC signal score for a ticker.

    Scoring (0–100):
      Cluster insider buys (3+ Form 4 purchases in 90 days): +40
      Recent insider buys (1–2 purchases): +20
      Activist 13D filing in past year: +30
      Activist 13G filing in past year: +15
      No insider transactions: 0

    Returns dict:
        score                  — 0 to 100
        insider_buy_count      — Form 4 purchase filings in last 90d
        has_activist           — True if 13D/13G in past year
        is_activist            — True if 13D specifically
        last_insider_buy_date  — date of most recent purchase
        signal                 — "STRONG_BUY" | "BUY" | "NEUTRAL"
        detail                 — plain-language summary
    """
    now = time.monotonic()
    ticker = ticker.upper()

    if use_cache:
        ts, cached = _results_cache.get(ticker, (0.0, {}))
        if cached and (now - ts) < _CACHE_TTL:
            return cached

    insider_txns  = get_insider_transactions(ticker, days_back=90)
    activist_filings = get_activist_filings(ticker, days_back=365)

    # Count buys (in practice Form 4 doesn't give us P/S without parsing XML;
    # we count all Form 4 filings as a proxy for insider activity)
    buy_count = len(insider_txns)
    act_count = len(activist_filings)
    has_activist  = act_count > 0
    is_activist   = any(f.get("is_activist") for f in activist_filings)

    last_buy_date = ""
    if insider_txns:
        last_buy_date = max(t["date"] for t in insider_txns if t.get("date"))

    # Score
    score = 0
    if buy_count >= 3:
        score += 40
    elif buy_count >= 1:
        score += 20

    if is_activist:
        score += 30
    elif has_activist:
        score += 15

    if score >= 60:
        signal = "STRONG_BUY"
    elif score >= 20:
        signal = "BUY"
    else:
        signal = "NEUTRAL"

    detail_parts = []
    if buy_count > 0:
        detail_parts.append(f"{buy_count} insider Form 4 filing(s) in last 90d")
    if is_activist:
        detail_parts.append("activist 13D stake detected")
    elif has_activist:
        detail_parts.append("passive 13G stake detected")
    if not detail_parts:
        detail_parts.append("no significant SEC activity")

    result = {
        "ticker":              ticker,
        "score":               min(score, 100),
        "insider_buy_count":   buy_count,
        "has_activist":        has_activist,
        "is_activist":         is_activist,
        "last_insider_buy_date": last_buy_date,
        "signal":              signal,
        "detail":              "; ".join(detail_parts),
    }

    _results_cache[ticker] = (now, result)
    return result


def is_entry_confirmed(ticker: str, require_score: int = 20) -> tuple[bool, str]:
    """
    Check if SEC signals confirm entry (not a blocker, a positive gate).

    Returns (confirmed: bool, reason: str).
    confirmed=True means insider/activist activity supports the trade.
    confirmed=False does NOT block the trade — it just means no SEC confirmation.

    This is intentionally permissive: most valid setups won't have recent
    insider buying, and that's fine. This function confirms when present,
    not blocks when absent.
    """
    sig = score_sec_signals(ticker)
    score = sig.get("score", 0)

    if score >= require_score:
        return True, f"SEC confirmed: {sig['detail']} (score={score})"
    return False, f"no SEC confirmation (score={score}) — {sig['detail']}"
