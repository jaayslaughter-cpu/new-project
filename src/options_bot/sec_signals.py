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


def _parse_form4_xml(cik: str, accession: str) -> list[dict]:
    """Parse Form 4 XML. Returns list of {transaction_type, shares, price, value}."""
    try:
        acc_nodash = accession.replace('-', '')
        acc_dashes = f'{acc_nodash[:10]}-{acc_nodash[10:12]}-{acc_nodash[12:]}'
        cik_plain  = str(int(cik))
        index_url  = (f'https://www.sec.gov/Archives/edgar/data/{cik_plain}/'
                      f'{acc_nodash}/{acc_dashes}-index.json')
        index = _sec_get(index_url)
        if not index: return []
        xml_name = next((d['document'] for d in index.get('documents',[]) if d.get('document','').endswith('.xml')), None)
        if not xml_name: return []
        import urllib.request as _ur, xml.etree.ElementTree as ET
        xml_url = f'https://www.sec.gov/Archives/edgar/data/{cik_plain}/{acc_nodash}/{xml_name}'
        r = _ur.Request(xml_url, headers={'User-Agent': 'OptionsBot research@localhost'})
        with _ur.urlopen(r, timeout=10) as resp:
            root = ET.fromstring(resp.read().decode('utf-8', errors='ignore'))
        txns = []
        for txn in root.iter('nonDerivativeTransaction'):
            code_el   = txn.find('.//transactionCode')
            shares_el = txn.find('.//transactionShares/value')
            price_el  = txn.find('.//transactionPricePerShare/value')
            code   = code_el.text.strip()  if code_el   is not None else ''
            shares = float(shares_el.text) if shares_el is not None and shares_el.text else 0.0
            price  = float(price_el.text)  if price_el  is not None and price_el.text  else 0.0
            txns.append({'transaction_type': code, 'shares': shares, 'price': price, 'value': shares*price})
        return txns
    except Exception as exc:
        logger.debug('[SEC] Form 4 XML parse failed (%s/%s): %s', cik, accession, exc)
        return []


def get_insider_transactions(ticker: str, days_back: int = 90) -> list[dict]:
    """Return confirmed insider PURCHASE transactions (type=P only, XML-parsed)."""
    cik = get_cik(ticker)
    if not cik: return []
    data = _sec_get(f'{_EDGAR_BASE}/submissions/CIK{cik}.json')
    if not data: return []
    recent     = data.get('filings', {}).get('recent', {})
    forms      = recent.get('form', [])
    dates      = recent.get('filingDate', [])
    accessions = recent.get('accessionNumber', [])
    cutoff     = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    purchases  = []
    xml_fetches = 0
    for i, form in enumerate(forms):
        if form != '4': continue
        filing_date = dates[i] if i < len(dates) else ''
        if filing_date < cutoff: continue
        if xml_fetches >= 5: break
        accession = accessions[i] if i < len(accessions) else ''
        if not accession: continue
        xml_fetches += 1
        for txn in _parse_form4_xml(cik, accession):
            if txn.get('transaction_type') == 'P':
                purchases.append({'date': filing_date, 'form': form,
                                  'accession': accession, 'transaction_type': 'P',
                                  'shares': txn['shares'], 'price': txn['price'], 'value': txn['value']})
    logger.debug('[SEC] %s: %d purchase(s) in last %dd', ticker, len(purchases), days_back)
    return purchases






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
    """Re-enabled — Form 4 XML parsing now filters type=P purchases only."""
    return _score_sec_signals_impl(ticker, use_cache=use_cache)


def _score_sec_signals_impl(ticker: str, use_cache: bool = True) -> dict:
    """Core SEC scoring — confirmed open-market purchases only."""
    now = time.monotonic()
    ticker = ticker.upper()
    if use_cache:
        ts, cached = _results_cache.get(ticker, (0.0, {}))
        if cached and (now - ts) < _CACHE_TTL:
            return cached
    insider_txns     = get_insider_transactions(ticker, days_back=90)
    activist_filings = get_activist_filings(ticker, days_back=365)
    buy_count    = len(insider_txns)
    has_activist = len(activist_filings) > 0
    is_activist  = any(f.get('is_activist') for f in activist_filings)
    last_buy_date = max((t['date'] for t in insider_txns if t.get('date')), default='')
    score = 0
    if buy_count >= 3: score += 40
    elif buy_count >= 1: score += 20
    if is_activist: score += 30
    elif has_activist: score += 15
    signal = 'STRONG_BUY' if score >= 60 else ('BUY' if score >= 20 else 'NEUTRAL')
    parts = []
    if buy_count > 0: parts.append(f'{buy_count} confirmed insider purchase(s) in last 90d')
    if is_activist: parts.append('activist 13D stake')
    elif has_activist: parts.append('passive 13G stake')
    if not parts: parts.append('no significant SEC activity')
    result = {'ticker': ticker, 'score': min(score,100), 'insider_buy_count': buy_count,
              'has_activist': has_activist, 'is_activist': is_activist,
              'last_insider_buy_date': last_buy_date, 'signal': signal,
              'detail': '; '.join(parts)}
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
