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
from dataclasses import dataclass

import json
import logging
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

from .circuit_breaker import data_circuit_breaker as _cb
import os as _os

logger = logging.getLogger(__name__)

_AV_KEY     = _os.getenv("ALPHA_VANTAGE_KEY", "")
_AV_PCR_URL = ("https://www.alphavantage.co/query"
               "?function=REALTIME_PUT_CALL_RATIO&symbol={sym}&apikey={key}")


def get_av_put_call_ratio(ticker: str) -> Optional[float]:
    """
    Realtime put/call ratio from Alpha Vantage.
    PCR >= 1.2 → elevated put demand → good short-put-spread entry signal.
    Returns None if AV key not set or call fails.
    """
    if not _AV_KEY or not _cb.is_available("av_pcr"):
        return None
    try:
        url = _AV_PCR_URL.format(sym=ticker.upper().strip(), key=_AV_KEY)
        req = urllib.request.Request(url, headers={"User-Agent": "OptionsBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        _cb.record_success("av_pcr")
        pcr = data.get("put_call_ratio") or data.get("ratio")
        if pcr is None:
            for v in data.values():
                if isinstance(v, dict):
                    pcr = v.get("put_call_ratio") or v.get("ratio")
                    if pcr: break
        return float(pcr) if pcr is not None else None
    except Exception as exc:
        _cb.record_failure("av_pcr", str(exc))
        logger.debug("[AV PCR] %s: %s", ticker, exc)
        return None

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
    # Boost score with Alpha Vantage insider data (faster updates than EDGAR)
    av_boost = get_av_insider_score(ticker)
    if av_boost > 0:
        result["score"] = result.get("score", 0) + av_boost
        result["detail"] = result.get("detail", "") + f"; AV insider boost=+{av_boost}"
        logger.debug("[SEC] %s: AV insider boost +%d → total score=%d", ticker, av_boost, result["score"])

    # Congressional trades boost — filtered by news sentiment confirmation
    # Fetch news signal to cross-reference against congressional activity
    _news_sig   = result.get("news_signal")    # injected by orchestrator if available
    _news_score = result.get("news_score")     # injected by orchestrator if available
    congress_boost = score_congress_signals(
        ticker,
        news_signal=_news_sig,
        news_score=_news_score,
    )
    if congress_boost != 0:
        congress_trades = get_congress_trades(ticker)
        buys  = [t for t in congress_trades if "purchase" in t.trade_type or "buy" in t.trade_type]
        sells = [t for t in congress_trades if "sale" in t.trade_type]
        news_note = f" [news={_news_sig}]" if _news_sig else ""
        result["score"] = result.get("score", 0) + congress_boost
        result["detail"] = (
            result.get("detail", "") +
            f"; Congress boost={congress_boost:+d}{news_note}"
            f" ({len(buys)} buys/{len(sells)} sells in 90d)"
        )
        result["congress_buys"]  = len(buys)
        result["congress_sells"] = len(sells)
        result["news_confirmed"]  = _news_sig == "BUY" if _news_sig else None
        logger.info(
            "[SEC] %s: Congress %+d%s buys=%d sells=%d → total=%d",
            ticker, congress_boost, news_note, len(buys), len(sells), result["score"],
        )

    _results_cache[ticker] = (now, result)
    return result


def score_sec_with_news(
    ticker: str,
    news_signal: Optional[str] = None,
    news_score: Optional[float] = None,
) -> dict:
    """
    Run SEC scoring with news sentiment pre-injected for congress confirmation.

    Call this instead of score_sec_signals() when you have sentiment data available.
    The news_signal and news_score are temporarily stored in the result dict
    so score_congress_signals() can read them during scoring.

    Parameters
    ----------
    ticker       : ETF ticker
    news_signal  : "BUY" | "SELL" | "HOLD" from SentimentAnalyzer.get_signals()
    news_score   : weighted score -1.0 to +1.0 from TickerSignal.weighted_score
    """
    # Use a fresh (uncached) score with the news data injected
    result = _score_sec_signals_impl(ticker, use_cache=False)
    # The congress call inside _score_sec_signals_impl will have already used
    # result["news_signal"] if it was set — but we need to pre-seed it.
    # Re-run congress scoring with news context and update result.
    result["news_signal"] = news_signal
    result["news_score"]  = news_score
    congress_boost = score_congress_signals(ticker, news_signal=news_signal, news_score=news_score)
    if congress_boost != 0:
        trades = get_congress_trades(ticker)
        buys  = [t for t in trades if "purchase" in t.trade_type or "buy" in t.trade_type]
        sells = [t for t in trades if "sale" in t.trade_type]
        news_note = f" [news={news_signal}]" if news_signal else ""
        # Avoid double-counting: subtract any previous congress boost and re-add
        prev_boost = result.get("_prev_congress_boost", 0)
        result["score"] = max(0, result["score"] - prev_boost) + congress_boost
        result["_prev_congress_boost"] = congress_boost
        result["detail"] = (
            result.get("detail", "").split("; Congress boost=")[0] +
            f"; Congress boost={congress_boost:+d}{news_note}"
            f" ({len(buys)} buys/{len(sells)} sells)"
        )
        result["congress_buys"]  = len(buys)
        result["congress_sells"] = len(sells)
        result["news_confirmed"]  = news_signal == "BUY" if news_signal else None
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


# ---------------------------------------------------------------------------
# Policy Intelligence Monitor
# ---------------------------------------------------------------------------
# Monitors upstream policy signals BEFORE they become congressional trades or
# SEC filings. Four levels (highest urgency first):
#   L1 — White House Fact Sheets + Federal Register EOs (immediate sector catalyst)
#   L2 — DoD contract announcements (specific stock catalyst)
#
# Source: adapted from agentic-trading-system/execution/policy_monitor.py
# Key changes: stdlib only (no requests/BeautifulSoup), integrated with our
# circuit_breaker, sector map extended with ETF universe overlap.
#
# Returns PolicySignal objects. The orchestrator calls scan_policy_signals()
# during the daily scan and logs findings to Discord + SEC signal score.
# ---------------------------------------------------------------------------

import hashlib as _hashlib
import xml.etree.ElementTree as _ET

# Sector keyword → ticker mapping. Tickers must overlap with your universe
# (SPY, QQQ, XLE, XLF, XLI, XLK, SMH, etc.) to produce actionable signals.
POLICY_SECTOR_MAP: dict[str, dict] = {
    "defense": {
        "keywords": ["defense", "military", "pentagon", "armed forces", "weapon",
                     "drone", "missile", "munition", "national security",
                     "golden dome", "darpa"],
        "etfs": ["XLI"],           # industrials — broad defense proxy in your universe
        "watch_tickers": ["LMT", "RTX", "NOC", "GD", "LHX"],
    },
    "energy_fossil": {
        "keywords": ["oil", "gas", "lng", "coal", "drill", "offshore leasing",
                     "energy dominance", "pipeline", "permian"],
        "etfs": ["XLE"],
        "watch_tickers": ["XOM", "CVX", "COP"],
    },
    "semiconductors": {
        "keywords": ["semiconductor", "chips act", "chip", "fab", "wafer",
                     "domestic manufacturing", "advanced computing", "tsmc"],
        "etfs": ["SMH", "XLK"],
        "watch_tickers": [],
    },
    "financials": {
        "keywords": ["bank", "deregulation", "interest rate", "federal reserve",
                     "dodd-frank", "cfpb", "lending"],
        "etfs": ["XLF"],
        "watch_tickers": [],
    },
    "healthcare": {
        "keywords": ["medicare", "medicaid", "drug pricing", "fda", "prescription",
                     "healthcare", "pharma", "aca"],
        "etfs": ["XLV"],
        "watch_tickers": [],
    },
    "real_estate": {
        "keywords": ["housing", "mortgage", "fannie mae", "freddie mac",
                     "hud", "zoning", "real estate"],
        "etfs": ["XLRE"],
        "watch_tickers": [],
    },
    "materials_industrial": {
        "keywords": ["reshoring", "made in america", "tariff", "steel",
                     "aluminum", "infrastructure", "industrial"],
        "etfs": ["XLI", "XLB"],
        "watch_tickers": [],
    },
    "bonds_rates": {
        "keywords": ["treasury", "federal reserve", "interest rate", "yield",
                     "debt ceiling", "federal deficit", "monetary policy"],
        "etfs": ["TLT", "HYG"],
        "watch_tickers": [],
    },
    "commodities": {
        "keywords": ["gold", "silver", "oil", "natural gas", "commodity",
                     "strategic reserve", "critical mineral"],
        "etfs": ["GLD", "XLE"],
        "watch_tickers": [],
    },
    "emerging_markets": {
        "keywords": ["china", "tariff", "trade war", "sanctions", "export control",
                     "emerging market", "brics"],
        "etfs": ["EEM"],
        "watch_tickers": [],
    },
}

_POLICY_SOURCES = [
    {
        "name": "White House Fact Sheets",
        "url": "https://www.whitehouse.gov/fact-sheets/",
        "level": 1,
        "parse": "html_links",
        "selector_hint": "wp-block-post-title",
    },
    {
        "name": "Federal Register EOs",
        "url": (
            "https://www.federalregister.gov/api/v1/documents.json"
            "?conditions[presidential_document_type]=executive_order"
            "&per_page=20&order=newest&fields[]=title"
        ),
        "level": 1,
        "parse": "json_results",
    },
    {
        "name": "DoD Contract Announcements",
        "url": "https://www.defense.gov/News/Contracts/",
        "level": 2,
        "parse": "html_links",
        "selector_hint": "title",
    },
]

# In-process seen-set — persists for the lifetime of the process.
# Prevents refiring on repeated hourly scans within the same session.
_policy_seen: set[str] = set()


@dataclass
class PolicySignal:
    """One upstream policy signal from a government source."""
    source:   str
    headline: str
    level:    int           # 1 = immediate catalyst, 2 = stock-specific
    sectors:  list[str]
    etfs:     list[str]     # ETFs in your universe affected
    signal_id: str = ""

    def __post_init__(self):
        raw = f"{self.source}:{self.headline}"
        self.signal_id = _hashlib.md5(raw.encode()).hexdigest()[:12]


def _policy_classify(text: str) -> tuple[list[str], list[str]]:
    """Map headline text to matched sectors and their ETFs."""
    text_lower = text.lower()
    sectors, etfs = [], []
    for sector, cfg in POLICY_SECTOR_MAP.items():
        if any(kw in text_lower for kw in cfg["keywords"]):
            sectors.append(sector)
            etfs.extend(cfg["etfs"])
    return sectors, list(set(etfs))


def _policy_fetch(source: dict) -> list[str]:
    """Fetch headlines from one policy source. Stdlib only, no BeautifulSoup."""
    cb_key = f"policy_{source['name'].lower().replace(' ', '_')}"
    if not _cb.is_available(cb_key):
        return []
    try:
        import urllib.request as _ur
        import ssl as _ssl
        ctx = _ssl.create_default_context()
        req = _ur.Request(
            source["url"],
            headers={"User-Agent": "OptionsBot policy-monitor/1.0 research@localhost"}
        )
        with _ur.urlopen(req, timeout=12, context=ctx) as r:
            body = r.read().decode("utf-8", errors="replace")
        _cb.record_success(cb_key)

        # JSON API path (Federal Register)
        if source.get("parse") == "json_results":
            import json as _json
            data = _json.loads(body)
            return [r["title"] for r in data.get("results", [])[:20] if r.get("title")]

        # Simple HTML: extract text from <a> tags containing the selector hint
        # Avoids BeautifulSoup dependency — good enough for headline extraction
        import re as _re
        hint = source.get("selector_hint", "title")
        headlines = []
        # Match anchor tag content near the hint class
        for m in _re.finditer(r'<a[^>]*>([^<]{15,200})</a>', body):
            text = m.group(1).strip()
            if text and len(text) > 15:
                headlines.append(text)
        return headlines[:20]

    except Exception as exc:
        cb_key2 = f"policy_{source['name'].lower().replace(' ', '_')}"
        _cb.record_failure(cb_key2, str(exc))
        logger.debug("[Policy] Fetch failed %s: %s", source["name"], exc)
        return []


def scan_policy_signals() -> list[PolicySignal]:
    """
    Scan all policy sources for new headlines that match sector keywords.

    Returns only NEW signals not seen in this process lifetime.
    Called by the orchestrator during the daily scan.
    Fail-open: any source failure is logged and skipped.

    ETF signals map directly to your 20-ticker universe (XLE, XLF, SMH, etc.)
    so the orchestrator can flag affected tickers for the day's scan.
    """
    new_signals: list[PolicySignal] = []

    for source in _POLICY_SOURCES:
        headlines = _policy_fetch(source)
        for headline in headlines:
            sectors, etfs = _policy_classify(headline)
            if not sectors:
                continue

            sig = PolicySignal(
                source=source["name"],
                headline=headline,
                level=source["level"],
                sectors=sectors,
                etfs=etfs,
            )

            if sig.signal_id in _policy_seen:
                continue

            _policy_seen.add(sig.signal_id)
            new_signals.append(sig)
            logger.info(
                "[Policy L%d] %s: %s | sectors=%s etfs=%s",
                sig.level, sig.source, sig.headline[:80],
                ",".join(sig.sectors), ",".join(sig.etfs),
            )

    return new_signals


# ---------------------------------------------------------------------------
# Congressional Trade Monitor
# Data model from TradingBotTest-Claude/congress-copy/src/scraper.py
# Implementation: stdlib HTTP (no Playwright — Railway compatible)
# Source: CapitolTrades public pages + EDGAR Form 4 cross-reference
#
# Congressional trades beat the market by ~6% annually (STOCK Act data).
# For options: a senator buying calls or puts in a specific sector ETF
# before legislation is a strong signal in that direction.
# ---------------------------------------------------------------------------

@dataclass
class CongressTrade:
    """One congressional trade disclosure."""
    politician:     str
    party:          str           # "D" | "R" | "I"
    chamber:        str           # "senate" | "house"
    ticker:         str
    trade_type:     str           # "purchase" | "sale" | "sale_partial"
    amount_low:     float         # disclosure range low (e.g. 1000)
    amount_high:    float         # disclosure range high (e.g. 15000)
    filed_date:     str           # YYYY-MM-DD
    transaction_date: str         # YYYY-MM-DD
    days_since:     int


_CONGRESS_CACHE: dict[str, tuple[float, list[CongressTrade]]] = {}
_CONGRESS_TTL = 6 * 3600   # 6 hours

# Quiver Quantitative free API — no key required, rate limited to ~10/min
_QUIVER_URL = "https://api.quiverquant.com/beta/live/congresstrading"
_QUIVER_HEADERS = {
    "User-Agent": "OptionsBot congress-monitor/1.0 research@localhost",
    "Accept": "application/json",
}


def get_congress_trades(ticker: str, lookback_days: int = 90) -> list[CongressTrade]:
    """
    Fetch recent congressional trade disclosures for a ticker.

    Uses Quiver Quantitative's free congressional trading API.
    Falls back to empty list on any failure — fail-open.
    Caches per ticker for 6 hours.

    Returns disclosures filed within lookback_days sorted by recency.
    """
    cache_key = f"{ticker.upper()}_{lookback_days}"
    cached = _CONGRESS_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _CONGRESS_TTL:
        return cached[1]

    if not _cb.is_available("congress_trades"):
        return []

    try:
        import urllib.request as _ur
        import ssl as _ssl
        from datetime import date as _date, datetime as _dt

        url = f"{_QUIVER_URL}/{ticker.upper()}"
        req = _ur.Request(url, headers=_QUIVER_HEADERS)
        ctx = _ssl.create_default_context()
        with _ur.urlopen(req, timeout=10, context=ctx) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))

        _cb.record_success("congress_trades")
        today = _date.today()
        results = []

        for row in (data if isinstance(data, list) else []):
            try:
                filed = row.get("FiledAfter") or row.get("Date") or ""
                txn   = row.get("TransactionDate") or filed
                filed_date = _dt.strptime(filed[:10], "%Y-%m-%d").date()
                days = (today - filed_date).days
                if days > lookback_days:
                    continue

                # Parse amount range: "1K-15K", "15K-50K"
                amt_str = str(row.get("Amount") or "0-0")
                lo, hi = _parse_amount_range(amt_str)

                results.append(CongressTrade(
                    politician=str(row.get("Representative") or "Unknown"),
                    party=str(row.get("Party") or ""),
                    chamber="senate" if row.get("Chamber","").lower() == "senate" else "house",
                    ticker=ticker.upper(),
                    trade_type=str(row.get("Transaction") or "").lower(),
                    amount_low=lo,
                    amount_high=hi,
                    filed_date=filed[:10],
                    transaction_date=txn[:10],
                    days_since=days,
                ))
            except Exception:
                continue

        results.sort(key=lambda x: x.days_since)
        _CONGRESS_CACHE[cache_key] = (time.time(), results)
        if results:
            logger.info("[Congress] %s: %d trades in last %dd", ticker, len(results), lookback_days)
        return results

    except Exception as exc:
        _cb.record_failure("congress_trades", str(exc))
        logger.debug("[Congress] %s fetch failed: %s", ticker, exc)
        return []


def _parse_amount_range(text: str) -> tuple[float, float]:
    """Parse '1K-15K' or '$1,000-$15,000' into (low, high) floats."""
    import re as _re
    text = text.replace(",", "").replace("$", "").upper().strip()
    parts = _re.split(r"[-–]", text)
    if len(parts) != 2:
        return 0.0, 0.0

    def _parse(s: str) -> float:
        s = s.strip()
        mult = 1.0
        if s.endswith("M"):
            mult = 1_000_000; s = s[:-1]
        elif s.endswith("K"):
            mult = 1_000; s = s[:-1]
        try:
            return float(s) * mult
        except ValueError:
            return 0.0

    return _parse(parts[0]), _parse(parts[1])


def score_congress_signals(
    ticker: str,
    news_signal: Optional[str] = None,   # "BUY" | "SELL" | "HOLD" | None
    news_score: Optional[float] = None,  # weighted sentiment -1.0 to +1.0
) -> int:
    """
    Score congressional trades filtered and weighted by news confirmation.

    Congressional trades alone are useful but noisy — politicians sometimes
    trade on sector-wide macro views, not company-specific knowledge. Requiring
    a corroborating news signal significantly improves signal quality:

      Congress BUY + News BUY  → full boost + 10pt bonus (strong combined signal)
      Congress BUY + News HOLD → full boost (neutral confirmation, proceed)
      Congress BUY + News SELL → 50% boost (conflicting signals — halved)
      Congress SELL + any news → penalty applied (insider distribution)

    If news_signal is not provided (None), falls back to standalone scoring.

    Scoring (before news adjustment):
      Recent purchase (≤30 days):  +12 each (cap +36)
      Older purchase (31-90 days):  +5 each (cap +15)
      Sale:                         -3 each

    Returns integer score to add to SEC signal total.
    """
    trades = get_congress_trades(ticker)
    if not trades:
        return 0

    # Base score from trade activity
    raw_score = 0
    buy_count = 0
    for t in trades:
        if "purchase" in t.trade_type or "buy" in t.trade_type:
            raw_score += 12 if t.days_since <= 30 else 5
            buy_count += 1
        elif "sale" in t.trade_type:
            raw_score -= 3

    raw_score = max(-15, min(36, raw_score))

    # News confirmation adjustment
    if news_signal is not None and buy_count > 0:
        news_sig = (news_signal or "HOLD").upper()
        if news_sig == "BUY":
            # Strong combined signal: congress buying + positive news
            adjusted = raw_score + 10
            logger.info(
                "[Congress+News] %s: congress_score=%d + news_BUY_bonus=+10 → %d",
                ticker, raw_score, adjusted,
            )
            raw_score = adjusted
        elif news_sig == "SELL":
            # Conflicting: congress buying but negative news cycle
            adjusted = int(raw_score * 0.5)
            logger.info(
                "[Congress+News] %s: congress_score=%d × 0.5 (news=SELL conflict) → %d",
                ticker, raw_score, adjusted,
            )
            raw_score = adjusted
        # HOLD: no adjustment — congress signal stands on its own

    return max(-15, min(46, raw_score))   # cap raised to 46 for combined BUY+BUY


def policy_etf_boost(ticker: str, signals: list[PolicySignal]) -> int:
    """
    Return a score boost (+15 per L1 signal, +8 per L2) if a ticker is in the
    ETF list of any active policy signal.

    Pass the result of scan_policy_signals() for the day. Used by
    score_sec_signals() callers to layer policy intelligence on top of
    insider-buying scores.
    """
    boost = 0
    ticker_upper = ticker.upper().strip()
    for sig in signals:
        if ticker_upper in sig.etfs:
            boost += 15 if sig.level == 1 else 8
    return boost
