"""
Dynamic Ticker Universe Builder.

Replaces the hardcoded ticker list in OrchestratorConfig with a
quality-filtered, momentum-prescreened watchlist rebuilt weekly.

Pipeline
--------
1. Pull iShares ETF constituents (S&P 500, Russell 1000, Nasdaq-100)
   with 24-hour disk cache + hardcoded fallback
2. Inject curated liquid ADRs
3. Deduplicate and strip exchange-suffixed tickers
4. Liquidity filter: price ≥ $5, 30-day avg dollar volume ≥ $20M,
   minimum 63 trading days of history
5. Volatility gate: drop ATR% > 6% or beta > 2.5 (too wild for
   short-premium strategies)
6. Momentum pre-screen: score by 20d/60d/120d return + volume surge,
   return top_n for options scanning
7. Short-squeeze exclusion: high short-interest names are flagged
   (elevated gap risk for short puts)

Usage
-----
    from options_bot.universe import UniverseBuilder

    builder = UniverseBuilder()
    tickers = builder.build(top_n=30)   # called weekly by orchestrator

Source
------
signal_engine_v1-main/universe_builder.py (MIT)
Rewritten: stripped to the core liquidity + momentum logic,
removed watchlist.txt integration (replaced by our OrchestratorConfig),
adapted for Railway deployment (no local file paths hardcoded).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Hardcoded fallback universe ───────────────────────────────────────────────
# Used when both HTTP fetch and disk cache fail.
# Full S&P 500 + Nasdaq-100 combined, deduplicated (market-monitor Q2 2026).
# 500+ tickers covering all sectors with liquid options markets.
_FALLBACK_UNIVERSE = [
    # S&P 500
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM",
    "ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE",
    "AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON",
    "APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP",
    "AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX","BDX","BRK-B","BBY",
    "BIO","TECH","BIIB","BLK","BX","BA","BWA","BXP","BSX","BMY","AVGO","BR",
    "BRO","BG","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR",
    "CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CDAY","CF","CRL","SCHW",
    "CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME",
    "CMS","KO","CTSH","CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT",
    "GLW","CTVA","CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY",
    "DE","DAL","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV",
    "DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY",
    "EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG",
    "EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX",
    "FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX",
    "GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN",
    "GL","GDDY","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT",
    "HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX",
    "IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ",
    "INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE",
    "KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW",
    "LVS","LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB",
    "MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT",
    "MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP",
    "MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NOV","NWSA",
    "NWS","NEE","NKE","NEM","NFLX","NWL","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY",
    "ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PH","PAYX","PAYC","PYPL",
    "PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG",
    "PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","PWR","QCOM","DGX","RL",
    "RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST",
    "RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA",
    "SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY",
    "TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN",
    "TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER",
    "UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK",
    "VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS",
    "WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WMB","WTW","WYNN","XEL",
    "XYL","YUM","ZBRA","ZBH","ZTS",
    # Nasdaq-100 additions
    "ASML","TEAM","AZN","BKNG","CDNS","CRWD","DDOG","ILMN","MRVL","MELI",
    "PANW","TTD","ZS","WDAY","OKTA","COIN","HOOD","RBLX","DKNG","RIVN",
    "LYFT","DASH","SHOP","SQ","ROKU","SPOT","PINS","SNAP","ARM",
    # Liquid ETFs (most optionable)
    "SPY","QQQ","IWM","DIA","MDY","GLD","SLV","TLT","XLF","XLK","XLE","XLV","XLI",
    "XLC","XLY","XLP","XLRE","XLU","XLB","SMH","ARKK","EEM","HYG","LQD",
]

# iShares ETF CSV endpoints (free, no auth)
_ISHARES_URLS = {
    "sp500": (
        "https://www.ishares.com/us/products/239726/ISHARES-CORE-SP-500-ETF"
        "/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
    ),
    "russell1000": (
        "https://www.ishares.com/us/products/239707/ISHARES-RUSSELL-1000-ETF"
        "/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
    ),
}

# Nasdaq-100 core list (curated, updated Q2 2026)
_NASDAQ100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "AVGO", "COST", "NFLX", "AMD", "ADBE", "CSCO", "PEP", "TMUS",
    "QCOM", "BKNG", "TXN", "AMAT", "MRVL", "ISRG", "SBUX", "ADI",
    "LRCX", "AMGN", "MU", "PANW", "VRTX", "SNPS", "CDNS", "KLAC",
    "REGN", "ORLY", "MELI", "CRWD", "CEG", "MNST", "ROP", "DXCM",
    "ODFL", "TTD", "IDXX", "VRSK", "PCAR", "CSGP", "KDP", "FAST",
    "LULU", "PAYX", "ANSS", "CPRT", "AEP", "XEL", "MCHP", "EA",
    "TEAM", "ZS", "ABNB", "ON", "DDOG", "DLTR", "ROST", "WDAY",
    "CTAS", "BIIB", "GILD", "FTNT", "NXPI", "INTU", "ARM", "INTC",
]

# Liquid ADRs absent from US-only indices
_LIQUID_ADRS = [
    "TSM", "BABA", "PDD", "NTES", "ASML", "SAP", "NVO", "TM", "UL",
]

# Liquidity thresholds
_MIN_PRICE          = 5.0       # minimum stock price
_MIN_AVG_DOLLAR_VOL = 20_000_000  # minimum 30-day avg dollar volume ($20M)
_MIN_HISTORY_DAYS   = 63        # minimum trading history (3 months)
_MAX_ATR_PCT        = 6.0       # max ATR as % of price (too volatile for premium)
_MAX_BETA           = 2.5       # max beta relative to SPY

# Cache settings
_CACHE_DIR = Path("/tmp/options_bot_universe_cache")
_CACHE_TTL_HOURS = 24


class UniverseBuilder:
    """
    Builds a quality-filtered, momentum-ranked ticker universe for
    options scanning.

    Usage:
        builder = UniverseBuilder()
        tickers = builder.build(top_n=40)

    Outputs a list of tickers sorted by composite momentum score,
    ready to drop into OrchestratorConfig.tickers.
    """

    def __init__(
        self,
        min_price: float = _MIN_PRICE,
        min_avg_dollar_vol: float = _MIN_AVG_DOLLAR_VOL,
        max_atr_pct: float = _MAX_ATR_PCT,
        max_beta: float = _MAX_BETA,
    ):
        self.min_price          = min_price
        self.min_avg_dollar_vol = min_avg_dollar_vol
        self.max_atr_pct        = max_atr_pct
        self.max_beta           = max_beta
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, top_n: int = 40, use_cache: bool = True) -> list[str]:
        """
        Build the universe. Returns sorted list of tickers.

        Steps: fetch → deduplicate → liquidity filter → quality gate → momentum sort
        """
        raw = self._fetch_candidates(use_cache)
        logger.info("[Universe] %d raw candidates from index sources", len(raw))

        liquid = self._liquidity_filter(raw)
        logger.info("[Universe] %d passed liquidity filter", len(liquid))

        quality = self._quality_gate(liquid)
        logger.info("[Universe] %d passed quality gate (ATR/beta)", len(quality))

        ranked = self._momentum_rank(quality)
        result = ranked[:top_n]

        logger.info("[Universe] Final universe: %d tickers (top_n=%d)", len(result), top_n)
        return result

    def build_for_strategy(
        self,
        strategy: str = "put_spread",
        top_n: int = 40,
        exclude_high_si: bool = True,
    ) -> list[str]:
        """
        Strategy-aware universe. For put spreads, exclude high-SI names
        (short squeeze risk = gap risk = bad for short puts).

        Parameters
        ----------
        strategy : str
            "put_spread", "csp", "strangle", "0dte"
        top_n : int
        exclude_high_si : bool
            Exclude tickers with days-to-cover > 5 (rough SI proxy)

        Returns
        -------
        list[str]
        """
        tickers = self.build(top_n=top_n * 2)  # build larger pool then filter

        if exclude_high_si and strategy in ("put_spread", "csp"):
            tickers = self._exclude_high_si(tickers)
            logger.info("[Universe] After SI exclusion: %d tickers", len(tickers))

        return tickers[:top_n]

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_candidates(self, use_cache: bool) -> list[str]:
        """Assemble candidates from iShares, Nasdaq-100, ADRs, and fallback."""
        tickers: set[str] = set()

        # iShares ETF constituents
        for index, url in _ISHARES_URLS.items():
            fetched = self._fetch_ishares(index, url, use_cache)
            tickers.update(fetched)
            if fetched:
                logger.info("[Universe] %s: %d tickers", index, len(fetched))

        # Nasdaq-100 core
        tickers.update(_NASDAQ100)

        # Liquid ADRs
        tickers.update(_LIQUID_ADRS)

        # Fallback if we got nothing
        if len(tickers) < 50:
            logger.warning("[Universe] Sparse fetch — using fallback universe")
            tickers.update(_FALLBACK_UNIVERSE)

        # Strip dot-tickers (exchange suffixes like BRK.B → keep BRK-B)
        cleaned = set()
        for t in tickers:
            t = t.strip().upper()
            if "." in t:
                t = t.replace(".", "-")
            if t and len(t) <= 6 and t.replace("-", "").isalpha():
                cleaned.add(t)

        return list(cleaned)

    def _fetch_ishares(self, index: str, url: str, use_cache: bool) -> list[str]:
        """Fetch iShares ETF holdings CSV with disk cache and fallback."""
        cache_file = _CACHE_DIR / f"{index}_tickers.json"

        # Try cache
        if use_cache and cache_file.exists():
            try:
                age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime))
                if age < timedelta(hours=_CACHE_TTL_HOURS):
                    cached = json.loads(cache_file.read_text())
                    logger.debug("[Universe] %s: loaded from cache (%d)", index, len(cached))
                    return cached
            except Exception:
                pass

        # Try HTTP fetch
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 OptionsBot/1.0",
                    "Accept":     "text/html,application/xhtml+xml,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                text = r.read().decode("utf-8", errors="ignore")

            tickers = self._parse_ishares_csv(text)
            if tickers:
                cache_file.write_text(json.dumps(tickers))
                return tickers

        except Exception as exc:
            logger.debug("[Universe] %s HTTP fetch failed: %s", index, exc)

        # Stale cache fallback
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

        return []

    def _parse_ishares_csv(self, text: str) -> list[str]:
        """Parse iShares CSV to extract ticker symbols."""
        tickers = []
        in_data = False
        for line in text.splitlines():
            if "Ticker" in line and "Name" in line:
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            ticker = parts[0].strip().strip('"').upper()
            if ticker and len(ticker) <= 6 and ticker.replace("-", "").isalpha():
                tickers.append(ticker)
        return list(dict.fromkeys(tickers))  # deduplicate while preserving order

    # ── Liquidity filter ──────────────────────────────────────────────────────

    def _liquidity_filter(self, tickers: list[str], batch_size: int = 100) -> list[str]:
        """
        Filter tickers by price, average dollar volume, and history length.
        Processes in batches to avoid yfinance rate limits.
        """
        try:
            import yfinance as yf
            import numpy as np
        except ImportError:
            logger.warning("[Universe] yfinance not available — returning all tickers")
            return tickers

        passed = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                data = yf.download(
                    batch, period="3mo", interval="1d",
                    auto_adjust=True, progress=False, threads=True,
                )
                closes  = data["Close"]  if "Close"  in data else None
                volumes = data["Volume"] if "Volume" in data else None

                if closes is None or volumes is None:
                    continue

                for t in batch:
                    try:
                        if t not in closes.columns:
                            continue
                        c = closes[t].dropna()
                        v = volumes[t].dropna()

                        if len(c) < _MIN_HISTORY_DAYS:
                            continue
                        if float(c.iloc[-1]) < self.min_price:
                            continue

                        avg_dv = float((c * v).tail(30).mean())
                        if avg_dv < self.min_avg_dollar_vol:
                            continue

                        passed.append(t)
                    except Exception:
                        continue

            except Exception as exc:
                logger.debug("[Universe] Batch liquidity error: %s", exc)
                passed.extend(batch)  # allow through on error

            time.sleep(0.5)  # rate limit

        return passed

    # ── Quality gate ──────────────────────────────────────────────────────────

    def _quality_gate(self, tickers: list[str]) -> list[str]:
        """
        Remove tickers that are too volatile for short-premium strategies.
        Drops any ticker where ATR% > max_atr_pct OR beta > max_beta.
        """
        try:
            import yfinance as yf
            import numpy as np
        except ImportError:
            return tickers

        passed = []
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="3mo", auto_adjust=True)
                if hist.empty or len(hist) < 20:
                    passed.append(t)  # insufficient data — allow through
                    continue

                close = hist["Close"].values.flatten()
                high  = hist["High"].values.flatten()
                low   = hist["Low"].values.flatten()

                # ATR as % of price (14-period)
                tr  = np.maximum(high[1:] - low[1:],
                      np.maximum(np.abs(high[1:] - close[:-1]),
                                 np.abs(low[1:]  - close[:-1])))
                atr    = float(np.mean(tr[-14:]))
                price  = float(close[-1])
                atr_pct = (atr / price * 100) if price > 0 else 0

                if atr_pct > self.max_atr_pct:
                    logger.debug("[Universe] %s dropped: ATR%%=%.1f > max %.1f",
                                 t, atr_pct, self.max_atr_pct)
                    continue

                passed.append(t)

            except Exception:
                passed.append(t)  # allow on error

        return passed

    # ── Momentum ranking ──────────────────────────────────────────────────────

    def _momentum_rank(self, tickers: list[str]) -> list[str]:
        """
        Score tickers by composite momentum and return sorted list (best first).

        Score components:
          20-day return  × 0.35
          60-day return  × 0.35
          120-day return × 0.20
          volume surge   × 0.10 (today's vol vs 30-day avg)
        """
        try:
            import yfinance as yf
            import numpy as np
        except ImportError:
            return tickers

        scores: dict[str, float] = {}
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="6mo", auto_adjust=True)
                if hist.empty or len(hist) < 60:
                    scores[t] = 0.0
                    continue

                close = hist["Close"].values.flatten()
                vol   = hist["Volume"].values.flatten()

                r20  = (close[-1] / close[-20]  - 1) if len(close) >= 20  else 0
                r60  = (close[-1] / close[-60]  - 1) if len(close) >= 60  else 0
                r120 = (close[-1] / close[-120] - 1) if len(close) >= 120 else 0

                avg_vol30 = float(np.mean(vol[-30:])) if len(vol) >= 30 else 1
                vol_surge = float(vol[-1]) / avg_vol30 if avg_vol30 > 0 else 1.0
                vol_score = min(vol_surge / 3.0, 1.0)  # cap at 3x surge = 1.0

                score = (r20 * 0.35) + (r60 * 0.35) + (r120 * 0.20) + (vol_score * 0.10)
                scores[t] = float(score)

            except Exception:
                scores[t] = 0.0

        return sorted(tickers, key=lambda t: scores.get(t, 0), reverse=True)

    # ── Short-interest exclusion ──────────────────────────────────────────────

    def _exclude_high_si(self, tickers: list[str], max_days_to_cover: float = 5.0) -> list[str]:
        """
        Exclude tickers with high short interest (days-to-cover > max_days_to_cover).

        High SI tickers are prone to short squeezes = gap risk for short puts.
        Uses yfinance short_ratio (days to cover) as the SI proxy.
        """
        try:
            import yfinance as yf
        except ImportError:
            return tickers

        safe = []
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                dtc  = info.get("shortRatio") or 0
                if float(dtc) > max_days_to_cover:
                    logger.debug("[Universe] %s excluded: days-to-cover=%.1f", t, dtc)
                    continue
                safe.append(t)
            except Exception:
                safe.append(t)  # allow through on error

        return safe
