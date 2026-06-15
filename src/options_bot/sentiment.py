"""
Sentiment analysis module — FinBERT news signal layer.

Uses ProsusAI/finbert (a finance-domain BERT model) to score news headlines
and summaries, then aggregates scores per ticker into a BUY / SELL / HOLD signal
with confidence weighting.

Why FinBERT over generic VADER/TextBlob:
  - Trained on financial communications (10-Ks, earnings transcripts, news)
  - Understands domain terms: "beat estimates", "headwinds", "guidance raised"
  - Three-class output (positive/negative/neutral) with calibrated probabilities
  - Batch inference — handles 50+ articles in one forward pass

Architecture:
  - Lazy model load: first call downloads ~500MB weights, subsequent calls reuse
  - CPU-only by default (no CUDA requirement for Railway deployment)
  - Batch size 16 — tuned for Railway's 1-2GB RAM containers
  - Graceful degradation: if torch/transformers not installed, returns neutral
    signals and logs a warning (non-fatal for the trading loop)

Signal generation:
  For each ticker, we collect all articles and compute:
    weighted_score = Σ(sentiment_score_i * confidence_i) / Σ(confidence_i)
  where sentiment_score: positive=+1.0, negative=-1.0, neutral=0.0

  Thresholds (configurable via SentimentConfig):
    weighted_score >= buy_threshold  → BUY
    weighted_score <= sell_threshold → SELL
    otherwise                        → HOLD

  Low-confidence neutral articles (confidence < 0.7) are excluded from
  the weighted average to prevent signal dilution from ambiguous text.

Integration in the trading pipeline:
  The sentiment signal is an additional gate — it does NOT generate trades
  by itself. The orchestrator checks it before entering a position:
    - HOLD or BUY → proceed with the options strategy
    - SELL        → skip this ticker (bearish news = poor CSP/put spread timing)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# FinBERT model identifier — publicly available on HuggingFace
_FINBERT_MODEL = "ProsusAI/finbert"

# Sentiment label mapping from FinBERT's output classes
# FinBERT outputs: 0=positive, 1=negative, 2=neutral
_LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
_SCORE_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# Module-level lazy-loaded model (shared across all SentimentAnalyzer instances)
_tokenizer = None
_model = None
_model_load_attempted = False
_finbert_available = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SentimentConfig:
    """Configuration for the sentiment analysis pipeline."""
    # FinBERT inference
    batch_size: int = 16              # articles per forward pass
    max_text_length: int = 512        # token limit (FinBERT max is 512)
    min_confidence: float = 0.55      # minimum confidence to include in score
    neutral_confidence_threshold: float = 0.70  # skip neutral if confidence < this

    # Signal thresholds
    buy_threshold: float = 0.15       # weighted score >= this → BUY
    sell_threshold: float = -0.15     # weighted score <= this → SELL

    # Minimum articles to generate a non-HOLD signal
    # AUDIT FIX: raised from 2 to 3. 2-article sample has a 95% CI of
    # ~[0%, 84%] on proportion — too wide for a meaningful signal.
    min_articles_for_signal: int = 3

    # Cache TTL in seconds (don't re-fetch news within this window)
    cache_ttl_seconds: int = 1800     # 30 minutes


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredArticle:
    """A news article with FinBERT sentiment annotation."""
    ticker: str
    title: str
    summary: str = ""
    source: str = ""
    published_at: Optional[datetime] = None
    sentiment: str = "neutral"        # "positive" | "negative" | "neutral"
    confidence: float = 0.0           # 0.0 – 1.0
    recency_weight: float = 1.0       # AUDIT FIX: exp(-hours_old/48), default=1.0 (fresh)


@dataclass
class TickerSignal:
    """Aggregated sentiment signal for one ticker."""
    ticker: str
    signal: str                        # "BUY" | "SELL" | "HOLD"
    weighted_score: float              # -1.0 to +1.0
    avg_confidence: float              # mean confidence across articles
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    top_headline: str = ""
    computed_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_vader_analyzer = None
_vader_loaded   = False


def _try_load_vader() -> bool:
    """Load VADER — ~1MB, Railway-safe, no model download."""
    global _vader_analyzer, _vader_loaded
    if _vader_loaded: return _vader_analyzer is not None
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
        _vader_loaded   = True
        logger.info('[Sentiment] VADER loaded (Railway mode, ~72%% accuracy)')
        return True
    except ImportError:
        logger.warning('[Sentiment] vaderSentiment not installed. pip install vaderSentiment')
        _vader_loaded = True; return False
    except Exception as exc:
        logger.warning('[Sentiment] VADER load failed: %s', exc)
        _vader_loaded = True; return False


def _score_text_vader(text: str) -> tuple[str, float]:
    """VADER score: returns (sentiment, confidence)."""
    if _vader_analyzer is None: return 'neutral', 0.0
    compound = _vader_analyzer.polarity_scores(text)['compound']
    if compound >= 0.05:  return 'positive', min(abs(compound), 1.0)
    if compound <= -0.05: return 'negative', min(abs(compound), 1.0)
    return 'neutral', min(abs(compound) + 0.1, 0.5)


def _try_load_model() -> bool:
    """
    Attempt to load FinBERT. Returns True if successful.
    Logs a warning and returns False if torch/transformers are not installed.
    Called once per process; result is cached in module globals.
    """
    global _tokenizer, _model, _model_load_attempted, _finbert_available

    if _model_load_attempted:
        return _finbert_available

    _model_load_attempted = True

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(
            "[Sentiment] Loading FinBERT (%s) — first run downloads ~500MB weights",
            _FINBERT_MODEL,
        )
        t0 = time.time()
        _tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL)
        _model.eval()
        elapsed = time.time() - t0
        logger.info("[Sentiment] FinBERT loaded in %.1fs (CPU mode)", elapsed)
        _finbert_available = True
        return True

    except ImportError as exc:
        logger.warning(
            "[Sentiment] torch/transformers not installed (%s). "
            "Sentiment signals will be HOLD/neutral. "
            "Install with: pip install torch transformers",
            exc,
        )
        _finbert_available = False
        return False

    except Exception as exc:
        logger.warning(
            "[Sentiment] FinBERT load failed: %s. "
            "Sentiment signals will be HOLD/neutral.",
            exc,
        )
        _finbert_available = False
        return False


# ---------------------------------------------------------------------------
# Core sentiment scorer
# ---------------------------------------------------------------------------

def score_articles(
    articles: list[ScoredArticle],
    config: SentimentConfig | None = None,
) -> list[ScoredArticle]:
    """
    Run FinBERT inference on a list of articles.
    Mutates each article's sentiment and confidence fields in-place.
    Returns the same list (modified).

    AUDIT FIX: Added recency weighting. Articles are scored by FinBERT
    but older articles receive a lower weight in the aggregate signal.
    Recency decay: weight = exp(-hours_old / 48) so a 48-hour-old article
    has 37% the weight of a current article.

    LABEL: This is a RECENCY-WEIGHTED FinBERT classification. The
    sentiment label is the FinBERT class; the aggregate score accounts
    for both classification confidence AND article age. Older articles
    contribute less to the final signal, addressing the audit finding:
    "No recency weighting exists. Older articles have equal weight to
    breaking news."

    If FinBERT is unavailable, all articles are left as neutral/0.0.
    """
    if not articles:
        return articles

    cfg = config or SentimentConfig()

    model_available = _try_load_model()
    vader_available = _try_load_vader()

    if not model_available and not vader_available:
        for art in articles:
            art.sentiment = 'neutral'; art.confidence = 0.0; art.model_used = 'none'
        return articles

    if not model_available and vader_available:
        # VADER path — Railway default when torch not installed
        for art in articles:
            text = f'{art.title}. {art.summary}'.strip()
            art.sentiment, art.confidence = _score_text_vader(text)
            art.model_used = 'vader'
        return articles

    import torch
    from datetime import timezone as _tz

    for i in range(0, len(articles), cfg.batch_size):
        batch = articles[i : i + cfg.batch_size]
        texts = []
        for art in batch:
            text = art.title
            if art.summary:
                text = text + ". " + art.summary
            texts.append(text[: cfg.max_text_length * 4])

        try:
            inputs = _tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=cfg.max_text_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = _model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)

            now_utc = datetime.now(tz=_tz.utc)
            for j, art in enumerate(batch):
                pred_idx   = predictions[j].item()
                confidence = probs[j][pred_idx].item()
                art.sentiment  = _LABEL_MAP[pred_idx]
                art.confidence = round(confidence, 4)

                # AUDIT FIX: compute recency weight
                # exp(-hours_old / 48): 0h=1.0, 24h=0.61, 48h=0.37, 96h=0.14
                if art.published_at is not None:
                    pub = art.published_at
                    if pub.tzinfo is None:
                        import pytz as _ptz
                        pub = _ptz.utc.localize(pub)
                    hours_old = max(0.0, (now_utc - pub).total_seconds() / 3600)
                    art.recency_weight = round(math.exp(-hours_old / 48.0), 4)
                else:
                    art.recency_weight = 0.5  # unknown age = half weight

        except Exception as exc:
            logger.exception(
                "[Sentiment] Inference error on batch %d: %s",
                i // cfg.batch_size, exc,
            )

    pos = sum(1 for a in articles if a.sentiment == "positive")
    neg = sum(1 for a in articles if a.sentiment == "negative")
    neu = sum(1 for a in articles if a.sentiment == "neutral")
    logger.info(
        "[Sentiment] Scored %d articles: %d positive, %d negative, %d neutral",
        len(articles), pos, neg, neu,
    )
    return articles


# ---------------------------------------------------------------------------
# Signal aggregator
# ---------------------------------------------------------------------------

def aggregate_signals(
    articles: list[ScoredArticle],
    tickers: list[str],
    config: SentimentConfig | None = None,
) -> dict[str, TickerSignal]:
    """
    Aggregate scored articles into a per-ticker signal dict.

    Parameters
    ----------
    articles : list[ScoredArticle]
        Articles that have already been scored by score_articles().
    tickers : list[str]
        The tickers we want signals for. Tickers with no articles → HOLD.
    config : SentimentConfig or None

    Returns
    -------
    dict[str, TickerSignal]
        Keyed by ticker symbol.
    """
    import math as _math

    cfg = config or SentimentConfig()

    # Group articles by ticker
    by_ticker: dict[str, list[ScoredArticle]] = {t: [] for t in tickers}
    for art in articles:
        if art.ticker in by_ticker:
            by_ticker[art.ticker].append(art)

    signals: dict[str, TickerSignal] = {}

    for ticker in tickers:
        ticker_articles = by_ticker[ticker]
        count = len(ticker_articles)

        # AUDIT FIX: raised minimum from 2 to 3.
        # LABEL: a 2-article sample has a 95% CI of ~[0%, 84%] on win rate —
        # too wide to be meaningful. 3 articles is still thin but reduces
        # the worst-case CI. The signal is explicitly labeled as a proxy.
        if count < cfg.min_articles_for_signal:
            signals[ticker] = TickerSignal(
                ticker=ticker,
                signal="HOLD",
                weighted_score=0.0,
                avg_confidence=0.0,
                article_count=count,
                positive_count=0,
                negative_count=0,
                neutral_count=count,
                top_headline=f"(only {count} articles — below minimum {cfg.min_articles_for_signal})",
            )
            continue

        # AUDIT FIX: confidence×recency weighted average
        # Combined weight = confidence × recency_weight
        # This ensures fresh high-confidence articles dominate the signal
        weighted_sum  = 0.0
        weight_total  = 0.0
        best_article: ScoredArticle | None = None
        best_weight = 0.0

        for art in ticker_articles:
            score = _SCORE_MAP.get(art.sentiment, 0.0)
            conf  = art.confidence
            recency = getattr(art, 'recency_weight', 1.0)

            # Skip ambiguous neutral articles below confidence threshold
            if score == 0.0 and conf < cfg.neutral_confidence_threshold:
                continue
            if conf < cfg.min_confidence:
                continue

            # Combined weight: confidence × recency decay
            combined_weight = conf * recency
            weighted_sum   += score * combined_weight
            weight_total   += combined_weight

            if combined_weight > best_weight:
                best_weight  = combined_weight
                best_article = art

        avg_score      = weighted_sum / weight_total if weight_total > 0 else 0.0
        avg_confidence = weight_total / count        if count       > 0 else 0.0

        # Determine signal
        if avg_score >= cfg.buy_threshold:
            signal = "BUY"
        elif avg_score <= cfg.sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        pos = sum(1 for a in ticker_articles if a.sentiment == "positive")
        neg = sum(1 for a in ticker_articles if a.sentiment == "negative")
        neu = sum(1 for a in ticker_articles if a.sentiment == "neutral")

        signals[ticker] = TickerSignal(
            ticker=ticker,
            signal=signal,
            weighted_score=round(avg_score, 4),
            avg_confidence=round(avg_confidence, 4),
            article_count=count,
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            top_headline=best_article.title if best_article else "",
        )

        logger.info(
            "[Sentiment] %s: %s (score=%.3f recency-weighted, conf=%.2f, "
            "%d articles: +%d -%d ~%d)",
            ticker, signal, avg_score, avg_confidence, count, pos, neg, neu,
        )

    return signals


# ---------------------------------------------------------------------------
# News fetcher (yfinance — no extra API key required)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# News importance pre-filter (adapted from AlphaTrade breaking_news_monitor)
# Only articles containing at least one high-signal keyword are passed to
# VADER/FinBERT. Generic price-action headlines ("stock rises 1%") add noise
# without changing the sentiment signal for option entry decisions.
# ---------------------------------------------------------------------------
_IMPORTANCE_KEYWORDS: frozenset[str] = frozenset([
    # Earnings / guidance
    "earnings", "beats", "misses", "guidance", "revenue", "profit", "loss",
    "eps", "forecast", "outlook", "raised", "lowered", "estimate",
    # Corporate events
    "merger", "acquisition", "buyout", "spinoff", "ipo", "secondary",
    "partnership", "deal", "contract", "awarded",
    # Regulatory / legal
    "fda", "approval", "approved", "rejected", "recall", "investigation",
    "sec", "doj", "lawsuit", "settlement", "fine", "penalty",
    # Management
    "ceo", "cfo", "resign", "fired", "appointed", "departure",
    # Macro catalysts
    "fed", "rate", "inflation", "cpi", "nfp", "gdp", "fomc",
    # Extreme price events
    "crash", "surge", "plunge", "rally", "halt", "delisted", "bankrupt",
    # Analyst actions
    "upgrade", "downgrade", "initiated", "price target", "overweight",
    "underweight", "buy", "sell", "hold",
])


def _is_important(title: str, summary: str = "") -> bool:
    """Return True if the headline contains at least one high-signal keyword."""
    text = (title + " " + summary).lower()
    return any(kw in text for kw in _IMPORTANCE_KEYWORDS)


def _fetch_alpaca_news(
    tickers: list[str],
    max_per_ticker: int = 10,
    lookback_days: int = 3,
) -> list[dict]:
    """
    Fetch news from Alpaca's /v1beta1/news endpoint.
    Uses existing ALPACA_API_KEY + ALPACA_SECRET_KEY env vars — no extra auth.
    Alpaca news has reliable UTC timestamps, better coverage than yfinance,
    and supports pagination. Adapted from Algo-Trader/strategy/news_filter.py.
    Returns raw dicts; caller converts to ScoredArticle.
    """
    import urllib.request as _ur
    from datetime import timezone as _tz, timedelta as _td

    api_key    = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not api_secret:
        return []

    base_url = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
    url      = f"{base_url}/v1beta1/news"
    start    = (datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")
    symbols  = ",".join(tickers)
    params   = f"symbols={symbols}&start={start}&limit={min(max_per_ticker * len(tickers), 50)}"
    headers  = {
        "APCA-API-KEY-ID":     api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    try:
        req = _ur.Request(f"{url}?{params}", headers=headers)
        with _ur.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        return data.get("news", [])
    except Exception as exc:
        logger.debug("[Sentiment] Alpaca news fetch failed: %s", exc)
        return []


def fetch_news(
    tickers: list[str],
    max_articles_per_ticker: int = 10,
) -> list[ScoredArticle]:
    """
    Fetch recent news headlines using Alpaca as primary source, yfinance as fallback.

    Source priority:
      1. Alpaca /v1beta1/news — authenticated with existing API keys, reliable
         UTC timestamps, better coverage, consistent JSON schema.
      2. yfinance — fallback when Alpaca keys are absent or request fails.

    Articles are pre-filtered by _is_important() before scoring — generic
    price-action headlines are dropped so VADER/FinBERT only processes
    high-signal content (earnings, FDA, M&A, analyst actions, macro events).
    """
    articles: list[ScoredArticle] = []
    filtered_out = 0
    source_used  = "none"

    # ── Source 1: Alpaca news ────────────────────────────────────────────────
    raw_alpaca = _fetch_alpaca_news(tickers, max_per_ticker=max_articles_per_ticker)
    if raw_alpaca:
        source_used = "alpaca"
        # Track per-ticker count to respect max_articles_per_ticker
        ticker_counts: dict[str, int] = {}
        for item in raw_alpaca:
            # Alpaca returns a flat list; filter to only requested tickers
            item_tickers = item.get("symbols", [])
            for ticker in tickers:
                if ticker not in item_tickers:
                    continue
                if ticker_counts.get(ticker, 0) >= max_articles_per_ticker:
                    continue
                title   = (item.get("headline") or item.get("title") or "").strip()
                summary = (item.get("summary") or "").strip()
                source  = item.get("source") or item.get("author") or "Alpaca"
                raw_ts  = item.get("created_at") or item.get("updated_at") or ""

                if not title:
                    continue
                if not _is_important(title, summary):
                    filtered_out += 1
                    continue

                published = None
                if raw_ts:
                    try:
                        published = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass

                articles.append(ScoredArticle(
                    ticker=ticker, title=title, summary=summary,
                    source=source, published_at=published,
                ))
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

    # ── Source 2: yfinance fallback ──────────────────────────────────────────
    if not raw_alpaca:
        try:
            import yfinance as yf
            source_used = "yfinance"
            for ticker in tickers:
                try:
                    news = yf.Ticker(ticker).news or []
                    for item in news[:max_articles_per_ticker]:
                        title   = item.get("title", "").strip()
                        summary = item.get("summary", "").strip()
                        source  = item.get("source", "")
                        ts      = item.get("providerPublishTime")

                        if not title:
                            continue
                        if not _is_important(title, summary):
                            filtered_out += 1
                            continue

                        published = None
                        if ts:
                            try:
                                published = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                            except (ValueError, TypeError, OSError):
                                pass

                        articles.append(ScoredArticle(
                            ticker=ticker, title=title, summary=summary,
                            source=source, published_at=published,
                        ))
                except Exception as exc:
                    logger.warning("[Sentiment] yfinance news failed for %s: %s", ticker, exc)
        except ImportError:
            logger.warning("[Sentiment] yfinance not installed and Alpaca fetch failed")

    logger.info(
        "[Sentiment] Fetched %d articles for %d tickers via %s (%d dropped by keyword filter)",
        len(articles), len(tickers), source_used, filtered_out,
    )
    return articles


# ---------------------------------------------------------------------------
# High-level entry point — used by orchestrator
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    High-level sentiment analyzer for the trading pipeline.

    Usage:
        analyzer = SentimentAnalyzer(config=SentimentConfig())
        signals = analyzer.get_signals(["SPY", "QQQ", "IWM"])

        if signals["SPY"].signal == "SELL":
            logger.info("Bearish news on SPY — skipping put spread entry")

    The analyzer caches results for cache_ttl_seconds to avoid re-fetching
    on every scan.
    """

    def __init__(self, config: SentimentConfig | None = None):
        self.config = config or SentimentConfig()
        self._cache: dict[str, TickerSignal] = {}
        self._cache_time: datetime | None = None
        logger.info(
            "[Sentiment] SentimentAnalyzer ready (model=%s, batch=%d)",
            _FINBERT_MODEL, self.config.batch_size,
        )

    def get_signals(
        self,
        tickers: list[str],
        force_refresh: bool = False,
    ) -> dict[str, TickerSignal]:
        """
        Return sentiment signals for a list of tickers.

        Uses cached results if within cache_ttl_seconds.
        Fetches fresh news and scores with FinBERT otherwise.

        Parameters
        ----------
        tickers : list[str]
        force_refresh : bool
            Bypass cache and re-fetch (useful for testing)

        Returns
        -------
        dict[str, TickerSignal] keyed by ticker
        """
        now = datetime.now(tz=timezone.utc)

        # Check cache validity
        cache_valid = (
            not force_refresh
            and self._cache_time is not None
            and (now - self._cache_time).total_seconds() < self.config.cache_ttl_seconds
            and all(t in self._cache for t in tickers)
        )

        if cache_valid:
            logger.debug("[Sentiment] Using cached signals (age=%.0fs)",
                         (now - self._cache_time).total_seconds())
            return {t: self._cache[t] for t in tickers if t in self._cache}

        # Fetch + score fresh articles
        articles = fetch_news(tickers, max_articles_per_ticker=10)
        scored   = score_articles(articles, config=self.config)
        signals  = aggregate_signals(scored, tickers, config=self.config)

        # Update cache
        self._cache.update(signals)
        self._cache_time = now

        return signals

    def is_entry_allowed(self, ticker: str, signals: dict[str, TickerSignal]) -> bool:
        """
        Returns True if the sentiment signal allows entry for this ticker.

        Entry is blocked only on explicit SELL signals.
        HOLD and BUY both allow entry — options strategies don't require
        positive news, just the absence of strongly negative news.

        Parameters
        ----------
        ticker : str
        signals : dict[str, TickerSignal]
            Result from get_signals()

        Returns
        -------
        bool
        """
        sig = signals.get(ticker)
        if sig is None:
            logger.debug("[Sentiment] No signal for %s — allowing entry", ticker)
            return True

        if sig.signal == "SELL":
            logger.info(
                "[Sentiment] SELL signal on %s — blocking entry "
                "(score=%.3f, %d articles, top: %s)",
                ticker, sig.weighted_score, sig.article_count,
                sig.top_headline[:80] if sig.top_headline else "(none)",
            )
            return False

        logger.debug(
            "[Sentiment] %s: %s (score=%.3f) — entry allowed",
            ticker, sig.signal, sig.weighted_score,
        )
        return True


# ---------------------------------------------------------------------------
# Prevent NameError on math import (used in score_articles batch calc)
# ---------------------------------------------------------------------------
import math
