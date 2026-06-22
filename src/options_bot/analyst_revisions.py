"""Analyst rating-revision signal — net upgrades/downgrades over a window.

GATED FEATURE. Built + tested but DORMANT. Activation mirrors the Iron Condor
pattern: blocked until (1) >=30 closed trades AND (2) an explicit enable flag
(analyst_revisions_enabled in OrchestratorConfig).

Reframed onto data already in the stack: yfinance (already a dependency)
exposes `Ticker.upgrades_downgrades` directly — firm, from-grade, to-grade,
action, date — so no subscription and no scraping.

New edge vs current stack: the bot has FinBERT *headline sentiment* and an
earnings *blackout* filter, but nothing tracking analyst rating *changes*. A
cluster of upgrades (or downgrades) is a distinct, slower-moving signal from
news tone — it confirms or contradicts a directional read, helping avoid
selling premium against a wall of fresh downgrades (capital preservation) and
better timing income trades with analyst momentum.

The fetch is isolated behind a guarded import so this module is importable and
testable even where yfinance/network is unavailable; the scoring core is pure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# --- PROVISIONAL_WEIGHTS (require paper-trading data to calibrate) ---
PROVISIONAL_LOOKBACK_DAYS = 30
# Net-revision score above/below these flips the directional read.
PROVISIONAL_BULLISH_NET = 1.5
PROVISIONAL_BEARISH_NET = -1.5

# Weight a rating change by how far it moves on a normalized grade ladder, so a
# Sell->Buy double-upgrade counts more than Hold->Buy.
_GRADE_LADDER = {
    "strong sell": 0, "sell": 1, "underperform": 1, "reduce": 1,
    "underweight": 1, "hold": 2, "neutral": 2, "market perform": 2,
    "equal-weight": 2, "equal weight": 2, "sector perform": 2,
    "overweight": 3, "buy": 3, "outperform": 3, "accumulate": 3, "add": 3,
    "strong buy": 4,
}


def _grade_value(grade: "str | None") -> "int | None":
    if not grade:
        return None
    return _GRADE_LADDER.get(grade.strip().lower())


@dataclass(frozen=True)
class RevisionEvent:
    date: datetime
    firm: str
    from_grade: "str | None"
    to_grade: "str | None"
    action: "str | None"  # 'up' / 'down' / 'main' / 'init' (yfinance vocab)


@dataclass(frozen=True)
class RevisionSignal:
    ticker: str
    net_score: float            # ladder-weighted net of up vs down moves
    n_upgrades: int
    n_downgrades: int
    direction: str              # 'bullish' / 'bearish' / 'neutral'
    confidence: float           # 0..1, scales with event count
    days: int


def score_revisions(ticker: str, events: "list[RevisionEvent]",
                    now: "datetime | None" = None) -> RevisionSignal:
    """Pure scoring core — no network. Tally ladder-weighted moves in window."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=PROVISIONAL_LOOKBACK_DAYS)

    net = 0.0
    ups = downs = 0
    for e in events:
        if e.date < cutoff:
            continue
        fv, tv = _grade_value(e.from_grade), _grade_value(e.to_grade)
        if fv is not None and tv is not None:
            delta = tv - fv
        else:  # fall back to the action verb when grades don't map
            delta = {"up": 1, "down": -1}.get((e.action or "").lower(), 0)
        if delta > 0:
            ups += 1
        elif delta < 0:
            downs += 1
        net += delta

    if net >= PROVISIONAL_BULLISH_NET:
        direction = "bullish"
    elif net <= PROVISIONAL_BEARISH_NET:
        direction = "bearish"
    else:
        direction = "neutral"

    total = ups + downs
    confidence = min(1.0, total / 5.0)  # ~5 events saturates confidence
    return RevisionSignal(ticker, round(net, 2), ups, downs,
                          direction, round(confidence, 2),
                          PROVISIONAL_LOOKBACK_DAYS)


def fetch_revisions(ticker: str) -> "list[RevisionEvent]":
    """Network fetch via yfinance. Guarded — returns [] on any failure so the
    bot degrades gracefully rather than raising into the trade path."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        df = tk.upgrades_downgrades
        if df is None or df.empty:
            return []
        out: list[RevisionEvent] = []
        for idx, row in df.iterrows():
            dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            out.append(RevisionEvent(
                date=dt,
                firm=str(row.get("Firm", "")),
                from_grade=row.get("FromGrade") or None,
                to_grade=row.get("ToGrade") or None,
                action=row.get("Action") or None,
            ))
        return out
    except Exception:
        return []
