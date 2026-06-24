"""
Macro-event blackout gate — don't open new short-premium positions into a
scheduled high-impact macro event.

WHY (capital preservation)
--------------------------
Every strategy in this book is net short gamma/vega. Opening a fresh spread or
strangle the day before an FOMC decision (or a CPI / jobs print) means selling
premium straight into a *scheduled* volatility event — the underlying gaps, IV
crushes or explodes, and a position entered hours earlier can blow through its
risk budget before any management logic runs. The bot already blacks out
*company earnings*; this adds the *macro* calendar, which is what actually moves
the index ETFs that make up the core universe (SPY/QQQ/IWM/sector ETFs).

This is a pure RISK-REDUCER: it can only ever *prevent* an entry, never create
one. It is therefore NOT milestone-gated like the signal features — you want
event protection active during the paper-trading evaluation itself, so the edge
estimate isn't polluted by event-driven entries. It is dormant by default
(macro_blackout_enabled=False) and fail-open: any calendar/parse error allows
the trade rather than halting the bot.

DATA SOURCE
-----------
FOMC decision dates are the reliable backbone — published a year-plus ahead,
they change rarely, and they are the single highest-impact scheduled event for
index ETFs. They are baked in below (2026-2027) and trivially updated. Other
recurring prints (CPI, the jobs report, PCE, GDP) are higher-maintenance and
their dates vary, so they are supplied via config (`macro_blackout_extra_events`)
rather than scraped from a fragile endpoint. A future upgrade can auto-refresh
those from the FRED release calendar (already an integrated data source); that
hook is intentionally kept out of the hot path for v1.

Not applied to the 0DTE GEX scalper (separate intraday fast path; its own
filters already stand it down on volatile days).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


# FOMC interest-rate DECISION dates (day 2 of each meeting — the 2:00pm ET
# announcement is the vol event). Source: federalreserve.gov FOMC calendars.
# Update annually; 2027 is already published. Verify against federalreserve.gov.
_FOMC_DECISION_DATES: tuple[str, ...] = (
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    # 2027
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
)


@dataclass(frozen=True)
class MacroEvent:
    on: date
    label: str
    impact: str = "high"   # high | medium | low


@dataclass(frozen=True)
class MacroBlackoutResult:
    in_blackout: bool
    event_label: str | None = None
    event_date: date | None = None
    days_until: int | None = None   # 0 = event is today


def _parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError, TypeError):
        return None


def default_events() -> list[MacroEvent]:
    """The baked-in reliable backbone: FOMC decision days."""
    out: list[MacroEvent] = []
    for s in _FOMC_DECISION_DATES:
        d = _parse_date(s)
        if d is not None:
            out.append(MacroEvent(on=d, label=f"FOMC decision {d.isoformat()}", impact="high"))
    return out


def parse_extra_events(items) -> list[MacroEvent]:
    """Parse config-supplied events. Each item is 'YYYY-MM-DD' or
    'YYYY-MM-DD:Label' (e.g. '2026-07-15:CPI'). Bad items are skipped
    (fail-open per item)."""
    out: list[MacroEvent] = []
    for item in (items or ()):
        try:
            raw = str(item)
            if ":" in raw:
                ds, label = raw.split(":", 1)
            else:
                ds, label = raw, "macro event"
            d = _parse_date(ds)
            if d is not None:
                out.append(
                    MacroEvent(on=d, label=f"{label.strip()} {d.isoformat()}", impact="high")
                )
        except Exception:  # noqa: BLE001 — fail-open per item
            continue
    return out


def _eastern_today() -> date:
    """Current US/Eastern calendar date (market tz), fail-open to system UTC date."""
    try:
        import pytz
        return datetime.now(pytz.timezone("US/Eastern")).date()
    except Exception:  # noqa: BLE001
        return datetime.now(timezone.utc).date()


def check_macro_blackout(
    as_of: date | None = None,
    lookahead_days: int = 1,
    extra_events=None,
    events: list[MacroEvent] | None = None,
) -> MacroBlackoutResult:
    """Return whether ``as_of`` falls within the blackout window of a high-impact
    macro event: the ``lookahead_days`` days BEFORE the event through the event
    day itself (inclusive). The window deliberately does NOT extend past the
    event — the post-event vol collapse is favorable for premium sellers, so
    there's no reason to keep blocking after it.

    Fail-open: on any error returns in_blackout=False (entry allowed).
    """
    try:
        today = as_of or _eastern_today()
        cal = list(events) if events is not None else default_events()
        cal.extend(parse_extra_events(extra_events))
        window = max(0, int(lookahead_days))

        hit: MacroEvent | None = None
        hit_days: int | None = None
        for ev in cal:
            delta = (ev.on - today).days
            if 0 <= delta <= window:
                if hit is None or delta < hit_days:
                    hit, hit_days = ev, delta
        if hit is not None:
            return MacroBlackoutResult(
                in_blackout=True,
                event_label=hit.label,
                event_date=hit.on,
                days_until=hit_days,
            )
        return MacroBlackoutResult(in_blackout=False)
    except Exception as exc:  # noqa: BLE001 — never halt trading on a calendar bug
        logger.warning(
            "[MacroBlackout] check failed (%s) — fail-open (entry allowed)", exc
        )
        return MacroBlackoutResult(in_blackout=False)
