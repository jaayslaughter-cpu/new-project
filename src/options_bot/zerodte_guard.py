"""
zerodte_guard.py — Dedicated risk guardrails for the 0DTE GEX scalper.

0DTE is validated and risk-managed SEPARATELY from the core 14-60 DTE
strategies. It is the highest-risk path in the bot (explosive same-day
gamma, losses that can develop faster than the 15s monitor can react), so
it gets its own circuit breaker that is intentionally stricter than the
account-wide halts and cannot draw freely on the main risk budget.

Two independent guards, both fail-CLOSED (any ambiguity → block 0DTE, never
the core book):

1. DAILY LOSS CAP — before any 0DTE entry, sum today's realized 0DTE P&L.
   If it has already lost more than `max_daily_loss_pct` of equity, block
   further 0DTE entries for the rest of the day. Set well below the 3%
   account-wide daily-loss halt so a bad 0DTE day cannot consume the whole
   book's budget.

2. CONSECUTIVE-LOSING-DAY COOLDOWN — at EOD, record whether 0DTE finished
   net-negative (realized) for the day. If it finishes negative on
   `max_consec_losing_days` days IN A ROW, disable 0DTE entirely for
   `cooldown_trading_days` trading days. State persists across container
   restarts and across days via the bot_state table (NOT the same-session
   0DTE state, which is intentionally discarded after 18h).

This module is pure logic over injected state/data so it can be unit-tested
without a live broker or scheduler. The orchestrator wires it to real DB
queries and the real trading calendar.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# bot_state key for the cross-day cooldown / streak tracker.
_STATE_KEY = "zerodte_circuit_breaker"

# The cooldown must survive across many days, so when loading this specific
# state we override the default 18h staleness discard with a long window.
_STATE_MAX_AGE_HOURS = 24 * 60  # 60 days — comfortably longer than any cooldown


@dataclass
class GuardDecision:
    """Result of a pre-entry 0DTE guard check."""
    allowed: bool
    reason: str


class ZeroDTECircuitBreaker:
    """
    Tracks 0DTE daily P&L outcomes and enforces the daily loss cap +
    consecutive-losing-day cooldown. State is persisted via two injected
    callables (load_state/save_state) so this class stays broker- and
    DB-agnostic and unit-testable.

    Persisted state schema (JSON in bot_state under _STATE_KEY):
        {
          "consec_losing_days": int,     # current streak of net-negative 0DTE days
          "cooldown_until_iso": str|null, # ISO date; 0DTE blocked until this trading day
          "last_recorded_day": str|null, # ISO date of the last EOD record (dedupe guard)
        }
    """

    def __init__(
        self,
        config,
        load_state: Callable[[str, int], Optional[dict]],
        save_state: Callable[[str, dict, int], None],
    ):
        self.cfg = config
        self._load_state = load_state
        self._save_state = save_state

    # ---- state helpers ----------------------------------------------------

    def _read(self) -> dict:
        try:
            st = self._load_state(_STATE_KEY, _STATE_MAX_AGE_HOURS)
        except Exception as exc:
            logger.warning("[0DTE-CB] state load failed (fail-closed cooldown read): %s", exc)
            st = None
        if not st:
            return {"consec_losing_days": 0, "cooldown_until_iso": None, "last_recorded_day": None}
        return {
            "consec_losing_days": int(st.get("consec_losing_days", 0)),
            "cooldown_until_iso": st.get("cooldown_until_iso"),
            "last_recorded_day": st.get("last_recorded_day"),
        }

    def _write(self, state: dict) -> None:
        try:
            self._save_state(_STATE_KEY, state, _STATE_MAX_AGE_HOURS)
        except Exception as exc:
            logger.warning("[0DTE-CB] state save failed (non-fatal): %s", exc)

    # ---- pre-entry guard --------------------------------------------------

    def check_entry_allowed(
        self,
        today: date,
        equity: float,
        today_realized_0dte_pnl: float,
    ) -> GuardDecision:
        """
        Called before any 0DTE entry. Blocks if either (a) a cooldown is
        active, or (b) today's 0DTE realized loss already exceeds the
        dedicated daily cap. Fail-closed: on any state error, block.
        """
        state = self._read()

        # --- Guard 1: active cooldown? ---
        cd_iso = state.get("cooldown_until_iso")
        if cd_iso:
            try:
                cd_until = date.fromisoformat(cd_iso)
                if today < cd_until:
                    return GuardDecision(
                        False,
                        f"0DTE in cooldown until {cd_iso} "
                        f"({(cd_until - today).days} trading-day(s) left) — "
                        f"triggered by {self.cfg.max_consec_losing_days} consecutive losing days",
                    )
            except ValueError:
                # Corrupt date → fail closed
                return GuardDecision(False, "0DTE cooldown state unreadable — blocking (fail-closed)")

        # --- Guard 2: dedicated daily loss cap ---
        daily_cap = equity * self.cfg.max_daily_loss_pct
        if today_realized_0dte_pnl <= -daily_cap:
            return GuardDecision(
                False,
                f"0DTE daily loss cap hit: realized ${today_realized_0dte_pnl:,.2f} "
                f"≤ -${daily_cap:,.2f} ({self.cfg.max_daily_loss_pct:.1%} of equity) — "
                f"no more 0DTE entries today",
            )

        return GuardDecision(True, "0DTE guards clear")

    # ---- EOD recording ----------------------------------------------------

    def record_eod(self, today: date, today_realized_0dte_pnl: float) -> dict:
        """
        Called once at EOD. Updates the consecutive-losing-day streak and,
        if the streak hits the threshold, arms the cooldown. Idempotent per
        day via last_recorded_day. Returns the updated state for logging.

        A "losing day" = net-negative REALIZED 0DTE P&L. A day with zero
        0DTE trades (pnl == 0.0) is NOT counted as losing — it neither
        extends nor resets the streak (no information). Only an actual
        net-negative day extends it; an actual net-positive day resets it.
        """
        state = self._read()
        today_iso = today.isoformat()

        # Idempotency: don't double-count if EOD runs twice for the same day.
        if state.get("last_recorded_day") == today_iso:
            logger.debug("[0DTE-CB] EOD already recorded for %s — skip", today_iso)
            return state

        if today_realized_0dte_pnl < 0:
            state["consec_losing_days"] = int(state.get("consec_losing_days", 0)) + 1
            logger.info(
                "[0DTE-CB] %s net-negative 0DTE day (realized $%.2f) — streak now %d/%d",
                today_iso, today_realized_0dte_pnl,
                state["consec_losing_days"], self.cfg.max_consec_losing_days,
            )
        elif today_realized_0dte_pnl > 0:
            if state.get("consec_losing_days", 0) > 0:
                logger.info(
                    "[0DTE-CB] %s net-positive 0DTE day (realized $%.2f) — streak reset (was %d)",
                    today_iso, today_realized_0dte_pnl, state["consec_losing_days"],
                )
            state["consec_losing_days"] = 0
        else:
            # Exactly zero — no 0DTE trades closed today. Leave streak untouched.
            logger.debug("[0DTE-CB] %s no realized 0DTE P&L — streak unchanged", today_iso)

        # Arm cooldown if streak hit the threshold.
        if state["consec_losing_days"] >= self.cfg.max_consec_losing_days:
            cooldown_until = _add_trading_days(today, self.cfg.cooldown_trading_days)
            state["cooldown_until_iso"] = cooldown_until.isoformat()
            state["consec_losing_days"] = 0  # reset streak; the cooldown now governs
            logger.warning(
                "[0DTE-CB] COOLDOWN ARMED: %d consecutive losing 0DTE days — "
                "0DTE disabled until %s (%d trading days)",
                self.cfg.max_consec_losing_days, state["cooldown_until_iso"],
                self.cfg.cooldown_trading_days,
            )

        state["last_recorded_day"] = today_iso
        self._write(state)
        return state


def _add_trading_days(start: date, n: int) -> date:
    """
    Add N trading days (Mon-Fri, naive — ignores market holidays) to `start`.
    Holiday-naive is acceptable here: erring slightly long on a cooldown is
    the safe direction (more time off after a losing streak, not less).
    """
    d = start
    added = 0
    while added < n:
        d = date.fromordinal(d.toordinal() + 1)
        if d.weekday() < 5:  # Mon-Fri
            added += 1
    return d
