"""
Adaptive strategy tuner — learns from closed trade history.

Analyzes closed trades from the database every N trades and adjusts
strategy parameters when performance deviates from targets.

Design principles:
  - Rule-based adaptation, not ML — every adjustment has an explicit reason
  - Hard bounds on all parameters — cannot drift past safe limits
  - Full audit trail — every change logged to Discord and the DB
  - Conservative by default — tightens on bad performance, relaxes slowly
  - Per-strategy tuning — CSP, ShortPutSpread, ShortStrangle tuned independently

Tuning logic per metric window (default: last 20 closed trades):

  Win rate:
    < 40%  → tighten delta (further OTM), raise min_credit, shorten max_dte
    40-50% → tighten delta slightly
    50-65% → no change (target zone)
    > 65% AND profit_factor > 1.8 → relax delta slightly (more trades)

  Profit factor:
    < 0.8  → increase stop_multiplier (hold losers less long)
    0.8-1.0 → small stop tightening
    > 2.0  → may relax stop (but only if win_rate also healthy)

  Average loss size vs average win size:
    If avg_loss > 2x avg_win → something wrong with stop discipline
    → force stop_multiplier back to minimum safe level

All adjustments are additive steps, not jumps. One step per evaluation cycle.
Hard limits enforce the parameter never exceeds safe operational bounds.

Usage (called automatically by orchestrator after every EVAL_INTERVAL trades):
    tuner = AdaptiveTuner(db, discord_webhook)
    new_config = tuner.evaluate_and_tune("short_put_spread", current_config)
    # new_config is a modified copy; original unchanged if no adjustment needed
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning targets and bounds
# ---------------------------------------------------------------------------

# How many closed trades to look back over
EVAL_WINDOW = 20

# How many closed trades must exist before we start tuning at all
MIN_TRADES_TO_TUNE = 10

# Minimum interval between tune cycles (in closed trades)
# Prevents over-fitting to a small recent streak
TUNE_INTERVAL = 10

# Performance targets
TARGET_WIN_RATE_LOW  = 0.45   # below this → tighten
TARGET_WIN_RATE_HIGH = 0.65   # above this (with good PF) → relax slightly
TARGET_PROFIT_FACTOR = 1.0    # minimum acceptable

# Per-parameter hard limits (absolute, never crossed regardless of tuning)
PARAM_LIMITS = {
    # CSP
    "csp_target_delta":    (-0.30, -0.10),   # (most conservative, most aggressive)
    "csp_min_delta":       (-0.40, -0.15),
    "csp_max_delta":       (-0.20, -0.05),
    "csp_stop_multiplier": (1.5,    3.5),
    "csp_min_dte":         (14,     35),
    "csp_max_dte":         (30,     60),

    # ShortPutSpread
    "sps_short_delta":     (-0.35, -0.15),
    "sps_long_delta":      (-0.15, -0.05),
    "sps_stop_multiplier": (1.5,    3.5),
    "sps_min_credit":      (0.25,   2.00),
    "sps_min_dte":         (14,     35),
    "sps_max_dte":         (30,     60),

    # ShortStrangle
    "ss_call_delta":       (0.15,   0.30),
    "ss_put_delta":        (-0.30, -0.15),
    "ss_stop_multiplier":  (2.0,    5.0),
    "ss_min_total_credit": (0.75,   3.00),
    "ss_min_dte":          (21,     45),
    "ss_max_dte":          (45,     75),
}

# Step sizes for each adjustment (one step per cycle)
STEP_SIZES = {
    "delta":           0.02,   # move delta by 0.02 toward/away from ATM
    "stop_multiplier": 0.25,   # increase/decrease stop by 0.25x
    "min_credit":      0.10,   # raise/lower min credit by $0.10
    "dte":             5,      # shift DTE window by 5 days
}


# ---------------------------------------------------------------------------
# Performance snapshot
# ---------------------------------------------------------------------------

@dataclass
class PerfSnapshot:
    """Performance metrics computed over the last EVAL_WINDOW trades."""
    strategy:        str
    window_size:     int
    win_rate:        float
    profit_factor:   float
    avg_win:         float
    avg_loss:        float        # always positive (absolute value)
    win_loss_ratio:  float
    total_pnl:       float
    computed_at:     datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def is_healthy(self) -> bool:
        return (
            self.win_rate   >= TARGET_WIN_RATE_LOW
            and self.profit_factor >= TARGET_PROFIT_FACTOR
        )

    @property
    def is_thriving(self) -> bool:
        return (
            self.win_rate   >= TARGET_WIN_RATE_HIGH
            and self.profit_factor >= 1.8
        )

    @property
    def is_struggling(self) -> bool:
        return self.win_rate < TARGET_WIN_RATE_LOW or self.profit_factor < 0.8

    def to_discord_line(self) -> str:
        status = "✅" if self.is_healthy else ("🚨" if self.is_struggling else "⚠️")
        return (
            f"{status} **{self.strategy}** last {self.window_size} trades: "
            f"win={self.win_rate:.0%} PF={self.profit_factor:.2f} "
            f"avgW=${self.avg_win:.0f} avgL=${self.avg_loss:.0f} "
            f"total=${self.total_pnl:+.0f}"
        )


# ---------------------------------------------------------------------------
# Adjustment record
# ---------------------------------------------------------------------------

@dataclass
class ParamAdjustment:
    """One parameter change applied during a tune cycle."""
    strategy:   str
    param:      str
    old_value:  float
    new_value:  float
    reason:     str
    applied_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def to_discord_line(self) -> str:
        direction = "↗" if self.new_value > self.old_value else "↘"
        return (
            f"  {direction} `{self.param}`: "
            f"{self.old_value:.3f} → {self.new_value:.3f} — {self.reason}"
        )


# ---------------------------------------------------------------------------
# Core tuner
# ---------------------------------------------------------------------------

class AdaptiveTuner:
    """
    Analyzes closed trade history and adjusts strategy parameters.

    Parameters
    ----------
    db : TradeDatabase
        The live trade database instance.
    discord_webhook : str
        Webhook URL for Discord notifications. Empty string = silent.
    eval_window : int
        Number of recent closed trades to analyze (default 20).
    tune_interval : int
        Minimum closed trades between evaluation cycles (default 10).
    """

    def __init__(
        self,
        db,
        discord_webhook: str = "",
        eval_window: int    = EVAL_WINDOW,
        tune_interval: int  = TUNE_INTERVAL,
    ):
        self.db              = db
        self.discord_webhook = discord_webhook
        self.eval_window     = eval_window
        self.tune_interval   = tune_interval
        self._last_tune_count: dict[str, int] = {}   # strategy → trade count at last tune
        self._adjustment_log: list[ParamAdjustment]  = []

    # ------------------------------------------------------------------
    # Public API — called by orchestrator
    # ------------------------------------------------------------------

    def should_evaluate(self, strategy_name: str) -> bool:
        """
        Returns True if enough new closed trades have accumulated since
        the last evaluation cycle for this strategy.
        """
        closed = self._count_closed(strategy_name)
        if closed < MIN_TRADES_TO_TUNE:
            return False
        last = self._last_tune_count.get(strategy_name, 0)
        return (closed - last) >= self.tune_interval

    def evaluate_and_tune(self, strategy_name: str, config) -> tuple:
        """
        Analyze recent performance and return (new_config, snapshot, adjustments).

        The returned config is a modified deepcopy. If no adjustments are needed,
        the copy is identical to the input — always safe to use the return value.

        Parameters
        ----------
        strategy_name : str
            One of: "csp", "short_put_spread", "short_strangle"
        config : CSPConfig | ShortPutSpreadConfig | ShortStrangleConfig
            Current strategy configuration dataclass.

        Returns
        -------
        (new_config, PerfSnapshot, list[ParamAdjustment])
        """
        snap    = self._compute_snapshot(strategy_name)
        new_cfg = deepcopy(config)
        adjustments: list[ParamAdjustment] = []

        if snap is None or snap.window_size < MIN_TRADES_TO_TUNE:
            logger.info(
                "[Adaptive] %s: insufficient history (%s trades) — no tuning",
                strategy_name, snap.window_size if snap else 0
            )
            return new_cfg, snap, adjustments

        logger.info("[Adaptive] %s: %s", strategy_name, snap.to_discord_line())

        # Route to strategy-specific tuner
        if strategy_name == "csp":
            adjustments = self._tune_csp(snap, new_cfg)
        elif strategy_name == "short_put_spread":
            adjustments = self._tune_sps(snap, new_cfg)
        elif strategy_name == "short_strangle":
            adjustments = self._tune_strangle(snap, new_cfg)
        else:
            logger.warning("[Adaptive] Unknown strategy '%s' — no tuning", strategy_name)

        # Update last-tuned count
        self._last_tune_count[strategy_name] = self._count_closed(strategy_name)

        # Notify and log
        if adjustments:
            self._adjustment_log.extend(adjustments)
            self._notify(strategy_name, snap, adjustments)
        else:
            logger.info("[Adaptive] %s: performance healthy — no adjustments", strategy_name)

        return new_cfg, snap, adjustments

    # ------------------------------------------------------------------
    # Strategy-specific tuning logic
    # ------------------------------------------------------------------

    def _tune_csp(self, snap: PerfSnapshot, cfg) -> list[ParamAdjustment]:
        adj = []

        if snap.is_struggling:
            # Move target delta further OTM (more negative = safer)
            if cfg.target_delta > PARAM_LIMITS["csp_target_delta"][0]:
                adj.append(self._adjust(cfg, "target_delta",
                    cfg.target_delta - STEP_SIZES["delta"],
                    "csp_target_delta",
                    f"win_rate={snap.win_rate:.0%} below target — moving further OTM"))

            # Also tighten the delta band
            if cfg.min_delta > PARAM_LIMITS["csp_min_delta"][0]:
                adj.append(self._adjust(cfg, "min_delta",
                    cfg.min_delta - STEP_SIZES["delta"],
                    "csp_min_delta",
                    f"tightening delta band — low win rate"))

            # Loosen stop to give more room
            if cfg.stop_multiplier < PARAM_LIMITS["csp_stop_multiplier"][1]:
                adj.append(self._adjust(cfg, "stop_multiplier",
                    cfg.stop_multiplier + STEP_SIZES["stop_multiplier"],
                    "csp_stop_multiplier",
                    f"PF={snap.profit_factor:.2f} — giving positions more room"))

        elif snap.is_thriving:
            # Slightly closer to ATM (more premium, more trades)
            if cfg.target_delta < PARAM_LIMITS["csp_target_delta"][1]:
                adj.append(self._adjust(cfg, "target_delta",
                    cfg.target_delta + STEP_SIZES["delta"],
                    "csp_target_delta",
                    f"win_rate={snap.win_rate:.0%} PF={snap.profit_factor:.2f} — relaxing slightly"))

        # Stop discipline: if average loss > 2.5x average win, reset stop tighter
        if snap.avg_loss > 2.5 * snap.avg_win and snap.avg_win > 0:
            min_stop = PARAM_LIMITS["csp_stop_multiplier"][0]
            if cfg.stop_multiplier > min_stop:
                adj.append(self._adjust(cfg, "stop_multiplier",
                    max(cfg.stop_multiplier - STEP_SIZES["stop_multiplier"], min_stop),
                    "csp_stop_multiplier",
                    f"avg_loss=${snap.avg_loss:.0f} > 2.5x avg_win=${snap.avg_win:.0f} — tightening stop"))

        return adj

    def _tune_sps(self, snap: PerfSnapshot, cfg) -> list[ParamAdjustment]:
        adj = []

        if snap.is_struggling:
            # Tighten short delta (further OTM)
            if cfg.short_delta > PARAM_LIMITS["sps_short_delta"][0]:
                adj.append(self._adjust(cfg, "short_delta",
                    cfg.short_delta - STEP_SIZES["delta"],
                    "sps_short_delta",
                    f"win_rate={snap.win_rate:.0%} — moving short leg further OTM"))

            # Raise minimum credit requirement
            if cfg.min_credit < PARAM_LIMITS["sps_min_credit"][1]:
                adj.append(self._adjust(cfg, "min_credit",
                    cfg.min_credit + STEP_SIZES["min_credit"],
                    "sps_min_credit",
                    "raising quality bar — only higher-credit setups"))

            # Tighten max spread (narrower width = defined max loss)
            if cfg.max_spread_width > 5.0:
                cfg.max_spread_width = max(cfg.max_spread_width - 2.0, 5.0)
                adj.append(ParamAdjustment(
                    strategy="short_put_spread",
                    param="max_spread_width",
                    old_value=cfg.max_spread_width + 2.0,
                    new_value=cfg.max_spread_width,
                    reason="narrowing max spread width to reduce max loss per trade"
                ))

        elif snap.is_thriving:
            # Allow slightly wider spread (more premium)
            if cfg.short_delta < PARAM_LIMITS["sps_short_delta"][1]:
                adj.append(self._adjust(cfg, "short_delta",
                    cfg.short_delta + STEP_SIZES["delta"],
                    "sps_short_delta",
                    f"win_rate={snap.win_rate:.0%} PF={snap.profit_factor:.2f} — relaxing"))

            # Slightly lower min credit (catch more setups)
            if cfg.min_credit > PARAM_LIMITS["sps_min_credit"][0]:
                adj.append(self._adjust(cfg, "min_credit",
                    cfg.min_credit - STEP_SIZES["min_credit"],
                    "sps_min_credit",
                    "healthy performance — accepting slightly lower credit setups"))

        # Stop discipline check
        if snap.avg_loss > 2.5 * snap.avg_win and snap.avg_win > 0:
            min_stop = PARAM_LIMITS["sps_stop_multiplier"][0]
            if cfg.stop_multiplier > min_stop:
                adj.append(self._adjust(cfg, "stop_multiplier",
                    max(cfg.stop_multiplier - STEP_SIZES["stop_multiplier"], min_stop),
                    "sps_stop_multiplier",
                    f"avg_loss=${snap.avg_loss:.0f} too large — tightening stop"))

        return adj

    def _tune_strangle(self, snap: PerfSnapshot, cfg) -> list[ParamAdjustment]:
        adj = []

        if snap.is_struggling:
            # Move both legs further OTM
            if cfg.call_delta > PARAM_LIMITS["ss_call_delta"][0]:
                adj.append(self._adjust(cfg, "call_delta",
                    cfg.call_delta - STEP_SIZES["delta"],
                    "ss_call_delta",
                    f"win_rate={snap.win_rate:.0%} — call leg further OTM"))

            if cfg.put_delta > PARAM_LIMITS["ss_put_delta"][0]:
                adj.append(self._adjust(cfg, "put_delta",
                    cfg.put_delta - STEP_SIZES["delta"],
                    "ss_put_delta",
                    f"win_rate={snap.win_rate:.0%} — put leg further OTM"))

            # Raise minimum credit (strangles need to pay enough to be worth the risk)
            if cfg.min_total_credit < PARAM_LIMITS["ss_min_total_credit"][1]:
                adj.append(self._adjust(cfg, "min_total_credit",
                    cfg.min_total_credit + STEP_SIZES["min_credit"],
                    "ss_min_total_credit",
                    "raising credit floor — only high-premium strangles"))

        elif snap.is_thriving:
            # Slightly closer to ATM (more premium)
            if cfg.call_delta < PARAM_LIMITS["ss_call_delta"][1]:
                adj.append(self._adjust(cfg, "call_delta",
                    cfg.call_delta + STEP_SIZES["delta"],
                    "ss_call_delta",
                    f"win_rate={snap.win_rate:.0%} PF={snap.profit_factor:.2f} — relaxing"))
            if cfg.put_delta < PARAM_LIMITS["ss_put_delta"][1]:
                adj.append(self._adjust(cfg, "put_delta",
                    cfg.put_delta + STEP_SIZES["delta"],
                    "ss_put_delta",
                    "healthy — relaxing put delta"))

        # Strangles have larger max loss — extra stop discipline
        if snap.avg_loss > 2.0 * snap.avg_win and snap.avg_win > 0:
            min_stop = PARAM_LIMITS["ss_stop_multiplier"][0]
            if cfg.stop_multiplier > min_stop:
                adj.append(self._adjust(cfg, "stop_multiplier",
                    max(cfg.stop_multiplier - STEP_SIZES["stop_multiplier"], min_stop),
                    "ss_stop_multiplier",
                    f"avg_loss=${snap.avg_loss:.0f} too large vs avg_win=${snap.avg_win:.0f}"))

        return adj

    # ------------------------------------------------------------------
    # Performance analysis
    # ------------------------------------------------------------------

    def _compute_snapshot(self, strategy_name: str) -> Optional[PerfSnapshot]:
        """Pull closed trades from DB and compute performance metrics."""
        try:
            with self.db._get_conn() as conn:
                cur = conn.execute(
                    """
                    SELECT realized_pnl FROM trades
                    WHERE strategy = ?
                      AND status IN ('stopped_out','closed_profit_target',
                                     'closed_expiry','closed_external')
                      AND realized_pnl IS NOT NULL
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (strategy_name, self.eval_window)
                )
                rows = [row[0] for row in cur.fetchall()]
        except Exception as exc:
            logger.error("[Adaptive] DB query failed: %s", exc)
            return None

        if len(rows) < MIN_TRADES_TO_TUNE:
            return PerfSnapshot(
                strategy=strategy_name, window_size=len(rows),
                win_rate=0.0, profit_factor=0.0,
                avg_win=0.0, avg_loss=0.0,
                win_loss_ratio=0.0, total_pnl=0.0,
            )

        wins   = [p for p in rows if p > 0]
        losses = [p for p in rows if p < 0]

        win_rate     = len(wins) / len(rows)
        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))
        pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_win      = gross_profit / len(wins)  if wins   else 0.0
        avg_loss     = gross_loss   / len(losses) if losses else 0.0

        return PerfSnapshot(
            strategy       = strategy_name,
            window_size    = len(rows),
            win_rate       = round(win_rate,  4),
            profit_factor  = round(pf,        3),
            avg_win        = round(avg_win,   2),
            avg_loss       = round(avg_loss,  2),
            win_loss_ratio = round(avg_win / avg_loss, 3) if avg_loss > 0 else 0.0,
            total_pnl      = round(sum(rows), 2),
        )

    def _count_closed(self, strategy_name: str) -> int:
        """Total number of closed trades for this strategy in the DB."""
        try:
            with self.db._get_conn() as conn:
                cur = conn.execute(
                    """
                    SELECT COUNT(*) FROM trades
                    WHERE strategy = ?
                      AND status IN ('stopped_out','closed_profit_target',
                                     'closed_expiry','closed_external')
                      AND realized_pnl IS NOT NULL
                    """,
                    (strategy_name,)
                )
                return cur.fetchone()[0]
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adjust(self, cfg, attr: str, new_val: float, limit_key: str, reason: str) -> ParamAdjustment:
        """Apply one clamped parameter adjustment and return the record."""
        lo, hi = PARAM_LIMITS[limit_key]
        old_val  = getattr(cfg, attr)
        clamped  = max(lo, min(hi, round(new_val, 4)))
        setattr(cfg, attr, clamped)

        strategy = limit_key.split("_")[0]   # "csp", "sps", "ss"
        strategy_map = {"csp": "csp", "sps": "short_put_spread", "ss": "short_strangle"}

        logger.info(
            "[Adaptive] %s.%s: %.4f → %.4f (clamped from %.4f) — %s",
            strategy, attr, old_val, clamped, new_val, reason
        )
        return ParamAdjustment(
            strategy  = strategy_map.get(strategy, strategy),
            param     = attr,
            old_value = old_val,
            new_value = clamped,
            reason    = reason,
        )

    def _notify(
        self,
        strategy_name: str,
        snap: PerfSnapshot,
        adjustments: list[ParamAdjustment],
    ) -> None:
        """Send a Discord message summarising the tuning cycle."""
        if not self.discord_webhook:
            return

        lines = [
            f"🧠 **Adaptive tuner — {strategy_name.upper()}**",
            snap.to_discord_line(),
            f"Applied {len(adjustments)} adjustment(s):",
        ]
        for a in adjustments:
            lines.append(a.to_discord_line())

        msg = "\n".join(lines)
        logger.info("[Adaptive] Discord: %s", msg.replace("\n", " | "))

        import urllib.request
        try:
            import json as _json
            payload = _json.dumps({"content": msg}).encode()
            req = urllib.request.Request(
                self.discord_webhook, data=payload,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.warning("[Adaptive] Discord notify failed: %s", exc)

    def adjustment_history(self) -> list[dict]:
        """Return all adjustments made this session as a list of dicts."""
        return [
            {
                "strategy":   a.strategy,
                "param":      a.param,
                "old_value":  a.old_value,
                "new_value":  a.new_value,
                "reason":     a.reason,
                "applied_at": a.applied_at.isoformat(),
            }
            for a in self._adjustment_log
        ]
