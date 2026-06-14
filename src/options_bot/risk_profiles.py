"""
Risk Profile System — tiered trading parameter sets.

Exposes three immutable profiles (LOW / MEDIUM / HIGH) that control every
tunable parameter in the trading system. Instead of editing raw numbers
in config, pick a profile and all parameters adjust consistently.

Extracted from AI-trader/config/risk_profiles.py and adapted for our
Alpaca-based US options model (removed India broker references, converted
lot sizes to contract counts, aligned parameter names with our strategy
configs, updated time filters to PT).

Usage:
    from options_bot.risk_profiles import get_risk_profile, RiskLevel, apply_profile

    profile = get_risk_profile(RiskLevel.MEDIUM)
    print(profile.stop_multiplier)   # 2.0

    # Apply to an existing OrchestratorConfig:
    config = apply_profile(config, RiskLevel.MEDIUM)

Design notes
------------
Each profile is frozen — parameters cannot drift at runtime. The
AdaptiveTuner adjusts within the profile's defined bounds, but cannot
exceed them. Think of the profile as the outer fence and the tuner as
the gate that moves within it.

Profile summaries:
  LOW    Conservative. Max 2 contracts, tight stops, high entry bar.
         Good for: first 3 months of paper trading. Prioritizes capital
         preservation over returns.

  MEDIUM Balanced. Default. Max 3 contracts, standard 2x stop, 50% TP.
         Good for: consistent paper trading with proven edge. Our default.

  HIGH   Aggressive. Max 5 contracts, wider stops, lower entry bar.
         Good for: live trading only after 6+ months of MEDIUM results
         with a Sharpe ratio > 1.0 and profit factor > 1.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class RiskLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass(frozen=True)
class RiskProfile:
    """
    Immutable parameter set for one risk tier.

    All time values are in PT (Pacific Time) — the scheduler timezone.
    Stop/profit percentages are fractions of premium received (e.g. 0.50 = 50%).
    """
    name:  str
    level: RiskLevel

    # ── Position sizing ────────────────────────────────────────────────
    max_contracts:        int    # absolute max contracts per trade
    max_capital_pct:      float  # max fraction of account per trade

    # ── Stop / profit ──────────────────────────────────────────────────
    stop_multiplier:      float  # close when spread costs N× credit (2.0 = 2x)
    profit_target_pct:    float  # close at N% of max profit captured (0.50 = 50%)
    stop_multiplier_min:  float  # adaptive tuner lower bound for stop_multiplier
    stop_multiplier_max:  float  # adaptive tuner upper bound for stop_multiplier

    # ── Trailing stop (for 0DTE monitor) ──────────────────────────────
    trailing_trigger_pct: float  # activate trailing after N% profit
    trailing_floor_pct:   float  # lock in at least N% profit once trailing

    # ── Entry selectivity ──────────────────────────────────────────────
    min_pop:              float  # minimum probability of profit
    min_iv_rank:          float  # minimum IV rank for strangle entry (0 = disabled)
    min_delta:            float  # most aggressive delta allowed (e.g. -0.30)
    max_delta:            float  # most conservative delta allowed (e.g. -0.10)
    min_credit:           float  # minimum net credit per spread

    # ── DTE window ────────────────────────────────────────────────────
    min_dte:              int    # minimum days to expiration
    max_dte:              int    # maximum days to expiration

    # ── Daily limits ──────────────────────────────────────────────────
    max_positions_total:  int    # max open positions at any time
    max_trades_per_day:   int    # max new trades per day

    # ── Time filters (PT) ─────────────────────────────────────────────
    scan_hour:            int    # PT hour for daily scan
    scan_minute:          int    # PT minute for daily scan
    no_entry_after_hour:  int    # PT hour after which no new entries (0DTE)
    eod_close_hour:       int    # PT hour for EOD cleanup
    eod_close_minute:     int    # PT minute for EOD cleanup

    # ── Regime multipliers ─────────────────────────────────────────────
    # Applied to max_contracts when regime is detected
    regime_multipliers:   Dict[str, float]

    # ── Adaptive tuner bounds ──────────────────────────────────────────
    # The tuner cannot move delta outside these bounds regardless of performance
    delta_bound_conservative: float  # furthest OTM the tuner can go
    delta_bound_aggressive:   float  # closest to ATM the tuner can go


# ── LOW — Conservative, capital preservation ──────────────────────────────────
LOW = RiskProfile(
    name  = "Conservative",
    level = RiskLevel.LOW,

    # Position sizing: max 2 contracts, risk ≤ 0.5% of account per trade
    max_contracts     = 2,
    max_capital_pct   = 0.005,

    # Stops: tight (1.5x), take profit at 50%
    stop_multiplier      = 1.5,
    profit_target_pct    = 0.50,
    stop_multiplier_min  = 1.25,
    stop_multiplier_max  = 2.50,

    # Trailing (0DTE): activate at 20% profit, floor at 12%
    trailing_trigger_pct = 0.20,
    trailing_floor_pct   = 0.12,

    # Entry: high bar — 70% PoP minimum, IV rank ≥ 30 for strangles
    min_pop       = 0.70,
    min_iv_rank   = 0.30,
    min_delta     = -0.25,   # sell no more aggressive than 25-delta
    max_delta     = -0.10,   # sell no more conservative than 10-delta
    min_credit    = 0.60,

    # DTE: slightly longer window for more time value
    min_dte = 25,
    max_dte = 50,

    # Daily limits: conservative
    max_positions_total = 2,
    max_trades_per_day  = 2,

    # Schedule (PT): scan at 9:45 AM PT, EOD at 3:45 PM PT
    scan_hour          = 9,
    scan_minute        = 45,
    no_entry_after_hour = 11,   # 11 AM PT = 2 PM ET for 0DTE
    eod_close_hour     = 15,
    eod_close_minute   = 45,

    # Regime: reduce size significantly in bad conditions
    regime_multipliers = {
        "trending":        0.50,   # directional trends are dangerous for short premium
        "mean_reverting":  1.00,   # ideal for short premium
        "high_volatility": 0.25,   # VIX spike — very small
        "unknown":         0.50,
    },

    # Adaptive tuner cannot push delta past these bounds
    delta_bound_conservative = -0.30,
    delta_bound_aggressive   = -0.10,
)


# ── MEDIUM — Balanced, default mode ───────────────────────────────────────────
MEDIUM = RiskProfile(
    name  = "Balanced",
    level = RiskLevel.MEDIUM,

    # Position sizing: max 3 contracts, risk ≤ 1% of account per trade
    max_contracts     = 3,
    max_capital_pct   = 0.010,

    # Stops: standard (2x), take profit at 50%
    stop_multiplier      = 2.0,
    profit_target_pct    = 0.50,
    stop_multiplier_min  = 1.50,
    stop_multiplier_max  = 3.00,

    # Trailing (0DTE): activate at 25% profit, floor at 10%
    trailing_trigger_pct = 0.25,
    trailing_floor_pct   = 0.10,

    # Entry: 65% PoP, IV rank ≥ 25 for strangles
    min_pop       = 0.65,
    min_iv_rank   = 0.25,
    min_delta     = -0.30,
    max_delta     = -0.10,
    min_credit    = 0.50,

    # DTE: standard 21-45 window
    min_dte = 21,
    max_dte = 45,

    # Daily limits
    max_positions_total = 5,
    max_trades_per_day  = 3,

    # Schedule (PT): scan at 9:45 AM PT, EOD at 3:45 PM PT
    scan_hour          = 9,
    scan_minute        = 45,
    no_entry_after_hour = 11,
    eod_close_hour     = 15,
    eod_close_minute   = 45,

    # Regime: balanced
    regime_multipliers = {
        "trending":        0.75,
        "mean_reverting":  1.25,
        "high_volatility": 0.50,
        "unknown":         0.75,
    },

    delta_bound_conservative = -0.35,
    delta_bound_aggressive   = -0.10,
)


# ── HIGH — Aggressive, maximum profit potential ────────────────────────────────
# WARNING: Only use HIGH after 6+ months of MEDIUM paper/live results
# with Sharpe > 1.0 and profit factor > 1.5. The edge in HIGH is 50%
# more contracts vs MEDIUM — same signal quality, more size.
HIGH = RiskProfile(
    name  = "Aggressive",
    level = RiskLevel.HIGH,

    # Position sizing: max 5 contracts, risk ≤ 1.5% of account per trade
    max_contracts     = 5,
    max_capital_pct   = 0.015,

    # Stops: slightly wider (2x), take profit at 50%
    stop_multiplier      = 2.0,
    profit_target_pct    = 0.50,
    stop_multiplier_min  = 1.50,
    stop_multiplier_max  = 3.50,

    # Trailing (0DTE): activate at 20% profit, floor at 8%
    trailing_trigger_pct = 0.20,
    trailing_floor_pct   = 0.08,

    # Entry: slightly lower bar — 60% PoP, IV rank ≥ 20
    min_pop       = 0.60,
    min_iv_rank   = 0.20,
    min_delta     = -0.30,
    max_delta     = -0.10,
    min_credit    = 0.45,

    # DTE: can go slightly shorter for more trades
    min_dte = 18,
    max_dte = 45,

    # Daily limits
    max_positions_total = 8,
    max_trades_per_day  = 5,

    # Schedule (PT)
    scan_hour          = 9,
    scan_minute        = 45,
    no_entry_after_hour = 11,
    eod_close_hour     = 15,
    eod_close_minute   = 45,

    # Regime: more aggressive in favorable conditions
    regime_multipliers = {
        "trending":        1.00,
        "mean_reverting":  1.50,
        "high_volatility": 0.75,
        "unknown":         1.00,
    },

    delta_bound_conservative = -0.35,
    delta_bound_aggressive   = -0.08,
)


_PROFILES = {
    RiskLevel.LOW:    LOW,
    RiskLevel.MEDIUM: MEDIUM,
    RiskLevel.HIGH:   HIGH,
}


def get_risk_profile(level: RiskLevel | str) -> RiskProfile:
    """Return the RiskProfile for a given level."""
    if isinstance(level, str):
        level = RiskLevel(level.lower())
    return _PROFILES[level]


def apply_profile(config, level: RiskLevel | str) -> object:
    """
    Apply a RiskProfile to an OrchestratorConfig, returning the updated config.

    Overwrites: max_positions_total, scan_hour, scan_minute, close_hour,
    close_minute, and strategy_config fields that match the profile.

    Parameters
    ----------
    config : OrchestratorConfig
        The orchestrator configuration dataclass instance.
    level : RiskLevel or str

    Returns
    -------
    The same config object with profile fields applied (mutated in place).
    """
    import logging
    logger = logging.getLogger(__name__)

    profile = get_risk_profile(level)

    # Orchestrator-level fields
    if hasattr(config, "max_positions_total"):
        config.max_positions_total = profile.max_positions_total
    if hasattr(config, "scan_hour"):
        config.scan_hour   = profile.scan_hour
        config.scan_minute = profile.scan_minute
    if hasattr(config, "close_hour"):
        config.close_hour   = profile.eod_close_hour
        config.close_minute = profile.eod_close_minute

    # Strategy-level fields (if strategy_config is present)
    sc = getattr(config, "strategy_config", None)
    if sc is not None:
        for attr, val in [
            ("stop_multiplier",   profile.stop_multiplier),
            ("profit_target_pct", profile.profit_target_pct),
            ("min_pop",           profile.min_pop),
            ("min_credit",        profile.min_credit),
            ("min_dte",           profile.min_dte),
            ("max_dte",           profile.max_dte),
        ]:
            if hasattr(sc, attr):
                setattr(sc, attr, val)

    # 0DTE config
    zc = getattr(config, "zero_dte_config", None)
    if zc is not None:
        for attr, val in [
            ("max_contracts",          profile.max_contracts),
            ("stop_loss_pct",          profile.stop_multiplier - 1.0),
            ("profit_target_pct",      profile.profit_target_pct),
            ("trailing_stop_trigger_pct", profile.trailing_trigger_pct),
            ("trailing_stop_floor_pct",   profile.trailing_floor_pct),
            ("min_credit_absolute",    profile.min_credit),
        ]:
            if hasattr(zc, attr):
                setattr(zc, attr, val)

    logger.info("[RiskProfile] Applied %s profile to config", profile.name)
    return config


def list_profiles() -> list[dict]:
    """Return a summary of all profiles for display."""
    return [
        {
            "level":            p.level.value,
            "name":             p.name,
            "max_contracts":    p.max_contracts,
            "stop_multiplier":  p.stop_multiplier,
            "profit_target":    f"{p.profit_target_pct:.0%}",
            "min_pop":          f"{p.min_pop:.0%}",
            "max_positions":    p.max_positions_total,
            "delta_range":      f"{p.min_delta:.2f} to {p.max_delta:.2f}",
        }
        for p in _PROFILES.values()
    ]
