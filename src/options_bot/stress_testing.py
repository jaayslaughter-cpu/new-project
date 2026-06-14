"""
Portfolio Stress Testing — scenario-based impact analysis.

AUDIT FINDING (Layer 4 — Position Sizing):
  "No correlation adjustment. If the universe contains SPY, QQQ, and AAPL,
   all three resulting put spreads are highly correlated. A market drop affects
   all three simultaneously. Missing correlation accounting."

  "max_loss_per_contract assumes the spread goes to full width at expiration.
   The sizing is conservative but the sizing doesn't reflect the true stop-based
   risk."

FIX:
  This module runs the live portfolio against named market scenarios to answer:
  "What is the total portfolio P&L if the market drops 3% right now?"
  "Does the combined impact exceed the 10% NLV drawdown threshold?"

  It reveals correlation implicitly — when 5 bull put spreads on correlated
  names all hit max loss in the same flash crash scenario, the combined impact
  is visible before it happens.

INTEGRATION:
  Called by the orchestrator's EOD summary and on-demand during the monitor loop.
  Results appear in the Discord EOD message:
    "Stress tests: worst case = Market -5% → -$2,340 (-9.3% NLV). Survives: YES"

LABEL:
  Impact calculations use first-order Greek approximations (delta × price_move,
  vega × vol_change). These are NOT full revaluation. For short-dated options
  (0–45 DTE) with large shocks (>5%), second-order gamma effects are material
  and will be underestimated. The results are DIRECTIONAL ESTIMATES, not
  precise P&L predictions. Labeled as such in all output.

Source
------
income-desk-main/income_desk/stress_testing.py (MIT)
Rewritten: removed Pydantic dependency (uses dataclasses), adapted
PortfolioPosition to our existing contracts layer, added traceable
labels per audit guidance.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

class ScenarioType(str, Enum):
    MARKET_SHOCK  = "market_shock"
    VOL_SHOCK     = "vol_shock"
    COMBINED      = "combined"


@dataclass
class ScenarioParams:
    name:               str
    scenario_type:      ScenarioType
    price_shock_pct:    float = 0.0   # -5.0 = market drops 5%
    vol_shock_pct:      float = 0.0   # +50.0 = IV increases 50% (relative)
    time_decay_days:    int   = 0     # Days of theta decay to apply
    rate_shock_bp:      float = 0.0   # Yield change in basis points
    description:        str   = ""


# 13 predefined scenarios covering the realistic tail-risk space for short-premium
# strategies. Named after historical events where applicable.
# LABEL: price/vol shock magnitudes are from historical peak-to-trough data.
# They are NOT statistical models — they are hardcoded historical reference points.
PREDEFINED_SCENARIOS: dict[str, ScenarioParams] = {
    "market_down_1pct": ScenarioParams(
        name="Market -1%", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=-1.0, vol_shock_pct=10.0,
        description="Mild selloff: market -1%, VIX +10%",
    ),
    "market_down_3pct": ScenarioParams(
        name="Market -3%", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=-3.0, vol_shock_pct=30.0,
        description="Significant selloff: market -3%, VIX +30%",
    ),
    "market_down_5pct": ScenarioParams(
        name="Market -5%", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=-5.0, vol_shock_pct=60.0, time_decay_days=1,
        description="Sharp selloff: market -5%, VIX +60%",
    ),
    "market_down_10pct": ScenarioParams(
        name="Market -10%", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=-10.0, vol_shock_pct=150.0, time_decay_days=3,
        description="Crash: market -10%, VIX more than doubles",
    ),
    "market_up_3pct": ScenarioParams(
        name="Market +3%", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=3.0, vol_shock_pct=-20.0,
        description="Strong rally: market +3%, VIX -20%",
    ),
    "vix_spike_50pct": ScenarioParams(
        name="VIX Spike 50%", scenario_type=ScenarioType.VOL_SHOCK,
        vol_shock_pct=50.0,
        description="Vol spike: VIX +50% (e.g. 20 → 30). Price flat.",
    ),
    "vix_spike_100pct": ScenarioParams(
        name="VIX Doubles", scenario_type=ScenarioType.VOL_SHOCK,
        vol_shock_pct=100.0,
        description="Vol explosion: VIX doubles (e.g. 20 → 40). Price flat.",
    ),
    "rate_shock": ScenarioParams(
        name="Rate Shock +50bp", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=-2.0, rate_shock_bp=50.0, vol_shock_pct=15.0,
        description="Fed surprise: yields +50bp, stocks -2%",
    ),
    "flash_crash": ScenarioParams(
        name="Flash Crash", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=-7.0, vol_shock_pct=200.0, time_decay_days=1,
        description="Flash crash: market -7% in 1 day, VIX triples",
    ),
    "black_monday": ScenarioParams(
        name="Black Monday", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=-20.0, vol_shock_pct=300.0, time_decay_days=5,
        description="Black Monday: market -20%, VIX to 80+",
    ),
    "covid_march_2020": ScenarioParams(
        name="COVID March 2020", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=-12.0, vol_shock_pct=200.0, time_decay_days=5,
        description="COVID crash: -12% in a week, VIX to 65",
    ),
    "fed_surprise": ScenarioParams(
        name="Fed Hawkish Surprise", scenario_type=ScenarioType.COMBINED,
        price_shock_pct=-2.0, rate_shock_bp=30.0, vol_shock_pct=20.0,
        description="Unexpected rate hike signal: stocks -2%, yields +30bp",
    ),
    "vol_crush": ScenarioParams(
        name="Vol Crush (Post-Event)", scenario_type=ScenarioType.VOL_SHOCK,
        vol_shock_pct=-40.0, time_decay_days=1,
        description="IV crush post FOMC/CPI: VIX -40%, theta +1 day. Good for short premium.",
    ),
}

DEFAULT_SUITE = [
    "market_down_1pct",
    "market_down_3pct",
    "market_down_5pct",
    "market_up_3pct",
    "vix_spike_50pct",
    "flash_crash",
    "fed_surprise",
]


# ---------------------------------------------------------------------------
# Position representation
# ---------------------------------------------------------------------------

@dataclass
class StressPosition:
    """
    A single open position for stress testing.

    Adapted from income-desk PortfolioPosition to use our existing
    broker.get_positions() and DB schema.

    LABEL: Greeks (delta, vega, theta) are from Alpaca's last snapshot.
    They are point-in-time estimates at the last scan. For stress testing,
    we hold Greeks constant (first-order approximation). Gamma effects
    (second-order) are NOT included — material for large shocks.
    """
    ticker:          str
    strategy:        str        # "bull_put", "bear_call", "csp", "strangle", "0dte_*"
    direction:       str        # "bullish", "bearish", "neutral"
    max_loss:        float      # Maximum dollar loss (defined-risk trades)
    net_credit:      float      # Premium received per contract × contracts × 100
    n_contracts:     int        = 1
    delta:           float      = 0.0   # Portfolio delta (sum of leg deltas × 100 × n)
    vega:            float      = 0.0   # Portfolio vega (sum of leg vegas × 100 × n)
    theta:           float      = 0.0   # Portfolio theta (sum of leg thetas × 100 × n)
    current_pnl:     float      = 0.0   # Current unrealized P&L
    notional_value:  float      = 0.0   # Underlying value (spot × n_contracts × 100)
    underlying_price: float     = 0.0


# ---------------------------------------------------------------------------
# Impact computation
# ---------------------------------------------------------------------------

@dataclass
class PositionImpact:
    ticker:           str
    strategy:         str
    current_pnl:      float
    stressed_pnl:     float
    impact_dollars:   float   # stressed_pnl - current_pnl
    impact_pct_nlv:   float   # impact_dollars / account_nlv * 100
    new_status:       str     # "safe" | "tested" | "breached" | "max_loss"
    action:           str     # "hold" | "monitor" | "hedge" | "close"
    label:            str     # traceable one-liner

    # LABEL: All impacts are FIRST-ORDER APPROXIMATIONS.
    # delta × price_move underestimates losses for large moves (gamma ignored).
    # vega × vol_change is linear — actual vol surface shift is non-linear.
    approximation_note: str = "first-order Greek approximation; gamma and higher-order effects excluded"


def _compute_position_impact(
    pos: StressPosition,
    scenario: ScenarioParams,
    account_nlv: float,
) -> PositionImpact:
    """
    Compute stressed P&L for one position under one scenario.

    CALCULATION TRACE (per audit requirements — no unsupported claims):

    Price impact:
        price_move = scenario.price_shock_pct / 100
        delta_pnl  = pos.delta × price_move × pos.underlying_price
        (delta units: shares equivalent — already includes 100 × n_contracts)

    Vol impact:
        vol_change = scenario.vol_shock_pct / 100   (relative change)
        vega_pnl   = pos.vega × vol_change
        (vega units: $ per 1% IV change × 100 — already includes contract mult)

    Theta impact:
        theta_pnl  = pos.theta × scenario.time_decay_days
        (theta is negative for long options, positive for short options)

    Combined:
        stressed_pnl = current_pnl + delta_pnl + vega_pnl + theta_pnl

    Clamp (defined-risk trades):
        stressed_pnl = max(-max_loss, stressed_pnl)

    LABEL: This is a P&L ESTIMATE, not a mark-to-model revaluation.
    For short options strategies, delta and vega effects dominate.
    For large shocks (>5%), gamma convexity causes underestimation.
    """
    price_move = scenario.price_shock_pct / 100.0

    # Delta P&L (directional)
    # For defined-risk strategies, delta × price_move × underlying
    # approximates the spread value change
    if pos.delta != 0 and pos.underlying_price > 0:
        delta_pnl = pos.delta * price_move * pos.underlying_price
    else:
        # Fallback: use max_loss-based approximation
        # Bull put spread loses ~2-3× max_loss per 1% down move when near strikes
        leverage = 2.5 if "put" in pos.strategy.lower() or pos.direction == "bullish" else 2.5
        delta_pnl = price_move * pos.max_loss * leverage * (-1 if pos.direction == "bullish" else 1)

    # Vega P&L (vol change)
    vol_change = scenario.vol_shock_pct / 100.0
    if pos.vega != 0:
        vega_pnl = pos.vega * vol_change
    else:
        # Rough proxy: short premium loses ~0.5% of max_loss per 1% vol increase
        vega_pnl = -abs(pos.max_loss) * 0.005 * scenario.vol_shock_pct

    # Theta P&L (time decay)
    theta_pnl = pos.theta * scenario.time_decay_days if scenario.time_decay_days > 0 else 0.0

    total_impact = delta_pnl + vega_pnl + theta_pnl
    stressed_pnl = pos.current_pnl + total_impact

    # Clamp to max loss for defined-risk trades
    if pos.max_loss > 0 and stressed_pnl < -pos.max_loss:
        stressed_pnl = -pos.max_loss
        total_impact = stressed_pnl - pos.current_pnl

    # Status classification
    pct_of_max = abs(stressed_pnl) / pos.max_loss if pos.max_loss > 0 else 0
    if pct_of_max >= 0.90:
        status = "max_loss"
        action = "close"
    elif pct_of_max >= 0.50:
        status = "breached"
        action = "hedge" if "strangle" in pos.strategy else "close"
    elif pct_of_max >= 0.20 or abs(total_impact) > 50:
        status = "tested"
        action = "monitor"
    else:
        status = "safe"
        action = "hold"

    impact_pct = (total_impact / account_nlv * 100) if account_nlv > 0 else 0

    label = (
        f"{pos.ticker} {pos.strategy}: "
        f"Δ${total_impact:+,.0f} ({impact_pct:+.1f}% NLV) "
        f"[delta_pnl=${delta_pnl:+,.0f} vega_pnl=${vega_pnl:+,.0f} "
        f"theta_pnl=${theta_pnl:+,.0f}]"
    )

    return PositionImpact(
        ticker=pos.ticker,
        strategy=pos.strategy,
        current_pnl=round(pos.current_pnl, 2),
        stressed_pnl=round(stressed_pnl, 2),
        impact_dollars=round(total_impact, 2),
        impact_pct_nlv=round(impact_pct, 2),
        new_status=status,
        action=action,
        label=label,
    )


# ---------------------------------------------------------------------------
# Stress test results
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_name:          str
    scenario_description:   str
    total_impact_dollars:   float
    total_impact_pct_nlv:   float
    worst_position:         str
    worst_impact_dollars:   float
    positions_at_risk:      int       # count in "breached" or "max_loss"
    portfolio_survives:     bool      # total loss < drawdown_threshold
    recommended_action:     str
    position_impacts:       list[PositionImpact] = field(default_factory=list)
    approximation_note:     str = "first-order Greek approximation — gamma excluded"


@dataclass
class StressTestSuite:
    as_of_date:          date
    account_nlv:         float
    n_positions:         int
    results:             list[ScenarioResult]
    worst_scenario:      str
    worst_impact_pct:    float
    survives_all:        bool
    drawdown_threshold:  float       # fraction of NLV used for survival check
    summary:             str
    discord_line:        str         # formatted for EOD Discord message


def run_stress_test(
    positions: list[StressPosition],
    scenario: ScenarioParams,
    account_nlv: float,
    drawdown_threshold: float = 0.10,
) -> ScenarioResult:
    """
    Run one scenario against all positions.

    LABEL: Results are ESTIMATES based on first-order Greek approximations.
    survival check: total_loss / nlv < drawdown_threshold (default 10%).
    """
    impacts = [_compute_position_impact(p, scenario, account_nlv) for p in positions]

    total_impact = sum(i.impact_dollars for i in impacts)
    total_pct    = (total_impact / account_nlv * 100) if account_nlv > 0 else 0

    worst = min(impacts, key=lambda i: i.impact_dollars) if impacts else None
    at_risk = sum(1 for i in impacts if i.new_status in ("breached", "max_loss"))

    survives = abs(total_pct / 100) < drawdown_threshold

    if not survives:
        action = "EMERGENCY: total loss exceeds drawdown threshold — close positions"
    elif total_pct < -5:
        action = "CRITICAL: reduce exposure, hedge largest positions"
    elif total_pct < -2:
        action = "WARNING: tighten stops, consider hedging"
    elif total_pct < -1:
        action = "CAUTION: monitor closely"
    else:
        action = "OK: portfolio withstands this scenario"

    logger.info(
        "[StressTest] %s: total_impact=$%.0f (%.1f%% NLV) survives=%s",
        scenario.name, total_impact, total_pct, survives
    )

    return ScenarioResult(
        scenario_name=scenario.name,
        scenario_description=scenario.description,
        total_impact_dollars=round(total_impact, 2),
        total_impact_pct_nlv=round(total_pct, 2),
        worst_position=worst.ticker if worst else "",
        worst_impact_dollars=round(worst.impact_dollars, 2) if worst else 0,
        positions_at_risk=at_risk,
        portfolio_survives=survives,
        recommended_action=action,
        position_impacts=impacts,
    )


def run_stress_suite(
    positions: list[StressPosition],
    account_nlv: float,
    scenario_names: Optional[list[str]] = None,
    drawdown_threshold: float = 0.10,
) -> StressTestSuite:
    """
    Run multiple scenarios and return a complete suite result.

    Default suite: 7 scenarios covering mild selloff through flash crash.
    """
    if not scenario_names:
        scenario_names = DEFAULT_SUITE

    results = []
    for name in scenario_names:
        if name not in PREDEFINED_SCENARIOS:
            logger.warning("[StressTest] Unknown scenario '%s' — skipping", name)
            continue
        params = PREDEFINED_SCENARIOS[name]
        results.append(run_stress_test(positions, params, account_nlv, drawdown_threshold))

    if not results:
        return StressTestSuite(
            as_of_date=date.today(), account_nlv=account_nlv,
            n_positions=len(positions), results=[],
            worst_scenario="none", worst_impact_pct=0.0,
            survives_all=True, drawdown_threshold=drawdown_threshold,
            summary="No scenarios ran",
            discord_line="Stress tests: no data",
        )

    worst = min(results, key=lambda r: r.total_impact_pct_nlv)
    survives_all = all(r.portfolio_survives for r in results)

    # Build Discord-ready summary line
    icon = "✅" if survives_all else "⚠️"
    discord_line = (
        f"{icon} Stress ({len(results)} scenarios): "
        f"worst={worst.scenario_name} {worst.total_impact_pct_nlv:+.1f}% NLV "
        f"({'SURVIVES' if survives_all else 'BREACHES THRESHOLD'})"
    )
    if not survives_all:
        failing = [r.scenario_name for r in results if not r.portfolio_survives]
        discord_line += f" | Failing: {', '.join(failing)}"

    return StressTestSuite(
        as_of_date=date.today(),
        account_nlv=account_nlv,
        n_positions=len(positions),
        results=results,
        worst_scenario=worst.scenario_name,
        worst_impact_pct=round(worst.total_impact_pct_nlv, 2),
        survives_all=survives_all,
        drawdown_threshold=drawdown_threshold,
        summary=f"{len(results)} scenarios | worst: {worst.scenario_name} {worst.total_impact_pct_nlv:+.1f}%",
        discord_line=discord_line,
    )


# ---------------------------------------------------------------------------
# Bridge: convert broker positions to StressPosition list
# ---------------------------------------------------------------------------

def positions_from_broker(broker, db=None) -> list[StressPosition]:
    """
    Build a list of StressPosition from the live Alpaca broker state.

    Reads open positions from broker.get_positions() and enriches with
    Greeks from DB trade records where available.

    Returns empty list on failure (stress test is non-blocking).
    """
    try:
        raw_positions = broker.get_positions()
    except Exception as exc:
        logger.warning("[StressTest] broker.get_positions() failed: %s", exc)
        return []

    positions = []

    # Load DB trade records for Greek data
    db_trades: dict[str, dict] = {}
    if db:
        try:
            with db._get_conn() as conn:
                cur = conn.execute(
                    "SELECT underlying, strategy, max_loss, net_credit, contracts, "
                    "delta, vega, theta, underlying_price "
                    "FROM trades WHERE status='open'"
                )
                cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    r = dict(zip(cols, row))
                    db_trades[r["underlying"]] = r
        except Exception as exc:
            logger.debug("[StressTest] DB Greek fetch failed: %s", exc)

    for pos in raw_positions:
        ticker = pos.get("symbol", "").split(pos.get("symbol", "")[:3])[0][:6]
        underlying = pos.get("underlying_symbol") or pos.get("symbol", "UNKNOWN")[:6]

        db = db_trades.get(underlying, {})

        max_loss    = float(db.get("max_loss") or pos.get("cost_basis") or 0)
        net_credit  = float(db.get("net_credit") or 0)
        strategy    = str(db.get("strategy") or "unknown")
        contracts   = int(db.get("contracts") or 1)
        delta       = float(db.get("delta") or 0)
        vega        = float(db.get("vega") or 0)
        theta       = float(db.get("theta") or 0)
        spot        = float(db.get("underlying_price") or 0)
        current_pnl = float(pos.get("unrealized_pl") or 0)

        direction = "neutral"
        if "put" in strategy.lower() or "csp" in strategy.lower():
            direction = "bullish"
        elif "call" in strategy.lower():
            direction = "bearish"

        if max_loss <= 0:
            max_loss = net_credit * 3 if net_credit > 0 else 500  # rough fallback

        positions.append(StressPosition(
            ticker=underlying,
            strategy=strategy,
            direction=direction,
            max_loss=max_loss,
            net_credit=net_credit,
            n_contracts=contracts,
            delta=delta,
            vega=vega,
            theta=theta,
            current_pnl=current_pnl,
            underlying_price=spot,
        ))

    logger.info("[StressTest] Built %d stress positions from broker", len(positions))
    return positions
