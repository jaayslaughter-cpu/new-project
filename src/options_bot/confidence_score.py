"""
ConfidenceScorer — per-trade and system-wide confidence scoring.

Aggregates signals from every pipeline layer into seven section scores
(0–100 each) and one overall score (0–100). Scores are computed at signal
evaluation time and attached to StrategySignal metadata, logged to Discord
in the EOD summary, and used by the orchestrator to suppress low-confidence
trades in high-risk market conditions.

Section definitions
-------------------
1. REGIME          — How well does the current market environment suit
                     short-premium strategies? Based on VIX, yield curve,
                     Hurst exponent, breadth.

2. SIGNAL QUALITY  — How clean and reliable are the technical and IV inputs?
                     Based on TechnicalScore, IV contamination level,
                     structural break recency.

3. ENTRY INTEL     — Independent confirmation signals: insider buying,
                     policy catalyst alignment, sentiment, GEX regime.

4. STRATEGY FIT    — Does this specific contract set up well? Based on
                     probability of profit, delta quality, DTE, credit ratio.

5. RISK POSTURE    — How much headroom does the risk layer have? Based on
                     daily P&L, open positions, max-loss budget remaining.

6. EXECUTION       — Can we actually get filled at a good price? Based on
                     bid/ask spread quality, live mid availability, API health.

7. TRACK RECORD    — What does recent performance say? Based on win rate,
                     profit factor, and N-trade sample size confidence.

Aggregation
-----------
Overall = weighted average of all seven sections.
Weights are regime-adaptive: high-volatility regimes upweight risk posture
and execution; trending regimes upweight signal quality and strategy fit.

Interpretation
--------------
  90–100  VERY HIGH    — all systems aligned, proceed normally
  75–89   HIGH         — strong setup, proceed with standard sizing
  60–74   MODERATE     — proceed but log the gaps; consider 75% size
  45–59   LOW          — borderline; skip unless top-tier ticker
  < 45    VERY LOW     — do not trade; surface via Discord warning
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section score dataclass
# ---------------------------------------------------------------------------

@dataclass
class SectionScore:
    """One component of the overall confidence score."""
    name:        str
    score:       float          # 0–100
    weight:      float          # relative weight in overall (sum of weights = 1.0)
    signals:     dict[str, Any] = field(default_factory=dict)   # raw inputs
    detail:      str = ""       # human-readable explanation

    @property
    def weighted(self) -> float:
        return self.score * self.weight

    def emoji(self) -> str:
        if self.score >= 80: return "🟢"
        if self.score >= 60: return "🟡"
        if self.score >= 45: return "🟠"
        return "🔴"

    def discord_line(self) -> str:
        bar = _bar(self.score)
        return (
            f"{self.emoji()} **{self.name}** `{self.score:.0f}/100`  {bar}\n"
            f"   _{self.detail}_"
        )


@dataclass
class ConfidenceReport:
    """Complete confidence assessment for one trade signal."""
    ticker:          str
    strategy:        str
    sections:        list[SectionScore]
    overall:         float          # 0–100, weighted average
    computed_at:     datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Convenience
    @property
    def grade(self) -> str:
        if self.overall >= 90: return "VERY HIGH"
        if self.overall >= 75: return "HIGH"
        if self.overall >= 60: return "MODERATE"
        if self.overall >= 45: return "LOW"
        return "VERY LOW"

    @property
    def should_trade(self) -> bool:
        return self.overall >= 45

    def emoji(self) -> str:
        if self.overall >= 75: return "🟢"
        if self.overall >= 60: return "🟡"
        if self.overall >= 45: return "🟠"
        return "🔴"

    def get_section(self, name: str) -> Optional[SectionScore]:
        return next((s for s in self.sections if s.name == name), None)

    def to_dict(self) -> dict:
        return {
            "ticker":     self.ticker,
            "strategy":   self.strategy,
            "overall":    round(self.overall, 1),
            "grade":      self.grade,
            "sections":   {s.name: round(s.score, 1) for s in self.sections},
            "computed_at": self.computed_at.isoformat(),
        }

    def discord_message(self) -> str:
        """Full Discord-ready confidence report for EOD or per-trade alerts."""
        lines = [
            f"{self.emoji()} **CONFIDENCE SCORE — {self.ticker} {self.strategy.upper()}**",
            f"Overall: `{self.overall:.0f}/100`  [{self.grade}]  {_bar(self.overall)}",
            "",
        ]
        for s in self.sections:
            lines.append(s.discord_line())
        lines.append("")
        lines.append(
            f"_Computed {self.computed_at.strftime('%H:%M:%S UTC')} "
            f"{'— ✅ PROCEED' if self.should_trade else '— ⛔ SKIP'}_"
        )
        return "\n".join(lines)

    def short_line(self) -> str:
        """One-line summary for trade entry Discord alert."""
        section_str = "  ".join(
            f"{s.name[:4]}={s.score:.0f}" for s in self.sections
        )
        return (
            f"{self.emoji()} Confidence `{self.overall:.0f}` [{self.grade}]  "
            f"|  {section_str}"
        )


# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------

def _bar(score: float, width: int = 10) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Section scorers
# ---------------------------------------------------------------------------

def _score_regime(regime: dict) -> SectionScore:
    """
    Section 1: Market Regime  (weight 0.20)

    Short-premium strategies profit most in low/stable vol, mean-reverting,
    positively-sloped yield curve environments.

    Inputs from regime.detect():
      vix_level        — raw VIX
      vix_percentile   — VIX percentile (0-100)
      vix_trend        — 'falling'|'stable'|'rising'
      yield_curve_slope — 10Y-2Y spread (positive = normal, negative = inverted)
      hurst            — 0.5 = random walk, >0.5 = trending, <0.5 = mean-reverting
      breadth_composite — 0-1 market breadth
      regime           — classified regime string
    """
    score = 100.0
    signals = {}
    parts = []

    vix     = regime.get("vix_level", 20.0)
    vix_pct = regime.get("vix_percentile", 50.0)
    trend   = regime.get("vix_trend", "stable")
    yclope  = regime.get("yield_curve_slope", 0.5)
    hurst   = regime.get("hurst", 0.5)
    breadth = regime.get("breadth_composite", 0.5)
    reg     = regime.get("regime", "neutral").lower()

    signals["vix"] = vix
    signals["vix_pct"] = vix_pct
    signals["vix_trend"] = trend
    signals["yield_slope"] = yclope
    signals["hurst"] = hurst
    signals["breadth"] = breadth
    signals["regime"] = reg

    # VIX level: ideal 12-20, penalise extremes
    if vix > 35:
        score -= 35; parts.append(f"VIX={vix:.0f} (crisis)")
    elif vix > 25:
        score -= 20; parts.append(f"VIX={vix:.0f} (elevated)")
    elif vix < 12:
        score -= 10; parts.append(f"VIX={vix:.0f} (crushed)")
    else:
        parts.append(f"VIX={vix:.0f} ✓")

    # VIX trend: falling = best, rising = worst
    if trend == "rising":
        score -= 15; parts.append("VIX rising")
    elif trend == "falling":
        score += 5;  parts.append("VIX falling ✓")

    # Yield curve: inverted = recession risk, suppresses premium strategies
    if yclope < -0.3:
        score -= 15; parts.append(f"curve inverted({yclope:.2f})")
    elif yclope < 0:
        score -= 8;  parts.append(f"curve flat({yclope:.2f})")
    else:
        parts.append(f"curve={yclope:.2f} ✓")

    # Hurst: mean-reverting (<0.45) is ideal for short premium
    if hurst < 0.45:
        score += 5;  parts.append(f"Hurst={hurst:.3f}(MR ✓)")
    elif hurst > 0.6:
        score -= 10; parts.append(f"Hurst={hurst:.3f}(trending)")

    # Breadth: wide participation supports stable underlying prices
    if breadth < 0.35:
        score -= 10; parts.append(f"breadth={breadth:.2f}(weak)")
    elif breadth > 0.65:
        parts.append(f"breadth={breadth:.2f} ✓")

    # Regime override: crisis or high-vol regimes get a hard floor
    if "crisis" in reg or "extreme" in reg:
        score = min(score, 30)
        parts.append("CRISIS regime override")
    elif "high_vol" in reg or "volatile" in reg:
        score = min(score, 55)

    score = max(0.0, min(100.0, score))
    return SectionScore(
        name="REGIME",
        score=score,
        weight=0.20,
        signals=signals,
        detail=", ".join(parts) or f"regime={reg}",
    )


def _score_signal_quality(
    tech_score: Optional[Any],          # TechnicalScore or None
    iv_report:  Optional[Any],          # IVQualityReport or None
) -> SectionScore:
    """
    Section 2: Signal Quality  (weight 0.15)

    How clean are the inputs? Penalises bearish technicals, IV contamination,
    and recent structural breaks in the IV series.
    """
    score = 75.0    # start at 75 — neutral, not perfect
    signals = {}
    parts = []

    # Technical score: 0–7, threshold ~3.0
    if tech_score is not None:
        ts = tech_score.score
        signals["tech_score"] = ts
        signals["is_bullish"] = tech_score.is_bullish
        if ts >= 5.5:
            score += 20; parts.append(f"tech={ts:.1f} (strong ✓)")
        elif ts >= 3.5:
            score += 10; parts.append(f"tech={ts:.1f} (bullish ✓)")
        elif ts >= 2.5:
            parts.append(f"tech={ts:.1f} (neutral)")
        else:
            score -= 20; parts.append(f"tech={ts:.1f} (bearish)")
    else:
        score -= 5; parts.append("tech=unavailable")

    # IV quality: TRADE=full points, CAUTION=-10, BLOCK=-40
    if iv_report is not None:
        rec = iv_report.recommendation
        div = iv_report.divergence
        signals["iv_quality"] = rec
        signals["iv_divergence"] = round(div, 1)
        signals["contamination"] = iv_report.contamination.value
        if rec == "TRADE":
            score += 5; parts.append(f"IV={rec} ✓")
        elif rec == "CAUTION":
            score -= 10; parts.append(f"IV={rec}(div={div:.0f}pt)")
        else:   # BLOCK
            score -= 40; parts.append(f"IV=BLOCK(div={div:.0f}pt) ⚠")
        # Structural break recency
        brk = iv_report.structural_break
        if brk:
            days = brk.get("days_since_break", 999)
            signals["struct_break_days"] = days
            if days < 90:
                score -= 15; parts.append(f"struct_break={days}d ago")
            elif days < 180:
                score -= 7; parts.append(f"struct_break={days}d(caution)")
    else:
        parts.append("IV=unknown (fail-open)")

    score = max(0.0, min(100.0, score))
    return SectionScore(
        name="SIGNAL QUALITY",
        score=score,
        weight=0.15,
        signals=signals,
        detail=", ".join(parts),
    )


def _score_entry_intel(
    sec_score:      int,           # from score_sec_signals()["score"]
    policy_boost:   int,           # from policy_etf_boost()
    sentiment_allowed: bool,       # from SentimentAnalyzer.is_entry_allowed()
    sentiment_score:   float,      # aggregate compound score from sentiment
    gex_sign:       str,           # 'positive'|'negative'|'flat'|None
    pc_oi:          Optional[float],  # put/call OI ratio from GEX
) -> SectionScore:
    """
    Section 3: Entry Intelligence  (weight 0.15)

    Independent confirmation: insider buying, policy signals, sentiment,
    GEX regime. These don't gate entry — they boost or reduce conviction.
    """
    score = 50.0    # neutral base — absence of signals is neutral, not bearish
    signals = {}
    parts = []

    # SEC insider score
    signals["sec_score"] = sec_score
    if sec_score >= 40:
        score += 20; parts.append(f"insider=+{sec_score}(strong ✓)")
    elif sec_score >= 20:
        score += 10; parts.append(f"insider=+{sec_score}(confirmed ✓)")
    elif sec_score >= 5:
        score += 3;  parts.append(f"insider=+{sec_score}(weak)")

    # Policy boost
    signals["policy_boost"] = policy_boost
    if policy_boost >= 15:
        score += 15; parts.append(f"policy=+{policy_boost}(L1 ✓)")
    elif policy_boost >= 8:
        score += 8;  parts.append(f"policy=+{policy_boost}(L2 ✓)")

    # Sentiment
    signals["sentiment_allowed"] = sentiment_allowed
    signals["sentiment_score"] = round(sentiment_score, 3)
    if not sentiment_allowed:
        score -= 15; parts.append("sentiment=blocked")
    elif sentiment_score > 0.1:
        score += 10; parts.append(f"sentiment=+{sentiment_score:.2f} ✓")
    elif sentiment_score < -0.1:
        score -= 10; parts.append(f"sentiment={sentiment_score:.2f}")

    # GEX: positive = dealers long gamma = mean-reverting = good for spreads
    signals["gex_sign"] = gex_sign
    signals["pc_oi"] = pc_oi
    if gex_sign == "positive":
        score += 10; parts.append("GEX=positive(pin ✓)")
    elif gex_sign == "negative":
        score -= 10; parts.append("GEX=negative(vol-amp)")
    # Put-heavy OI: bearish hedge demand elevated — support for put-spread entry
    if pc_oi is not None:
        if pc_oi >= 1.2:
            score += 5; parts.append(f"PC={pc_oi:.2f}(hedged ✓)")
        elif pc_oi <= 0.7:
            score -= 5; parts.append(f"PC={pc_oi:.2f}(complacent)")

    score = max(0.0, min(100.0, score))
    return SectionScore(
        name="ENTRY INTEL",
        score=score,
        weight=0.15,
        signals=signals,
        detail=", ".join(parts) or "no independent signals",
    )


def _score_strategy_fit(
    pop:               float,          # probability of profit (0-1)
    delta:             float,          # short leg delta (negative for puts)
    dte:               int,            # days to expiration
    credit_to_width:   float,          # credit / spread_width (spreads only, else 0)
    estimated_credit:  float,          # absolute credit in dollars per contract
    strategy_name:     str,
) -> SectionScore:
    """
    Section 4: Strategy Fit  (weight 0.20)

    Does this specific contract structure make mathematical sense?
    Ideal: PoP >= 68%, delta -0.20 to -0.28, DTE 21-45, credit >= 25% width.
    """
    score = 50.0
    signals = {}
    parts = []

    signals["pop"] = round(pop, 3)
    signals["delta"] = round(delta, 3)
    signals["dte"] = dte
    signals["credit_to_width"] = round(credit_to_width, 3)
    signals["estimated_credit"] = round(estimated_credit, 2)

    # Probability of profit: 70%+ = excellent, 60-70% = acceptable, <60% = weak
    if pop >= 0.75:
        score += 30; parts.append(f"PoP={pop:.0%}(✓)")
    elif pop >= 0.68:
        score += 20; parts.append(f"PoP={pop:.0%}(good)")
    elif pop >= 0.60:
        score += 5;  parts.append(f"PoP={pop:.0%}(ok)")
    else:
        score -= 20; parts.append(f"PoP={pop:.0%}(weak)")

    # Delta: -0.20 to -0.28 is ideal for risk-adjusted short premium
    abs_delta = abs(delta)
    if 0.20 <= abs_delta <= 0.28:
        score += 15; parts.append(f"Δ={delta:.2f}(✓)")
    elif 0.15 <= abs_delta < 0.20:
        score += 5;  parts.append(f"Δ={delta:.2f}(OTM)")
    elif 0.28 < abs_delta <= 0.35:
        score += 5;  parts.append(f"Δ={delta:.2f}(ITM risk)")
    elif abs_delta > 0.35:
        score -= 10; parts.append(f"Δ={delta:.2f}(too close)")

    # DTE: 21-45 days is the theta decay sweet spot for short spreads
    if 21 <= dte <= 45:
        score += 15; parts.append(f"DTE={dte}(✓)")
    elif 14 <= dte < 21 or 45 < dte <= 60:
        score += 5;  parts.append(f"DTE={dte}(ok)")
    elif dte < 14:
        score -= 15; parts.append(f"DTE={dte}(too short)")
    else:
        score -= 5;  parts.append(f"DTE={dte}(too long)")

    # Credit-to-width: >= 0.25 means collecting >= 25% of the spread width
    if "spread" in strategy_name.lower() or "strangle" in strategy_name.lower():
        if credit_to_width >= 0.33:
            score += 10; parts.append(f"C/W={credit_to_width:.2f}(✓)")
        elif credit_to_width >= 0.25:
            score += 5;  parts.append(f"C/W={credit_to_width:.2f}(ok)")
        elif credit_to_width < 0.20:
            score -= 15; parts.append(f"C/W={credit_to_width:.2f}(thin)")

    # Absolute credit: minimum threshold to be worth the commission
    if estimated_credit < 0.20:
        score -= 20; parts.append(f"credit=${estimated_credit:.2f}(too thin)")
    elif estimated_credit >= 0.75:
        score += 5;  parts.append(f"credit=${estimated_credit:.2f} ✓")

    score = max(0.0, min(100.0, score))
    return SectionScore(
        name="STRATEGY FIT",
        score=score,
        weight=0.20,
        signals=signals,
        detail=", ".join(parts),
    )


def _score_risk_posture(
    daily_pnl_pct:       float,     # today's P&L as % of equity (negative = loss)
    max_daily_loss_pct:  float,     # halt threshold (e.g. 0.05 = 5%)
    open_positions:      int,       # number of currently open positions
    max_positions:       int,       # configured max (e.g. 5)
    risk_budget_used:    float,     # fraction of daily risk budget consumed (0-1)
) -> SectionScore:
    """
    Section 5: Risk Posture  (weight 0.15)

    How much capacity does the risk layer have? Penalises days where we're
    already losing, near the position cap, or running out of risk budget.
    """
    score = 90.0    # start high — fresh day with no positions is ideal
    signals = {}
    parts = []

    signals["daily_pnl_pct"] = round(daily_pnl_pct, 4)
    signals["open_positions"] = open_positions
    signals["max_positions"] = max_positions
    signals["risk_budget_used"] = round(risk_budget_used, 3)

    # Daily P&L: losing days get increasing penalties
    loss_pct = abs(daily_pnl_pct) if daily_pnl_pct < 0 else 0
    halt_pct = max_daily_loss_pct
    if daily_pnl_pct < 0:
        fraction_of_limit = loss_pct / halt_pct if halt_pct > 0 else 1.0
        if fraction_of_limit >= 0.8:
            score -= 50; parts.append(f"P&L={daily_pnl_pct:.1%}(near halt ⚠)")
        elif fraction_of_limit >= 0.5:
            score -= 25; parts.append(f"P&L={daily_pnl_pct:.1%}(losing)")
        elif fraction_of_limit >= 0.25:
            score -= 10; parts.append(f"P&L={daily_pnl_pct:.1%}")
        else:
            score -= 5;  parts.append(f"P&L={daily_pnl_pct:.1%}(small loss)")
    else:
        parts.append(f"P&L={daily_pnl_pct:+.1%} ✓")

    # Open positions vs cap
    pos_pct = open_positions / max(max_positions, 1)
    if pos_pct >= 0.9:
        score -= 25; parts.append(f"positions={open_positions}/{max_positions}(near cap)")
    elif pos_pct >= 0.7:
        score -= 10; parts.append(f"positions={open_positions}/{max_positions}")
    elif open_positions == 0:
        parts.append("positions=0 ✓")
    else:
        parts.append(f"positions={open_positions}/{max_positions} ✓")

    # Risk budget remaining
    if risk_budget_used >= 0.9:
        score -= 20; parts.append(f"budget={risk_budget_used:.0%}(exhausted)")
    elif risk_budget_used >= 0.7:
        score -= 10; parts.append(f"budget={risk_budget_used:.0%}")
    else:
        parts.append(f"budget={risk_budget_used:.0%} ✓")

    score = max(0.0, min(100.0, score))
    return SectionScore(
        name="RISK POSTURE",
        score=score,
        weight=0.15,
        signals=signals,
        detail=", ".join(parts),
    )


def _score_execution(
    spread_pct:         Optional[float],   # (ask-bid)/mid of short leg
    live_mid_available: bool,              # was a live Alpaca midpoint fetched?
    api_healthy:        bool,              # circuit breakers not tripped
    slippage_budget_pct: float = 0.02,    # configured slippage tolerance
) -> SectionScore:
    """
    Section 6: Execution Readiness  (weight 0.10)

    Can we actually get a good fill? Wide spreads, stale quotes, and open
    circuit breakers all reduce the chance of a clean execution.
    """
    score = 80.0
    signals = {}
    parts = []

    signals["spread_pct"] = round(spread_pct, 4) if spread_pct is not None else None
    signals["live_mid"] = live_mid_available
    signals["api_healthy"] = api_healthy

    # Bid/ask spread: <5% = excellent, 5-15% = ok, >15% = wide
    if spread_pct is not None:
        if spread_pct > 0.30:
            score -= 35; parts.append(f"spread={spread_pct:.0%}(very wide)")
        elif spread_pct > 0.15:
            score -= 20; parts.append(f"spread={spread_pct:.0%}(wide)")
        elif spread_pct > 0.08:
            score -= 10; parts.append(f"spread={spread_pct:.0%}(moderate)")
        elif spread_pct > 0.05:
            score -= 5;  parts.append(f"spread={spread_pct:.0%}(ok)")
        else:
            score += 10; parts.append(f"spread={spread_pct:.0%}(tight ✓)")
    else:
        score -= 15; parts.append("spread=unknown")

    # Live mid: stale/unavailable midpoint means we're pricing blind
    if not live_mid_available:
        score -= 20; parts.append("live_mid=unavailable")
    else:
        parts.append("live_mid=fresh ✓")

    # API health: open circuit breakers indicate recent failures
    if not api_healthy:
        score -= 15; parts.append("API=degraded ⚠")
    else:
        parts.append("API=healthy ✓")

    score = max(0.0, min(100.0, score))
    return SectionScore(
        name="EXECUTION",
        score=score,
        weight=0.10,
        signals=signals,
        detail=", ".join(parts),
    )


def _score_track_record(
    win_rate:       Optional[float],    # 0-1, None if < min_trades
    profit_factor:  Optional[float],    # > 1 = profitable
    n_trades:       int,                # total closed trades in history
    min_trades:     int = 10,           # minimum for meaningful stats
) -> SectionScore:
    """
    Section 7: Track Record  (weight 0.05)

    What does the bot's own history say about this strategy's performance?
    Low weight (0.05) because recent performance alone shouldn't override
    a strong setup — but consistent losses should reduce conviction.
    """
    score = 70.0    # neutral — insufficient history doesn't penalise
    signals = {}
    parts = []

    signals["n_trades"] = n_trades
    signals["win_rate"] = round(win_rate, 3) if win_rate is not None else None
    signals["profit_factor"] = round(profit_factor, 3) if profit_factor is not None else None

    # Sample size: below min_trades, score stays neutral
    if n_trades < min_trades:
        parts.append(f"n={n_trades}(building history — neutral)")
        score = 70.0
    else:
        # Win rate: target 65%+ for short premium
        if win_rate is not None:
            if win_rate >= 0.70:
                score += 20; parts.append(f"WR={win_rate:.0%} ✓")
            elif win_rate >= 0.60:
                score += 10; parts.append(f"WR={win_rate:.0%}(ok)")
            elif win_rate >= 0.50:
                parts.append(f"WR={win_rate:.0%}(neutral)")
            else:
                score -= 25; parts.append(f"WR={win_rate:.0%}(struggling ⚠)")

        # Profit factor: > 1.5 = strong, 1.0-1.5 = ok, < 1.0 = net loss
        if profit_factor is not None:
            if profit_factor >= 1.8:
                score += 15; parts.append(f"PF={profit_factor:.2f} ✓")
            elif profit_factor >= 1.3:
                score += 5;  parts.append(f"PF={profit_factor:.2f}(ok)")
            elif profit_factor >= 1.0:
                parts.append(f"PF={profit_factor:.2f}(breakeven)")
            else:
                score -= 25; parts.append(f"PF={profit_factor:.2f}(net loss ⚠)")

    score = max(0.0, min(100.0, score))
    return SectionScore(
        name="TRACK RECORD",
        score=score,
        weight=0.05,
        signals=signals,
        detail=", ".join(parts),
    )


# ---------------------------------------------------------------------------
# Overall aggregation
# ---------------------------------------------------------------------------

_REGIME_WEIGHTS = {
    # (regime_string_substring -> {section_name: weight_override})
    "crisis":     {"REGIME": 0.30, "RISK POSTURE": 0.25, "EXECUTION": 0.15,
                   "SIGNAL QUALITY": 0.10, "ENTRY INTEL": 0.10,
                   "STRATEGY FIT": 0.07, "TRACK RECORD": 0.03},
    "high_vol":   {"REGIME": 0.25, "RISK POSTURE": 0.20, "STRATEGY FIT": 0.20,
                   "EXECUTION": 0.12, "SIGNAL QUALITY": 0.10,
                   "ENTRY INTEL": 0.10, "TRACK RECORD": 0.03},
    "trending":   {"REGIME": 0.18, "STRATEGY FIT": 0.25, "SIGNAL QUALITY": 0.20,
                   "RISK POSTURE": 0.12, "ENTRY INTEL": 0.12,
                   "EXECUTION": 0.10, "TRACK RECORD": 0.03},
}

_DEFAULT_WEIGHTS = {
    "REGIME":         0.20,
    "SIGNAL QUALITY": 0.15,
    "ENTRY INTEL":    0.15,
    "STRATEGY FIT":   0.20,
    "RISK POSTURE":   0.15,
    "EXECUTION":      0.10,
    "TRACK RECORD":   0.05,
}


def _get_weights(regime_str: str) -> dict[str, float]:
    for key, weights in _REGIME_WEIGHTS.items():
        if key in regime_str.lower():
            return weights
    return _DEFAULT_WEIGHTS


def _aggregate(sections: list[SectionScore], regime_str: str) -> float:
    """Weighted average, regime-adaptive."""
    weights = _get_weights(regime_str)
    total_w = 0.0
    total_wv = 0.0
    for s in sections:
        w = weights.get(s.name, s.weight)
        s.weight = w   # update section weight for reporting
        total_wv += s.score * w
        total_w  += w
    return round(total_wv / total_w, 1) if total_w > 0 else 50.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """
    Compute a ConfidenceReport for one StrategySignal.

    Usage in orchestrator (inside TradingPipeline.run_for_ticker):

        scorer = ConfidenceScorer(self.regime_detector, self.db, self.rm)
        report = scorer.score(
            signal=signal,
            ticker=ticker,
            tech_score=tech_score,      # from TickerGate.scanner.compute()
            iv_report=iv_report,        # from IVQualityGate.check()
            sec_data=sec_data,          # from score_sec_signals()
            gex=gex_analysis,           # from GEXEngine.analyze()
            sentiment=sentiment_signal, # from SentimentAnalyzer
            open_trades=open_trades,    # from TradeDatabase.get_open_trades()
        )
        if not report.should_trade:
            logger.info("[Pipeline] %s: confidence too low (%s) — skip", ticker, report.grade)
            return None
        send_discord(webhook, report.short_line())
    """

    def __init__(self, regime_detector, db, risk_manager):
        self._regime   = regime_detector
        self._db       = db
        self._rm       = risk_manager

    def score(
        self,
        signal,                          # StrategySignal
        ticker: str,
        tech_score=None,                 # TechnicalScore | None
        iv_report=None,                  # IVQualityReport | None
        sec_data: Optional[dict] = None, # score_sec_signals() result
        policy_boost: int = 0,
        gex=None,                        # GEXAnalysis | None
        gex_cboe: Optional[dict] = None, # fetch_cboe_gex() result | None
        sentiment_allowed: bool = True,
        sentiment_compound: float = 0.0,
        open_trades: Optional[list] = None,
    ) -> ConfidenceReport:
        """
        Compute a full ConfidenceReport from available pipeline data.
        Every input is Optional — missing data degrades gracefully to
        neutral scores rather than blocking the trade.
        """
        regime = self._regime.detect()
        regime_str = regime.get("regime", "neutral")

        # ── Section 1: Regime ────────────────────────────────────────
        s1 = _score_regime(regime)

        # ── Section 2: Signal Quality ────────────────────────────────
        s2 = _score_signal_quality(tech_score, iv_report)

        # ── Section 3: Entry Intel ───────────────────────────────────
        sec_score_val = (sec_data or {}).get("score", 0)
        gex_obj  = gex or {}
        gex_cboe_obj = gex_cboe or {}
        # Prefer live GEX, fall back to CBOE
        gex_sign = (
            getattr(gex_obj, "gamma_regime", None)
            or gex_cboe_obj.get("gex_sign")
        )
        pc_oi = gex_cboe_obj.get("pc_oi")
        if gex is not None and hasattr(gex, "levels") and gex.levels:
            # Use net GEX sign from analyze_gex result
            net = sum(lv.gex_notional for lv in gex.levels)
            gex_sign = "positive" if net > 0 else "negative" if net < 0 else "flat"

        s3 = _score_entry_intel(
            sec_score=sec_score_val,
            policy_boost=policy_boost,
            sentiment_allowed=sentiment_allowed,
            sentiment_score=sentiment_compound,
            gex_sign=gex_sign,
            pc_oi=pc_oi,
        )

        # ── Section 4: Strategy Fit ──────────────────────────────────
        legs = signal.legs if signal else []
        short_leg = next((l for l in legs if "sell" in (l.side or "")), None)
        delta = 0.0
        pop   = 0.5
        if short_leg and signal.source_contracts:
            src = next(
                (c for c in signal.source_contracts
                 if c.strike == short_leg.strike and c.option_type == short_leg.option_type),
                None
            )
            if src:
                delta = src.delta or 0.0
                pop   = getattr(src, "_pop", 0.5)   # set by greeks.probability_of_profit

        dte = signal.dte or 30
        credit = abs(signal.estimated_fill_price) if signal else 0.0
        # For spreads: credit-to-width
        if len(legs) == 2:
            strikes = sorted([l.strike for l in legs])
            width = strikes[1] - strikes[0] if len(strikes) == 2 else 1
            c2w = credit / width if width > 0 else 0.0
        else:
            c2w = 0.0

        s4 = _score_strategy_fit(
            pop=pop,
            delta=delta,
            dte=dte,
            credit_to_width=c2w,
            estimated_credit=credit,
            strategy_name=signal.strategy_name if signal else "",
        )

        # ── Section 5: Risk Posture ──────────────────────────────────
        equity = max(self._rm.equity, 1.0)
        daily_pnl = getattr(self._rm, "_state", None)
        # Try to get daily P&L from risk manager state
        try:
            from .risk import RiskManager
            daily_pnl_pct = self._rm._state.daily_realized_pnl / equity
        except Exception:
            daily_pnl_pct = 0.0

        open_count = len(open_trades or [])
        max_pos    = getattr(
            getattr(self._rm, "_config", None), "max_positions_per_strategy", 5
        ) or 5
        # Risk budget used = trades_today / max_trades_per_day
        try:
            trades_today  = self._rm._state.trades_today if hasattr(self._rm, "_state") else 0
            max_trades    = self._rm._config.max_trades_per_day if hasattr(self._rm, "_config") else 5
            budget_used   = trades_today / max(max_trades, 1)
        except Exception:
            budget_used = 0.0
        max_loss_pct = (
            self._rm._config.max_daily_loss_pct
            if hasattr(self._rm, "_config") else 0.05
        )

        s5 = _score_risk_posture(
            daily_pnl_pct=daily_pnl_pct,
            max_daily_loss_pct=max_loss_pct,
            open_positions=open_count,
            max_positions=max_pos,
            risk_budget_used=budget_used,
        )

        # ── Section 6: Execution Readiness ───────────────────────────
        spread_pct = None
        if short_leg and signal and signal.source_contracts:
            src = next(
                (c for c in signal.source_contracts if c.strike == short_leg.strike),
                None
            )
            if src:
                spread_pct = src.spread_pct
        live_mid = spread_pct is not None   # if we have a contract, we had a live quote
        # API health: check circuit breakers
        try:
            from .circuit_breaker import data_circuit_breaker as _cb
            cb_status = _cb.status()
            api_ok = all(
                v.get("state") in ("closed", "half_open")
                for v in cb_status.values()
            )
        except Exception:
            api_ok = True

        s6 = _score_execution(
            spread_pct=spread_pct,
            live_mid_available=live_mid,
            api_healthy=api_ok,
        )

        # ── Section 7: Track Record ──────────────────────────────────
        try:
            from .adaptive import AdaptiveTuner
            all_pnls  = self._db.get_all_closed_pnls()
            n_trades  = len(all_pnls)
            wins      = [p for p in all_pnls if p > 0]
            losses    = [abs(p) for p in all_pnls if p < 0]
            win_rate  = len(wins) / n_trades if n_trades >= 10 else None
            pf        = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else None
        except Exception:
            n_trades, win_rate, pf = 0, None, None

        s7 = _score_track_record(
            win_rate=win_rate,
            profit_factor=pf,
            n_trades=n_trades,
        )

        # ── Overall ──────────────────────────────────────────────────
        sections = [s1, s2, s3, s4, s5, s6, s7]
        overall  = _aggregate(sections, regime_str)

        report = ConfidenceReport(
            ticker=ticker,
            strategy=signal.strategy_name if signal else "",
            sections=sections,
            overall=overall,
        )

        logger.info(
            "[Confidence] %s %s: overall=%.0f [%s]  "
            "regime=%.0f signal=%.0f intel=%.0f fit=%.0f "
            "risk=%.0f exec=%.0f track=%.0f",
            ticker, signal.strategy_name if signal else "",
            overall, report.grade,
            s1.score, s2.score, s3.score, s4.score,
            s5.score, s6.score, s7.score,
        )
        return report


__all__ = [
    "ConfidenceScorer",
    "ConfidenceReport",
    "SectionScore",
]
