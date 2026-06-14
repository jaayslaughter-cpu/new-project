"""
Walk-Forward Edge Validator.

AUDIT FINDING (Layer 5 — Adaptive Tuner):
  "20 trades is statistically insufficient for the conclusions drawn.
   A 45% win rate over 20 trades has a 95% CI of approximately [23%, 68%].
   Tightening delta based on 20-trade win rate is acting on noise."
  "No regime adjustment in the tuner — poor performance may be regime-caused
   not strategy-caused."

FIX:
  This module wraps the adaptive tuner with a walk-forward validation check.
  Before any parameter adjustment is applied, we verify that the detected
  edge (higher win rate in recent trades) survives an out-of-sample split.

VALIDATION METHOD:
  For each strategy with N >= MIN_WF_SAMPLES closed trades:
    1. Sort trades chronologically (oldest → newest)
    2. Split at midpoint: training = first N//2, validation = last N//2
    3. Compute on each half independently:
         - bootstrap mean CI (B=500 resamples, 95% CI)
         - Wilson lower bound on win rate
    4. Classify edge survival:
         "survives" : μ_train > 0 AND μ_val > 0 AND |μ_val - μ_train| ≤ 0.5σ
         "reverses" : μ_train > 0 but μ_val ≤ 0
         "decays"   : μ_train > 0 AND μ_val > 0 but μ_val < μ_train - 0.5σ
         "no_edge"  : μ_train ≤ 0 AND μ_val ≤ 0
         "emerged"  : μ_train ≤ 0 but μ_val > 0 (new edge developing)
         "insufficient": N < MIN_WF_SAMPLES

  ADAPTIVE TUNER GATE:
    The adaptive tuner should only apply parameter changes when:
      verdict == "survives" OR verdict == "emerged"
    If verdict is "reverses", "decays", or "no_edge", adjustments are
    suppressed with a clear log entry explaining why.

LABELS:
  "survives" does NOT mean the strategy is profitable — only that the
  first-half edge (positive mean) persisted into the second half.
  A negative mean that persists would be "no_edge", not "reverses".

  Bootstrap CIs at 500 resamples are approximate. For N < 30, the CI
  will be wide. The method is transparent and sample-size appropriate;
  a paired t-test would require normality assumptions unsafe at small N.

Source
------
diablotrading-main/inferno_walk_forward.py
Core math preserved exactly; rewritten interface for our adaptive.py.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum closed trades per strategy before walk-forward can run
# Below this, any edge claim is statistically meaningless
MIN_WF_SAMPLES = 30   # Audit recommended; 20 is insufficient

# Bootstrap parameters
N_BOOTSTRAP = 500     # Resamples for mean CI
CI_LEVEL    = 0.95    # 95% confidence interval

# Survival tolerance: allow mean to decay by up to 0.5 std before flagging
DECAY_TOLERANCE_STD = 0.5


# ---------------------------------------------------------------------------
# Wilson lower bound on win rate
# ---------------------------------------------------------------------------

def wilson_lower_bound(n_wins: int, n_total: int, z: float = 1.645) -> float:
    """
    Wilson lower bound on win rate at 90% confidence (z=1.645).

    LABEL: The Wilson interval is preferred over naive p±z*se because it
    performs better at small sample sizes and near the boundaries (0%, 100%).
    z=1.645 gives a one-sided 90% lower bound — the minimum plausible win rate.

    Returns 0.0 if n_total == 0.
    """
    if n_total == 0:
        return 0.0
    p = n_wins / n_total
    numerator = p + z**2 / (2 * n_total) - z * math.sqrt(
        (p * (1 - p) + z**2 / (4 * n_total)) / n_total
    )
    denominator = 1 + z**2 / n_total
    return max(0.0, numerator / denominator)


# ---------------------------------------------------------------------------
# Bootstrap mean CI
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(
    values: list[float],
    n_resamples: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean of values.

    Returns (mean, ci_low, ci_high).

    LABEL: Non-parametric bootstrap. Does not assume normality.
    Appropriate for small samples where t-test normality assumption is unsafe.
    At N < 10, CI will be very wide — this is correct, not a bug.
    """
    if not values:
        return 0.0, 0.0, 0.0

    n   = len(values)
    obs = float(sum(values) / n)

    if n == 1:
        return obs, obs, obs

    boot_means = []
    for _ in range(n_resamples):
        resample = [random.choice(values) for _ in range(n)]
        boot_means.append(sum(resample) / n)

    boot_means.sort()
    alpha = 1 - ci
    lo_idx = int(alpha / 2 * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples)
    lo_idx = max(0, min(lo_idx, len(boot_means) - 1))
    hi_idx = max(0, min(hi_idx, len(boot_means) - 1))

    return round(obs, 4), round(boot_means[lo_idx], 4), round(boot_means[hi_idx], 4)


# ---------------------------------------------------------------------------
# Walk-forward result
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    strategy:         str
    n_total:          int
    n_train:          int
    n_val:            int

    # Training half
    mean_train:       float
    ci_lo_train:      float
    ci_hi_train:      float
    winrate_train:    float
    wilson_lo_train:  float

    # Validation half
    mean_val:         float
    ci_lo_val:        float
    ci_hi_val:        float
    winrate_val:      float
    wilson_lo_val:    float

    # Verdict
    verdict:          str     # survives | reverses | decays | no_edge | emerged | insufficient
    tuner_may_adjust: bool    # True if adaptive tuner is permitted to adjust parameters
    explanation:      str     # Plain-language explanation

    @property
    def edge_degradation(self) -> float:
        """How much the mean decayed from train to val (negative = improvement)."""
        return self.mean_val - self.mean_train


def run_walk_forward(
    strategy: str,
    closed_pnls: list[tuple[str, float]],  # (timestamp_str, pnl_dollars)
) -> WalkForwardResult:
    """
    Run walk-forward validation on a strategy's closed trade history.

    Parameters
    ----------
    strategy : str
        Strategy name (e.g. "short_put_spread")
    closed_pnls : list of (timestamp, pnl_dollars)
        Closed trades sorted by timestamp (oldest first from DB ORDER BY updated_at ASC).
        PnL in dollars per trade (already multiplied by contracts × 100).

    Returns
    -------
    WalkForwardResult with verdict and tuner gate decision.
    """
    n = len(closed_pnls)

    if n < MIN_WF_SAMPLES:
        return WalkForwardResult(
            strategy=strategy, n_total=n, n_train=0, n_val=0,
            mean_train=0, ci_lo_train=0, ci_hi_train=0,
            winrate_train=0, wilson_lo_train=0,
            mean_val=0, ci_lo_val=0, ci_hi_val=0,
            winrate_val=0, wilson_lo_val=0,
            verdict="insufficient",
            tuner_may_adjust=False,
            explanation=(
                f"{strategy}: only {n} closed trades (min {MIN_WF_SAMPLES} required). "
                f"Adaptive tuner SUPPRESSED — cannot validate edge with this sample size. "
                f"LABEL: Any pattern in {n} trades is likely noise."
            ),
        )

    # Sort by timestamp (should already be sorted, but enforce)
    sorted_pnls = sorted(closed_pnls, key=lambda x: x[0])
    values = [pnl for _, pnl in sorted_pnls]

    mid = n // 2
    train = values[:mid]
    val   = values[mid:]

    # Training metrics
    mean_t, ci_lo_t, ci_hi_t = bootstrap_mean_ci(train)
    wins_t = sum(1 for v in train if v > 0)
    wr_t   = wins_t / len(train)
    wl_t   = wilson_lower_bound(wins_t, len(train))

    # Validation metrics
    mean_v, ci_lo_v, ci_hi_v = bootstrap_mean_ci(val)
    wins_v = sum(1 for v in val if v > 0)
    wr_v   = wins_v / len(val)
    wl_v   = wilson_lower_bound(wins_v, len(val))

    # Standard deviation of training set (for decay tolerance band)
    if len(train) >= 2:
        train_std = (sum((x - mean_t) ** 2 for x in train) / (len(train) - 1)) ** 0.5
    else:
        train_std = abs(mean_t) + 1

    # Verdict classification
    if mean_t > 0 and mean_v > 0:
        if abs(mean_v - mean_t) <= DECAY_TOLERANCE_STD * train_std:
            verdict          = "survives"
            tuner_may_adjust = True
            explanation = (
                f"{strategy}: edge SURVIVES walk-forward split. "
                f"Train μ=${mean_t:.1f} (WR={wr_t:.0%}), "
                f"Val μ=${mean_v:.1f} (WR={wr_v:.0%}). "
                f"Decay ${mean_v - mean_t:.1f} ≤ tolerance ${DECAY_TOLERANCE_STD * train_std:.1f}. "
                f"Adaptive tuner PERMITTED to adjust parameters."
            )
        else:
            verdict          = "decays"
            tuner_may_adjust = False
            explanation = (
                f"{strategy}: edge DECAYS in validation. "
                f"Train μ=${mean_t:.1f} → Val μ=${mean_v:.1f} "
                f"(decay ${mean_v - mean_t:.1f} > tolerance ${DECAY_TOLERANCE_STD * train_std:.1f}). "
                f"Adaptive tuner SUPPRESSED — likely overfitting to recent data."
            )
    elif mean_t > 0 and mean_v <= 0:
        verdict          = "reverses"
        tuner_may_adjust = False
        explanation = (
            f"{strategy}: edge REVERSES in validation. "
            f"Train μ=${mean_t:.1f} (profitable) → Val μ=${mean_v:.1f} (losing). "
            f"Adaptive tuner SUPPRESSED — strategy may be overfit or regime-dependent. "
            f"AUDIT NOTE: check if regime changed between train and val periods."
        )
    elif mean_t <= 0 and mean_v > 0:
        verdict          = "emerged"
        tuner_may_adjust = True
        explanation = (
            f"{strategy}: edge EMERGED in validation (new edge developing). "
            f"Train μ=${mean_t:.1f} (losing) → Val μ=${mean_v:.1f} (profitable). "
            f"Adaptive tuner PERMITTED — recent improvement is confirmed out-of-sample. "
            f"CAUTION: watch for mean reversion; check regime conditions."
        )
    else:
        verdict          = "no_edge"
        tuner_may_adjust = False
        explanation = (
            f"{strategy}: NO EDGE in either half. "
            f"Train μ=${mean_t:.1f}, Val μ=${mean_v:.1f}. "
            f"Adaptive tuner SUPPRESSED — no positive mean to preserve. "
            f"Consider pausing strategy or reviewing setup filters."
        )

    logger.info("[WalkForward] %s → %s (tuner=%s)", strategy, verdict.upper(),
                "PERMITTED" if tuner_may_adjust else "SUPPRESSED")

    return WalkForwardResult(
        strategy=strategy, n_total=n, n_train=len(train), n_val=len(val),
        mean_train=mean_t, ci_lo_train=ci_lo_t, ci_hi_train=ci_hi_t,
        winrate_train=round(wr_t, 4), wilson_lo_train=round(wl_t, 4),
        mean_val=mean_v, ci_lo_val=ci_lo_v, ci_hi_val=ci_hi_v,
        winrate_val=round(wr_v, 4), wilson_lo_val=round(wl_v, 4),
        verdict=verdict,
        tuner_may_adjust=tuner_may_adjust,
        explanation=explanation,
    )


def check_tuner_permission(
    strategy: str,
    db,
) -> tuple[bool, WalkForwardResult]:
    """
    Query DB for closed trades and run walk-forward validation.

    Returns (tuner_may_adjust: bool, result: WalkForwardResult).
    Call this in adaptive.py before applying any parameter change.

    If DB is unavailable, returns (False, insufficient_result) — conservative.
    """
    try:
        with db._get_conn() as conn:
            cur = conn.execute(
                """SELECT updated_at, realized_pnl FROM trades
                   WHERE strategy = ?
                     AND status IN ('stopped_out','closed_profit_target',
                                    'closed_expiry','closed_external')
                     AND realized_pnl IS NOT NULL
                   ORDER BY updated_at ASC""",
                (strategy,)
            )
            rows = [(row[0], float(row[1])) for row in cur.fetchall()]
    except Exception as exc:
        logger.warning("[WalkForward] DB query failed for %s: %s", strategy, exc)
        return False, WalkForwardResult(
            strategy=strategy, n_total=0, n_train=0, n_val=0,
            mean_train=0, ci_lo_train=0, ci_hi_train=0,
            winrate_train=0, wilson_lo_train=0,
            mean_val=0, ci_lo_val=0, ci_hi_val=0,
            winrate_val=0, wilson_lo_val=0,
            verdict="insufficient",
            tuner_may_adjust=False,
            explanation=f"{strategy}: DB unavailable — tuner SUPPRESSED (conservative).",
        )

    result = run_walk_forward(strategy, rows)
    return result.tuner_may_adjust, result
