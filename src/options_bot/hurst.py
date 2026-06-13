"""
Hurst exponent — market regime classifier.

Computes the Hurst exponent for a price series using log-lag variance regression
(R/S analysis approximation). Classifies the series as:

  H < 0.48  → MEAN_REVERTING  — price oscillates around a mean (sell-premium friendly)
  H > 0.52  → TRENDING        — directional momentum (avoid short-premium, use spreads)
  0.48-0.52 → RANDOM_WALK     — no exploitable structure (reduce size or skip)

Mathematical basis:
  For lags τ = 2..19, compute the standard deviation of lagged differences:
    τ_i = sqrt( std( price[τ:] - price[:-τ] ) )
  Fit log(τ) vs log(τ_i) with OLS. The slope × 2 = Hurst exponent H.
  H = 0.5 is pure random walk (Brownian motion).
  H < 0.5 → mean-reversion (anti-persistent).
  H > 0.5 → trending (persistent).

No external dependencies beyond numpy (already required).
Pure function — no side effects, safe to call from any context.
"""
from __future__ import annotations

import numpy as np


def hurst_exponent(prices: np.ndarray | list) -> float:
    """
    Compute the Hurst exponent for a price series.

    Parameters
    ----------
    prices : array-like
        Closing prices (or any time series). Needs at least 20 observations.
        Passing log prices is acceptable — the slope is scale-invariant.

    Returns
    -------
    float
        Hurst exponent in [0, 1]. Returns 0.5 (random walk) if the series
        is too short or degenerate.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 20:
        return 0.5

    lags = range(2, 20)
    try:
        tau = [
            np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag])))
            for lag in lags
        ]
        # Guard against all-zero tau (flat price series)
        tau = np.array(tau)
        if np.all(tau == 0):
            return 0.5
        # OLS fit in log-log space
        poly = np.polyfit(np.log(list(lags)), np.log(tau + 1e-12), 1)
        return float(poly[0] * 2.0)
    except Exception:
        return 0.5


def classify_regime(h: float) -> str:
    """
    Classify a Hurst exponent value into a market regime string.

    Returns one of: "mean_reverting", "trending", "random_walk"
    """
    if h < 0.48:
        return "mean_reverting"
    if h > 0.52:
        return "trending"
    return "random_walk"


def hurst_options_weight(h: float) -> float:
    """
    Map a Hurst exponent to an options allocation weight modifier.

    Mean-reverting regimes are ideal for short-premium strategies (high weight).
    Trending regimes are dangerous for short-premium (reduce weight).
    Random walk is neutral (moderate weight).

    Returns a multiplier in [0.0, 1.5]:
      mean_reverting (H < 0.40) → 1.5  (most favorable)
      mean_reverting (0.40-0.48) → 1.2
      random_walk    (0.48-0.52) → 0.8
      trending       (0.52-0.60) → 0.4
      trending       (H > 0.60)  → 0.0  (suppress entirely)
    """
    if h < 0.40:
        return 1.5
    if h < 0.48:
        return 1.2
    if h < 0.52:
        return 0.8
    if h < 0.60:
        return 0.4
    return 0.0
