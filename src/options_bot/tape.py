"""
Tape reading and factor orthogonalization utilities.

Two tools extracted from Hedge_Fund_Algo/Short_Engine.py:

1. detect_wall(trades) — identifies hidden buy/sell walls in trade tape.
   A buy wall (absorption): 70%+ of volume is buy-side but price is flat.
   A sell wall (absorption): 70%+ of volume is sell-side but price is flat.
   Used as a pre-entry filter — if a hidden wall is detected against your
   intended direction, skip the trade.

2. FactorOrthogonalizer — removes cross-factor correlation from signal scores
   before combining them into a composite score. Prevents double-counting
   when two signals are correlated (e.g. Z-score and CMF often move together).
   Requires a pre-computed weights file (factor_weights.json). Falls back
   gracefully to raw scores if weights are unavailable.

Both are pure utilities with no broker or strategy dependencies.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tape wall detector
# ---------------------------------------------------------------------------

WallSignal = Literal["sell_wall", "buy_wall", "neutral"]


@dataclass
class TapeResult:
    signal: WallSignal
    buy_ratio: float       # fraction of volume that was buy-aggressor
    price_delta: float     # price move over the lookback period (fraction)
    reason: str


def detect_wall(
    trades: list[dict],
    buy_wall_threshold: float = 0.30,
    sell_wall_threshold: float = 0.70,
    stagnation_threshold: float = 0.0005,
) -> TapeResult:
    """
    Detect hidden buy or sell walls from a list of trade ticks.

    A sell wall (absorption): most volume is buy-aggressor but price barely
    moves — there's a large seller absorbing all the buying.

    A buy wall (absorption): most volume is sell-aggressor but price barely
    moves — there's a large buyer absorbing all the selling.

    Parameters
    ----------
    trades : list of dicts
        Each dict must have keys 'price' (float) and 'size' (float).
        Trades in chronological order.
    buy_wall_threshold : float
        If buy_ratio < this and price is flat → hidden buy wall detected.
    sell_wall_threshold : float
        If buy_ratio > this and price is flat → hidden sell wall detected.
    stagnation_threshold : float
        Max fractional price move to be considered "flat" (default 0.05%).

    Returns
    -------
    TapeResult
    """
    if not trades or len(trades) < 30:
        return TapeResult(
            signal="neutral",
            buy_ratio=0.5,
            price_delta=0.0,
            reason="insufficient_data",
        )

    buy_vol = 0.0
    sell_vol = 0.0
    last_aggressor = 0

    for i in range(1, len(trades)):
        price     = float(trades[i].get("price", trades[i].get("p", 0)))
        prev_price = float(trades[i-1].get("price", trades[i-1].get("p", 0)))
        size      = float(trades[i].get("size", trades[i].get("s", 0)))

        if price > prev_price:
            buy_vol += size
            last_aggressor = 1
        elif price < prev_price:
            sell_vol += size
            last_aggressor = -1
        else:
            # Tick rule: assign to last known aggressor
            if last_aggressor == 1:
                buy_vol += size
            elif last_aggressor == -1:
                sell_vol += size

    total_vol = buy_vol + sell_vol
    if total_vol == 0:
        return TapeResult(signal="neutral", buy_ratio=0.5, price_delta=0.0, reason="zero_volume")

    buy_ratio = buy_vol / total_vol

    start_price = float(trades[0].get("price", trades[0].get("p", 0)))
    end_price   = float(trades[-1].get("price", trades[-1].get("p", 0)))
    price_delta = (end_price - start_price) / start_price if start_price != 0 else 0.0

    if buy_ratio > sell_wall_threshold and abs(price_delta) < stagnation_threshold:
        return TapeResult(
            signal="sell_wall",
            buy_ratio=round(buy_ratio, 4),
            price_delta=round(price_delta, 6),
            reason=f"absorption: {buy_ratio:.1%} buys but price flat ({price_delta:.3%})",
        )

    if buy_ratio < buy_wall_threshold and abs(price_delta) < stagnation_threshold:
        return TapeResult(
            signal="buy_wall",
            buy_ratio=round(buy_ratio, 4),
            price_delta=round(price_delta, 6),
            reason=f"absorption: {buy_ratio:.1%} sells but price flat ({price_delta:.3%})",
        )

    return TapeResult(
        signal="neutral",
        buy_ratio=round(buy_ratio, 4),
        price_delta=round(price_delta, 6),
        reason=f"normal flow: {buy_ratio:.1%} buy ratio, {price_delta:.3%} price move",
    )


# ---------------------------------------------------------------------------
# Factor orthogonalizer
# ---------------------------------------------------------------------------

class FactorOrthogonalizer:
    """
    Remove cross-factor correlation from raw signal scores.

    When multiple signals are correlated (Z-score, CMF, SMA distance often
    move together), simply summing them double-counts shared information.
    Orthogonalization regresses each factor on the prior ones and uses
    the residuals — each residual captures only the unique information in
    that factor not already explained by the others.

    Requires a pre-computed weights JSON file with per-symbol regression
    coefficients. Falls back to raw scores if the file is missing or the
    symbol isn't in the weights dict.

    Weights file format (per symbol):
    {
      "AAPL": {
        "sigma_Z": 1.2,
        "alpha_C": 0.01, "beta_C_Z": 0.3, "Sigma_C_res": 0.8,
        "alpha_S": 0.0,  "beta_S_Z": 0.1, "beta_S_C": 0.2, "sigma_S_res": 0.5
      },
      ...
    }
    """

    def __init__(self, weights_path: str | Path = "factor_weights.json"):
        self._weights: dict = {}
        path = Path(weights_path)
        if path.exists():
            try:
                with open(path) as f:
                    self._weights = json.load(f)
                logger.info("[FactorOrthogonalizer] Loaded weights for %d symbols", len(self._weights))
            except Exception as exc:
                logger.warning("[FactorOrthogonalizer] Failed to load weights: %s — using raw scores", exc)
        else:
            logger.debug("[FactorOrthogonalizer] No weights file at %s — using raw scores", path)

    def orthogonalize(
        self,
        symbol: str,
        raw_z: float,
        raw_cmf: float,
        raw_sma_dist: float,
    ) -> tuple[float, float, float]:
        """
        Return orthogonalized (z, cmf, sma_dist) scores for a symbol.

        If no weights exist for the symbol, returns the raw inputs unchanged.

        Parameters
        ----------
        symbol : str
        raw_z : float
            Raw 20-day Z-score of the price.
        raw_cmf : float
            Raw Chaikin Money Flow value.
        raw_sma_dist : float
            Raw distance from SMA50 as a fraction.

        Returns
        -------
        tuple[float, float, float]
            (orth_z, orth_cmf, orth_sma_dist) — decorrelated signal scores.
        """
        w = self._weights.get(symbol)
        if not w:
            return raw_z, raw_cmf, raw_sma_dist

        try:
            # Z-score: normalize by its own sigma
            orth_z = raw_z / w["sigma_Z"]

            # CMF residual: remove component explained by Z
            sigma_c = w.get("Sigma_C_res", w.get("sigma_C_res", 1.0))
            orth_cmf = (raw_cmf - w["alpha_C"] - w["beta_C_Z"] * orth_z) / sigma_c

            # SMA distance residual: remove components explained by Z and CMF
            orth_sma = (
                raw_sma_dist
                - w["alpha_S"]
                - w["beta_S_Z"] * orth_z
                - w["beta_S_C"] * orth_cmf
            ) / w["sigma_S_res"]

            return orth_z, orth_cmf, orth_sma

        except (KeyError, ZeroDivisionError) as exc:
            logger.debug("[FactorOrthogonalizer] %s: orthogonalization failed (%s) — using raw", symbol, exc)
            return raw_z, raw_cmf, raw_sma_dist
