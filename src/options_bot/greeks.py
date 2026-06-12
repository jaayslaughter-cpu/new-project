"""
Greeks enrichment layer.

Takes OptionChainRow objects from the market data layer and computes:
  - Implied Volatility (Newton-Raphson solver)
  - Delta, Gamma, Theta, Vega, Rho (analytical Black-Scholes)

Also fetches the live risk-free rate from the US Treasury website,
with a daily cache and configurable fallback.

Mathematical rationale (written before any code, per system directive):

Black-Scholes for a European option:

  d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
  d2 = d1 - σ·√T

  Call price  = S·N(d1) - K·e^(-rT)·N(d2)
  Put price   = K·e^(-rT)·N(-d2) - S·N(-d1)

  Delta (call) = N(d1)
  Delta (put)  = N(d1) - 1
  Gamma        = N'(d1) / (S·σ·√T)
  Theta (call) = -[S·N'(d1)·σ/(2√T)] - r·K·e^(-rT)·N(d2)      (per calendar day)
  Vega         = S·N'(d1)·√T  (per 1-point IV move, divide by 100 for per-1% move)
  Rho (call)   = K·T·e^(-rT)·N(d2)

IV is solved by Newton-Raphson: find σ such that BS_price(σ) = market_price.
If the solver fails to converge (e.g. deep ITM/OTM with no vol), iv=None
and IVSolveError is raised — never substitute an estimate.

Note: Black-Scholes assumes European-style exercise.
For American single-stock options, use the binomial CRR engine instead
(gamma-scalping repo, QuantLib). Index options (SPX, SPY) are European-style
and BS is appropriate.
"""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from scipy.stats import norm

from .contracts import EnrichedOptionRow, OptionChainRow
from .exceptions import DataValidationError, IVSolveError, PipelineConnectionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Treasury rate fetcher
# ---------------------------------------------------------------------------

_RATE_CACHE: dict[str, tuple[float, float]] = {}  # date_str → (rate, fetch_epoch)
_RATE_CACHE_TTL = 86_400  # refresh once per day
_FALLBACK_RATE = float(os.getenv("FALLBACK_RISK_FREE_RATE", "0.045"))  # 4.5% default
_TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month="
)


def get_risk_free_rate() -> float:
    """
    Fetches the current 3-month US Treasury yield as the risk-free rate.

    Caches the result for 24 hours. Falls back to FALLBACK_RISK_FREE_RATE
    (env var, default 4.5%) if the request fails — but logs a warning so
    you know it's using the fallback.

    Returns
    -------
    float
        Annual risk-free rate as a decimal (e.g. 0.045 = 4.5%)
    """
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m")
    cache_key = today

    if cache_key in _RATE_CACHE:
        rate, fetched_at = _RATE_CACHE[cache_key]
        if time.time() - fetched_at < _RATE_CACHE_TTL:
            logger.debug("[RateCache] Using cached rate %.4f", rate)
            return rate

    try:
        import urllib.request
        url = _TREASURY_URL + today.replace("-", "")
        req = urllib.request.Request(url, headers={"User-Agent": "options-bot/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Parse 3-month rate from Treasury HTML table
        # The table contains rows like: <td>4.32</td>
        import re
        # Find the 3-month column value — it appears after "3 Mo" in the table
        matches = re.findall(r"3 Mo</th>.*?<td[^>]*>([\d.]+)</td>", html, re.DOTALL)
        if not matches:
            # Try alternate format
            matches = re.findall(r"<td[^>]*>([\d.]+)</td>", html)

        if matches:
            rate = float(matches[0]) / 100.0  # convert percent to decimal
            _RATE_CACHE[cache_key] = (rate, time.time())
            logger.info("[RateFetcher] Treasury 3-month rate: %.4f (%.2f%%)", rate, rate * 100)
            return rate
        else:
            raise ValueError("Could not parse rate from Treasury HTML")

    except Exception as exc:
        logger.warning(
            "[RateFetcher] Failed to fetch Treasury rate: %s — using fallback %.4f",
            exc, _FALLBACK_RATE
        )
        return _FALLBACK_RATE


# ---------------------------------------------------------------------------
# Black-Scholes pricing and Greeks
# ---------------------------------------------------------------------------

def _bs_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """
    Compute d1 and d2 for Black-Scholes.

    Parameters
    ----------
    S : float  — underlying spot price
    K : float  — strike price
    T : float  — time to expiry in years
    r : float  — risk-free rate (annual decimal)
    sigma : float — implied volatility (annual decimal)
    """
    if T <= 0:
        raise DataValidationError("T", f"Time to expiry must be positive, got {T}")
    if sigma <= 0:
        raise DataValidationError("sigma", f"Volatility must be positive, got {sigma}")
    if S <= 0:
        raise DataValidationError("S", f"Spot price must be positive, got {S}")
    if K <= 0:
        raise DataValidationError("K", f"Strike must be positive, got {K}")

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Black-Scholes option price."""
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> dict[str, float]:
    """
    Compute all first-order Greeks analytically.

    Returns dict with keys: delta, gamma, theta, vega, rho
    theta is per calendar day (not per year).
    vega is per 1-point move in IV (i.e. per 0.01 in sigma decimal).
    """
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    exp_rT = math.exp(-r * T)

    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0

    # Gamma (same for call and put)
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # Theta (per calendar day — divide annual theta by 365)
    if option_type == "call":
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            - r * K * exp_rT * norm.cdf(d2)
        )
    else:
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            + r * K * exp_rT * norm.cdf(-d2)
        )
    theta = theta_annual / 365.0

    # Vega (per 1-point move = 0.01 in sigma)
    vega = S * pdf_d1 * sqrt_T * 0.01

    # Rho (per 1% rate move = 0.01)
    if option_type == "call":
        rho = K * T * exp_rT * norm.cdf(d2) * 0.01
    else:
        rho = -K * T * exp_rT * norm.cdf(-d2) * 0.01

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


def solve_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    initial_guess: float = 0.25,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """
    Newton-Raphson implied volatility solver.

    Finds σ such that BS_price(S, K, T, r, σ) ≈ market_price.

    Parameters
    ----------
    market_price : float  — observed mid-price of the option
    initial_guess : float — starting σ (default 0.25 = 25% IV)

    Returns
    -------
    float — implied volatility as a decimal

    Raises
    ------
    IVSolveError
        If the solver fails to converge or produces an unreasonable result.
        Per zero-hallucination policy: never return an estimate.
    """
    if market_price <= 0:
        raise IVSolveError("?", f"market_price={market_price} must be positive")
    if T <= 0:
        raise IVSolveError("?", f"T={T} must be positive (not expired)")

    # Boundary check: price below intrinsic value means IV can't be solved
    if option_type == "call":
        intrinsic = max(0.0, S - K * math.exp(-r * T))
    else:
        intrinsic = max(0.0, K * math.exp(-r * T) - S)

    if market_price < intrinsic - 0.01:
        raise IVSolveError(
            "?",
            f"market_price={market_price:.4f} < intrinsic={intrinsic:.4f} "
            "(arbitrage violation — cannot solve IV)"
        )

    sigma = initial_guess
    for i in range(max_iterations):
        try:
            price = bs_price(S, K, T, r, sigma, option_type)
            d1, _ = _bs_d1_d2(S, K, T, r, sigma)
            # Vega = S * N'(d1) * sqrt(T)  (NOT divided by 100 here)
            vega = S * norm.pdf(d1) * math.sqrt(T)

            if abs(vega) < 1e-10:
                raise IVSolveError("?", "Vega near zero — deep ITM/OTM, cannot solve")

            diff = price - market_price
            if abs(diff) < tolerance:
                if sigma < 0.001 or sigma > 20.0:
                    raise IVSolveError(
                        "?",
                        f"IV={sigma:.4f} is outside reasonable range [0.1%, 2000%]"
                    )
                return sigma

            sigma = sigma - diff / vega

            # Clamp to prevent explosion
            sigma = max(1e-6, min(sigma, 20.0))

        except (DataValidationError, ZeroDivisionError, ValueError) as exc:
            raise IVSolveError("?", f"Solver error at iteration {i}: {exc}") from exc

    raise IVSolveError(
        "?",
        f"Newton-Raphson did not converge after {max_iterations} iterations "
        f"(last σ={sigma:.6f}, diff={diff:.6f})"
    )


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

class GreeksEnricher:
    """
    Enriches OptionChainRow objects with IV and Greeks.

    Plugs into the pipeline after market_data and before the strategy engine.
    """

    def __init__(self, risk_free_rate: Optional[float] = None, pricing_model: str = "black_scholes"):
        """
        Parameters
        ----------
        risk_free_rate : float or None
            If None, fetches from Treasury website (cached daily).
            Pass a float to override (useful in backtesting).
        pricing_model : str
            "black_scholes" for European-style (SPX, SPY index options).
            "binomial_crr" for American-style single-stock (requires QuantLib).
        """
        self._override_rate = risk_free_rate
        self.pricing_model = pricing_model
        logger.info(
            "[GreeksEnricher] Initialized (model=%s, rate_override=%s)",
            pricing_model, risk_free_rate
        )

    def get_rate(self) -> float:
        if self._override_rate is not None:
            return self._override_rate
        return get_risk_free_rate()

    def enrich(self, row: OptionChainRow) -> EnrichedOptionRow:
        """
        Enriches one OptionChainRow with IV and Greeks.

        If IV solve fails, returns an EnrichedOptionRow with iv=None and
        all Greeks=None. The row is still returned (not discarded) so
        callers can decide whether to skip it.

        Parameters
        ----------
        row : OptionChainRow

        Returns
        -------
        EnrichedOptionRow
        """
        logger.debug("[GreeksEnricher] Enriching %s", row.symbol)

        rate = self.get_rate()
        T = row.dte / 365.0  # time to expiry in years

        # Need mid_price to solve IV
        market_price = row.mid_price
        if market_price is None or market_price <= 0:
            logger.debug(
                "[GreeksEnricher] %s: no valid mid_price (%.4f) — Greeks will be None",
                row.symbol, market_price or 0
            )
            return EnrichedOptionRow(
                raw=row,
                iv=None,
                risk_free_rate=rate,
                pricing_model=self.pricing_model,
            )

        # Solve IV
        try:
            iv = solve_iv(
                market_price=market_price,
                S=row.underlying_price,
                K=row.strike,
                T=T,
                r=rate,
                option_type=row.option_type,
            )
        except IVSolveError as exc:
            logger.debug("[GreeksEnricher] IV solve failed for %s: %s", row.symbol, exc)
            return EnrichedOptionRow(
                raw=row,
                iv=None,
                risk_free_rate=rate,
                pricing_model=self.pricing_model,
            )

        # Compute Greeks
        try:
            greeks = bs_greeks(
                S=row.underlying_price,
                K=row.strike,
                T=T,
                r=rate,
                sigma=iv,
                option_type=row.option_type,
            )
        except (DataValidationError, ZeroDivisionError, ValueError) as exc:
            logger.warning(
                "[GreeksEnricher] Greeks failed for %s (iv=%.4f): %s",
                row.symbol, iv, exc
            )
            return EnrichedOptionRow(
                raw=row,
                iv=iv,
                risk_free_rate=rate,
                pricing_model=self.pricing_model,
            )

        logger.debug(
            "[GreeksEnricher] %s: iv=%.4f delta=%.4f gamma=%.6f theta=%.4f vega=%.4f",
            row.symbol, iv, greeks["delta"], greeks["gamma"], greeks["theta"], greeks["vega"]
        )

        return EnrichedOptionRow(
            raw=row,
            iv=iv,
            delta=greeks["delta"],
            gamma=greeks["gamma"],
            theta=greeks["theta"],
            vega=greeks["vega"],
            rho=greeks["rho"],
            risk_free_rate=rate,
            pricing_model=self.pricing_model,
        )

    def enrich_chain(self, rows: list[OptionChainRow]) -> list[EnrichedOptionRow]:
        """
        Enriches a full chain. Logs progress and returns all rows
        (including those where IV solve failed).
        """
        logger.info("[GreeksEnricher] Enriching %d rows", len(rows))
        enriched = [self.enrich(row) for row in rows]
        solved = sum(1 for r in enriched if r.iv is not None)
        logger.info(
            "[GreeksEnricher] Done: %d/%d rows have IV", solved, len(enriched)
        )
        return enriched

    def enrich_chain_filtered(
        self,
        rows: list[OptionChainRow],
        require_iv: bool = True,
        min_abs_delta: Optional[float] = None,
        max_abs_delta: Optional[float] = None,
    ) -> list[EnrichedOptionRow]:
        """
        Enriches and then filters by Greeks.

        Parameters
        ----------
        require_iv : bool
            Drop rows where IV could not be solved (default True).
        min_abs_delta : float or None
            Minimum |delta| to keep (e.g. 0.10 = 10-delta minimum).
        max_abs_delta : float or None
            Maximum |delta| to keep (e.g. 0.40 = 40-delta maximum).
        """
        enriched = self.enrich_chain(rows)
        result = []

        for row in enriched:
            if require_iv and row.iv is None:
                continue
            if row.delta is not None:
                abs_delta = abs(row.delta)
                if min_abs_delta is not None and abs_delta < min_abs_delta:
                    continue
                if max_abs_delta is not None and abs_delta > max_abs_delta:
                    continue
            result.append(row)

        logger.info(
            "[GreeksEnricher] After delta filter: %d rows remain", len(result)
        )
        return result
