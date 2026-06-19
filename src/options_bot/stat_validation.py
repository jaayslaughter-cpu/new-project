"""
Statistical Validation Module — short-premium options bot.

Ported from Alphaglyph (MIT licence, Danny-397/alphaglyph).

Three independent tests that answer the question:
  "Is the bot's edge statistically real, or just noise?"

1. Monte Carlo bootstrap (1 000 paths)
   Resamples the daily P&L sequence; returns the actual result's
   percentile rank in the simulated distribution.

2. Deflated Sharpe Ratio (Lopez de Prado, 2014)
   PSR corrected for multiple-testing bias.  For N strategies tested,
   raises the Sharpe benchmark to the expected max of N random strategies.
   DSR > 0.95  →  "STATISTICALLY SIGNIFICANT"

3. Fama-French 3-Factor Decomposition
   OLS regression against Mkt-RF, SMB, HML.
   Separates true alpha from passive factor exposure.
   Requires internet access to pull Ken French's factor CSV.

Usage (called from orchestrator._check_milestones):
    from .stat_validation import run_all_validations, format_validation_discord

    curve  = orchestrator._query_equity_curve()   # list of (date_str, equity)
    result = run_all_validations(curve)
    msg    = format_validation_discord(result)
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

logger = logging.getLogger(__name__)

_TRADING_DAYS     = 252
_EULER_MASCHERONI = 0.5772156649
_FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "data_library/F-F_Research_Data_Factors_daily_CSV.zip"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _daily_returns(equity_curve: list[float]) -> np.ndarray:
    """Simple daily returns from an equity curve."""
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size < 2:
        return np.array([])
    prev = arr[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(prev != 0, np.diff(arr) / prev, 0.0)


# ---------------------------------------------------------------------------
# 1. Monte Carlo bootstrap
# ---------------------------------------------------------------------------

def run_monte_carlo(
    port_hist: list[dict],
    initial_capital: float,
    actual_sharpe: float,
    n_simulations: int = 1_000,
) -> dict:
    """
    Bootstrap-resample the daily return sequence n_simulations times.

    port_hist : list of {"date": str, "value": float}
    Returns an 'enabled: False' dict when there are fewer than 5 points.
    """
    if len(port_hist) < 5:
        return {"enabled": False, "reason": "Need at least 5 equity curve points."}

    values = np.array([p["value"] for p in port_hist], dtype=float)
    dates  = [p["date"] for p in port_hist]

    returns       = _daily_returns(values)
    n             = len(returns)
    actual_final  = float(values[-1])
    actual_return = (actual_final / initial_capital - 1) * 100

    # Bootstrap: shape (n_simulations, n)
    sim_returns   = np.random.choice(returns, size=(n_simulations, n), replace=True)
    equity_matrix = initial_capital * np.cumprod(1 + sim_returns, axis=1)
    final_values  = equity_matrix[:, -1]
    final_returns = (final_values / initial_capital - 1) * 100

    stds        = sim_returns.std(axis=1)
    means       = sim_returns.mean(axis=1)
    rf_daily    = 0.04 / 252
    sim_sharpes = np.where(stds > 0, (means - rf_daily) / stds * np.sqrt(252), 0.0)

    actual_pct = float(np.mean(final_values <= actual_final) * 100)
    sharpe_pct = float(np.mean(sim_sharpes  <= actual_sharpe) * 100)

    # Fan chart — ~60 sampled time points
    step    = max(1, n // 60)
    idx     = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    sampled   = equity_matrix[:, idx]
    fan_dates = [dates[i + 1] for i in idx]

    def _band(pct):
        return [round(float(v), 2) for v in np.percentile(sampled, pct, axis=0)]

    return {
        "enabled":             True,
        "n_simulations":       n_simulations,
        "actual_return_pct":   round(actual_return, 2),
        "actual_percentile":   round(actual_pct,    1),
        "sharpe_percentile":   round(sharpe_pct,    1),
        "return_distribution": {
            "p5":  round(float(np.percentile(final_returns,  5)), 2),
            "p25": round(float(np.percentile(final_returns, 25)), 2),
            "p50": round(float(np.percentile(final_returns, 50)), 2),
            "p75": round(float(np.percentile(final_returns, 75)), 2),
            "p95": round(float(np.percentile(final_returns, 95)), 2),
        },
        "sharpe_distribution": {
            "p5":  round(float(np.percentile(sim_sharpes,  5)), 2),
            "p25": round(float(np.percentile(sim_sharpes, 25)), 2),
            "p50": round(float(np.percentile(sim_sharpes, 50)), 2),
            "p75": round(float(np.percentile(sim_sharpes, 75)), 2),
            "p95": round(float(np.percentile(sim_sharpes, 95)), 2),
        },
        "fan_chart": {"dates": fan_dates, "p5": _band(5), "p25": _band(25),
                      "p50": _band(50), "p75": _band(75), "p95": _band(95)},
    }


# ---------------------------------------------------------------------------
# 2. Probabilistic & Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    daily_returns: np.ndarray,
    sr_benchmark_annual: float = 0.0,
) -> float:
    """
    P(SR_true > sr_benchmark_annual) given the observed sample.
    Corrects for skewness, excess kurtosis, and finite sample size.
    Lopez de Prado (2014), eq. 1.
    """
    r = np.asarray(daily_returns, dtype=float)
    n = len(r)
    if n < 10:
        return float("nan")

    mu, sigma = r.mean(), r.std(ddof=1)
    if sigma < 1e-12:
        return float("nan")

    sr_hat = mu / sigma
    sr_b   = sr_benchmark_annual / np.sqrt(_TRADING_DAYS)

    skew_r = float(skew(r))
    exkurt = float(kurtosis(r, fisher=True))

    var_correction = 1.0 - skew_r * sr_hat + (exkurt + 2) / 4 * sr_hat ** 2
    if var_correction <= 0:
        return float("nan")

    z = (sr_hat - sr_b) * np.sqrt(n - 1) / np.sqrt(var_correction)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    daily_returns: np.ndarray,
    n_strategies: int = 4,
) -> dict:
    """
    DSR: PSR where SR* = expected max Sharpe from N independent random strategies.

    n_strategies: how many strategies were implicitly tested when selecting
                  the best one (ShortPutSpread, CashSecuredPut, ShortStrangle,
                  ZeroDTE = 4 by default).
    """
    r  = np.asarray(daily_returns, dtype=float)
    T  = len(r)
    s  = r.std(ddof=1)

    _null = {
        "sr_annual": None, "sr_benchmark": None,
        "psr": None, "dsr": None,
        "is_significant": False, "n_strategies": n_strategies,
    }
    if s < 1e-12 or T < 10:
        return _null

    mu        = r.mean()
    sr_annual = float(mu / s * np.sqrt(_TRADING_DAYS))

    if n_strategies > 1:
        ez_max = (
            (1 - _EULER_MASCHERONI) * norm.ppf(1 - 1 / n_strategies) +
            _EULER_MASCHERONI     * norm.ppf(1 - 1 / (n_strategies * np.e))
        )
    else:
        ez_max = 0.0

    sr_star_annual = float(ez_max * np.sqrt(_TRADING_DAYS / T))

    psr_val = probabilistic_sharpe_ratio(r, sr_benchmark_annual=0.0)
    dsr_val = probabilistic_sharpe_ratio(r, sr_benchmark_annual=sr_star_annual)

    return {
        "sr_annual":      round(sr_annual,      4),
        "sr_benchmark":   round(sr_star_annual, 4),
        "psr":            round(psr_val, 4) if not np.isnan(psr_val) else None,
        "dsr":            round(dsr_val, 4) if not np.isnan(dsr_val) else None,
        "is_significant": bool(not np.isnan(dsr_val) and dsr_val > 0.95),
        "n_strategies":   n_strategies,
    }


# ---------------------------------------------------------------------------
# 3. Fama-French 3-Factor Decomposition
# ---------------------------------------------------------------------------

def _fetch_ff3_raw() -> Optional[str]:
    try:
        import urllib.request as ur
        with ur.urlopen(_FF3_URL, timeout=15) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            csv_name = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
            return z.read(csv_name).decode("latin-1")
    except Exception as exc:
        logger.warning("[StatValidation] Could not fetch Fama-French factors: %s", exc)
        return None


def _parse_ff3_csv(text: str) -> pd.DataFrame:
    rows, data_started = [], False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if data_started:
                break
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 5:
            continue
        try:
            date   = pd.to_datetime(parts[0], format="%Y%m%d")
            values = [float(p) / 100.0 for p in parts[1:5]]
            rows.append({"date": date, "mkt_rf": values[0],
                         "smb": values[1], "hml": values[2], "rf": values[3]})
            data_started = True
        except (ValueError, IndexError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def fama_french_decomposition(
    port_hist: list[dict],
    ff3_data: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Regress daily portfolio excess returns against FF3 factors.

    port_hist : list of {"date": str, "value": float}
    Returns alpha (annualised), factor betas, R², t-stats, and plain-English
    interpretation.
    """
    if len(port_hist) < 30:
        return {"enabled": False, "reason": "Need at least 30 equity curve points."}

    dates  = pd.to_datetime([p["date"] for p in port_hist])
    values = np.array([p["value"] for p in port_hist], dtype=float)
    rets   = pd.Series(np.diff(values) / values[:-1], index=dates[1:])

    if ff3_data is None:
        raw = _fetch_ff3_raw()
        if raw is None:
            return {"enabled": False,
                    "reason": "Could not download Fama-French factors (no internet?)."}
        ff3_data = _parse_ff3_csv(raw)

    if ff3_data.empty:
        return {"enabled": False, "reason": "Failed to parse Fama-French data."}

    merged = pd.DataFrame({"port": rets}).join(ff3_data, how="inner")
    if len(merged) < 30:
        return {"enabled": False,
                "reason": f"Only {len(merged)} overlapping trading days (need 30+)."}

    excess = merged["port"].values - merged["rf"].values
    X = np.column_stack([
        np.ones(len(merged)),
        merged["mkt_rf"].values,
        merged["smb"].values,
        merged["hml"].values,
    ])
    betas, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)

    y_hat  = X @ betas
    resid  = excess - y_hat
    ss_res = float(resid @ resid)
    ss_tot = float(((excess - excess.mean()) ** 2).sum())
    r_sq   = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    n, k    = len(excess), 4
    mse     = ss_res / max(n - k, 1)
    xtx_inv = np.linalg.inv(X.T @ X + np.eye(k) * 1e-12)
    se      = np.sqrt(np.abs(np.diag(mse * xtx_inv)))
    t_stats = betas / (se + 1e-15)

    alpha_annual = float(betas[0]) * _TRADING_DAYS

    sig      = "significant" if abs(float(t_stats[0])) > 2.0 else "not significant"
    smb_desc = ("small-cap tilt" if betas[2] > 0.1 else
                "large-cap tilt" if betas[2] < -0.1 else "size-neutral")
    hml_desc = ("value tilt"  if betas[3] > 0.1 else
                "growth tilt" if betas[3] < -0.1 else "style-neutral")
    interpretation = (
        f"Annual alpha {alpha_annual*100:+.2f}% ({sig}, |t|={abs(t_stats[0]):.2f}). "
        f"Market beta {betas[1]:.2f}x. "
        f"SMB {betas[2]:+.2f} ({smb_desc}). "
        f"HML {betas[3]:+.2f} ({hml_desc}). "
        f"R²={r_sq:.3f} — {r_sq*100:.1f}% of variance explained by the 3 factors."
    )

    return {
        "enabled":        True,
        "n_obs":          n,
        "alpha_annual":   round(alpha_annual * 100, 2),
        "alpha_t_stat":   round(float(t_stats[0]), 3),
        "beta_market":    round(float(betas[1]), 4),
        "beta_smb":       round(float(betas[2]), 4),
        "beta_hml":       round(float(betas[3]), 4),
        "r_squared":      round(r_sq, 4),
        "t_stats": {
            "alpha":  round(float(t_stats[0]), 3),
            "market": round(float(t_stats[1]), 3),
            "smb":    round(float(t_stats[2]), 3),
            "hml":    round(float(t_stats[3]), 3),
        },
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_validations(
    port_hist: list[dict],
    initial_capital: float = 100_000.0,
    n_strategies: int = 4,
) -> dict:
    """
    Run Monte Carlo, DSR, and Fama-French on a portfolio equity curve.

    port_hist : list of {"date": str, "value": float} sorted oldest → newest.
    Returns a combined dict with keys: monte_carlo, deflated_sharpe, fama_french,
    verdict (one of STATISTICALLY_SIGNIFICANT / PROMISING / INCONCLUSIVE / NOISE).
    """
    values = [p["value"] for p in port_hist]
    rets   = _daily_returns(values)

    # Sharpe for Monte Carlo
    if len(rets) >= 2:
        std = rets.std(ddof=1)
        sharpe = float((rets.mean() - 0.04/252) / std * np.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    mc  = run_monte_carlo(port_hist, initial_capital, sharpe)
    dsr = deflated_sharpe_ratio(rets, n_strategies=n_strategies)
    ff3 = fama_french_decomposition(port_hist)

    # Overall verdict
    verdict = _verdict(mc, dsr, ff3)

    return {
        "monte_carlo":     mc,
        "deflated_sharpe": dsr,
        "fama_french":     ff3,
        "verdict":         verdict,
    }


def _verdict(mc: dict, dsr: dict, ff3: dict) -> str:
    signals = 0
    total   = 0

    if mc.get("enabled"):
        total += 1
        if mc.get("actual_percentile", 0) >= 70:
            signals += 1

    if dsr.get("dsr") is not None:
        total += 1
        if dsr.get("is_significant"):
            signals += 1

    if ff3.get("enabled"):
        total += 1
        if ff3.get("alpha_t_stat") is not None and abs(ff3["alpha_t_stat"]) > 2.0:
            signals += 1

    if total == 0:
        return "INSUFFICIENT_DATA"
    ratio = signals / total
    if ratio >= 0.67:
        return "STATISTICALLY_SIGNIFICANT"
    if ratio >= 0.34:
        return "PROMISING_NEEDS_MORE_DATA"
    return "INCONCLUSIVE_MAY_BE_NOISE"


# ---------------------------------------------------------------------------
# Discord formatter
# ---------------------------------------------------------------------------

def format_validation_discord(result: dict) -> str:
    NL = "\n"
    mc  = result.get("monte_carlo",     {})
    dsr = result.get("deflated_sharpe", {})
    ff3 = result.get("fama_french",     {})
    verdict = result.get("verdict", "UNKNOWN")

    verdict_emoji = {
        "STATISTICALLY_SIGNIFICANT":   "✅",
        "PROMISING_NEEDS_MORE_DATA":   "🟡",
        "INCONCLUSIVE_MAY_BE_NOISE":   "🔴",
        "INSUFFICIENT_DATA":           "⚪",
    }.get(verdict, "❓")

    lines = [
        f"**📊 Statistical Edge Validation**",
        f"{verdict_emoji} Verdict: **{verdict.replace('_', ' ')}**",
        "",
    ]

    # Monte Carlo
    if mc.get("enabled"):
        pct = mc.get("actual_percentile", 0)
        mc_emoji = "✅" if pct >= 70 else "🟡" if pct >= 50 else "🔴"
        lines += [
            f"**Monte Carlo** (1 000 bootstrap paths)",
            f"  {mc_emoji} Return percentile: **{pct:.0f}th** "
            f"(actual {mc.get('actual_return_pct', 0):+.1f}%)",
            f"  Median random path: {mc['return_distribution']['p50']:+.1f}%  "
            f"P5/P95: {mc['return_distribution']['p5']:+.1f}% / "
            f"{mc['return_distribution']['p95']:+.1f}%",
            "",
        ]
    else:
        lines += [f"**Monte Carlo**: ⚪ {mc.get('reason', 'Not run')}", ""]

    # DSR
    if dsr.get("dsr") is not None:
        dsr_val = dsr["dsr"]
        dsr_emoji = "✅" if dsr.get("is_significant") else "🟡" if dsr_val > 0.80 else "🔴"
        lines += [
            f"**Deflated Sharpe Ratio** (Lopez de Prado)",
            f"  {dsr_emoji} DSR: **{dsr_val:.3f}** "
            f"(Sharpe {dsr.get('sr_annual', 0):.2f}  "
            f"benchmark {dsr.get('sr_benchmark', 0):.2f}  "
            f"N={dsr.get('n_strategies', 4)} strategies)",
            f"  {'Edge is REAL after multiple-testing correction' if dsr.get('is_significant') else 'Edge not yet confirmed vs random selection'}",
            "",
        ]
    else:
        lines += [f"**Deflated Sharpe**: ⚪ insufficient data", ""]

    # Fama-French
    if ff3.get("enabled"):
        alpha = ff3.get("alpha_annual", 0)
        t     = ff3.get("alpha_t_stat", 0)
        ff_emoji = "✅" if abs(t) > 2.0 else "🟡" if abs(t) > 1.5 else "🔴"
        lines += [
            f"**Fama-French 3-Factor**",
            f"  {ff_emoji} Annual alpha: **{alpha:+.2f}%** (t={t:.2f})",
            f"  Mkt β={ff3.get('beta_market',0):.2f}  "
            f"SMB β={ff3.get('beta_smb',0):.2f}  "
            f"HML β={ff3.get('beta_hml',0):.2f}  "
            f"R²={ff3.get('r_squared',0):.3f}",
            "",
        ]
    else:
        lines += [f"**Fama-French**: ⚪ {ff3.get('reason', 'Not run')}", ""]

    return NL.join(lines)
