"""
Performance metrics for the options bot.

Pure functions operating on numpy arrays — no pandas dependency, no side effects.
Used by the orchestrator's EOD summary and by backtests.

All functions accept plain lists or numpy arrays.
Annualization assumes daily bars (252 trading days/year) by default.
Pass periods_per_year explicitly for intraday or weekly data.

Extracted and rewritten from pairs-divergence-strategy/metrics.py
(original: MIT license, pure numpy implementation).
"""
from __future__ import annotations

import math
import numpy as np

TRADING_DAYS_PER_YEAR = 252


def equity_to_returns(equity) -> np.ndarray:
    """Convert an equity curve to simple period returns."""
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2:
        return np.array([])
    prev = equity[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(prev != 0, np.diff(equity) / prev, 0.0)
    return returns


def total_return(equity) -> float:
    """Total return over the full curve (e.g. 0.12 = +12%)."""
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2 or equity[0] == 0:
        return 0.0
    return float(equity[-1] / equity[0] - 1.0)


def sharpe_ratio(
    returns,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Period returns (not cumulative).
    periods_per_year : int
        252 for daily, 52 for weekly, 12 for monthly.
    risk_free_rate : float
        Annual risk-free rate (de-annualized internally).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / sd)


def max_drawdown(equity) -> float:
    """
    Maximum peak-to-trough decline as a negative fraction (e.g. -0.12 = -12%).
    """
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(
            running_max != 0,
            (equity - running_max) / running_max,
            0.0,
        )
    return float(drawdowns.min())


def annualized_return(
    equity,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compound annual growth rate implied by the equity curve."""
    equity = np.asarray(equity, dtype=float)
    n = equity.size - 1
    if n < 1 or equity[0] <= 0:
        return 0.0
    growth = equity[-1] / equity[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n) - 1.0)


def win_rate(pnls) -> float:
    """Fraction of trades with positive P&L (e.g. 0.62 = 62% win rate)."""
    pnls = np.asarray(pnls, dtype=float)
    if pnls.size == 0:
        return 0.0
    return float((pnls > 0).mean())


def profit_factor(pnls) -> float:
    """
    Gross profit / gross loss. > 1.0 means more won than lost in dollar terms.
    Returns inf if there are no losing trades.
    """
    pnls = np.asarray(pnls, dtype=float)
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def avg_win_loss_ratio(pnls) -> float:
    """Average winning trade / average losing trade (absolute values)."""
    pnls = np.asarray(pnls, dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    if wins.size == 0 or losses.size == 0:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def summary(pnls, equity_curve=None) -> dict:
    """
    Compute a full performance summary dict.

    Parameters
    ----------
    pnls : array-like
        Per-trade P&L values in dollars.
    equity_curve : array-like or None
        Running equity curve. If None, computed from cumsum of pnls
        starting at 100_000.

    Returns
    -------
    dict with keys:
        total_pnl, trade_count, win_rate, profit_factor,
        avg_win_loss_ratio, sharpe, max_drawdown, annualized_return
    """
    pnls = np.asarray(pnls, dtype=float)

    if equity_curve is None and pnls.size > 0:
        equity_curve = np.concatenate([[100_000.0], 100_000.0 + np.cumsum(pnls)])
    elif equity_curve is None:
        equity_curve = np.array([100_000.0])

    returns = equity_to_returns(equity_curve)

    return {
        "total_pnl":         round(float(pnls.sum()), 2),
        "trade_count":       int(pnls.size),
        "win_rate":          round(win_rate(pnls), 4),
        "profit_factor":     round(profit_factor(pnls), 3),
        "avg_win_loss_ratio": round(avg_win_loss_ratio(pnls), 3),
        "sharpe":            round(sharpe_ratio(returns), 3),
        "max_drawdown":      round(max_drawdown(equity_curve), 4),
        "annualized_return": round(annualized_return(equity_curve), 4),
    }
