#!/usr/bin/env python3
"""
scripts/backtest_strategy.py
─────────────────────────────
Backtests the short-put-spread and short-call-spread strategies against
historical Alpaca stock bars using the bot's own Black-Scholes pricing,
realized-vol engine, and spread math.  Runs entirely offline — does NOT
touch the live bot, scheduler, or database.

What it calibrates
──────────────────
1.  VRP by ticker          — which names consistently have IV > RV?
                              (tells you which tickers to prioritize)
2.  PoP accuracy sweep     — does PoP ≥ 65% actually produce 65%+ winners?
                              test PoP floors from 55% → 75%.
3.  Filter attribution     — what % of days does each filter block entry?
                              (pinpoints which lever costs you the most fills)
4.  Spread width vs P&L    — expected value at different widths / budget levels.
5.  Per-ticker P&L summary — Sharpe, win rate, avg credit, total trades.

IV assumption
─────────────
Alpaca historical options data only goes back to Feb 2024 and requires
pre-knowing OCC symbols — full historical chain replay is not possible.
Instead we use:

    IV_proxy = rv_yang_zhang(21-day window) × VRP_FACTOR

where VRP_FACTOR defaults to 1.20 (the average IV premium over realized vol
historically observed for liquid US ETFs).  All outputs are labelled with
this assumption.  The PoP, strike, and credit figures will be systematically
slightly higher than real-chain results; the relative ordering across tickers
and parameter sweeps is the calibration signal, not the absolute numbers.

Usage
─────
  # Full run (Feb 2024 → today, all 15 tickers, both spreads):
  ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python scripts/backtest_strategy.py

  # Custom range / tickers:
  python scripts/backtest_strategy.py \
      --start 2024-02-01 --end 2025-12-31 \
      --tickers SPY QQQ IWM GLD \
      --output results/

  # Quick smoke test (90 days, SPY only):
  python scripts/backtest_strategy.py --tickers SPY --start 2025-01-01

Outputs (written to --output dir, default: backtest_results/)
──────
  trades.csv           — every simulated trade (entry/exit/P&L)
  summary.txt          — human-readable calibration tables
  filter_attribution.csv
  pop_sweep.csv
  vrp_by_ticker.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import json
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Bot modules (same pricing logic as the live bot) ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from options_bot.greeks import bs_price, bs_greeks, probability_of_profit, pop_spread
from options_bot.realized_vol import rv_yang_zhang
from options_bot.spread_math import bull_put_entry, calc_spread, profit_target_price

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS   = 252
VRP_FACTOR     = 1.20   # IV proxy = RV × 1.20  (documented assumption)
RISK_FREE_RATE = 0.05   # approximate for the backtest period

DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "TLT", "XLF", "XLK", "XLE",
    "XLV", "XLI", "GLD", "EEM", "HYG", "SMH", "VXX", "XBI",
]

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    # Strategy parameters matching the live bot defaults
    short_put_delta:   float = -0.25
    long_put_delta:    float = -0.10
    short_call_delta:  float =  0.15
    long_call_delta:   float =  0.07
    min_dte:           int   = 28
    max_dte:           int   = 45
    target_dte:        int   = 35       # DTE we aim for at entry
    min_credit:        float = 0.25
    min_pop:           float = 0.65
    min_spread_width:  float = 1.0
    max_spread_width:  float = 20.0
    profit_target_pct: float = 0.50     # close at 50% of credit
    stop_multiplier:   float = 2.0      # stop at 2× credit
    equity:            float = 100_000  # baseline account size
    risk_pct:          float = 0.01     # 1% per trade
    vrp_factor:        float = VRP_FACTOR

    # PoP sweep thresholds to evaluate
    pop_sweep_values: list = field(
        default_factory=lambda: [0.55, 0.60, 0.65, 0.70, 0.75]
    )

# ── Trade record ──────────────────────────────────────────────────────────────
@dataclass
class Trade:
    date:           str
    ticker:         str
    strategy:       str     # "short_put_spread" | "short_call_spread"
    spot:           float
    short_strike:   float
    long_strike:    float
    width:          float
    net_credit:     float
    max_loss:       float
    pop:            float
    dte_at_entry:   int
    iv_proxy:       float
    realized_vol:   float
    exit_date:      str  = ""
    exit_reason:    str  = ""   # "expired_profit" | "stopped_out" | "profit_target"
    pnl:            float = 0.0
    won:            bool  = False

# ── Blocked-entry record (for filter attribution) ─────────────────────────────
@dataclass
class BlockedEntry:
    date:       str
    ticker:     str
    strategy:   str
    reason:     str     # "pop_min" | "credit_min" | "budget" | "width" | "no_rv"

# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_bars(tickers: list[str], start: date, end: date,
               api_key: str, secret_key: str) -> dict[str, pd.DataFrame]:
    """
    Pull daily OHLCV bars from Alpaca for every ticker.
    Returns {ticker: DataFrame(open, high, low, close, volume)} indexed by date.
    """
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=str(start),
        end=str(end),
        adjustment="all",   # split/dividend adjusted
        feed="iex",         # free tier
    )
    bars = client.get_stock_bars(req)

    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = bars[ticker].df if hasattr(bars[ticker], "df") else pd.DataFrame(bars[ticker])
            if df.empty:
                print(f"  WARNING: no data for {ticker}")
                continue
            # Normalise index to date (drop time component)
            df.index = pd.to_datetime(df.index).date
            df.columns = [c.lower() for c in df.columns]
            result[ticker] = df[["open", "high", "low", "close", "volume"]].copy()
        except Exception as exc:
            print(f"  WARNING: {ticker} fetch failed — {exc}")
    return result

# ── IV computation ────────────────────────────────────────────────────────────

def compute_iv_series(df: pd.DataFrame, window: int = 21,
                      vrp_factor: float = VRP_FACTOR) -> pd.Series:
    """
    IV proxy = rv_yang_zhang(window) × vrp_factor.
    Returns a Series indexed by date; first (window-1) rows are NaN.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values

    ivs = []
    dates = list(df.index)
    need = window + 1  # rv_yang_zhang needs window+1 due to overnight gap calc
    for i in range(len(dates)):
        if i < need:
            ivs.append(float("nan"))
            continue
        rv = rv_yang_zhang(
            open_=opens[i - need:i],
            high=highs[i - need:i],
            low=lows[i - need:i],
            close=closes[i - need:i],
            window=window,
        )
        ivs.append(float(rv) * vrp_factor if rv is not None else float("nan"))

    return pd.Series(ivs, index=dates, name="iv_proxy")

# ── Strike selection from target delta ───────────────────────────────────────

def strike_from_delta(S: float, sigma: float, T: float, r: float,
                      target_delta: float, option_type: str) -> float:
    """
    Solve analytically for the strike that gives approximately target_delta.

    For a European put:  delta = N(d1) - 1  → N(d1) = delta + 1
    For a European call: delta = N(d1)      → N(d1) = delta

    d1 = (ln(S/K) + (r + 0.5σ²)T) / (σ√T)
    → K = S × exp(-(N⁻¹(z) × σ√T - (r + 0.5σ²)T))
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return float("nan")
    z = (target_delta + 1) if option_type == "put" else target_delta
    z = max(0.001, min(0.999, z))
    d1_target = norm.ppf(z)
    log_moneyness = -(d1_target * sigma * np.sqrt(T) - (r + 0.5 * sigma**2) * T)
    K = S * np.exp(log_moneyness)
    # Round to nearest valid strike increment
    increment = 0.5 if S < 100 else 1.0
    return round(K / increment) * increment

# ── Single-day spread evaluation ──────────────────────────────────────────────

def evaluate_spread(
    sim_date: date,
    ticker: str,
    strategy: str,
    S: float,
    iv: float,
    rv: float,
    cfg: BacktestConfig,
) -> tuple[Optional[Trade], Optional[BlockedEntry]]:
    """
    Attempt to construct a qualifying spread on `sim_date`.
    Returns (Trade, None) if entry passes all filters, else (None, BlockedEntry).
    """
    T = cfg.target_dte / TRADING_DAYS
    r = RISK_FREE_RATE
    budget = cfg.equity * cfg.risk_pct

    if strategy == "short_put_spread":
        short_delta = cfg.short_put_delta   # e.g. -0.25
        long_delta  = cfg.long_put_delta    # e.g. -0.10
        otype       = "put"
    else:  # short_call_spread
        short_delta = cfg.short_call_delta  # e.g.  0.15
        long_delta  = cfg.long_call_delta   # e.g.  0.07
        otype       = "call"

    # 1. Find strikes at target deltas
    short_K = strike_from_delta(S, iv, T, r, short_delta, otype)
    long_K  = strike_from_delta(S, iv, T, r, long_delta,  otype)
    if np.isnan(short_K) or np.isnan(long_K):
        return None, BlockedEntry(str(sim_date), ticker, strategy, "no_rv")

    # Ensure correct ordering: puts → short > long, calls → short < long
    if otype == "put"  and short_K <= long_K:
        long_K = short_K - 1.0
    if otype == "call" and short_K >= long_K:
        long_K = short_K + 1.0

    width = abs(long_K - short_K)
    if width < cfg.min_spread_width or width > cfg.max_spread_width:
        return None, BlockedEntry(str(sim_date), ticker, strategy, "width")

    # 2. Option prices via Black-Scholes
    short_price = bs_price(S, short_K, T, r, iv, otype)
    long_price  = bs_price(S, long_K,  T, r, iv, otype)
    net_credit  = short_price - long_price

    if net_credit < cfg.min_credit:
        return None, BlockedEntry(str(sim_date), ticker, strategy, "credit_min")

    # 3. PoP — use pop_spread with correct signature
    spread_type = "bull_put" if otype == "put" else "bear_call"
    pop_result  = pop_spread(
        spread_type=spread_type,
        short_strike=short_K,
        long_strike=long_K,
        net_credit=net_credit,
        spot=S,
        sigma=iv,
        rate=r,
        days_to_expiry=cfg.target_dte,
    )
    pop = float(pop_result.get("pop", 0.0)) if isinstance(pop_result, dict) else float(pop_result)
    if pop < cfg.min_pop:
        return None, BlockedEntry(str(sim_date), ticker, strategy, "pop_min")

    # 4. Risk budget
    max_loss = (width - net_credit) * 100
    if max_loss > budget:
        # Attempt budget-fit: narrow the spread
        trial_width = width - 1.0
        while trial_width >= cfg.min_spread_width:
            if otype == "put":
                fit_long_K = short_K - trial_width
            else:
                fit_long_K = short_K + trial_width
            fit_long_price = bs_price(S, fit_long_K, T, r, iv, otype)
            fit_credit     = short_price - fit_long_price
            fit_max_loss   = (trial_width - fit_credit) * 100
            if fit_max_loss <= budget and fit_credit >= cfg.min_credit:
                long_K     = fit_long_K
                long_price = fit_long_price
                net_credit = fit_credit
                width      = trial_width
                max_loss   = fit_max_loss
                break
            trial_width -= 1.0
        else:
            return None, BlockedEntry(str(sim_date), ticker, strategy, "budget")

    trade = Trade(
        date=str(sim_date),
        ticker=ticker,
        strategy=strategy,
        spot=S,
        short_strike=short_K,
        long_strike=long_K,
        width=width,
        net_credit=round(net_credit, 3),
        max_loss=round(max_loss, 2),
        pop=round(pop, 4),
        dte_at_entry=cfg.target_dte,
        iv_proxy=round(iv, 4),
        realized_vol=round(rv, 4),
    )
    return trade, None

# ── Exit simulation ───────────────────────────────────────────────────────────

def simulate_exit(trade: Trade, price_series: pd.Series,
                  cfg: BacktestConfig) -> Trade:
    """
    Walk forward from entry date and determine exit reason + P&L.
    Rules (matching live bot):
      - At 50% of DTE (≈ 17 trading days): take profit if net credit decayed 50%
      - At expiry (DTE trading days out): check if underlying within profit zone
      - Stop: if spread mark reaches 2× credit before either of the above
    """
    entry_date = date.fromisoformat(trade.date)
    otype = "put" if "put" in trade.strategy else "call"
    r     = RISK_FREE_RATE
    profit_target = trade.net_credit * (1 - cfg.profit_target_pct)
    stop_price    = trade.net_credit * cfg.stop_multiplier
    expiry_days   = trade.dte_at_entry
    half_dte      = expiry_days // 2

    future_dates = sorted(
        d for d in price_series.index
        if isinstance(d, date) and d > entry_date
    )

    for i, sim_date in enumerate(future_dates[:expiry_days + 5]):
        days_elapsed   = i + 1
        days_remaining = max(0, expiry_days - days_elapsed)
        T_rem          = days_remaining / TRADING_DAYS
        S              = float(price_series[sim_date])

        # Approximate spread mark using BS
        try:
            short_mark = bs_price(S, trade.short_strike, T_rem, r,
                                  trade.iv_proxy, otype)
            long_mark  = bs_price(S, trade.long_strike,  T_rem, r,
                                  trade.iv_proxy, otype)
            spread_cost = short_mark - long_mark  # cost to close
        except Exception:
            spread_cost = trade.net_credit  # fallback: no change

        # Stop: spread value has doubled from entry credit
        if spread_cost >= stop_price:
            pnl = -(spread_cost - trade.net_credit) * 100
            trade.exit_date   = str(sim_date)
            trade.exit_reason = "stopped_out"
            trade.pnl         = round(pnl, 2)
            trade.won         = False
            return trade

        # Profit target at half-DTE
        if days_elapsed >= half_dte and spread_cost <= profit_target:
            pnl = (trade.net_credit - spread_cost) * 100
            trade.exit_date   = str(sim_date)
            trade.exit_reason = "profit_target"
            trade.pnl         = round(pnl, 2)
            trade.won         = True
            return trade

        # At expiry: intrinsic value
        if days_remaining == 0 or days_elapsed >= expiry_days:
            if otype == "put":
                short_intrinsic = max(0, trade.short_strike - S)
                long_intrinsic  = max(0, trade.long_strike  - S)
            else:
                short_intrinsic = max(0, S - trade.short_strike)
                long_intrinsic  = max(0, S - trade.long_strike)
            spread_intrinsic = short_intrinsic - long_intrinsic
            pnl = (trade.net_credit - spread_intrinsic) * 100
            won = spread_intrinsic <= 0.05  # expired essentially worthless
            trade.exit_date   = str(sim_date)
            trade.exit_reason = "expired_profit" if won else "expired_loss"
            trade.pnl         = round(pnl, 2)
            trade.won         = won
            return trade

    # Ran out of data — mark as open
    trade.exit_date   = "open"
    trade.exit_reason = "open_at_end"
    trade.pnl         = 0.0
    trade.won         = False
    return trade

# ── Main backtest engine ──────────────────────────────────────────────────────

def run_backtest(
    bars: dict[str, pd.DataFrame],
    cfg: BacktestConfig,
    strategies: list[str],
) -> tuple[list[Trade], list[BlockedEntry]]:
    """
    For each ticker and each trading day, attempt to enter one spread per
    strategy (if not already in a position for that ticker/strategy).
    """
    trades:  list[Trade]        = []
    blocked: list[BlockedEntry] = []

    for ticker, df in bars.items():
        iv_series = compute_iv_series(df, vrp_factor=cfg.vrp_factor)
        rv_series = compute_iv_series(df, vrp_factor=1.0)  # RV without VRP factor
        dates     = sorted(df.index)
        close_s   = df["close"]

        for strategy in strategies:
            open_trade: Optional[Trade] = None

            for sim_date in dates:
                S  = float(close_s[sim_date])
                iv = float(iv_series.get(sim_date, float("nan")))
                rv = float(rv_series.get(sim_date, float("nan")))

                if np.isnan(iv) or iv <= 0:
                    continue  # not enough history for vol estimate yet

                # If in an open position, check for exit first
                if open_trade is not None:
                    # Check days elapsed since entry
                    entry_d = date.fromisoformat(open_trade.date)
                    days_in = (sim_date - entry_d).days
                    if days_in >= open_trade.dte_at_entry:
                        # Exit at expiry — simulate properly
                        open_trade = simulate_exit(open_trade, close_s, cfg)
                        trades.append(open_trade)
                        open_trade = None
                    elif days_in >= 1:
                        # Check stop / profit target
                        otype = "put" if "put" in strategy else "call"
                        T_rem = max(0, open_trade.dte_at_entry - days_in) / TRADING_DAYS
                        try:
                            sm = bs_price(S, open_trade.short_strike, T_rem,
                                          RISK_FREE_RATE, iv, otype)
                            lm = bs_price(S, open_trade.long_strike, T_rem,
                                          RISK_FREE_RATE, iv, otype)
                            spread_cost = sm - lm
                        except Exception:
                            spread_cost = open_trade.net_credit

                        stop   = open_trade.net_credit * cfg.stop_multiplier
                        target = open_trade.net_credit * (1 - cfg.profit_target_pct)

                        if spread_cost >= stop:
                            pnl = -(spread_cost - open_trade.net_credit) * 100
                            open_trade.exit_date   = str(sim_date)
                            open_trade.exit_reason = "stopped_out"
                            open_trade.pnl         = round(pnl, 2)
                            open_trade.won         = False
                            trades.append(open_trade)
                            open_trade = None

                        elif days_in >= open_trade.dte_at_entry // 2 and \
                                spread_cost <= target:
                            pnl = (open_trade.net_credit - spread_cost) * 100
                            open_trade.exit_date   = str(sim_date)
                            open_trade.exit_reason = "profit_target"
                            open_trade.pnl         = round(pnl, 2)
                            open_trade.won         = True
                            trades.append(open_trade)
                            open_trade = None

                # Only attempt new entry if flat
                if open_trade is None:
                    trade, block = evaluate_spread(
                        sim_date, ticker, strategy, S, iv, rv, cfg
                    )
                    if trade:
                        open_trade = trade
                    elif block:
                        blocked.append(block)

            # Close any still-open position at end of data
            if open_trade is not None:
                open_trade.exit_date   = "end_of_data"
                open_trade.exit_reason = "open_at_end"
                trades.append(open_trade)

    return trades, blocked

# ── Analysis ──────────────────────────────────────────────────────────────────

def vrp_analysis(bars: dict[str, pd.DataFrame],
                 cfg: BacktestConfig) -> pd.DataFrame:
    rows = []
    for ticker, df in bars.items():
        iv_s  = compute_iv_series(df, vrp_factor=cfg.vrp_factor)
        rv_s  = compute_iv_series(df, vrp_factor=1.0)
        valid = pd.DataFrame({"iv": iv_s, "rv": rv_s}).dropna()
        if valid.empty:
            continue
        vrp = valid["iv"] - valid["rv"]
        rows.append({
            "ticker":        ticker,
            "mean_iv_proxy": round(valid["iv"].mean(), 4),
            "mean_rv":       round(valid["rv"].mean(), 4),
            "mean_vrp":      round(vrp.mean(), 4),
            "pct_days_iv_gt_rv": round((vrp > 0).mean(), 3),
            "vrp_consistency":   round(vrp.std(), 4),
        })
    if not rows:
        return pd.DataFrame(columns=["ticker", "mean_iv_proxy", "mean_rv",
                                     "mean_vrp", "pct_days_iv_gt_rv", "vrp_consistency"])
    df_out = pd.DataFrame(rows).sort_values("mean_vrp", ascending=False)
    return df_out

def pop_sweep(trades: list[Trade], cfg: BacktestConfig) -> pd.DataFrame:
    """
    For each PoP threshold, report what the actual win rate would have been
    if we only took trades above that threshold.
    """
    rows = []
    for threshold in cfg.pop_sweep_values:
        subset = [t for t in trades if t.pop >= threshold
                  and t.exit_reason not in ("open_at_end", "end_of_data")]
        if not subset:
            continue
        won = sum(1 for t in subset if t.won)
        rows.append({
            "pop_threshold":    threshold,
            "trades_taken":     len(subset),
            "actual_win_rate":  round(won / len(subset), 3),
            "avg_credit":       round(np.mean([t.net_credit for t in subset]), 3),
            "avg_pnl":          round(np.mean([t.pnl for t in subset]), 2),
            "total_pnl":        round(sum(t.pnl for t in subset), 2),
        })
    return pd.DataFrame(rows)

def filter_attribution(blocked: list[BlockedEntry],
                       trades: list[Trade]) -> pd.DataFrame:
    """
    Per ticker: how many days was entry attempted, and what fraction was
    blocked by each filter vs successfully entered?
    """
    rows = []
    all_tickers = sorted(set(b.ticker for b in blocked) | set(t.ticker for t in trades))
    reasons = ["pop_min", "credit_min", "budget", "width", "no_rv"]

    for ticker in all_tickers:
        t_blocked = [b for b in blocked if b.ticker == ticker]
        t_entered = [t for t in trades if t.ticker == ticker]
        total = len(t_blocked) + len(t_entered)
        if total == 0:
            continue
        row = {"ticker": ticker, "total_attempts": total,
               "entered": len(t_entered)}
        for r in reasons:
            row[f"blocked_{r}"] = round(
                sum(1 for b in t_blocked if b.reason == r) / total, 3
            )
        row["entry_rate"] = round(len(t_entered) / total, 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_rate", ascending=False)

def per_ticker_summary(trades: list[Trade]) -> pd.DataFrame:
    closed = [t for t in trades if t.exit_reason not in ("open_at_end", "end_of_data")]
    rows   = []
    for ticker in sorted(set(t.ticker for t in closed)):
        sub = [t for t in closed if t.ticker == ticker]
        pnls = np.array([t.pnl for t in sub])
        rows.append({
            "ticker":       ticker,
            "trades":       len(sub),
            "win_rate":     round(sum(t.won for t in sub) / len(sub), 3),
            "avg_credit":   round(np.mean([t.net_credit for t in sub]), 3),
            "avg_width":    round(np.mean([t.width for t in sub]), 2),
            "avg_pnl":      round(pnls.mean(), 2),
            "total_pnl":    round(pnls.sum(), 2),
            "pnl_std":      round(pnls.std(), 2),
            "sharpe":       round(pnls.mean() / pnls.std(), 3) if pnls.std() > 0 else 0,
            "pct_stopped":  round(sum(1 for t in sub if t.exit_reason == "stopped_out") / len(sub), 3),
            "pct_target":   round(sum(1 for t in sub if t.exit_reason == "profit_target") / len(sub), 3),
            "pct_expiry":   round(sum(1 for t in sub if "expired" in t.exit_reason) / len(sub), 3),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)

# ── Output formatting ─────────────────────────────────────────────────────────

def write_summary(
    out_dir: Path,
    trades:  list[Trade],
    blocked: list[BlockedEntry],
    bars:    dict[str, pd.DataFrame],
    cfg:     BacktestConfig,
    start:   date,
    end:     date,
    strategies: list[str],
) -> None:
    closed = [t for t in trades if t.exit_reason not in ("open_at_end", "end_of_data")]

    vrp_df      = vrp_analysis(bars, cfg)
    pop_df      = pop_sweep(trades, cfg)
    filter_df   = filter_attribution(blocked, trades)
    ticker_df   = per_ticker_summary(trades)

    # ── Save CSVs
    pd.DataFrame([asdict(t) for t in trades]).to_csv(out_dir / "trades.csv", index=False)
    vrp_df.to_csv(out_dir    / "vrp_by_ticker.csv",      index=False)
    pop_df.to_csv(out_dir    / "pop_sweep.csv",           index=False)
    filter_df.to_csv(out_dir / "filter_attribution.csv",  index=False)
    ticker_df.to_csv(out_dir / "ticker_summary.csv",      index=False)

    # ── Summary text
    lines = []
    lines += [
        "=" * 70,
        "  OPTIONS BOT — STRATEGY BACKTEST & CALIBRATION REPORT",
        "=" * 70,
        f"  Period   : {start} → {end}",
        f"  Tickers  : {', '.join(sorted(bars))}",
        f"  Strategies: {', '.join(strategies)}",
        f"  IV method: realized_vol × {cfg.vrp_factor} (documented assumption)",
        f"  Budget   : ${cfg.equity:,.0f} × {cfg.risk_pct:.0%} = "
        f"${cfg.equity * cfg.risk_pct:,.0f}/trade",
        "",
        "  NOTE: All credits/widths are synthetic (BS + RV-based IV).",
        "  Relative rankings across tickers and parameter sweeps are the",
        "  calibration signal — absolute numbers will differ from real fills.",
        "=" * 70,
        "",
    ]

    # Overall stats
    if closed:
        pnls = [t.pnl for t in closed]
        lines += [
            "── OVERALL SUMMARY ──────────────────────────────────────",
            f"  Total closed trades : {len(closed)}",
            f"  Win rate            : {sum(t.won for t in closed)/len(closed):.1%}",
            f"  Avg credit          : ${np.mean([t.net_credit for t in closed]):.3f}",
            f"  Avg P&L per trade   : ${np.mean(pnls):.2f}",
            f"  Total P&L           : ${sum(pnls):,.2f}",
            f"  Sharpe (per trade)  : {np.mean(pnls)/np.std(pnls):.2f}",
            f"  Stopped out         : {sum(1 for t in closed if t.exit_reason=='stopped_out')/len(closed):.1%}",
            f"  Profit target hit   : {sum(1 for t in closed if t.exit_reason=='profit_target')/len(closed):.1%}",
            f"  Expired (win/loss)  : {sum(1 for t in closed if 'expired' in t.exit_reason)/len(closed):.1%}",
            "",
        ]

    # ── Calibration 1: VRP by ticker
    lines += ["── CALIBRATION 1: VRP BY TICKER ─────────────────────────",
              "  Tickers with highest VRP = best short-premium candidates.",
              "  IV > RV consistently = you're collecting genuine premium.",
              ""]
    lines.append(f"  {'Ticker':<8} {'Mean IV':>8} {'Mean RV':>8} "
                 f"{'VRP':>8} {'IV>RV %':>8} {'Rank'}") 
    lines.append("  " + "-" * 55)
    for _, row in vrp_df.head(15).iterrows():
        rank = "★★★" if row["pct_days_iv_gt_rv"] >= 0.75 else (
               "★★"  if row["pct_days_iv_gt_rv"] >= 0.60 else "★")
        lines.append(
            f"  {row['ticker']:<8} {row['mean_iv_proxy']:>8.1%} "
            f"{row['mean_rv']:>8.1%} {row['mean_vrp']:>8.1%} "
            f"{row['pct_days_iv_gt_rv']:>7.1%} {rank}"
        )
    lines.append("")

    # ── Calibration 2: PoP sweep
    lines += ["── CALIBRATION 2: POP THRESHOLD SWEEP ───────────────────",
              "  Which PoP floor actually produces PoP% winners?",
              "  Live bot currently uses 65%. Compare to 'actual_win_rate'.",
              ""]
    lines.append(f"  {'PoP floor':>10} {'Trades':>8} {'Actual WR':>10} "
                 f"{'Avg Credit':>11} {'Avg P&L':>9}")
    lines.append("  " + "-" * 52)
    for _, row in pop_df.iterrows():
        calibrated = "✓ CALIBRATED" if abs(row["actual_win_rate"] - row["pop_threshold"]) < 0.05 else ""
        lines.append(
            f"  {row['pop_threshold']:>9.0%} {row['trades_taken']:>8} "
            f"{row['actual_win_rate']:>9.1%}  "
            f"${row['avg_credit']:>9.3f} ${row['avg_pnl']:>8.2f}  {calibrated}"
        )
    lines.append("")

    # ── Calibration 3: Filter attribution
    lines += ["── CALIBRATION 3: FILTER ATTRIBUTION PER TICKER ─────────",
              "  What fraction of scan days is each filter killing?",
              "  High 'blocked_pop_min' → PoP floor may be over-tight.",
              "  High 'blocked_credit_min' → credit floor over-tight.",
              ""]
    lines.append(f"  {'Ticker':<8} {'Entry%':>7} {'PoP':>7} "
                 f"{'Credit':>7} {'Budget':>7} {'Width':>7}")
    lines.append("  " + "-" * 50)
    for _, row in filter_df.iterrows():
        lines.append(
            f"  {row['ticker']:<8} {row['entry_rate']:>6.1%} "
            f"{row['blocked_pop_min']:>6.1%} "
            f"{row['blocked_credit_min']:>6.1%} "
            f"{row['blocked_budget']:>6.1%} "
            f"{row['blocked_width']:>6.1%}"
        )
    lines.append("")

    # ── Calibration 4: Per-ticker P&L
    lines += ["── CALIBRATION 4: PER-TICKER P&L SUMMARY ────────────────",
              "  Sorted by Sharpe ratio (risk-adjusted per-trade performance).",
              ""]
    lines.append(f"  {'Ticker':<8} {'N':>5} {'WR':>6} {'AvgCr':>7} "
                 f"{'AvgPnL':>8} {'TotPnL':>9} {'Sharpe':>7}")
    lines.append("  " + "-" * 58)
    for _, row in ticker_df.iterrows():
        lines.append(
            f"  {row['ticker']:<8} {row['trades']:>5} "
            f"{row['win_rate']:>5.1%} "
            f"${row['avg_credit']:>5.3f} "
            f"${row['avg_pnl']:>7.2f} "
            f"${row['total_pnl']:>8.2f} "
            f"{row['sharpe']:>7.2f}"
        )
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  CSV outputs written to: {out_dir}/")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print(summary_text)
    (out_dir / "summary.txt").write_text(summary_text)

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Options strategy backtest & calibration")
    ap.add_argument("--start",    default="2024-02-01",
                    help="Start date YYYY-MM-DD (default: 2024-02-01, matching Alpaca opt data)")
    ap.add_argument("--end",      default=str(date.today()),
                    help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--tickers",  nargs="+", default=DEFAULT_TICKERS,
                    help="Tickers to backtest")
    ap.add_argument("--strategies", nargs="+",
                    default=["short_put_spread", "short_call_spread"],
                    help="Strategies: short_put_spread short_call_spread")
    ap.add_argument("--output",   default="backtest_results",
                    help="Output directory (default: backtest_results/)")
    ap.add_argument("--equity",   type=float, default=100_000,
                    help="Simulated account equity (default: 100000)")
    ap.add_argument("--vrp-factor", type=float, default=VRP_FACTOR,
                    help=f"IV = RV × factor (default: {VRP_FACTOR})")
    ap.add_argument("--pop-min",  type=float, default=0.65,
                    help="PoP floor for entry filter (default: 0.65)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Fetch data and print shapes, do not run backtest")
    args = ap.parse_args()

    # API keys — same env vars as the live bot
    api_key    = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.", file=sys.stderr)
        return 1

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    cfg = BacktestConfig(
        equity=args.equity,
        vrp_factor=args.vrp_factor,
        min_pop=args.pop_min,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(args.tickers)} tickers from Alpaca ({start} → {end})…")
    bars = fetch_bars(args.tickers, start, end, api_key, secret_key)
    print(f"  Got data for {len(bars)} tickers")
    for ticker, df in bars.items():
        print(f"    {ticker}: {len(df)} trading days, "
              f"{df.index[0]} → {df.index[-1]}")

    if args.dry_run:
        print("\n--dry-run: data fetch OK. Exiting without running backtest.")
        return 0

    print(f"\nRunning backtest ({', '.join(args.strategies)})…")
    trades, blocked = run_backtest(bars, cfg, args.strategies)
    closed = [t for t in trades if t.exit_reason not in ("open_at_end", "end_of_data")]
    print(f"  {len(closed)} closed trades, {len(blocked)} blocked attempts")

    write_summary(out_dir, trades, blocked, bars, cfg, start, end, args.strategies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
