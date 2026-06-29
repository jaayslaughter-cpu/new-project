#!/usr/bin/env python3
"""
scripts/backtest_strategy.py
─────────────────────────────
Backtests short-put-spread and short-call-spread against historical
Alpaca stock bars using the bot's own Black-Scholes pricing, realized-vol
engine, and spread math.  Runs entirely offline — no live bot, scheduler,
or database involvement.

IV sourcing (real where possible, synthetic fallback)
─────────────────────────────────────────────────────
Ticker   Source          Series / method
SPY      FRED VIXCLS     Exact — VIX is the 30-day IV of S&P 500 options
QQQ      yfinance ^VXN   CBOE Nasdaq-100 Volatility Index
IWM      FRED RVX        CBOE Russell 2000 Volatility Index
GLD      FRED GVZ        CBOE Gold ETF Volatility Index
EEM      FRED VXEEM      CBOE Emerging Markets ETF Volatility
TLT      VIX × 0.35      Bond vol ~35% of equity vol (TYVIX discontinued 2019)
VXX      VIX × 1.30      VIX futures premium
XLF      VIX × 1.10      Financials slightly higher than SPY
XLK      VIX × 1.05      Tech close to SPY
XLE      VIX × 1.25      Energy higher vol
XLV      VIX × 0.90      Healthcare defensive, lower vol
XLI      VIX × 1.00      Industrials ~SPY
HYG      VIX × 0.45      Credit, much lower vol than equity
SMH      VIX × 1.15      Semis track Nasdaq, higher vol
XBI      VIX × 1.60      Biotech, highest vol in universe

Real FRED series (VIXCLS, RVX, GVZ, VXEEM) require FRED_API_KEY.
Without it, all tickers fall back to RV × VRP_FACTOR.
Each output table notes which tickers used real vs synthetic IV.

Usage
─────
  ALPACA_API_KEY=... ALPACA_SECRET_KEY=... FRED_API_KEY=... \\
      python scripts/backtest_strategy.py

  # Custom range / tickers:
  python scripts/backtest_strategy.py \\
      --start 2024-02-01 --end 2025-12-31 \\
      --tickers SPY QQQ IWM

  # Dry run (data fetch only):
  python scripts/backtest_strategy.py --dry-run

Outputs (written to --output dir, default: backtest_results/)
──────
  trades.csv            per simulated trade (entry/exit/P&L/iv_source)
  summary.txt           human-readable calibration tables
  filter_attribution.csv
  pop_sweep.csv
  vrp_by_ticker.csv
  ticker_summary.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from options_bot.greeks import bs_price, bs_greeks, pop_spread
from options_bot.realized_vol import rv_yang_zhang
from options_bot.dividends import DIVIDEND_YIELDS

# ── IV source mapping ─────────────────────────────────────────────────────────
# Each entry: (source_type, series_id_or_yf_ticker, vix_multiplier)
# source_type: "fred" | "yf" | "vix_scaled"
# vix_multiplier only used when source_type == "vix_scaled"
# When real source unavailable → fall back to RV × VRP_FACTOR

IV_SOURCES: dict[str, tuple[str, str, float]] = {
    "SPY":  ("fred",       "VIXCLS", 1.00),
    "QQQ":  ("yf",         "^VXN",   1.00),
    "IWM":  ("fred",       "RVX",    1.00),
    "GLD":  ("fred",       "GVZ",    1.00),
    "EEM":  ("fred",       "VXEEM",  1.00),
    "TLT":  ("vix_scaled", "VIXCLS", 0.35),
    "VXX":  ("vix_scaled", "VIXCLS", 1.30),
    "XLF":  ("vix_scaled", "VIXCLS", 1.10),
    "XLK":  ("vix_scaled", "VIXCLS", 1.05),
    "XLE":  ("vix_scaled", "VIXCLS", 1.25),
    "XLV":  ("vix_scaled", "VIXCLS", 0.90),
    "XLI":  ("vix_scaled", "VIXCLS", 1.00),
    "HYG":  ("vix_scaled", "VIXCLS", 0.45),
    "SMH":  ("yf",         "^VXN",   1.15),  # ^VXN × 1.15 for semis
    "XBI":  ("vix_scaled", "VIXCLS", 1.60),
}

TRADING_DAYS   = 252
VRP_FACTOR     = 1.20   # fallback: RV × 1.20 when real IV unavailable
RISK_FREE_RATE = 0.05

DEFAULT_TICKERS = list(IV_SOURCES.keys())


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    short_put_delta:    float = -0.25
    long_put_delta:     float = -0.10
    short_call_delta:   float =  0.15
    long_call_delta:    float =  0.07
    min_dte:            int   = 28
    max_dte:            int   = 45
    target_dte:         int   = 35
    min_credit:         float = 0.25
    min_pop:            float = 0.65
    min_spread_width:   float = 1.0
    max_spread_width:   float = 20.0
    profit_target_pct:  float = 0.50
    stop_multiplier:    float = 2.0
    equity:             float = 100_000
    risk_pct:           float = 0.01
    vrp_factor:         float = VRP_FACTOR
    pop_sweep_values: list = field(
        default_factory=lambda: [0.55, 0.60, 0.65, 0.70, 0.75]
    )


# ── Trade + blocked records ───────────────────────────────────────────────────
@dataclass
class Trade:
    date:           str
    ticker:         str
    strategy:       str
    spot:           float
    short_strike:   float
    long_strike:    float
    width:          float
    net_credit:     float
    max_loss:       float
    pop:            float
    dte_at_entry:   int
    iv_used:        float
    realized_vol:   float
    iv_source:      str   = "synthetic"   # "real" | "vix_scaled" | "synthetic"
    exit_date:      str   = ""
    exit_reason:    str   = ""
    pnl:            float = 0.0
    won:            bool  = False


@dataclass
class BlockedEntry:
    date:     str
    ticker:   str
    strategy: str
    reason:   str


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_bars(tickers: list[str], start: date, end: date,
               api_key: str, secret_key: str) -> dict[str, pd.DataFrame]:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=str(start), end=str(end),
        adjustment="all", feed="iex",
    )
    bars = client.get_stock_bars(req)
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = bars[ticker].df if hasattr(bars[ticker], "df") else pd.DataFrame(bars[ticker])
            if df.empty:
                print(f"  WARNING: no data for {ticker}"); continue
            df.index = pd.to_datetime(df.index).date
            df.columns = [c.lower() for c in df.columns]
            result[ticker] = df[["open", "high", "low", "close", "volume"]].copy()
        except Exception as exc:
            print(f"  WARNING: {ticker} fetch failed — {exc}")
    return result


def fetch_vol_indices(start: date, end: date,
                      fred_key: str) -> dict[str, pd.Series]:
    """
    Fetch real vol indices from FRED and yfinance.
    Returns {series_id: pd.Series(date -> decimal_iv)}.
    FRED VIX/RVX/GVZ/VXEEM are in percent (17.5 = 17.5%), divide by 100.
    """
    result: dict[str, pd.Series] = {}

    # ── FRED series ───────────────────────────────────────────────────────────
    fred_series = ["VIXCLS", "RVX", "GVZ", "VXEEM"]
    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            for series_id in fred_series:
                try:
                    raw = fred.get_series(series_id,
                                          observation_start=str(start),
                                          observation_end=str(end))
                    s = raw.dropna() / 100.0   # percent → decimal
                    s.index = pd.to_datetime(s.index).date
                    result[series_id] = s
                    print(f"  FRED {series_id}: {len(s)} observations")
                except Exception as exc:
                    print(f"  WARNING: FRED {series_id} failed — {exc}")
        except ImportError:
            print("  WARNING: fredapi not installed, FRED vol indices unavailable")
    else:
        print("  INFO: FRED_API_KEY not set — FRED vol indices skipped")

    # ── yfinance series (^VXN) ────────────────────────────────────────────────
    yf_series = {"^VXN": "^VXN"}
    for label, yticker in yf_series.items():
        try:
            import yfinance as yf
            df = yf.download(yticker, start=str(start), end=str(end),
                             progress=False, auto_adjust=True)
            if df.empty:
                print(f"  WARNING: yfinance {yticker} returned empty")
                continue
            s = (df["Close"] / 100.0).dropna()
            s.index = pd.to_datetime(s.index).date
            result[label] = s
            print(f"  yfinance {yticker}: {len(s)} observations")
        except Exception as exc:
            print(f"  WARNING: yfinance {yticker} failed — {exc}")

    return result


def build_iv_lookup(ticker: str,
                    vol_data: dict[str, pd.Series],
                    rv_series: pd.Series,
                    cfg: BacktestConfig) -> tuple[pd.Series, str]:
    """
    For a given ticker, return (iv_series, source_label) where:
      iv_series: date -> decimal IV
      source_label: "real" | "vix_scaled" | "synthetic"

    Priority:
      1. Direct real index  (FRED or yfinance, mult=1.0)
      2. VIX-scaled         (VIX × ticker multiplier)
      3. Synthetic fallback (RV × vrp_factor)
    """
    if ticker not in IV_SOURCES:
        return rv_series * cfg.vrp_factor, "synthetic"

    src_type, series_id, mult = IV_SOURCES[ticker]

    # Case 1: real index available
    if src_type in ("fred", "yf") and series_id in vol_data:
        base = vol_data[series_id]
        iv   = base * mult   # mult is 1.0 for these, but handle SMH (^VXN × 1.15)
        label = "real" if mult == 1.00 else "vix_scaled"
        return iv, label

    # Case 2: VIX-scaled (or real failed, fall through)
    if "VIXCLS" in vol_data:
        iv = vol_data["VIXCLS"] * mult
        return iv, "vix_scaled"

    # Case 3: synthetic fallback
    return rv_series * cfg.vrp_factor, "synthetic"


# ── RV computation ────────────────────────────────────────────────────────────

def compute_rv_series(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Yang-Zhang realized vol series (decimal annualized), one value per date."""
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    ivs    = []
    dates  = list(df.index)
    need   = window + 1

    for i in range(len(dates)):
        if i < need:
            ivs.append(float("nan")); continue
        rv = rv_yang_zhang(
            open_=opens[i - need:i], high=highs[i - need:i],
            low=lows[i - need:i],   close=closes[i - need:i],
            window=window,
        )
        ivs.append(float(rv) if rv is not None else float("nan"))
    return pd.Series(ivs, index=dates, name="rv")


# ── Strike selection ──────────────────────────────────────────────────────────

def strike_from_delta(S: float, sigma: float, T: float, r: float,
                      target_delta: float, option_type: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return float("nan")
    z = (target_delta + 1) if option_type == "put" else target_delta
    z = max(0.001, min(0.999, z))
    d1_target    = norm.ppf(z)
    log_moneyness = -(d1_target * sigma * np.sqrt(T) - (r + 0.5 * sigma**2) * T)
    K             = S * np.exp(log_moneyness)
    increment     = 0.5 if S < 100 else 1.0
    return round(K / increment) * increment


# ── Spread evaluation ─────────────────────────────────────────────────────────

def evaluate_spread(
    sim_date: date, ticker: str, strategy: str,
    S: float, iv: float, rv: float, iv_source: str,
    cfg: BacktestConfig,
) -> tuple[Optional[Trade], Optional[BlockedEntry]]:
    T      = cfg.target_dte / TRADING_DAYS
    r      = RISK_FREE_RATE
    budget = cfg.equity * cfg.risk_pct
    q      = DIVIDEND_YIELDS.get(ticker, 0.0)   # per-ticker dividend yield

    if strategy == "short_put_spread":
        short_delta = cfg.short_put_delta
        long_delta  = cfg.long_put_delta
        otype       = "put"
    else:
        short_delta = cfg.short_call_delta
        long_delta  = cfg.long_call_delta
        otype       = "call"

    short_K = strike_from_delta(S, iv, T, r, short_delta, otype)
    long_K  = strike_from_delta(S, iv, T, r, long_delta,  otype)
    if np.isnan(short_K) or np.isnan(long_K):
        return None, BlockedEntry(str(sim_date), ticker, strategy, "no_iv")

    if otype == "put"  and short_K <= long_K: long_K = short_K - 1.0
    if otype == "call" and short_K >= long_K: long_K = short_K + 1.0

    width = abs(long_K - short_K)
    if width < cfg.min_spread_width or width > cfg.max_spread_width:
        return None, BlockedEntry(str(sim_date), ticker, strategy, "width")

    short_price = bs_price(S, short_K, T, r, iv, otype, q)
    long_price  = bs_price(S, long_K,  T, r, iv, otype, q)
    net_credit  = short_price - long_price

    if net_credit < cfg.min_credit:
        return None, BlockedEntry(str(sim_date), ticker, strategy, "credit_min")

    spread_type = "bull_put" if otype == "put" else "bear_call"
    pop_result  = pop_spread(
        spread_type=spread_type,
        short_strike=short_K, long_strike=long_K,
        net_credit=net_credit, spot=S,
        sigma=iv, rate=r, days_to_expiry=cfg.target_dte,
        dividend_yield=q,
    )
    pop = float(pop_result.get("pop", 0.0)) if isinstance(pop_result, dict) \
          else float(pop_result)

    if pop < cfg.min_pop:
        return None, BlockedEntry(str(sim_date), ticker, strategy, "pop_min")

    max_loss = (width - net_credit) * 100
    if max_loss > budget:
        # Budget-fit: walk narrower until it fits
        trial_w = width - 1.0
        while trial_w >= cfg.min_spread_width:
            fit_K     = short_K - trial_w if otype == "put" else short_K + trial_w
            fit_lp    = bs_price(S, fit_K, T, r, iv, otype, q)
            fit_cred  = short_price - fit_lp
            fit_loss  = (trial_w - fit_cred) * 100
            if fit_loss <= budget and fit_cred >= cfg.min_credit:
                long_K = fit_K; long_price = fit_lp
                net_credit = fit_cred; width = trial_w; max_loss = fit_loss
                break
            trial_w -= 1.0
        else:
            return None, BlockedEntry(str(sim_date), ticker, strategy, "budget")

    return Trade(
        date=str(sim_date), ticker=ticker, strategy=strategy,
        spot=S, short_strike=short_K, long_strike=long_K,
        width=width, net_credit=round(net_credit, 3),
        max_loss=round(max_loss, 2), pop=round(pop, 4),
        dte_at_entry=cfg.target_dte,
        iv_used=round(iv, 4), realized_vol=round(rv, 4),
        iv_source=iv_source,
    ), None


# ── Exit simulation ───────────────────────────────────────────────────────────

def simulate_exit(trade: Trade, close_s: pd.Series, iv_s: pd.Series,
                  cfg: BacktestConfig) -> Trade:
    entry_date = date.fromisoformat(trade.date)
    otype      = "put" if "put" in trade.strategy else "call"
    r          = RISK_FREE_RATE
    q          = DIVIDEND_YIELDS.get(trade.ticker, 0.0)
    stop_p     = trade.net_credit * cfg.stop_multiplier
    target_p   = trade.net_credit * (1 - cfg.profit_target_pct)
    half_dte   = trade.dte_at_entry // 2

    future = sorted(d for d in close_s.index
                    if isinstance(d, date) and d > entry_date)

    for i, sim_date in enumerate(future[:trade.dte_at_entry + 5]):
        days_elapsed   = i + 1
        days_remaining = max(0, trade.dte_at_entry - days_elapsed)
        T_rem          = days_remaining / TRADING_DAYS
        S              = float(close_s[sim_date])
        # Use the same IV source if available on this date, else trade entry IV
        iv = float(iv_s.get(sim_date, trade.iv_used)) if iv_s is not None \
             else trade.iv_used

        try:
            sm = bs_price(S, trade.short_strike, T_rem, r, iv, otype, q)
            lm = bs_price(S, trade.long_strike,  T_rem, r, iv, otype, q)
            spread_cost = sm - lm
        except Exception:
            spread_cost = trade.net_credit

        if spread_cost >= stop_p:
            trade.exit_date   = str(sim_date)
            trade.exit_reason = "stopped_out"
            trade.pnl         = round(-(spread_cost - trade.net_credit) * 100, 2)
            trade.won         = False
            return trade

        if days_elapsed >= half_dte and spread_cost <= target_p:
            trade.exit_date   = str(sim_date)
            trade.exit_reason = "profit_target"
            trade.pnl         = round((trade.net_credit - spread_cost) * 100, 2)
            trade.won         = True
            return trade

        if days_remaining == 0 or days_elapsed >= trade.dte_at_entry:
            if otype == "put":
                si = max(0, trade.short_strike - S)
                li = max(0, trade.long_strike  - S)
            else:
                si = max(0, S - trade.short_strike)
                li = max(0, S - trade.long_strike)
            intrinsic = si - li
            won = intrinsic <= 0.05
            trade.exit_date   = str(sim_date)
            trade.exit_reason = "expired_profit" if won else "expired_loss"
            trade.pnl         = round((trade.net_credit - intrinsic) * 100, 2)
            trade.won         = won
            return trade

    trade.exit_date = "end_of_data"; trade.exit_reason = "open_at_end"
    return trade


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    bars:     dict[str, pd.DataFrame],
    vol_data: dict[str, pd.Series],
    cfg:      BacktestConfig,
    strategies: list[str],
) -> tuple[list[Trade], list[BlockedEntry]]:
    trades:  list[Trade]        = []
    blocked: list[BlockedEntry] = []

    for ticker, df in bars.items():
        rv_s = compute_rv_series(df)

        # Build real IV series for this ticker
        iv_s, iv_source_label = build_iv_lookup(ticker, vol_data, rv_s, cfg)

        # Align to daily close index
        close_s = df["close"]
        dates   = sorted(df.index)

        # For exit simulation we need IV series as a Series indexed by date
        if isinstance(iv_s, pd.Series):
            iv_aligned = iv_s.reindex(dates)
        else:
            iv_aligned = pd.Series(dtype=float)

        for strategy in strategies:
            open_trade: Optional[Trade] = None

            for sim_date in dates:
                S  = float(close_s[sim_date])
                rv = float(rv_s.get(sim_date, float("nan"))
                          if isinstance(rv_s, pd.Series) else float("nan"))

                # Get IV for this date
                if isinstance(iv_s, pd.Series):
                    iv = float(iv_s.get(sim_date, float("nan")))
                    src = iv_source_label
                else:
                    iv = float("nan")
                    src = "synthetic"

                # Fallback chain: real → vix_scaled → synthetic RV
                if np.isnan(iv) or iv <= 0:
                    if not np.isnan(rv) and rv > 0:
                        iv  = rv * cfg.vrp_factor
                        src = "synthetic"
                    else:
                        continue

                # Manage open position
                if open_trade is not None:
                    entry_d   = date.fromisoformat(open_trade.date)
                    days_in   = (sim_date - entry_d).days
                    otype_ot  = "put" if "put" in strategy else "call"
                    T_rem     = max(0, open_trade.dte_at_entry - days_in) / TRADING_DAYS

                    if days_in >= open_trade.dte_at_entry:
                        open_trade = simulate_exit(open_trade, close_s, iv_aligned, cfg)
                        trades.append(open_trade); open_trade = None
                    elif days_in >= 1:
                        try:
                            _q = DIVIDEND_YIELDS.get(ticker, 0.0)
                            sm = bs_price(S, open_trade.short_strike, T_rem,
                                          RISK_FREE_RATE, iv, otype_ot, _q)
                            lm = bs_price(S, open_trade.long_strike, T_rem,
                                          RISK_FREE_RATE, iv, otype_ot, _q)
                            sc = sm - lm
                        except Exception:
                            sc = open_trade.net_credit

                        stop_p   = open_trade.net_credit * cfg.stop_multiplier
                        target_p = open_trade.net_credit * (1 - cfg.profit_target_pct)

                        if sc >= stop_p:
                            open_trade.exit_date   = str(sim_date)
                            open_trade.exit_reason = "stopped_out"
                            open_trade.pnl         = round(-(sc - open_trade.net_credit)*100, 2)
                            open_trade.won         = False
                            trades.append(open_trade); open_trade = None
                        elif days_in >= open_trade.dte_at_entry // 2 and sc <= target_p:
                            open_trade.exit_date   = str(sim_date)
                            open_trade.exit_reason = "profit_target"
                            open_trade.pnl         = round((open_trade.net_credit - sc)*100, 2)
                            open_trade.won         = True
                            trades.append(open_trade); open_trade = None

                if open_trade is None:
                    trade, block = evaluate_spread(
                        sim_date, ticker, strategy, S, iv, rv if not np.isnan(rv) else 0.0,
                        src, cfg
                    )
                    if trade:
                        open_trade = trade
                    elif block:
                        blocked.append(block)

            if open_trade is not None:
                open_trade.exit_date = "end_of_data"
                open_trade.exit_reason = "open_at_end"
                trades.append(open_trade)

    return trades, blocked


# ── Analysis ──────────────────────────────────────────────────────────────────

def vrp_analysis(bars: dict[str, pd.DataFrame],
                 vol_data: dict[str, pd.Series],
                 cfg: BacktestConfig) -> pd.DataFrame:
    rows = []
    for ticker, df in bars.items():
        rv_s = compute_rv_series(df)
        iv_s, src = build_iv_lookup(ticker, vol_data, rv_s, cfg)
        if isinstance(iv_s, pd.Series):
            valid = pd.DataFrame({"iv": iv_s, "rv": rv_s}).dropna()
        else:
            valid = pd.DataFrame({"iv": rv_s * cfg.vrp_factor, "rv": rv_s}).dropna()
        if valid.empty:
            continue
        vrp = valid["iv"] - valid["rv"]
        rows.append({
            "ticker":            ticker,
            "iv_source":         src,
            "mean_iv":           round(valid["iv"].mean(), 4),
            "mean_rv":           round(valid["rv"].mean(), 4),
            "mean_vrp":          round(vrp.mean(), 4),
            "pct_days_iv_gt_rv": round((vrp > 0).mean(), 3),
            "vrp_std":           round(vrp.std(), 4),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mean_vrp", ascending=False)


def pop_sweep(trades: list[Trade], cfg: BacktestConfig) -> pd.DataFrame:
    rows = []
    closed = [t for t in trades if t.exit_reason not in ("open_at_end","end_of_data")]
    for threshold in cfg.pop_sweep_values:
        sub = [t for t in closed if t.pop >= threshold]
        if not sub:
            continue
        won = sum(t.won for t in sub)
        rows.append({
            "pop_threshold": threshold,
            "trades_taken":  len(sub),
            "actual_win_rate": round(won / len(sub), 3),
            "avg_credit":    round(np.mean([t.net_credit for t in sub]), 3),
            "avg_pnl":       round(np.mean([t.pnl for t in sub]), 2),
            "total_pnl":     round(sum(t.pnl for t in sub), 2),
        })
    return pd.DataFrame(rows)


def filter_attribution(blocked: list[BlockedEntry],
                       trades: list[Trade]) -> pd.DataFrame:
    all_tickers = sorted(set(b.ticker for b in blocked) | set(t.ticker for t in trades))
    reasons = ["pop_min", "credit_min", "budget", "width", "no_iv"]
    rows = []
    for ticker in all_tickers:
        tb = [b for b in blocked if b.ticker == ticker]
        te = [t for t in trades if t.ticker == ticker]
        total = len(tb) + len(te)
        if total == 0:
            continue
        row = {"ticker": ticker, "total_attempts": total, "entered": len(te)}
        for r in reasons:
            row[f"blocked_{r}"] = round(sum(1 for b in tb if b.reason == r) / total, 3)
        row["entry_rate"] = round(len(te) / total, 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_rate", ascending=False)


def per_ticker_summary(trades: list[Trade]) -> pd.DataFrame:
    closed = [t for t in trades if t.exit_reason not in ("open_at_end","end_of_data")]
    rows   = []
    for ticker in sorted(set(t.ticker for t in closed)):
        sub  = [t for t in closed if t.ticker == ticker]
        pnls = np.array([t.pnl for t in sub])
        # Determine primary IV source
        src_counts = {}
        for t in sub:
            src_counts[t.iv_source] = src_counts.get(t.iv_source, 0) + 1
        primary_src = max(src_counts, key=src_counts.get)
        rows.append({
            "ticker":      ticker,
            "iv_source":   primary_src,
            "trades":      len(sub),
            "win_rate":    round(sum(t.won for t in sub) / len(sub), 3),
            "avg_credit":  round(np.mean([t.net_credit for t in sub]), 3),
            "avg_width":   round(np.mean([t.width for t in sub]), 2),
            "avg_pnl":     round(pnls.mean(), 2),
            "total_pnl":   round(pnls.sum(), 2),
            "sharpe":      round(pnls.mean()/pnls.std(), 3) if pnls.std() > 0 else 0,
            "pct_stopped": round(sum(1 for t in sub if t.exit_reason=="stopped_out")/len(sub),3),
            "pct_target":  round(sum(1 for t in sub if t.exit_reason=="profit_target")/len(sub),3),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# ── Output ────────────────────────────────────────────────────────────────────

def write_outputs(
    out_dir: Path, trades: list[Trade], blocked: list[BlockedEntry],
    bars: dict[str, pd.DataFrame], vol_data: dict[str, pd.Series],
    cfg: BacktestConfig, start: date, end: date, strategies: list[str],
) -> None:
    closed = [t for t in trades if t.exit_reason not in ("open_at_end","end_of_data")]
    vrp_df    = vrp_analysis(bars, vol_data, cfg)
    pop_df    = pop_sweep(trades, cfg)
    filter_df = filter_attribution(blocked, trades)
    ticker_df = per_ticker_summary(trades)

    pd.DataFrame([asdict(t) for t in trades]).to_csv(out_dir/"trades.csv", index=False)
    vrp_df.to_csv(out_dir/"vrp_by_ticker.csv",     index=False)
    pop_df.to_csv(out_dir/"pop_sweep.csv",          index=False)
    filter_df.to_csv(out_dir/"filter_attribution.csv", index=False)
    ticker_df.to_csv(out_dir/"ticker_summary.csv",  index=False)

    # ── IV source legend
    real_tickers      = [t for t in bars if IV_SOURCES.get(t, ("","",""))[0] in ("fred","yf")
                         and IV_SOURCES[t][1] in vol_data]
    vix_scaled_tickers = [t for t in bars if t not in real_tickers
                          and "VIXCLS" in vol_data]
    synthetic_tickers  = [t for t in bars if t not in real_tickers
                          and t not in vix_scaled_tickers]

    lines = [
        "=" * 70,
        "  OPTIONS BOT — STRATEGY BACKTEST & CALIBRATION REPORT",
        "=" * 70,
        f"  Period     : {start} → {end}",
        f"  Tickers    : {', '.join(sorted(bars))}",
        f"  Strategies : {', '.join(strategies)}",
        "",
        "  IV QUALITY KEY:",
        f"  [REAL]      : {', '.join(real_tickers) or 'none — FRED_API_KEY not set'}",
        "                Real market-priced implied vol (VIX/RVX/GVZ/VXEEM/VXN)",
        f"  [VIX-SCALED]: {', '.join(vix_scaled_tickers) or 'none'}",
        "                Real VIX level × calibrated beta multiplier",
        f"  [SYNTHETIC] : {', '.join(synthetic_tickers) or 'none'}",
        "                RV × 1.20 — assumed average VRP premium",
        "",
        "  Rankings and relative comparisons are reliable for all tiers.",
        "  Absolute credit/P&L most accurate for [REAL] tickers.",
        "=" * 70,
        "",
    ]

    if closed:
        pnls = [t.pnl for t in closed]
        real_pct = sum(1 for t in closed if t.iv_source=="real") / len(closed)
        lines += [
            "── OVERALL SUMMARY ──────────────────────────────────────",
            f"  Total closed trades : {len(closed)}",
            f"  Trades with real IV : {real_pct:.0%}",
            f"  Win rate            : {sum(t.won for t in closed)/len(closed):.1%}",
            f"  Avg credit          : ${np.mean([t.net_credit for t in closed]):.3f}",
            f"  Avg P&L per trade   : ${np.mean(pnls):.2f}",
            f"  Total P&L           : ${sum(pnls):,.2f}",
            f"  Sharpe (per trade)  : {np.mean(pnls)/np.std(pnls):.2f}",
            f"  Stopped out         : {sum(1 for t in closed if t.exit_reason=='stopped_out')/len(closed):.1%}",
            f"  Profit target       : {sum(1 for t in closed if t.exit_reason=='profit_target')/len(closed):.1%}",
            "",
        ]

    # VRP table
    lines += ["── CALIBRATION 1: VRP BY TICKER ─────────────────────────",
              "  ★★★ = IV>RV on ≥75% of days (strong VRP — prioritize these)",
              "  ★★  = 60-75%  ★ = <60%",
              ""]
    if not vrp_df.empty:
        lines.append(f"  {'Ticker':<8} {'Src':>10} {'Mean IV':>8} "
                     f"{'Mean RV':>8} {'VRP':>8} {'IV>RV%':>8}")
        lines.append("  " + "─" * 58)
        for _, row in vrp_df.iterrows():
            star = "★★★" if row["pct_days_iv_gt_rv"]>=0.75 else (
                   "★★"  if row["pct_days_iv_gt_rv"]>=0.60 else "★")
            src_tag = f"[{row['iv_source'].upper()[:5]}]"
            lines.append(
                f"  {row['ticker']:<8} {src_tag:>10} "
                f"{row['mean_iv']:>7.1%} {row['mean_rv']:>8.1%} "
                f"{row['mean_vrp']:>8.1%} {row['pct_days_iv_gt_rv']:>7.1%} {star}"
            )
    lines.append("")

    # PoP sweep
    lines += ["── CALIBRATION 2: POP THRESHOLD SWEEP ───────────────────",
              "  Current live bot uses 65%. Compare 'pop_threshold' to 'actual_win_rate'.",
              "  If actual WR >> threshold → model is conservative → can tighten.",
              "  If actual WR << threshold → model is optimistic → keep 65% or raise.",
              ""]
    if not pop_df.empty:
        lines.append(f"  {'PoP floor':>10} {'Trades':>8} {'Actual WR':>10} "
                     f"{'Avg Cred':>10} {'Avg P&L':>9} {'Calibrated?'}")
        lines.append("  " + "─" * 62)
        for _, row in pop_df.iterrows():
            diff = abs(row["actual_win_rate"] - row["pop_threshold"])
            tag  = "✓ well-calibrated" if diff <= 0.05 else (
                   "→ model CONSERVATIVE" if row["actual_win_rate"] > row["pop_threshold"]+0.05
                   else "⚠ model OPTIMISTIC")
            lines.append(
                f"  {row['pop_threshold']:>9.0%} {row['trades_taken']:>8} "
                f"{row['actual_win_rate']:>9.1%} "
                f"${row['avg_credit']:>8.3f} "
                f"${row['avg_pnl']:>8.2f}  {tag}"
            )
    lines.append("")

    # Filter attribution
    lines += ["── CALIBRATION 3: FILTER ATTRIBUTION ────────────────────",
              "  % of scan days blocked by each filter per ticker.",
              "  High 'blocked_pop_min'    → PoP floor may be over-tight",
              "  High 'blocked_credit_min' → min credit floor over-tight",
              "  High 'blocked_budget'     → account too small for this ticker",
              ""]
    if not filter_df.empty:
        lines.append(f"  {'Ticker':<8} {'Entry%':>7} {'PoP':>7} "
                     f"{'Credit':>7} {'Budget':>7} {'Width':>7}")
        lines.append("  " + "─" * 50)
        for _, row in filter_df.iterrows():
            lines.append(
                f"  {row['ticker']:<8} {row['entry_rate']:>6.1%} "
                f"{row['blocked_pop_min']:>6.1%} {row['blocked_credit_min']:>6.1%} "
                f"{row['blocked_budget']:>6.1%} {row['blocked_width']:>6.1%}"
            )
    lines.append("")

    # Ticker P&L summary
    lines += ["── CALIBRATION 4: PER-TICKER P&L (sorted by Sharpe) ─────",
              "  [REAL] IV tickers are most reliable for absolute P&L figures.",
              ""]
    if not ticker_df.empty:
        lines.append(f"  {'Ticker':<8} {'Src':>10} {'N':>5} {'WR':>6} "
                     f"{'AvgCr':>7} {'AvgPnL':>8} {'TotPnL':>9} {'Sharpe':>7}")
        lines.append("  " + "─" * 68)
        for _, row in ticker_df.iterrows():
            src_tag = f"[{str(row['iv_source']).upper()[:5]}]"
            lines.append(
                f"  {row['ticker']:<8} {src_tag:>10} {row['trades']:>5} "
                f"{row['win_rate']:>5.1%} "
                f"${row['avg_credit']:>5.3f} "
                f"${row['avg_pnl']:>7.2f} "
                f"${row['total_pnl']:>8.2f} "
                f"{row['sharpe']:>7.2f}"
            )
    lines += ["", "=" * 70,
              f"  Outputs written to: {out_dir}/",
              "=" * 70]

    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",      default="2024-02-01")
    ap.add_argument("--end",        default=str(date.today()))
    ap.add_argument("--tickers",    nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--strategies", nargs="+",
                    default=["short_put_spread", "short_call_spread"])
    ap.add_argument("--output",     default="backtest_results")
    ap.add_argument("--equity",     type=float, default=100_000)
    ap.add_argument("--vrp-factor", type=float, default=VRP_FACTOR)
    ap.add_argument("--pop-min",    type=float, default=0.65)
    ap.add_argument("--dry-run",    action="store_true")
    args = ap.parse_args()

    api_key    = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    fred_key   = os.getenv("FRED_API_KEY", "")

    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.")
        return 1

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    cfg   = BacktestConfig(equity=args.equity, vrp_factor=args.vrp_factor,
                           min_pop=args.pop_min)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFetching {len(args.tickers)} tickers from Alpaca ({start} → {end})…")
    bars = fetch_bars(args.tickers, start, end, api_key, secret_key)
    print(f"  Got data for {len(bars)} tickers")

    print("\nFetching vol indices (FRED + yfinance)…")
    vol_data = fetch_vol_indices(start, end, fred_key)
    print(f"  Got {len(vol_data)} vol series: {', '.join(vol_data)}")

    # Report IV coverage
    for ticker in args.tickers:
        if ticker not in IV_SOURCES:
            print(f"  {ticker}: synthetic (no IV source configured)")
            continue
        src_type, series_id, mult = IV_SOURCES[ticker]
        if series_id in vol_data:
            tag = "REAL" if mult == 1.00 else f"REAL × {mult}"
            print(f"  {ticker}: {tag} ({series_id})")
        elif "VIXCLS" in vol_data and src_type == "vix_scaled":
            print(f"  {ticker}: VIX-scaled (× {mult})")
        else:
            print(f"  {ticker}: synthetic fallback (RV × {cfg.vrp_factor})")

    if args.dry_run:
        print("\n--dry-run: fetch OK. Exiting.")
        return 0

    print(f"\nRunning backtest ({', '.join(args.strategies)})…")
    trades, blocked = run_backtest(bars, vol_data, cfg, args.strategies)
    closed = [t for t in trades if t.exit_reason not in ("open_at_end","end_of_data")]
    print(f"  {len(closed)} closed trades, {len(blocked)} blocked attempts")

    write_outputs(out_dir, trades, blocked, bars, vol_data, cfg,
                  start, end, args.strategies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
