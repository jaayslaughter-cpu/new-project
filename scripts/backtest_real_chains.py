#!/usr/bin/env python3
"""
scripts/backtest_real_chains.py
─────────────────────────────────
Backtests ShortPutSpread / ShortCallSpread against REAL, market-quoted
options chains — no Black-Scholes reconstruction, no synthetic IV. Every
credit, every delta, every exit cost comes directly from actual recorded
bid/ask/greeks.

Data source
───────────
post-no-preference/options on DoltHub (CC-licensed, free, 2019–present,
~2,098 underlyings). Two tables:

  option_chain        date, act_symbol, expiration, strike, call_put,
                       bid, ask, vol, delta, gamma, theta, vega, rho
  volatility_history   date, act_symbol, hv_current/week/month/year_high/low,
                       iv_current/week/month/year_high/low  (IV-rank context)

This script reads CSV exports of those tables — it does not call the
DoltHub API directly (no network access required to run it).

Getting the export
───────────────────
DoltHub's SQL workbench lets you run arbitrary SQL and download the result
as CSV. Run these two queries (replace the ticker list / date range as
needed) and download each result:

    SELECT * FROM option_chain
    WHERE act_symbol IN ('SPY','QQQ','IWM','TLT','XLF','XLK','XLE','XLV',
                          'XLI','GLD','EEM','HYG','SMH','VXX','XBI')
      AND date >= '2020-01-01'
      AND DATEDIFF(expiration, date) BETWEEN 7 AND 65
      AND ABS(delta) BETWEEN 0.02 AND 0.40;

    SELECT * FROM volatility_history
    WHERE act_symbol IN ('SPY','QQQ','IWM','TLT','XLF','XLK','XLE','XLV',
                          'XLI','GLD','EEM','HYG','SMH','VXX','XBI')
      AND date >= '2020-01-01';

The DTE and delta filters above match the bot's actual trading range
(DTE 14–60, |delta| up to 0.40 covers every strategy's target deltas with
margin) and cut the export from "all strikes of every expiration" down to
only the rows the strategies could ever actually select — this keeps the
file size manageable instead of pulling the full chain.

Why real quotes change everything vs the synthetic backtest
─────────────────────────────────────────────────────────────
- Entry credit = short_bid − long_ask (you receive the bid when selling,
  pay the ask when buying) — the bid-ask spread is real, built-in slippage,
  not assumed away.
- Delta is the recorded market delta — no IV solving, no spot price needed.
- Exit cost-to-close is read from the SAME contract's real quote on a later
  date — by expiration this naturally converges to intrinsic value, with
  no separate intrinsic-value formula needed.
- PoP is approximated as 1 − |short_delta|, the standard practitioner
  shorthand (not a Black-Scholes touch-probability calc, since this
  dataset has no underlying spot price column to feed one). Documented
  and clearly labeled — this is the one approximation in an otherwise
  fully real-data backtest.

Usage
─────
    python scripts/backtest_real_chains.py \\
        --chain option_chain_export.csv \\
        --output backtest_real_results/

Outputs: trades.csv, ticker_summary.csv, summary.txt
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ── Strategy config — mirrors the live bot's actual ShortPutSpread /
# ShortCallSpread defaults (src/options_bot/strategy.py) ──────────────────

@dataclass
class RealChainConfig:
    short_put_delta:   float = -0.15
    long_put_delta:    float = -0.07
    short_call_delta:  float =  0.15
    long_call_delta:   float =  0.07
    min_dte:           int   = 14
    max_dte:           int   = 60
    min_credit:        float = 0.25
    min_pop:           float = 0.65
    min_spread_width:  float = 1.0
    max_spread_width:  float = 20.0
    profit_target_pct: float = 0.50     # close at 50% of credit captured
    stop_multiplier:   float = 2.0      # stop at 2x credit received
    equity:            float = 100_000
    risk_pct:          float = 0.01
    half_dte_target:   bool  = True     # only take profit at/after half-DTE
    pop_sweep_values: list = field(
        default_factory=lambda: [0.55, 0.60, 0.65, 0.70, 0.75]
    )


@dataclass
class Trade:
    entry_date:     str
    ticker:         str
    strategy:       str
    short_strike:   float
    long_strike:    float
    width:          float
    net_credit:     float
    max_loss:       float
    short_delta:    float
    pop_approx:     float
    dte_at_entry:   int
    expiration:     str
    exit_date:      str  = ""
    exit_reason:    str  = ""
    pnl:            float = 0.0
    won:            bool  = False


@dataclass
class BlockedEntry:
    date:     str
    ticker:   str
    strategy: str
    reason:   str


# ── Data loading ──────────────────────────────────────────────────────────

def load_chain(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "expiration"])
    df.columns = [c.strip().lower() for c in df.columns]
    df["call_put"] = df["call_put"].str.lower()
    # Sanity: required columns
    required = {"date", "act_symbol", "expiration", "strike", "call_put",
                "bid", "ask", "delta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"option_chain CSV missing required columns: {missing}")
    df["dte"] = (df["expiration"] - df["date"]).dt.days
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    return df


# ── Strike selection from REAL chain rows (no BS solving needed) ─────────

def select_spread(
    day_chain: pd.DataFrame,
    option_type: str,
    short_target_delta: float,
    long_target_delta: float,
    cfg: RealChainConfig,
) -> Optional[dict]:
    """
    From one day's chain rows (already filtered to one ticker, one
    expiration, one option_type), pick the short and long legs closest
    to their target deltas using REAL recorded delta values.
    """
    candidates = day_chain[
        (day_chain["bid"] > 0) | (day_chain["ask"] > 0)
    ].copy()
    if candidates.empty or len(candidates) < 2:
        return None

    candidates["delta_dist_short"] = (candidates["delta"] - short_target_delta).abs()
    short_row = candidates.loc[candidates["delta_dist_short"].idxmin()]

    # Long leg must be further OTM than short (lower |delta|), correct side
    if option_type == "put":
        long_pool = candidates[candidates["strike"] < short_row["strike"]]
    else:
        long_pool = candidates[candidates["strike"] > short_row["strike"]]
    if long_pool.empty:
        return None

    long_pool = long_pool.copy()
    long_pool["delta_dist_long"] = (long_pool["delta"] - long_target_delta).abs()
    long_row = long_pool.loc[long_pool["delta_dist_long"].idxmin()]

    return {"short": short_row, "long": long_row}


def evaluate_entry(
    chain_on_date: pd.DataFrame,
    ticker: str,
    sim_date,
    strategy: str,
    cfg: RealChainConfig,
) -> tuple[Optional[Trade], Optional[BlockedEntry]]:
    option_type = "put" if strategy == "short_put_spread" else "call"
    short_target = cfg.short_put_delta if option_type == "put" else cfg.short_call_delta
    long_target  = cfg.long_put_delta  if option_type == "put" else cfg.long_call_delta

    type_chain = chain_on_date[
        (chain_on_date["call_put"] == option_type) &
        (chain_on_date["dte"] >= cfg.min_dte) &
        (chain_on_date["dte"] <= cfg.max_dte)
    ]
    if type_chain.empty:
        return None, BlockedEntry(str(sim_date), ticker, strategy, "no_chain_data")

    # Try each expiration present, closest-to-target-DTE first (mirrors the
    # live bot's preference for nearer expirations within the DTE window)
    expirations = sorted(type_chain["expiration"].unique())
    for exp in expirations:
        exp_chain = type_chain[type_chain["expiration"] == exp]
        picked = select_spread(exp_chain, option_type, short_target, long_target, cfg)
        if picked is None:
            continue

        short_row, long_row = picked["short"], picked["long"]
        width = abs(short_row["strike"] - long_row["strike"])
        if width < cfg.min_spread_width or width > cfg.max_spread_width:
            continue

        # Real entry credit: sell short at bid, buy long at ask
        net_credit = short_row["bid"] - long_row["ask"]
        if net_credit < cfg.min_credit:
            continue

        # PoP approximation (documented): 1 - |short delta|
        pop_approx = 1.0 - abs(short_row["delta"])
        if pop_approx < cfg.min_pop:
            continue

        max_loss = (width - net_credit) * 100
        budget = cfg.equity * cfg.risk_pct
        if max_loss > budget:
            continue  # real chains don't let us "narrow" — strikes are discrete & fixed per day

        return Trade(
            entry_date=str(sim_date.date() if hasattr(sim_date, "date") else sim_date),
            ticker=ticker, strategy=strategy,
            short_strike=float(short_row["strike"]),
            long_strike=float(long_row["strike"]),
            width=width, net_credit=round(float(net_credit), 3),
            max_loss=round(max_loss, 2),
            short_delta=round(float(short_row["delta"]), 4),
            pop_approx=round(pop_approx, 4),
            dte_at_entry=int(short_row["dte"]),
            expiration=str(exp.date() if hasattr(exp, "date") else exp),
        ), None

    return None, BlockedEntry(str(sim_date), ticker, strategy, "no_qualifying_spread")


def simulate_exit(trade: Trade, ticker_chain: pd.DataFrame, cfg: RealChainConfig) -> Trade:
    """
    Walk forward through later dates in the dataset, looking up the SAME
    contract (matched on expiration + strike + call_put) to get its real
    bid/ask on each subsequent day. Cost to close a credit spread:
        cost = short_ask (buy back the short) - long_bid (sell the long)
    """
    option_type = "put" if "put" in trade.strategy else "call"
    entry_date  = pd.Timestamp(trade.entry_date)
    expiry_date = pd.Timestamp(trade.expiration)

    contract_rows = ticker_chain[
        (ticker_chain["call_put"] == option_type) &
        (ticker_chain["expiration"] == expiry_date) &
        (ticker_chain["date"] > entry_date)
    ].sort_values("date")

    short_rows = contract_rows[contract_rows["strike"] == trade.short_strike]
    long_rows  = contract_rows[contract_rows["strike"] == trade.long_strike]

    stop_cost   = trade.net_credit * cfg.stop_multiplier
    target_cost = trade.net_credit * (1 - cfg.profit_target_pct)
    half_dte    = trade.dte_at_entry // 2

    merged = pd.merge(
        short_rows[["date", "ask"]].rename(columns={"ask": "short_ask"}),
        long_rows[["date", "bid"]].rename(columns={"bid": "long_bid"}),
        on="date", how="inner",
    ).sort_values("date")

    for _, row in merged.iterrows():
        days_elapsed = (row["date"] - entry_date).days
        cost_to_close = row["short_ask"] - row["long_bid"]

        if cost_to_close >= stop_cost:
            trade.exit_date   = str(row["date"].date())
            trade.exit_reason = "stopped_out"
            trade.pnl         = round(-(cost_to_close - trade.net_credit) * 100, 2)
            trade.won         = False
            return trade

        if days_elapsed >= half_dte and cost_to_close <= target_cost:
            trade.exit_date   = str(row["date"].date())
            trade.exit_reason = "profit_target"
            trade.pnl         = round((trade.net_credit - cost_to_close) * 100, 2)
            trade.won         = True
            return trade

        if row["date"] >= expiry_date:
            won = cost_to_close <= 0.05
            trade.exit_date   = str(row["date"].date())
            trade.exit_reason = "expired_profit" if won else "expired_loss"
            trade.pnl         = round((trade.net_credit - cost_to_close) * 100, 2)
            trade.won         = won
            return trade

    # No further quotes found before expiration (data ends, or contract
    # stopped being recorded) — mark unresolved rather than guessing.
    trade.exit_date   = "no_data"
    trade.exit_reason = "open_at_end"
    return trade


# ── Main engine ───────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, cfg: RealChainConfig,
                 strategies: list[str]) -> tuple[list[Trade], list[BlockedEntry]]:
    trades, blocked = [], []
    tickers = sorted(df["act_symbol"].unique())

    for ticker in tickers:
        ticker_chain = df[df["act_symbol"] == ticker]
        dates = sorted(ticker_chain["date"].unique())

        for strategy in strategies:
            open_trade: Optional[Trade] = None
            for sim_date in dates:
                if open_trade is not None:
                    entry_d = pd.Timestamp(open_trade.entry_date)
                    if pd.Timestamp(sim_date) <= entry_d:
                        continue
                    continue  # exit handled separately once at entry time (see below)

                day_chain = ticker_chain[ticker_chain["date"] == sim_date]
                trade, block = evaluate_entry(day_chain, ticker, sim_date, strategy, cfg)
                if trade:
                    trade = simulate_exit(trade, ticker_chain, cfg)
                    trades.append(trade)
                elif block:
                    blocked.append(block)

    return trades, blocked


# ── Analysis ──────────────────────────────────────────────────────────────

def per_ticker_summary(trades: list[Trade]) -> pd.DataFrame:
    closed = [t for t in trades if t.exit_reason not in ("open_at_end",)]
    rows = []
    for ticker in sorted(set(t.ticker for t in closed)):
        sub = [t for t in closed if t.ticker == ticker]
        if not sub:
            continue
        pnls = np.array([t.pnl for t in sub])
        rows.append({
            "ticker":      ticker,
            "trades":      len(sub),
            "win_rate":    round(sum(t.won for t in sub) / len(sub), 3),
            "avg_credit":  round(np.mean([t.net_credit for t in sub]), 3),
            "avg_pnl":     round(pnls.mean(), 2),
            "total_pnl":   round(pnls.sum(), 2),
            "sharpe":      round(pnls.mean() / pnls.std(), 3) if pnls.std() > 0 else 0,
            "pct_stopped": round(sum(1 for t in sub if t.exit_reason == "stopped_out") / len(sub), 3),
            "pct_target":  round(sum(1 for t in sub if t.exit_reason == "profit_target") / len(sub), 3),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False) if rows else pd.DataFrame()


def pop_calibration(trades: list[Trade], cfg: RealChainConfig) -> pd.DataFrame:
    closed = [t for t in trades if t.exit_reason not in ("open_at_end",)]
    rows = []
    for threshold in cfg.pop_sweep_values:
        sub = [t for t in closed if t.pop_approx >= threshold]
        if not sub:
            continue
        won = sum(t.won for t in sub)
        rows.append({
            "pop_threshold":   threshold,
            "trades_taken":    len(sub),
            "actual_win_rate": round(won / len(sub), 3),
            "avg_pnl":         round(np.mean([t.pnl for t in sub]), 2),
        })
    return pd.DataFrame(rows)


def write_summary(out_dir: Path, trades: list[Trade], blocked: list[BlockedEntry],
                  cfg: RealChainConfig, source_file: str) -> None:
    closed = [t for t in trades if t.exit_reason not in ("open_at_end",)]

    ticker_df = per_ticker_summary(trades)
    pop_df    = pop_calibration(trades, cfg)

    pd.DataFrame([t.__dict__ for t in trades]).to_csv(out_dir / "trades.csv", index=False)
    ticker_df.to_csv(out_dir / "ticker_summary.csv", index=False)
    pop_df.to_csv(out_dir / "pop_calibration.csv", index=False)

    lines = [
        "=" * 70,
        "  REAL-CHAIN BACKTEST — option_chain (DoltHub) market-quoted data",
        "=" * 70,
        f"  Source file : {source_file}",
        f"  Tickers     : {', '.join(sorted(set(t.ticker for t in trades))) or 'none'}",
        f"  Credit      : short_bid - long_ask (real bid-ask spread, no synthetic slippage)",
        f"  PoP         : 1 - |short delta|  (practitioner approximation — no spot price",
        f"                in this dataset, so a true BS touch-probability isn't computable;",
        f"                this is the one non-literal number in the backtest, documented)",
        f"  Exit cost   : real quote of the SAME contract on later dates — converges to",
        f"                intrinsic value naturally by expiration, no formula needed",
        "=" * 70, "",
    ]

    if closed:
        pnls = [t.pnl for t in closed]
        lines += [
            "── OVERALL ──────────────────────────────────────────────",
            f"  Closed trades       : {len(closed)}",
            f"  Blocked attempts    : {len(blocked)}",
            f"  Win rate            : {sum(t.won for t in closed)/len(closed):.1%}",
            f"  Avg credit          : ${np.mean([t.net_credit for t in closed]):.3f}",
            f"  Avg P&L per trade   : ${np.mean(pnls):.2f}",
            f"  Total P&L           : ${sum(pnls):,.2f}",
            f"  Sharpe (per trade)  : {np.mean(pnls)/np.std(pnls):.2f}" if np.std(pnls) > 0 else "  Sharpe: N/A",
            f"  Stopped out         : {sum(1 for t in closed if t.exit_reason=='stopped_out')/len(closed):.1%}",
            f"  Profit target       : {sum(1 for t in closed if t.exit_reason=='profit_target')/len(closed):.1%}",
            f"  Expired             : {sum(1 for t in closed if 'expired' in t.exit_reason)/len(closed):.1%}",
            "",
        ]
    else:
        lines += ["── NO CLOSED TRADES — check ticker coverage in your CSV export ──", ""]

    if not pop_df.empty:
        lines += ["── POP CALIBRATION (real delta vs actual outcome) ────────",
                  f"  {'PoP floor':>10} {'Trades':>8} {'Actual WR':>10} {'Avg P&L':>9}",
                  "  " + "-" * 42]
        for _, row in pop_df.iterrows():
            lines.append(f"  {row['pop_threshold']:>9.0%} {row['trades_taken']:>8} "
                         f"{row['actual_win_rate']:>9.1%} ${row['avg_pnl']:>8.2f}")
        lines.append("")

    if not ticker_df.empty:
        lines += ["── PER-TICKER (sorted by Sharpe) ──────────────────────────",
                  f"  {'Ticker':>8} {'N':>5} {'WR':>6} {'AvgCr':>7} {'AvgPnL':>8} {'TotPnL':>9} {'Sharpe':>7}",
                  "  " + "-" * 56]
        for _, row in ticker_df.iterrows():
            lines.append(
                f"  {row['ticker']:>8} {row['trades']:>5} {row['win_rate']:>5.1%} "
                f"${row['avg_credit']:>5.3f} ${row['avg_pnl']:>7.2f} "
                f"${row['total_pnl']:>8.2f} {row['sharpe']:>7.2f}"
            )

    lines += ["", "=" * 70, f"  Outputs: {out_dir}/", "=" * 70]
    text = "\n".join(lines)
    print(text)
    (out_dir / "summary.txt").write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, help="option_chain CSV export")
    ap.add_argument("--strategies", nargs="+",
                    default=["short_put_spread", "short_call_spread"])
    ap.add_argument("--output", default="backtest_real_results")
    ap.add_argument("--equity", type=float, default=100_000)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.chain}...")
    df = load_chain(args.chain)
    print(f"  {len(df):,} rows, {df['act_symbol'].nunique()} tickers, "
          f"{df['date'].nunique()} unique dates "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    cfg = RealChainConfig(equity=args.equity)
    print(f"\nRunning backtest ({', '.join(args.strategies)})...")
    trades, blocked = run_backtest(df, cfg, args.strategies)
    closed = [t for t in trades if t.exit_reason != "open_at_end"]
    print(f"  {len(closed)} closed trades, {len(blocked)} blocked attempts\n")

    write_summary(out_dir, trades, blocked, cfg, args.chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
