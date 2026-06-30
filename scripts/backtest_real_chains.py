#!/usr/bin/env python3
"""
scripts/backtest_real_chains.py
─────────────────────────────────
Backtests ShortPutSpread / ShortCallSpread against REAL, market-quoted
options chains — no Black-Scholes reconstruction, no synthetic IV. Every
credit, every delta, every exit cost comes directly from actual recorded
bid/ask/greeks. Optionally ingests real underlying OHLCV data to compute
true Black-Scholes PoP and build a real SPY buy-and-hold benchmark into
the same report.

Data sources (both from DoltHub, same publisher, same daily cadence)
──────────────────────────────────────────────────────────────────────
post-no-preference/options  (option_chain table)
    date, act_symbol, expiration, strike, call_put, bid, ask, vol,
    delta, gamma, theta, vega, rho
    `vol` here is the contract's own recorded implied volatility —
    no IV solving needed.

post-no-preference/stocks   (ohlcv table) — OPTIONAL but recommended
    date, act_symbol, open, high, low, close, volume
    Used for: (1) real Black-Scholes PoP via the bot's own pop_spread()
    function, now that we have real spot price + real IV + real DTE —
    no more delta-shorthand approximation. (2) A genuine SPY buy-and-hold
    benchmark computed from real closing prices over the exact same
    period as the options backtest, not a hardcoded yearly-return table.

This script reads CSV exports of those tables — it does not call the
DoltHub API directly (no network access required to run it).

Getting the exports
────────────────────
Run on DoltHub's SQL workbench and download each result as CSV:

    -- option_chain (post-no-preference/options)
    SELECT * FROM option_chain
    WHERE act_symbol IN ('SPY','QQQ','IWM','TLT','XLF','XLK','XLE','XLV',
                          'XLI','GLD','EEM','HYG','SMH','VXX','XBI')
      AND date >= '2020-01-01'
      AND DATEDIFF(expiration, date) BETWEEN 7 AND 65
      AND ABS(delta) BETWEEN 0.02 AND 0.40;

    -- ohlcv (post-no-preference/stocks) — same ticker list, no DTE/delta
    -- filters needed since it's one row per symbol per day (small table)
    SELECT * FROM ohlcv
    WHERE act_symbol IN ('SPY','QQQ','IWM','TLT','XLF','XLK','XLE','XLV',
                          'XLI','GLD','EEM','HYG','SMH','VXX','XBI')
      AND date >= '2020-01-01';

The DTE and delta filters on option_chain match the bot's actual trading
range (DTE 14–60, |delta| up to 0.40 covers every strategy's target
deltas with margin) and keep the export to only rows the strategies could
ever select.

Why real quotes change everything vs the synthetic backtest
─────────────────────────────────────────────────────────────
- Entry credit = short_bid − long_ask (you receive the bid when selling,
  pay the ask when buying) — the bid-ask spread is real, built-in slippage,
  not assumed away.
- IV is the contract's own recorded `vol` — no solving required.
- Delta is the recorded market delta.
- PoP, when OHLCV is supplied, comes from the bot's own pop_spread()
  Black-Scholes formula fed real spot/IV/DTE — the same math the live bot
  uses for real trades. Without OHLCV, falls back to the 1-|delta|
  practitioner shorthand (clearly labeled either way).
- Exit cost-to-close is read from the SAME contract's real quote on a
  later date — by expiration this naturally converges to intrinsic value.
- The SPY benchmark, when OHLCV is supplied, is computed from SPY's own
  real closing prices over the identical date range as the options
  backtest — a genuine apples-to-apples comparison, not a separately
  sourced yearly-return table.

Usage
─────
    python scripts/backtest_real_chains.py \\
        --chain option_chain_export.csv \\
        --ohlcv ohlcv_export.csv \\
        --output backtest_real_results/

`--ohlcv` is optional — the script runs fine without it, just with the
delta-shorthand PoP and no SPY benchmark section.

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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from options_bot.greeks import pop_spread

TRADING_DAYS = 252
RISK_FREE_RATE = 0.05   # flat approximation; not recorded in either dataset

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
    pop_is_real:    bool  = False     # True = real BS PoP via spot+IV; False = delta shorthand
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


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "act_symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ohlcv CSV missing required columns: {missing}")
    return df[["date", "act_symbol", "close"]].copy()


def spy_buy_and_hold(ohlcv: pd.DataFrame, equity: float) -> Optional[dict]:
    """Real SPY buy-and-hold return over the exact date range present in
    the ohlcv export — same period as the options backtest, not a
    separately sourced yearly-return table."""
    spy = ohlcv[ohlcv["act_symbol"] == "SPY"].sort_values("date")
    if spy.empty:
        return None
    start_price = float(spy.iloc[0]["close"])
    end_price   = float(spy.iloc[-1]["close"])
    shares = equity / start_price
    end_value = shares * end_price
    total_return = (end_value / equity) - 1.0
    n_days = (spy.iloc[-1]["date"] - spy.iloc[0]["date"]).days
    years = max(n_days / 365.25, 0.01)
    cagr = (end_value / equity) ** (1 / years) - 1.0
    return {
        "start_date": str(spy.iloc[0]["date"].date()),
        "end_date":   str(spy.iloc[-1]["date"].date()),
        "start_price": round(start_price, 2),
        "end_price":   round(end_price, 2),
        "start_value": round(equity, 2),
        "end_value":   round(end_value, 2),
        "total_return": round(total_return, 4),
        "cagr":         round(cagr, 4),
    }


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
    spot_price: Optional[float] = None,
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

        # PoP: real Black-Scholes via the bot's own pop_spread() when a real
        # spot price is available (from ohlcv) — uses the contract's own
        # recorded `vol` (real IV, no solving). Falls back to the
        # 1-|delta| practitioner shorthand when no spot price is supplied.
        pop_is_real = False
        if spot_price is not None and "vol" in short_row and short_row["vol"] > 0:
            try:
                spread_type = "bull_put" if option_type == "put" else "bear_call"
                pop_result = pop_spread(
                    spread_type=spread_type,
                    short_strike=float(short_row["strike"]),
                    long_strike=float(long_row["strike"]),
                    net_credit=float(net_credit),
                    spot=float(spot_price),
                    sigma=float(short_row["vol"]),
                    rate=RISK_FREE_RATE,
                    days_to_expiry=int(short_row["dte"]),
                )
                pop_approx = float(pop_result["pop"]) if isinstance(pop_result, dict) else float(pop_result)
                pop_is_real = True
            except Exception:
                pop_approx = 1.0 - abs(short_row["delta"])
        else:
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
            pop_is_real=pop_is_real,
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
                 strategies: list[str],
                 ohlcv: Optional[pd.DataFrame] = None,
                 ) -> tuple[list[Trade], list[BlockedEntry]]:
    trades, blocked = [], []
    tickers = sorted(df["act_symbol"].unique())

    # Build a fast (ticker, date) -> close lookup if OHLCV was supplied
    spot_lookup: dict = {}
    if ohlcv is not None:
        for _, row in ohlcv.iterrows():
            spot_lookup[(row["act_symbol"], pd.Timestamp(row["date"]))] = float(row["close"])

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
                spot = spot_lookup.get((ticker, pd.Timestamp(sim_date)))
                trade, block = evaluate_entry(day_chain, ticker, sim_date, strategy, cfg, spot)
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
                  cfg: RealChainConfig, source_file: str,
                  spy_benchmark: Optional[dict] = None) -> None:
    closed = [t for t in trades if t.exit_reason not in ("open_at_end",)]

    ticker_df = per_ticker_summary(trades)
    pop_df    = pop_calibration(trades, cfg)

    pd.DataFrame([t.__dict__ for t in trades]).to_csv(out_dir / "trades.csv", index=False)
    ticker_df.to_csv(out_dir / "ticker_summary.csv", index=False)
    pop_df.to_csv(out_dir / "pop_calibration.csv", index=False)

    real_pop_pct = (sum(1 for t in closed if t.pop_is_real) / len(closed)) if closed else 0.0
    pop_label = (
        f"  PoP         : Black-Scholes via pop_spread() using real spot price + real IV\n"
        f"                ({real_pop_pct:.0%} of trades) where OHLCV data was available;\n"
        f"                falls back to 1-|short delta| shorthand otherwise."
        if spy_benchmark is not None else
        f"  PoP         : 1 - |short delta|  (practitioner approximation — no spot price\n"
        f"                supplied via --ohlcv, so a true BS PoP isn't computable here)"
    )

    lines = [
        "=" * 70,
        "  REAL-CHAIN BACKTEST — option_chain (DoltHub) market-quoted data",
        "=" * 70,
        f"  Source file : {source_file}",
        f"  Tickers     : {', '.join(sorted(set(t.ticker for t in trades))) or 'none'}",
        f"  Credit      : short_bid - long_ask (real bid-ask spread, no synthetic slippage)",
        pop_label,
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

        if spy_benchmark is not None:
            strategy_return_on_equity = sum(pnls) / cfg.equity
            lines += [
                "── STRATEGY vs SPY BUY-AND-HOLD (same real data, same period) ──",
                f"  Period              : {spy_benchmark['start_date']} -> {spy_benchmark['end_date']}",
                f"  Starting capital    : ${cfg.equity:,.0f}",
                "",
                f"  SPY buy-and-hold:",
                f"    SPY price         : ${spy_benchmark['start_price']} -> ${spy_benchmark['end_price']}",
                f"    Ending value      : ${spy_benchmark['end_value']:,.2f}",
                f"    Total return      : {spy_benchmark['total_return']:+.1%}",
                f"    CAGR              : {spy_benchmark['cagr']:+.1%}",
                "",
                f"  This strategy (options P&L only, capital otherwise idle/uninvested):",
                f"    Total P&L         : ${sum(pnls):,.2f}",
                f"    Return on capital : {strategy_return_on_equity:+.1%}",
                "",
                "  NOTE: not a true apples-to-apples comparison — SPY buy-and-hold puts",
                "  100% of capital at market risk the whole period; this strategy risks",
                "  only ~1% of equity per trade and the remainder earns nothing in this",
                "  simulation (no cash-management/margin modeling). The honest reading:",
                "  this shows the strategy's raw edge in isolation, not a fund-vs-fund",
                "  comparison. A fair comparison would add T-bill yield on idle cash.",
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
    ap.add_argument("--ohlcv", default=None,
                    help="optional ohlcv CSV export (post-no-preference/stocks) — "
                         "enables real Black-Scholes PoP and a real SPY buy-and-hold benchmark")
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

    ohlcv = None
    spy_benchmark = None
    if args.ohlcv:
        print(f"\nLoading {args.ohlcv}...")
        ohlcv = load_ohlcv(args.ohlcv)
        print(f"  {len(ohlcv):,} rows, {ohlcv['act_symbol'].nunique()} tickers, "
              f"{ohlcv['date'].nunique()} unique dates")
        spy_benchmark = spy_buy_and_hold(ohlcv, args.equity)
        if spy_benchmark is None:
            print("  WARNING: no SPY rows found in ohlcv export — "
                  "no benchmark section will be included. Add SPY to your ohlcv query.")

    cfg = RealChainConfig(equity=args.equity)
    print(f"\nRunning backtest ({', '.join(args.strategies)})...")
    trades, blocked = run_backtest(df, cfg, args.strategies, ohlcv)
    closed = [t for t in trades if t.exit_reason != "open_at_end"]
    print(f"  {len(closed)} closed trades, {len(blocked)} blocked attempts\n")

    write_summary(out_dir, trades, blocked, cfg, args.chain, spy_benchmark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
