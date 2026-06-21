"""
screen_etf_universe.py — Data-driven ETF candidate screener.

Answers "which ETFs should we add to the universe?" objectively, by running
each candidate through the SAME gates the live bot uses to accept a trade —
not a hand-picked guess. An ETF only earns a spot if it actually clears:

  1. IV-quality gate (options_bot.iv_quality.IVQualityGate) — the real gate,
     same TRADE/CAUTION/BLOCK logic the scanner uses. Needs 252d of history;
     fails open to CAUTION on thin data (mirrors production).

  2. Real option-chain liquidity probe — fetches the actual near-the-money
     options for a ~30-45 DTE expiry and measures:
       - count of strikes with open interest >= the strategy OI floor
       - median bid-ask spread % across near-the-money strikes
     against the bot's real per-strategy thresholds:
       ShortPutSpread : OI >= 100, spread <= 25%
       CSP            : OI >= 500, spread <= 15%   (strictest)
       ShortStrangle  : OI >= 200, spread <= 20%
       ShortCallSpread: OI >= 100, spread <= 25%

  3. Underlying price sanity — CSP's max loss is ~(strike * 100), so the 1%
     risk budget on $100k equity ($1,000/trade) can't fit a cash-secured put
     on a high-priced underlying. This flags which candidates are even
     viable for CSP at the current account size.

OUTPUT: a ranked table showing, per candidate, IV-quality verdict + which
strategies' liquidity bars it clears. An ETF that clears ShortPutSpread but
not CSP is still useful (just not for every strategy). One that clears
nothing should not be added.

Run with real internet access (yfinance) — your machine or a Railway job.
This sandbox can't reach yfinance.

USAGE:
    python screen_etf_universe.py                 # screens the default candidate list
    python screen_etf_universe.py SPY QQQ XLU SLV # screen specific tickers

NOTE: This is a DECISION-SUPPORT tool, not an auto-add. It tells you what
WOULD qualify. Adding to the live universe is still a manual edit to
OrchestratorConfig.tickers + __main__.py --tickers default (keep them in
sync — that divergence has bitten this project before). And per the project
gate: stay ETF-only until the 30-trade milestone regardless of what scores
well here.
"""

from __future__ import annotations

import sys
import statistics
from datetime import date, datetime, timedelta

# --- Candidate ETFs to evaluate (excludes the 15 already live, but you can
#     pass any tickers as args). Chosen for plausibly-liquid options + a
#     short-premium-friendly profile; the gates decide, not this list. ---
DEFAULT_CANDIDATES = [
    # broad / style gaps not currently covered
    "DIA",   # Dow 30 — large-cap, deep options
    "MDY",   # S&P midcap 400
    "VTI",   # total market
    "IWF",   # large growth
    "IWD",   # large value
    # sector ETFs with deep options not currently in the 15
    "XLU",   # utilities
    "XLY",   # consumer discretionary
    "XLP",   # consumer staples
    "XLC",   # communication services
    "XOP",   # oil & gas E&P (high IV)
    "XME",   # metals & mining (high IV)
    # commodity / rates vol
    "USO",   # oil
    "SLV",   # silver
    "UNG",   # natural gas (very high IV — gate will judge)
    "TLT",   # already live, included as a sanity benchmark
    "SPY",   # already live, included as a sanity benchmark
]

# Per-strategy liquidity thresholds — copied verbatim from strategy.py configs.
STRATEGY_THRESHOLDS = {
    "ShortPutSpread":  {"min_oi": 100, "max_spread_pct": 0.25},
    "CSP":             {"min_oi": 500, "max_spread_pct": 0.15},
    "ShortStrangle":   {"min_oi": 200, "max_spread_pct": 0.20},
    "ShortCallSpread": {"min_oi": 100, "max_spread_pct": 0.25},
}

# Risk budget context for CSP viability (1% of $100k = $1,000/trade).
ACCOUNT_EQUITY = 100_000.0
RISK_PCT = 0.01
RISK_BUDGET = ACCOUNT_EQUITY * RISK_PCT


def probe_chain_liquidity(ticker: str) -> dict:
    """
    Fetch a real ~30-45 DTE option chain for `ticker` and measure
    near-the-money OI and spread. Returns a dict of measured stats, or
    {"error": ...} if the chain can't be fetched.
    """
    import yfinance as yf

    t = yf.Ticker(ticker)
    spot_hist = t.history(period="5d")
    if spot_hist.empty:
        return {"error": "no price history"}
    spot = float(spot_hist["Close"].iloc[-1])

    expirations = t.options
    if not expirations:
        return {"error": "no listed options"}

    # Pick the expiry closest to 35 DTE (the bot's sweet spot)
    today = date.today()
    target = today + timedelta(days=35)
    def dte_of(exp_str):
        return abs((datetime.strptime(exp_str, "%Y-%m-%d").date() - target).days)
    best_exp = min(expirations, key=dte_of)
    actual_dte = (datetime.strptime(best_exp, "%Y-%m-%d").date() - today).days

    chain = t.option_chain(best_exp)
    rows = []
    for df, otype in ((chain.calls, "call"), (chain.puts, "put")):
        for _, r in df.iterrows():
            strike = float(r["strike"])
            # near-the-money only: within 10% of spot
            if abs(strike - spot) / spot > 0.10:
                continue
            bid = float(r["bid"]) if r["bid"] == r["bid"] else 0.0
            ask = float(r["ask"]) if r["ask"] == r["ask"] else 0.0
            oi = int(r["openInterest"]) if r["openInterest"] == r["openInterest"] else 0
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0.0
            spread_pct = (ask - bid) / mid if mid > 0 else None
            rows.append({"otype": otype, "strike": strike, "oi": oi, "spread_pct": spread_pct})

    if not rows:
        return {"error": "no near-the-money strikes"}

    spreads = [r["spread_pct"] for r in rows if r["spread_pct"] is not None]
    return {
        "spot": spot,
        "expiry": best_exp,
        "dte": actual_dte,
        "ntm_strike_count": len(rows),
        "median_spread_pct": statistics.median(spreads) if spreads else None,
        "rows": rows,
    }


def evaluate_strategy_fit(probe: dict, thresholds: dict) -> tuple[bool, str]:
    """Does this chain clear a given strategy's OI + spread bar?"""
    qualifying = [
        r for r in probe["rows"]
        if r["oi"] >= thresholds["min_oi"]
        and (r["spread_pct"] is not None and r["spread_pct"] <= thresholds["max_spread_pct"])
    ]
    n = len(qualifying)
    # Need at least a few qualifying NTM strikes to actually build a position
    ok = n >= 4
    return ok, f"{n} NTM strikes clear OI>={thresholds['min_oi']} & spread<={thresholds['max_spread_pct']:.0%}"


def main(candidates: list[str]):
    sys.path.insert(0, "src")
    from options_bot.iv_quality import IVQualityGate

    iv_gate = IVQualityGate(block_on_block=True, block_on_caution=False)

    print("=" * 100)
    print(f"ETF UNIVERSE SCREEN — {len(candidates)} candidates")
    print(f"Risk budget context: {RISK_PCT:.0%} of ${ACCOUNT_EQUITY:,.0f} = ${RISK_BUDGET:,.0f}/trade")
    print("=" * 100)

    results = []
    for ticker in candidates:
        print(f"\n--- {ticker} ---")

        # 1. IV-quality gate (the real one)
        try:
            iv_ok, report = iv_gate.check(ticker)
            iv_verdict = report.recommendation if report else "NO_DATA (fail-open)"
            iv_score = report.quality_score if report else None
        except Exception as e:
            iv_ok, iv_verdict, iv_score = True, f"ERROR: {str(e)[:40]}", None
        print(f"  IV quality: {iv_verdict}" + (f" (score={iv_score})" if iv_score is not None else ""))

        # 2. Chain liquidity probe
        try:
            probe = probe_chain_liquidity(ticker)
        except Exception as e:
            probe = {"error": f"{type(e).__name__}: {str(e)[:50]}"}

        if "error" in probe:
            print(f"  Chain: UNAVAILABLE — {probe['error']}")
            results.append({"ticker": ticker, "iv": iv_verdict, "strategies": [], "note": probe["error"]})
            continue

        print(f"  Chain: spot=${probe['spot']:.2f} expiry={probe['expiry']} ({probe['dte']}DTE) "
              f"{probe['ntm_strike_count']} NTM strikes, "
              f"median spread={probe['median_spread_pct']*100:.1f}%"
              if probe['median_spread_pct'] is not None else "  Chain: spreads unavailable")

        # 3. Per-strategy fit
        passed_strategies = []
        for strat_name, thr in STRATEGY_THRESHOLDS.items():
            ok, detail = evaluate_strategy_fit(probe, thr)
            mark = "✓" if ok else "✗"
            print(f"    [{mark}] {strat_name:16s}: {detail}")
            if ok:
                passed_strategies.append(strat_name)

        # CSP price viability note
        if probe["spot"] * 100 > RISK_BUDGET and "CSP" in passed_strategies:
            print(f"    ⚠️  CSP liquidity OK but spot ${probe['spot']:.2f} × 100 = "
                  f"${probe['spot']*100:,.0f} > ${RISK_BUDGET:,.0f} budget — "
                  f"CSP won't size at current equity")

        results.append({
            "ticker": ticker, "iv": iv_verdict,
            "strategies": passed_strategies,
            "spot": probe["spot"],
        })

    # Summary ranking
    print("\n" + "=" * 100)
    print("SUMMARY — candidates that clear at least one strategy's bar (excl. IV BLOCK)")
    print("=" * 100)
    keepers = [r for r in results if r.get("strategies") and r["iv"] != "BLOCK"]
    keepers.sort(key=lambda r: len(r["strategies"]), reverse=True)
    if not keepers:
        print("  None cleared the bar — keep the current universe.")
    for r in keepers:
        print(f"  {r['ticker']:5s}  IV={r['iv']:20s}  clears: {', '.join(r['strategies'])}")

    print("\nReminder: per the project gate, stay ETF-only until 30+ closed trades validate edge,")
    print("and keep OrchestratorConfig.tickers in sync with __main__.py --tickers if you add any.")


if __name__ == "__main__":
    args = sys.argv[1:]
    candidates = args if args else DEFAULT_CANDIDATES
    main(candidates)
