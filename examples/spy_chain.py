"""
Example: pull live SPY chain, enrich with Greeks, filter by delta.

Run on your local machine or Railway (not in Claude's sandbox — Yahoo
Finance is blocked from here).

    python examples/spy_chain.py
"""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)

import sys
sys.path.insert(0, "src")

from options_bot.market_data import YFinanceDataLoader
from options_bot.greeks import GreeksEnricher
from options_bot.exceptions import LiquidityFilterError, PipelineConnectionError

def main():
    # --- Step 1: Fetch raw chain ---
    loader = YFinanceDataLoader(ticker="SPY")
    expirations = loader.get_expirations()
    print(f"\nAvailable expirations: {expirations[:6]} ...\n")

    # Pick the 2nd expiration (~2 weeks out)
    target_expiry = expirations[1]
    print(f"Fetching chain for {target_expiry}...")

    try:
        raw_rows = loader.get_chain_filtered(
            expiry=target_expiry,
            min_open_interest=100,
            max_spread_pct=0.25,
            option_type=None,   # both calls and puts
        )
    except LiquidityFilterError as e:
        print(f"No liquid contracts found: {e}")
        return

    print(f"Liquid contracts: {len(raw_rows)}\n")

    # --- Step 2: Enrich with Greeks ---
    enricher = GreeksEnricher()   # fetches live Treasury rate
    enriched = enricher.enrich_chain_filtered(
        raw_rows,
        require_iv=True,
        min_abs_delta=0.10,
        max_abs_delta=0.40,
    )

    print(f"Contracts with IV in 10-40 delta range: {len(enriched)}\n")

    # --- Step 3: Print sample ---
    print(f"{'Symbol':<28} {'Type':<5} {'Strike':>7} {'DTE':>4} "
          f"{'Bid':>6} {'Ask':>6} {'IV':>7} {'Delta':>7} {'Theta':>8}")
    print("-" * 95)

    for row in sorted(enriched, key=lambda r: (r.option_type, r.strike))[:20]:
        iv_pct = f"{row.iv*100:.1f}%" if row.iv else "N/A"
        delta_s = f"{row.delta:.3f}" if row.delta else "N/A"
        theta_s = f"{row.theta:.4f}" if row.theta else "N/A"
        print(
            f"{row.symbol:<28} {row.option_type:<5} {row.strike:>7.1f} {row.dte:>4} "
            f"{row.bid or 0:>6.2f} {row.ask or 0:>6.2f} {iv_pct:>7} {delta_s:>7} {theta_s:>8}"
        )

    print(f"\nUnderlying SPY price: ${enriched[0].underlying_price:.2f}")


if __name__ == "__main__":
    main()
