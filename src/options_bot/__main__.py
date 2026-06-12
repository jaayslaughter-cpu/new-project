"""
Entry point for the options bot.

Usage:
    # Paper trading (default — safe)
    python -m options_bot

    # With custom config
    python -m options_bot --tickers SPY QQQ --strategy short_put_spread

    # Single manual scan (no scheduler)
    python -m options_bot --once

    # Live trading (requires ALPACA_PAPER=false)
    ALPACA_PAPER=false python -m options_bot

Environment variables (all optional except API keys):
    ALPACA_API_KEY          — required
    ALPACA_SECRET_KEY       — required
    ALPACA_PAPER            — "true" (default) or "false"
    DISCORD_WEBHOOK_URL     — Discord webhook for trade dispatch
    DATABASE_URL            — PostgreSQL URL (Railway), falls back to SQLite

Example .env file:
    ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ALPACA_PAPER=true
    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
"""

import argparse
import logging
import os
import sys

# Configure structured JSON logging before any other imports
# This ensures Railway captures all logs correctly
from options_bot.logging_config import setup_logging
setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Options Bot — automated options trading via Alpaca",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["SPY", "QQQ", "IWM"],
        help="Tickers to scan",
    )
    parser.add_argument(
        "--strategy", default="short_put_spread",
        choices=["csp", "short_put_spread", "short_strangle"],
        help="Options strategy",
    )
    parser.add_argument(
        "--equity", type=float, default=None,
        help="Override starting equity (default: fetch from Alpaca)",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=0.02,
        help="Risk budget per trade as fraction of equity (e.g. 0.02 = 2%%)",
    )
    parser.add_argument(
        "--max-trades", type=int, default=5,
        help="Max trades per day",
    )
    parser.add_argument(
        "--max-positions", type=int, default=5,
        help="Max concurrent open positions",
    )
    parser.add_argument(
        "--scan-time", default="09:45",
        help="Scan time in PT (HH:MM)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single scan then exit (no scheduler)",
    )
    parser.add_argument(
        "--paper", action="store_true", default=True,
        help="Paper trading mode (default)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live trading mode (overrides --paper)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use PaperBroker stub (no API calls at all)",
    )

    args = parser.parse_args()

    # Validate API keys exist before doing anything
    if not args.dry_run:
        if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
            print(
                "ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.\n"
                "Add them to your environment or a .env file.\n"
                "For dry-run testing without API keys, use --dry-run.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from options_bot.orchestrator import Orchestrator, OrchestratorConfig
    from options_bot.risk import RiskConfig

    # Parse scan time
    try:
        scan_h, scan_m = map(int, args.scan_time.split(":"))
    except ValueError:
        print(f"ERROR: Invalid --scan-time '{args.scan_time}'. Use HH:MM format.")
        sys.exit(1)

    paper = not args.live

    config = OrchestratorConfig(
        tickers=args.tickers,
        strategy_name=args.strategy,
        risk_config=RiskConfig(
            risk_budget_pct=args.risk_pct,
            max_trades_per_day=args.max_trades,
        ),
        scan_hour=scan_h,
        scan_minute=scan_m,
        max_positions_total=args.max_positions,
        paper=paper,
    )

    if args.dry_run:
        # Swap in paper stub — no network calls
        from options_bot.broker import PaperBroker
        from options_bot.orchestrator import (
            TradingPipeline, PositionMonitor, TradeDatabase, SessionState,
        )
        from options_bot.risk import RiskManager

        print("DRY RUN mode — using PaperBroker stub (no API calls)")
        broker = PaperBroker(starting_equity=args.equity or 100_000)
        rm = RiskManager(equity=args.equity or 100_000, config=config.risk_config)
        db = TradeDatabase(sqlite_path=config.sqlite_path)
        state = SessionState()
        pipeline = TradingPipeline(config, rm, broker, db, state)

        print(f"Scanning: {config.tickers}")
        for ticker in config.tickers:
            result = pipeline.run_for_ticker(ticker)
            if result:
                print(f"  {ticker}: order {result.order_id} fill={result.fill_price:.2f}")
            else:
                print(f"  {ticker}: no trade")
        return

    mode_label = "LIVE" if not paper else "PAPER"
    print(f"Starting Options Bot [{mode_label}]")
    print(f"  Tickers:  {config.tickers}")
    print(f"  Strategy: {config.strategy_name}")
    print(f"  Risk:     {config.risk_config.risk_budget_pct:.1%} per trade")
    print(f"  Scan:     {args.scan_time} PT daily")

    orch = Orchestrator(config)

    if args.once:
        print("Running single scan...")
        filled = orch.run_scan()
        print(f"Scan complete: {len(filled)} trade(s) entered")
        for f in filled:
            print(f"  {f.approved_order.underlying} {f.approved_order.strategy_name} "
                  f"— {f.order_id}")
    else:
        orch.run()   # blocks


if __name__ == "__main__":
    main()
