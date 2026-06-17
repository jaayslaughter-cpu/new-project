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

# Start health server immediately — before Orchestrator init, API key checks,
# or any network calls — so Railway sees /health respond 200 within 2 seconds.
# The server stays at status="starting" until Orchestrator.__init__ completes,
# then transitions to "ready". If init fails, status stays "starting" (503).
import threading, os
from http.server import BaseHTTPRequestHandler, HTTPServer as _HS
import json as _json

_boot_status = {"status": "starting", "message": "Bot initialising..."}

class _BH(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] in ("/health", "/ready", "/"):
            b = _json.dumps(_boot_status).encode()
            # Return 200 during startup so Railway doesn't kill the container
            # while the bot is still initialising (can take 20-30s on cold start).
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *a): pass

def _start_boot_health(port: int = 8080) -> None:
    try:
        s = _HS(("0.0.0.0", port), _BH)
        threading.Thread(target=s.serve_forever, daemon=True).start()
    except OSError:
        pass  # port already in use (Orchestrator health server took it)

_start_boot_health(int(os.getenv("HEALTH_PORT", "8080")))


def _live_preflight(config) -> None:
    """
    Interactive pre-flight checklist for live (real money) trading.

    Runs once at process startup when ALPACA_PAPER=false or --live is passed.
    Prints a mandatory checklist, requires the operator to confirm each item,
    then demands they type a specific phrase before the bot is allowed to start.

    If any step fails or the user does not confirm, the process exits immediately.
    This is intentionally verbose — a moment of friction that prevents accidental
    live launches.
    """
    import sys

    CONFIRM_PHRASE = "I understand this is real money"

    checks = [
        ("ALPACA_PAPER env var is set to 'false'",
         lambda: os.getenv("ALPACA_PAPER", "true").lower() == "false"),
        ("ALPACA_API_KEY is set",
         lambda: bool(os.getenv("ALPACA_API_KEY", ""))),
        ("ALPACA_SECRET_KEY is set",
         lambda: bool(os.getenv("ALPACA_SECRET_KEY", ""))),
        ("Risk budget is 2% or less per trade",
         lambda: config.risk_config.risk_budget_pct <= 0.02),
        ("Max daily loss is 5% or less",
         lambda: config.risk_config.max_daily_loss_pct <= 0.05),
        ("Max contracts is 10 or less",
         lambda: config.risk_config.max_contracts <= 10),
        ("Discord webhook is set (for trade alerts)",
         lambda: bool(config.discord_webhook_url)),
    ]

    print()
    print("=" * 60)
    print("  LIVE TRADING PRE-FLIGHT CHECKLIST")
    print("  Real money will be at risk. Read every item carefully.")
    print("=" * 60)
    print()

    all_pass = True
    for label, check_fn in checks:
        try:
            passed = check_fn()
        except Exception:
            passed = False
        status = "OK " if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed:
            all_pass = False

    print()

    if not all_pass:
        print("ERROR: One or more pre-flight checks failed.")
        print("Fix the issues above before switching to live trading.")
        print()
        sys.exit(1)

    print("All checks passed.")
    print()
    print(f'Type exactly: {CONFIRM_PHRASE!r}')
    print("(or press Ctrl-C to abort)")
    print()

    try:
        response = input("Confirmation: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        print("Aborted.")
        sys.exit(0)

    if response != CONFIRM_PHRASE:
        print()
        print(f"Incorrect confirmation. Expected: {CONFIRM_PHRASE!r}")
        print("Bot NOT started.")
        sys.exit(1)

    print()
    print("Confirmation accepted. Starting LIVE trading bot...")
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Options Bot — automated options trading via Alpaca",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=[
            "SPY", "QQQ", "IWM", "DIA", "MDY",
            "XLF", "XLK", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLB", "XLRE",
            "GLD", "TLT", "EEM", "HYG", "SMH",
        ],
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
        "--scan-time", default="06:45",
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
        "--no-zero-dte", action="store_true", default=False,
        dest="no_zero_dte",
        help="Disable 0DTE intraday module (enabled by default)",
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

    # Load .env FIRST before any credential checks
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Validate API keys exist
    if not args.dry_run:
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            print(
                "ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.\n"
                "Add them to your environment or a .env file.\n"
                "For dry-run testing without API keys, use --dry-run.",
                file=sys.stderr,
            )
            sys.exit(1)

    paper = not args.live

    # ── System validation — boot-sequence firewall ───────────────────────────
    # Runs all 4 checks (env, broker, market data, internal modules) before
    # constructing anything. Raises SystemValidationError and exits on failure.
    # Skipped in --dry-run mode (PaperBroker, no network calls).
    from options_bot.system_validator import run_system_validation, SystemValidationError
    try:
        run_system_validation(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            discord_webhook=os.getenv("DISCORD_WEBHOOK_URL"),
        )
    except SystemValidationError as sv_exc:
        logging.getLogger(__name__).critical("Boot aborted — system validation failed: %s", sv_exc)
        # Send Discord alert if webhook is available
        _webhook = os.getenv("DISCORD_WEBHOOK_URL")
        if _webhook:
            try:
                import urllib.request, json as _json
                _url = _webhook.replace("discordapp.com", "discord.com")
                _payload = _json.dumps({
                    "content": f"🔴 **Options Bot failed to start**\n```{sv_exc}```"
                }).encode()
                _req = urllib.request.Request(
                    _url, data=_payload,
                    headers={"Content-Type": "application/json"}, method="POST"
                )
                urllib.request.urlopen(_req, timeout=5)
            except Exception:
                pass
        sys.exit(1)
    # ─────────────────────────────────────────────────────────────────────────

    from options_bot.orchestrator import Orchestrator, OrchestratorConfig
    from options_bot.risk import RiskConfig

    # Parse scan time
    try:
        scan_h, scan_m = map(int, args.scan_time.split(":"))
    except ValueError:
        print(f"ERROR: Invalid --scan-time '{args.scan_time}'. Use HH:MM format.")
        sys.exit(1)

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
        zero_dte_enabled=not args.no_zero_dte,
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
    scan_pt = args.scan_time
    try:
        _h, _m = map(int, scan_pt.split(":"))
        _et_h  = (_h + 3) % 24
        scan_et = f"{_et_h:02d}:{_m:02d}"
    except Exception:
        scan_et = "?"
    print(f"  Scan:     {scan_pt} PT = {scan_et} ET daily")

    # ── Live trading pre-flight ───────────────────────────────────────────────
    # When ALPACA_PAPER=false (or --live flag), run an interactive checklist
    # before constructing the Orchestrator. Aborts the process if any check
    # fails or the user does not type the confirmation phrase.
    if not paper:
        _live_preflight(config)
    # ─────────────────────────────────────────────────────────────────────────

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
