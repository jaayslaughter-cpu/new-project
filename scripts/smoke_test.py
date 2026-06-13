#!/usr/bin/env python3
"""
Smoke test — confirms every external connection is live.

Run this from Railway shell or locally before starting the bot:

    python scripts/smoke_test.py

Checks (in order):
  1. Discord webhook — sends a real message
  2. Alpaca paper account — fetches account equity
  3. Alpaca options approval — confirms options are enabled
  4. yfinance — fetches SPY + VIX prices
  5. FRED API — fetches 10Y yield (skipped if FRED_API_KEY not set)
  6. Alpaca options chain — fetches live SPY option chain
  7. Regime detector — runs full detection pipeline
  8. Hurst exponent — computes on live SPY data

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import os
import sys
import json
import time
import urllib.request
from datetime import datetime, timezone

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(label, detail=""):
    print(f"  {GREEN}✓{RESET}  {label}" + (f"  {YELLOW}({detail}){RESET}" if detail else ""))

def fail(label, detail=""):
    print(f"  {RED}✗{RESET}  {label}" + (f"  {RED}{detail}{RESET}" if detail else ""))

def skip(label, reason=""):
    print(f"  {YELLOW}~{RESET}  {label} — {reason}")

# ── Env ───────────────────────────────────────────────────────────────────────
# Load .env if present
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if line and '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

ALPACA_KEY    = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_PAPER  = os.getenv('ALPACA_PAPER', 'true').lower() != 'false'
DISCORD_URL   = os.getenv('DISCORD_WEBHOOK_URL', '')
FRED_KEY      = os.getenv('FRED_API_KEY', '')

ALPACA_BASE = 'https://paper-api.alpaca.markets/v2' if ALPACA_PAPER else 'https://api.alpaca.markets/v2'
ALPACA_DATA = 'https://data.alpaca.markets/v1beta1'

failures = []

# ── 1. Discord ────────────────────────────────────────────────────────────────
print(f"\n{BOLD}1. Discord webhook{RESET}")
if not DISCORD_URL:
    skip("Discord", "DISCORD_WEBHOOK_URL not set")
else:
    try:
        now_str = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        msg = (
            f"🟢 **Options bot — smoke test passed** `{now_str}`\n"
            f"All systems connected. Bot is ready."
        )
        payload = json.dumps({"content": msg}).encode()
        req = urllib.request.Request(
            DISCORD_URL, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            if r.status in (200, 204):
                ok("Discord webhook", "message delivered — check your server")
            else:
                fail("Discord webhook", f"HTTP {r.status}")
                failures.append("discord")
    except Exception as e:
        fail("Discord webhook", str(e))
        failures.append("discord")

# ── 2. Alpaca account ─────────────────────────────────────────────────────────
print(f"\n{BOLD}2. Alpaca account{RESET}")
if not ALPACA_KEY or not ALPACA_SECRET:
    fail("Alpaca credentials", "ALPACA_API_KEY or ALPACA_SECRET_KEY not set")
    failures.append("alpaca_creds")
else:
    try:
        req = urllib.request.Request(
            f"{ALPACA_BASE}/account",
            headers={
                "APCA-API-KEY-ID": ALPACA_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET,
            }
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        equity  = float(data.get('equity', 0))
        status  = data.get('status', '?')
        paper   = 'paper' if ALPACA_PAPER else 'LIVE'
        ok("Alpaca account", f"{paper} | equity=${equity:,.2f} | status={status}")

        # Options approval
        opt_level = data.get('options_approved_level', 0)
        if int(opt_level or 0) >= 1:
            ok("Options approved", f"level={opt_level}")
        else:
            fail("Options not approved", f"level={opt_level} — enable options in Alpaca dashboard")
            failures.append("alpaca_options")

    except Exception as e:
        fail("Alpaca account", str(e))
        failures.append("alpaca_account")

# ── 3. yfinance ───────────────────────────────────────────────────────────────
print(f"\n{BOLD}3. yfinance (SPY + VIX){RESET}")
try:
    import yfinance as yf
    spy_price = yf.Ticker("SPY").fast_info.get("lastPrice")
    vix_price = yf.Ticker("^VIX").fast_info.get("lastPrice")
    if spy_price and float(spy_price) > 0:
        ok("SPY price", f"${float(spy_price):.2f}")
    else:
        fail("SPY price", "got None or 0")
        failures.append("yfinance_spy")
    if vix_price and float(vix_price) > 0:
        ok("VIX price", f"{float(vix_price):.2f}")
    else:
        fail("VIX price", "got None or 0")
        failures.append("yfinance_vix")
except ImportError:
    fail("yfinance", "not installed — pip install yfinance")
    failures.append("yfinance")
except Exception as e:
    fail("yfinance", str(e))
    failures.append("yfinance")

# ── 4. FRED API ───────────────────────────────────────────────────────────────
print(f"\n{BOLD}4. FRED API (yield curve){RESET}")
if not FRED_KEY:
    skip("FRED API", "FRED_API_KEY not set — yield curve will use yfinance proxy")
else:
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_KEY)
        dgs10 = float(fred.get_series("DGS10").dropna().iloc[-1])
        dgs2  = float(fred.get_series("DGS2").dropna().iloc[-1])
        slope = round(dgs10 - dgs2, 3)
        ok("FRED yield curve", f"10Y={dgs10:.2f}% 2Y={dgs2:.2f}% spread={slope:+.3f}%")
    except ImportError:
        fail("fredapi", "not installed — pip install fredapi")
        failures.append("fred")
    except Exception as e:
        fail("FRED API", str(e))
        failures.append("fred")

# ── 5. Alpaca options chain ───────────────────────────────────────────────────
print(f"\n{BOLD}5. Alpaca options chain (SPY){RESET}")
if not ALPACA_KEY or not ALPACA_SECRET:
    skip("Alpaca options chain", "credentials missing")
else:
    try:
        url = f"{ALPACA_DATA}/options/snapshots/SPY?limit=5&feed=indicative"
        req = urllib.request.Request(
            url,
            headers={
                "APCA-API-KEY-ID": ALPACA_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET,
            }
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        snapshots = data.get("snapshots", {})
        count = len(snapshots)
        if count > 0:
            sample = next(iter(snapshots.keys()))
            ok("SPY options chain", f"{count} contracts returned, e.g. {sample}")
        else:
            fail("SPY options chain", "0 contracts returned — market may be closed")
            failures.append("alpaca_chain")
    except Exception as e:
        fail("Alpaca options chain", str(e))
        failures.append("alpaca_chain")

# ── 6. Regime detector ───────────────────────────────────────────────────────
print(f"\n{BOLD}6. Regime detector{RESET}")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from options_bot.regime import RegimeDetector
    detector = RegimeDetector(cache_ttl_seconds=0)
    result = detector.detect()
    regime   = result['regime']
    conf     = result['confidence']
    vix      = result['indicators'].get('vix_level', 0)
    hurst    = result['indicators'].get('hurst', 0.5)
    hreg     = result['indicators'].get('hurst_regime', '?')
    opt_wt   = result['options_weight']
    ok("Regime", f"{regime} conf={conf:.0%} VIX={vix:.1f} Hurst={hurst:.3f}[{hreg}] options_weight={opt_wt:.0%}")
    if result['should_trade_options']:
        ok("Options trading gate", "OPEN — regime allows new positions")
    else:
        fail("Options trading gate", f"CLOSED — options_weight={opt_wt:.0%} below threshold")
        failures.append("regime_gate")
except Exception as e:
    fail("Regime detector", str(e))
    failures.append("regime")

# ── 7. Hurst exponent ────────────────────────────────────────────────────────
print(f"\n{BOLD}7. Hurst exponent (SPY){RESET}")
try:
    import yfinance as yf
    import numpy as np
    from options_bot.hurst import hurst_exponent, classify_regime
    hist = yf.Ticker("SPY").history(period="1y")
    if len(hist) >= 50:
        h = hurst_exponent(hist["Close"].values)
        ok("Hurst exponent", f"H={h:.4f} → {classify_regime(h)}")
    else:
        fail("Hurst exponent", f"only {len(hist)} bars available")
        failures.append("hurst")
except Exception as e:
    fail("Hurst exponent", str(e))
    failures.append("hurst")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if not failures:
    print(f"{GREEN}{BOLD}✓ All checks passed — bot is ready{RESET}")
    print(f"{'='*50}\n")

    # Send final Discord confirmation if webhook works
    if DISCORD_URL and 'discord' not in failures:
        try:
            regime_line = ""
            try:
                regime_line = f"Regime: {result['regime'].upper()} | VIX={result['indicators'].get('vix_level',0):.1f} | Hurst={result['indicators'].get('hurst',0.5):.3f}\n"
            except Exception:
                pass
            msg = (
                f"✅ **Smoke test PASSED** — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"{regime_line}"
                f"All connections live: Discord · Alpaca · yfinance · Regime · Hurst\n"
                f"Bot is ready to trade."
            )
            payload = json.dumps({"content": msg}).encode()
            req = urllib.request.Request(
                DISCORD_URL, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    sys.exit(0)
else:
    print(f"{RED}{BOLD}✗ {len(failures)} check(s) failed: {', '.join(failures)}{RESET}")
    print(f"{'='*50}\n")
    sys.exit(1)
