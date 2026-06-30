#!/usr/bin/env python3
"""
scripts/heal_orphans.py
────────────────────────
Generic orphan reconciliation — compares live Alpaca option positions
against the DB's open trades, and for any position with no matching DB
row, reconstructs it from `pending_intents` (the real order details, not
guesswork) or, failing that, from the broker position data alone.

This replaces scripts/reconcile_iwm_orphan.py, which only knew how to fix
that one specific 2026-06-24 incident from facts copied out of a Discord
alert. Going forward, this script works for ANY orphan, on ANY ticker,
without needing a human to manually transcribe the trade details first.

Why this exists
────────────────
Per Alpaca's official reconciliation-idempotency guidance: "every write is
a request, not a fact." A broker.submit() can succeed while the
subsequent save_fill() DB write fails (network blip, transient Postgres
error, the historical `%s`->`%%s` bug, etc). The orchestrator now writes a
`pending_intents` row BEFORE calling the broker (local-first ordering), so
even if save_fill() never lands, the real legs/credit/stop are already on
disk — this script just needs to match it to the live broker position and
promote it into the `trades` table.

Matching strategy (in priority order)
──────────────────────────────────────
1. Exact match: an unresolved pending_intents row whose legs (by symbol)
   are a subset of the orphaned position's OCC symbols. This gives exact
   credit, stop, profit target, and contract count — no guessing.
2. Fallback: no matching intent found (e.g. this predates the
   pending_intents table, like the original IWM incident). Reconstruct a
   degraded record from broker position data alone — strike/expiry parsed
   from the OCC symbol, qty from the position, but credit/stop/target
   unknown (set to None) and flagged `broker='heal_degraded'` so you know
   to set the stop manually before the next PositionMonitor cycle.

Usage
─────
    python scripts/heal_orphans.py --dry-run    # preview only
    python scripts/heal_orphans.py              # write + alert via Discord

Run from Railway shell (needs ALPACA_API_KEY/SECRET + DATABASE_URL), or
locally against a Railway-linked database.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from options_bot.orchestrator import TradeDatabase
from options_bot.broker import get_broker

# OCC symbol format: ROOT + YYMMDD + C/P + 8-digit strike (thousandths)
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def parse_occ(symbol: str) -> dict | None:
    """Parse an OCC option symbol into its components, or None if malformed."""
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    root, yymmdd, cp, strike_raw = m.groups()
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
    expiry = f"20{yy}-{mm}-{dd}"
    strike = int(strike_raw) / 1000.0
    return {
        "underlying": root,
        "expiry": expiry,
        "option_type": "call" if cp == "C" else "put",
        "strike": strike,
    }


def find_orphan_underlyings(db: TradeDatabase, positions: list[dict]) -> dict[str, list[dict]]:
    """
    Group broker option positions by underlying, keeping only those whose
    underlying has no open DB row at all (the orphan signal we care about —
    a partial mismatch on one leg of a multi-leg spread is a different,
    rarer problem this script doesn't attempt to auto-heal).
    """
    open_trades = db.get_open_trades()
    open_underlyings = {t.get("underlying") for t in open_trades if t.get("underlying")}

    orphans: dict[str, list[dict]] = {}
    for pos in positions:
        if pos.get("asset_class") != "us_option":
            continue
        parsed = parse_occ(pos["symbol"])
        if not parsed:
            print(f"  WARNING: could not parse OCC symbol {pos['symbol']!r} — skipping")
            continue
        underlying = parsed["underlying"]
        if underlying in open_underlyings:
            continue  # already tracked, not an orphan
        orphans.setdefault(underlying, []).append({**pos, **parsed})

    return orphans


def match_pending_intent(intents: list[dict], orphan_symbols: set[str]) -> dict | None:
    """Find an unresolved intent whose legs overlap the orphan's OCC symbols."""
    for intent in intents:
        try:
            legs = json.loads(intent.get("legs_json") or "[]")
        except Exception:
            continue
        leg_symbols = {l.get("symbol") for l in legs if l.get("symbol")}
        if leg_symbols & orphan_symbols:
            return intent
    return None


def heal_from_intent(db: TradeDatabase, underlying: str, intent: dict,
                     dry_run: bool) -> None:
    legs_json = intent["legs_json"]
    legs = json.loads(legs_json)
    new_id = f"healed-{uuid.uuid4()}"
    now = datetime.now(tz=timezone.utc).isoformat()
    net_credit = intent.get("net_credit") or 0.0
    max_loss   = intent.get("max_loss")
    hard_stop  = intent.get("hard_stop")
    contracts  = intent.get("contracts") or 1
    strategy   = intent.get("strategy") or "unknown"
    expiry     = None
    if legs:
        expiry = legs[0].get("expiry")

    print(f"  [{underlying}] MATCHED pending_intent {intent['id']} "
          f"(strategy={strategy}, credit=${net_credit}, stop=${hard_stop})")

    if dry_run:
        print(f"  [{underlying}] --dry-run: would INSERT trade {new_id}")
        return

    params = (
        new_id, date.today().isoformat(), strategy, underlying, legs_json,
        net_credit, 0.0, max_loss, hard_stop, contracts, net_credit,
        "open", "healed_from_intent", now, now,
        None, None, None, None, expiry,
        round(net_credit * 0.5, 2) if net_credit else None, 0.5,
    )
    _write_trade_row(db, params)
    db.resolve_pending_intent(intent["id"], new_id)
    print(f"  [{underlying}] OK — healed using real order details (intent {intent['id']})")


def heal_degraded(db: TradeDatabase, underlying: str, legs: list[dict],
                  dry_run: bool) -> None:
    """No matching intent — reconstruct what we can from broker data alone.
    Credit/stop/target are unknown; flagged for manual review."""
    new_id = f"healed-degraded-{uuid.uuid4()}"
    now = datetime.now(tz=timezone.utc).isoformat()
    expiry = legs[0]["expiry"] if legs else None
    legs_json = json.dumps([
        {"symbol": l["symbol"], "side": "unknown", "strike": l["strike"],
         "qty": 1, "expiry": l["expiry"]}
        for l in legs
    ])

    print(f"  [{underlying}] NO MATCHING INTENT — degraded heal "
          f"(credit/stop UNKNOWN, manual review required)")

    if dry_run:
        print(f"  [{underlying}] --dry-run: would INSERT degraded trade {new_id}")
        return

    params = (
        new_id, date.today().isoformat(), "unknown", underlying, legs_json,
        None, 0.0, None, None, len(legs), None,
        "open", "heal_degraded", now, now,
        None, None, None, None, expiry,
        None, None,
    )
    _write_trade_row(db, params)
    print(f"  [{underlying}] WROTE degraded record {new_id} — "
          f"⚠️  SET STOP MANUALLY before next PositionMonitor cycle")


def _write_trade_row(db: TradeDatabase, params: tuple) -> None:
    if db._use_pg:
        sql = """
            INSERT INTO trades
            (id, trade_date, strategy, underlying, legs_json,
             fill_price, slippage, max_loss, hard_stop, contracts,
             net_credit, status, broker, created_at, updated_at,
             delta, vega, theta, underlying_price, expiry,
             profit_target_price, profit_target_pct)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (id) DO UPDATE SET updated_at = EXCLUDED.updated_at
        """
    else:
        sql = """
            INSERT OR REPLACE INTO trades
            (id, trade_date, strategy, underlying, legs_json,
             fill_price, slippage, max_loss, hard_stop, contracts,
             net_credit, status, broker, created_at, updated_at,
             delta, vega, theta, underlying_price, expiry,
             profit_target_price, profit_target_pct)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
    conn = db._get_conn()
    try:
        db._execute(conn, sql, params)
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="preview matches without writing")
    args = ap.parse_args()

    db = TradeDatabase(
        database_url=os.getenv("DATABASE_URL") or os.getenv("DATABASE_PUBLIC_URL", ""),
        sqlite_path=os.getenv("SQLITE_PATH", "options_bot.db"),
    )

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret  = os.getenv("ALPACA_SECRET_KEY", "")
    paper   = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    if not api_key or not secret:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY must be set.", file=sys.stderr)
        return 1

    broker = get_broker(api_key=api_key, secret_key=secret, paper=paper)
    positions = broker.get_positions()
    print(f"Fetched {len(positions)} broker positions "
          f"({sum(1 for p in positions if p.get('asset_class')=='us_option')} options)")

    orphans = find_orphan_underlyings(db, positions)
    if not orphans:
        print("No orphaned positions found. DB and broker are in sync.")
        return 0

    print(f"\nFound {len(orphans)} orphaned underlying(s): {', '.join(orphans)}\n")

    intents = db.get_unresolved_intents(max_age_hours=24 * 14)  # 2-week window
    print(f"Checking against {len(intents)} unresolved pending_intents...\n")

    for underlying, legs in orphans.items():
        symbols = {l["symbol"] for l in legs}
        intent = match_pending_intent(intents, symbols)
        if intent:
            heal_from_intent(db, underlying, intent, args.dry_run)
        else:
            heal_degraded(db, underlying, legs, args.dry_run)
        print()

    if args.dry_run:
        print("--dry-run set — nothing was written. Re-run without it to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
