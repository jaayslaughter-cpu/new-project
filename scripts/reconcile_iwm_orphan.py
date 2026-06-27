#!/usr/bin/env python3
"""
One-off reconciliation: insert the orphaned IWM short call spread into the
trades DB so PositionMonitor begins managing it (stop + profit target).

Background
----------
On 2026-06-24 06:46 PT the bot dispatched and filled a real trade:

    SHORTCALLSPREAD — IWM
    SELL TO OPEN 320C / BUY TO OPEN 330C
    Credit $0.84  Stop $1.68  Max loss $916  Contracts 1  Expiry 2026-07-31
    Order ID 4df16876-df8d-41d9-9704-a4cb3181ced1
    short=320C delta=0.135  long=330C delta=0.056  width=10.0

The fill never persisted to the DB because of the save_fill `%s`->`%%s`
double-escape bug (fixed since). The position is live at Alpaca but invisible
to PositionMonitor, so its stop/target are not being enforced. This script
writes the record using the live TradeDatabase so the row matches exactly
what save_fill would have written (same 22 columns, same `?`->`%s` path),
and the monitor adopts it on the next 15-min cycle.

Safety
------
* Idempotent: refuses to insert if an open IWM short-call-spread already
  exists (matched on underlying + strategy + the two leg symbols).
* Reads it back after commit to confirm the write landed.
* Run ONCE, inside the Railway environment (needs DATABASE_URL):
      python scripts/reconcile_iwm_orphan.py
  Add --dry-run to preview without writing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import date, datetime, timezone

from options_bot.orchestrator import TradeDatabase

# --- Known facts about the orphaned trade (from the 06-24 Discord alert) ----
UNDERLYING   = "IWM"
STRATEGY     = "short_call_spread"          # canonical lowercase form used by
                                            # scan routing, adaptive tuner, and
                                            # walk_forward WHERE strategy=? queries
TRADE_DATE   = "2026-06-24"
EXPIRY       = "2026-07-31"
SHORT_SYMBOL = "IWM260731C00320000"         # SELL TO OPEN 320C
LONG_SYMBOL  = "IWM260731C00330000"         # BUY  TO OPEN 330C
NET_CREDIT   = 0.84
HARD_STOP    = 1.68                          # 2x net credit (bot convention)
PROFIT_TGT   = round(NET_CREDIT * 0.50, 2)   # 0.42 — 50% of credit
PROFIT_PCT   = 0.50
MAX_LOSS     = 916.0                         # (10 wide - 0.84) * 100
CONTRACTS    = 1
SHORT_DELTA  = 0.135
LONG_DELTA   = 0.056
ORIG_ORDER   = "4df16876-df8d-41d9-9704-a4cb3181ced1"

LEGS = [
    {"symbol": SHORT_SYMBOL, "side": "sell_to_open", "strike": 320.0,
     "qty": 1, "expiry": EXPIRY},
    {"symbol": LONG_SYMBOL,  "side": "buy_to_open",  "strike": 330.0,
     "qty": 1, "expiry": EXPIRY},
]


def _already_present(db: TradeDatabase) -> bool:
    """True if an open IWM short-call-spread with these legs is already stored."""
    for t in db.get_open_trades():
        if t.get("underlying") != UNDERLYING:
            continue
        try:
            legs = json.loads(t.get("legs_json") or "[]")
        except Exception:
            legs = []
        syms = {l.get("symbol") for l in legs}
        if SHORT_SYMBOL in syms and LONG_SYMBOL in syms:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="preview without writing")
    args = ap.parse_args()

    db = TradeDatabase(
        database_url=os.getenv("DATABASE_URL") or os.getenv("DATABASE_PUBLIC_URL", ""),
        sqlite_path=os.getenv("SQLITE_PATH", "options_bot.db"),
    )  # mirrors OrchestratorConfig.database_url; runs schema/migrations on init

    if _already_present(db):
        print("[reconcile] An open IWM short-call-spread is already in the DB. "
              "Nothing to do.")
        return 0

    order_id = f"reconcile-{uuid.uuid4()}"
    now = datetime.now(tz=timezone.utc).isoformat()
    legs_json = json.dumps(LEGS)

    params = (
        order_id,
        TRADE_DATE,
        STRATEGY,
        UNDERLYING,
        legs_json,
        NET_CREDIT,        # fill_price = net credit received
        0.0,               # slippage (unknown post-hoc)
        MAX_LOSS,          # max_loss (known: defined risk)
        HARD_STOP,         # hard_stop = 2x credit, monitor compares short-leg ask
        CONTRACTS,
        NET_CREDIT,        # net_credit
        "open",
        "reconciled",      # broker tag distinguishes from live fills & "adopted"
        now, now,
        SHORT_DELTA, None, None,   # delta (short leg), vega, theta
        None, EXPIRY,              # underlying_price (unknown post-hoc), expiry
        PROFIT_TGT, PROFIT_PCT,
    )

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

    print(f"[reconcile] backend       : {'PostgreSQL' if db._use_pg else 'SQLite'}")
    print(f"[reconcile] new order id   : {order_id}")
    print(f"[reconcile] original order : {ORIG_ORDER}")
    print(f"[reconcile] {UNDERLYING} {STRATEGY}  SELL {SHORT_SYMBOL} / BUY {LONG_SYMBOL}")
    print(f"[reconcile] credit ${NET_CREDIT}  stop ${HARD_STOP}  "
          f"target ${PROFIT_TGT}  max_loss ${MAX_LOSS}  exp {EXPIRY}")

    if args.dry_run:
        print("[reconcile] --dry-run set — not writing.")
        return 0

    conn = db._get_conn()
    try:
        db._execute(conn, sql, params)
        conn.commit()
    finally:
        conn.close()

    # Read-back confirmation
    if _already_present(db):
        print("[reconcile] OK — row written and confirmed via get_open_trades(). "
              "PositionMonitor will manage it on the next 15-min cycle.")
        return 0
    print("[reconcile] ERROR — write did not appear on read-back. Investigate.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
