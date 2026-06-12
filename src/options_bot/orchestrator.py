"""
Orchestrator — the main trading loop.

Ties every pipeline module together:

  Market data → Greeks → Strategy → Risk → ExecutionGuard → Broker
      ↑                                                         ↓
  Staleness check                                         FilledOrder
      ↑                                                         ↓
  Session gate                                        Position monitor
      ↑                                                         ↓
  Account check                                       Discord dispatch

Architecture follows your existing PropIQ pattern:
  - APScheduler owns all scheduled jobs (no loose threads)
  - Single shared state object passed by reference
  - Jobs fire at configurable times (default: scan at 9:45 AM PT,
    monitor every 15 min during market hours, close at 3:45 PM PT)
  - Discord webhook for pick dispatch (matches PropIQ setup)
  - PostgreSQL / SQLite for trade state persistence

Environment variables:
  ALPACA_API_KEY       — required
  ALPACA_SECRET_KEY    — required
  ALPACA_PAPER         — "true" (default) or "false"
  DISCORD_WEBHOOK_URL  — optional, enables Discord dispatch
  DATABASE_URL         — optional, PostgreSQL URL for Railway
                         falls back to SQLite if not set

Session gates:
  - Market must be open (NYSE calendar)
  - Account must not be blocked
  - Not within 15 min of market close (avoids illiquid fills)
  - Daily loss limit not already hit
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from .broker import AlpacaBroker, PaperBroker, get_broker
from .contracts import ApprovedOrder, FilledOrder
from .exceptions import (
    LiquidityFilterError,
    PipelineConnectionError,
    RiskVetoError,
    StalenessError,
)
from .greeks import GreeksEnricher
from .market_data import YFinanceDataLoader
from .risk import ExecutionGuard, RiskConfig, RiskManager
from .strategy import BaseStrategy, StrategySignal, get_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator configuration
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """
    All orchestrator parameters in one place.
    Override any field to customise the session.
    """
    # --- Tickers to scan ---
    tickers: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])

    # --- Strategy ---
    strategy_name: str = "short_put_spread"   # csp | short_put_spread | short_strangle
    strategy_config: object = None             # strategy-specific config dataclass

    # --- Risk ---
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # --- Scheduling (hour, minute in PT) ---
    scan_hour: int = 9       # 9:45 AM PT = 12:45 PM ET (after open volatility settles)
    scan_minute: int = 45
    monitor_interval_minutes: int = 15   # position check frequency
    close_hour: int = 15     # 3:45 PM PT = EOD, close stale orders
    close_minute: int = 45

    # --- Chain filter ---
    min_dte: int = 21
    max_dte: int = 45
    min_open_interest: int = 100
    max_spread_pct: float = 0.25

    # --- Session gates ---
    min_minutes_to_close: int = 15    # don't enter new positions within 15 min of close
    max_positions_total: int = 5      # max concurrent open positions

    # --- Broker ---
    paper: bool = True                # always default to paper

    # --- Notifications ---
    discord_webhook_url: str = field(
        default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", "")
    )

    # --- Database ---
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "")
    )
    sqlite_path: str = "options_bot.db"


# ---------------------------------------------------------------------------
# Session state — shared between all jobs
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Live state shared across scheduled jobs within one trading day."""
    trade_date: date = field(default_factory=date.today)
    equity: float = 0.0
    open_positions: list[dict] = field(default_factory=list)
    filled_today: list[FilledOrder] = field(default_factory=list)
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    scan_ran: bool = False
    errors_today: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset_for_new_day(self) -> None:
        today = date.today()
        if self.trade_date != today:
            logger.info("[SessionState] New day %s — resetting", today)
            with self._lock:
                self.trade_date = today
                self.filled_today.clear()
                self.daily_realized_pnl = 0.0
                self.daily_unrealized_pnl = 0.0
                self.scan_ran = False
                self.errors_today.clear()

    def record_error(self, msg: str) -> None:
        with self._lock:
            self.errors_today.append(f"{datetime.now(tz=timezone.utc).isoformat()}: {msg}")
        logger.error("[SessionState] %s", msg)


# ---------------------------------------------------------------------------
# Database (SQLite fallback / PostgreSQL if DATABASE_URL set)
# ---------------------------------------------------------------------------

class TradeDatabase:
    """
    Persists FilledOrders to SQLite (local) or PostgreSQL (Railway).
    Schema is intentionally flat for simplicity — one row per leg per order.
    """

    def __init__(self, database_url: str = "", sqlite_path: str = "options_bot.db"):
        self._use_pg = bool(database_url and database_url.startswith("postgres"))
        self._database_url = database_url
        self._sqlite_path = sqlite_path
        self._init_schema()

    def _get_conn(self):
        if self._use_pg:
            import psycopg2
            return psycopg2.connect(self._database_url)
        return sqlite3.connect(self._sqlite_path)

    def _init_schema(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id           TEXT PRIMARY KEY,
                    trade_date   TEXT NOT NULL,
                    strategy     TEXT,
                    underlying   TEXT,
                    legs_json    TEXT,
                    fill_price   REAL,
                    slippage     REAL,
                    max_loss     REAL,
                    hard_stop    REAL,
                    contracts    INTEGER,
                    net_credit   REAL,
                    status       TEXT DEFAULT 'open',
                    close_price  REAL,
                    realized_pnl REAL,
                    broker       TEXT,
                    created_at   TEXT,
                    updated_at   TEXT
                )
            """)
            conn.commit()
        logger.debug("[TradeDB] Schema ready (%s)", "PostgreSQL" if self._use_pg else "SQLite")

    def save_fill(self, filled: FilledOrder) -> None:
        order = filled.approved_order
        legs_json = json.dumps([
            {"symbol": l.symbol, "side": l.side,
             "strike": l.strike, "qty": l.quantity}
            for l in order.legs
        ])
        now = datetime.now(tz=timezone.utc).isoformat()
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trades
                    (id, trade_date, strategy, underlying, legs_json,
                     fill_price, slippage, max_loss, hard_stop, contracts,
                     net_credit, status, broker, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    filled.order_id,
                    date.today().isoformat(),
                    order.strategy_name,
                    order.underlying,
                    legs_json,
                    filled.fill_price,
                    filled.slippage_actual,
                    order.max_loss_dollars,
                    order.hard_stop_price,
                    order.position_size_contracts,
                    order.net_debit_credit,
                    filled.status,
                    filled.broker,
                    now, now,
                ))
                conn.commit()
            logger.info("[TradeDB] Saved: %s %s", filled.order_id, order.strategy_name)
        except Exception as exc:
            logger.error("[TradeDB] Save failed: %s", exc)

    def update_status(
        self, order_id: str, status: str,
        close_price: Optional[float] = None,
        realized_pnl: Optional[float] = None,
    ) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE trades SET status=?, close_price=?, realized_pnl=?, "
                    "updated_at=? WHERE id=?",
                    (status, close_price, realized_pnl, now, order_id)
                )
                conn.commit()
        except Exception as exc:
            logger.error("[TradeDB] Update failed for %s: %s", order_id, exc)

    def get_open_trades(self) -> list[dict]:
        try:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "SELECT id, strategy, underlying, hard_stop, contracts, "
                    "net_credit, fill_price, legs_json FROM trades WHERE status='open'"
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("[TradeDB] get_open_trades failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Discord notifier
# ---------------------------------------------------------------------------

def send_discord(webhook_url: str, message: str) -> None:
    """
    POST a message to a Discord webhook.
    Fails silently — never let a notification failure crash the trading loop.
    """
    if not webhook_url:
        return
    try:
        payload = json.dumps({"content": message}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
        logger.debug("[Discord] Message sent")
    except Exception as exc:
        logger.warning("[Discord] Send failed (non-fatal): %s", exc)


def _format_signal_message(signal: StrategySignal, filled: FilledOrder) -> str:
    """Format a Discord dispatch message for a new trade."""
    order = filled.approved_order
    legs_str = " / ".join(
        f"{l.side.replace('_', ' ').upper()} {l.strike:.0f}{l.option_type[0].upper()}"
        for l in signal.legs
    )
    return (
        f"**{signal.strategy_name.upper()} — {signal.underlying}**\n"
        f"Legs: {legs_str}\n"
        f"Credit: ${signal.estimated_fill_price:.2f}  "
        f"Stop: ${order.hard_stop_price:.2f}  "
        f"Max loss: ${order.max_loss_dollars:.0f}\n"
        f"Contracts: {order.position_size_contracts}  "
        f"DTE: {signal.dte}  "
        f"Expiry: {signal.expiry}\n"
        f"Order ID: `{filled.order_id}`\n"
        f"{signal.notes}"
    )


def _format_stop_message(trade: dict, current_price: float) -> str:
    return (
        f"⚠️ **STOP HIT — {trade['underlying']} {trade['strategy']}**\n"
        f"Order ID: `{trade['id']}`\n"
        f"Current: ${current_price:.2f}  Stop: ${trade['hard_stop']:.2f}\n"
        f"Closing position..."
    )


# ---------------------------------------------------------------------------
# Market hours helpers
# ---------------------------------------------------------------------------

def _market_is_open() -> bool:
    """
    Returns True if the US equity market is currently open.
    Uses a simple time-based check (ET 9:30–16:00 on weekdays).
    For production use pandas_market_calendars if available.
    """
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        now_et = datetime.now(tz=timezone.utc).astimezone(
            __import__("zoneinfo").ZoneInfo("America/New_York")
        )
        schedule = nyse.schedule(
            start_date=now_et.date().isoformat(),
            end_date=now_et.date().isoformat(),
        )
        if schedule.empty:
            return False
        open_t  = schedule.iloc[0]["market_open"].to_pydatetime()
        close_t = schedule.iloc[0]["market_close"].to_pydatetime()
        return open_t <= datetime.now(tz=timezone.utc) <= close_t
    except ImportError:
        pass

    # Fallback: simple ET time check
    from zoneinfo import ZoneInfo
    now_et = datetime.now(tz=timezone.utc).astimezone(ZoneInfo("America/New_York"))
    weekday = now_et.weekday()     # 0=Mon … 4=Fri
    if weekday >= 5:               # weekend
        return False
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now_et <= market_close


def _minutes_to_close() -> float:
    """Returns minutes until NYSE close (negative if already closed)."""
    from zoneinfo import ZoneInfo
    now_et = datetime.now(tz=timezone.utc).astimezone(ZoneInfo("America/New_York"))
    close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return (close_et - now_et).total_seconds() / 60


# ---------------------------------------------------------------------------
# Core pipeline — one scan for one ticker
# ---------------------------------------------------------------------------

class TradingPipeline:
    """
    Executes the full pipeline for a single ticker:
      market_data → greeks → strategy → risk → broker → db → discord
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        risk_manager: RiskManager,
        broker: AlpacaBroker | PaperBroker,
        db: TradeDatabase,
        state: SessionState,
    ):
        self.config = config
        self.rm = risk_manager
        self.broker = broker
        self.db = db
        self.state = state
        self.enricher = GreeksEnricher()   # fetches Treasury rate on init
        self.strategy = get_strategy(
            config.strategy_name, config.strategy_config
        )

    def run_for_ticker(self, ticker: str) -> Optional[FilledOrder]:
        """
        Run the full pipeline for one ticker.
        Returns FilledOrder if a trade was entered, None otherwise.
        """
        logger.info("[Pipeline] Running for %s", ticker)

        # --- Step 1: Fetch chain ---
        try:
            loader = YFinanceDataLoader(ticker)
            expirations = loader.get_expirations()
        except PipelineConnectionError as exc:
            self.state.record_error(f"{ticker} fetch failed: {exc}")
            return None

        # Pick expiration in target DTE window
        target_expiry = self._pick_expiry(expirations, loader)
        if target_expiry is None:
            logger.info("[Pipeline] %s: no expiry in %d-%d DTE window",
                        ticker, self.config.min_dte, self.config.max_dte)
            return None

        try:
            raw_rows = loader.get_chain_filtered(
                expiry=target_expiry,
                min_open_interest=self.config.min_open_interest,
                max_spread_pct=self.config.max_spread_pct,
            )
        except LiquidityFilterError as exc:
            logger.info("[Pipeline] %s %s: liquidity filter — %s", ticker, target_expiry, exc)
            return None
        except PipelineConnectionError as exc:
            self.state.record_error(f"{ticker} chain failed: {exc}")
            return None

        # --- Step 2: Enrich with Greeks ---
        enriched = self.enricher.enrich_chain_filtered(
            raw_rows, require_iv=True, min_abs_delta=0.05
        )
        if not enriched:
            logger.info("[Pipeline] %s: no enriched rows after delta filter", ticker)
            return None

        # --- Step 3: Strategy evaluation ---
        try:
            signal = self.strategy.evaluate(enriched)
        except LiquidityFilterError as exc:
            logger.info("[Pipeline] %s strategy: no qualifying contracts — %s", ticker, exc)
            return None
        except PipelineConnectionError as exc:
            self.state.record_error(f"{ticker} strategy failed: {exc}")
            return None

        # --- Step 4: Risk evaluation ---
        # Update equity from broker before sizing
        try:
            current_equity = self.broker.get_equity()
            self.rm.update_equity(current_equity)
            self.state.equity = current_equity
        except PipelineConnectionError as exc:
            logger.warning("[Pipeline] Equity fetch failed, using cached: %s", exc)

        decision = self.rm.evaluate(
            max_loss_per_contract=signal.max_loss_per_contract,
            hard_stop_price=signal.hard_stop_price,
            option=signal.source_contracts[0] if signal.source_contracts else None,
            strategy_name=signal.strategy_name,
        )
        if not decision.approved:
            logger.info("[Pipeline] %s risk veto: %s", ticker, decision.rejection_reason)
            return None

        # --- Step 5: Build order ---
        order = self.rm.build_approved_order(
            legs=signal.legs,
            decision=decision,
            net_debit_credit=signal.net_debit_credit,
            estimated_fill_price=signal.estimated_fill_price,
            hard_stop_price=signal.hard_stop_price,
            profit_target_price=signal.profit_target_price,
            strategy_name=signal.strategy_name,
            underlying=signal.underlying,
            notes=signal.notes,
        )

        # --- Step 6: ExecutionGuard (redundant — defense in depth) ---
        try:
            ExecutionGuard.check(order)
        except RiskVetoError as exc:
            self.state.record_error(f"ExecutionGuard veto on {ticker}: {exc}")
            return None

        # --- Step 7: Submit ---
        try:
            filled = self.broker.submit(order)
        except (RiskVetoError, PipelineConnectionError) as exc:
            self.state.record_error(f"{ticker} submit failed: {exc}")
            return None

        # --- Step 8: Record ---
        self.rm.record_trade_opened()
        self.db.save_fill(filled)
        with self.state._lock:
            self.state.filled_today.append(filled)

        msg = _format_signal_message(signal, filled)
        logger.info("[Pipeline] Trade entered: %s", msg.replace("\n", " | "))
        send_discord(self.config.discord_webhook_url, msg)

        return filled

    def _pick_expiry(self, expirations: list[str], loader: YFinanceDataLoader) -> Optional[str]:
        """
        Select the expiration closest to the mid-point of the DTE window.
        e.g. for min_dte=21, max_dte=45 → target DTE ≈ 33
        """
        today = date.today()
        target_dte = (self.config.min_dte + self.config.max_dte) / 2
        best = None
        best_diff = float("inf")
        for exp_str in expirations:
            try:
                exp_date = date.fromisoformat(exp_str)
                dte = (exp_date - today).days
                if self.config.min_dte <= dte <= self.config.max_dte:
                    diff = abs(dte - target_dte)
                    if diff < best_diff:
                        best_diff = diff
                        best = exp_str
            except ValueError:
                continue
        return best


# ---------------------------------------------------------------------------
# Position monitor — checks stops and marks-to-market
# ---------------------------------------------------------------------------

class PositionMonitor:
    """
    Monitors open positions against hard stops.
    Called on a schedule every N minutes during market hours.
    """

    def __init__(
        self,
        broker: AlpacaBroker | PaperBroker,
        db: TradeDatabase,
        state: SessionState,
        discord_webhook_url: str = "",
    ):
        self.broker = broker
        self.db = db
        self.state = state
        self.discord_webhook_url = discord_webhook_url

    def run(self) -> None:
        """Check all open trades against their stop levels."""
        logger.debug("[Monitor] Running position check")
        open_trades = self.db.get_open_trades()
        if not open_trades:
            logger.debug("[Monitor] No open trades")
            return

        # Collect all symbols to batch-quote
        all_symbols = []
        for trade in open_trades:
            try:
                legs = json.loads(trade.get("legs_json", "[]"))
                all_symbols.extend(l["symbol"] for l in legs)
            except (json.JSONDecodeError, KeyError):
                continue

        if not all_symbols:
            return

        # Fetch latest quotes
        try:
            quotes = self.broker.get_latest_quotes(all_symbols)
        except PipelineConnectionError as exc:
            logger.warning("[Monitor] Quote fetch failed: %s", exc)
            return

        # Check each trade
        for trade in open_trades:
            self._check_trade(trade, quotes)

        # Update unrealized P&L in session state
        positions = self.broker.get_positions()
        total_unreal = sum(p.get("unrealized_pl", 0) for p in positions)
        with self.state._lock:
            self.state.open_positions = positions
            self.state.daily_unrealized_pnl = total_unreal

    def _check_trade(self, trade: dict, quotes: dict) -> None:
        """Check one trade against its stop. Submit close order if stop is hit."""
        try:
            legs = json.loads(trade.get("legs_json", "[]"))
            hard_stop = float(trade.get("hard_stop", 0))
            if hard_stop <= 0:
                return

            # For single-leg: check the short leg's ask price
            # For multi-leg: check the net value of the spread
            short_legs = [l for l in legs if "sell" in l.get("side", "")]
            if not short_legs:
                return

            # Use the first short leg as the primary price signal
            short_symbol = short_legs[0]["symbol"]
            quote = quotes.get(short_symbol, {})
            ask_price = quote.get("ask")

            if ask_price is None:
                logger.debug("[Monitor] No ask for %s — skipping stop check", short_symbol)
                return

            if ask_price >= hard_stop:
                logger.warning(
                    "[Monitor] STOP HIT: %s %s — ask=%.2f >= stop=%.2f",
                    trade["underlying"], trade["strategy"], ask_price, hard_stop
                )
                self._close_trade(trade, ask_price)

        except Exception as exc:
            logger.error("[Monitor] Error checking trade %s: %s", trade.get("id"), exc)

    def _close_trade(self, trade: dict, current_price: float) -> None:
        """Submit closing orders for a stopped-out trade."""
        try:
            legs = json.loads(trade.get("legs_json", "[]"))
            for leg in legs:
                symbol = leg["symbol"]
                # Reverse the position intent: sell_to_open → buy_to_close, etc.
                close_side = _reverse_side(leg.get("side", ""))
                if close_side:
                    self.broker.close_position(symbol)
                    logger.info("[Monitor] Closed: %s", symbol)

            # Update DB
            entry_price = float(trade.get("fill_price", 0) or 0)
            realized_pnl = (entry_price - current_price) * float(trade.get("contracts", 1)) * 100
            self.db.update_status(
                trade["id"],
                status="stopped_out",
                close_price=current_price,
                realized_pnl=realized_pnl,
            )
            with self.state._lock:
                self.state.daily_realized_pnl += realized_pnl

            msg = _format_stop_message(trade, current_price)
            send_discord(self.discord_webhook_url, msg)

        except Exception as exc:
            logger.error("[Monitor] Close failed for %s: %s", trade.get("id"), exc)


def _reverse_side(side: str) -> Optional[str]:
    mapping = {
        "sell_to_open":  "buy_to_close",
        "buy_to_open":   "sell_to_close",
        "sell_to_close": "buy_to_open",
        "buy_to_close":  "sell_to_open",
    }
    return mapping.get(side)


# ---------------------------------------------------------------------------
# Orchestrator — main entry point
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Main trading orchestrator. Manages the full lifecycle:
      - Session setup (account check, equity sync)
      - Scheduled scan jobs (APScheduler)
      - Position monitoring
      - EOD cleanup
      - Discord reporting

    Usage:
        config = OrchestratorConfig(
            tickers=["SPY", "QQQ"],
            strategy_name="short_put_spread",
            paper=True,
        )
        orch = Orchestrator(config)
        orch.run()               # blocks — runs scheduler loop
        # OR for single manual scan:
        orch.run_scan()          # one-shot scan, returns immediately
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.state = SessionState()

        # Broker
        self.broker = get_broker(paper=config.paper, use_paper_stub=False)

        # Risk manager — equity synced from broker on each scan
        initial_equity = self._safe_get_equity()
        self.rm = RiskManager(equity=initial_equity, config=config.risk_config)

        # Database
        self.db = TradeDatabase(
            database_url=config.database_url,
            sqlite_path=config.sqlite_path,
        )

        # Pipeline and monitor
        self.pipeline = TradingPipeline(
            config=config,
            risk_manager=self.rm,
            broker=self.broker,
            db=self.db,
            state=self.state,
        )
        self.monitor = PositionMonitor(
            broker=self.broker,
            db=self.db,
            state=self.state,
            discord_webhook_url=config.discord_webhook_url,
        )

        logger.info(
            "[Orchestrator] Ready: tickers=%s strategy=%s paper=%s",
            config.tickers, config.strategy_name, config.paper
        )

    def run(self) -> None:
        """
        Start the APScheduler loop. Blocks indefinitely.
        Press Ctrl+C to stop.
        """
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            raise PipelineConnectionError(
                "apscheduler not installed. Run: pip install apscheduler"
            )

        scheduler = BlockingScheduler(timezone="America/Los_Angeles")

        # Scan job: fire once per day at scan_hour:scan_minute PT
        scheduler.add_job(
            self.run_scan,
            CronTrigger(
                hour=self.config.scan_hour,
                minute=self.config.scan_minute,
                timezone="America/Los_Angeles",
            ),
            id="scan",
            name="Daily option scan",
            misfire_grace_time=300,
        )

        # Monitor job: fire every N minutes during market hours
        scheduler.add_job(
            self.run_monitor,
            IntervalTrigger(minutes=self.config.monitor_interval_minutes),
            id="monitor",
            name="Position monitor",
        )

        # EOD job: cancel unfilled orders, send daily summary
        scheduler.add_job(
            self.run_eod,
            CronTrigger(
                hour=self.config.close_hour,
                minute=self.config.close_minute,
                timezone="America/Los_Angeles",
            ),
            id="eod",
            name="EOD cleanup",
        )

        logger.info(
            "[Orchestrator] Scheduler starting: scan=%02d:%02d PT, "
            "monitor=every %dmin, eod=%02d:%02d PT",
            self.config.scan_hour, self.config.scan_minute,
            self.config.monitor_interval_minutes,
            self.config.close_hour, self.config.close_minute,
        )
        send_discord(
            self.config.discord_webhook_url,
            f"🟢 **Options Bot started** — {self.config.strategy_name.upper()} "
            f"{'[PAPER]' if self.config.paper else '[LIVE]'}\n"
            f"Tickers: {', '.join(self.config.tickers)}"
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("[Orchestrator] Shutting down")
            scheduler.shutdown(wait=False)

    def run_scan(self) -> list[FilledOrder]:
        """
        One scan across all configured tickers.
        Can be called manually for testing without the scheduler.
        Returns list of FilledOrders (may be empty).
        """
        self.state.reset_for_new_day()
        logger.info("[Orchestrator] === SCAN START %s ===", date.today())

        # Session gates
        if not _market_is_open():
            logger.info("[Orchestrator] Market closed — skipping scan")
            return []

        if _minutes_to_close() < self.config.min_minutes_to_close:
            logger.info("[Orchestrator] Too close to market close — skipping scan")
            return []

        try:
            self.broker.check_account_ready()
        except PipelineConnectionError as exc:
            self.state.record_error(f"Account not ready: {exc}")
            return []

        # Check position count
        positions = self.broker.get_positions()
        option_positions = [p for p in positions if p.get("asset_class") == "us_option"]
        if len(option_positions) >= self.config.max_positions_total:
            logger.info(
                "[Orchestrator] Max positions reached (%d/%d) — skipping scan",
                len(option_positions), self.config.max_positions_total
            )
            return []

        filled_orders = []
        for ticker in self.config.tickers:
            # Don't exceed max positions mid-scan
            positions = self.broker.get_positions()
            current_count = len([p for p in positions if p.get("asset_class") == "us_option"])
            if current_count >= self.config.max_positions_total:
                logger.info("[Orchestrator] Max positions reached — stopping scan")
                break

            filled = self.pipeline.run_for_ticker(ticker)
            if filled:
                filled_orders.append(filled)

        self.state.scan_ran = True
        logger.info(
            "[Orchestrator] === SCAN END: %d trades entered ===",
            len(filled_orders)
        )
        return filled_orders

    def run_monitor(self) -> None:
        """Check all open positions against stops. Safe to call any time."""
        if not _market_is_open():
            return
        self.state.reset_for_new_day()
        self.monitor.run()

    def run_eod(self) -> None:
        """EOD cleanup: cancel unfilled orders, send daily summary to Discord."""
        logger.info("[Orchestrator] EOD cleanup")
        cancelled = self.broker.cancel_all_orders()
        if cancelled:
            logger.info("[Orchestrator] Cancelled %d unfilled orders", cancelled)

        open_trades = self.db.get_open_trades()
        summary = (
            f"📊 **Daily Summary — {date.today()}**\n"
            f"Strategy: {self.config.strategy_name.upper()}  "
            f"{'[PAPER]' if self.config.paper else '[LIVE]'}\n"
            f"Trades today: {len(self.state.filled_today)}\n"
            f"Open positions: {len(open_trades)}\n"
            f"Realized P&L: ${self.state.daily_realized_pnl:+.2f}\n"
            f"Unrealized P&L: ${self.state.daily_unrealized_pnl:+.2f}\n"
            f"Errors: {len(self.state.errors_today)}"
        )
        logger.info("[Orchestrator] %s", summary.replace("\n", " | "))
        send_discord(self.config.discord_webhook_url, summary)

    def _safe_get_equity(self) -> float:
        """Get equity from broker, fall back to config default on failure."""
        try:
            return self.broker.get_equity()
        except Exception as exc:
            logger.warning("[Orchestrator] Equity fetch failed, using 50000: %s", exc)
            return 50_000.0
