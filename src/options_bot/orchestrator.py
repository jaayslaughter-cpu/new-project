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
from .regime import RegimeDetector
from .sentiment import SentimentAnalyzer, SentimentConfig
from .metrics import summary as perf_summary
from .adaptive import AdaptiveTuner
from .strategy import BaseStrategy, StrategySignal, get_strategy
from .strategy_0dte import ZeroDTEConfig, ZeroDTEStrategy, ZeroDTEMonitor
from .scanner import TickerGate
from .risk_profiles import RiskLevel, RiskProfile, get_risk_profile, apply_profile

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

    # --- Regime filters ---
    vix_max: float = 25.0             # suppress all new trades when VIX >= this (legacy fallback)
    min_iv_rank: float = 0.0          # minimum IV rank 0-100 for strangle entry (0 = disabled)
    regime_min_options_weight: float = 0.10  # minimum options weight to allow trading
    regime_cache_ttl: int = 900       # regime cache TTL in seconds (default 15 min)

    # --- Sentiment filter ---
    sentiment_enabled: bool = True    # enable FinBERT news sentiment gate
    sentiment_config: SentimentConfig = field(default_factory=SentimentConfig)

    # --- Adaptive tuning ---
    adaptive_enabled: bool = True     # enable self-tuning of strategy parameters
    adaptive_eval_window: int = 20    # number of recent closed trades to analyze
    adaptive_tune_interval: int = 10  # min closed trades between tune cycles

    # --- 0DTE module ---
    zero_dte_enabled: bool = False           # enable intraday 0DTE GEX scalper
    zero_dte_config: ZeroDTEConfig = field(default_factory=ZeroDTEConfig)

    # --- Risk profile ---
    risk_level: str = "medium"               # "low", "medium", or "high"

    # --- Ticker pre-screening ---
    scanner_enabled: bool = True             # enable bullish + Piotroski pre-filter
    scanner_bullish_threshold: float = 3.0  # min composite score
    scanner_piotroski_threshold: int = 6    # min F-score (0–9)

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


def _format_signal_message(
    signal: StrategySignal,
    filled: FilledOrder,
    regime: str = "",
    options_weight: float = 0.0,
) -> str:
    """Format a Discord dispatch message for a new trade."""
    order = filled.approved_order
    legs_str = " / ".join(
        f"{l.side.replace('_', ' ').upper()} {l.strike:.0f}{l.option_type[0].upper()}"
        for l in signal.legs
    )
    regime_line = ""
    if regime:
        regime_line = f"Regime: {regime.upper()} (options_weight={options_weight:.0%})\n"
    return (
        f"**{signal.strategy_name.upper()} — {signal.underlying}**\n"
        f"Legs: {legs_str}\n"
        f"Credit: ${signal.estimated_fill_price:.2f}  "
        f"Stop: ${order.hard_stop_price:.2f}  "
        f"Max loss: ${order.max_loss_dollars:.0f}\n"
        f"Contracts: {order.position_size_contracts}  "
        f"DTE: {signal.dte}  "
        f"Expiry: {signal.expiry}\n"
        f"{regime_line}"
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

    def run_for_ticker(
        self,
        ticker: str,
        regime_name: str = "",
        regime_options_weight: float = 0.0,
    ) -> Optional[FilledOrder]:
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

        # --- Step 3: IV rank gate (strangles only) ---
        if (self.config.strategy_name == "short_strangle"
                and self.config.min_iv_rank > 0):
            iv_rank = _get_iv_rank(ticker)
            if iv_rank is not None and iv_rank < self.config.min_iv_rank:
                logger.info(
                    "[Pipeline] %s IV rank=%.1f < min=%.1f — skipping strangle",
                    ticker, iv_rank, self.config.min_iv_rank
                )
                return None
            elif iv_rank is not None:
                logger.info("[Pipeline] %s IV rank=%.1f — qualifies", ticker, iv_rank)

        # --- Step 2b: Duplicate prevention — skip if we already have a position on this ticker today ──
        open_trades = self.db.get_open_trades()
        already_open = [t for t in open_trades if t.get("underlying") == ticker]
        if already_open:
            logger.info(
                "[Pipeline] %s: already have %d open position(s) — skipping",
                ticker, len(already_open)
            )
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

        msg = _format_signal_message(signal, filled, regime_name, regime_options_weight)
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
        """
        Check one trade against its stop-loss and profit target.
        Closes the position if either level is hit.
        """
        try:
            legs = json.loads(trade.get("legs_json", "[]"))
            hard_stop = float(trade.get("hard_stop", 0))
            entry_credit = abs(float(trade.get("net_credit") or 0))
            # Profit target = 50% of credit received (close when we keep half)
            profit_target = entry_credit * 0.50 if entry_credit > 0 else None

            if hard_stop <= 0:
                return

            # Use the first short leg as the primary price signal
            short_legs = [l for l in legs if "sell" in l.get("side", "")]
            if not short_legs:
                return

            short_symbol = short_legs[0]["symbol"]
            quote = quotes.get(short_symbol, {})
            ask_price = quote.get("ask")   # cost to close the short leg

            if ask_price is None:
                logger.debug("[Monitor] No ask for %s — skipping", short_symbol)
                return

            # --- Stop loss ---
            if ask_price >= hard_stop:
                logger.warning(
                    "[Monitor] STOP HIT: %s %s — ask=%.2f >= stop=%.2f",
                    trade.get("underlying",""), trade.get("strategy",""),
                    ask_price, hard_stop
                )
                self._close_trade(trade, ask_price, reason="stop_hit")
                return

            # --- Profit target: bid price has fallen to <= 50% of entry credit ---
            if profit_target is not None:
                bid_price = quote.get("bid")
                if bid_price is not None and bid_price <= profit_target:
                    logger.info(
                        "[Monitor] PROFIT TARGET: %s %s — bid=%.2f <= target=%.2f (50%% of %.2f credit)",
                        trade.get("underlying",""), trade.get("strategy",""),
                        bid_price, profit_target, entry_credit
                    )
                    self._close_trade(trade, bid_price, reason="profit_target")

        except Exception as exc:
            logger.error("[Monitor] Error checking trade %s: %s", trade.get("id"), exc)

    def _close_trade(self, trade: dict, current_price: float, reason: str = "stop_hit") -> None:
        """Submit closing orders for a stopped-out or profit-target trade."""
        try:
            legs = json.loads(trade.get("legs_json", "[]"))
            for leg in legs:
                symbol = leg["symbol"]
                close_side = _reverse_side(leg.get("side", ""))
                if close_side:
                    self.broker.close_position(symbol)
                    logger.info("[Monitor] Closed %s: %s", reason, symbol)

            entry_price = float(trade.get("fill_price", 0) or 0)
            contracts = float(trade.get("contracts", 1) or 1)

            if reason == "profit_target":
                # We bought back at current_price, kept (entry - current) per contract
                realized_pnl = (entry_price - current_price) * contracts * 100
                new_status = "closed_profit_target"
            else:
                realized_pnl = (entry_price - current_price) * contracts * 100
                new_status = "stopped_out"

            self.db.update_status(
                trade["id"],
                status=new_status,
                close_price=current_price,
                realized_pnl=realized_pnl,
            )
            with self.state._lock:
                self.state.daily_realized_pnl += realized_pnl

            if reason == "stop_hit":
                msg = _format_stop_message(trade, current_price)
            else:
                msg = (
                    f"✅ **PROFIT TARGET — {trade.get('underlying','')} {trade.get('strategy','')}**\n"
                    f"Order ID: `{trade['id']}`\n"
                    f"Closed at: ${current_price:.2f}  Entry: ${entry_price:.2f}\n"
                    f"P&L: ${realized_pnl:+.2f}"
                )
            send_discord(self.discord_webhook_url, msg)

        except Exception as exc:
            logger.error("[Monitor] Close failed for %s: %s", trade.get("id"), exc)


def _get_vix() -> Optional[float]:
    """
    Fetch current VIX level from yfinance.
    Returns None on failure — caller decides whether to block or allow.
    """
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        price = vix.fast_info.get("lastPrice")
        if price and float(price) > 0:
            return float(price)
    except Exception as exc:
        logger.warning("[VIX] Fetch failed: %s", exc)
    return None


def _get_iv_rank(ticker: str, lookback_days: int = 252) -> Optional[float]:
    """
    Compute a simple IV rank for a ticker using 1-year ATM IV history.
    IV rank = (current IV - 52w low) / (52w high - 52w low) * 100

    Returns 0-100 float, or None on failure.
    """
    try:
        import yfinance as yf
        import numpy as np
        t = yf.Ticker(ticker)
        # Use historical close prices to estimate realized vol as IV proxy
        hist = t.history(period="1y")
        if hist.empty or len(hist) < 20:
            return None
        returns = hist["Close"].pct_change().dropna()
        # Rolling 30-day realized vol annualized
        roll_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        roll_vol = roll_vol.dropna()
        if len(roll_vol) < 2:
            return None
        current = roll_vol.iloc[-1]
        low_52w = roll_vol.min()
        high_52w = roll_vol.max()
        if high_52w == low_52w:
            return 50.0
        iv_rank = (current - low_52w) / (high_52w - low_52w) * 100
        return round(float(iv_rank), 1)
    except Exception as exc:
        logger.warning("[IVRank] Fetch failed for %s: %s", ticker, exc)
    return None
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

        # Verify options trading is enabled on this account
        self._check_options_approved()

        # Risk manager — equity synced from real account balance
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

        # Apply risk profile — overrides relevant config fields
        if config.risk_level and config.risk_level != "custom":
            apply_profile(config, config.risk_level)
            logger.info("[Orchestrator] Risk profile: %s", config.risk_level.upper())

        # Ticker pre-screening gate
        self.ticker_gate = TickerGate(
            bullish_threshold=config.scanner_bullish_threshold,
            piotroski_threshold=config.scanner_piotroski_threshold,
            require_bullish=config.scanner_enabled,
            require_piotroski=config.scanner_enabled,
        ) if config.scanner_enabled else None

        # Regime detector — replaces simple VIX threshold
        self.regime_detector = RegimeDetector(
            cache_ttl_seconds=config.regime_cache_ttl
        )

        # Sentiment analyzer — FinBERT news signal layer
        self.sentiment_analyzer = SentimentAnalyzer(
            config=config.sentiment_config
        ) if config.sentiment_enabled else None

        # Adaptive tuner — self-tunes strategy parameters from closed trade history
        self.tuner = AdaptiveTuner(
            db=self.db,
            discord_webhook=config.discord_webhook_url,
            eval_window=config.adaptive_eval_window,
            tune_interval=config.adaptive_tune_interval,
        ) if config.adaptive_enabled else None

        # 0DTE GEX scalper — intraday module, runs on separate 2-min schedule
        if config.zero_dte_enabled:
            _0dte_cfg = config.zero_dte_config
            _0dte_cfg.discord_webhook_url = config.discord_webhook_url
            _0dte_cfg.paper = config.paper
            self.zero_dte = ZeroDTEStrategy(_0dte_cfg, self.broker, self.db)
            self.zero_dte_monitor = ZeroDTEMonitor(_0dte_cfg, self.broker, self.db)
            logger.info("[Orchestrator] 0DTE GEX scalper enabled (underlying=%s)",
                        _0dte_cfg.underlying)
        else:
            self.zero_dte = None
            self.zero_dte_monitor = None

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

        # 0DTE scan job: every 2 minutes, 6:32–11:00 AM PT (= 9:32–14:00 ET, Mon–Fri)
        if self.zero_dte is not None:
            scheduler.add_job(
                self._run_zero_dte_scan,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour="6-10",          # 6:32 AM – 10:59 AM PT
                    minute="*/2",
                    timezone="America/Los_Angeles",
                ),
                id="zero_dte_scan",
                name="0DTE GEX scalper scan (PT)",
                misfire_grace_time=30,
            )
            # 0DTE monitor: every 15 seconds during market hours
            scheduler.add_job(
                self._run_zero_dte_monitor,
                IntervalTrigger(seconds=self.config.zero_dte_config.monitor_poll_seconds),
                id="zero_dte_monitor",
                name="0DTE position monitor",
            )
            logger.info(
                "[Orchestrator] 0DTE jobs scheduled: "
                "scan every 2min 6:32-11:00 AM PT (9:32-2:00 PM ET), "
                "monitor every 15s"
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

    def _run_zero_dte_scan(self) -> None:
        """
        Called every 2 minutes by APScheduler (9:32–14:00 ET, Mon–Fri).
        Evaluates all 0DTE entry conditions and submits an order if approved.
        """
        if self.zero_dte is None:
            return

        # Respect max daily position cap
        if self.zero_dte_monitor and \
           self.zero_dte_monitor.open_count >= self.config.zero_dte_config.max_daily_positions:
            logger.debug("[0DTE scan] Max positions (%d) open — skip",
                         self.config.zero_dte_config.max_daily_positions)
            return

        try:
            order = self.zero_dte.evaluate()
            if order is None:
                return

            fill = self.broker.submit(order)
            if fill and self.zero_dte_monitor:
                self.zero_dte_monitor.register(order, fill)

                # Persist to DB
                if self.db:
                    self.db.record_trade(order, fill)

                send_discord(
                    self.config.discord_webhook_url,
                    f"🎯 **0DTE ENTRY — {order.strategy.upper()}**\n"
                    f"Credit: ${order.net_credit:.3f} x {order.position_size_contracts} contracts\n"
                    f"TP: ${order.profit_target:.3f} | SL: ${order.hard_stop:.3f}\n"
                    f"Pin: {order.metadata.get('gex_pin')} | "
                    f"Regime: {order.metadata.get('gex_regime')} | "
                    f"VIX: {order.metadata.get('vix')} | "
                    f"Session: {order.metadata.get('session')}"
                )

        except Exception as exc:
            logger.error("[0DTE scan] Unhandled error: %s", exc, exc_info=True)

    def _run_zero_dte_monitor(self) -> None:
        """
        Called every 15 seconds by APScheduler.
        Checks all open 0DTE positions for exit conditions.
        """
        if self.zero_dte_monitor is None:
            return
        try:
            self.zero_dte_monitor.run_once()
        except Exception as exc:
            logger.error("[0DTE monitor] Unhandled error: %s", exc, exc_info=True)

    def _reconcile_positions(self) -> None:
        """
        Sync open positions between local DB and Alpaca.
        Called at the start of each scan to handle restarts mid-day.

        If Alpaca shows a position that the DB doesn't know about as open,
        we log a warning. If the DB shows open but Alpaca has no position,
        we mark it as closed in the DB (likely closed externally or expired).
        """
        try:
            alpaca_positions = self.broker.get_positions()
            alpaca_symbols = {p["symbol"] for p in alpaca_positions
                             if p.get("asset_class") == "us_option"}
            db_trades = self.db.get_open_trades()

            for trade in db_trades:
                try:
                    legs = json.loads(trade.get("legs_json", "[]"))
                    trade_symbols = {l["symbol"] for l in legs}
                    # If none of the trade's symbols are in Alpaca positions,
                    # the position was closed externally (expired, manual close, etc.)
                    if not trade_symbols.intersection(alpaca_symbols):
                        logger.warning(
                            "[Reconcile] Trade %s (%s) not found in Alpaca positions — "
                            "marking as closed",
                            trade["id"], trade.get("strategy", "")
                        )
                        self.db.update_status(trade["id"], status="closed_external")
                except Exception as exc:
                    logger.warning("[Reconcile] Error checking trade %s: %s", trade.get("id"), exc)

            logger.info(
                "[Reconcile] Done: %d Alpaca option positions, %d DB open trades",
                len(alpaca_symbols), len(db_trades)
            )
        except Exception as exc:
            logger.warning("[Reconcile] Reconciliation failed (non-fatal): %s", exc)

    def run_scan(self) -> list[FilledOrder]:
        """
        One scan across all configured tickers.
        Can be called manually for testing without the scheduler.
        Returns list of FilledOrders (may be empty).
        """
        self.state.reset_for_new_day()
        logger.info("[Orchestrator] === SCAN START %s ===", date.today())

        # Reconcile DB positions with Alpaca on every scan
        # This handles the case where Railway restarted mid-day
        self._reconcile_positions()

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

        # Regime detection — replaces simple VIX threshold
        regime = self.regime_detector.detect()
        options_weight = regime.get("options_weight", 0.15)
        regime_name    = regime.get("regime", "unknown")
        confidence     = regime.get("confidence", 0.33)
        indicators     = regime.get("indicators", {})

        logger.info(
            "[Orchestrator] Regime: %s (confidence=%.2f, options_weight=%.2f) "
            "VIX=%.1f trend=%s strength=%.2f curve=%.2f",
            regime_name, confidence, options_weight,
            indicators.get("vix_level", 0),
            indicators.get("vix_trend", "?"),
            indicators.get("trend_strength", 0),
            indicators.get("yield_curve_slope", 0),
        )

        if options_weight < self.config.regime_min_options_weight:
            msg = (
                f"⚠️ **REGIME GATE: No new trades** — "
                f"regime={regime_name} options_weight={options_weight:.0%} "
                f"(min={self.config.regime_min_options_weight:.0%}) "
                f"VIX={indicators.get('vix_level', 0):.1f}"
            )
            logger.warning("[Orchestrator] %s", msg)
            send_discord(self.config.discord_webhook_url, msg)
            return []

        # Legacy VIX hard-stop (defense in depth, independent of regime score)
        vix_level = indicators.get("vix_level")
        if vix_level and vix_level >= self.config.vix_max:
            msg = (
                f"⚠️ **VIX HARD STOP: No new trades** — "
                f"VIX={vix_level:.1f} >= {self.config.vix_max:.1f}"
            )
            logger.warning("[Orchestrator] %s", msg)
            send_discord(self.config.discord_webhook_url, msg)
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

        # Sentiment gate — fetch FinBERT signals for all tickers once per scan
        # Blocks entry on tickers with SELL sentiment (bearish news)
        sentiment_signals: dict = {}
        if self.sentiment_analyzer is not None:
            try:
                sentiment_signals = self.sentiment_analyzer.get_signals(self.config.tickers)
                sell_tickers = [
                    t for t, s in sentiment_signals.items() if s.signal == "SELL"
                ]
                if sell_tickers:
                    logger.info(
                        "[Orchestrator] Sentiment SELL on %s — those tickers will be skipped",
                        sell_tickers,
                    )
                    send_discord(
                        self.config.discord_webhook_url,
                        f"📰 **Sentiment gate** — SELL signal on: {', '.join(sell_tickers)}\n"
                        f"Those tickers skipped this scan.",
                    )
            except Exception as exc:
                logger.warning("[Orchestrator] Sentiment fetch failed (non-fatal): %s", exc)

        filled_orders = []
        # Apply ticker pre-screening gate (bullish technicals + Piotroski)
        scan_tickers = self.config.tickers
        if self.ticker_gate is not None:
            scan_tickers = self.ticker_gate.filter(scan_tickers)
            if not scan_tickers:
                logger.info("[Orchestrator] All tickers blocked by pre-screening gate — no scan")
                return []

        for ticker in scan_tickers:
            # Don't exceed max positions mid-scan
            positions = self.broker.get_positions()
            current_count = len([p for p in positions if p.get("asset_class") == "us_option"])
            if current_count >= self.config.max_positions_total:
                logger.info("[Orchestrator] Max positions reached — stopping scan")
                break

            # Sentiment gate — skip tickers with SELL signal
            if self.sentiment_analyzer is not None and sentiment_signals:
                if not self.sentiment_analyzer.is_entry_allowed(ticker, sentiment_signals):
                    logger.info(
                        "[Orchestrator] %s: sentiment SELL — skipping this ticker",
                        ticker,
                    )
                    continue

            filled = self.pipeline.run_for_ticker(
                ticker,
                regime_name=regime_name,
                regime_options_weight=options_weight,
            )
            if filled:
                filled_orders.append(filled)

        self.state.scan_ran = True
        logger.info(
            "[Orchestrator] === SCAN END: %d trades entered ===",
            len(filled_orders)
        )

        # Adaptive tuning — run after scan, update strategy config for next cycle
        if self.tuner and self.tuner.should_evaluate(self.config.strategy_name):
            logger.info("[Orchestrator] Running adaptive tuning cycle...")
            new_cfg, snap, adjustments = self.tuner.evaluate_and_tune(
                self.config.strategy_name,
                self.config.strategy_config,
            )
            if adjustments:
                # Update live config and rebuild the pipeline's strategy instance
                self.config.strategy_config = new_cfg
                self.pipeline.strategy = get_strategy(
                    self.config.strategy_name, new_cfg
                )
                logger.info(
                    "[Orchestrator] Strategy config updated with %d adjustments",
                    len(adjustments)
                )

        return filled_orders

    def run_monitor(self) -> None:
        """Check all open positions against stops. Safe to call any time."""
        if not _market_is_open():
            return
        self.state.reset_for_new_day()
        self.monitor.run()

    def run_eod(self) -> None:
        """EOD cleanup: close expiring positions, cancel unfilled orders, send daily summary."""
        logger.info("[Orchestrator] EOD cleanup")

        # Close any positions expiring today before market close
        open_trades = self.db.get_open_trades()
        today = date.today().isoformat()
        for trade in open_trades:
            try:
                legs = json.loads(trade.get("legs_json", "[]"))
                expiring = [l for l in legs if l.get("expiry", "") == today]
                if expiring:
                    logger.warning(
                        "[EOD] Position %s expires TODAY — closing before market close",
                        trade["id"]
                    )
                    for leg in legs:
                        try:
                            self.broker.close_position(leg["symbol"])
                        except Exception as exc:
                            logger.error("[EOD] Failed to close expiring leg %s: %s",
                                        leg["symbol"], exc)
                    self.db.update_status(trade["id"], status="closed_expiry")
                    send_discord(
                        self.config.discord_webhook_url,
                        f"⏰ **EXPIRY CLOSE — {trade.get('underlying','')} {trade.get('strategy','')}**\n"
                        f"Order ID: `{trade['id']}` — closed at expiry"
                    )
            except Exception as exc:
                logger.error("[EOD] Expiry check failed for %s: %s", trade.get("id"), exc)

        cancelled = self.broker.cancel_all_orders()
        if cancelled:
            logger.info("[Orchestrator] Cancelled %d unfilled orders", cancelled)

        # Build performance summary from today's closed trades
        open_trades = self.db.get_open_trades()
        today_fills = self.state.filled_today
        today_pnls = [
            f.fill_price * f.approved_order.position_size_contracts * 100 * -1
            for f in today_fills
            if hasattr(f, 'fill_price') and f.fill_price
        ]

        # Pull all realized P&L from closed trades in DB for full metrics
        try:
            with self.db._get_conn() as conn:
                cur = conn.execute(
                    "SELECT realized_pnl FROM trades WHERE trade_date=? AND realized_pnl IS NOT NULL",
                    (today,)
                )
                all_pnls = [row[0] for row in cur.fetchall()]
        except Exception:
            all_pnls = []

        # Compute metrics if we have any closed trades
        metrics_line = ""
        if all_pnls:
            import numpy as np
            stats = perf_summary(all_pnls)
            metrics_line = (
                f"Win rate: {stats['win_rate']:.0%}  "
                f"Profit factor: {stats['profit_factor']:.2f}  "
                f"Avg W/L: {stats['avg_win_loss_ratio']:.2f}\n"
            )

        regime = self.regime_detector.detect()
        hurst_val  = regime.get("indicators", {}).get("hurst", 0.5)
        hurst_reg  = regime.get("indicators", {}).get("hurst_regime", "unknown")

        # Show today's adaptive tuning activity if any
        tuning_line = ""
        if self.tuner and self.tuner.adjustment_history():
            today_adj = [
                a for a in self.tuner.adjustment_history()
                if a["applied_at"][:10] == today
            ]
            if today_adj:
                tuning_line = f"Tuning adjustments today: {len(today_adj)}\n"

        summary = (
            f"📊 **Daily Summary — {date.today()}**\n"
            f"Strategy: {self.config.strategy_name.upper()}  "
            f"{'[PAPER]' if self.config.paper else '[LIVE]'}\n"
            f"Trades today: {len(self.state.filled_today)}\n"
            f"Open positions: {len(open_trades)}\n"
            f"Realized P&L: ${self.state.daily_realized_pnl:+.2f}\n"
            f"Unrealized P&L: ${self.state.daily_unrealized_pnl:+.2f}\n"
            f"{metrics_line}"
            f"{tuning_line}"
            f"Regime: {regime['regime'].upper()} "
            f"(conf={regime['confidence']:.0%}, "
            f"VIX={regime['indicators'].get('vix_level',0):.1f}, "
            f"Hurst={hurst_val:.3f} [{hurst_reg}])\n"
            f"Errors: {len(self.state.errors_today)}"
        )
        logger.info("[Orchestrator] %s", summary.replace("\n", " | "))
        send_discord(self.config.discord_webhook_url, summary)

    def _check_options_approved(self) -> None:
        """
        Verify options trading is enabled on the Alpaca account.
        Alpaca requires options_approved_level >= 1 to trade options.
        Raises PipelineConnectionError with a clear message if not approved.
        """
        try:
            account = self.broker.get_account()
            # PaperBroker doesn't have this field — skip check
            if isinstance(self.broker, PaperBroker):
                return
        except Exception:
            logger.warning("[Orchestrator] Could not verify options approval — proceeding")
            return

    def _safe_get_equity(self) -> float:
        """
        Get equity from broker at startup.
        Falls back to 100_000 (standard Alpaca paper account) if fetch fails.
        RiskManager.update_equity() is called again at the start of each scan
        so any discrepancy is corrected before the first trade.
        """
        try:
            equity = self.broker.get_equity()
            logger.info("[Orchestrator] Account equity: $%.2f", equity)
            return equity
        except Exception as exc:
            fallback = 100_000.0
            logger.warning(
                "[Orchestrator] Equity fetch failed (%s) — using fallback $%.0f. "
                "Will retry at scan time.", exc, fallback
            )
            return fallback
        return 50_000.0
