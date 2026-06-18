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
  - Jobs fire at configurable times (default: scan at 6:45 AM PT = 9:45 AM ET,
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
from http.server import BaseHTTPRequestHandler, HTTPServer
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
from .regime import RegimeDetector, get_regime_policy, REGIME_POLICY
from .sentiment import SentimentAnalyzer, SentimentConfig
from .metrics import summary as perf_summary
from .adaptive import AdaptiveTuner
from .strategy import BaseStrategy, StrategySignal, get_strategy, STRATEGY_REGISTRY
from .strategy_0dte import ZeroDTEConfig, ZeroDTEStrategy, ZeroDTEMonitor
from .scanner import TickerGate
from .risk_profiles import RiskLevel, RiskProfile, get_risk_profile, apply_profile
from .universe import UniverseBuilder
from .volume_profile import volume_profile_cache
from .stress_testing import run_stress_suite, positions_from_broker
from .sec_signals import (is_entry_confirmed, score_sec_signals,
                           score_sec_with_news, get_dynamic_tickers)
from .confidence_score import ConfidenceScorer

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
    # Full 20-ticker universe: index ETFs, sector ETFs, volatility-sensitive ETFs.
    # All have liquid options markets on Alpaca. The bot ranks and filters daily
    # and trades only the top candidates that pass all signal gates.
    tickers: list[str] = field(default_factory=lambda: [
        # ── Core index ETFs ─────────────────────────────────────────
        "SPY",    # S&P 500 — highest liquidity, 0DTE module
        "QQQ",    # Nasdaq 100 — second-deepest options market
        "IWM",    # Russell 2000 — small-cap, 166 contracts daily
        # ── Fixed income ────────────────────────────────────────────
        "TLT",    # 20-yr Treasuries — rate regime signal, 80 contracts
        # ── Sector ETFs (liquid options, weeklies available) ────────
        "XLF",    # Financials — 51 contracts, rate-sensitive
        "XLK",    # Technology — strong OI
        "XLE",    # Energy — 8M vol, good premium
        "XLV",    # Healthcare — defensive, 71 contracts
        "XLI",    # Industrials — 73 contracts
        # ── Commodity & alternatives ─────────────────────────────────
        "GLD",    # Gold — commodity regime hedge
        "EEM",    # Emerging markets — carries EM risk premium
        "HYG",    # High yield bonds — credit spread indicator
        "SMH",    # Semiconductors — high beta, elevated IV
        # ── Added (higher liquidity than dropped tickers) ───────────
        "VXX",    # VIX short-term futures — fat premiums, crisis hedge
        "XBI",    # Biotech — elevated IV year-round from FDA catalysts
        # ── Removed (thin options markets, failed liquidity filters) ─
        # MDY:  0/40 contracts passed yesterday. No weeklies.
        # XLB:  3M volume, few strikes, no weeklies
        # XLC:  2M volume, bearish tech, thin chains
        # XLRE: 1M volume, almost no options OI
        # XLP:  2M volume, staples — tiny premiums
        # DIA:  Wide spreads, low OI, redundant with SPY
    ])

    # --- Strategy ---
    # Primary strategy — runs first on every ticker in the shortlist
    strategy_name: str = "short_put_spread"   # csp | short_put_spread | short_call_spread | short_strangle
    strategy_config: object = None             # strategy-specific config dataclass

    # Additional strategies — run after primary on the same shortlist each scan.
    # short_call_spread: sells calls when ticker is overbought / near upper BB
    # short_strangle:    sells both call and put on same expiry (neutral premium)
    extra_strategies: list = field(default_factory=lambda: [
        "short_call_spread",   # calls — bearish/overbought setups
        "short_strangle",      # both sides — neutral high-IV setups
    ])

    # --- Risk ---
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # --- Scheduling (hour, minute in PT) ---
    scan_hour: int = 6       # 6:45 AM PT = 9:45 AM ET (15 min after open)
    scan_minute: int = 45
    monitor_interval_minutes: int = 15   # position check frequency
    close_hour: int = 12     # 12:45 PM PT = 3:45 PM ET (15 min before close)
    close_minute: int = 45

    # --- Chain filter ---
    min_dte: int = 14     # widened: captures monthly-only ETF expirations
    max_dte: int = 60
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
    zero_dte_enabled: bool = True            # enable intraday 0DTE GEX scalper
    zero_dte_config: ZeroDTEConfig = field(default_factory=ZeroDTEConfig)

    # --- Risk profile ---
    risk_level: str = "medium"               # "low", "medium", or "high"

    # --- Ticker pre-screening ---
    scanner_enabled: bool = True             # enable bullish + Piotroski pre-filter
    scanner_bullish_threshold: float = 3.0  # min composite score
    scanner_piotroski_threshold: int = 6    # min F-score (0–9)
    scanner_shortlist_top_n: int = 10       # top N by score proceed to chain fetch

    # --- Dynamic universe ---
    universe_enabled: bool = False           # auto-rebuild ticker list weekly
    universe_top_n: int = 30                 # number of tickers in auto universe
    universe_exclude_high_si: bool = True    # exclude high short-interest names

    # --- Broker ---
    paper: bool = True                # always default to paper

    # --- Notifications ---
    discord_webhook_url: str = field(
        default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", "")
    )

    # --- Database ---
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL") or os.getenv("DATABASE_PUBLIC_URL", "")
    )
    sqlite_path: str = "options_bot.db"


# ---------------------------------------------------------------------------
# Health server — /health and /ready for Railway monitoring
# ---------------------------------------------------------------------------

_hs: dict = {"status":"starting","last_scan":None,"last_monitor":None,
             "open_positions":0,"daily_pnl":0.0,"started_at":""}


def _update_health(k: str, v) -> None:
    _hs[k] = v


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] in ("/health", "/ready", "/"):
            import json as _j, datetime as _dt
            ts = _hs["started_at"] or _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
            b = _j.dumps({**_hs, "uptime": int((_dt.datetime.now(tz=_dt.timezone.utc)
                - _dt.datetime.fromisoformat(ts)).total_seconds())}).encode()
            self.send_response(200 if _hs["status"] in ("ready", "running") else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def start_health_server(port: int = 8080) -> None:
    try:
        s = HTTPServer(("0.0.0.0", port), _HealthHandler)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        logger.info("[Health] Listening on :%d /health", port)
    except OSError:
        # Port already bound by the boot server in __main__.py — fine.
        # Module-level _hs dict is shared so _update_health() still works.
        logger.debug("[Health] Port %d already in use — boot server active", port)


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
    Persists FilledOrders to SQLite (local dev) or PostgreSQL (Railway).

    Dual-backend design:
      - SQLite  — uses connection.execute() directly, ? placeholders
      - psycopg2 — requires a cursor, %s placeholders, different introspection

    All SQL is routed through _execute() and _executemany() helpers that
    abstract the cursor/placeholder differences between the two backends.
    """

    def __init__(self, database_url: str = "", sqlite_path: str = "options_bot.db"):
        self._use_pg = bool(database_url and database_url.startswith("postgres"))
        self._database_url = database_url
        self._sqlite_path = sqlite_path
        self._init_schema()

    # ------------------------------------------------------------------
    # Backend abstraction helpers
    # ------------------------------------------------------------------

    def _get_conn(self):
        """Open and return a raw DB connection (caller must close/commit)."""
        if self._use_pg:
            import psycopg2
            return psycopg2.connect(self._database_url)
        return sqlite3.connect(self._sqlite_path)

    def _ph(self) -> str:
        """Return the placeholder character for this backend: %s (pg) or ? (sqlite)."""
        return "%s" if self._use_pg else "?"

    def _execute(self, conn, sql: str, params=()) -> "Any":
        """
        Execute a single SQL statement, returning the cursor.

        Abstracts the psycopg2 cursor vs SQLite direct-execute difference.
        SQL must use ? placeholders — they are rewritten to %s for psycopg2.
        """
        if self._use_pg:
            sql = sql.replace("?", "%s")
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur
        # SQLite: connection.execute() returns a cursor directly
        return conn.execute(sql, params)

    def _get_existing_columns(self, conn) -> set:
        """
        Return the set of column names in the trades table.
        Uses information_schema for PostgreSQL, PRAGMA for SQLite.
        """
        if self._use_pg:
            cur = conn.cursor()
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'trades'
            """)
            return {row[0] for row in cur.fetchall()}
        return {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}

    # ------------------------------------------------------------------
    # Schema init + migration
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        conn = self._get_conn()
        try:
            # PostgreSQL uses TEXT for primary key (same as SQLite).
            # INSERT OR REPLACE is SQLite-only — PostgreSQL uses
            # INSERT ... ON CONFLICT DO UPDATE (upsert syntax).
            if self._use_pg:
                self._execute(conn, """
                    CREATE TABLE IF NOT EXISTS trades (
                        id               TEXT PRIMARY KEY,
                        trade_date       TEXT NOT NULL,
                        strategy         TEXT,
                        underlying       TEXT,
                        legs_json        TEXT,
                        fill_price       REAL,
                        slippage         REAL,
                        max_loss         REAL,
                        hard_stop        REAL,
                        contracts        INTEGER,
                        net_credit       REAL,
                        status           TEXT DEFAULT 'open',
                        close_price      REAL,
                        realized_pnl     REAL,
                        broker           TEXT,
                        created_at       TEXT,
                        updated_at       TEXT,
                        delta                REAL,
                        vega                 REAL,
                        theta                REAL,
                        underlying_price     REAL,
                        expiry               TEXT,
                        profit_target_price  REAL,
                        profit_target_pct    REAL
                    )
                """)
            else:
                self._execute(conn, """
                    CREATE TABLE IF NOT EXISTS trades (
                        id               TEXT PRIMARY KEY,
                        trade_date       TEXT NOT NULL,
                        strategy         TEXT,
                        underlying       TEXT,
                        legs_json        TEXT,
                        fill_price       REAL,
                        slippage         REAL,
                        max_loss         REAL,
                        hard_stop        REAL,
                        contracts        INTEGER,
                        net_credit       REAL,
                        status           TEXT DEFAULT 'open',
                        close_price      REAL,
                        realized_pnl     REAL,
                        broker           TEXT,
                        created_at       TEXT,
                        updated_at       TEXT,
                        delta                REAL,
                        vega                 REAL,
                        theta                REAL,
                        underlying_price     REAL,
                        expiry               TEXT,
                        profit_target_price  REAL,
                        profit_target_pct    REAL
                    )
                """)
            conn.commit()

            # Migration: add columns to existing DBs that predate this schema version
            existing = self._get_existing_columns(conn)
            for col, typedef in [
                ("delta",               "REAL"),
                ("vega",                "REAL"),
                ("theta",               "REAL"),
                ("underlying_price",    "REAL"),
                ("expiry",              "TEXT"),
                ("profit_target_price", "REAL"),
                ("profit_target_pct",   "REAL"),
            ]:
                if col not in existing:
                    self._execute(conn, f"ALTER TABLE trades ADD COLUMN {col} {typedef}")
                    logger.info("[TradeDB] Migration: added column %s %s", col, typedef)
            conn.commit()
        finally:
            conn.close()
        logger.debug("[TradeDB] Schema ready (%s)", "PostgreSQL" if self._use_pg else "SQLite")

    # ------------------------------------------------------------------
    # Strategy name map
    # ------------------------------------------------------------------

    # Map BaseStrategy class names → OrchestratorConfig strategy keys
    # so AdaptiveTuner WHERE strategy=? queries match DB records.
    _STRATEGY_NAME_MAP = {
        "CashSecuredPut":  "csp",
        "ShortPutSpread":  "short_put_spread",
        "ShortStrangle":   "short_strangle",
        "ZeroDTEStrategy": "zero_dte",
    }

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_fill(self, filled: FilledOrder) -> None:
        order = filled.approved_order
        legs_json = json.dumps([
            {"symbol": l.symbol, "side": l.side,
             "strike": l.strike, "qty": l.quantity,
             "expiry": l.expiry.isoformat() if l.expiry else None}
            for l in order.legs
        ])
        _profit_target_price = order.profit_target_price
        _profit_target_pct   = order.profit_target_pct
        now = datetime.now(tz=timezone.utc).isoformat()
        src = order.legs[0] if order.legs else None
        _delta = _vega = _theta = _spot = None
        _expiry_str = None
        if src:
            _expiry_str = src.expiry.isoformat() if hasattr(src, "expiry") and src.expiry else None

        # PostgreSQL uses INSERT ... ON CONFLICT; SQLite uses INSERT OR REPLACE
        if self._use_pg:
            sql = """
                INSERT INTO trades
                (id, trade_date, strategy, underlying, legs_json,
                 fill_price, slippage, max_loss, hard_stop, contracts,
                 net_credit, status, broker, created_at, updated_at,
                 delta, vega, theta, underlying_price, expiry,
                 profit_target_price, profit_target_pct)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO UPDATE SET
                  updated_at = EXCLUDED.updated_at,
                  status     = EXCLUDED.status
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
        params = (
            filled.order_id,
            date.today().isoformat(),
            self._STRATEGY_NAME_MAP.get(order.strategy_name, order.strategy_name),
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
            _delta, _vega, _theta, _spot, _expiry_str,
            _profit_target_price, _profit_target_pct,
        )
        conn = self._get_conn()
        try:
            self._execute(conn, sql, params)
            conn.commit()
            logger.info("[TradeDB] Saved: %s %s", filled.order_id, order.strategy_name)
        except Exception as exc:
            logger.error("[TradeDB] Save failed: %s", exc)
        finally:
            conn.close()

    def update_status(
        self, order_id: str, status: str,
        close_price: Optional[float] = None,
        realized_pnl: Optional[float] = None,
    ) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            self._execute(
                conn,
                "UPDATE trades SET status=?, close_price=?, realized_pnl=?, "
                "updated_at=? WHERE id=?",
                (status, close_price, realized_pnl, now, order_id)
            )
            conn.commit()
        except Exception as exc:
            logger.error("[TradeDB] Update failed for %s: %s", order_id, exc)
        finally:
            conn.close()

    def update_greeks(
        self,
        order_id: str,
        delta: float,
        vega: float,
        theta: float,
        underlying_price: float,
    ) -> None:
        """Backfill Greeks on an open trade after the first monitor cycle snapshot."""
        now = datetime.now(tz=timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            self._execute(
                conn,
                "UPDATE trades SET delta=?, vega=?, theta=?, underlying_price=?, "
                "updated_at=? WHERE id=? AND status='open'",
                (delta, vega, theta, underlying_price, now, order_id)
            )
            conn.commit()
        except Exception as exc:
            logger.error("[TradeDB] update_greeks failed for %s: %s", order_id, exc)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_open_trades(self) -> list[dict]:
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT id, strategy, underlying, hard_stop, contracts, "
                "net_credit, fill_price, legs_json, "
                "delta, vega, theta, underlying_price, expiry, "
                "profit_target_price, profit_target_pct "
                "FROM trades WHERE status='open'"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("[TradeDB] get_open_trades failed: %s", exc)
            return []
        finally:
            conn.close()

    def get_all_closed_pnls(self) -> list[float]:
        """Return all realized_pnl values for closed trades.
        Used by ConfidenceScorer track record section."""
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT realized_pnl FROM trades "
                "WHERE status IN ('closed','closed_expiry',"
                "'profit_target','stop_hit') "
                "AND realized_pnl IS NOT NULL"
            )
            return [float(row[0]) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("[TradeDB] get_all_closed_pnls failed: %s", exc)
            return []
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Discord notifier
# ---------------------------------------------------------------------------

def send_discord(webhook_url: str, message: str) -> None:
    """
    POST a message to a Discord webhook.
    Fails silently — never let a notification failure crash the trading loop.

    Normalises discordapp.com → discord.com automatically. Discord's old
    domain (discordapp.com) redirects to discord.com, but Railway's egress
    allowlist only permits discord.com, so the redirect is never followed.
    """
    if not webhook_url:
        return
    # Normalise legacy discordapp.com → discord.com
    url = webhook_url.replace("discordapp.com", "discord.com")
    try:
        payload = json.dumps({"content": message}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
        logger.debug("[Discord] Message sent")
    except Exception as exc:
        logger.warning("[Discord] Send failed (non-fatal): %s", exc)


def _classify_vol_regime_inline(
    iv_rank: Optional[float],
    skew_ratio: Optional[float],
    term_slope: Optional[float],
) -> Optional[dict]:
    """
    Classify vol regime from per-contract signal data.
    Adapted from Quantops/options_vol_regime.py classify_vol_regime().

    Returns dict with premium/skew/term regime + favored strategy list,
    or None if insufficient data.
    """
    if iv_rank is None:
        return None

    # Premium regime
    if iv_rank >= 75:
        premium = "rich"
    elif iv_rank <= 25:
        premium = "cheap"
    else:
        premium = "neutral"

    # Skew regime
    if skew_ratio is not None:
        if skew_ratio >= 1.30:
            skew = "steep_put"
        elif skew_ratio <= 0.85:
            skew = "steep_call"
        else:
            skew = "neutral"
    else:
        skew = "neutral"

    # Term structure
    if term_slope is not None:
        if term_slope >= 0.02:
            term = "contango"
        elif term_slope <= -0.02:
            term = "backwardation"
        else:
            term = "flat"
    else:
        term = "flat"

    # Strategy routing
    favored = []
    if premium == "rich":
        if skew == "steep_put":
            favored = ["short_put_spread", "iron_condor"]
        elif skew == "neutral":
            favored = ["short_put_spread", "short_strangle"]
        else:
            favored = ["short_strangle"]
    elif premium == "cheap":
        favored = ["csp"]   # collect what's available; avoid spreads
    else:
        favored = ["short_put_spread"]

    return {
        "premium": premium,
        "skew":    skew,
        "term":    term,
        "favored": ", ".join(favored),
    }


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
        regime_detector: Optional[RegimeDetector] = None,
    ):
        self.config = config
        self.rm = risk_manager
        self.broker = broker
        self.db = db
        self.state = state
        self._regime    = regime_detector or RegimeDetector()
        self.enricher   = GreeksEnricher()   # fetches Treasury rate on init
        self.strategy   = get_strategy(config.strategy_name, config.strategy_config)
        self._sentiment = SentimentAnalyzer(config=SentimentConfig())
        self._confidence = ConfidenceScorer(
            regime_detector=self._regime,
            db=db,
            risk_manager=risk_manager,
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

        # --- Step 2c: Sentiment gate ---
        # VADER (vaderSentiment, ~1MB, no GPU) runs as the default on Railway.
        # FinBERT runs automatically if torch is locally installed.
        # SELL signal blocks entry; HOLD and BUY allow through.
        # Non-fatal: news fetch failure defaults to HOLD (never blocks).
        try:
            sentiment_signals = self._sentiment.get_signals(
                tickers=[ticker],
                config=self._sentiment.config,
            )
            sig = sentiment_signals.get(ticker)
            if sig and sig.signal == "SELL":
                logger.info(
                    "[Pipeline] %s sentiment SELL — skipping "
                    "(score=%.3f, articles=%d, model=%s)",
                    ticker, sig.weighted_score, sig.article_count, sig.model_used,
                )
                return None
            if sig:
                logger.debug(
                    "[Pipeline] %s sentiment %s (score=%.3f, articles=%d, model=%s)",
                    ticker, sig.signal, sig.weighted_score,
                    sig.article_count, sig.model_used,
                )
        except Exception as exc:
            logger.debug("[Pipeline] %s sentiment gate skipped (non-fatal): %s", ticker, exc)

        # --- Step 2d: SEC insider signal (positive confirmation gate) ---
        # Never blocks a trade — only logs when insider buying is confirmed.
        # Form 4 XML parsed for type=P (open-market purchases) only.
        # Non-fatal: EDGAR unavailable silently allows through.
        try:
            confirmed, sec_reason = is_entry_confirmed(ticker, require_score=20)
            if confirmed:
                logger.info("[Pipeline] %s SEC confirmed: %s", ticker, sec_reason)
            else:
                logger.debug("[Pipeline] %s SEC: %s", ticker, sec_reason)
        except Exception as exc:
            logger.debug("[Pipeline] %s SEC gate skipped (non-fatal): %s", ticker, exc)

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

        # --- Step 4b: Confidence scoring ---
        try:
            _sig_obj = None
            try:
                _sig_obj = sentiment_signals.get(ticker)
            except Exception:
                pass
            # Pass news sentiment into SEC scoring for congress+news confirmation
            _news_sig   = _sig_obj.signal         if _sig_obj else None
            _news_score = _sig_obj.weighted_score  if _sig_obj else None
            _sec_data_full = score_sec_with_news(
                ticker,
                news_signal=_news_sig,
                news_score=_news_score,
            )
            if _sec_data_full.get("news_confirmed") is True:
                logger.info(
                    "[Pipeline] %s ⭐ Congress+News aligned: buys=%d news=%s sec_score=%d",
                    ticker, _sec_data_full.get("congress_buys",0),
                    _news_sig, _sec_data_full.get("score",0),
                )
            _confidence_report = self._confidence.score(
                signal=signal,
                ticker=ticker,
                sec_data=_sec_data_full,
                open_trades=open_trades,
                sentiment_allowed=_sig_obj.signal != "SELL" if _sig_obj else True,
                sentiment_compound=_sig_obj.weighted_score if _sig_obj else 0.0,
            )
            if not _confidence_report.should_trade:
                logger.info(
                    "[Pipeline] %s confidence too low: %.0f [%s] — skip  %s",
                    ticker, _confidence_report.overall, _confidence_report.grade,
                    _confidence_report.short_line(),
                )
                return None
            logger.info(
                "[Pipeline] %s confidence %.0f [%s]",
                ticker, _confidence_report.overall, _confidence_report.grade,
            )
        except Exception as _conf_exc:
            logger.debug("[Pipeline] Confidence scoring non-fatal: %s", _conf_exc)
            _confidence_report = None

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
        # Deduct committed max-loss from working equity so the next ticker
        # in the same scan is sized against the remaining risk budget.
        self.rm.update_equity_after_fill(order.max_loss_dollars)
        self.db.save_fill(filled)
        with self.state._lock:
            self.state.filled_today.append(filled)

        _conf_line = _confidence_report.short_line() if _confidence_report else ""
        # Vol regime classification (skew-aware strategy routing)
        # Adapted from Quantops/options_vol_regime.py
        try:
            _iv_rank = getattr(signal, "iv_rank", None)
            _iv_skew = getattr(signal, "iv_skew", None)    # put_iv/call_iv ratio
            _term_sl = getattr(signal, "term_slope", None) # back_iv - front_iv
            _vol_regime = _classify_vol_regime_inline(
                iv_rank=_iv_rank, skew_ratio=_iv_skew, term_slope=_term_sl
            )
            if _vol_regime:
                logger.info(
                    "[Pipeline] %s vol regime: premium=%s skew=%s term=%s → %s",
                    ticker, _vol_regime["premium"], _vol_regime["skew"],
                    _vol_regime["term"], _vol_regime["favored"],
                )
        except Exception:
            pass

        msg = _format_signal_message(signal, filled, regime_name, regime_options_weight)
        if _conf_line:
            msg = msg + "\n" + _conf_line
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
        risk_manager=None,
    ):
        self.broker = broker
        self.db = db
        self.state = state
        self.discord_webhook_url = discord_webhook_url
        self.rm = risk_manager  # used to update daily P&L for loss-halt enforcement

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

        # Fetch latest quotes — explicit rate-limit and disconnect logging
        try:
            quotes = self.broker.get_latest_quotes(all_symbols)
        except PipelineConnectionError as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "too many" in err_str or "rate" in err_str:
                logger.warning(
                    "[Monitor] ⚠ RATE LIMIT (HTTP 429) — Alpaca throttled quote fetch "
                    "for %d symbols. Will retry next monitor cycle (%ds). "
                    "If this persists, reduce monitor_poll_seconds in OrchestratorConfig.",
                    len(all_symbols), self.config.monitor_interval_minutes * 60,
                )
            elif "connect" in err_str or "timeout" in err_str or "network" in err_str:
                logger.warning(
                    "[Monitor] ⚠ CONNECTION LOST — Quote fetch timed out or network "
                    "dropped for %d symbols (%s). Positions unmonitored this cycle — "
                    "stops are exchange-managed via hard_stop_price on single-leg orders.",
                    len(all_symbols), exc,
                )
            else:
                logger.warning(
                    "[Monitor] Quote fetch failed for %d symbols: %s",
                    len(all_symbols), exc,
                )
            return

        # Backfill Greeks for any open trades that still have NULL delta/vega/theta.
        # Uses get_option_snapshots() which returns Greeks from Alpaca's Black-Scholes model.
        # This runs on every monitor cycle but the DB write is skipped if Greeks already set.
        trades_needing_greeks = [
            t for t in open_trades
            if t.get("delta") is None or t.get("vega") is None
        ]
        if trades_needing_greeks:
            try:
                snapshots = self.broker.get_option_snapshots(all_symbols)
                for trade in trades_needing_greeks:
                    legs = json.loads(trade.get("legs_json", "[]"))
                    if not legs:
                        continue

                    # Compute NET Greeks across all legs.
                    # For spreads: short put delta + long put delta (long delta is positive,
                    # partially offsetting short delta — net is smaller magnitude).
                    # For strangles: short put delta + short call delta (both contribute).
                    # Side convention: sell_to_open legs contribute negative delta (short),
                    # buy_to_open legs contribute positive delta (long hedge).
                    net_delta = 0.0
                    net_vega  = 0.0
                    net_theta = 0.0
                    spot      = float(trade.get("underlying_price") or 0)
                    any_valid = False

                    for leg in legs:
                        sym  = leg.get("symbol", "")
                        snap = snapshots.get(sym, {})
                        d    = snap.get("delta")
                        v    = snap.get("vega")
                        t    = snap.get("theta")
                        if d is None:
                            continue
                        any_valid = True
                        # sell_to_open = short position: flip sign convention so
                        # portfolio delta reflects net directional exposure
                        if "sell" in leg.get("side", ""):
                            net_delta += d          # short put: d is already negative
                            net_vega  += (v or 0)   # short options: vega is negative
                            net_theta += (t or 0)   # short options: theta is positive
                        else:
                            net_delta += d          # long put hedge: d is negative but smaller
                            net_vega  += (v or 0)
                            net_theta += (t or 0)
                        # Use spot from any snapshot that has it
                        if spot == 0 and snap.get("underlying_price"):
                            spot = float(snap["underlying_price"])

                    if any_valid:
                        self.db.update_greeks(
                            order_id=trade["id"],
                            delta=round(net_delta, 6),
                            vega=round(net_vega, 6),
                            theta=round(net_theta, 6),
                            underlying_price=spot,
                        )
                        logger.debug(
                            "[Monitor] Greeks backfilled for %s: "
                            "net_delta=%.4f net_vega=%.4f net_theta=%.4f (%d legs)",
                            trade["id"], net_delta, net_vega, net_theta, len(legs),
                        )
            except Exception as exc:
                logger.debug("[Monitor] Greek backfill failed (non-fatal): %s", exc)

        # Check each trade
        for trade in open_trades:
            self._check_trade(trade, quotes)

        # Update unrealized P&L in session state and RiskManager
        positions = self.broker.get_positions()
        total_unreal = sum(p.get("unrealized_pl", 0) for p in positions)
        with self.state._lock:
            self.state.open_positions = positions
            self.state.daily_unrealized_pnl = total_unreal
        # Keep RiskManager in sync so daily loss halt sees current unrealized exposure
        if self.rm is not None:
            self.rm.record_pnl(unrealized=total_unreal)
        _update_health("open_positions", len(positions))
        _update_health("daily_pnl", round(self.state.daily_realized_pnl + total_unreal, 2))
        _update_health("last_monitor", datetime.now(tz=timezone.utc).isoformat())

    def _check_trade(self, trade: dict, quotes: dict) -> None:
        """
        Check one trade against its stop-loss and profit target.
        Closes the position if either level is hit.
        """
        try:
            legs = json.loads(trade.get("legs_json", "[]"))
            hard_stop = float(trade.get("hard_stop", 0))
            entry_credit = abs(float(trade.get("net_credit") or 0))
            # Use stored profit_target_price if available (set from strategy config).
            # Falls back to 50% of credit if not stored (legacy trades).
            stored_target = trade.get("profit_target_price")
            if stored_target is not None:
                profit_target = float(stored_target)
            elif entry_credit > 0:
                profit_target = entry_credit * 0.50
            else:
                profit_target = None

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
            # Update RiskManager daily P&L so the daily loss halt can fire
            if self.rm is not None:
                self.rm.record_pnl(realized=realized_pnl)

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

        # Pipeline and monitor — constructed below after regime_detector is ready

        # Apply risk profile — overrides relevant config fields.
        # strategy_config must be instantiated first so apply_profile() can
        # set stop_multiplier, min_pop, min_credit, min_dte, max_dte on it.
        # Without this, profile only sets orchestrator-level fields (scan times etc).
        if config.strategy_config is None and config.strategy_name in STRATEGY_REGISTRY:
            config.strategy_config = STRATEGY_REGISTRY[config.strategy_name]().config
            logger.debug(
                "[Orchestrator] Instantiated default strategy_config for '%s' "
                "so risk profile can apply strategy-level parameters.",
                config.strategy_name,
            )

        if config.risk_level and config.risk_level != "custom":
            apply_profile(config, config.risk_level)
            logger.info("[Orchestrator] Risk profile: %s applied (strategy params included)",
                        config.risk_level.upper())

        _hs["started_at"] = datetime.now(tz=timezone.utc).isoformat()
        start_health_server(int(os.getenv("HEALTH_PORT", "8080")))
        _update_health("status", "ready")

        # Ticker pre-screening gate
        self.ticker_gate = TickerGate(
            bullish_threshold=config.scanner_bullish_threshold,
            piotroski_threshold=config.scanner_piotroski_threshold,
            require_bullish=config.scanner_enabled,
            require_piotroski=config.scanner_enabled,
        ) if config.scanner_enabled else None

        # Dynamic universe builder (weekly refresh)
        self.universe_builder = UniverseBuilder() if config.universe_enabled else None

        # Regime detector — replaces simple VIX threshold
        self.regime_detector = RegimeDetector(
            cache_ttl_seconds=config.regime_cache_ttl
        )

        # Pipeline and monitor (needs self.regime_detector, so constructed here)
        self.pipeline = TradingPipeline(
            config=config,
            risk_manager=self.rm,
            broker=self.broker,
            db=self.db,
            state=self.state,
            regime_detector=self.regime_detector,
        )
        self.monitor = PositionMonitor(
            broker=self.broker,
            db=self.db,
            state=self.state,
            discord_webhook_url=config.discord_webhook_url,
            risk_manager=self.rm,
        )

        # Sentiment analyzer — owned by TradingPipeline._sentiment to avoid
        # double-initialisation. Expose as property for external callers.
        # (TradingPipeline.__init__ already creates SentimentAnalyzer above)
        self.sentiment_analyzer = self.pipeline._sentiment

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

        # Weekly universe rebuild: Sunday 6 PM PT (before Monday open)
        if self.universe_builder is not None:
            scheduler.add_job(
                self._rebuild_universe,
                CronTrigger(
                    day_of_week="sun",
                    hour=18,
                    minute=0,
                    timezone="America/Los_Angeles",
                ),
                id="universe_rebuild",
                name="Weekly universe rebuild",
            )
            logger.info("[Orchestrator] Universe rebuild scheduled: Sundays 6 PM PT")

        et_scan = (self.config.scan_hour + 3) % 24
        et_eod  = (self.config.close_hour + 3) % 24
        logger.info(
            "[Orchestrator] Scheduler starting: "
            "scan=%02d:%02d PT (%02d:%02d ET), "
            "monitor=every %dmin, "
            "eod=%02d:%02d PT (%02d:%02d ET)",
            self.config.scan_hour, self.config.scan_minute,
            et_scan, self.config.scan_minute,
            self.config.monitor_interval_minutes,
            self.config.close_hour, self.config.close_minute,
            et_eod, self.config.close_minute,
        )
        # Startup announcement — sent at scheduler start so the env var
        # has had time to be fully injected (avoids the boot-timing issue)
        _strategies = [self.config.strategy_name] + (
            getattr(self.config, "extra_strategies", None) or []
        )
        send_discord(
            self.config.discord_webhook_url,
            f"🟢 **Options Bot started** — "
            f"{'PAPER' if self.config.paper else 'LIVE'} mode\n"
            f"**Tickers:** {', '.join(self.config.tickers)}\n"
            f"**Strategies:** {', '.join(s.upper() for s in _strategies)}\n"
            f"**Scan:** {self.config.scan_hour:02d}:{self.config.scan_minute:02d} PT "
            f"= {et_scan:02d}:{self.config.scan_minute:02d} ET | "
            f"**Risk:** {self.config.risk_config.risk_budget_pct:.0%}/trade",
        )
        send_discord(
            self.config.discord_webhook_url,
            f"🟢 **Options Bot started** — {self.config.strategy_name.upper()} "
            f"{'[PAPER]' if self.config.paper else '[LIVE]'}\n"
            f"Tickers ({len(self.config.tickers)}): {', '.join(self.config.tickers)}\n"
            f"Scan: {self.config.scan_hour:02d}:{self.config.scan_minute:02d} PT "
            f"({et_scan:02d}:{self.config.scan_minute:02d} ET) daily\n"
            f"EOD:  {self.config.close_hour:02d}:{self.config.close_minute:02d} PT "
            f"({et_eod:02d}:{self.config.close_minute:02d} ET) daily\n"
            f"Risk: {self.config.risk_config.risk_budget_pct:.1%}/trade | "
            f"Max {self.config.risk_config.max_trades_per_day} trades/day | "
            f"Halt at -{self.config.risk_config.max_daily_loss_pct:.0%} daily loss"
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("[Orchestrator] Shutting down")
            scheduler.shutdown(wait=False)

    def _rebuild_universe(self) -> None:
        """
        Called every Sunday at 6 PM PT by APScheduler.
        Rebuilds the ticker universe from iShares ETF constituents and
        sends a Discord summary of the new watchlist.
        """
        if self.universe_builder is None:
            return
        try:
            tickers = self.universe_builder.build_for_strategy(
                strategy=self.config.strategy_name,
                top_n=self.config.universe_top_n,
                exclude_high_si=self.config.universe_exclude_high_si,
            )
            if tickers:
                # Store rebuilt universe for this week's scans
                self.config.tickers = tickers
                logger.info("[Universe] Rebuilt: %d tickers", len(tickers))
                send_discord(
                    self.config.discord_webhook_url,
                    f"🌎 **Weekly universe rebuilt** — {len(tickers)} tickers\n"
                    f"Top 10: {', '.join(tickers[:10])}"
                    + (f" +{len(tickers)-10} more" if len(tickers) > 10 else "")
                )
        except Exception as exc:
            logger.error("[Universe] Weekly rebuild failed: %s", exc)

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
                        # Grace window: don't mark fresh fills as closed_external.
                        # Alpaca positions can take 5-30s to appear after submission.
                        # Only mark closed_external if the fill is older than 15 minutes.
                        from datetime import datetime, timezone as _tz
                        created_at = trade.get("created_at", "")
                        is_fresh = False
                        if created_at:
                            try:
                                age_seconds = (
                                    datetime.now(tz=_tz.utc) -
                                    datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                                ).total_seconds()
                                is_fresh = age_seconds < 900  # 15-minute grace window
                            except Exception:
                                pass
                        if is_fresh:
                            logger.debug(
                                "[Reconcile] Trade %s is fresh (<15 min) — skipping closure",
                                trade["id"]
                            )
                        else:
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

        # Apply regime policy — single source of truth for regime-driven constraints
        policy = get_regime_policy(regime_name)

        if policy.block_new_entries:
            msg = (
                f"🔴 **REGIME GATE: Trading halted** — "
                f"regime={regime_name.upper()} | "
                f"VIX={indicators.get('vix_level', 0):.1f} | "
                f"options_weight={options_weight:.0%}\n"
                f"_All new entries blocked until regime improves._"
            )
            logger.warning("[Orchestrator] %s", msg)
            send_discord(self.config.discord_webhook_url, msg)
            return []

        if options_weight < self.config.regime_min_options_weight:
            msg = (
                f"⚠️ **REGIME GATE: No new trades today** — "
                f"regime={regime_name.upper()} | "
                f"options_weight={options_weight:.0%} "
                f"(min={self.config.regime_min_options_weight:.0%}) | "
                f"VIX={indicators.get('vix_level', 0):.1f} | "
                f"curve={indicators.get('yield_curve_slope', 0):.2f} | "
                f"Hurst={indicators.get('hurst', 0):.3f}\n"
                f"_Bot is protecting capital — will resume when conditions improve._"
            )
            logger.warning("[Orchestrator] %s", msg)
            send_discord(self.config.discord_webhook_url, msg)
            return []

        # Log regime policy being applied
        logger.info(
            "[Orchestrator] Regime policy: size×%.2f max_trades=%d "
            "conf_boost=%d favored=%s",
            policy.size_multiplier, policy.max_trades_per_scan,
            policy.min_confidence_boost, policy.favored_strategy,
        )

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

        # Correlation warning — logs Herfindahl N_eff and flags concentrated groups
        if option_positions:
            open_underlyings = [p.get("symbol", "")[:6] for p in option_positions]
            self.rm.warn_correlation(open_underlyings)

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

        # Congress+news dynamic tickers — discovered at scan time, not static config
        # Only tickers confirmed by BOTH sources are included (intersection gate)
        try:
            _dynamic = get_dynamic_tickers(max_tickers=5)
        except Exception as _dyn_exc:
            _dynamic = []
            logger.debug("[Orchestrator] Dynamic ticker fetch failed: %s", _dyn_exc)

        # Dynamic universe rebuild (if enabled, replaces config.tickers for this scan)
        scan_tickers = self.config.tickers
        if self.universe_builder is not None:
            try:
                rebuilt = self.universe_builder.build_for_strategy(
                    strategy=self.config.strategy_name,
                    top_n=self.config.universe_top_n,
                    exclude_high_si=self.config.universe_exclude_high_si,
                )
                if rebuilt:
                    scan_tickers = rebuilt
                    logger.info("[Orchestrator] Dynamic universe: %d tickers", len(rebuilt))
            except Exception as exc:
                logger.warning("[Orchestrator] Universe rebuild failed (using config list): %s", exc)

        # Score all universe tickers, take top N by composite score.
        # Chain fetching only runs on the shortlist — avoids 20 slow yfinance
        # options chain calls when most tickers would be rejected anyway.
        if self.ticker_gate is not None:
            scan_tickers = self.ticker_gate.filter_ranked(
                scan_tickers,
                top_n=self.config.scanner_shortlist_top_n,
            )
            if not scan_tickers:
                logger.info("[Orchestrator] All tickers blocked by pre-screening gate — no scan")
                return []

        # Append congress+news confirmed individual stocks to scan list
        # These bypass the BullishScanner gate (they have their own signal gate)
        # but still go through the full pipeline: IV, Greeks, risk, execution
        if _dynamic:
            new_stocks = [t for t in _dynamic if t not in scan_tickers]
            if new_stocks:
                scan_tickers = list(scan_tickers) + new_stocks
                logger.info(
                    "[Orchestrator] +%d dynamic stock(s) appended to scan: %s",
                    len(new_stocks), ", ".join(new_stocks),
                )

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

        # Multi-strategy pass — run extra_strategies against same shortlist
        # Each extra strategy uses its own default config, independent of the
        # main strategy. Respects daily position caps across all strategies.
        for extra_name in (self.config.extra_strategies or []):
            if extra_name == self.config.strategy_name:
                continue  # already ran this one above

            # Check positions cap before starting next strategy pass
            positions = self.broker.get_positions()
            current_count = len([p for p in positions if p.get("asset_class") == "us_option"])
            if current_count >= self.config.max_positions_total:
                logger.info(
                    "[Orchestrator] Max positions reached — skipping extra strategy %s",
                    extra_name,
                )
                break

            logger.info("[Orchestrator] === EXTRA STRATEGY PASS: %s ===", extra_name.upper())
            try:
                # Build a fresh pipeline using the extra strategy
                extra_cfg = get_strategy(extra_name, None).config                     if hasattr(get_strategy(extra_name, None), "config") else None
                self.pipeline.strategy = get_strategy(extra_name, extra_cfg)

                for ticker in scan_tickers:
                    positions = self.broker.get_positions()
                    current_count = len([
                        p for p in positions if p.get("asset_class") == "us_option"
                    ])
                    if current_count >= self.config.max_positions_total:
                        break

                    try:
                        filled = self.pipeline.run_for_ticker(
                            ticker=ticker,
                            regime=regime,
                            regime_name=regime_name,
                            regime_options_weight=options_weight,
                            open_trades=self.db.get_open_trades(),
                            sentiment_signals=sentiment_signals,
                        )
                        if filled:
                            filled_orders.append(filled)
                            logger.info(
                                "[Orchestrator] %s (%s) → FILLED",
                                ticker, extra_name,
                            )
                    except Exception as exc:
                        logger.warning(
                            "[Orchestrator] %s (%s) pipeline error: %s",
                            ticker, extra_name, exc,
                        )
                        self.state.record_error(str(exc))

            except Exception as exc:
                logger.warning("[Orchestrator] Extra strategy %s failed: %s", extra_name, exc)
            finally:
                # Restore primary strategy
                self.pipeline.strategy = get_strategy(
                    self.config.strategy_name, self.config.strategy_config
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

        # Fetch fresh equity from broker so stress test NLV reflects intraday P&L.
        # self.state.equity is stale (set at 6:45 AM PT / 9:45 AM ET scan time).
        try:
            equity = self.broker.get_equity()
            self.state.equity = equity
            self.rm.update_equity(equity)
        except Exception as exc:
            equity = self.state.equity if self.state.equity > 0 else self.rm.equity
            logger.warning("[EOD] Equity fetch failed, using cached $%.2f: %s", equity, exc)

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

        # Run stress test suite against current open positions
        stress_line = ""
        try:
            stress_positions = positions_from_broker(self.broker, self.db)
            if stress_positions:
                suite = run_stress_suite(
                    stress_positions,
                    account_nlv=self.state.equity if self.state.equity > 0 else self.rm.equity,
                    drawdown_threshold=self.config.risk_config.max_daily_loss_pct
                    if hasattr(self.config, 'risk_config') else 0.10,
                )
                stress_line = suite.discord_line + "\n"
                if not suite.survives_all:
                    logger.warning(
                        "[EOD] Stress test: portfolio does NOT survive all scenarios. "
                        "Worst: %s (%.1f%% NLV). Review positions.",
                        suite.worst_scenario, suite.worst_impact_pct,
                    )
        except Exception as exc:
            logger.warning("[EOD] Stress test failed (non-fatal): %s", exc)
        tuning_line = ""
        if self.tuner and self.tuner.adjustment_history():
            today_adj = [
                a for a in self.tuner.adjustment_history()
                if a["applied_at"][:10] == today
            ]
            if today_adj:
                tuning_line = f"Tuning adjustments today: {len(today_adj)}\n"

        week_start = date.today() - timedelta(days=date.today().weekday())
        wtd_pnls   = self._query_pnls_since(week_start)
        wtd_pnl    = sum(wtd_pnls)
        n_total    = len(self._query_all_pnls())
        _eod_conf_line = ""
        try:
            from .confidence_score import _score_regime, _score_risk_posture, _score_track_record
            _s_r = _score_regime(regime)
            _dpnl = self.state.daily_realized_pnl / max(self.state.equity or 1.0, 1.0)
            _s_rp = _score_risk_posture(
                daily_pnl_pct=_dpnl,
                max_daily_loss_pct=getattr(self.config.risk_config, "max_daily_loss_pct", 0.05),
                open_positions=len(open_trades),
                max_positions=getattr(self.config, "max_positions_total", 5),
                risk_budget_used=len(self.state.filled_today) / max(
                    getattr(self.config.risk_config, "max_trades_per_day", 5), 1),
            )
            _wins_e  = [p for p in all_pnls if p > 0]
            _losses_e = [abs(p) for p in all_pnls if p < 0]
            _wr_e = (len(_wins_e) / len(all_pnls)) if len(all_pnls) >= 10 else None
            _pf_e = (sum(_wins_e) / sum(_losses_e)) if _losses_e else None
            _s_tr = _score_track_record(_wr_e, _pf_e, len(all_pnls))
            _sys = round(_s_r.score*0.40 + _s_rp.score*0.40 + _s_tr.score*0.20, 1)
            _grd = ("VERY HIGH" if _sys>=90 else "HIGH" if _sys>=75 else
                    "MODERATE" if _sys>=60 else "LOW" if _sys>=45 else "VERY LOW")
            _em = "\U0001f7e2" if _sys>=75 else "\U0001f7e1" if _sys>=60 else "\U0001f534"
            _eod_conf_line = (
                f"{_em} System confidence: `{_sys:.0f}/100` [{_grd}]"
                f"  regime={_s_r.score:.0f}  risk={_s_rp.score:.0f}"
                f"  track={_s_tr.score:.0f}\n"
            )
        except Exception as _eod_ce:
            logger.debug("[EOD] Confidence snapshot non-fatal: %s", _eod_ce)

        summary = (
            f"Daily Summary - {date.today()}\n"
            f"Strategy: {self.config.strategy_name.upper()}  "
            f"{'[PAPER]' if self.config.paper else '[LIVE]'}\n"
            f"Trades today: {len(self.state.filled_today)}  |  Open: {len(open_trades)}\n"
            f"Today P&L: ${self.state.daily_realized_pnl:+.2f} realized  "
            f"${self.state.daily_unrealized_pnl:+.2f} unrealized\n"
            f"Week-to-date: ${wtd_pnl:+.2f} ({len(wtd_pnls)} closed trades)\n"
            f"Total trades: {n_total}/30 to walk-forward unlock\n"
            f"{metrics_line}"
            f"{stress_line}"
            f"{tuning_line}"
            f"{_eod_conf_line}"
            f"Regime: {regime['regime'].upper()} "
            f"(conf={regime['confidence']:.0%}, "
            f"VIX={regime['indicators'].get('vix_level',0):.1f}, "
            f"Hurst={hurst_val:.3f} [{hurst_reg}])\n"
            f"Errors: {len(self.state.errors_today)}"
        )
        logger.info("[Orchestrator] %s", summary.replace("\n", " | "))
        send_discord(self.config.discord_webhook_url, summary)

        # Milestone + periodic summaries
        self._check_milestones()
        if date.today().weekday() == 4:  # Friday
            self._send_weekly_summary()
        tomorrow = date.today() + timedelta(days=1)
        if tomorrow.month != date.today().month:
            self._send_monthly_summary()

    def _query_all_pnls(self):
        try:
            with self.db._get_conn() as conn:
                cur = conn.execute(
                    "SELECT realized_pnl FROM trades "
                    "WHERE status NOT IN ('open') AND realized_pnl IS NOT NULL"
                )
                return [row[0] for row in cur.fetchall()]
        except Exception:
            return []

    def _query_pnls_since(self, since_date):
        try:
            with self.db._get_conn() as conn:
                cur = conn.execute(
                    "SELECT realized_pnl FROM trades "
                    "WHERE status NOT IN ('open') AND realized_pnl IS NOT NULL "
                    "AND trade_date >= ?",
                    (since_date.isoformat(),)
                )
                return [row[0] for row in cur.fetchall()]
        except Exception:
            return []

    def _pnl_block(self, pnls, label):
        NL = chr(10)
        if not pnls:
            return label + ": no closed trades yet" + NL
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total  = sum(pnls)
        wr     = len(wins) / len(pnls) * 100
        avg_w  = sum(wins)   / len(wins)   if wins   else 0.0
        avg_l  = sum(losses) / len(losses) if losses else 0.0
        pf     = sum(wins) / abs(sum(losses)) if losses else 999.0
        return (
            "**" + label + "** (" + str(len(pnls)) + " trades)" + NL +
            "Total P&L: **$" + "{:+.2f}".format(total) + "**  "
            "Win rate: **" + "{:.0f}".format(wr) + "%**" + NL +
            "Avg win: $" + "{:+.2f}".format(avg_w) + "  "
            "Avg loss: $" + "{:+.2f}".format(avg_l) + "  "
            "Profit factor: " + "{:.2f}".format(pf) + NL
        )

    def _check_milestones(self):
        all_pnls = self._query_all_pnls()
        n = len(all_pnls)
        NL = chr(10)
        MILESTONES = {
            10: ("Milestone: 10 trades closed",
                 "Adaptive tuner has initial data. 20 more to walk-forward unlock."),
            20: ("Milestone: 20 trades closed",
                 "10 more trades until the adaptive tuner walk-forward gate opens."),
            30: ("Milestone: 30 trades - WALK-FORWARD UNLOCKED",
                 "The tuner can now make statistically valid parameter adjustments."),
            60: ("Milestone: 60 trades (~2 months)",
                 "At least one full market regime cycle covered. Review performance before going live."),
            90: ("Milestone: 90 trades (~3 months)",
                 "Breadth weights can now be calibrated. Final go-live assessment."),
        }
        if n not in MILESTONES:
            return
        title, detail = MILESTONES[n]
        wins   = [p for p in all_pnls if p > 0]
        losses = [p for p in all_pnls if p <= 0]
        wr     = len(wins) / n * 100 if n else 0
        pf     = sum(wins) / abs(sum(losses)) if losses else 999.0
        pnl_block = self._pnl_block(all_pnls, "All-time")
        readiness = ""
        if n >= 30:
            checks = [
                ("Win rate >= 60%",      wr >= 60,  "{:.0f}%".format(wr)),
                ("Profit factor >= 1.3", pf >= 1.3, "{:.2f}".format(pf)),
            ]
            lines_r = ["Go-live readiness:"]
            for chk, passed, val in checks:
                lines_r.append(("OK " if passed else "X  ") + chk + ": " + val)
            all_pass = all(p for _, p, _ in checks)
            lines_r.append("READY FOR LIVE TRADING" if all_pass else "Continue paper trading")
            readiness = NL.join(lines_r) + NL
        msg = title + NL + detail + NL + NL + pnl_block + readiness
        send_discord(self.config.discord_webhook_url, msg)

    def _send_weekly_summary(self):
        NL = chr(10)
        week_start = date.today() - timedelta(days=4)
        pnls      = self._query_pnls_since(week_start)
        all_pnls  = self._query_all_pnls()
        n_all     = len(all_pnls)
        equity    = self.state.equity or 100_000.0
        bar_filled = min(int(n_all / 3), 10)
        bar = ("X" * bar_filled) + ("-" * (10 - bar_filled))
        unlock_status = "UNLOCKED" if n_all >= 30 else (str(30 - n_all) + " to unlock")
        msg = (
            "Weekly Summary - " + str(date.today()) + NL +
            "Account equity: $" + "{:,.2f}".format(equity) + NL + NL +
            self._pnl_block(pnls, "This week") + NL +
            self._pnl_block(all_pnls, "All-time") + NL +
            "Paper progress: [" + bar + "] " + str(n_all) + "/30 trades (" + unlock_status + ")" + NL
        )
        send_discord(self.config.discord_webhook_url, msg)

    def _send_monthly_summary(self):
        NL = chr(10)
        today       = date.today()
        month_start = today.replace(day=1)
        pnls        = self._query_pnls_since(month_start)
        all_pnls    = self._query_all_pnls()
        equity      = self.state.equity or 100_000.0
        n           = len(all_pnls)
        wins        = [p for p in all_pnls if p > 0]
        losses      = [p for p in all_pnls if p <= 0]
        wr          = len(wins) / n * 100 if n else 0
        pf          = sum(wins) / abs(sum(losses)) if losses else 999.0
        ready       = n >= 30 and wr >= 60 and pf >= 1.3
        msg = (
            "Monthly Summary - " + today.strftime("%B %Y") + NL +
            "Account equity: $" + "{:,.2f}".format(equity) + NL + NL +
            self._pnl_block(pnls, today.strftime("%B") + " P&L") + NL +
            self._pnl_block(all_pnls, "All-time P&L") + NL +
            "Go-live status: " + ("READY" if ready else "NOT YET") +
            " (" + str(n) + " trades, " + "{:.0f}".format(wr) + "% WR, " + "{:.2f}".format(pf) + " PF)" + NL
        )
        send_discord(self.config.discord_webhook_url, msg)

    def _check_options_approved(self) -> None:
        """
        Verify options trading is enabled on the Alpaca account.

        Alpaca requires options_approved_level >= 1 for simple options and
        >= 2 for spreads. This check runs at Orchestrator.__init__() so the
        bot fails fast at startup rather than silently at order time.

        Paper accounts always pass — Alpaca paper trading auto-enables options.
        Live accounts must have applied for options approval in the Alpaca dashboard
        at https://app.alpaca.markets → Account → Options Trading.
        """
        # PaperBroker stub has no account info — skip
        if isinstance(self.broker, PaperBroker):
            return

        try:
            account = self.broker.get_account()
        except Exception as exc:
            logger.warning(
                "[Orchestrator] Could not verify options approval (%s) — proceeding. "
                "Ensure options trading is enabled in your Alpaca account dashboard "
                "before going live.", exc
            )
            return

        # Paper Alpaca accounts return is_paper=True — options always enabled
        if getattr(self.broker, '_paper', True):
            return

        # For live accounts, check the options approval level
        # Alpaca returns this as account.options_approved_level (int 0-4)
        # 0 = not approved, 1 = covered calls/cash-secured puts, 2 = spreads
        level = None
        if hasattr(account, 'options_approved_level'):
            level = account.options_approved_level
        elif isinstance(account, dict):
            level = account.get('options_approved_level')

        if level is None:
            logger.warning(
                "[Orchestrator] options_approved_level not found in account data — "
                "verify options trading is enabled at app.alpaca.markets before going live."
            )
            return

        try:
            level = int(level)
        except (TypeError, ValueError):
            logger.warning("[Orchestrator] Could not parse options_approved_level=%r", level)
            return

        strategy = self.config.strategy_name
        # Spreads require level 2; CSP/covered calls require level 1
        requires_level = 2 if strategy in ("short_put_spread", "short_strangle") else 1

        if level < requires_level:
            raise PipelineConnectionError(
                f"Options trading approval insufficient for strategy '{strategy}'. "
                f"Account options_approved_level={level}, need >={requires_level}. "
                f"Apply for options approval at: https://app.alpaca.markets → Account → Options Trading"
            )

        logger.info(
            "[Orchestrator] Options approval verified: level=%d (strategy=%s requires>=%d)",
            level, strategy, requires_level
        )

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
