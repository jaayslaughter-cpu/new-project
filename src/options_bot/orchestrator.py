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
import re
import sqlite3
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from .broker import AlpacaBroker, PaperBroker, get_broker
from .contracts import ApprovedOrder, FilledOrder
from .exceptions import (
    LiquidityFilterError,
    PipelineConnectionError,
    RiskVetoError,
    StalenessError,
)
from .greeks import GreeksEnricher, get_risk_free_rate
from .parity_check import check_parity
from .vrp_gate import evaluate_vrp_gate, PROVISIONAL_RV_WINDOW
from .market_data import YFinanceDataLoader
from .risk import ExecutionGuard, RiskConfig, RiskManager
from .regime import RegimeDetector, get_regime_policy, REGIME_POLICY
from .sentiment import SentimentAnalyzer, SentimentConfig
from .metrics import summary as perf_summary
from .adaptive import AdaptiveTuner
from .strategy import BaseStrategy, StrategySignal, get_strategy, STRATEGY_REGISTRY
from .strategy_0dte import ZeroDTEConfig, ZeroDTEStrategy, ZeroDTEMonitor
from .zerodte_guard import ZeroDTECircuitBreaker
from .scanner import TickerGate
from .risk_profiles import RiskLevel, RiskProfile, get_risk_profile, apply_profile
from .universe import UniverseBuilder
from .volume_profile import volume_profile_cache
from .stress_testing import run_stress_suite, positions_from_broker
from .sec_signals import (is_entry_confirmed, score_sec_signals,
                           score_sec_with_news, get_dynamic_tickers)
from .confidence_score import ConfidenceScorer
from .stat_validation import run_all_validations, format_validation_discord

logger = logging.getLogger(__name__)


# Extra strategies that share the PRIMARY strategy's bullish/neutral thesis and
# may therefore reuse the bullish shortlist when the directional router gives
# them no dedicated bucket. csp (naked puts) is the same directional bet as
# short_put_spread, so it has no bearish/neutral router bucket and the bullish
# shortlist is correct for it. Every OTHER non-primary strategy is
# direction-specific: see _select_extra_tickers.
_BULLISH_EQUIVALENT_STRATEGIES = frozenset({"csp", "cash_secured_put"})


def _select_extra_tickers(
    extra_name: str,
    routes: dict[str, list[str]],
    routing_ran: bool,
    scan_tickers: list[str],
) -> list[str]:
    """Pick the ticker list for one extra-strategy pass.

    Returns ``[]`` to mean "skip this strategy this scan".

    The rule that matters: a direction-specific strategy (short_call_spread,
    short_strangle, iron_condor, …) is SKIPPED when routing ran but assigned it
    no tickers — rather than falling back to the bullish shortlist. Selling call
    spreads on tickers the scorer just confirmed bullish fights the trend; that
    "structurally backwards" fallback is the bug this guards against.

    The bullish shortlist (``scan_tickers``) is reused only when:
      * routing did NOT run (ticker_gate disabled / routing errored) — in that
        case ``scan_tickers`` is the unfiltered candidate pool, not a bullish
        subset, so it's a safe legacy fallback; or
      * the strategy is bullish-equivalent (csp) and has no router bucket.
    """
    routed = routes.get(extra_name, [])
    if routed:
        return routed
    if not routing_ran:
        return scan_tickers
    if extra_name in _BULLISH_EQUIVALENT_STRATEGIES:
        return scan_tickers
    return []


# ---------------------------------------------------------------------------
# OCC option-symbol parser
# ---------------------------------------------------------------------------

# OCC symbol format: <UNDERLYING><YYMMDD><C|P><STRIKE_x1000 zero-padded to 8>
# Example: IWM260731C00320000 → IWM, 2026-07-31, call, 320.00
_OCC_RE = re.compile(r'^([A-Z]+)(\d{6})([CP])(\d{8})$')


def _parse_occ_symbol(symbol: str) -> Optional[dict]:
    """Parse an OCC option symbol into its components.

    Returns a dict with keys ``underlying``, ``expiry`` (date), ``option_type``
    ('call'|'put'), and ``strike`` (float), or ``None`` if *symbol* is not a
    valid OCC option symbol (e.g. bare equity ticker, garbage string, empty).

    Used by ``_adopt_orphans`` to reconstruct position metadata from the raw
    Alpaca symbol returned in broker positions.
    """
    if not symbol:
        return None
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    underlying, expiry_str, cp, strike_str = m.groups()
    try:
        expiry = date(
            2000 + int(expiry_str[:2]),
            int(expiry_str[2:4]),
            int(expiry_str[4:6]),
        )
        strike = int(strike_str) / 1000.0
        return {
            "underlying": underlying,
            "expiry": expiry,
            "option_type": "call" if cp == "C" else "put",
            "strike": strike,
        }
    except Exception:
        return None


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

    # Additional strategies — run after primary each scan, each on a
    # direction-appropriate shortlist (see filter_ranked_multi_strategy).
    # csp:               naked puts — same bullish/neutral bet as the
    #                     primary strategy. Falls back to the bullish
    #                     shortlist since it has no dedicated routing bucket
    #                     (CSP and ShortPutSpread are directionally the same
    #                     bet, just different structures — defined-risk
    #                     spread vs naked put). Risk sizing naturally limits
    #                     CSP to lower-priced underlyings — see earlier
    #                     audit note: CSP's theoretical max loss formula
    #                     ((strike - credit) x 100) makes it self-reject via
    #                     the 1% risk budget on anything pricier than a
    #                     small-cap-ish per-share level.
    # short_call_spread: sells calls when ticker is overbought / near upper BB
    # short_strangle:    sells both call and put on same expiry (neutral premium)
    #
    # AUDIT FIX: "csp" was previously missing from this list entirely.
    # CashSecuredPut was a fully working, tested strategy that simply never
    # ran in production because nothing ever selected it — strategy_name
    # defaults to short_put_spread and CSP wasn't in extra_strategies either.
    extra_strategies: list = field(default_factory=lambda: [
        "csp",                  # naked puts — bullish/neutral, same shortlist as primary
        "short_call_spread",    # calls — bearish/overbought setups
        "short_strangle",       # both sides — neutral high-IV setups
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

    # --- Iron Condor (gated future strategy) ---
    # Defined-risk neutral premium seller — the upgrade to ShortStrangle for
    # neutral-direction tickers. Built and tested but DORMANT by default.
    # TWO gates must BOTH clear before it ever trades:
    #   1. The 30-trade walk-forward milestone must be passed (block removed
    #      automatically once >= iron_condor_min_trades closed trades exist).
    #   2. iron_condor_enabled must be explicitly set True (deliberate human
    #      go-live decision — the milestone makes it ELIGIBLE, not automatic).
    # This mirrors how 0DTE was enabled: the gate prevents premature
    # activation; the flag preserves the human checkpoint at the riskiest
    # moment (a strategy's first live trades).
    iron_condor_enabled: bool = False        # explicit human go-live flag
    iron_condor_min_trades: int = 30         # milestone gate (walk-forward unlock)

    # --- Gated regime/signal inputs (built, tested, DORMANT) ---
    # Same two-gate pattern as the Iron Condor: each stays inert until BOTH
    # (1) >= *_min_trades closed trades (walk-forward milestone removes the
    # block automatically) AND (2) the *_enabled flag is explicitly True
    # (deliberate human go-live decision). These are SIGNAL INPUTS that nudge
    # the regime score / confirm directional reads — not new trade types — so
    # they serve the capital-preservation + income mandate (avoid bad trades,
    # better time good ones) without adding new risk surface.
    #
    # credit_regime: HYG/IEF credit-spread proxy — pulls the bot defensive
    #   when credit markets signal stress (a cross-asset read the regime
    #   detector currently lacks).
    # analyst_revisions: net analyst upgrade/downgrade momentum over 30d —
    #   a slower-moving directional confirmation distinct from FinBERT
    #   headline sentiment.
    credit_regime_enabled: bool = False
    credit_regime_min_trades: int = 30
    analyst_revisions_enabled: bool = False
    analyst_revisions_min_trades: int = 30
    #
    # vrp_gate: vol-risk-premium entry gate. The whole short-premium book lives
    #   on IV exceeding subsequently-realized vol; this gate confirms IV is
    #   actually rich vs realized (Yang-Zhang RV from OHLC) before allowing a
    #   premium-selling entry, and shrinks/vetoes when the premium isn't there.
    #   A gate, not a trade type. NOT applied to the 0DTE fast path.
    vrp_gate_enabled: bool = False
    vrp_gate_min_trades: int = 30

    # macro_blackout: don't open NEW premium-selling positions into a scheduled
    #   high-impact macro event (FOMC by default; add CPI/jobs/PCE/GDP via
    #   macro_blackout_extra_events as 'YYYY-MM-DD:Label'). Pure risk-reducer
    #   (only ever vetoes an entry), so it is intentionally NOT milestone-gated:
    #   you want event protection active during the paper-trading window so
    #   event-driven entries don't pollute the edge estimate. Dormant by default;
    #   fail-open; NOT applied to the 0DTE fast path. See macro_blackout.py.
    macro_blackout_enabled: bool = False
    macro_blackout_lookahead_days: int = 1
    macro_blackout_extra_events: tuple[str, ...] = ()

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

    # --- Orphan adoption (gated — safe only after ?-placeholder fix is confirmed deployed) ---
    # Reads broker positions with no matching DB record and writes them into the
    # trades table so the position monitor can manage them.  Uses the same INSERT
    # path as save_fill, so it must stay OFF until the 63fc7d8 fix is confirmed
    # running in the deployed Railway container.  Enable manually after verifying
    # the deployed container is on the current main.
    adopt_orphans_enabled: bool = False

    # --- Catch-up scan (default ON — guards against missed scan on restart) ---
    # On startup, if no scan has run today and the market is open past the
    # scheduled 6:45 AM PT window with enough time left before close, fire one
    # scan immediately.  Idempotency marker (last_scan_date in bot_state) ensures
    # multiple container restarts on the same day produce exactly one scan.
    # Set False only if you want zero startup-triggered entries.
    catchup_scan_enabled: bool = True
    catchup_min_minutes_to_close: int = 90  # don't catch up within 90 min of close


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
    peak_equity: float = 0.0          # all-time high equity; drawdown measured from here
    drawdown_halt: bool = False        # True when max_drawdown_pct is breached
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
            # psycopg2 uses %-style param formatting, so any LITERAL % in the
            # SQL (e.g. LIKE '0dte_%') is read as a format target and throws
            # "tuple index out of range". Double literal % FIRST, then rewrite
            # ? -> %s (the placeholder we actually want bound).
            sql = sql.replace("%", "%%").replace("?", "%s")
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

            # Generic key-value state table — survives container restarts.
            # Used to persist in-memory strategy state (0DTE cooldown timer,
            # momentum engine EMAs) that would otherwise silently reset every
            # time Railway redeploys mid-session.
            # (idea from IntelliStock's strategy_cache_persistence.py pattern)
            if self._use_pg:
                self._execute(conn, """
                    CREATE TABLE IF NOT EXISTS bot_state (
                        key        TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
            else:
                self._execute(conn, """
                    CREATE TABLE IF NOT EXISTS bot_state (
                        key        TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
            conn.commit()
        finally:
            conn.close()
        logger.debug("[TradeDB] Schema ready (%s)", "PostgreSQL" if self._use_pg else "SQLite")

    # ------------------------------------------------------------------
    # Generic state persistence (survives container restarts)
    # ------------------------------------------------------------------

    def save_state(self, key: str, value: dict, max_age_hours: int = 18) -> None:
        """
        Persist an arbitrary JSON-serialisable dict under a key.

        Used for in-memory strategy state that needs to survive Railway
        container restarts mid-session (0DTE cooldown, momentum EMAs).
        Fail-open: any error here must never break the trading loop.

        max_age_hours is not enforced here — it's read by load_state() at
        load time, so old state (e.g. from a previous trading day) is
        automatically discarded rather than incorrectly resumed.
        """
        try:
            conn = self._get_conn()
            try:
                now = datetime.now(timezone.utc).isoformat()
                payload = json.dumps(value)
                if self._use_pg:
                    self._execute(conn, """
                        INSERT INTO bot_state (key, value_json, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT (key) DO UPDATE
                        SET value_json = EXCLUDED.value_json,
                            updated_at = EXCLUDED.updated_at
                    """, (key, payload, now))
                else:
                    self._execute(conn, """
                        INSERT OR REPLACE INTO bot_state (key, value_json, updated_at)
                        VALUES (?, ?, ?)
                    """, (key, payload, now))
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("[TradeDB] save_state(%s) failed (non-fatal): %s", key, exc)

    def load_state(self, key: str, max_age_hours: int = 18) -> Optional[dict]:
        """
        Load a previously saved state dict, or None if not found / stale.

        Discards state older than max_age_hours — this prevents a Monday
        morning boot from resuming Friday's stale 0DTE cooldown/momentum
        state, which would be meaningless after a weekend gap.
        """
        try:
            conn = self._get_conn()
            try:
                cur = self._execute(
                    conn, "SELECT value_json, updated_at FROM bot_state WHERE key = ?", (key,)
                )
                row = cur.fetchone()
                if not row:
                    return None
                value_json, updated_at = row[0], row[1]
                try:
                    saved_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    if saved_at.tzinfo is None:
                        saved_at = saved_at.replace(tzinfo=timezone.utc)
                    age_hours = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        logger.debug(
                            "[TradeDB] load_state(%s) stale (%.1fh > %dh) — ignoring",
                            key, age_hours, max_age_hours,
                        )
                        return None
                except Exception:
                    pass
                return json.loads(value_json)
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("[TradeDB] load_state(%s) failed (non-fatal): %s", key, exc)
            return None

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

        # PostgreSQL uses INSERT ... ON CONFLICT; SQLite uses INSERT OR REPLACE.
        # BOTH branches use ? placeholders — _execute() rewrites ? -> %s for
        # psycopg2. Writing %s here directly is a bug: _execute escapes % -> %%
        # first, turning %s into %%s, which Postgres receives as literal "%s"
        # text ("syntax error at or near %").
        if self._use_pg:
            sql = """
                INSERT INTO trades
                (id, trade_date, strategy, underlying, legs_json,
                 fill_price, slippage, max_loss, hard_stop, contracts,
                 net_credit, status, broker, created_at, updated_at,
                 delta, vega, theta, underlying_price, expiry,
                 profit_target_price, profit_target_pct)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            conn.close()
            raise  # re-raise so the call site can alert via Discord
        # Read-back verification: confirm the row actually persisted.
        # A miss triggers a warning (DB inconsistency) but does NOT raise —
        # the trade is already filled at the broker and the scan must continue.
        try:
            cur = self._execute(
                conn, "SELECT id FROM trades WHERE id = ?", (filled.order_id,)
            )
            row = cur.fetchone() if cur else None
            if row is None:
                logger.warning(
                    "[TradeDB] Read-back miss for %s — row not found after commit; "
                    "DB may be inconsistent",
                    filled.order_id,
                )
        except Exception as rb_exc:
            logger.warning(
                "[TradeDB] Read-back check error for %s: %s", filled.order_id, rb_exc
            )
        finally:
            conn.close()

    def adopt_orphan(
        self,
        underlying: str,
        strategy: str,
        net_credit: float,
        legs: list,
        profit_target_price: float,
    ) -> bool:
        """Record a broker position that has no existing DB entry as an open trade.

        Mirrors ``save_fill``'s INSERT exactly (same 22 columns, same ?
        placeholders so ``_execute`` can rewrite them for Postgres).  The
        broker tag is set to ``"adopted"`` so the position monitor and analytics
        can distinguish adopted records from normal fills.

        Returns True on success, False on DB error (never raises — orphan
        adoption is best-effort; the position is already live at the broker).
        """
        import uuid as _uuid
        order_id = f"adopted-{_uuid.uuid4()}"
        now = datetime.now(tz=timezone.utc).isoformat()
        legs_json = json.dumps(legs)

        if self._use_pg:
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
        params = (
            order_id,
            date.today().isoformat(),
            strategy,
            underlying,
            legs_json,
            net_credit,            # fill_price (net credit received)
            0.0,                   # slippage (unknown at adoption)
            0.0,                   # max_loss (unknown at adoption)
            net_credit * 2.0,      # hard_stop (2× credit default)
            1,                     # contracts (position qty assumed = 1 lot)
            net_credit,
            "open",
            "adopted",             # broker tag distinguishes from normal fills
            now, now,
            None, None, None,      # delta, vega, theta
            None, None,            # underlying_price, expiry
            profit_target_price,
            0.5,                   # profit_target_pct (50% of credit)
        )
        conn = self._get_conn()
        try:
            self._execute(conn, sql, params)
            conn.commit()
            logger.info(
                "[TradeDB] Adopted orphan: %s %s %s net_credit=%.2f",
                order_id, underlying, strategy, net_credit,
            )
            return True
        except Exception as exc:
            logger.error("[TradeDB] adopt_orphan failed for %s/%s: %s", underlying, strategy, exc)
            return False
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

    def get_today_realized_0dte_pnl(self, trade_date: str) -> float:
        """
        Sum today's REALIZED P&L from closed 0DTE trades only (strategy
        LIKE '0dte_%'). Used by the dedicated 0DTE circuit breaker — kept
        strictly separate from the core book's P&L so the two strategies'
        risk accounting never co-mingle. Returns 0.0 if none / on error
        (caller treats 0.0 as "no 0DTE activity today").
        """
        conn = self._get_conn()
        try:
            cur = self._execute(
                conn,
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades "
                "WHERE trade_date = ? "
                "AND strategy LIKE '0dte_%' "
                "AND realized_pnl IS NOT NULL",
                (trade_date,),
            )
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception as exc:
            logger.error("[TradeDB] get_today_realized_0dte_pnl failed: %s", exc)
            return 0.0
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
            headers={
                "Content-Type": "application/json",
                # Discord's API sits behind Cloudflare, which 403s requests
                # carrying Python's default urllib User-Agent regardless of
                # webhook validity. A descriptive UA avoids the block.
                # NOTE: this fix was once lost to a stale-copy push regression
                # (added in 0ac6aa1, silently reverted in a311ab8) — restored
                # and guarded by the audit. Do not remove.
                "User-Agent": "OptionsBot/1.0 (+https://github.com/jaayslaughter-cpu/new-project)",
            },
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
        # VRP gate activation — both gates must clear (enabled flag AND the
        # >=N-trade milestone). Evaluated once at construction from the live
        # closed-trade count. Mirrors the Iron Condor / credit_regime pattern.
        try:
            _n_closed = len(self.db.get_all_closed_pnls()) if self.db else 0
        except Exception:
            _n_closed = 0
        self._vrp_gate_active = (
            getattr(config, "vrp_gate_enabled", False)
            and _n_closed >= getattr(config, "vrp_gate_min_trades", 30)
        )
        if getattr(config, "vrp_gate_enabled", False) and not self._vrp_gate_active:
            logger.info(
                "[Pipeline] vrp_gate enabled but gated: %d/%d closed trades",
                _n_closed, getattr(config, "vrp_gate_min_trades", 30),
            )

        # Macro-event blackout — enable-flag only (no milestone gate; it's a
        # pure risk-reducer, see macro_blackout.py).
        self._macro_blackout_active = getattr(config, "macro_blackout_enabled", False)
        if self._macro_blackout_active:
            logger.info(
                "[Pipeline] macro_blackout ACTIVE: vetoing new entries within "
                "%d day(s) of a high-impact macro event",
                getattr(config, "macro_blackout_lookahead_days", 1),
            )

    def _evaluate_vrp_for_signal(self, ticker: str, signal):
        """Compute the VRP gate result for a produced signal.

        Uses the signal's short-leg IV as the structure IV, and recent daily
        OHLC bars for the underlying. Returns a VRPGateResult or None ('no
        read'). Pure orchestration glue — the math lives in vrp_gate.py.
        """
        # Structure IV: prefer the short leg's IV from the source contracts.
        iv = None
        for c in (signal.source_contracts or []):
            if getattr(c, "iv", None):
                iv = c.iv
                break
        if not iv:
            return None
        # Recent daily OHLC for the underlying (need > RV_WINDOW bars).
        bars = self.broker.get_bars(ticker, timeframe="1Day", limit=60)
        if not bars or len(bars) < PROVISIONAL_RV_WINDOW + 2:
            return None
        open_ = [b["o"] for b in bars]
        high  = [b["h"] for b in bars]
        low   = [b["l"] for b in bars]
        close = [b["c"] for b in bars]
        return evaluate_vrp_gate(iv, open_, high, low, close)

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

        # --- Macro-event blackout (capital preservation) ---
        # Veto a NEW entry into a scheduled high-impact macro event (FOMC, etc.)
        # before doing any chain work. Pure risk-reducer; fail-open.
        if getattr(self, "_macro_blackout_active", False):
            try:
                from .macro_blackout import check_macro_blackout
                _bo = check_macro_blackout(
                    lookahead_days=getattr(
                        self.config, "macro_blackout_lookahead_days", 1
                    ),
                    extra_events=getattr(
                        self.config, "macro_blackout_extra_events", ()
                    ),
                )
                if _bo.in_blackout:
                    logger.info(
                        "[Pipeline] %s: macro blackout — %s in %s day(s); "
                        "skipping new entry",
                        ticker, _bo.event_label, _bo.days_until,
                    )
                    return None
            except Exception as exc:  # noqa: BLE001 — fail-open
                logger.debug("[Pipeline] macro blackout check skipped: %s", exc)

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

        # --- Step 2a: Put-call parity data-quality gate ---
        # Pure-math sanity check that the chain's near-the-money calls and
        # puts are internally consistent (C - P ≈ S - K·e^(-rT)). NOT an
        # arbitrage screen — it's a guard against pricing a spread on stale,
        # crossed, or otherwise broken quotes. Deliberately loose (only trips
        # on gross >3%-of-spot violations across a majority of NTM pairs) so
        # legitimate American-exercise/dividend deviations never false-trip.
        # Fails open on thin/untestable chains (a separate liquidity gate
        # handles those). Capital-preservation: skip rather than size risk
        # off data we can't trust.
        try:
            parity = check_parity(enriched, rate=get_risk_free_rate())
            if not parity.ok:
                logger.warning("[Pipeline] %s parity gate: %s", ticker, parity.reason)
                return None
            logger.debug("[Pipeline] %s parity: %s", ticker, parity.reason)
        except Exception as exc:
            # Never let the data-quality gate itself crash the scan — fail open.
            logger.debug("[Pipeline] %s parity check skipped (non-fatal): %s", ticker, exc)

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
            _risk_budget = (
                self.rm._equity * self.rm.config.risk_budget_pct
                if hasattr(self, "rm") and self.rm is not None
                else None
            )
            signal = self.strategy.evaluate(enriched, risk_budget_dollars=_risk_budget)
        except LiquidityFilterError as exc:
            logger.info("[Pipeline] %s strategy: no qualifying contracts — %s", ticker, exc)
            return None
        except PipelineConnectionError as exc:
            self.state.record_error(f"{ticker} strategy failed: {exc}")
            return None
        except Exception as exc:
            # Defense in depth: any unexpected bug in strategy code (bad variable
            # reference, type error, etc.) must not kill the entire scan job.
            # Skip this ticker, log loudly, and let the scan continue to the
            # next ticker in the shortlist.
            logger.error(
                "[Pipeline] %s: UNEXPECTED error in strategy.evaluate() — %s. "
                "Skipping this ticker. This indicates a bug — investigate.",
                ticker, exc, exc_info=True,
            )
            self.state.record_error(f"{ticker} strategy UNEXPECTED error: {exc}")
            return None

        # --- Step 3b: Vol-risk-premium gate (GATED) ---
        # The short-premium book profits only when IV exceeds subsequently
        # realized vol. This gate confirms IV is actually rich vs realized
        # (Yang-Zhang RV from daily OHLC) before allowing the entry. When the
        # premium isn't there (thin/negative VRP) it vetoes; otherwise it
        # passes through a size_factor the risk manager can use to shrink size.
        # Active only when BOTH gates clear (>=30 trades AND enabled). Fully
        # off / no-op when inactive. Fail-OPEN on a 'no read' (None) — only an
        # explicit thin/negative VRP blocks, consistent with the core book's
        # data-gate posture. NOT applied to the 0DTE path (separate fast path).
        signal.vrp_size_factor = 1.0  # default: no VRP shrink
        if getattr(self, "_vrp_gate_active", False):
            try:
                vrp_res = self._evaluate_vrp_for_signal(ticker, signal)
                if vrp_res is not None:
                    if not vrp_res.passes:
                        logger.info(
                            "[Pipeline] %s VRP gate veto: IV=%.3f RV=%.3f vrp=%.3f "
                            "ratio=%.2f — premium not rich vs realized, skipping",
                            ticker, vrp_res.iv, vrp_res.rv, vrp_res.vrp, vrp_res.iv_rv_ratio,
                        )
                        return None
                    signal.vrp_size_factor = vrp_res.size_factor
                    logger.info(
                        "[Pipeline] %s VRP gate pass: vrp=%.3f size_factor=%.2f",
                        ticker, vrp_res.vrp, vrp_res.size_factor,
                    )
                else:
                    logger.debug("[Pipeline] %s VRP gate: no read (insufficient RV data) — fail-open", ticker)
            except Exception as exc:
                logger.debug("[Pipeline] %s VRP gate skipped (non-fatal): %s", ticker, exc)

        # --- Step 4: Risk evaluation ---
        # Update equity from broker before sizing
        try:
            current_equity = self.broker.get_equity()
            self.rm.update_equity(current_equity)
            self.state.equity = current_equity
            # Ratchet peak and check drawdown halt
            self.rm.update_peak_equity(current_equity)
            if self.rm.is_drawdown_halted() and not self.state.drawdown_halt:
                self.state.drawdown_halt = True
                dd_pct = self.rm.current_drawdown_pct()
                _dd_msg = (
                    f"🛑 **MAX DRAWDOWN HALT** — bot paused.\n"
                    f"Account is down **{dd_pct:.1%}** from peak "
                    f"(equity=${current_equity:,.2f}, peak=${self.rm._peak_equity:,.2f}, "
                    f"limit={self.config.risk_config.max_drawdown_pct:.1%}).\n"
                    f"No new trades will be opened. Review performance and restart the bot manually."
                )
                logger.critical("[Orchestrator] %s", _dd_msg.replace("**", ""))
                send_discord(self.config.discord_webhook_url, _dd_msg)
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
            # Pass news sentiment into SEC scoring for congress+news confirmation.
            # score_sec_with_news() is independently hardened, but wrap here too —
            # a single ticker's SEC/congress lookup must never crash the whole scan.
            try:
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
            except Exception as _sec_exc:
                logger.warning(
                    "[Pipeline] %s SEC/Congress scoring failed (non-fatal, using neutral): %s",
                    ticker, _sec_exc,
                )
                _sec_data_full = {"score": 0, "detail": "scoring unavailable"}
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
        try:
            self.db.save_fill(filled)
        except Exception as _sf_exc:
            # Trade IS filled at the broker — never abort the scan here.
            # Alert loudly so the orphan can be adopted manually (or via
            # adopt_orphans once that gate is enabled).
            _sf_msg = (
                f"🔴 **DB write failed** — {filled.order_id} IS FILLED AT BROKER "
                f"but NOT in database. Manual reconciliation required.\n"
                f"Strategy: {getattr(order, 'strategy_name', '?')} | "
                f"Underlying: {getattr(order, 'underlying', '?')} | "
                f"Error: {_sf_exc}"
            )
            logger.error("[Pipeline] save_fill failed for %s: %s", filled.order_id, _sf_exc)
            self.state.record_error(f"save_fill failed {filled.order_id}: {_sf_exc}")
            send_discord(self.config.discord_webhook_url, _sf_msg)
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
        Two-pass expiry selection:
          Pass 1 — prefer sweet spot 30-45 DTE (peak theta/gamma ratio)
          Pass 2 — accept full 14-60 DTE window (rescues monthly-only ETFs
                   that have no expiry in the 30-45 range on a given day)
        """
        today = date.today()
        sweet_min, sweet_max = 30, 45
        sweet_target = (sweet_min + sweet_max) / 2

        # Pass 1 — sweet spot
        best = None
        best_diff = float("inf")
        for exp_str in expirations:
            try:
                dte = (date.fromisoformat(exp_str) - today).days
                if sweet_min <= dte <= sweet_max:
                    diff = abs(dte - sweet_target)
                    if diff < best_diff:
                        best_diff = diff; best = exp_str
            except ValueError:
                continue
        if best:
            return best

        # Pass 2 — full window fallback
        full_target = (self.config.min_dte + self.config.max_dte) / 2
        best_diff = float("inf")
        for exp_str in expirations:
            try:
                dte = (date.fromisoformat(exp_str) - today).days
                if self.config.min_dte <= dte <= self.config.max_dte:
                    diff = abs(dte - full_target)
                    if diff < best_diff:
                        best_diff = diff; best = exp_str
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
        # Gated regime/signal inputs — active only when BOTH the milestone
        # (>=N closed trades) AND the explicit enable flag clear. Mirrors the
        # Iron Condor activation pattern: the milestone removes the block; the
        # flag is the deliberate human go-live decision. Evaluated once at
        # construction from the current closed-trade count.
        try:
            _n_closed = len(self.db.get_all_closed_pnls()) if self.db else 0
        except Exception:
            _n_closed = 0
        _credit_active = (
            getattr(config, "credit_regime_enabled", False)
            and _n_closed >= getattr(config, "credit_regime_min_trades", 30)
        )
        _revisions_active = (
            getattr(config, "analyst_revisions_enabled", False)
            and _n_closed >= getattr(config, "analyst_revisions_min_trades", 30)
        )
        if getattr(config, "credit_regime_enabled", False) and not _credit_active:
            logger.info(
                "[Orchestrator] credit_regime enabled but gated: %d/%d closed trades",
                _n_closed, getattr(config, "credit_regime_min_trades", 30),
            )
        if getattr(config, "analyst_revisions_enabled", False) and not _revisions_active:
            logger.info(
                "[Orchestrator] analyst_revisions enabled but gated: %d/%d closed trades",
                _n_closed, getattr(config, "analyst_revisions_min_trades", 30),
            )
        self.regime_detector = RegimeDetector(
            cache_ttl_seconds=config.regime_cache_ttl,
            credit_regime_active=_credit_active,
            analyst_revisions_active=_revisions_active,
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
            # Dedicated 0DTE circuit breaker — daily loss cap + consecutive
            # losing-day cooldown, persisted via bot_state. Kept entirely
            # separate from the core book's risk halts.
            self.zero_dte_cb = ZeroDTECircuitBreaker(
                _0dte_cfg,
                load_state=self.db.load_state,
                save_state=self.db.save_state,
            )
            logger.info("[Orchestrator] 0DTE GEX scalper enabled (underlying=%s)",
                        _0dte_cfg.underlying)
        else:
            self.zero_dte = None
            self.zero_dte_monitor = None
            self.zero_dte_cb = None

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

        # Fire catch-up scan if this boot missed the scheduled 6:45 AM PT window.
        # Fail-open: any error in the catch-up must never prevent the scheduler
        # from starting.
        try:
            self._maybe_catchup_scan()
        except Exception as _cs_exc:
            logger.warning(
                "[Orchestrator] Catch-up scan startup error (non-fatal): %s", _cs_exc
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

        # Market holiday / closed guard — APScheduler's cron trigger does not
        # know about NYSE holidays (Juneteenth, Thanksgiving, etc.), so without
        # this check the job fires every 2 minutes against stale or missing
        # data on closed days.
        if not _market_is_open():
            logger.debug("[0DTE scan] Market closed — skip")
            return

        # Respect max daily position cap
        if self.zero_dte_monitor and \
           self.zero_dte_monitor.open_count >= self.config.zero_dte_config.max_daily_positions:
            logger.debug("[0DTE scan] Max positions (%d) open — skip",
                         self.config.zero_dte_config.max_daily_positions)
            return

        # Dedicated 0DTE circuit breaker — daily loss cap + cooldown.
        # Blocks NEW 0DTE entries only; never touches the core book or any
        # already-open 0DTE position (the monitor still manages exits).
        if self.zero_dte_cb is not None:
            try:
                today_str = date.today().isoformat()
                today_0dte_pnl = self.db.get_today_realized_0dte_pnl(today_str) if self.db else 0.0
                equity = self.state.equity or self.config.zero_dte_config.starting_capital
                decision = self.zero_dte_cb.check_entry_allowed(
                    today=date.today(),
                    equity=equity,
                    today_realized_0dte_pnl=today_0dte_pnl,
                )
                if not decision.allowed:
                    logger.info("[0DTE scan] Blocked by guard: %s", decision.reason)
                    return
            except Exception as exc:
                # Fail-CLOSED: if the guard itself errors, do NOT trade 0DTE.
                logger.warning("[0DTE scan] Guard check errored — blocking 0DTE (fail-closed): %s", exc)
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

            # Adopt orphan positions — broker has them, DB doesn't.
            # Gated by adopt_orphans_enabled (default False) until the
            # ?-placeholder fix is confirmed deployed and INSERT path is clean.
            if self.config.adopt_orphans_enabled:
                _db_symbols: set = set()
                for _t in db_trades:
                    try:
                        for _leg in json.loads(_t.get("legs_json", "[]")):
                            _db_symbols.add(_leg.get("symbol", ""))
                    except Exception:
                        pass
                _orphans = [p for p in alpaca_positions if p.get("symbol") not in _db_symbols]
                if _orphans:
                    logger.info(
                        "[Reconcile] %d potential orphan legs — attempting adoption",
                        len(_orphans),
                    )
                    self._adopt_orphans(_orphans)

        except Exception as exc:
            logger.warning("[Reconcile] Reconciliation failed (non-fatal): %s", exc)

    def _adopt_orphans(self, positions: list) -> None:
        """Reconstruct DB records for broker positions that have no DB entry.

        Called from ``_reconcile_positions`` when ``adopt_orphans_enabled=True``.
        Can also be called directly for manual reconciliation.

        Recognizes two defined-risk premium-selling shapes only:
          * 1 short put (no long) → cash_secured_put
          * 1 short + 1 long, same option type → short_put_spread or short_call_spread

        Everything else (naked calls, >2 legs, mismatched types, long-only) is
        logged and skipped — never adopted without a clear structure.

        net_credit is the per-share price (e.g. 0.83), NOT × 100.
        """
        if not self.config.adopt_orphans_enabled:
            return

        # Filter to options only, parse each OCC symbol
        parsed = []
        for pos in positions:
            if pos.get("asset_class") != "us_option":
                continue
            sym = pos.get("symbol", "")
            info = _parse_occ_symbol(sym)
            if info is None:
                logger.warning("[Orphan] Cannot parse OCC symbol %r — skipping", sym)
                continue
            parsed.append({**pos, **info})

        # Group by (underlying, expiry)
        groups: dict = {}
        for p in parsed:
            key = (p["underlying"], p["expiry"])
            if key not in groups:
                groups[key] = []
            groups[key].append(p)

        for (underlying, expiry), legs in groups.items():
            try:
                n_legs = len(legs)
                short_legs = [l for l in legs if l.get("side") == "short"]
                long_legs  = [l for l in legs if l.get("side") == "long"]

                if n_legs > 2:
                    logger.warning(
                        "[Orphan] %s %s: %d-leg shape not recognized — skipping",
                        underlying, expiry, n_legs,
                    )
                    send_discord(
                        self.config.discord_webhook_url,
                        f"⚠️ **Orphan skip** — {underlying} {expiry}: "
                        f"{n_legs}-leg structure not recognized",
                    )
                    continue

                if not short_legs:
                    logger.info(
                        "[Orphan] %s %s: long-only position — skipping",
                        underlying, expiry,
                    )
                    continue

                short = short_legs[0]

                if n_legs == 1:
                    # Single short leg → only adopt as CSP (short put)
                    if short["option_type"] != "put":
                        logger.warning(
                            "[Orphan] %s %s: naked short call — skipping (undefined risk)",
                            underlying, expiry,
                        )
                        continue
                    strategy = "cash_secured_put"
                    net_credit = float(short["avg_entry_price"])
                else:
                    # 2 legs: 1 short + 1 long, same option type
                    long_ = long_legs[0]
                    if short["option_type"] != long_["option_type"]:
                        logger.warning(
                            "[Orphan] %s %s: mixed call/put legs — skipping",
                            underlying, expiry,
                        )
                        continue
                    strategy = (
                        "short_put_spread"
                        if short["option_type"] == "put"
                        else "short_call_spread"
                    )
                    net_credit = float(short["avg_entry_price"]) - float(long_["avg_entry_price"])

                profit_target_price = round(net_credit * 0.5, 4)

                leg_records = [
                    {
                        "symbol": l["symbol"],
                        "side": "sell_to_open" if l["side"] == "short" else "buy_to_open",
                        "option_type": l["option_type"],
                        "strike": l["strike"],
                        "expiry": expiry.isoformat(),
                    }
                    for l in legs
                ]

                success = self.db.adopt_orphan(
                    underlying=underlying,
                    strategy=strategy,
                    net_credit=net_credit,
                    legs=leg_records,
                    profit_target_price=profit_target_price,
                )

                if success:
                    logger.info(
                        "[Orphan] Adopted %s %s %s net_credit=%.2f profit_target=%.3f",
                        underlying, expiry, strategy, net_credit, profit_target_price,
                    )
                    send_discord(
                        self.config.discord_webhook_url,
                        f"🔧 **Orphan adopted** — {underlying} {expiry} "
                        f"{strategy} net_credit={net_credit:.2f}",
                    )

            except Exception as exc:
                logger.error(
                    "[Orphan] Error processing %s %s: %s", underlying, expiry, exc
                )

    def _maybe_catchup_scan(self) -> None:
        """Fire a catch-up scan after a container restart that missed the 6:45 AM
        PT scheduled window.

        Five guards must ALL pass before a scan is fired:
          1. ``catchup_scan_enabled`` config flag (default True)
          2. Market is currently open
          3. Current PT time is at or past the scheduled scan window (6:45 AM PT)
          4. At least ``catchup_min_minutes_to_close`` minutes remain before close
          5. No scan has run today yet (idempotency via last_scan_date in bot_state)

        The idempotency marker is written by ``run_scan`` itself; multiple container
        restarts on the same day therefore produce exactly one scan (guard 5 skips
        the subsequent calls once the marker is present).
        """
        from zoneinfo import ZoneInfo

        if not self.config.catchup_scan_enabled:
            return

        if not _market_is_open():
            return

        LA = ZoneInfo("America/Los_Angeles")
        now_pt = datetime.now(tz=LA)

        # Guard 3: past the scheduled scan window (6:45 AM PT = 9:45 AM ET)
        if now_pt.time() < time(6, 45):
            return

        # Guard 4: enough trading day remaining to warrant new entries
        if _minutes_to_close() < self.config.catchup_min_minutes_to_close:
            logger.info(
                "[CatchupScan] Too close to market close (<%d min) — skipping",
                self.config.catchup_min_minutes_to_close,
            )
            return

        # Guard 5: idempotency — skip if a scan already ran today
        today_iso = date.today().isoformat()
        try:
            marker = self.db.load_state("last_scan_date", max_age_hours=48)
        except Exception:
            marker = None

        if marker and marker.get("date") == today_iso:
            logger.debug(
                "[CatchupScan] Scan already ran today (%s) — skipping", today_iso
            )
            return

        logger.info(
            "[CatchupScan] No scan recorded for %s — firing catch-up scan now",
            today_iso,
        )
        send_discord(
            self.config.discord_webhook_url,
            "⚡ **Catch-up scan** — container restarted after scheduled window; "
            "running now",
        )
        try:
            self.run_scan()
        except Exception as exc:
            logger.error("[CatchupScan] Catch-up scan failed: %s", exc)

    def run_scan(self) -> list[FilledOrder]:
        """
        One scan across all configured tickers.
        Can be called manually for testing without the scheduler.
        Returns list of FilledOrders (may be empty).
        """
        self.state.reset_for_new_day()
        logger.info("[Orchestrator] === SCAN START %s ===", date.today())

        # Persist scan-date marker so _maybe_catchup_scan skips on subsequent
        # container restarts within the same trading day.
        try:
            self.db.save_state("last_scan_date", {"date": date.today().isoformat()})
        except Exception:
            pass

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

        # Cumulative drawdown halt — highest priority gate
        if self.state.drawdown_halt or self.rm.is_drawdown_halted():
            dd_pct = self.rm.current_drawdown_pct()
            logger.warning(
                "[Orchestrator] DRAWDOWN HALT active (%.1f%% from peak). "
                "No new trades. Restart bot manually after review.",
                dd_pct * 100,
            )
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

        # Capture the pre-bullish-filter candidate pool — needed below so the
        # extra-strategy pass (short_call_spread, short_strangle) can route
        # off direction-appropriate shortlists instead of reusing the
        # bullish-only one. Must be captured before filter_ranked() narrows
        # scan_tickers down to bullish-only candidates.
        candidate_pool = scan_tickers

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

        # Multi-strategy pass — run extra_strategies on DIRECTION-APPROPRIATE
        # shortlists, not the bullish-only one used by the primary strategy.
        #
        # AUDIT FIX: previously every extra strategy (including
        # short_call_spread, a BEARISH bet) ran against `scan_tickers`, which
        # `filter_ranked()` had already narrowed down to bullish-only
        # candidates. Selling calls on a ticker the bot just confirmed is
        # bullish fights the trend instead of using it — structurally
        # backwards. Now each extra strategy gets the shortlist that actually
        # matches its market thesis:
        #   short_call_spread -> tickers routed "bearish" by the technical scorer
        #   short_strangle     -> tickers routed "neutral" (no strong direction)
        # Each extra strategy uses its own default config, independent of the
        # main strategy. Respects daily position caps across all strategies.
        _multi_strategy_routes: dict[str, list[str]] = {}
        _routing_ran = False
        if self.ticker_gate is not None and self.config.extra_strategies:
            try:
                _routed = self.ticker_gate.filter_ranked_multi_strategy(
                    candidate_pool,
                    top_n=self.config.scanner_shortlist_top_n,
                )
                for _t, _strategy_name in _routed:
                    _multi_strategy_routes.setdefault(_strategy_name, []).append(_t)
                _routing_ran = True
            except Exception as exc:
                logger.warning(
                    "[Orchestrator] Multi-strategy routing failed (extra "
                    "strategies will be skipped this scan): %s", exc
                )

        # --- Iron Condor gate (both conditions must clear) ---
        # Defined-risk neutral strategy, built and tested but DORMANT until:
        #   (1) the 30-trade walk-forward milestone has passed, AND
        #   (2) iron_condor_enabled is explicitly True.
        # When eligible, it joins the NEUTRAL routing bucket (same tickers the
        # short_strangle serves — it's the defined-risk upgrade to that bet).
        # The milestone removes the block; the flag is the human go-live
        # decision. Until both clear, it never trades.
        _extra_strategies = list(self.config.extra_strategies or [])
        if getattr(self.config, "iron_condor_enabled", False):
            try:
                _n_closed = len(self.db.get_all_closed_pnls()) if self.db else 0
            except Exception:
                _n_closed = 0
            _ic_min = getattr(self.config, "iron_condor_min_trades", 30)
            if _n_closed >= _ic_min:
                if "iron_condor" not in _extra_strategies:
                    _extra_strategies.append("iron_condor")
                    # Route it to the same neutral bucket as the strangle.
                    _neutral = _multi_strategy_routes.get("short_strangle", [])
                    if _neutral:
                        _multi_strategy_routes["iron_condor"] = list(_neutral)
                    logger.info(
                        "[Orchestrator] Iron Condor ACTIVE: %d closed trades >= %d "
                        "milestone and iron_condor_enabled=True", _n_closed, _ic_min,
                    )
            else:
                logger.info(
                    "[Orchestrator] Iron Condor enabled but gated: %d/%d closed "
                    "trades — stays dormant until walk-forward milestone", _n_closed, _ic_min,
                )

        for extra_name in _extra_strategies:
            if extra_name == self.config.strategy_name:
                continue  # already ran this one above

            # Direction-appropriate shortlist for this specific strategy.
            # A direction-specific strategy whose router bucket is empty is
            # SKIPPED (returns []), not run against the bullish shortlist — the
            # previous `... or scan_tickers` fallback did exactly that, selling
            # e.g. call spreads on tickers just confirmed bullish. See
            # _select_extra_tickers for the full rule.
            extra_tickers = _select_extra_tickers(
                extra_name, _multi_strategy_routes, _routing_ran, scan_tickers
            )
            if not extra_tickers:
                logger.info(
                    "[Orchestrator] No tickers routed to %s this scan — skipping",
                    extra_name,
                )
                continue

            # Check positions cap before starting next strategy pass
            positions = self.broker.get_positions()
            current_count = len([p for p in positions if p.get("asset_class") == "us_option"])
            if current_count >= self.config.max_positions_total:
                logger.info(
                    "[Orchestrator] Max positions reached — skipping extra strategy %s",
                    extra_name,
                )
                break

            logger.info(
                "[Orchestrator] === EXTRA STRATEGY PASS: %s (%d tickers: %s) ===",
                extra_name.upper(), len(extra_tickers), ", ".join(extra_tickers),
            )
            try:
                # Build a fresh pipeline using the extra strategy
                extra_cfg = get_strategy(extra_name, None).config                     if hasattr(get_strategy(extra_name, None), "config") else None
                self.pipeline.strategy = get_strategy(extra_name, extra_cfg)

                for ticker in extra_tickers:
                    positions = self.broker.get_positions()
                    current_count = len([
                        p for p in positions if p.get("asset_class") == "us_option"
                    ])
                    if current_count >= self.config.max_positions_total:
                        break

                    try:
                        # NOTE: run_for_ticker's signature is
                        # (ticker, regime_name="", regime_options_weight=0.0).
                        # The earlier call here passed regime=/open_trades=/
                        # sentiment_signals= — none of which exist on the method —
                        # so EVERY extra-strategy eval (csp, short_call_spread,
                        # short_strangle) raised TypeError and 3 of 4 live
                        # strategies never traded. Match the working main pass.
                        filled = self.pipeline.run_for_ticker(
                            ticker,
                            regime_name=regime_name,
                            regime_options_weight=options_weight,
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
            self.rm.update_peak_equity(equity)
        except Exception as exc:
            equity = self.state.equity if self.state.equity > 0 else self.rm.equity
            logger.warning("[EOD] Equity fetch failed, using cached $%.2f: %s", equity, exc)

        # Record today's 0DTE outcome for the dedicated circuit breaker's
        # consecutive-losing-day streak (arms the 14-trading-day cooldown
        # after 3 net-negative 0DTE days in a row). Must run at EOD AFTER
        # exits have closed, so realized P&L for the day is final.
        if getattr(self, "zero_dte_cb", None) is not None:
            try:
                _today_iso = date.today().isoformat()
                _0dte_pnl = self.db.get_today_realized_0dte_pnl(_today_iso) if self.db else 0.0
                _cb_state = self.zero_dte_cb.record_eod(date.today(), _0dte_pnl)
                logger.info(
                    "[EOD] 0DTE day recorded: realized=$%.2f streak=%d cooldown_until=%s",
                    _0dte_pnl, _cb_state.get("consec_losing_days", 0),
                    _cb_state.get("cooldown_until_iso"),
                )
            except Exception as exc:
                logger.warning("[EOD] 0DTE circuit-breaker record failed (non-fatal): %s", exc)

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
                cur = self.db._execute(
                    conn,
                    "SELECT realized_pnl FROM trades WHERE trade_date=? AND realized_pnl IS NOT NULL",
                    (today,)
                )
                all_pnls = [row[0] for row in cur.fetchall()]
        except Exception as exc:
            logger.error("[Orchestrator] EOD realized_pnl query failed: %s", exc)
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

        _dd_pct  = self.rm.current_drawdown_pct()
        _dd_line = (
            f"⚠️ DRAWDOWN HALT ACTIVE: {_dd_pct:.1%} from peak — no new trades\n"
            if self.state.drawdown_halt else
            f"Drawdown: {_dd_pct:.1%} from peak (limit={self.config.risk_config.max_drawdown_pct:.1%})\n"
        )
        summary = (
            f"Daily Summary - {date.today()}\n"
            f"Strategy: {self.config.strategy_name.upper()}  "
            f"{'[PAPER]' if self.config.paper else '[LIVE]'}\n"
            f"Trades today: {len(self.state.filled_today)}  |  Open: {len(open_trades)}\n"
            f"Today P&L: ${self.state.daily_realized_pnl:+.2f} realized  "
            f"${self.state.daily_unrealized_pnl:+.2f} unrealized\n"
            f"Week-to-date: ${wtd_pnl:+.2f} ({len(wtd_pnls)} closed trades)\n"
            f"Total trades: {n_total}/30 to walk-forward unlock\n"
            f"{_dd_line}"
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
                cur = self.db._execute(
                    conn,
                    "SELECT realized_pnl FROM trades "
                    "WHERE status NOT IN ('open') AND realized_pnl IS NOT NULL"
                )
                return [row[0] for row in cur.fetchall()]
        except Exception as exc:
            logger.error("[Orchestrator] _query_all_pnls failed: %s", exc)
            return []

    def _query_pnls_since(self, since_date):
        try:
            with self.db._get_conn() as conn:
                cur = self.db._execute(
                    conn,
                    "SELECT realized_pnl FROM trades "
                    "WHERE status NOT IN ('open') AND realized_pnl IS NOT NULL "
                    "AND trade_date >= ?",
                    (since_date.isoformat(),)
                )
                return [row[0] for row in cur.fetchall()]
        except Exception as exc:
            logger.error("[Orchestrator] _query_pnls_since failed: %s", exc)
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

    def _query_equity_curve(self) -> list[dict]:
        """
        Pull the daily account equity from the DB and return as a list of
        {"date": str, "value": float} sorted oldest -> newest.
        Falls back to reconstructing from closed trade P&L if the equity
        snapshot table does not exist.
        """
        try:
            with self.db._get_conn() as conn:
                try:
                    cur = self.db._execute(
                        conn,
                        "SELECT snapshot_date, equity "
                        "FROM equity_snapshots "
                        "ORDER BY snapshot_date ASC"
                    )
                    rows = cur.fetchall()
                    if rows:
                        return [{"date": r[0], "value": float(r[1])} for r in rows]
                except Exception:
                    pass  # table may not exist yet — fall through to reconstruction

                # Reconstruction: build cumulative curve from closed trade P&L
                cur = self.db._execute(
                    conn,
                    "SELECT trade_date, realized_pnl FROM trades "
                    "WHERE status NOT IN ('open') AND realized_pnl IS NOT NULL "
                    "ORDER BY trade_date ASC"
                )
                rows = cur.fetchall()
                if not rows:
                    return []
                equity = self.state.equity or 100_000.0
                # Walk backwards to get starting capital
                total_pnl = sum(r[1] for r in rows)
                start_eq  = equity - total_pnl
                curve, running = [], start_eq
                for trade_date, pnl in rows:
                    running += pnl
                    curve.append({"date": trade_date, "value": round(running, 2)})
                return curve
        except Exception as exc:
            logger.debug("[StatValidation] Could not build equity curve: %s", exc)
            return []

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
                 "The tuner can now make statistically valid parameter adjustments. "
                 "GATED FEATURES NOW ELIGIBLE: Iron Condor (defined-risk neutral "
                 "strategy) via iron_condor_enabled=True; credit_regime (HYG/IEF "
                 "credit-stress regime input) via credit_regime_enabled=True; "
                 "analyst_revisions (upgrade/downgrade momentum) via "
                 "analyst_revisions_enabled=True. Confirm stat_validation shows "
                 "genuine edge below before enabling. Position rolling can also "
                 "be revisited per the saved plan."),
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

        # Statistical edge validation — runs at 30, 60, 90 trade milestones
        stat_block = ""
        if n >= 30:
            try:
                curve = self._query_equity_curve()
                initial_capital = self.state.equity or 100_000.0
                if len(curve) >= 10:
                    validation = run_all_validations(
                        curve,
                        initial_capital=initial_capital,
                        n_strategies=4,   # ShortPutSpread, CashSecuredPut, ShortStrangle, 0DTE
                    )
                    stat_block = NL + format_validation_discord(validation) + NL
                else:
                    stat_block = NL + "*(Statistical validation: equity curve too short — run again after more trading days)*" + NL
            except Exception as exc:
                logger.warning("[Milestones] Statistical validation failed (non-fatal): %s", exc)

        msg = title + NL + detail + NL + NL + pnl_block + readiness + stat_block
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
