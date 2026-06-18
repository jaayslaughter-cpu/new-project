"""
0DTE GEX Scalper — intraday SPX/SPY options strategy.

Sells 0DTE vertical spreads anchored to the Gamma Exposure (GEX) pin strike.
The GEX pin is the strike with the highest positive dealer gamma — price
gravitates toward it as dealers delta-hedge, creating a mean-reverting force.

Architecture
------------
  ZeroDTEConfig      — all tunable parameters in one dataclass
  EventCalendar      — FOMC/CPI/NFP/OPEX gate (blocks or halves size)
  SessionClassifier  — time-of-day gate (OPENING/CORE/GAMMA_RAMP/CLOSE)
  GEXEngine          — computes GEX pin from live Alpaca options chain
  KellySizer         — half-Kelly position sizing from rolling trade history
  SlippageGuard      — spread quality check + alternative strike finder
  GammaAdjuster      — adjusts params when live GEX diverges from plan GEX
  ZeroDTEStrategy    — orchestrates all of the above; returns ApprovedOrder
  ZeroDTEMonitor     — 15-second exit loop (TP/trailing stop/hard stop/EOD)

Integration
-----------
  The orchestrator calls:
    strategy = ZeroDTEStrategy(config, broker)
    monitor  = ZeroDTEMonitor(config, broker, db)

  Scan schedule (apscheduler): every 2 minutes from 9:32 to 14:00 ET.
  Monitor schedule: every 15 seconds from first fill until 15:35 ET.

Key differences vs monthly strategies
--------------------------------------
  - 1-minute scan cycle (not 15 min)
  - Hard no-new-trades cutoff at 14:00 ET (2 PM)
  - Hard force-close at 15:30 ET (30 min before expiry)
  - Kelly sizing not fixed-percent
  - GEX pin replaces delta-selection for strike choice
  - All spreads are SPX weekly (SPXW) expiring today

Sources
-------
  GEX pin calculation    — gamma-main/scalper.py (Tradier → Alpaca adapted)
  Session classifier     — 0dte-strategy-main/signal/generator.py
  Event calendar         — 0dte-strategy-main/risk/event_calendar.py
  Kelly sizing           — gamma-main/scalper.py calculate_position_size_kelly()
  Slippage guard         — 0dte_bot-main/slippage_minimizer.py
  Gamma adjuster         — 0dte_bot-main/gamma_adjuster.py
  Stop/TP logic          — gamma-main/monitor.py
  All rewritten for Alpaca API and our existing broker/contracts layer.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pytz

from .broker import AlpacaBroker
from .contracts import ApprovedOrder, FilledOrder, OrderLeg, OptionType
from .exceptions import LiquidityFilterError, PipelineConnectionError
from .spread_math import validate_spread_inputs, calc_spread
from .circuit_breaker import data_circuit_breaker as _cb

logger = logging.getLogger(__name__)
ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ZeroDTEConfig:
    """All tunable parameters for the 0DTE GEX scalper."""

    # --- Underlying ---
    underlying: str = "SPY"          # SPY (not SPX — Alpaca supports SPY options)
    spread_width: float = 5.0        # Dollars between short and long strike

    # --- Entry gates ---
    vix_floor: float = 12.0          # Skip if VIX below this (not enough premium)
    vix_ceiling: float = 35.0        # Skip if VIX above this (too dangerous)
    rsi_min: float = 40.0            # SPY RSI must be above this
    rsi_max: float = 80.0            # SPY RSI must be below this
    max_gap_pct: float = 1.5         # Skip if overnight gap > 1.5% (SPY gaps >0.5% on ~40% of days)
    max_consec_down_days: int = 5    # Skip if N+ consecutive red days
    skip_friday: bool = False        # SPY 0DTE trades every day
    min_expected_move_pct: float = 0.08  # Min expected 2-hr move as % of price (lowered from 0.15)

    # --- Credit filters ---
    min_credit_morning: float = 0.30  # Before 11 AM ET
    min_credit_midday: float = 0.40   # 11 AM–1 PM ET
    min_credit_afternoon: float = 0.55 # After 1 PM ET
    min_credit_absolute: float = 0.20  # Never trade below this regardless of time
    max_spread_pct: float = 0.30       # Max bid-ask spread as % of mid

    # --- GEX parameters ---
    gex_window_pct: float = 0.03      # Look for pin within 3% of spot
    gex_min_oi: int = 100             # Minimum OI to include in GEX calc
    pin_distance_min: float = 2.0     # Min $ between spot and short strike
    pin_distance_max: float = 15.0    # Max $ between spot and short strike

    # --- Position sizing (Kelly) ---
    starting_capital: float = 25_000.0
    max_contracts: int = 3
    stop_loss_per_contract: float = 75.0   # Max dollar risk per contract for sizing
    bootstrap_win_rate: float = 0.55        # Bootstrap WR before real data
    bootstrap_avg_win: float = 45.0         # Bootstrap avg win ($) per contract
    bootstrap_avg_loss: float = 60.0        # Bootstrap avg loss ($) per contract
    min_kelly_trades: int = 15              # Trades needed before using real stats
    kelly_fraction: float = 0.5             # Use half-Kelly

    # --- Exit parameters ---
    profit_target_pct: float = 0.50        # Close at 50% of max profit
    stop_loss_pct: float = 1.00            # Close when spread costs 2x credit (100% loss)
    trailing_stop_trigger_pct: float = 0.25 # Activate trailing at 25% profit
    trailing_stop_floor_pct: float = 0.10  # Lock in 10% profit minimum once trailing
    max_daily_positions: int = 3

    # --- Time gates (ET) ---
    entry_cutoff_hour: int = 14       # No new trades at or after 2:00 PM
    force_close_hour: int = 15
    force_close_minute: int = 30      # Force close all at 3:30 PM

    # --- Monitor ---
    monitor_poll_seconds: int = 15

    # --- VWAP stretch signal ---
    vwap_stretch_threshold: float = 0.003   # 0.3%: price must be this far from VWAP to qualify
    vwap_reclaim_threshold: float = 0.0021  # 0.21%: how close price must return to VWAP to confirm reclaim
    vwap_require_reclaim: bool = False       # If True, only enter after a partial VWAP reclaim is detected
    vwap_cooldown_seconds: int = 120         # Min seconds between VWAP stretch signals

    # --- Misc ---
    discord_webhook_url: str = ""
    paper: bool = True
    account_state_file: str = "/tmp/0dte_account_state.json"


# ---------------------------------------------------------------------------
# Event Calendar
# ---------------------------------------------------------------------------

# 2026 FOMC statement dates (federalreserve.gov)
_FOMC_2026 = {
    date(2026, 1, 28), date(2026, 1, 29),
    date(2026, 3, 17), date(2026, 3, 18),
    date(2026, 5, 5),  date(2026, 5, 6),
    date(2026, 6, 16), date(2026, 6, 17),
    date(2026, 7, 28), date(2026, 7, 29),
    date(2026, 9, 15), date(2026, 9, 16),
    date(2026, 10, 27), date(2026, 10, 28),
    date(2026, 12, 15), date(2026, 12, 16),
}

# 2027 FOMC statement dates (federalreserve.gov — published in advance)
_FOMC_2027 = {
    date(2027, 1, 26), date(2027, 1, 27),
    date(2027, 3, 16), date(2027, 3, 17),
    date(2027, 5, 4),  date(2027, 5, 5),
    date(2027, 6, 15), date(2027, 6, 16),
    date(2027, 7, 27), date(2027, 7, 28),
    date(2027, 9, 21), date(2027, 9, 22),
    date(2027, 10, 26), date(2027, 10, 27),
    date(2027, 12, 14), date(2027, 12, 15),
}

# Hardcoded CPI release dates — Bureau of Labor Statistics publishes the
# schedule annually at bls.gov/schedule/news_release/cpi.htm
# CPI is released at 8:30 AM ET on these dates; 0DTE gets half-size.
_CPI_2026 = {
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 10), date(2026, 5, 13), date(2026, 6, 11),
    date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 11),
    date(2026, 10, 14), date(2026, 11, 12), date(2026, 12, 11),
}
_CPI_2027 = {
    date(2027, 1, 13), date(2027, 2, 10), date(2027, 3, 11),
    date(2027, 4, 13), date(2027, 5, 12), date(2027, 6, 11),
    date(2027, 7, 14), date(2027, 8, 11), date(2027, 9, 14),
    date(2027, 10, 13), date(2027, 11, 10), date(2027, 12, 14),
}

# Hardcoded NFP (Non-Farm Payroll) release dates — BLS publishes at 8:30 AM ET
# on the first Friday of each month. 0DTE gets half-size on these days.
_NFP_2026 = {
    date(2026, 1, 9),  date(2026, 2, 6),  date(2026, 3, 6),
    date(2026, 4, 3),  date(2026, 5, 8),  date(2026, 6, 5),
    date(2026, 7, 10), date(2026, 8, 7),  date(2026, 9, 4),
    date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
}
_NFP_2027 = {
    date(2027, 1, 8),  date(2027, 2, 5),  date(2027, 3, 5),
    date(2027, 4, 2),  date(2027, 5, 7),  date(2027, 6, 4),
    date(2027, 7, 9),  date(2027, 8, 6),  date(2027, 9, 3),
    date(2027, 10, 1), date(2027, 11, 5), date(2027, 12, 3),
}


def _monthly_opex(year: int) -> set[date]:
    """3rd Friday of each month."""
    dates = set()
    for month in range(1, 13):
        first = date(year, month, 1)
        days_to_fri = (4 - first.weekday()) % 7
        dates.add(first + timedelta(days=days_to_fri + 14))
    return dates


class EventCalendar:
    """
    Classifies today's risk level based on macro events.

    Returns a risk_multiplier (0.0 = no trade, 0.5 = half size, 1.0 = full).
    """

    def __init__(self):
        year = date.today().year
        # Combine all known FOMC dates — 0DTE never trades on FOMC announcement days
        self._fomc = _FOMC_2026 | _FOMC_2027
        self._opex = _monthly_opex(year)
        self._triple = {d for d in self._opex if d.month in (3, 6, 9, 12)}
        # Seed CPI and NFP from hardcoded BLS schedule — always populated,
        # not dependent on Finnhub API key being present.
        self._cpi: set[date] = _CPI_2026 | _CPI_2027
        self._nfp: set[date] = _NFP_2026 | _NFP_2027
        # Finnhub can add additional dates if the key is available
        self._try_load_finnhub()

    def _try_load_finnhub(self) -> None:
        key = os.getenv("FINNHUB_API_KEY", "")
        if not key:
            return
        try:
            import httpx
            year = date.today().year
            resp = httpx.get(
                "https://finnhub.io/api/v1/calendar/economic",
                params={"from": f"{year}-01-01", "to": f"{year}-12-31", "token": key},
                timeout=10,
            )
            for ev in resp.json().get("economicCalendar", []):
                name = (ev.get("event") or "").lower()
                ds   = (ev.get("date") or "")[:10]
                if not ds:
                    continue
                try:
                    d = date.fromisoformat(ds)
                except ValueError:
                    continue
                if "cpi" in name or "consumer price" in name:
                    self._cpi.add(d)
                elif "nonfarm" in name or "nfp" in name:
                    self._nfp.add(d)
            logger.info("[EventCalendar] Loaded %d CPI, %d NFP dates from Finnhub",
                        len(self._cpi), len(self._nfp))
        except Exception as exc:
            logger.debug("[EventCalendar] Finnhub load skipped: %s", exc)

    def classify(self, d: Optional[date] = None) -> dict:
        if d is None:
            d = date.today()
        is_fomc = d in self._fomc
        is_cpi  = d in self._cpi
        is_nfp  = d in self._nfp
        is_opex = d in self._opex
        is_tw   = d in self._triple

        if is_fomc or is_cpi or is_nfp:
            mode, mult = "VANNA_DOMINANT", 0.5
        elif is_tw:
            mode, mult = "TRIPLE_WITCHING", 0.7
        elif is_opex:
            mode, mult = "OPEX", 0.85
        else:
            mode, mult = "NORMAL", 1.0

        return {
            "is_fomc": is_fomc, "is_cpi": is_cpi, "is_nfp": is_nfp,
            "is_opex": is_opex, "is_triple_witching": is_tw,
            "risk_multiplier": mult, "mode": mode,
        }


# ---------------------------------------------------------------------------
# Session Classifier
# ---------------------------------------------------------------------------

class SessionClassifier:
    """
    Classifies the intraday time session for GEX-based trading.

    OPENING    09:30–10:00  Walls unstable — no new positions
    CORE       10:00–14:00  Best signal window — full size
    GAMMA_RAMP 14:00–15:30  Exponential gamma — reduced confidence
    CLOSE      15:30–16:00  Force-close only — no new entries

    Source: 0dte-strategy-main/signal/generator.py classify_session()
    Extended with exact minute boundaries matching gamma-main cutoffs.
    """

    SESSIONS = {
        "OPENING":    (9*60+30,  10*60),
        "CORE":       (10*60,    14*60),
        "GAMMA_RAMP": (14*60,    15*60+30),
        "CLOSE":      (15*60+30, 16*60),
    }

    # Confidence modifier per session
    MODIFIERS = {
        "OPENING":    0.0,    # never trade
        "CORE":       1.0,
        "GAMMA_RAMP": 0.85,
        "CLOSE":      0.0,    # never open
    }

    def classify(self, now: Optional[datetime] = None) -> str:
        if now is None:
            now = datetime.now(tz=ET)
        m = now.hour * 60 + now.minute
        for name, (start, end) in self.SESSIONS.items():
            if start <= m < end:
                return name
        return "CLOSED"

    def can_enter(self, session: str) -> bool:
        return session in ("CORE", "GAMMA_RAMP")

    def confidence_modifier(self, session: str) -> float:
        return self.MODIFIERS.get(session, 0.0)


# ---------------------------------------------------------------------------
# GEX Engine
# ---------------------------------------------------------------------------

@dataclass
class GEXPin:
    strike: float
    gex_value: float          # Total GEX at this strike (B$)
    distance_from_spot: float # Absolute distance in dollars
    side: str                 # "ABOVE" or "BELOW"
    regime: str               # "POSITIVE_GAMMA" or "NEGATIVE_GAMMA"


class GEXEngine:
    """
    Computes the GEX pin strike from the live Alpaca options chain.

    GEX formula per strike:
        calls: +gamma * OI * 100 * spot^2
        puts:  -gamma * OI * 100 * spot^2

    The pin strike is the one with the highest positive net GEX within
    gex_window_pct of the current spot price.

    If total GEX is positive, market is in positive-gamma regime (pinning).
    If negative, dealers are short gamma (trending/volatile).

    Source: gamma-main/scalper.py calculate_gex_pin() — adapted for Alpaca.
    """

    def __init__(self, broker: AlpacaBroker, config: ZeroDTEConfig):
        self.broker = broker
        self.cfg    = config

    def compute(self, spot: float) -> Optional[GEXPin]:
        """Fetch today's option chain and return the GEX pin strike."""
        if not _cb.is_available("alpaca_chain_0dte"):
            logger.debug("[GEX] Chain fetch skipped — circuit breaker OPEN")
            return None

        try:
            today_str = date.today().isoformat()
            chain = self.broker.get_option_chain(
                self.cfg.underlying,
                expiration_date=today_str,
                option_type=None,   # both calls and puts
            )
            _cb.record_success("alpaca_chain_0dte")
        except Exception as exc:
            _cb.record_failure("alpaca_chain_0dte", str(exc))
            logger.warning("[GEX] Chain fetch failed: %s", exc)
            return None

        if not chain:
            logger.warning("[GEX] Empty chain for %s today", self.cfg.underlying)
            return None

        window = spot * self.cfg.gex_window_pct
        gex_by_strike: dict[float, float] = defaultdict(float)
        total_gex = 0.0

        for symbol, snap in chain.items():
            # Parse strike and option_type from OCC symbol string
            # Format: SPY260617C00580000 -> last 9 chars = C00580000
            # Broker returns flat dict: {bid, ask, iv, delta, gamma, theta, vega, rho}
            try:
                # OCC: ticker(var) + YYMMDD(6) + C/P(1) + strike*1000 zero-padded(8)
                opt_type_char = symbol[-9]          # 'C' or 'P'
                strike        = int(symbol[-8:]) / 1000.0
                opt_type      = "call" if opt_type_char == "C" else "put"
            except (IndexError, ValueError):
                continue

            # Gamma comes from flat dict directly (broker normalises greeks)
            gamma = snap.get("gamma") or 0
            if not gamma:
                continue

            # OI: broker doesn't always return it — use bid/ask proxy
            # If gamma is present and non-zero, treat as having valid OI
            # Real OI gating via gex_min_oi is skipped when OI unavailable
            oi = snap.get("open_interest") or 1   # default 1 so gamma*OI is non-zero

            try:
                strike = float(strike)
                oi     = int(oi)
                gamma  = float(gamma)
            except (TypeError, ValueError):
                continue

            if abs(strike - spot) > window:
                continue   # outside GEX window — skip early

            # GEX = gamma * OI * contract_size * spot^2
            gex = gamma * oi * 100 * (spot ** 2)

            if opt_type == "call":
                gex_by_strike[strike] += gex
                total_gex += gex
            elif opt_type == "put":
                gex_by_strike[strike] -= gex
                total_gex -= gex

        if not gex_by_strike:
            logger.warning("[GEX] No valid strikes in chain")
            return None

        regime = "POSITIVE_GAMMA" if total_gex >= 0 else "NEGATIVE_GAMMA"

        # Find the highest positive-GEX strike near spot
        near = [(s, g) for s, g in gex_by_strike.items()
                if abs(s - spot) <= window and g > 0]

        if near:
            pin_strike, pin_gex = max(near, key=lambda x: x[1])
        else:
            # Fallback: highest absolute GEX near spot
            near_all = [(s, g) for s, g in gex_by_strike.items()
                        if abs(s - spot) <= window]
            if not near_all:
                logger.warning("[GEX] No strikes found within %.1f%% of spot", self.cfg.gex_window_pct * 100)
                return None
            pin_strike, pin_gex = max(near_all, key=lambda x: abs(x[1]))

        side = "ABOVE" if pin_strike > spot else "BELOW"
        logger.info("[GEX] Pin=%s distance=%.2f regime=%s total_gex=%.1fB",
                    pin_strike, abs(spot - pin_strike),
                    regime, total_gex / 1e9)

        return GEXPin(
            strike=pin_strike,
            gex_value=pin_gex,
            distance_from_spot=abs(spot - pin_strike),
            side=side,
            regime=regime,
        )

    def select_strikes(
        self, spot: float, pin: GEXPin, cfg: ZeroDTEConfig
    ) -> Optional[dict]:
        """
        Given a GEX pin, select the optimal spread strikes.

        Logic (from 574-day validated signal in 0dte-strategy-main):
          NEGATIVE_GAMMA + spot near put_wall (pin below spot) -> BULLISH
            -> sell put spread below spot
          NEGATIVE_GAMMA + spot near call_wall (pin above spot) -> BEARISH
            -> sell call spread above spot
          POSITIVE_GAMMA (pinning)
            -> sell whichever side has more distance from spot

        Returns dict with: strategy, short_strike, long_strike, direction
        """
        dist = pin.distance_from_spot

        if dist < cfg.pin_distance_min or dist > cfg.pin_distance_max:
            logger.info("[GEX] Pin distance %.2f outside [%.2f, %.2f] — no trade",
                        dist, cfg.pin_distance_min, cfg.pin_distance_max)
            return None

        if pin.regime == "NEGATIVE_GAMMA":
            if pin.side == "BELOW":
                # Pin below spot -> bounce expected -> sell put spread
                short_strike = round(pin.strike - 0.50, 0)
                long_strike  = short_strike - cfg.spread_width
                strategy     = "PUT_SPREAD"
                direction    = "BULLISH"
            else:
                # Pin above spot -> ceiling expected -> sell call spread
                short_strike = round(pin.strike + 0.50, 0)
                long_strike  = short_strike + cfg.spread_width
                strategy     = "CALL_SPREAD"
                direction    = "BEARISH"
        else:
            # POSITIVE_GAMMA: sell the side further from spot
            if pin.side == "ABOVE":
                short_strike = round(pin.strike + cfg.spread_width, 0)
                long_strike  = short_strike + cfg.spread_width
                strategy     = "CALL_SPREAD"
                direction    = "NEUTRAL_BEAR"
            else:
                short_strike = round(pin.strike - cfg.spread_width, 0)
                long_strike  = short_strike - cfg.spread_width
                strategy     = "PUT_SPREAD"
                direction    = "NEUTRAL_BULL"

        # Sanity: short strike must be OTM
        if strategy == "PUT_SPREAD" and short_strike >= spot:
            logger.info("[GEX] Short put %.0f >= spot %.2f — too close", short_strike, spot)
            return None
        if strategy == "CALL_SPREAD" and short_strike <= spot:
            logger.info("[GEX] Short call %.0f <= spot %.2f — too close", short_strike, spot)
            return None

        return {
            "strategy":     strategy,
            "direction":    direction,
            "short_strike": short_strike,
            "long_strike":  long_strike,
        }


# ---------------------------------------------------------------------------
# Kelly Position Sizer
# ---------------------------------------------------------------------------

class KellySizer:
    """
    Half-Kelly position sizing using rolling trade history.

    Bootstraps with configured stats until min_kelly_trades closed trades
    are available. Halts trading if account drops below 50% of starting capital.

    Source: gamma-main/scalper.py calculate_position_size_kelly() — rewritten
    to use our DB instead of a JSON file, with explicit safety checks.
    """

    def __init__(self, cfg: ZeroDTEConfig, db=None):
        self.cfg = cfg
        self.db  = db

    def size(self, account_equity: float, closed_pnls: Optional[list] = None) -> int:
        """
        Returns number of contracts to trade (0 = don't trade).

        Parameters
        ----------
        account_equity : float
            Current account equity from Alpaca.
        closed_pnls : list of float or None
            Recent per-contract P&L from closed 0DTE trades.
        """
        # Safety halt
        floor = self.cfg.starting_capital * 0.50
        if account_equity < floor:
            logger.warning(
                "[Kelly] Account $%.0f below 50%% floor $%.0f — halt",
                account_equity, floor
            )
            return 0

        pnls = closed_pnls or []

        if len(pnls) >= self.cfg.min_kelly_trades:
            wins   = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            if len(wins) >= 5 and len(losses) >= 2:
                win_rate = len(wins) / len(pnls)
                avg_win  = sum(wins)   / len(wins)
                avg_loss = abs(sum(losses) / len(losses))
            else:
                win_rate, avg_win, avg_loss = self._bootstrap()
        else:
            win_rate, avg_win, avg_loss = self._bootstrap()

        if avg_win <= 0:
            return 1

        # Kelly fraction: f = (p*W - (1-p)*L) / W
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        half_k  = kelly_f * self.cfg.kelly_fraction

        if half_k <= 0:
            logger.info("[Kelly] Negative Kelly (%.3f) — sizing to 1 contract", half_k)
            return 1

        contracts = int((account_equity * half_k) / self.cfg.stop_loss_per_contract)
        contracts = max(1, min(contracts, self.cfg.max_contracts))

        logger.info(
            "[Kelly] equity=$%.0f WR=%.0f%% W=$%.0f L=$%.0f -> %d contract(s)",
            account_equity, win_rate * 100, avg_win, avg_loss, contracts
        )
        return contracts

    def _bootstrap(self) -> tuple[float, float, float]:
        return (
            self.cfg.bootstrap_win_rate,
            self.cfg.bootstrap_avg_win,
            self.cfg.bootstrap_avg_loss,
        )


# ---------------------------------------------------------------------------
# Slippage Guard
# ---------------------------------------------------------------------------

class SlippageGuard:
    """
    Checks bid-ask spread quality before entry.
    Finds alternative strikes if primary is too wide.

    Source: 0dte_bot-main/slippage_minimizer.py — simplified for our use case.
    """

    def __init__(self, cfg: ZeroDTEConfig):
        self.cfg = cfg
        self._spread_history: dict[float, list] = {}

    def check(
        self,
        short_bid: float, short_ask: float,
        long_bid: float,  long_ask: float,
        expected_credit: float,
    ) -> tuple[bool, str]:
        """
        Returns (ok, reason).
        ok=False means don't trade due to wide spread.
        """
        if expected_credit <= 0:
            return False, "zero or negative credit"

        short_spread = short_ask - short_bid
        long_spread  = long_ask  - long_bid
        net_spread   = abs(short_spread - long_spread)
        max_spread   = expected_credit * self.cfg.max_spread_pct

        if net_spread > max_spread:
            return False, (
                f"net spread ${net_spread:.3f} > max ${max_spread:.3f} "
                f"({self.cfg.max_spread_pct:.0%} of credit ${expected_credit:.2f})"
            )

        return True, "ok"

    def mid_credit(
        self,
        short_bid: float, short_ask: float,
        long_bid: float,  long_ask: float,
    ) -> float:
        """Net credit at mid-price."""
        return round(((short_bid + short_ask) / 2) - ((long_bid + long_ask) / 2), 3)

    def limit_price(self, credit: float, haircut: float = 0.05) -> float:
        """Limit order price: slightly below mid to ensure fill."""
        return round(credit * (1.0 - haircut), 2)


# ---------------------------------------------------------------------------
# Gamma Adjuster
# ---------------------------------------------------------------------------

class GammaAdjuster:
    """
    Dynamically adjusts position size and stop when live GEX diverges
    from the GEX at time of plan creation.

    Source: 0dte_bot-main/gamma_adjuster.py — simplified to two adjustments:
      1. Position size reduction (if GEX shifted significantly against us)
      2. Stop width expansion (if total gamma spiked = more volatile)
    """

    TOTAL_GEX_THRESHOLD = 0.15    # 15% change in total gamma
    FLIP_THRESHOLD      = 0.005   # 0.5% of spot change in flip point

    def adjust(
        self,
        plan_gex: float,    # Total GEX at plan creation (signed $B)
        live_gex: float,    # Current total GEX
        plan_flip: float,   # Pin strike at plan creation
        live_flip: float,   # Current pin strike
        spot: float,
        n_contracts: int,
        stop_multiplier: float,
    ) -> tuple[int, float]:
        """
        Returns (adjusted_contracts, adjusted_stop_multiplier).
        """
        if plan_gex == 0:
            return n_contracts, stop_multiplier

        total_gex_change = abs(live_gex - plan_gex) / abs(plan_gex)
        flip_change      = abs(live_flip - plan_flip) / spot

        size_factor = 1.0
        stop_factor = 1.0

        if total_gex_change > self.TOTAL_GEX_THRESHOLD:
            size_factor *= 0.8   # Reduce size 20% on high gamma change
            stop_factor *= 1.2   # Widen stop 20%
            logger.info(
                "[GammaAdj] GEX changed %.0f%% — reducing size 20%%, widening stop 20%%",
                total_gex_change * 100
            )

        if flip_change > self.FLIP_THRESHOLD:
            if (live_flip > spot and plan_flip <= spot) or (live_flip < spot and plan_flip >= spot):
                # Flip point crossed spot — significant regime shift
                size_factor *= 0.5
                logger.warning("[GammaAdj] GEX flip crossed spot — halving size")

        new_contracts = max(1, int(n_contracts * size_factor))
        new_stop      = round(stop_multiplier * stop_factor, 2)

        return new_contracts, new_stop


# ---------------------------------------------------------------------------
# Pre-entry filters
# ---------------------------------------------------------------------------

def _get_spy_data() -> dict:
    """Fetch SPY RSI, consecutive down days, and overnight gap via yfinance."""
    if not _cb.is_available("yfinance_spy_0dte"):
        return {}
    try:
        import yfinance as yf
        import numpy as np

        hist = yf.Ticker("SPY").history(period="35d", interval="1d")
        if hist.empty or len(hist) < 15:
            _cb.record_failure("yfinance_spy_0dte", "insufficient bars")
            return {}

        _cb.record_success("yfinance_spy_0dte")
        closes = hist["Close"].values.flatten()
        opens  = hist["Open"].values.flatten()

        # RSI (14-period)
        delta = np.diff(closes)
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        ag = np.convolve(gain, np.ones(14) / 14, mode="valid")[-1]
        al = np.convolve(loss, np.ones(14) / 14, mode="valid")[-1]
        rsi = 100.0 if al == 0 else 0.0 if ag == 0 else 100 - 100 / (1 + ag / al)

        # Consecutive down days
        consec_down = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                consec_down += 1
            else:
                break

        # Overnight gap (today open vs yesterday close)
        gap_pct = 0.0
        if len(closes) >= 2 and closes[-2] != 0:
            gap_pct = abs(opens[-1] - closes[-2]) / closes[-2] * 100

        # Current price and VIX
        spy_price = float(closes[-1])

        return {
            "rsi":         round(rsi, 1),
            "consec_down": consec_down,
            "gap_pct":     round(gap_pct, 3),
            "spy_price":   spy_price,
        }
    except Exception as exc:
        _cb.record_failure("yfinance_spy_0dte", str(exc))
        logger.warning("[0DTE filters] SPY data failed: %s", exc)
        return {}


def _get_vix() -> Optional[float]:
    """Fetch VIX from yfinance."""
    if not _cb.is_available("yfinance_vix"):
        return None
    try:
        import yfinance as yf
        price = yf.Ticker("^VIX").fast_info.get("lastPrice")
        if price and float(price) > 0:
            _cb.record_success("yfinance_vix")
            return float(price)
        hist = yf.Ticker("^VIX").history(period="2d")
        if not hist.empty:
            _cb.record_success("yfinance_vix")
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        _cb.record_failure("yfinance_vix", str(exc))
    return None


def _expected_move_pct(spy_price: float, vix: float, hours: float = 2.0) -> float:
    """
    Expected N-hour move as a fraction of spot.
    Formula: VIX/100 * sqrt(hours / (252 * 6.5))
    """
    return (vix / 100) * math.sqrt(hours / (252 * 6.5))


# ---------------------------------------------------------------------------
# Main Strategy Class
# ---------------------------------------------------------------------------

class VWAPStretchFilter:
    """
    Intraday VWAP stretch/reclaim signal filter for 0DTE entries.

    Computes a running VWAP from today's 1-minute SPY bars (fetched from
    Alpaca) and detects whether price has stretched ≥ vwap_stretch_threshold
    away from VWAP in a direction that aligns with the GEX-selected spread.

    Optionally confirms a partial reclaim toward VWAP (the classic
    "stretch and snap" entry pattern from vwap-reclaim strategy).

    Fail-open: if Alpaca bar data is unavailable the filter returns
    confirmed=True so the GEX+credit gates still control the trade.

    Direction alignment:
        BULLISH / NEUTRAL_BULL  →  price must be stretched *below* VWAP
                                   (mean-reversion bounce expected)
        BEARISH / NEUTRAL_BEAR  →  price must be stretched *above* VWAP
                                   (mean-reversion fade expected)
    """

    def __init__(self, cfg: ZeroDTEConfig, broker: AlpacaBroker):
        self.cfg    = cfg
        self.broker = broker
        self._last_signal_time: Optional[datetime] = None

    def check(self, spot: float, direction: str) -> dict:
        """
        Returns a dict:
            confirmed   bool   — True if entry is allowed
            stretch_pct float  — % distance from VWAP (signed; negative = below)
            vwap        float  — running VWAP at last bar
            reason      str    — human-readable gate result
        """
        result = {
            "confirmed": True,
            "stretch_pct": 0.0,
            "vwap": 0.0,
            "reason": "vwap_data_unavailable_passthrough",
        }

        bars = self._fetch_today_bars()
        if not bars:
            logger.warning("[VWAPFilter] No intraday bars — filter bypassed")
            return result

        vwap = self._compute_vwap(bars)
        if vwap <= 0:
            logger.warning("[VWAPFilter] VWAP computed as 0 — filter bypassed")
            return result

        stretch_pct = (spot - vwap) / vwap  # positive = above VWAP, negative = below

        result["vwap"]        = round(vwap, 4)
        result["stretch_pct"] = round(stretch_pct * 100, 4)

        # Determine required stretch direction for this trade
        bullish = direction in ("BULLISH", "NEUTRAL_BULL")
        bearish = direction in ("BEARISH", "NEUTRAL_BEAR")

        threshold = self.cfg.vwap_stretch_threshold

        if bullish:
            # Need price stretched BELOW VWAP to buy put spread (mean-rev bounce)
            if stretch_pct > -threshold:
                result["confirmed"] = False
                result["reason"] = (
                    f"vwap_stretch_insufficient: spot {spot:.2f} is only "
                    f"{stretch_pct*100:.3f}% below VWAP {vwap:.2f} "
                    f"(need ≥ {threshold*100:.2f}% below)"
                )
                logger.info("[VWAPFilter] %s", result["reason"])
                return result
        elif bearish:
            # Need price stretched ABOVE VWAP to sell call spread (mean-rev fade)
            if stretch_pct < threshold:
                result["confirmed"] = False
                result["reason"] = (
                    f"vwap_stretch_insufficient: spot {spot:.2f} is only "
                    f"{stretch_pct*100:.3f}% above VWAP {vwap:.2f} "
                    f"(need ≥ {threshold*100:.2f}% above)"
                )
                logger.info("[VWAPFilter] %s", result["reason"])
                return result
        else:
            # Unknown direction — pass through
            result["reason"] = f"vwap_direction_unknown({direction})_passthrough"
            return result

        # Cooldown: don't fire again within vwap_cooldown_seconds
        now = datetime.now(tz=ET)
        if self._last_signal_time is not None:
            elapsed = (now - self._last_signal_time).total_seconds()
            if elapsed < self.cfg.vwap_cooldown_seconds:
                result["confirmed"] = False
                result["reason"] = (
                    f"vwap_cooldown: last signal {elapsed:.0f}s ago "
                    f"(cooldown={self.cfg.vwap_cooldown_seconds}s)"
                )
                logger.info("[VWAPFilter] %s", result["reason"])
                return result

        # Optional reclaim confirmation
        if self.cfg.vwap_require_reclaim:
            reclaim = self._check_reclaim(bars, vwap, bullish)
            if not reclaim:
                result["confirmed"] = False
                result["reason"] = "vwap_reclaim_not_detected"
                logger.info("[VWAPFilter] Reclaim required but not detected — skip")
                return result

        self._last_signal_time = now
        result["confirmed"] = True
        result["reason"] = (
            f"vwap_stretch_confirmed: {stretch_pct*100:+.3f}% from VWAP={vwap:.2f} "
            f"direction={direction}"
        )
        logger.info("[VWAPFilter] %s", result["reason"])
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_today_bars(self) -> list[dict]:
        """Fetch today's 1-minute SPY bars from Alpaca."""
        if not _cb.is_available("alpaca_bars_vwap"):
            return []
        try:
            today_str = date.today().isoformat()
            bars = self.broker.get_bars(
                "SPY",
                timeframe="1Min",
                start=today_str,
                end=today_str,
                limit=400,
            )
            _cb.record_success("alpaca_bars_vwap")
            return bars if bars else []
        except Exception as exc:
            _cb.record_failure("alpaca_bars_vwap", str(exc))
            logger.warning("[VWAPFilter] Bar fetch failed: %s", exc)
            return []

    @staticmethod
    def _compute_vwap(bars: list[dict]) -> float:
        """
        Running VWAP = cumsum(typical_price * volume) / cumsum(volume).
        Typical price = (high + low + close) / 3.
        Returns the VWAP at the most recent bar.
        """
        cum_tpv = 0.0
        cum_vol = 0.0
        for bar in bars:
            try:
                h = float(bar.get("h") or bar.get("high")  or 0)
                l = float(bar.get("l") or bar.get("low")   or 0)
                c = float(bar.get("c") or bar.get("close") or 0)
                v = float(bar.get("v") or bar.get("volume") or 0)
            except (TypeError, ValueError):
                continue
            if v <= 0:
                continue
            tp = (h + l + c) / 3.0
            cum_tpv += tp * v
            cum_vol  += v
        return cum_tpv / cum_vol if cum_vol > 0 else 0.0

    def _check_reclaim(self, bars: list[dict], vwap: float, bullish: bool) -> bool:
        """
        Detect if the most recent bars show a partial reclaim toward VWAP.
        Looks at the last 3 bars (≈3 minutes) for the reclaim move.
        """
        reclaim_threshold = self.cfg.vwap_reclaim_threshold
        recent = bars[-3:] if len(bars) >= 3 else bars
        for bar in recent:
            try:
                c = float(bar.get("c") or bar.get("close") or 0)
            except (TypeError, ValueError):
                continue
            if bullish:
                # Price was below VWAP; reclaim = price moved back within threshold
                if c >= vwap * (1.0 - reclaim_threshold):
                    return True
            else:
                # Price was above VWAP; reclaim = price moved back within threshold
                if c <= vwap * (1.0 + reclaim_threshold):
                    return True
        return False


class ZeroDTEStrategy:
    """
    Orchestrates all 0DTE entry logic and returns an ApprovedOrder.

    Intended to be called by the orchestrator's 2-minute intraday scan.
    Returns None if any pre-entry filter fails.
    """

    def __init__(self, cfg: ZeroDTEConfig, broker: AlpacaBroker, db=None):
        self.cfg      = cfg
        self.broker   = broker
        self.db       = db
        self.calendar    = EventCalendar()
        self.session     = SessionClassifier()
        self.gex         = GEXEngine(broker, cfg)
        self.sizer       = KellySizer(cfg, db)
        self.slippage    = SlippageGuard(cfg)
        self.gamma_adj   = GammaAdjuster()
        self.vwap_filter = VWAPStretchFilter(cfg, broker)

    def evaluate(self) -> Optional[ApprovedOrder]:
        """
        Run all pre-entry checks and return an ApprovedOrder if all pass,
        or None if any check fails.
        """
        now     = datetime.now(tz=ET)
        today   = date.today()
        session = self.session.classify(now)

        # 1. Session gate
        if not self.session.can_enter(session):
            logger.info("[0DTE] Session %s — no entry", session)
            return None

        # 2. Weekday check (SPY options trade Mon-Fri)
        if today.weekday() >= 5:
            logger.info("[0DTE] Weekend — no trade")
            return None

        if self.cfg.skip_friday and today.weekday() == 4:
            logger.info("[0DTE] Friday filter active — skipping")
            return None

        # 3. Event calendar
        event = self.calendar.classify(today)
        risk_mult = event["risk_multiplier"]
        if risk_mult == 0.0:
            logger.info("[0DTE] Event gate blocked: %s", event["mode"])
            return None
        logger.info("[0DTE] Event mode: %s (risk_mult=%.1f)", event["mode"], risk_mult)

        # 4. Market data filters
        spy_data = _get_spy_data()
        vix      = _get_vix()

        if not spy_data or not vix:
            logger.warning("[0DTE] Market data unavailable — skipping")
            return None

        spy_price   = spy_data["spy_price"]
        rsi         = spy_data["rsi"]
        consec_down = spy_data["consec_down"]
        gap_pct     = spy_data["gap_pct"]

        if vix < self.cfg.vix_floor:
            logger.info("[0DTE] VIX %.2f below floor %.1f — insufficient premium", vix, self.cfg.vix_floor)
            return None
        if vix > self.cfg.vix_ceiling:
            logger.info("[0DTE] VIX %.2f above ceiling %.1f — too dangerous", vix, self.cfg.vix_ceiling)
            return None

        if not (self.cfg.rsi_min <= rsi <= self.cfg.rsi_max):
            logger.info("[0DTE] RSI %.1f outside [%.0f, %.0f] — skip", rsi, self.cfg.rsi_min, self.cfg.rsi_max)
            return None

        if consec_down > self.cfg.max_consec_down_days:
            logger.info("[0DTE] %d consecutive down days > %d — skip", consec_down, self.cfg.max_consec_down_days)
            return None

        if gap_pct > self.cfg.max_gap_pct:
            logger.info("[0DTE] Gap %.2f%% > max %.1f%% — GEX pin disrupted", gap_pct, self.cfg.max_gap_pct)
            return None

        # Expected move filter
        exp_move = _expected_move_pct(spy_price, vix, hours=2.0)
        if exp_move < self.cfg.min_expected_move_pct / 100:
            logger.info("[0DTE] Expected 2hr move %.3f%% < %.3f%% — too quiet", exp_move * 100, self.cfg.min_expected_move_pct)
            return None

        # 5. GEX pin
        pin = self.gex.compute(spy_price)
        if pin is None:
            logger.info("[0DTE] GEX pin unavailable — skip")
            return None

        strikes = self.gex.select_strikes(spy_price, pin, self.cfg)
        if strikes is None:
            logger.info("[0DTE] No valid strike selection from GEX pin — skip")
            return None

        # 5b. VWAP stretch confirmation
        # Ensures price has actually stretched away from intraday VWAP in a
        # direction consistent with the spread before committing to quotes/sizing.
        # Fail-open: if Alpaca bars are unavailable the filter passes through.
        vwap_check = self.vwap_filter.check(spy_price, strikes["direction"])
        if not vwap_check["confirmed"]:
            logger.info("[0DTE] VWAP filter blocked: %s", vwap_check["reason"])
            return None
        logger.info(
            "[0DTE] VWAP filter passed: stretch=%.3f%% vwap=%.4f",
            vwap_check["stretch_pct"], vwap_check["vwap"],
        )

        # Bar cooldown: prevent re-entry too quickly after a recent fill
        self.bar_cooldown.tick()
        if not self.bar_cooldown.ready():
            logger.info(
                "[0DTE] Bar cooldown active — %d bars remaining before re-entry",
                self.bar_cooldown.remaining,
            )
            return None

        # Feed latest intraday bars into session momentum engine
        try:
            _bars_raw = getattr(self.vwap_filter, "_last_bars", [])
            if _bars_raw:
                _last = _bars_raw[-1]
                _mom = self.momentum.on_bar(
                    _last.get("open", spy_price), _last.get("high", spy_price),
                    _last.get("low",  spy_price), _last.get("close", spy_price),
                    _last.get("volume", 1.0),
                )
                logger.debug(
                    "[0DTE] Momentum: dir=%s ema5=%.2f ema20=%.2f "
                    "roc5=%.4f cg=%d cr=%d atr5=%.3f",
                    _mom["direction"], _mom["ema5"], _mom["ema20"],
                    _mom["roc5"], _mom["consec_green"],
                    _mom["consec_red"], _mom["atr5"],
                )
        except Exception:
            pass

        # 6. Fetch option quotes for the selected strikes
        today_str    = today.strftime("%y%m%d")
        short_strike = strikes["short_strike"]
        long_strike  = strikes["long_strike"]
        is_put       = "PUT" in strikes["strategy"]

        opt_char     = "P" if is_put else "C"
        # Alpaca OCC symbol: SPY + YYMMDD + C/P + 8-digit strike (x1000, zero-padded)
        short_sym    = f"SPY{today_str}{opt_char}{int(short_strike * 1000):08d}"
        long_sym     = f"SPY{today_str}{opt_char}{int(long_strike  * 1000):08d}"

        try:
            quotes = self.broker.get_option_snapshots([short_sym, long_sym])
        except Exception as exc:
            logger.warning("[0DTE] Quote fetch failed: %s", exc)
            return None

        def _q(sym: str) -> dict:
            snap = quotes.get(sym) or {}
            q    = snap.get("latest_quote") or snap.get("quote") or {}
            return q

        short_q = _q(short_sym)
        long_q  = _q(long_sym)

        short_bid = float(short_q.get("bid_price") or short_q.get("bp") or 0)
        short_ask = float(short_q.get("ask_price") or short_q.get("ap") or 0)
        long_bid  = float(long_q.get("bid_price")  or long_q.get("bp")  or 0)
        long_ask  = float(long_q.get("ask_price")  or long_q.get("ap")  or 0)

        if not all([short_bid, short_ask, long_bid, long_ask]):
            logger.warning("[0DTE] Incomplete quotes for %s / %s", short_sym, long_sym)
            return None

        # 7. Spread quality and credit
        ok, reason = self.slippage.check(short_bid, short_ask, long_bid, long_ask,
                                          (short_bid + short_ask) / 2)
        if not ok:
            logger.info("[0DTE] Slippage guard: %s", reason)
            return None

        credit = self.slippage.mid_credit(short_bid, short_ask, long_bid, long_ask)

        # Time-based minimum credit
        hour = now.hour
        if hour < 11:
            min_credit = self.cfg.min_credit_morning
        elif hour < 13:
            min_credit = self.cfg.min_credit_midday
        else:
            min_credit = self.cfg.min_credit_afternoon

        min_credit = max(min_credit, self.cfg.min_credit_absolute)

        if credit < min_credit:
            logger.info("[0DTE] Credit $%.3f < min $%.3f at %02d:xx ET", credit, min_credit, hour)
            return None

        # 8. Position sizing
        equity = self.broker.get_equity()
        closed_pnls = self._load_recent_pnls()
        n_contracts  = self.sizer.size(equity, closed_pnls)

        # Apply event risk multiplier (round down)
        n_contracts = max(1, int(n_contracts * risk_mult))

        if n_contracts == 0:
            logger.info("[0DTE] Kelly sizing returned 0 — account at safety threshold")
            return None

        # Apply session confidence modifier (GAMMA_RAMP = 85%)
        if session == "GAMMA_RAMP":
            n_contracts = max(1, int(n_contracts * 0.85))

        # 9. Build ApprovedOrder
        spread_math = calc_spread(
            spread_type="bull_put" if is_put else "bear_call",
            action="entry",
            low_strike=long_strike  if is_put else short_strike,
            low_bid=long_bid        if is_put else short_bid,
            low_ask=long_ask        if is_put else short_ask,
            high_strike=short_strike if is_put else long_strike,
            high_bid=short_bid      if is_put else long_bid,
            high_ask=short_ask      if is_put else long_ask,
            num_contracts=n_contracts,
            underlying_price=spy_price,
        )

        profit_target_price = round(credit * (1.0 - self.cfg.profit_target_pct), 3)
        stop_price          = round(credit * (1.0 + self.cfg.stop_loss_pct), 3)
        limit_price         = self.slippage.limit_price(credit)

        short_leg = OrderLeg(
            symbol=short_sym,
            side="sell_to_open",
            qty=n_contracts,
            option_type=OptionType.PUT if is_put else OptionType.CALL,
            strike=short_strike,
            expiry=today.isoformat(),
        )
        long_leg = OrderLeg(
            symbol=long_sym,
            side="buy_to_open",
            qty=n_contracts,
            option_type=OptionType.PUT if is_put else OptionType.CALL,
            strike=long_strike,
            expiry=today.isoformat(),
        )

        order = ApprovedOrder(
            underlying=self.cfg.underlying,
            strategy="0dte_" + strikes["strategy"].lower(),
            legs=[short_leg, long_leg],
            limit_price=limit_price,
            net_credit=credit,
            position_size_contracts=n_contracts,
            max_loss=spread_math["max_loss"],
            hard_stop=stop_price,
            profit_target=profit_target_price,
            metadata={
                "session":      session,
                "event_mode":   event["mode"],
                "gex_pin":      pin.strike,
                "gex_regime":   pin.regime,
                "vix":          vix,
                "rsi":          rsi,
                "direction":    strikes["direction"],
                "gap_pct":      gap_pct,
                "consec_down":  consec_down,
                "exp_move_pct":   round(exp_move * 100, 3),
                "vwap":           vwap_check["vwap"],
                "vwap_stretch_pct": vwap_check["stretch_pct"],
            },
        )

        logger.info(
            "[0DTE] APPROVED: %s %s credit=$%.3f contracts=%d "
            "short=%s long=%s pin=%.0f regime=%s",
            strikes["strategy"], strikes["direction"], credit, n_contracts,
            short_sym, long_sym, pin.strike, pin.regime,
        )
        return order

    def _load_recent_pnls(self) -> list[float]:
        """Load recent 0DTE closed trade P&Ls from DB for Kelly sizing."""
        if self.db is None:
            return []
        try:
            with self.db._get_conn() as conn:
                cur = conn.execute(
                    """SELECT realized_pnl FROM trades
                       WHERE strategy LIKE '0dte_%'
                         AND realized_pnl IS NOT NULL
                       ORDER BY updated_at DESC LIMIT 50"""
                )
                return [row[0] for row in cur.fetchall()]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Position Monitor
# ---------------------------------------------------------------------------

@dataclass
class _MonitoredPosition:
    trade_id:         str
    short_sym:        str
    long_sym:         str
    entry_credit:     float
    profit_target:    float    # spread mid at which to take profit
    hard_stop:        float    # spread mid at which to stop out
    n_contracts:      int
    trailing_active:  bool     = False
    best_profit_pct:  float    = 0.0
    trailing_floor:   float    = 0.0   # don't close below this credit when trailing


class ZeroDTEMonitor:
    """
    15-second polling loop that manages open 0DTE positions.

    Exit conditions (checked in order):
      1. Hard force-close at force_close_hour:force_close_minute ET
      2. Hard stop-loss (spread mid > stop price)
      3. Profit target (spread mid < profit target price)
      4. Trailing stop (once 25% profit reached, locks in 10% floor)

    Source: gamma-main/monitor.py — adapted for Alpaca broker interface.
    """

    def __init__(self, cfg: ZeroDTEConfig, broker: AlpacaBroker, db=None):
        self.cfg    = cfg
        self.broker = broker
        self.db     = db
        self._positions: list[_MonitoredPosition] = []

    def register(self, order: ApprovedOrder, fill: FilledOrder) -> None:
        """Register a newly filled 0DTE position for monitoring."""
        legs      = order.legs
        short_sym = next(l.symbol for l in legs if l.side == "sell_to_open")
        long_sym  = next(l.symbol for l in legs if l.side == "buy_to_open")
        credit    = order.net_credit

        pos = _MonitoredPosition(
            trade_id=fill.order_id,
            short_sym=short_sym,
            long_sym=long_sym,
            entry_credit=credit,
            profit_target=order.profit_target,
            hard_stop=order.hard_stop,
            n_contracts=order.position_size_contracts,
        )
        self._positions.append(pos)
        logger.info(
            "[0DTE Monitor] Registered %s: credit=$%.3f TP=$%.3f SL=$%.3f",
            fill.order_id, credit, pos.profit_target, pos.hard_stop,
        )

    def run_once(self) -> list[str]:
        """
        Check all monitored positions. Returns list of closed trade IDs.

        Call this every cfg.monitor_poll_seconds from the scheduler.
        """
        closed = []
        now_et = datetime.now(tz=ET)

        # Force-close time check
        force_close = (
            now_et.hour > self.cfg.force_close_hour
            or (now_et.hour == self.cfg.force_close_hour
                and now_et.minute >= self.cfg.force_close_minute)
        )

        remaining = []
        for pos in self._positions:
            result = self._check_position(pos, force_close, now_et)
            if result == "closed":
                closed.append(pos.trade_id)
            else:
                remaining.append(pos)

        self._positions = remaining
        return closed

    def _check_position(
        self, pos: _MonitoredPosition, force_close: bool, now_et: datetime
    ) -> str:
        """Returns 'closed' or 'open'."""

        # Fetch current spread mid
        try:
            quotes = self.broker.get_option_snapshots([pos.short_sym, pos.long_sym])
        except Exception as exc:
            logger.warning("[0DTE Monitor] Quote fetch error for %s: %s", pos.trade_id, exc)
            return "open"

        def _mid(sym: str) -> float:
            snap = quotes.get(sym) or {}
            q    = snap.get("latest_quote") or snap.get("quote") or {}
            bid  = float(q.get("bid_price") or q.get("bp") or 0)
            ask  = float(q.get("ask_price") or q.get("ap") or 0)
            return (bid + ask) / 2 if bid and ask else 0.0

        short_mid = _mid(pos.short_sym)
        long_mid  = _mid(pos.long_sym)
        spread_mid = max(0.0, short_mid - long_mid)

        profit_pct = (pos.entry_credit - spread_mid) / pos.entry_credit if pos.entry_credit > 0 else 0.0

        logger.debug(
            "[0DTE Monitor] %s spread_mid=$%.3f profit=%.1f%% "
            "(TP=$%.3f SL=$%.3f trailing=%s)",
            pos.trade_id, spread_mid, profit_pct * 100,
            pos.profit_target, pos.hard_stop, pos.trailing_active,
        )

        exit_reason = None

        # Force close
        if force_close:
            exit_reason = "force_close_eod"

        # Hard stop
        elif spread_mid >= pos.hard_stop:
            exit_reason = "stopped_out"

        # Profit target
        elif spread_mid <= pos.profit_target:
            exit_reason = "closed_profit_target"

        # Trailing stop logic
        elif profit_pct >= self.cfg.trailing_stop_trigger_pct:
            if not pos.trailing_active:
                pos.trailing_active = True
                pos.best_profit_pct = profit_pct
                pos.trailing_floor  = pos.entry_credit * (1.0 - self.cfg.trailing_stop_floor_pct)
                logger.info(
                    "[0DTE Monitor] %s: trailing stop activated at %.1f%% profit "
                    "(floor=$%.3f)",
                    pos.trade_id, profit_pct * 100, pos.trailing_floor,
                )
            else:
                pos.best_profit_pct = max(pos.best_profit_pct, profit_pct)
                # Close if spread expanded back above the floor
                if spread_mid >= pos.trailing_floor:
                    exit_reason = "trailing_stop"

        if exit_reason:
            self._close_position(pos, exit_reason, spread_mid, profit_pct)
            return "closed"

        return "open"

    def _close_position(
        self,
        pos: _MonitoredPosition,
        reason: str,
        spread_mid: float,
        profit_pct: float,
    ) -> None:
        """Submit closing order and update DB."""
        realized_pnl = round(
            (pos.entry_credit - spread_mid) * 100 * pos.n_contracts, 2
        )
        logger.info(
            "[0DTE Monitor] CLOSE %s reason=%s spread=$%.3f pnl=$%.2f",
            pos.trade_id, reason, spread_mid, realized_pnl,
        )

        # Close via broker (buy back the short, sell the long)
        for sym, side in [(pos.short_sym, "buy_to_close"), (pos.long_sym, "sell_to_close")]:
            try:
                self.broker.close_position(sym, qty=pos.n_contracts)
            except Exception as exc:
                logger.error("[0DTE Monitor] Close leg %s failed: %s", sym, exc)

        # Update DB
        if self.db:
            try:
                self.db.update_status(
                    pos.trade_id,
                    status=reason,
                    close_price=spread_mid,
                    realized_pnl=realized_pnl,
                )
            except Exception as exc:
                logger.error("[0DTE Monitor] DB update failed for %s: %s", pos.trade_id, exc)

        self._notify(pos, reason, spread_mid, realized_pnl)

    def _notify(
        self, pos: _MonitoredPosition, reason: str, spread_mid: float, pnl: float
    ) -> None:
        if not self.cfg.discord_webhook_url:
            return
        emoji = "✅" if pnl > 0 else "🛑"
        msg = (
            f"{emoji} **0DTE {reason.upper()}**\n"
            f"Trade: `{pos.trade_id}`\n"
            f"Short: `{pos.short_sym}`\n"
            f"Entry credit: ${pos.entry_credit:.3f} | Close: ${spread_mid:.3f}\n"
            f"P&L: ${pnl:+.2f} ({pos.n_contracts} contracts)"
        )
        import urllib.request
        try:
            payload = json.dumps({"content": msg}).encode()
            req = urllib.request.Request(
                self.cfg.discord_webhook_url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.debug("[0DTE Monitor] Discord notify failed: %s", exc)

    @property
    def open_count(self) -> int:
        return len(self._positions)
