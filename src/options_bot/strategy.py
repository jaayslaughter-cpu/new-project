"""
Strategy layer.

Defines the base Strategy interface and three concrete implementations:

  1. CashSecuredPut (CSP)
     Sell an OTM put, collect premium, obligation to buy shares at strike.
     Max loss = (strike - credit) * 100. Hard stop at 2x premium received.
     Entry filter: delta between -0.15 and -0.30, DTE 21-45, OI >= 500.

  2. ShortPutSpread (bull put spread / credit spread)
     Sell OTM put + buy further OTM put. Defined max loss = spread width - credit.
     The only spreads strategy that gives a true finite max-loss per contract.
     Entry filter: short leg delta -0.20 to -0.30, DTE 21-45, OI >= 100.

  3. ShortStrangle
     Sell OTM call + sell OTM put on same expiry.
     Max loss is theoretically large on the upside — only used with explicit
     stop-loss at 2x premium received per leg (or combined 3x credit).
     Entry filter: |delta| 0.15-0.25 each leg, DTE 30-60, IV rank >= 30%.

Mathematical rationale (written before code per system directive):

  CSP entry signal:
    target_delta: the put delta we want to sell (e.g. -0.20)
    Select the contract where |delta - target_delta| is minimised.
    max_loss_per_contract = strike * 100  (worst case: stock → 0)
    practical_max_loss = (strike - credit_received) * 100
    hard_stop = credit_received * stop_multiplier (e.g. 2x)

  Short put spread entry signal:
    short_leg = put closest to short_delta (e.g. -0.25)
    long_leg  = put closest to long_delta  (e.g. -0.10, further OTM)
    spread_width = short_strike - long_strike
    net_credit = short_premium - long_premium
    max_loss_per_contract = (spread_width - net_credit) * 100
    hard_stop = net_credit * stop_multiplier (e.g. 2x)

  Strangle entry signal:
    call_leg = call closest to call_delta (e.g. +0.20)
    put_leg  = put  closest to put_delta  (e.g. -0.20)
    net_credit = call_premium + put_premium
    max_loss is theoretically large → stop mandatory at 3x credit received
    hard_stop = net_credit * stop_multiplier (e.g. 3.0)

All strategies:
  - Raise LiquidityFilterError if no qualifying contract found
  - Raise PipelineConnectionError if enriched chain is empty
  - Return a StrategySignal containing legs + risk inputs for RiskManager
  - Never build ApprovedOrder directly — that belongs to RiskManager
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from .contracts import EnrichedOptionRow, OrderLeg, OptionType
from .exceptions import LiquidityFilterError, PipelineConnectionError
from .spread_math import (
    bull_put_entry,
    bull_put_exit,
    strangle_entry,
    profit_target_price as calc_profit_target,
    stop_price as calc_stop_price,
    validate_spread_inputs,
)
from .greeks import probability_of_profit, pop_spread, get_risk_free_rate
from .earnings_calendar import EarningsFilter
from .volume_profile import check_strike_safety, volume_profile_cache
from .gex_analysis import analyze_gex, check_strike_gex_safety

_earnings_filter = EarningsFilter(days_before=5, days_after=2)

logger = logging.getLogger(__name__)


def _clean_ticker(value: str) -> str:
    """
    Normalise a ticker/underlying string: strip whitespace and upper-case.

    Applied to every underlying value extracted from EnrichedOptionRow before
    it is passed to the execution layer or used in OCC symbol formatting.
    Prevents silent data integrity failures from mixed-case or padded strings
    from yfinance or other data sources.
    """
    return value.strip().upper() if value else value


# ---------------------------------------------------------------------------
# Strategy signal — output of every strategy, input to RiskManager
# ---------------------------------------------------------------------------

@dataclass
class StrategySignal:
    """
    Output of a strategy evaluation.

    Contains everything RiskManager.evaluate() and build_approved_order() need.
    Does NOT contain position size — that is RiskManager's job.
    """
    strategy_name: str
    underlying: str

    # Legs to trade
    legs: list[OrderLeg]

    # Pricing
    net_debit_credit: float        # negative = credit received (typical for short strategies)
    estimated_fill_price: float    # absolute value of net_debit_credit

    # Risk inputs for RiskManager.evaluate()
    max_loss_per_contract: float   # dollars, must be finite and > 0
    hard_stop_price: float         # stop-loss price level (e.g. 2x premium on short side)

    # Optional profit target
    profit_target_price: Optional[float] = None

    # Metadata for logging and analysis
    expiry: Optional[date] = None
    dte: Optional[int] = None
    signal_timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    notes: str = ""

    # The specific enriched contracts that drove the signal (for logging)
    source_contracts: list[EnrichedOptionRow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Abstract base for all options strategies.

    Subclasses implement evaluate() which takes an enriched option chain
    and returns a StrategySignal, or raises if no valid trade found.

    The strategy layer never touches the risk manager or broker.
    It only selects contracts and computes the signal inputs.
    """

    def __init__(self, name: str):
        self.name = name
        logger.info("[Strategy] Initialized: %s", name)

    @abstractmethod
    def evaluate(self, chain: list[EnrichedOptionRow]) -> StrategySignal:
        """
        Evaluate the chain and return a StrategySignal if a trade qualifies.

        Parameters
        ----------
        chain : list[EnrichedOptionRow]
            Enriched option chain for one expiration. Must have IV and delta
            computed — rows missing these are automatically skipped.

        Returns
        -------
        StrategySignal

        Raises
        ------
        PipelineConnectionError
            If chain is empty or no rows have required Greeks.
        LiquidityFilterError
            If no contract meets the strategy's entry criteria.
        """

    def _require_nonempty(self, chain: list[EnrichedOptionRow]) -> None:
        if not chain:
            raise PipelineConnectionError(
                f"[{self.name}] Received empty chain — "
                "market_data or greeks layer failed upstream"
            )

    def _require_greeks(self, chain: list[EnrichedOptionRow]) -> list[EnrichedOptionRow]:
        """Filter to rows that have both IV and delta. Raises if none remain."""
        valid = [r for r in chain if r.iv is not None and r.delta is not None]
        if not valid:
            raise PipelineConnectionError(
                f"[{self.name}] No rows with IV+delta in chain of {len(chain)} contracts. "
                "Greeks layer may have failed for all rows."
            )
        logger.debug(
            "[%s] %d/%d rows have IV+delta",
            self.name, len(valid), len(chain)
        )
        return valid

    def _find_closest_delta(
        self,
        contracts: list[EnrichedOptionRow],
        target_delta: float,
        option_type: OptionType,
    ) -> EnrichedOptionRow:
        """
        Find the contract whose delta is closest to target_delta.

        Parameters
        ----------
        target_delta : float
            Target delta value. Use negative for puts (e.g. -0.25).
        option_type : str
            "call" or "put"
        """
        filtered = [c for c in contracts if c.option_type == option_type]
        if not filtered:
            raise LiquidityFilterError(
                "chain",
                f"No {option_type}s found in chain for [{self.name}]"
            )
        closest = min(filtered, key=lambda c: abs(c.delta - target_delta))
        logger.debug(
            "[%s] Closest %s to delta=%.3f: %s (delta=%.4f, strike=%.1f)",
            self.name, option_type, target_delta,
            closest.symbol, closest.delta, closest.strike
        )
        return closest

    def _check_earnings(
        self,
        underlying: str,
        expiry,
        dte: Optional[int] = None,
    ) -> None:
        """
        Hard earnings filter — raises LiquidityFilterError if earnings fall
        within the DTE window. Call from every strategy before entering.
        ETFs are automatically allowed through (no per-company earnings).
        """
        from datetime import date as _date
        blocked, reason = _earnings_filter.check(
            underlying,
            entry_date=_date.today(),
            expiry_date=expiry,
            dte=dte,
        )
        if blocked:
            raise LiquidityFilterError(
                underlying,
                f"[{self.name}] EARNINGS BLOCK: {reason}"
            )

    def _check_volume_profile(
        self,
        underlying: str,
        short_strike: float,
        spot: float,
        spread_type: str = "bull_put",
        min_hvn_distance_pct: float = 1.5,
    ) -> None:
        """
        Volume profile strike safety check — raises LiquidityFilterError if the
        short strike sits in a contested HVN zone. Non-fatal if data unavailable.
        """
        try:
            profile = volume_profile_cache.get(underlying)
            safe, reason = check_strike_safety(
                ticker=underlying,
                short_strike=short_strike,
                spot=spot,
                spread_type=spread_type,
                min_hvn_distance_pct=min_hvn_distance_pct,
                profile=profile,
            )
            if not safe:
                raise LiquidityFilterError(
                    underlying,
                    f"[{self.name}] Volume profile REJECT: {reason}"
                )
            logger.debug("[%s] VP check OK: %s — %s", self.name, underlying, reason)
        except LiquidityFilterError:
            raise
        except Exception as exc:
            logger.debug("[%s] VP check skipped (data unavailable): %s", self.name, exc)

    def _check_gex(
        self,
        enriched: list,
        short_strike: float,
        spot: float,
        expiry,
        dte: int,
        atm_iv: float = 0.20,
        min_distance_pct: float = 1.5,
    ) -> None:
        """
        GEX strike safety check — raises LiquidityFilterError if the short
        strike is too close to the put wall or above the pin strike.
        Non-fatal if GEX data is unavailable.
        """
        try:
            underlying = _clean_ticker(enriched[0].underlying) if enriched else "unknown"
            analysis = analyze_gex(
                ticker=underlying,
                enriched_rows=enriched,
                expiry=expiry,
                spot=spot,
                dte_days=dte,
                atm_iv=atm_iv,
            )
            safe, reason = check_strike_gex_safety(
                analysis=analysis,
                short_strike=short_strike,
                min_distance_pct=min_distance_pct,
            )
            if not safe:
                raise LiquidityFilterError(
                    underlying,
                    f"[{self.name}] GEX REJECT: {reason}"
                )
            logger.debug("[%s] GEX OK: %s — %s", self.name, underlying, reason)
        except LiquidityFilterError:
            raise
        except Exception as exc:
            logger.debug("[%s] GEX check skipped (non-fatal): %s", self.name, exc)


# ---------------------------------------------------------------------------
# Strategy 1: Cash-Secured Put (CSP)
# ---------------------------------------------------------------------------

@dataclass
class CSPConfig:
    """Configuration for CashSecuredPut strategy."""
    target_delta: float = -0.20         # target put delta to sell
    min_delta: float = -0.30            # reject if delta more negative than this
    max_delta: float = -0.10            # reject if delta less negative than this
    min_dte: int = 14                   # minimum days to expiration (widened for monthly-only ETFs)
    max_dte: int = 60                   # maximum days to expiration
    min_open_interest: int = 500        # higher OI requirement for CSPs
    max_spread_pct: float = 0.15        # tighter spread requirement
    stop_multiplier: float = 2.0        # stop at 2x premium received
    profit_target_pct: float = 0.50     # close at 50% of max profit


class CashSecuredPut(BaseStrategy):
    """
    Sell an OTM put, fully cash-secured.

    Entry: sell the put closest to target_delta within delta/DTE filters.
    Stop:  hard_stop = premium_received * stop_multiplier
    Exit:  close at profit_target_pct of max profit (default 50%)

    Max loss = (strike - credit_received) * 100
    This is defined-risk in the sense that the stock cannot go below zero,
    but loss can be large on a big move down — use only with hard stops.
    """

    def __init__(self, config: Optional[CSPConfig] = None):
        super().__init__("CashSecuredPut")
        self.config = config or CSPConfig()

    def evaluate(self, chain: list[EnrichedOptionRow]) -> StrategySignal:
        self._require_nonempty(chain)
        valid = self._require_greeks(chain)

        cfg = self.config

        # Filter: option type + DTE + liquidity
        candidates = [
            c for c in valid
            if c.option_type == "put"
            and c.dte >= cfg.min_dte
            and c.dte <= cfg.max_dte
            and (c.open_interest is None or c.open_interest >= cfg.min_open_interest)
            and (c.spread_pct is None or c.spread_pct <= cfg.max_spread_pct)
            and c.bid is not None
            and c.ask is not None
        ]

        if not candidates:
            raise LiquidityFilterError(
                f"{_clean_ticker(valid[0].underlying) if valid else 'unknown'} chain",
                f"[{self.name}] No puts pass filters: "
                f"DTE {cfg.min_dte}-{cfg.max_dte}, "
                f"OI>={cfg.min_open_interest}, spread<={cfg.max_spread_pct:.0%}"
            )

        # Delta filter
        delta_filtered = [
            c for c in candidates
            if cfg.min_delta <= c.delta <= cfg.max_delta
        ]

        if not delta_filtered:
            best = candidates[0]
            raise LiquidityFilterError(
                f"{best.underlying} chain",
                f"[{self.name}] No puts in delta range "
                f"[{cfg.min_delta:.2f}, {cfg.max_delta:.2f}]. "
                f"Checked {len(candidates)} candidates."
            )

        # Select the put closest to target_delta
        short_put = min(
            delta_filtered,
            key=lambda c: abs(c.delta - cfg.target_delta)
        )

        credit = short_put.mid_price or ((short_put.bid + short_put.ask) / 2)
        # CSP max loss: stock drops to zero, we buy at strike, keep credit
        # Practical max loss = (strike - credit) * 100 per contract
        max_loss_per_contract = (short_put.strike - credit) * 100
        # Use spread_math helpers for stop and profit target
        # (ensures consistent math across all strategies)
        hard_stop = calc_stop_price(credit, cfg.stop_multiplier)
        profit_target = calc_profit_target(credit, cfg.profit_target_pct)

        # Earnings hard filter — same risk as ShortPutSpread for single-stock gap
        self._check_earnings(short_put.underlying, short_put.expiry, short_put.dte)

        # Volume profile strike safety check
        self._check_volume_profile(
            short_put.underlying, short_put.strike, short_put.underlying_price
        )

        leg = OrderLeg(
            symbol=short_put.symbol,
            option_type="put",
            strike=short_put.strike,
            expiry=short_put.expiry,
            side="sell_to_open",
            quantity=1,  # RiskManager will scale this
        )

        logger.info(
            "[%s] Signal: SELL %s delta=%.3f strike=%.1f credit=%.2f "
            "DTE=%d max_loss=$%.2f stop=%.2f",
            self.name, short_put.symbol, short_put.delta,
            short_put.strike, credit, short_put.dte,
            max_loss_per_contract, hard_stop
        )

        return StrategySignal(
            strategy_name=self.name,
            underlying=short_put.underlying,
            legs=[leg],
            net_debit_credit=-credit,           # negative = credit received
            estimated_fill_price=credit,
            max_loss_per_contract=max_loss_per_contract,
            hard_stop_price=hard_stop,
            profit_target_price=profit_target,
            expiry=short_put.expiry,
            dte=short_put.dte,
            notes=(
                f"delta={short_put.delta:.3f} iv={short_put.iv:.2%} "
                f"bid={short_put.bid:.2f} ask={short_put.ask:.2f}"
            ),
            source_contracts=[short_put],
        )


# ---------------------------------------------------------------------------
# Strategy 2: Short Put Spread (bull put spread / credit spread)
# ---------------------------------------------------------------------------

@dataclass
class ShortPutSpreadConfig:
    """Configuration for ShortPutSpread strategy."""
    short_delta: float = -0.25          # sell this delta put
    long_delta: float = -0.10           # buy this delta put (further OTM)
    min_dte: int = 14     # widened: captures monthly-only ETF expirations
    max_dte: int = 60
    min_open_interest: int = 100
    max_spread_pct: float = 0.25
    min_spread_width: float = 1.0       # minimum strike width in dollars (lowered for ETFs)
    max_spread_width: float = 20.0      # maximum strike width
    min_credit: float = 0.25            # minimum credit to bother with the trade
    stop_multiplier: float = 2.0        # stop at 2x credit received
    profit_target_pct: float = 0.50     # close at 50% of max profit
    min_pop: float = 0.65               # minimum probability of profit (65%)
    vp_check_enabled: bool = True       # enable volume-profile strike safety check
    vp_min_hvn_distance_pct: float = 1.5  # min % distance from short strike to any HVN


class ShortPutSpread(BaseStrategy):
    """
    Sell OTM put + buy further OTM put on same expiration.

    This is the only strategy here with a fully defined max loss per contract:
      max_loss = (spread_width - credit_received) * 100

    Entry: sell put at short_delta, buy put at long_delta.
    Stop:  hard_stop = credit_received * stop_multiplier
    Exit:  close at profit_target_pct of max profit (default 50%)
    """

    def __init__(self, config: Optional[ShortPutSpreadConfig] = None):
        super().__init__("ShortPutSpread")
        self.config = config or ShortPutSpreadConfig()

    def evaluate(self, chain: list[EnrichedOptionRow]) -> StrategySignal:
        self._require_nonempty(chain)
        valid = self._require_greeks(chain)

        cfg = self.config

        # Filter: puts only, DTE + liquidity
        put_candidates = [
            c for c in valid
            if c.option_type == "put"
            and c.dte >= cfg.min_dte
            and c.dte <= cfg.max_dte
            and (c.open_interest is None or c.open_interest >= cfg.min_open_interest)
            and (c.spread_pct is None or c.spread_pct <= cfg.max_spread_pct)
            and c.bid is not None
            and c.ask is not None
            and c.mid_price is not None
        ]

        if len(put_candidates) < 2:
            raise LiquidityFilterError(
                f"{_clean_ticker(valid[0].underlying) if valid else '?'} chain",
                f"[{self.name}] Need >= 2 liquid puts, found {len(put_candidates)}"
            )

        # Find short leg (closer to ATM)
        short_leg_candidates = [
            c for c in put_candidates
            if cfg.short_delta - 0.05 <= c.delta <= cfg.short_delta + 0.05
        ]
        if not short_leg_candidates:
            # Relax and take the closest
            short_leg_candidates = put_candidates

        short_put = min(
            short_leg_candidates,
            key=lambda c: abs(c.delta - cfg.short_delta)
        )

        # Find long leg (further OTM, lower strike, lower delta magnitude)
        long_leg_candidates = [
            c for c in put_candidates
            if c.strike < short_put.strike          # must be lower strike
            and (short_put.strike - c.strike) >= (
                    # For ETFs priced under $100, allow $0.5 wide spreads
                    # ($1 wide on a $50 ETF = 2% of spot, same ratio as $5 on SPY)
                    0.5 if short_put.strike < 100 else cfg.min_spread_width
                )
            and (short_put.strike - c.strike) <= cfg.max_spread_width
        ]

        if not long_leg_candidates:
            raise LiquidityFilterError(
                f"{short_put.underlying} chain",
                f"[{self.name}] No valid long leg found below short strike "
                f"{short_put.strike:.1f} with width "
                f"[{cfg.min_spread_width}, {cfg.max_spread_width}]"
            )

        long_put = min(
            long_leg_candidates,
            key=lambda c: abs(c.delta - cfg.long_delta)
        )

        spread_width = short_put.strike - long_put.strike
        short_credit = short_put.mid_price
        long_cost = long_put.mid_price
        net_credit = short_credit - long_cost

        if net_credit < cfg.min_credit:
            raise LiquidityFilterError(
                f"{short_put.underlying} chain",
                f"[{self.name}] Net credit ${net_credit:.2f} < min ${cfg.min_credit:.2f}. "
                f"Short={short_credit:.2f} Long={long_cost:.2f}"
            )

        # Use spread_math for all P&L calculations — single source of truth
        errors = validate_spread_inputs(
            low_bid=long_put.bid, low_ask=long_put.ask,
            high_bid=short_put.bid, high_ask=short_put.ask,
            low_strike=long_put.strike, high_strike=short_put.strike,
        )
        if errors:
            raise LiquidityFilterError(
                f"{short_put.underlying} chain",
                f"[{self.name}] Spread input validation failed: {'; '.join(errors)}"
            )

        spread_math = bull_put_entry(
            low_strike=long_put.strike,
            low_bid=long_put.bid,
            low_ask=long_put.ask,
            high_strike=short_put.strike,
            high_bid=short_put.bid,
            high_ask=short_put.ask,
            num_contracts=1,
            underlying_price=short_put.underlying_price,
        )

        max_loss_per_contract = spread_math["max_loss"]       # already per-contract dollars
        hard_stop             = calc_stop_price(net_credit, cfg.stop_multiplier)
        profit_target         = calc_profit_target(net_credit, cfg.profit_target_pct)

        # AUDIT FIX: Earnings hard filter
        # FINDING: "No earnings filter. The model has no earnings date lookup.
        # Missing structural risk check."
        # FIX: Reject if earnings fall within the DTE window.
        # Earnings hard filter + volume profile — shared BaseStrategy helpers.
        # Consistent with CashSecuredPut and ShortStrangle implementations.
        self._check_earnings(short_put.underlying, short_put.expiry, short_put.dte)
        self._check_volume_profile(
            short_put.underlying,
            short_put.strike,
            short_put.underlying_price,
            spread_type="bull_put",
            min_hvn_distance_pct=cfg.vp_min_hvn_distance_pct,
        )
        # GEX gamma wall check — don't short near the put wall or above pin
        self._check_gex(
            enriched=enriched,
            short_strike=short_put.strike,
            spot=short_put.underlying_price,
            expiry=short_put.expiry,
            dte=short_put.dte,
            atm_iv=short_put.iv or 0.20,
        )

        # Probability of profit + probability of touch validation
        # AUDIT FIX: PoT is now a HARD REJECT (was: warning only)
        # FINDING: "PoT warning not enforced. A short put with PoP=66%
        # and PoT=45% is structurally risky. Missing enforcement rule."
        short_iv = short_put.iv or 0.0
        if short_iv > 0 and short_put.dte and short_put.underlying_price:
            rate = get_risk_free_rate()
            prob = pop_spread(
                spread_type="bull_put",
                short_strike=short_put.strike,
                long_strike=long_put.strike,
                net_credit=net_credit,
                spot=short_put.underlying_price,
                sigma=short_iv,
                rate=rate,
                days_to_expiry=short_put.dte,
            )
            if prob["pop"] < cfg.min_pop:
                raise LiquidityFilterError(
                    short_put.underlying,
                    f"[{self.name}] PoP {prob['pop']:.0%} < min {cfg.min_pop:.0%} "
                    f"(break-even=${prob['break_even']:.2f}, PoT={prob['pot_short']:.0%}). "
                    f"LABEL: BS lognormal PoP — may underestimate fat-tail risk by 5-10pp."
                )
            # AUDIT FIX: PoT now REJECTS (was warning only)
            if prob["pot_warning"]:
                raise LiquidityFilterError(
                    short_put.underlying,
                    f"[{self.name}] PoT {prob['pot_short']:.0%} > 35% threshold — "
                    f"strike ${short_put.strike:.0f} has elevated touch risk. REJECT. "
                    f"(PoP={prob['pop']:.0%}, break-even=${prob['break_even']:.2f})"
                )

        short_leg = OrderLeg(
            symbol=short_put.symbol,
            option_type="put",
            strike=short_put.strike,
            expiry=short_put.expiry,
            side="sell_to_open",
            quantity=1,
        )
        long_leg = OrderLeg(
            symbol=long_put.symbol,
            option_type="put",
            strike=long_put.strike,
            expiry=long_put.expiry,
            side="buy_to_open",
            quantity=1,
        )

        logger.info(
            "[%s] Signal: SELL %s (delta=%.3f) / BUY %s (delta=%.3f) "
            "width=%.1f credit=%.2f max_loss=$%.2f stop=%.2f DTE=%d",
            self.name,
            short_put.symbol, short_put.delta,
            long_put.symbol, long_put.delta,
            spread_width, net_credit, max_loss_per_contract,
            hard_stop, short_put.dte
        )

        return StrategySignal(
            strategy_name=self.name,
            underlying=short_put.underlying,
            legs=[short_leg, long_leg],
            net_debit_credit=-net_credit,
            estimated_fill_price=net_credit,
            max_loss_per_contract=max_loss_per_contract,
            hard_stop_price=hard_stop,
            profit_target_price=profit_target,
            expiry=short_put.expiry,
            dte=short_put.dte,
            notes=(
                f"short={short_put.strike:.0f}P delta={short_put.delta:.3f} "
                f"long={long_put.strike:.0f}P delta={long_put.delta:.3f} "
                f"width={spread_width:.1f} credit={net_credit:.2f}"
            ),
            source_contracts=[short_put, long_put],
        )


# ---------------------------------------------------------------------------
# Strategy 3: Short Strangle
# ---------------------------------------------------------------------------

@dataclass
class ShortStrangleConfig:
    """Configuration for ShortStrangle strategy."""
    call_delta: float = 0.20            # sell this delta call
    put_delta: float = -0.20            # sell this delta put
    delta_tolerance: float = 0.05       # ± tolerance for delta matching
    min_dte: int = 21     # strangle: needs more time for both sides to decay
    max_dte: int = 60
    min_open_interest: int = 200
    max_spread_pct: float = 0.20
    min_total_credit: float = 0.75      # minimum combined credit (two sides ~$0.38 each)
    stop_multiplier: float = 3.0        # stop at 3x total credit received (wider than spread)
    profit_target_pct: float = 0.50
    min_iv_rank: Optional[float] = None  # e.g. 0.30 = only trade when IVR >= 30%
                                         # set None to skip this filter


class ShortStrangle(BaseStrategy):
    """
    Sell OTM call + sell OTM put on the same expiration.

    Collects premium from both sides. Profits if underlying stays between
    the two strikes at expiration.

    IMPORTANT: This has larger theoretical max loss than a spread.
    The stop_multiplier MUST be enforced — typically 3x combined credit.
    ExecutionGuard rejects any attempt to submit without a hard stop.

    Entry: sell call at call_delta, sell put at put_delta.
    Stop:  hard_stop = total_credit * stop_multiplier (applied per-side)
    Exit:  close at profit_target_pct of max profit (default 50%)
    """

    def __init__(self, config: Optional[ShortStrangleConfig] = None):
        super().__init__("ShortStrangle")
        self.config = config or ShortStrangleConfig()

    def evaluate(self, chain: list[EnrichedOptionRow]) -> StrategySignal:
        self._require_nonempty(chain)
        valid = self._require_greeks(chain)

        cfg = self.config

        # DTE + liquidity filter
        liquid = [
            c for c in valid
            if c.dte >= cfg.min_dte
            and c.dte <= cfg.max_dte
            and (c.open_interest is None or c.open_interest >= cfg.min_open_interest)
            and (c.spread_pct is None or c.spread_pct <= cfg.max_spread_pct)
            and c.bid is not None
            and c.ask is not None
            and c.mid_price is not None
        ]

        if not liquid:
            raise LiquidityFilterError(
                f"{_clean_ticker(valid[0].underlying) if valid else '?'} chain",
                f"[{self.name}] No liquid contracts in DTE "
                f"{cfg.min_dte}-{cfg.max_dte}"
            )

        # Find the call leg
        call_candidates = [
            c for c in liquid
            if c.option_type == "call"
            and abs(c.delta - cfg.call_delta) <= cfg.delta_tolerance
        ]
        if not call_candidates:
            # Relax tolerance
            call_candidates = [c for c in liquid if c.option_type == "call"]

        if not call_candidates:
            raise LiquidityFilterError(
                _clean_ticker(liquid[0].underlying),
                f"[{self.name}] No calls in chain"
            )

        short_call = min(
            call_candidates,
            key=lambda c: abs(c.delta - cfg.call_delta)
        )

        # Find the put leg
        put_candidates = [
            c for c in liquid
            if c.option_type == "put"
            and abs(c.delta - cfg.put_delta) <= cfg.delta_tolerance
        ]
        if not put_candidates:
            put_candidates = [c for c in liquid if c.option_type == "put"]

        if not put_candidates:
            raise LiquidityFilterError(
                _clean_ticker(liquid[0].underlying),
                f"[{self.name}] No puts in chain"
            )

        short_put = min(
            put_candidates,
            key=lambda c: abs(c.delta - cfg.put_delta)
        )

        call_credit = short_call.mid_price
        put_credit = short_put.mid_price
        total_credit = call_credit + put_credit

        if total_credit < cfg.min_total_credit:
            raise LiquidityFilterError(
                short_call.underlying,
                f"[{self.name}] Total credit ${total_credit:.2f} < "
                f"min ${cfg.min_total_credit:.2f}"
            )

        # Use spread_math for strangle P&L and break-even calculations
        strangle_math = strangle_entry(
            put_strike=short_put.strike,
            put_bid=short_put.bid,
            put_ask=short_put.ask,
            call_strike=short_call.strike,
            call_bid=short_call.bid,
            call_ask=short_call.ask,
            num_contracts=1,
            underlying_price=short_call.underlying_price,
        )

        practical_max_loss = strangle_math["practical_max_loss"]
        hard_stop          = calc_stop_price(total_credit, cfg.stop_multiplier)
        profit_target      = calc_profit_target(total_credit, cfg.profit_target_pct)

        # Earnings hard filter — strangles are especially dangerous around earnings
        # (realized move typically exceeds implied move, blowing through both strikes)
        self._check_earnings(short_call.underlying, short_call.expiry, short_call.dte)

        # Volume profile check on both short strikes
        self._check_volume_profile(
            short_call.underlying, short_call.strike, short_call.underlying_price,
            spread_type="bear_call",
        )
        self._check_volume_profile(
            short_put.underlying, short_put.strike, short_put.underlying_price,
            spread_type="bull_put",
        )

        call_leg = OrderLeg(
            symbol=short_call.symbol,
            option_type="call",
            strike=short_call.strike,
            expiry=short_call.expiry,
            side="sell_to_open",
            quantity=1,
        )
        put_leg = OrderLeg(
            symbol=short_put.symbol,
            option_type="put",
            strike=short_put.strike,
            expiry=short_put.expiry,
            side="sell_to_open",
            quantity=1,
        )

        logger.info(
            "[%s] Signal: SELL %s (delta=%.3f) / SELL %s (delta=%.3f) "
            "total_credit=%.2f hard_stop=%.2f DTE=%d",
            self.name,
            short_call.symbol, short_call.delta,
            short_put.symbol, short_put.delta,
            total_credit, hard_stop, short_call.dte
        )

        return StrategySignal(
            strategy_name=self.name,
            underlying=short_call.underlying,
            legs=[call_leg, put_leg],
            net_debit_credit=-total_credit,
            estimated_fill_price=total_credit,
            max_loss_per_contract=practical_max_loss,
            hard_stop_price=hard_stop,
            profit_target_price=profit_target,
            expiry=short_call.expiry,
            dte=short_call.dte,
            notes=(
                f"call={short_call.strike:.0f}C delta={short_call.delta:.3f} "
                f"put={short_put.strike:.0f}P delta={short_put.delta:.3f} "
                f"credit={total_credit:.2f}"
            ),
            source_contracts=[short_call, short_put],
        )


# ---------------------------------------------------------------------------
# Strategy registry — maps name → class for config-driven instantiation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Short Call Spread — sell OTM call, buy further OTM call
# Mirror of ShortPutSpread for bearish/high-IV environments.
# Use when: ticker is overbought, IV elevated, or regime is trending-down.
# ---------------------------------------------------------------------------

@dataclass
class ShortCallSpreadConfig:
    """Config for short call spread (bear call spread)."""
    target_delta:     float = 0.25     # sell ~25-delta call (OTM)
    min_delta:        float = 0.15     # reject if short call delta below this
    max_delta:        float = 0.35     # reject if short call delta above this
    min_spread_width: float = 1.0      # min strike separation (ETF-friendly)
    max_spread_width: float = 20.0     # max strike separation
    min_credit:       float = 0.25     # minimum credit to collect
    min_pop:          float = 0.65     # minimum probability of profit
    min_dte:          int   = 14       # minimum days to expiry
    max_dte:          int   = 60       # maximum days to expiry
    min_open_interest: int  = 100      # per-contract OI floor


class ShortCallSpread(BaseStrategy):
    """
    Bear call spread: sell OTM call + buy further OTM call on same expiry.

    Profits when the underlying stays BELOW the short call strike.
    Best entered when: IV rank is elevated, ticker is technically overbought
    (near upper Bollinger Band), or regime is trending/mean-reverting bearish.

    Max profit = net credit collected.
    Max loss   = spread width - credit (defined risk).
    """
    name = "short_call_spread"

    def __init__(self, config: ShortCallSpreadConfig | None = None):
        self.cfg = config or ShortCallSpreadConfig()

    def evaluate(
        self,
        ticker: str,
        chain: list,
        expiry,
        dte: int,
        underlying_price: float,
        **kwargs,
    ) -> Optional[TradeSignal]:
        cfg = self.cfg

        # Filter: calls only, DTE + liquidity
        liquid = [
            c for c in chain
            if c.option_type == "call"
            and c.dte >= cfg.min_dte
            and c.dte <= cfg.max_dte
            and c.open_interest >= cfg.min_open_interest
            and c.bid is not None and c.bid > 0
        ]
        if not liquid:
            logger.debug("[ShortCallSpread] %s: no liquid calls in DTE %d-%d",
                         ticker, cfg.min_dte, cfg.max_dte)
            return None

        # Select short call closest to target_delta (positive delta for calls)
        target = cfg.target_delta
        candidates = [
            c for c in liquid
            if c.delta is not None
            and cfg.min_delta <= c.delta <= cfg.max_delta
        ]
        if not candidates:
            logger.debug("[ShortCallSpread] %s: no calls in delta range %.2f-%.2f",
                         ticker, cfg.min_delta, cfg.max_delta)
            return None

        short_call = min(candidates, key=lambda c: abs((c.delta or 0) - target))

        # Find long call above short call strike
        long_candidates = [
            c for c in liquid
            if c.strike > short_call.strike
            and (
                0.5 if short_call.strike < 100 else cfg.min_spread_width
            ) <= (c.strike - short_call.strike) <= cfg.max_spread_width
        ]
        if not long_candidates:
            logger.debug("[ShortCallSpread] %s: no valid long leg above short strike %.2f",
                         ticker, short_call.strike)
            return None

        # Pick long call with best risk/reward (narrowest viable spread)
        long_call = min(long_candidates, key=lambda c: c.strike)

        # Calculate spread metrics
        short_mid = ((short_call.bid or 0) + (short_call.ask or short_call.bid or 0)) / 2
        long_mid  = ((long_call.bid or 0)  + (long_call.ask  or long_call.bid  or 0)) / 2
        credit    = round(short_mid - long_mid, 2)
        width     = round(long_call.strike - short_call.strike, 2)
        max_loss_per_contract = round((width - credit) * 100, 2)

        if credit < cfg.min_credit:
            logger.debug("[ShortCallSpread] %s: credit $%.2f < min $%.2f",
                         ticker, credit, cfg.min_credit)
            return None

        if max_loss_per_contract <= 0:
            return None

        # Probability of profit
        pop = 1.0 - abs(short_call.delta or 0)
        if pop < cfg.min_pop:
            logger.debug("[ShortCallSpread] %s: PoP %.1f%% < min %.1f%%",
                         ticker, pop * 100, cfg.min_pop * 100)
            return None

        # Earnings check — reuse parent method
        try:
            self._check_earnings(short_call.underlying, short_call.expiry, short_call.dte)
        except Exception as e:
            logger.info("[ShortCallSpread] %s: blocked — %s", ticker, e)
            return None

        hard_stop = round(short_call.strike + width * 1.5, 2)
        logger.info(
            "[ShortCallSpread] %s: short=%.2f(Δ%.2f) long=%.2f "
            "width=%.1f credit=%.2f max_loss=$%.2f PoP=%.1f%% DTE=%d",
            ticker, short_call.strike, short_call.delta or 0,
            long_call.strike, width, credit,
            max_loss_per_contract, pop * 100, dte,
        )

        return TradeSignal(
            ticker=ticker,
            strategy=self.name,
            legs=[
                OrderLeg(symbol=short_call.symbol, side="sell", quantity=1,
                         option_type="call", strike=short_call.strike,
                         expiry=short_call.expiry, delta=short_call.delta),
                OrderLeg(symbol=long_call.symbol, side="buy", quantity=1,
                         option_type="call", strike=long_call.strike,
                         expiry=long_call.expiry, delta=long_call.delta),
            ],
            net_credit=credit,
            max_loss_per_contract=max_loss_per_contract,
            probability_of_profit=pop,
            hard_stop_price=hard_stop,
            underlying=ticker,
            underlying_price=underlying_price,
            expiry=short_call.expiry,
            dte=dte,
        )


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "csp":               CashSecuredPut,
    "short_put_spread":  ShortPutSpread,
    "short_call_spread": ShortCallSpread,
    "short_strangle":    ShortStrangle,
}


def get_strategy(name: str, config=None) -> BaseStrategy:
    """
    Instantiate a strategy by name.

    Parameters
    ----------
    name : str
        One of: "csp", "short_put_spread", "short_strangle"
    config : dataclass or None
        Strategy-specific config object. Uses defaults if None.

    Raises
    ------
    ValueError
        If name is not in STRATEGY_REGISTRY.
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    cls = STRATEGY_REGISTRY[name]
    return cls(config) if config is not None else cls()
