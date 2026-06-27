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
    calc_spread,
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

    # VRP gate sizing multiplier (0..1). Default 1.0 = no shrink. Set by the
    # (gated) vol-risk-premium gate in the pipeline when active: 1.0 when the
    # premium is clearly rich, ramping toward 0 as VRP thins. The risk manager
    # can multiply position size by this. Default 1.0 keeps behavior unchanged
    # when the gate is inactive.
    vrp_size_factor: float = 1.0


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
    target_delta: float = -0.15         # target put delta to sell (lowered from -0.20:
                                         # PoT≈2×delta rule means -0.20 often breaches the
                                         # 35% PoT hard-reject; -0.15 keeps PoT in the 25-35% zone)
    min_delta: float = -0.22            # reject if delta more negative than this
    max_delta: float = -0.08            # reject if delta less negative than this
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
    short_delta: float = -0.15          # sell this delta put (lowered from -0.25:
                                         # PoT≈2×delta rule means -0.25 ~50% PoT, almost
                                         # always breaching the 35% PoT hard-reject)
    min_delta: float = -0.20            # reject if delta more negative than this
    max_delta: float = -0.10            # reject if delta less negative than this
    long_delta: float = -0.07           # buy this delta put (further OTM, scaled down to match)
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

    def evaluate(self, chain: list[EnrichedOptionRow], risk_budget_dollars: float | None = None) -> StrategySignal:
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
        # AUDIT FIX: was a "relax and take the closest" fallback, meaning
        # this strategy would trade an arbitrarily-far-off delta (and
        # therefore an arbitrarily different risk profile than intended)
        # rather than skip the trade. CSP and ShortCallSpread both hard-
        # reject when nothing is in range -- standardized on hard-reject
        # everywhere as the more conservative, capital-preservation-
        # consistent behavior. No principled reason ShortPutSpread alone
        # should be willing to force a trade the other strategies wouldn't.
        short_leg_candidates = [
            c for c in put_candidates
            if cfg.min_delta <= c.delta <= cfg.max_delta
        ]
        if not short_leg_candidates:
            raise LiquidityFilterError(
                f"{put_candidates[0].underlying} chain",
                f"[{self.name}] No puts in delta range "
                f"[{cfg.min_delta:.2f}, {cfg.max_delta:.2f}]"
            )

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

        # BUDGET-FIT: If the natural long leg produces a max loss that exceeds
        # the risk budget, walk the long leg closer to the short strike
        # (higher strike = narrower spread = lower max loss) until it fits.
        # Caller passes risk_budget_dollars; None means skip (backward-compat).
        if risk_budget_dollars is not None:
            # Sort ascending by strike (widest→narrowest for puts)
            _sorted_candidates = sorted(long_leg_candidates, key=lambda c: c.strike)
            _chosen = None
            for _cand in _sorted_candidates:
                _trial_width = short_put.strike - _cand.strike
                _trial_credit = short_put.mid_price - _cand.mid_price
                _trial_max_loss = (_trial_width - _trial_credit) * 100
                if _trial_max_loss <= risk_budget_dollars:
                    _chosen = _cand
                    break
            if _chosen is not None and _chosen is not long_put:
                logger.info(
                    "[%s] Budget-fit: narrowed spread from %.0f-wide (loss=$%.0f) "
                    "to %.0f-wide (loss=$%.0f) to fit risk budget $%.0f",
                    self.name,
                    short_put.strike - long_put.strike,
                    (short_put.strike - long_put.strike - (short_put.mid_price - long_put.mid_price)) * 100,
                    short_put.strike - _chosen.strike,
                    (short_put.strike - _chosen.strike - (short_put.mid_price - _chosen.mid_price)) * 100,
                    risk_budget_dollars,
                )
                long_put = _chosen
            elif _chosen is None:
                raise LiquidityFilterError(
                    f"{short_put.underlying} chain",
                    f"[{self.name}] No spread width fits risk budget ${risk_budget_dollars:.0f} "
                    f"(even 1-wide exceeds budget — equity too small for this ticker)"
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
            enriched=valid,
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
    call_delta: float = 0.15            # sell this delta call (lowered from 0.20 — see PoT note above)
    put_delta: float = -0.15            # sell this delta put (lowered from -0.20 — see PoT note above)
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

    def evaluate(self, chain: list[EnrichedOptionRow], risk_budget_dollars: float | None = None) -> StrategySignal:
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
        # AUDIT FIX: both legs previously had a "relax tolerance" fallback
        # that fell through to ANY call/put in the chain if nothing was
        # within delta_tolerance, regardless of how far off the resulting
        # delta was. CSP and ShortCallSpread hard-reject in the equivalent
        # situation -- standardized on hard-reject everywhere. See the same
        # note in ShortPutSpread.evaluate() for the full reasoning.
        call_candidates = [
            c for c in liquid
            if c.option_type == "call"
            and abs(c.delta - cfg.call_delta) <= cfg.delta_tolerance
        ]
        if not call_candidates:
            raise LiquidityFilterError(
                _clean_ticker(liquid[0].underlying),
                f"[{self.name}] No calls within {cfg.delta_tolerance:.2f} "
                f"of target delta {cfg.call_delta:.2f}"
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
            raise LiquidityFilterError(
                _clean_ticker(liquid[0].underlying),
                f"[{self.name}] No puts within {cfg.delta_tolerance:.2f} "
                f"of target delta {cfg.put_delta:.2f}"
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
    target_delta:      float = 0.15    # sell ~15-delta call (lowered from 0.25:
                                        # PoT≈2×delta rule means 0.25 ~50% PoT, almost
                                        # always breaching the 35% PoT hard-reject)
    long_delta:        float = 0.07    # buy this delta call (further OTM, mirrors ShortPutSpread)
    min_delta:         float = 0.08    # reject if short call delta below this
    max_delta:         float = 0.22    # reject if short call delta above this
    min_dte:           int   = 14      # minimum days to expiry
    max_dte:           int   = 60      # maximum days to expiry
    min_open_interest: int   = 100     # per-contract OI floor
    max_spread_pct:    float = 0.25    # max bid/ask spread as % of mid
    min_spread_width:  float = 1.0     # min strike separation (ETF-friendly)
    max_spread_width:  float = 20.0    # max strike separation
    min_credit:        float = 0.25    # minimum credit to collect
    stop_multiplier:   float = 2.0     # stop at 2x credit received
    profit_target_pct: float = 0.50    # close at 50% of max profit
    min_pop:           float = 0.65    # minimum probability of profit
    vp_check_enabled:  bool  = True    # enable volume-profile strike safety check
    vp_min_hvn_distance_pct: float = 1.5  # min % distance from short strike to any HVN


class ShortCallSpread(BaseStrategy):
    """
    Bear call spread: sell OTM call + buy further OTM call on same expiry.

    Profits when the underlying stays BELOW the short call strike.
    Best entered when: IV rank is elevated, ticker is technically overbought
    (near upper Bollinger Band), or regime is trending/mean-reverting bearish.

    Mirrors ShortPutSpread's structure exactly (same checks, same return
    contract) so it can be selected via STRATEGY_REGISTRY interchangeably.

    Max profit = net credit collected.
    Max loss   = spread width - credit (defined risk).
    """

    def __init__(self, config: Optional[ShortCallSpreadConfig] = None):
        super().__init__("ShortCallSpread")
        self.config = config or ShortCallSpreadConfig()

    def evaluate(self, chain: list[EnrichedOptionRow], risk_budget_dollars: float | None = None) -> StrategySignal:
        self._require_nonempty(chain)
        valid = self._require_greeks(chain)

        cfg = self.config

        # Filter: calls only, DTE + liquidity
        call_candidates = [
            c for c in valid
            if c.option_type == "call"
            and c.dte >= cfg.min_dte
            and c.dte <= cfg.max_dte
            and (c.open_interest is None or c.open_interest >= cfg.min_open_interest)
            and (c.spread_pct is None or c.spread_pct <= cfg.max_spread_pct)
            and c.bid is not None
            and c.ask is not None
            and c.mid_price is not None
        ]

        if len(call_candidates) < 2:
            raise LiquidityFilterError(
                f"{_clean_ticker(valid[0].underlying) if valid else '?'} chain",
                f"[{self.name}] Need >= 2 liquid calls, found {len(call_candidates)}"
            )

        # Find short leg (closer to ATM, positive delta for calls)
        short_leg_candidates = [
            c for c in call_candidates
            if cfg.min_delta <= c.delta <= cfg.max_delta
        ]
        if not short_leg_candidates:
            raise LiquidityFilterError(
                f"{call_candidates[0].underlying} chain",
                f"[{self.name}] No calls in delta range "
                f"[{cfg.min_delta:.2f}, {cfg.max_delta:.2f}]"
            )

        short_call = min(
            short_leg_candidates,
            key=lambda c: abs(c.delta - cfg.target_delta)
        )

        # Find long leg (further OTM, higher strike, lower delta magnitude)
        long_leg_candidates = [
            c for c in call_candidates
            if c.strike > short_call.strike          # must be higher strike
            and (c.strike - short_call.strike) >= (
                    0.5 if short_call.strike < 100 else cfg.min_spread_width
                )
            and (c.strike - short_call.strike) <= cfg.max_spread_width
        ]

        if not long_leg_candidates:
            raise LiquidityFilterError(
                f"{short_call.underlying} chain",
                f"[{self.name}] No valid long leg found above short strike "
                f"{short_call.strike:.1f} with width "
                f"[{cfg.min_spread_width}, {cfg.max_spread_width}]"
            )

        long_call = min(
            long_leg_candidates,
            key=lambda c: abs(c.delta - cfg.long_delta)
        )

        # BUDGET-FIT: mirror of ShortPutSpread logic — walk the long call leg
        # closer to the short strike (lower strike = narrower spread = lower
        # max loss) until the spread fits the risk budget.
        if risk_budget_dollars is not None:
            # Sort descending by strike (widest→narrowest for calls)
            _sorted_candidates = sorted(long_leg_candidates, key=lambda c: -c.strike)
            _chosen = None
            for _cand in _sorted_candidates:
                _trial_width = _cand.strike - short_call.strike
                _trial_credit = short_call.mid_price - _cand.mid_price
                _trial_max_loss = (_trial_width - _trial_credit) * 100
                if _trial_max_loss <= risk_budget_dollars:
                    _chosen = _cand
                    break
            if _chosen is not None and _chosen is not long_call:
                logger.info(
                    "[%s] Budget-fit: narrowed spread from %.0f-wide (loss=$%.0f) "
                    "to %.0f-wide (loss=$%.0f) to fit risk budget $%.0f",
                    self.name,
                    long_call.strike - short_call.strike,
                    (long_call.strike - short_call.strike - (short_call.mid_price - long_call.mid_price)) * 100,
                    _chosen.strike - short_call.strike,
                    (_chosen.strike - short_call.strike - (short_call.mid_price - _chosen.mid_price)) * 100,
                    risk_budget_dollars,
                )
                long_call = _chosen
            elif _chosen is None:
                raise LiquidityFilterError(
                    f"{short_call.underlying} chain",
                    f"[{self.name}] No spread width fits risk budget ${risk_budget_dollars:.0f} "
                    f"(even 1-wide exceeds budget — equity too small for this ticker)"
                )

        spread_width  = long_call.strike - short_call.strike
        short_credit  = short_call.mid_price
        long_cost     = long_call.mid_price
        net_credit    = short_credit - long_cost

        if net_credit < cfg.min_credit:
            raise LiquidityFilterError(
                f"{short_call.underlying} chain",
                f"[{self.name}] Net credit ${net_credit:.2f} < min ${cfg.min_credit:.2f}. "
                f"Short={short_credit:.2f} Long={long_cost:.2f}"
            )

        # Use spread_math for all P&L calculations — single source of truth
        errors = validate_spread_inputs(
            low_bid=short_call.bid, low_ask=short_call.ask,
            high_bid=long_call.bid, high_ask=long_call.ask,
            low_strike=short_call.strike, high_strike=long_call.strike,
        )
        if errors:
            raise LiquidityFilterError(
                f"{short_call.underlying} chain",
                f"[{self.name}] Spread input validation failed: {'; '.join(errors)}"
            )

        spread_math = calc_spread(
            spread_type="bear_call",
            action="entry",
            low_strike=short_call.strike,
            low_bid=short_call.bid,
            low_ask=short_call.ask,
            high_strike=long_call.strike,
            high_bid=long_call.bid,
            high_ask=long_call.ask,
            num_contracts=1,
            underlying_price=short_call.underlying_price,
        )

        max_loss_per_contract = spread_math["max_loss"]       # already per-contract dollars
        hard_stop             = calc_stop_price(net_credit, cfg.stop_multiplier)
        profit_target         = calc_profit_target(net_credit, cfg.profit_target_pct)

        # Earnings hard filter + volume profile — shared BaseStrategy helpers.
        # Consistent with ShortPutSpread / CashSecuredPut / ShortStrangle.
        self._check_earnings(short_call.underlying, short_call.expiry, short_call.dte)
        self._check_volume_profile(
            short_call.underlying,
            short_call.strike,
            short_call.underlying_price,
            spread_type="bear_call",
            min_hvn_distance_pct=cfg.vp_min_hvn_distance_pct,
        )
        # GEX gamma wall check — don't short near the call wall or below pin
        self._check_gex(
            enriched=valid,
            short_strike=short_call.strike,
            spot=short_call.underlying_price,
            expiry=short_call.expiry,
            dte=short_call.dte,
            atm_iv=short_call.iv or 0.20,
        )

        # Probability of profit + probability of touch validation —
        # same hard-reject pattern as ShortPutSpread.
        short_iv = short_call.iv or 0.0
        if short_iv > 0 and short_call.dte and short_call.underlying_price:
            rate = get_risk_free_rate()
            prob = pop_spread(
                spread_type="bear_call",
                short_strike=short_call.strike,
                long_strike=long_call.strike,
                net_credit=net_credit,
                spot=short_call.underlying_price,
                sigma=short_iv,
                rate=rate,
                days_to_expiry=short_call.dte,
            )
            if prob["pop"] < cfg.min_pop:
                raise LiquidityFilterError(
                    short_call.underlying,
                    f"[{self.name}] PoP {prob['pop']:.0%} < min {cfg.min_pop:.0%} "
                    f"(break-even=${prob['break_even']:.2f}, PoT={prob['pot_short']:.0%}). "
                    f"LABEL: BS lognormal PoP — may underestimate fat-tail risk by 5-10pp."
                )
            if prob["pot_warning"]:
                raise LiquidityFilterError(
                    short_call.underlying,
                    f"[{self.name}] PoT {prob['pot_short']:.0%} > 35% threshold — "
                    f"strike ${short_call.strike:.0f} has elevated touch risk. REJECT. "
                    f"(PoP={prob['pop']:.0%}, break-even=${prob['break_even']:.2f})"
                )

        short_leg = OrderLeg(
            symbol=short_call.symbol,
            option_type="call",
            strike=short_call.strike,
            expiry=short_call.expiry,
            side="sell_to_open",
            quantity=1,
        )
        long_leg = OrderLeg(
            symbol=long_call.symbol,
            option_type="call",
            strike=long_call.strike,
            expiry=long_call.expiry,
            side="buy_to_open",
            quantity=1,
        )

        logger.info(
            "[%s] Signal: SELL %s (delta=%.3f) / BUY %s (delta=%.3f) "
            "width=%.1f credit=%.2f max_loss=$%.2f stop=%.2f DTE=%d",
            self.name,
            short_call.symbol, short_call.delta,
            long_call.symbol, long_call.delta,
            spread_width, net_credit, max_loss_per_contract,
            hard_stop, short_call.dte
        )

        return StrategySignal(
            strategy_name=self.name,
            underlying=short_call.underlying,
            legs=[short_leg, long_leg],
            net_debit_credit=-net_credit,
            estimated_fill_price=net_credit,
            max_loss_per_contract=max_loss_per_contract,
            hard_stop_price=hard_stop,
            profit_target_price=profit_target,
            expiry=short_call.expiry,
            dte=short_call.dte,
            notes=(
                f"short={short_call.strike:.0f}C delta={short_call.delta:.3f} "
                f"long={long_call.strike:.0f}C delta={long_call.delta:.3f} "
                f"width={spread_width:.1f} credit={net_credit:.2f}"
            ),
            source_contracts=[short_call, long_call],
        )


# ---------------------------------------------------------------------------
# Iron Condor — defined-risk neutral premium-selling structure
# Sell OTM put + buy lower OTM put (bull put spread)
#   + sell OTM call + buy higher OTM call (bear call spread), same expiry.
# This is the defined-risk upgrade to ShortStrangle for neutral-direction
# tickers: capped max loss instead of the strangle's wide/undefined risk —
# strictly better for a capital-preservation-first mandate.
#
# DESIGN: composes the two ALREADY-HARDENED spread strategies rather than
# reimplementing leg selection. It runs ShortPutSpread.evaluate() and
# ShortCallSpread.evaluate() on the same chain, then merges the four legs
# into one defined-risk signal. This means every check those strategies
# already enforce (delta window hard-reject, PoP/PoT gate, GEX/volume-profile
# safety, liquidity) applies to BOTH sides of the condor for free, and any
# future fix to the underlying spreads automatically flows through.
#
# GATED: not in STRATEGY_REGISTRY's default routing — the orchestrator only
# routes to it after the 30-trade walk-forward milestone AND an explicit
# iron_condor_enabled flag (see OrchestratorConfig). Building it now keeps it
# tested and dormant; activation stays a deliberate human decision.
# ---------------------------------------------------------------------------

@dataclass
class IronCondorConfig:
    """
    Config for the iron condor. Delegates per-side leg selection to the
    underlying ShortPutSpread / ShortCallSpread configs so the condor stays
    consistent with the standalone spreads (single source of truth for
    delta targets, DTE window, liquidity, PoP). Only condor-level params
    live here directly.
    """
    put_side:  ShortPutSpreadConfig  = field(default_factory=ShortPutSpreadConfig)
    call_side: ShortCallSpreadConfig = field(default_factory=ShortCallSpreadConfig)
    # Condor-level minimum TOTAL credit (both spreads combined). Set a touch
    # above either single-side min_credit since we collect from both wings.
    min_total_credit:  float = 0.40
    stop_multiplier:   float = 2.0    # stop at 2x total credit received
    profit_target_pct: float = 0.50   # close at 50% of max profit


class IronCondor(BaseStrategy):
    """
    Defined-risk neutral premium seller.

    Profits when the underlying stays BETWEEN the short put and short call
    strikes through expiry — i.e. a low-volatility, range-bound view. Both
    sides are vertical spreads, so max loss is capped at the wider spread
    width minus total credit (the two sides can't both lose maximally — the
    underlying can only finish on one side).
    """

    def __init__(self, config: Optional[IronCondorConfig] = None):
        super().__init__("iron_condor")
        self.cfg = config or IronCondorConfig()
        # Compose the two hardened spread strategies.
        self._put_spread  = ShortPutSpread(self.cfg.put_side)
        self._call_spread = ShortCallSpread(self.cfg.call_side)

    def evaluate(self, chain: list[EnrichedOptionRow]) -> StrategySignal:
        # Each side runs its full evaluate() — including its own hard-reject
        # delta window, PoP/PoT gate, and GEX/volume-profile checks. If either
        # side can't find a qualifying spread it raises (LiquidityFilterError)
        # we let that propagate so the condor is only
        # built when BOTH sides independently qualify. A one-sided "condor"
        # is just a spread and would defeat the neutral, defined-risk purpose.
        put_signal  = self._put_spread.evaluate(chain)
        call_signal = self._call_spread.evaluate(chain)

        # Both sides must share an expiry (they will, since _pick_expiry feeds
        # a single-expiry chain — but assert it rather than assume).
        if put_signal.expiry != call_signal.expiry:
            raise LiquidityFilterError(
                put_signal.underlying,
                f"[{self.name}] put/call sides resolved different expiries "
                f"({put_signal.expiry} vs {call_signal.expiry}) — cannot form condor"
            )

        total_credit = put_signal.estimated_fill_price + call_signal.estimated_fill_price
        if total_credit < self.cfg.min_total_credit:
            raise LiquidityFilterError(
                put_signal.underlying,
                f"[{self.name}] total credit {total_credit:.2f} < "
                f"min {self.cfg.min_total_credit:.2f}"
            )

        # Max loss for an iron condor = max(put_side_width, call_side_width)
        # * 100 - total_credit*100. The underlying can only breach ONE side at
        # expiry, so we lose the WIDER spread's max, offset by the FULL credit
        # collected from both sides. Using max() (not sum) is the key
        # capital-efficiency property that makes the condor defined-risk.
        put_side_max_loss  = put_signal.max_loss_per_contract   # already (width - put_credit)*100
        call_side_max_loss = call_signal.max_loss_per_contract  # already (width - call_credit)*100
        # Each side's max_loss already netted its OWN credit; add back the
        # opposite side's credit since that's collected regardless of which
        # side is breached.
        condor_max_loss = max(
            put_side_max_loss  - call_signal.estimated_fill_price * 100,
            call_side_max_loss - put_signal.estimated_fill_price  * 100,
        )
        # Floor at a small positive value — defined-risk must be > 0 and finite.
        condor_max_loss = max(condor_max_loss, 1.0)

        # Hard stop on combined credit (2x total credit received).
        hard_stop = round(total_credit * self.cfg.stop_multiplier, 3)
        profit_target = round(total_credit * (1.0 - self.cfg.profit_target_pct), 3)

        legs = list(put_signal.legs) + list(call_signal.legs)

        logger.info(
            "[%s] %s condor: put-side credit=%.2f call-side credit=%.2f "
            "total=%.2f max_loss=%.2f dte=%s",
            self.name, put_signal.underlying,
            put_signal.estimated_fill_price, call_signal.estimated_fill_price,
            total_credit, condor_max_loss, put_signal.dte,
        )

        return StrategySignal(
            strategy_name=self.name,
            underlying=put_signal.underlying,
            legs=legs,
            net_debit_credit=-total_credit,
            estimated_fill_price=total_credit,
            max_loss_per_contract=condor_max_loss,
            hard_stop_price=hard_stop,
            profit_target_price=profit_target,
            expiry=put_signal.expiry,
            dte=put_signal.dte,
            notes=(
                f"iron_condor put_side_credit={put_signal.estimated_fill_price:.2f} "
                f"call_side_credit={call_signal.estimated_fill_price:.2f} "
                f"total_credit={total_credit:.2f} max_loss={condor_max_loss:.2f}"
            ),
            source_contracts=list(put_signal.source_contracts) + list(call_signal.source_contracts),
        )


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "csp":               CashSecuredPut,
    "short_put_spread":  ShortPutSpread,
    "short_call_spread": ShortCallSpread,
    "short_strangle":    ShortStrangle,
    "iron_condor":       IronCondor,
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
