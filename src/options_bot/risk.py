"""
Risk management module.

Two components:

1. RiskManager — validates trades before they reach the broker.
   Enforces:
     - Max risk % of equity per trade
     - Hard stop-loss mandatory on every order
     - Daily loss limit (halt trading when hit)
     - Max trades per day
     - Min/max position size in contracts
     - Liquidity sanity check (bid/ask present)

2. ExecutionGuard — final gate before broker submission.
   Raises RiskVetoError immediately if:
     - hard_stop_price is None
     - risk_approved is False
     - position_size_contracts < 1

Mathematical rationale (written before code, per system directive):

  Position sizing formula (risk-budget based):
    risk_budget_dollars = equity * risk_budget_pct
    max_loss_per_contract = spread_width * 100  (for spreads)
                          = premium_paid * 100  (for long options)
                          = (spread_width - credit) * 100 (for credit spreads)
    contracts = floor(risk_budget_dollars / max_loss_per_contract)
    contracts = clamp(contracts, min_contracts, max_contracts)

  Daily loss tracking:
    daily_pnl starts at 0 each trading day
    if daily_pnl <= -(equity * max_daily_loss_pct): halt all new trades

Every position sizing calculation strictly adheres to a maximum risk of
risk_budget_pct of total equity per trade — this is unbreakable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from .contracts import ApprovedOrder, EnrichedOptionRow, OrderLeg, OptionType
from .exceptions import DataValidationError, RiskVetoError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """
    All risk parameters in one place.
    Pass this into RiskManager at startup.
    Override per-environment via env vars or YAML config.
    """
    # Per-trade risk limit — UNBREAKABLE
    risk_budget_pct: float = 0.02          # 2% of equity max per trade

    # Daily halt rules
    max_daily_loss_pct: float = 0.05       # halt if down 5% on the day
    max_trades_per_day: int = 5            # max new positions opened per day

    # Position size bounds
    min_contracts: int = 1
    max_contracts: int = 10               # hard cap regardless of sizing math

    # Liquidity requirements (redundant with market_data filter but defense-in-depth)
    min_open_interest: int = 100
    max_spread_pct: float = 0.25           # reject if (ask-bid)/mid > 25%

    # Slippage assumption for realistic fill estimation
    slippage_pct: float = 0.02            # 2% of mid assumed as slippage cost

    # Profit target defaults (can be overridden per strategy)
    default_profit_target_pct: float = 0.50   # close at 50% of max profit

    def validate(self) -> None:
        """Raises DataValidationError if any parameter is out of bounds."""
        if not 0 < self.risk_budget_pct <= 0.25:
            raise DataValidationError(
                "risk_budget_pct",
                f"Must be in (0, 0.25], got {self.risk_budget_pct}. "
                "Never risk more than 25% of equity on a single trade."
            )
        if not 0 < self.max_daily_loss_pct <= 0.20:
            raise DataValidationError(
                "max_daily_loss_pct",
                f"Must be in (0, 0.20], got {self.max_daily_loss_pct}"
            )
        if self.max_trades_per_day < 1:
            raise DataValidationError(
                "max_trades_per_day", "Must be >= 1"
            )
        if self.min_contracts < 1:
            raise DataValidationError(
                "min_contracts", "Must be >= 1"
            )
        if self.max_contracts < self.min_contracts:
            raise DataValidationError(
                "max_contracts",
                f"Must be >= min_contracts ({self.min_contracts})"
            )


# ---------------------------------------------------------------------------
# Risk decision output
# ---------------------------------------------------------------------------

@dataclass
class RiskDecision:
    """
    Result from RiskManager.evaluate().

    approved=True means the trade passed all checks and is sized correctly.
    approved=False means it was vetoed — rejection_reason explains why.
    """
    approved: bool
    position_size_contracts: int
    max_loss_dollars: float
    risk_budget_dollars: float
    rejection_reason: str = ""

    # Computed fields for transparency
    equity_at_decision: float = 0.0
    trades_today: int = 0
    daily_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Daily state tracker
# ---------------------------------------------------------------------------

@dataclass
class DailyState:
    """Tracks intraday risk metrics. Resets at the start of each trading day."""
    trade_date: date = field(default_factory=date.today)
    trades_opened: int = 0
    realized_pnl: float = 0.0        # closed P&L today
    unrealized_pnl: float = 0.0      # mark-to-market on open positions

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    def reset_if_new_day(self) -> None:
        today = date.today()
        if self.trade_date != today:
            logger.info(
                "[DailyState] New trading day %s — resetting counters "
                "(previous day: trades=%d, pnl=%.2f)",
                today, self.trades_opened, self.total_pnl
            )
            self.trade_date = today
            self.trades_opened = 0
            self.realized_pnl = 0.0
            self.unrealized_pnl = 0.0


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Validates and sizes every trade before it reaches the broker.

    Usage:
        rm = RiskManager(equity=50_000, config=RiskConfig())

        decision = rm.evaluate(
            option=enriched_row,
            max_loss_per_contract=250.0,   # e.g. spread width * 100
            hard_stop_price=5.00,
            strategy_name="short_put_spread",
        )

        if decision.approved:
            order = rm.build_approved_order(...)
        else:
            logger.warning("Trade vetoed: %s", decision.rejection_reason)
    """

    def __init__(self, equity: float, config: Optional[RiskConfig] = None):
        """
        Parameters
        ----------
        equity : float
            Current account equity in dollars. Update this daily.
        config : RiskConfig or None
            Risk parameters. Defaults to conservative defaults if None.
        """
        if equity <= 0:
            raise DataValidationError("equity", f"Must be positive, got {equity}")

        self._equity = equity
        self.config = config or RiskConfig()
        self.config.validate()
        self._daily = DailyState()

        logger.info(
            "[RiskManager] Initialized: equity=$%.2f, risk_pct=%.1f%%, "
            "max_daily_loss=%.1f%%, max_trades=%d",
            equity,
            self.config.risk_budget_pct * 100,
            self.config.max_daily_loss_pct * 100,
            self.config.max_trades_per_day,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        return self._equity

    def update_equity(self, new_equity: float) -> None:
        """Update account equity (call at start of each session or after fills)."""
        if new_equity <= 0:
            raise DataValidationError("equity", f"Must be positive, got {new_equity}")
        logger.info(
            "[RiskManager] Equity updated: $%.2f → $%.2f (Δ$%.2f)",
            self._equity, new_equity, new_equity - self._equity
        )
        self._equity = new_equity

    def record_trade_opened(self) -> None:
        """Call this after a trade is confirmed filled."""
        self._daily.reset_if_new_day()
        self._daily.trades_opened += 1
        logger.info(
            "[RiskManager] Trade recorded: %d/%d today",
            self._daily.trades_opened, self.config.max_trades_per_day
        )

    def record_pnl(self, realized: float = 0.0, unrealized: float = 0.0) -> None:
        """Update daily P&L tracking."""
        self._daily.reset_if_new_day()
        self._daily.realized_pnl += realized
        self._daily.unrealized_pnl = unrealized   # replace, not add
        logger.debug(
            "[RiskManager] P&L update: realized=%.2f unrealized=%.2f total=%.2f",
            self._daily.realized_pnl, self._daily.unrealized_pnl, self._daily.total_pnl
        )

    def evaluate(
        self,
        max_loss_per_contract: float,
        hard_stop_price: float,
        option: Optional[EnrichedOptionRow] = None,
        strategy_name: str = "",
    ) -> RiskDecision:
        """
        Core risk evaluation. Returns a RiskDecision — never raises.

        Parameters
        ----------
        max_loss_per_contract : float
            Maximum dollar loss per contract in the worst case.
            For a credit spread: (spread_width - credit_received) * 100
            For a long option: premium_paid * 100
            For a naked short (DO NOT USE): theoretically unlimited — will be vetoed.

        hard_stop_price : float
            The stop-loss price. Must be > 0.

        option : EnrichedOptionRow or None
            The option being evaluated. Used for liquidity checks if provided.

        strategy_name : str
            For logging.
        """
        logger.info(
            "[RiskManager] Evaluating: strategy=%s max_loss_per_contract=$%.2f",
            strategy_name or "unnamed", max_loss_per_contract
        )

        self._daily.reset_if_new_day()

        # --- Pre-checks ---

        # Hard stop must be defined
        if hard_stop_price <= 0:
            return self._veto(
                f"hard_stop_price={hard_stop_price} must be > 0 — "
                "every position requires a defined exit"
            )

        # Max loss must be positive and finite
        if max_loss_per_contract <= 0:
            return self._veto(
                f"max_loss_per_contract={max_loss_per_contract} must be > 0. "
                "If this is a naked short, it has been correctly rejected."
            )

        if not math.isfinite(max_loss_per_contract):
            return self._veto(
                "max_loss_per_contract is infinite — "
                "naked/unhedged positions are prohibited"
            )

        # Max loss sanity — if a single contract costs more than the entire
        # risk budget, we can't even open 1 contract
        risk_budget = self._equity * self.config.risk_budget_pct
        if max_loss_per_contract > risk_budget:
            return self._veto(
                f"max_loss_per_contract=${max_loss_per_contract:.2f} exceeds "
                f"risk_budget=${risk_budget:.2f} "
                f"({self.config.risk_budget_pct:.1%} of ${self._equity:.2f}). "
                "Cannot open even 1 contract within risk limits."
            )

        # Daily loss halt
        daily_loss_limit = self._equity * self.config.max_daily_loss_pct
        if self._daily.total_pnl <= -daily_loss_limit:
            return self._veto(
                f"Daily loss limit hit: P&L=${self._daily.total_pnl:.2f} "
                f"≤ -${daily_loss_limit:.2f} "
                f"({self.config.max_daily_loss_pct:.1%} of equity). "
                "No new trades until tomorrow."
            )

        # Daily trade count limit
        if self._daily.trades_opened >= self.config.max_trades_per_day:
            return self._veto(
                f"Max daily trades reached: "
                f"{self._daily.trades_opened}/{self.config.max_trades_per_day}"
            )

        # Liquidity check on the option row if provided
        if option is not None:
            liq_veto = self._check_liquidity(option)
            if liq_veto:
                return self._veto(liq_veto)

        # --- Position sizing ---
        contracts = self._size_position(
            risk_budget_dollars=risk_budget,
            max_loss_per_contract=max_loss_per_contract,
        )

        max_loss_total = contracts * max_loss_per_contract

        logger.info(
            "[RiskManager] Approved: contracts=%d max_loss=$%.2f "
            "(budget=$%.2f, %.1f%% used)",
            contracts, max_loss_total, risk_budget,
            (max_loss_total / risk_budget) * 100
        )

        return RiskDecision(
            approved=True,
            position_size_contracts=contracts,
            max_loss_dollars=max_loss_total,
            risk_budget_dollars=risk_budget,
            equity_at_decision=self._equity,
            trades_today=self._daily.trades_opened,
            daily_pnl=self._daily.total_pnl,
        )

    def build_approved_order(
        self,
        legs: list[OrderLeg],
        decision: RiskDecision,
        net_debit_credit: float,
        estimated_fill_price: float,
        hard_stop_price: float,
        profit_target_price: Optional[float] = None,
        strategy_name: str = "",
        underlying: str = "",
        notes: str = "",
    ) -> ApprovedOrder:
        """
        Constructs an ApprovedOrder from a positive RiskDecision.

        Raises RiskVetoError if decision.approved is False — this is
        the final safety gate before the order reaches the broker.
        """
        # Final guard — never build an order from a vetoed decision
        if not decision.approved:
            raise RiskVetoError(
                f"Attempted to build order from vetoed decision: "
                f"{decision.rejection_reason}"
            )

        # Enforce profit target default if not specified
        pt_pct = self.config.default_profit_target_pct
        pt_price = profit_target_price

        order = ApprovedOrder(
            legs=legs,
            net_debit_credit=net_debit_credit,
            estimated_fill_price=estimated_fill_price,
            hard_stop_price=hard_stop_price,
            max_loss_dollars=decision.max_loss_dollars,
            position_size_contracts=decision.position_size_contracts,
            risk_approved=True,
            profit_target_price=pt_price,
            profit_target_pct=pt_pct,
            slippage_budget_pct=self.config.slippage_pct,
            strategy_name=strategy_name,
            underlying=underlying,
            signal_timestamp=datetime.now(tz=timezone.utc),
            notes=notes,
        )

        logger.info(
            "[RiskManager] Order built: %s %s %d contracts "
            "debit/credit=%.2f stop=%.2f max_loss=$%.2f",
            underlying, strategy_name,
            order.position_size_contracts,
            order.net_debit_credit,
            order.hard_stop_price,
            order.max_loss_dollars,
        )

        return order

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _size_position(
        self,
        risk_budget_dollars: float,
        max_loss_per_contract: float,
    ) -> int:
        """
        Compute contract count using risk-budget formula:
          contracts = floor(risk_budget / max_loss_per_contract)
          contracts = clamp(contracts, min_contracts, max_contracts)
        """
        raw = math.floor(risk_budget_dollars / max_loss_per_contract)
        sized = max(self.config.min_contracts, min(raw, self.config.max_contracts))

        if sized != raw:
            logger.debug(
                "[RiskManager] Position size clamped: raw=%d → final=%d "
                "(min=%d, max=%d)",
                raw, sized,
                self.config.min_contracts, self.config.max_contracts,
            )

        return sized

    def _check_liquidity(self, option: EnrichedOptionRow) -> Optional[str]:
        """
        Returns a rejection reason string if the option fails liquidity checks,
        or None if it passes.
        """
        if option.bid is None or option.ask is None:
            return f"missing bid/ask for {option.symbol}"

        if option.open_interest is not None:
            if option.open_interest < self.config.min_open_interest:
                return (
                    f"OI={option.open_interest} < min={self.config.min_open_interest} "
                    f"for {option.symbol}"
                )

        if option.spread_pct is not None:
            if option.spread_pct > self.config.max_spread_pct:
                return (
                    f"spread={option.spread_pct:.1%} > max={self.config.max_spread_pct:.1%} "
                    f"for {option.symbol}"
                )

        return None

    def _veto(self, reason: str) -> RiskDecision:
        logger.warning("[RiskManager] VETO: %s", reason)
        return RiskDecision(
            approved=False,
            position_size_contracts=0,
            max_loss_dollars=0.0,
            risk_budget_dollars=self._equity * self.config.risk_budget_pct,
            rejection_reason=reason,
            equity_at_decision=self._equity,
            trades_today=self._daily.trades_opened,
            daily_pnl=self._daily.total_pnl,
        )


# ---------------------------------------------------------------------------
# Execution guard — final gate before broker
# ---------------------------------------------------------------------------

class ExecutionGuard:
    """
    Last line of defense before an order touches the broker.

    Call ExecutionGuard.check(order) immediately before submitting.
    Raises RiskVetoError on any violation — never returns silently.

    This is intentionally redundant with RiskManager.evaluate().
    Defense in depth: the guard catches anything that slipped through
    (e.g. an order constructed manually or from a different code path).
    """

    @staticmethod
    def check(order: ApprovedOrder) -> None:
        """
        Validates an ApprovedOrder before broker submission.

        Raises
        ------
        RiskVetoError
            If any mandatory safety condition is violated.
        """
        logger.debug(
            "[ExecutionGuard] Checking order: %s %s %d contracts",
            order.underlying, order.strategy_name, order.position_size_contracts
        )

        # Must be explicitly approved by RiskManager
        if not order.risk_approved:
            raise RiskVetoError(
                "Order has risk_approved=False. "
                "Never submit an order that hasn't been reviewed by RiskManager."
            )

        # Hard stop is mandatory — no naked positions
        if order.hard_stop_price is None or order.hard_stop_price <= 0:
            raise RiskVetoError(
                f"hard_stop_price={order.hard_stop_price!r} is missing or invalid. "
                "Every position requires a defined stop-loss. "
                "Naked/unhedged positions are prohibited."
            )

        # Must have at least one contract
        if order.position_size_contracts < 1:
            raise RiskVetoError(
                f"position_size_contracts={order.position_size_contracts} < 1. "
                "Cannot submit a zero-size order."
            )

        # Must have at least one leg
        if not order.legs:
            raise RiskVetoError("Order has no legs defined.")

        # Each leg must have a valid symbol and positive quantity
        for i, leg in enumerate(order.legs):
            if not leg.symbol:
                raise RiskVetoError(f"Leg {i} has no symbol.")
            if leg.quantity < 1:
                raise RiskVetoError(
                    f"Leg {i} ({leg.symbol}) has quantity={leg.quantity} < 1."
                )

        # Max loss must be positive and finite
        if not math.isfinite(order.max_loss_dollars) or order.max_loss_dollars <= 0:
            raise RiskVetoError(
                f"max_loss_dollars={order.max_loss_dollars} is invalid. "
                "Refusing to submit — position may be unhedged."
            )

        logger.info(
            "[ExecutionGuard] ✓ Order passed all checks: %s %s "
            "%d contracts stop=%.2f max_loss=$%.2f",
            order.underlying,
            order.strategy_name,
            order.position_size_contracts,
            order.hard_stop_price,
            order.max_loss_dollars,
        )
