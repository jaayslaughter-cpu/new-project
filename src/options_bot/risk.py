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

# ---------------------------------------------------------------------------
# Vol-adaptive sizing helpers
# ---------------------------------------------------------------------------

def compute_atr_pct(symbol: str, period: int = 14) -> Optional[float]:
    """14-period ATR as % of current price."""
    try:
        import pandas as pd, yfinance as yf
        df = yf.download(symbol, period="30d", interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) < period + 1: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        atr = float(tr.rolling(period).mean().iloc[-1])
        price = float(c.iloc[-1])
        return round(atr/price*100,3) if price>0 and not math.isnan(atr) else None
    except Exception as exc:
        logger.debug("[Risk] compute_atr_pct(%s): %s", symbol, exc); return None

_GARCH_LOOKBACK=60; _GARCH_THRESHOLD=1.5; _GARCH_FLOOR=0.5

def compute_garch_vol_scalar(symbol: str) -> float:
    """GARCH(1,1) size multiplier [0.5,1.0]. Returns 1.0 if arch not installed."""
    if not symbol: return 1.0
    try:
        import pandas as pd, yfinance as yf
        from arch import arch_model
        df = yf.download(symbol, period="90d", interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) < _GARCH_LOOKBACK+5: return 1.0
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        ret = df["Close"].pct_change().dropna()*100
        ret = ret.tail(_GARCH_LOOKBACK)
        if len(ret) < 30: return 1.0
        res = arch_model(ret,vol="GARCH",p=1,q=1,rescale=False).fit(disp="off",show_warning=False)
        fv  = float(res.forecast(horizon=1,reindex=False).variance.iloc[-1,0])
        if fv<=0 or math.isnan(fv): return 1.0
        ratio = (fv**0.5) / max(float(ret.std()),1e-9)
        if ratio <= _GARCH_THRESHOLD: return 1.0
        scalar = max(_GARCH_FLOOR, _GARCH_THRESHOLD/ratio)
        logger.info("[Risk] GARCH %s: ratio=%.2f → ×%.2f", symbol, ratio, scalar)
        return round(scalar,3)
    except ImportError: return 1.0
    except Exception as exc:
        logger.debug("[Risk] compute_garch_vol_scalar(%s): %s", symbol, exc); return 1.0


@dataclass
class RiskConfig:
    """
    All risk parameters in one place.
    Pass this into RiskManager at startup.
    Override per-environment via env vars or YAML config.
    """
    # Per-trade risk limit — UNBREAKABLE
    risk_budget_pct: float = 0.01          # 1% of equity max per trade — capital preservation default

    # Daily halt rules
    max_daily_loss_pct: float = 0.03       # halt if down 3% on the day
    max_trades_per_day: int = 5            # max new positions opened per day

    # Cumulative drawdown halt — bot pauses when equity falls this far from peak
    max_drawdown_pct: float = 0.08            # pause if down 8% from equity peak

    # Position size bounds
    min_contracts: int = 1
    max_contracts: int = 5                # hard cap regardless of sizing math

    # Liquidity requirements (redundant with market_data filter but defense-in-depth)
    min_open_interest: int = 100
    max_spread_pct: float = 0.25           # reject if (ask-bid)/mid > 25%

    # Slippage assumption for realistic fill estimation
    slippage_pct: float = 0.02            # 2% of mid assumed as slippage cost

    # Profit target defaults (can be overridden per strategy)
    default_profit_target_pct: float = 0.50   # close at 50% of max profit

    def validate(self) -> None:
        """Raises DataValidationError if any parameter is out of bounds."""
        if not 0 < self.risk_budget_pct <= 0.10:
            raise DataValidationError(
                "risk_budget_pct",
                f"Must be in (0, 0.10], got {self.risk_budget_pct}. "
                "Capital preservation limit: never risk more than 10% of equity on a single trade. "
                "Recommended: 0.005-0.01 (0.5%-1%)."
            )
        if not 0 < self.max_daily_loss_pct <= 0.10:
            raise DataValidationError(
                "max_daily_loss_pct",
                f"Must be in (0, 0.10], got {self.max_daily_loss_pct}. "
                "Recommended: 0.02-0.03 (2%-3%)."
            )
        if not 0 < self.max_drawdown_pct <= 0.25:
            raise DataValidationError(
                "max_drawdown_pct",
                f"Must be in (0, 0.25], got {self.max_drawdown_pct}. "
                "Recommended: 0.05-0.10 (5%-10%). Bot pauses when this level is hit."
            )
        if self.max_drawdown_pct < self.max_daily_loss_pct:
            raise DataValidationError(
                "max_drawdown_pct",
                f"max_drawdown_pct ({self.max_drawdown_pct:.1%}) must be >= "
                f"max_daily_loss_pct ({self.max_daily_loss_pct:.1%})"
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
        self._peak_equity = equity        # all-time high equity for drawdown tracking
        self.config = config or RiskConfig()
        self.config.validate()
        self._daily = DailyState()

        logger.info(
            "[RiskManager] Initialized: equity=$%.2f, risk_pct=%.1f%%, "
            "max_daily_loss=%.1f%%, max_drawdown=%.1f%%, max_trades=%d",
            equity,
            self.config.risk_budget_pct * 100,
            self.config.max_daily_loss_pct * 100,
            self.config.max_drawdown_pct * 100,
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
        # Ratchet peak upward — never backward
        if new_equity > self._peak_equity:
            self._peak_equity = new_equity
            logger.debug("[RiskManager] New equity peak: $%.2f", self._peak_equity)

    def update_peak_equity(self, equity: float) -> None:
        """
        Explicitly ratchet peak equity upward.
        Call this at EOD or any time you want to confirm the peak is current.
        Peak only moves up — drawdown is always measured from the highest point reached.
        """
        if equity > self._peak_equity:
            self._peak_equity = equity
            logger.info("[RiskManager] Peak equity updated: $%.2f", self._peak_equity)

    def current_drawdown_pct(self) -> float:
        """
        Current drawdown from the all-time peak as a fraction (0.08 = 8%).
        Returns 0.0 if equity is at or above peak.
        """
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._equity) / self._peak_equity)

    def is_drawdown_halted(self) -> bool:
        """Returns True if the max drawdown limit has been breached."""
        return self.current_drawdown_pct() >= self.config.max_drawdown_pct

    def update_equity_after_fill(
        self, max_loss_committed: float, broker=None
    ) -> None:
        """
        AUDIT FIX: Update equity to reflect committed capital after a fill.

        FINDING: "broker.get_equity() is called once at scan time. If multiple
        positions are filled in the same scan, the equity figure is stale for
        the second and third position."

        FIX: After each fill, reduce the working equity by max_loss_committed
        so subsequent positions in the same scan are sized against the remaining
        risk budget, not the pre-scan equity.

        If a broker instance is provided, also fetch the actual equity to
        stay synchronized with Alpaca's clearing.

        LABEL: This is a CONSERVATIVE PROXY for intra-scan equity.
        It deducts committed max-loss (worst case), not actual buying power
        locked by Alpaca. Actual Alpaca buying power changes may differ.
        The deduction is always equal to or greater than actual buying power
        committed, making this a conservative (never too aggressive) estimate.
        """
        self._equity = max(0.0, self._equity - max_loss_committed)
        logger.info(
            "[RiskManager] Equity reduced by committed max-loss $%.2f → working equity $%.2f",
            max_loss_committed, self._equity
        )
        if broker is not None:
            try:
                actual = broker.get_equity()
                if actual > 0:
                    self._equity = actual
                    logger.info("[RiskManager] Equity refreshed from broker: $%.2f", actual)
            except Exception as exc:
                logger.debug("[RiskManager] Broker equity refresh failed (using estimate): %s", exc)

    def warn_correlation(self, open_underlyings: list[str]) -> None:
        """
        AUDIT FIX: Compute effective bet count (Herfindahl N_eff) and log
        correlation warning when multiple correlated positions exist.

        FINDING: "Multiple spreads on SPY, QQQ, and AAPL can behave like one
        large bet in a broad selloff. Missing correlation accounting."

        CALCULATION (from diablotrading/inferno_portfolio_correlation.py):
          Equal-weight assumption (no position-size data here):
            w_i = 1 / N  for all i
          Herfindahl N_eff = 1 / Σ(w_i²) = N  (trivially N for equal weights)

          For non-equal weights (when max_loss data is available):
            w_i = max_loss_i / Σ(max_loss_j)
            N_eff = 1 / Σ(w_i²)

          Dalio Holy Grail (equal-sigma, pairwise rho ρ):
            σ²_portfolio / σ²_individual = 1/N + (N-1)/N × ρ
          At ρ = 0.7 (typical SPY-correlated names):
            N=3: portfolio variance = 3x individual variance × (1/3 + 2/3×0.7) = 3×0.8 = 2.4x
            Effective diversification is lost.

        LABEL: This is a QUALITATIVE structural overlap warning.
        Correlation coefficients are not computed from actual return data here —
        that requires historical return correlation (see stress_testing.py for
        scenario-based approach). The group membership below is based on
        structural overlap (same index, same sector), not measured correlation.
        """
        import math as _math

        n = len(open_underlyings)
        if n == 0:
            return

        # Equal-weight Herfindahl — degenerate but establishes the framework
        # When all weights are equal: N_eff = N
        equal_w = 1.0 / n
        n_eff_equal = round(1.0 / (n * equal_w ** 2), 2)  # = N for equal weights

        # Group overlap detection
        spy_group  = {"SPY", "QQQ", "IWM", "DIA", "MDY"}
        tech_group = {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD"}
        fin_group  = {"JPM", "BAC", "GS", "MS", "WFC", "C", "V", "MA"}

        open_set    = set(s.upper() for s in open_underlyings)
        spy_overlap = open_set & spy_group
        tech_overlap = open_set & tech_group
        fin_overlap  = open_set & fin_group

        # Dalio estimate at assumed rho=0.7 for same-group positions
        # σ²_P / σ²_i = 1/N + (N-1)/N * rho
        rho = 0.70
        correlated_n = len(spy_overlap) + len(tech_overlap) + len(fin_overlap)
        if correlated_n >= 2:
            variance_ratio = (1 / correlated_n) + ((correlated_n - 1) / correlated_n) * rho
            # Individual position seems to be 1 risk unit; portfolio has variance_ratio × risk
            effective_risk_mult = round(_math.sqrt(variance_ratio * correlated_n), 2)
            logger.warning(
                "[RiskManager] CORRELATION WARNING: %d positions in correlated groups "
                "(spy=%d tech=%d fins=%d). "
                "At assumed rho=%.1f: portfolio vol ~%.1fx single position. "
                "N_eff (equal-weight)=%.1f. "
                "LABEL: rho is ASSUMED (0.70 for same-sector), not measured. "
                "See stress_testing.py for scenario-based correlation impact.",
                n, len(spy_overlap), len(tech_overlap), len(fin_overlap),
                rho, effective_risk_mult, n_eff_equal,
            )
        elif n >= 4:
            logger.info(
                "[RiskManager] Correlation check: %d open positions, "
                "N_eff=%.1f (equal-weight). Groups: spy=%d tech=%d fins=%d. "
                "No concentrated overlap detected.",
                n, n_eff_equal, len(spy_overlap), len(tech_overlap), len(fin_overlap),
            )

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

        # Cumulative drawdown halt — highest priority after hard stop / max loss checks
        drawdown = self.current_drawdown_pct()
        if drawdown >= self.config.max_drawdown_pct:
            return self._veto(
                f"MAX DRAWDOWN HALT: account is down {drawdown:.1%} from peak "
                f"(equity=${self._equity:,.2f}, peak=${self._peak_equity:,.2f}, "
                f"limit={self.config.max_drawdown_pct:.1%}). "
                "Bot paused. Review performance and resume manually."
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

        # --- Position sizing (GARCH vol scalar applied inside) ---
        self._current_underlying = getattr(option, "underlying", "") or getattr(option, "symbol", "")
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
        garch = compute_garch_vol_scalar(getattr(self,"_current_underlying",""))
        adj   = risk_budget_dollars * garch
        raw   = math.floor(adj / max_loss_per_contract)
        sized = max(self.config.min_contracts, min(raw, self.config.max_contracts))
        if garch < 1.0:
            logger.info("[Risk] GARCH: $%.2f→$%.2f (×%.2f) raw=%d sized=%d",
                        risk_budget_dollars, adj, garch, raw, sized)
        elif sized != raw:
            logger.debug("[Risk] Clamped: raw=%d→%d (min=%d max=%d)",
                         raw, sized, self.config.min_contracts, self.config.max_contracts)
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
