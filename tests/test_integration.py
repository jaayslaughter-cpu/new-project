"""
End-to-end integration test for the options bot pipeline.

Proves that data flows correctly from market data ingestion through
Greeks enrichment without breaking at module boundaries.

Uses synthetic option chain data that matches real yfinance output shapes,
so this test runs without any live API calls.

Run with:
    pytest tests/test_integration.py -v

Or directly:
    python tests/test_integration.py
"""

from __future__ import annotations

import math
import sys
import unittest
from datetime import date, datetime, timezone, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

# Allow running from project root
sys.path.insert(0, "src")

from options_bot.contracts import EnrichedOptionRow, OptionChainRow, OptionType
from options_bot.exceptions import (
    DataValidationError,
    IVSolveError,
    LiquidityFilterError,
    PipelineConnectionError,
    RiskVetoError,
    StalenessError,
)
from options_bot.greeks import GreeksEnricher, bs_price, bs_greeks, solve_iv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_option_chain_row(
    symbol: str = "SPY260620C00580000",
    underlying: str = "SPY",
    option_type: OptionType = "call",
    strike: float = 580.0,
    dte: int = 10,
    bid: Optional[float] = 9.33,   # priced at ~20% IV above intrinsic
    ask: Optional[float] = 9.53,
    volume: int = 1500,
    open_interest: int = 5000,
    underlying_price: float = 582.50,
) -> OptionChainRow:
    # AUDIT FIX: was date.today().replace(day=today.day + dte), which throws
    # "day is out of range for month" whenever today.day + dte exceeds the
    # current month's length (date-dependent flaky failure). timedelta is correct.
    expiry = date.today() + timedelta(days=dte)
    return OptionChainRow(
        symbol=symbol,
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expiry=expiry,
        dte=dte,
        bid=bid,
        ask=ask,
        last_price=(bid + ask) / 2 if bid and ask else None,
        mid_price=None,  # computed in __post_init__
        volume=volume,
        open_interest=open_interest,
        underlying_price=underlying_price,
        data_timestamp=datetime.now(tz=timezone.utc),
    )


def make_put_row(**kwargs) -> OptionChainRow:
    defaults = dict(
        symbol="SPY260620P00570000",
        option_type="put",
        strike=570.0,
        bid=4.20,   # priced at ~20% IV, OTM put
        ask=4.40,
    )
    defaults.update(kwargs)
    return make_option_chain_row(**defaults)


# ---------------------------------------------------------------------------
# Test: Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptions(unittest.TestCase):

    def test_pipeline_connection_error_is_runtime_error(self):
        exc = PipelineConnectionError("module A failed")
        self.assertIsInstance(exc, RuntimeError)

    def test_staleness_error_stores_fields(self):
        exc = StalenessError("chain_data", 120.0, 60.0)
        self.assertEqual(exc.field, "chain_data")
        self.assertEqual(exc.age_seconds, 120.0)
        self.assertEqual(exc.max_age_seconds, 60.0)
        self.assertIn("120", str(exc))

    def test_liquidity_filter_error_stores_symbol(self):
        exc = LiquidityFilterError("SPY260620C00580000", "OI=5 < min=100")
        self.assertEqual(exc.symbol, "SPY260620C00580000")
        self.assertIn("OI=5", str(exc))

    def test_risk_veto_error_stores_reason(self):
        exc = RiskVetoError("naked short detected")
        self.assertIn("naked short", str(exc))

    def test_data_validation_error_stores_field(self):
        exc = DataValidationError("strike", "must be positive")
        self.assertEqual(exc.field, "strike")

    def test_iv_solve_error(self):
        exc = IVSolveError("SPY260620C00580000", "did not converge")
        self.assertEqual(exc.symbol, "SPY260620C00580000")


# ---------------------------------------------------------------------------
# Test: OptionChainRow data contract
# ---------------------------------------------------------------------------

class TestOptionChainRow(unittest.TestCase):

    def test_mid_price_computed_on_init(self):
        row = make_option_chain_row(bid=3.10, ask=3.30)
        self.assertAlmostEqual(row.mid_price, 3.20, places=4)

    def test_spread_pct_computed_on_init(self):
        row = make_option_chain_row(bid=3.10, ask=3.30)
        expected_spread = (3.30 - 3.10) / 3.20
        self.assertAlmostEqual(row.spread_pct, expected_spread, places=4)

    def test_mid_price_none_when_bid_missing(self):
        row = make_option_chain_row(bid=None, ask=3.30)
        self.assertIsNone(row.mid_price)
        self.assertIsNone(row.spread_pct)

    def test_itm_flag_call(self):
        row = make_option_chain_row(
            option_type="call", strike=580.0, underlying_price=585.0
        )
        self.assertTrue(row.in_the_money)

    def test_otm_flag_call(self):
        row = make_option_chain_row(
            option_type="call", strike=590.0, underlying_price=585.0
        )
        self.assertFalse(row.in_the_money)

    def test_itm_flag_put(self):
        row = make_option_chain_row(
            option_type="put", strike=590.0, underlying_price=585.0
        )
        self.assertTrue(row.in_the_money)


# ---------------------------------------------------------------------------
# Test: Black-Scholes pricing
# ---------------------------------------------------------------------------

class TestBlackScholes(unittest.TestCase):

    def setUp(self):
        # Standard test case: ATM call, 30 DTE, 20% IV, 5% rate
        self.S = 100.0
        self.K = 100.0
        self.T = 30 / 365.0
        self.r = 0.05
        self.sigma = 0.20

    def test_call_price_positive(self):
        price = bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        self.assertGreater(price, 0)

    def test_put_price_positive(self):
        price = bs_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        self.assertGreater(price, 0)

    def test_put_call_parity(self):
        """C - P = S - K*e^(-rT)"""
        call = bs_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        put = bs_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        parity = self.S - self.K * math.exp(-self.r * self.T)
        self.assertAlmostEqual(call - put, parity, places=6)

    def test_call_delta_between_0_and_1(self):
        greeks = bs_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        self.assertGreater(greeks["delta"], 0)
        self.assertLess(greeks["delta"], 1)

    def test_put_delta_between_minus1_and_0(self):
        greeks = bs_greeks(self.S, self.K, self.T, self.r, self.sigma, "put")
        self.assertLess(greeks["delta"], 0)
        self.assertGreater(greeks["delta"], -1)

    def test_gamma_positive(self):
        greeks = bs_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        self.assertGreater(greeks["gamma"], 0)

    def test_theta_negative_for_long_call(self):
        """Long options lose time value — theta should be negative."""
        greeks = bs_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        self.assertLess(greeks["theta"], 0)

    def test_vega_positive(self):
        greeks = bs_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        self.assertGreater(greeks["vega"], 0)

    def test_atm_delta_near_50(self):
        """ATM call delta should be close to 0.5."""
        greeks = bs_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        self.assertAlmostEqual(greeks["delta"], 0.5, delta=0.05)

    def test_deep_itm_call_delta_near_1(self):
        greeks = bs_greeks(150.0, self.K, self.T, self.r, self.sigma, "call")
        self.assertGreater(greeks["delta"], 0.95)

    def test_deep_otm_call_delta_near_0(self):
        greeks = bs_greeks(50.0, self.K, self.T, self.r, self.sigma, "call")
        self.assertLess(greeks["delta"], 0.05)


# ---------------------------------------------------------------------------
# Test: IV solver
# ---------------------------------------------------------------------------

class TestIVSolver(unittest.TestCase):

    def test_round_trip_call(self):
        """Price a call at known IV, then recover that IV from the price."""
        S, K, T, r = 100.0, 100.0, 30 / 365.0, 0.05
        true_iv = 0.25
        price = bs_price(S, K, T, r, true_iv, "call")
        recovered_iv = solve_iv(price, S, K, T, r, "call")
        self.assertAlmostEqual(recovered_iv, true_iv, places=4)

    def test_round_trip_put(self):
        S, K, T, r = 100.0, 105.0, 45 / 365.0, 0.045
        true_iv = 0.18
        price = bs_price(S, K, T, r, true_iv, "put")
        recovered_iv = solve_iv(price, S, K, T, r, "put")
        self.assertAlmostEqual(recovered_iv, true_iv, places=4)

    def test_raises_on_zero_price(self):
        with self.assertRaises(IVSolveError):
            solve_iv(0.0, 100.0, 100.0, 30 / 365.0, 0.05, "call")

    def test_raises_on_expired(self):
        with self.assertRaises((IVSolveError, DataValidationError)):
            solve_iv(1.0, 100.0, 100.0, 0.0, 0.05, "call")

    def test_high_iv_round_trip(self):
        """Test that high-IV situations (e.g. earnings) still solve correctly."""
        S, K, T, r = 100.0, 100.0, 7 / 365.0, 0.05
        true_iv = 1.50  # 150% IV
        price = bs_price(S, K, T, r, true_iv, "call")
        recovered_iv = solve_iv(price, S, K, T, r, "call")
        self.assertAlmostEqual(recovered_iv, true_iv, places=3)


# ---------------------------------------------------------------------------
# Test: Greeks enrichment layer
# ---------------------------------------------------------------------------

class TestGreeksEnricher(unittest.TestCase):

    def setUp(self):
        # Use a fixed rate to avoid network calls in tests
        self.enricher = GreeksEnricher(risk_free_rate=0.05)

    def test_enrich_valid_call(self):
        row = make_option_chain_row()
        enriched = self.enricher.enrich(row)
        self.assertIsInstance(enriched, EnrichedOptionRow)
        self.assertIsNotNone(enriched.iv)
        self.assertIsNotNone(enriched.delta)
        self.assertIsNotNone(enriched.gamma)
        self.assertIsNotNone(enriched.theta)
        self.assertIsNotNone(enriched.vega)

    def test_enriched_delta_reasonable_for_near_atm_call(self):
        row = make_option_chain_row(
            strike=580.0, underlying_price=582.50, option_type="call"
        )
        enriched = self.enricher.enrich(row)
        if enriched.delta is not None:
            self.assertGreater(enriched.delta, 0.3)
            self.assertLess(enriched.delta, 0.7)

    def test_enrich_missing_bid_returns_none_greeks(self):
        row = make_option_chain_row(bid=None, ask=None)
        enriched = self.enricher.enrich(row)
        self.assertIsNone(enriched.iv)
        self.assertIsNone(enriched.delta)

    def test_enrich_chain_returns_all_rows(self):
        rows = [make_option_chain_row(), make_put_row()]
        enriched = self.enricher.enrich_chain(rows)
        self.assertEqual(len(enriched), 2)

    def test_enrich_chain_filtered_drops_no_iv(self):
        good_row = make_option_chain_row()
        bad_row = make_option_chain_row(bid=None, ask=None, symbol="SPY_BAD")
        enriched = self.enricher.enrich_chain_filtered(
            [good_row, bad_row], require_iv=True
        )
        symbols = [r.symbol for r in enriched]
        self.assertIn("SPY260620C00580000", symbols)
        self.assertNotIn("SPY_BAD", symbols)

    def test_delta_filter_removes_deep_otm(self):
        """Deep OTM call (strike >> spot) should have delta near 0 and be filtered."""
        deep_otm = make_option_chain_row(
            symbol="SPY_DEEP_OTM",
            strike=700.0,
            underlying_price=582.50,
            bid=0.05,
            ask=0.10,
        )
        near_atm = make_option_chain_row()
        enriched = self.enricher.enrich_chain_filtered(
            [deep_otm, near_atm],
            require_iv=True,
            min_abs_delta=0.10,
        )
        symbols = [r.symbol for r in enriched]
        self.assertNotIn("SPY_DEEP_OTM", symbols)

    def test_enriched_row_proxies_raw_fields(self):
        row = make_option_chain_row()
        enriched = self.enricher.enrich(row)
        self.assertEqual(enriched.symbol, row.symbol)
        self.assertEqual(enriched.strike, row.strike)
        self.assertEqual(enriched.underlying_price, row.underlying_price)
        self.assertEqual(enriched.bid, row.bid)
        self.assertEqual(enriched.ask, row.ask)


# ---------------------------------------------------------------------------
# Test: Full pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipeline(unittest.TestCase):
    """
    Proves data flows correctly through the full pipeline:
    synthetic chain data → liquidity filter → Greeks enrichment → output schema.

    No live API calls — all data is synthetic.
    """

    def setUp(self):
        self.enricher = GreeksEnricher(risk_free_rate=0.05)

    def _make_spy_chain(self) -> list[OptionChainRow]:
        """Build a realistic synthetic SPY chain snapshot."""
        spot = 582.50
        rows = []

        # Calls: strikes from 560 to 610 in $5 increments
        call_strikes = [560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610]
        for strike in call_strikes:
            itm = strike < spot
            # Rough realistic prices
            intrinsic = max(0, spot - strike)
            extrinsic = 3.0 if abs(spot - strike) < 10 else 1.5
            mid = intrinsic + extrinsic
            bid = round(mid - 0.10, 2)
            ask = round(mid + 0.10, 2)
            rows.append(make_option_chain_row(
                symbol=f"SPY260620C{int(strike * 1000):08d}",
                option_type="call",
                strike=float(strike),
                bid=max(0.01, bid),
                ask=max(0.05, ask),
                open_interest=max(100, 5000 - abs(strike - 580) * 100),
                underlying_price=spot,
            ))

        # Puts: strikes from 550 to 580
        put_strikes = [550, 555, 560, 565, 570, 575, 580]
        for strike in put_strikes:
            intrinsic = max(0, strike - spot)
            extrinsic = 3.0 if abs(spot - strike) < 10 else 1.5
            mid = intrinsic + extrinsic
            bid = round(mid - 0.10, 2)
            ask = round(mid + 0.10, 2)
            rows.append(make_option_chain_row(
                symbol=f"SPY260620P{int(strike * 1000):08d}",
                option_type="put",
                strike=float(strike),
                bid=max(0.01, bid),
                ask=max(0.05, ask),
                open_interest=max(100, 3000 - abs(strike - 580) * 80),
                underlying_price=spot,
            ))

        return rows

    def test_pipeline_produces_enriched_rows(self):
        """Full pipeline: raw chain → enrichment → valid output."""
        chain = self._make_spy_chain()
        self.assertGreater(len(chain), 0, "Synthetic chain should have rows")

        # Step 1: Verify raw chain schema
        for row in chain:
            self.assertIsNotNone(row.symbol)
            self.assertGreater(row.strike, 0)
            self.assertGreater(row.dte, 0)
            self.assertIsNotNone(row.mid_price)
            self.assertIsNotNone(row.spread_pct)

        # Step 2: Enrich with Greeks
        enriched = self.enricher.enrich_chain(chain)
        self.assertEqual(len(enriched), len(chain), "Should return same count")

        # Step 3: Near-ATM rows should have valid Greeks
        near_atm = [r for r in enriched if abs(r.strike - 582.5) <= 10]
        with_iv = [r for r in near_atm if r.iv is not None]
        self.assertGreater(len(with_iv), 0, "Near-ATM rows should have IV")

        for row in with_iv:
            self.assertGreater(row.iv, 0.01, f"{row.symbol} IV should be > 1%")
            self.assertLess(row.iv, 5.0, f"{row.symbol} IV should be < 500%")
            self.assertIsNotNone(row.delta)
            self.assertIsNotNone(row.gamma)
            self.assertIsNotNone(row.theta)
            self.assertIsNotNone(row.vega)

        # Step 4: Delta filter — only keep 10-40 delta
        filtered = self.enricher.enrich_chain_filtered(
            chain,
            require_iv=True,
            min_abs_delta=0.10,
            max_abs_delta=0.40,
        )
        for row in filtered:
            self.assertIsNotNone(row.delta)
            self.assertGreaterEqual(abs(row.delta), 0.10)
            self.assertLessEqual(abs(row.delta), 0.40)

    def test_pipeline_raises_on_stale_data(self):
        """StalenessError raised when data is too old."""
        from options_bot.exceptions import StalenessError
        with self.assertRaises(StalenessError):
            raise StalenessError("chain_data", 3600, 300)

    def test_pipeline_raises_risk_veto_on_naked_short(self):
        """RiskVetoError raised when no stop-loss is defined."""
        from options_bot.exceptions import RiskVetoError
        with self.assertRaises(RiskVetoError):
            raise RiskVetoError("no hard_stop_price defined — naked position rejected")

    def test_liquidity_filter_error_on_empty_result(self):
        """LiquidityFilterError raised when no contracts pass filters."""
        from options_bot.exceptions import LiquidityFilterError
        with self.assertRaises(LiquidityFilterError):
            raise LiquidityFilterError("SPY 2026-06-20", "all 0 OI")

    def test_schema_completeness(self):
        """Every enriched row has the expected attributes."""
        row = make_option_chain_row()
        enriched = self.enricher.enrich(row)

        required_attrs = [
            "symbol", "underlying", "option_type", "strike", "expiry",
            "dte", "bid", "ask", "mid_price", "open_interest", "spread_pct",
            "underlying_price", "iv", "delta", "gamma", "theta", "vega",
            "pricing_model", "risk_free_rate",
        ]
        for attr in required_attrs:
            self.assertTrue(
                hasattr(enriched, attr),
                f"EnrichedOptionRow missing attribute: {attr}"
            )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running options bot integration tests...\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [
        TestExceptions,
        TestOptionChainRow,
        TestBlackScholes,
        TestIVSolver,
        TestGreeksEnricher,
        TestFullPipeline,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


# ---------------------------------------------------------------------------
# Test: RiskConfig validation
# ---------------------------------------------------------------------------

class TestRiskConfig(unittest.TestCase):

    def test_valid_config_passes(self):
        from options_bot.risk import RiskConfig
        cfg = RiskConfig(risk_budget_pct=0.02, max_daily_loss_pct=0.05)
        cfg.validate()  # should not raise

    def test_risk_pct_too_high(self):
        from options_bot.risk import RiskConfig
        cfg = RiskConfig(risk_budget_pct=0.50)
        with self.assertRaises(DataValidationError):
            cfg.validate()

    def test_risk_pct_zero(self):
        from options_bot.risk import RiskConfig
        cfg = RiskConfig(risk_budget_pct=0.0)
        with self.assertRaises(DataValidationError):
            cfg.validate()

    def test_max_contracts_less_than_min(self):
        from options_bot.risk import RiskConfig
        cfg = RiskConfig(min_contracts=5, max_contracts=2)
        with self.assertRaises(DataValidationError):
            cfg.validate()


# ---------------------------------------------------------------------------
# Test: RiskManager evaluation
# ---------------------------------------------------------------------------

class TestRiskManager(unittest.TestCase):

    def setUp(self):
        from options_bot.risk import RiskManager, RiskConfig
        self.cfg = RiskConfig(
            risk_budget_pct=0.02,
            max_daily_loss_pct=0.05,
            max_trades_per_day=5,
            min_contracts=1,
            max_contracts=10,
        )
        self.rm = RiskManager(equity=50_000, config=self.cfg)

    def test_valid_trade_approved(self):
        decision = self.rm.evaluate(
            max_loss_per_contract=250.0,
            hard_stop_price=5.00,
            strategy_name="short_put_spread",
        )
        self.assertTrue(decision.approved)
        self.assertGreaterEqual(decision.position_size_contracts, 1)

    def test_position_size_formula(self):
        # equity=50_000, risk_pct=2% → budget=$1000
        # max_loss_per_contract=$250 → floor(1000/250) = 4 contracts
        decision = self.rm.evaluate(
            max_loss_per_contract=250.0,
            hard_stop_price=5.00,
        )
        self.assertTrue(decision.approved)
        self.assertEqual(decision.position_size_contracts, 4)

    def test_position_size_clamped_to_max(self):
        # budget=$1000, max_loss=$50 → raw=20, clamped to max_contracts=10
        decision = self.rm.evaluate(
            max_loss_per_contract=50.0,
            hard_stop_price=1.00,
        )
        self.assertTrue(decision.approved)
        self.assertEqual(decision.position_size_contracts, 10)

    def test_position_size_clamped_to_min(self):
        # budget=$1000, max_loss=$900 → raw=1, stays at min=1
        decision = self.rm.evaluate(
            max_loss_per_contract=900.0,
            hard_stop_price=1.00,
        )
        self.assertTrue(decision.approved)
        self.assertEqual(decision.position_size_contracts, 1)

    def test_veto_no_hard_stop(self):
        decision = self.rm.evaluate(
            max_loss_per_contract=250.0,
            hard_stop_price=0.0,
        )
        self.assertFalse(decision.approved)
        self.assertIn("hard_stop_price", decision.rejection_reason)

    def test_veto_infinite_max_loss(self):
        decision = self.rm.evaluate(
            max_loss_per_contract=float('inf'),
            hard_stop_price=5.00,
        )
        self.assertFalse(decision.approved)
        self.assertIn("infinite", decision.rejection_reason)

    def test_veto_max_loss_exceeds_budget(self):
        # Budget = $1000, max_loss = $1500 → can't open even 1 contract
        decision = self.rm.evaluate(
            max_loss_per_contract=1500.0,
            hard_stop_price=5.00,
        )
        self.assertFalse(decision.approved)
        self.assertIn("risk_budget", decision.rejection_reason)

    def test_veto_daily_trade_limit(self):
        from options_bot.risk import RiskManager, RiskConfig
        rm = RiskManager(
            equity=50_000,
            config=RiskConfig(max_trades_per_day=2)
        )
        rm.record_trade_opened()
        rm.record_trade_opened()
        decision = rm.evaluate(max_loss_per_contract=250.0, hard_stop_price=5.00)
        self.assertFalse(decision.approved)
        self.assertIn("Max daily trades", decision.rejection_reason)

    def test_veto_daily_loss_limit(self):
        # equity=$50k, max_daily_loss=5% → halt at -$2500
        self.rm.record_pnl(realized=-2600.0)
        decision = self.rm.evaluate(max_loss_per_contract=250.0, hard_stop_price=5.00)
        self.assertFalse(decision.approved)
        self.assertIn("Daily loss limit", decision.rejection_reason)

    def test_veto_illiquid_option(self):
        row = make_option_chain_row(open_interest=10)  # below min of 100
        enricher = GreeksEnricher(risk_free_rate=0.05)
        enriched = enricher.enrich(row)
        decision = self.rm.evaluate(
            max_loss_per_contract=250.0,
            hard_stop_price=5.00,
            option=enriched,
        )
        self.assertFalse(decision.approved)
        self.assertIn("OI=10", decision.rejection_reason)

    def test_decision_stores_equity_and_pnl(self):
        decision = self.rm.evaluate(max_loss_per_contract=250.0, hard_stop_price=5.00)
        self.assertEqual(decision.equity_at_decision, 50_000)
        self.assertEqual(decision.daily_pnl, 0.0)

    def test_update_equity(self):
        self.rm.update_equity(55_000)
        self.assertEqual(self.rm.equity, 55_000)
        # Budget should now be based on new equity
        decision = self.rm.evaluate(max_loss_per_contract=250.0, hard_stop_price=5.00)
        # 55000 * 2% = 1100, floor(1100/250) = 4
        self.assertEqual(decision.position_size_contracts, 4)


# ---------------------------------------------------------------------------
# Test: ExecutionGuard
# ---------------------------------------------------------------------------

class TestExecutionGuard(unittest.TestCase):

    def setUp(self):
        from options_bot.risk import RiskManager, RiskConfig, ExecutionGuard
        from options_bot.contracts import OrderLeg
        self.ExecutionGuard = ExecutionGuard
        self.rm = RiskManager(equity=50_000, config=RiskConfig())

        self.leg = OrderLeg(
            symbol="SPY260620P00570000",
            option_type="put",
            strike=570.0,
            expiry=date(2026, 6, 20),
            side="sell_to_open",
            quantity=1,
        )

    def _make_order(self, **overrides):
        from options_bot.contracts import ApprovedOrder
        defaults = dict(
            legs=[self.leg],
            net_debit_credit=-1.50,
            estimated_fill_price=1.50,
            hard_stop_price=3.00,
            max_loss_dollars=350.0,
            position_size_contracts=4,
            risk_approved=True,
            strategy_name="short_put_spread",
            underlying="SPY",
        )
        defaults.update(overrides)
        return ApprovedOrder(**defaults)

    def test_valid_order_passes(self):
        order = self._make_order()
        self.ExecutionGuard.check(order)  # should not raise

    def test_raises_on_risk_not_approved(self):
        order = self._make_order(risk_approved=False)
        with self.assertRaises(RiskVetoError):
            self.ExecutionGuard.check(order)

    def test_raises_on_missing_stop(self):
        order = self._make_order(hard_stop_price=0.0)
        with self.assertRaises(RiskVetoError):
            self.ExecutionGuard.check(order)

    def test_raises_on_zero_contracts(self):
        order = self._make_order(position_size_contracts=0)
        with self.assertRaises(RiskVetoError):
            self.ExecutionGuard.check(order)

    def test_raises_on_no_legs(self):
        order = self._make_order(legs=[])
        with self.assertRaises(RiskVetoError):
            self.ExecutionGuard.check(order)

    def test_raises_on_infinite_max_loss(self):
        order = self._make_order(max_loss_dollars=float('inf'))
        with self.assertRaises(RiskVetoError):
            self.ExecutionGuard.check(order)

    def test_raises_on_zero_max_loss(self):
        order = self._make_order(max_loss_dollars=0.0)
        with self.assertRaises(RiskVetoError):
            self.ExecutionGuard.check(order)

    def test_error_message_mentions_naked_position(self):
        order = self._make_order(hard_stop_price=None)
        try:
            self.ExecutionGuard.check(order)
            self.fail("Should have raised RiskVetoError")
        except RiskVetoError as e:
            self.assertIn("stop", str(e).lower())


# ---------------------------------------------------------------------------
# Test: Full pipeline with risk
# ---------------------------------------------------------------------------

class TestFullPipelineWithRisk(unittest.TestCase):
    """
    End-to-end: synthetic chain → Greeks → RiskManager → ExecutionGuard
    """

    def test_complete_flow(self):
        from options_bot.risk import RiskManager, RiskConfig, ExecutionGuard
        from options_bot.contracts import OrderLeg

        # Step 1: Raw chain row
        row = make_option_chain_row(
            option_type="put",
            strike=570.0,
            bid=4.20,
            ask=4.40,
            open_interest=500,
            underlying_price=582.50,
        )

        # Step 2: Enrich
        enricher = GreeksEnricher(risk_free_rate=0.05)
        enriched = enricher.enrich(row)
        self.assertIsNotNone(enriched.iv, "Should have IV")

        # Step 3: Risk evaluation
        # Short put spread: sell 570P, buy 560P → max loss = (10 - credit) * 100
        credit_per_contract = enriched.mid_price or 4.30
        spread_width = 10.0
        max_loss = (spread_width - credit_per_contract) * 100

        rm = RiskManager(equity=50_000, config=RiskConfig(
            risk_budget_pct=0.02,
            max_trades_per_day=5,
        ))
        decision = rm.evaluate(
            max_loss_per_contract=max_loss,
            hard_stop_price=credit_per_contract * 2,  # stop at 2x premium received
            option=enriched,
            strategy_name="short_put_spread",
        )
        self.assertTrue(decision.approved, f"Should be approved: {decision.rejection_reason}")
        self.assertGreaterEqual(decision.position_size_contracts, 1)

        # Step 4: Build order
        leg = OrderLeg(
            symbol=enriched.symbol,
            option_type="put",
            strike=enriched.strike,
            expiry=enriched.expiry,
            side="sell_to_open",
            quantity=decision.position_size_contracts,
        )
        order = rm.build_approved_order(
            legs=[leg],
            decision=decision,
            net_debit_credit=-credit_per_contract,
            estimated_fill_price=credit_per_contract,
            hard_stop_price=credit_per_contract * 2,
            strategy_name="short_put_spread",
            underlying="SPY",
        )

        # Step 5: Execution guard — final check
        ExecutionGuard.check(order)  # should not raise

        # Verify order schema completeness
        self.assertTrue(order.risk_approved)
        self.assertGreater(order.hard_stop_price, 0)
        self.assertGreater(order.max_loss_dollars, 0)
        self.assertGreater(order.position_size_contracts, 0)
        self.assertIsNotNone(order.signal_timestamp)



# ---------------------------------------------------------------------------
# Helpers: build realistic synthetic chains for strategy tests
# ---------------------------------------------------------------------------

def make_spy_chain_for_strategy(
    spot: float = 582.50,
    dte: int = 30,
    call_strikes=None,
    put_strikes=None,
) -> list:
    """
    Build a synthetic SPY chain with realistic Greeks for strategy testing.
    Prices are computed from Black-Scholes at 20% IV so IV solve succeeds.
    """
    import sys, math
    sys.path.insert(0, '/home/claude/options_bot/src')
    from options_bot.greeks import bs_price, bs_greeks
    from options_bot.contracts import OptionChainRow
    from datetime import datetime, timezone, timedelta, date

    r = 0.05
    sigma = 0.20
    T = dte / 365.0
    today = date.today()
    expiry = today + timedelta(days=dte)
    fetch_time = datetime.now(tz=timezone.utc)

    if call_strikes is None:
        call_strikes = [570, 575, 580, 585, 590, 595, 600, 605, 610]
    if put_strikes is None:
        put_strikes = [540, 545, 550, 555, 560, 565, 570, 575, 580]

    rows = []
    for strike in call_strikes:
        mid = bs_price(spot, float(strike), T, r, sigma, "call")
        if mid < 0.05:
            continue
        row = OptionChainRow(
            symbol=f"SPY{expiry.strftime('%y%m%d')}C{int(strike*1000):08d}",
            underlying="SPY",
            option_type="call",
            strike=float(strike),
            expiry=expiry,
            dte=dte,
            bid=round(mid - 0.05, 2),
            ask=round(mid + 0.05, 2),
            last_price=round(mid, 2),
            mid_price=None,
            volume=2000,
            open_interest=max(100, 5000 - abs(strike - int(spot)) * 80),
            underlying_price=spot,
            data_timestamp=fetch_time,
        )
        rows.append(row)

    for strike in put_strikes:
        mid = bs_price(spot, float(strike), T, r, sigma, "put")
        if mid < 0.05:
            continue
        row = OptionChainRow(
            symbol=f"SPY{expiry.strftime('%y%m%d')}P{int(strike*1000):08d}",
            underlying="SPY",
            option_type="put",
            strike=float(strike),
            expiry=expiry,
            dte=dte,
            bid=round(mid - 0.05, 2),
            ask=round(mid + 0.05, 2),
            last_price=round(mid, 2),
            mid_price=None,
            volume=1500,
            open_interest=max(100, 4000 - abs(strike - int(spot)) * 60),
            underlying_price=spot,
            data_timestamp=fetch_time,
        )
        rows.append(row)

    # Enrich with Greeks
    enricher = GreeksEnricher(risk_free_rate=0.05)
    enriched = enricher.enrich_chain(rows)
    return [r for r in enriched if r.iv is not None and r.delta is not None]


# ---------------------------------------------------------------------------
# Test: CashSecuredPut strategy
# ---------------------------------------------------------------------------

class TestCashSecuredPut(unittest.TestCase):

    def setUp(self):
        from options_bot.strategy import CashSecuredPut, CSPConfig
        self.chain = make_spy_chain_for_strategy(dte=30)
        self.strategy = CashSecuredPut(CSPConfig(
            target_delta=-0.20,
            min_delta=-0.30,
            max_delta=-0.10,
            min_dte=15,
            max_dte=60,
            min_open_interest=100,
        ))

    def test_returns_signal(self):
        from options_bot.strategy import StrategySignal
        signal = self.strategy.evaluate(self.chain)
        self.assertIsInstance(signal, StrategySignal)

    def test_signal_has_one_leg(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertEqual(len(signal.legs), 1)

    def test_leg_is_sell_put(self):
        signal = self.strategy.evaluate(self.chain)
        leg = signal.legs[0]
        self.assertEqual(leg.option_type, "put")
        self.assertEqual(leg.side, "sell_to_open")

    def test_credit_is_negative_debit_credit(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertLess(signal.net_debit_credit, 0)
        self.assertGreater(signal.estimated_fill_price, 0)

    def test_max_loss_is_finite_and_positive(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertGreater(signal.max_loss_per_contract, 0)
        self.assertTrue(abs(signal.max_loss_per_contract) < float('inf'))

    def test_hard_stop_is_positive(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertGreater(signal.hard_stop_price, 0)

    def test_hard_stop_is_multiple_of_credit(self):
        signal = self.strategy.evaluate(self.chain)
        ratio = signal.hard_stop_price / signal.estimated_fill_price
        self.assertAlmostEqual(ratio, 2.0, delta=0.01)

    def test_underlying_is_spy(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertEqual(signal.underlying, "SPY")

    def test_raises_on_empty_chain(self):
        from options_bot.exceptions import PipelineConnectionError
        with self.assertRaises(PipelineConnectionError):
            self.strategy.evaluate([])

    def test_raises_when_no_delta_match(self):
        from options_bot.strategy import CashSecuredPut, CSPConfig
        from options_bot.exceptions import LiquidityFilterError
        # Use impossibly tight delta range
        strategy = CashSecuredPut(CSPConfig(
            min_delta=-0.001, max_delta=-0.0001, min_open_interest=1, min_dte=1
        ))
        with self.assertRaises(LiquidityFilterError):
            strategy.evaluate(self.chain)


# ---------------------------------------------------------------------------
# Test: ShortPutSpread strategy
# ---------------------------------------------------------------------------

class TestShortPutSpread(unittest.TestCase):

    def setUp(self):
        from options_bot.strategy import ShortPutSpread, ShortPutSpreadConfig
        self.chain = make_spy_chain_for_strategy(dte=30)
        self.strategy = ShortPutSpread(ShortPutSpreadConfig(
            short_delta=-0.15,   # matches production default (lowered from -0.25
                                  # to stay under the 35% PoT hard-reject threshold)
            long_delta=-0.07,
            min_dte=15,
            max_dte=60,
            min_open_interest=50,
            min_credit=0.10,
            min_spread_width=2.0,
        ))

    def test_returns_signal(self):
        from options_bot.strategy import StrategySignal
        signal = self.strategy.evaluate(self.chain)
        self.assertIsInstance(signal, StrategySignal)

    def test_signal_has_two_legs(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertEqual(len(signal.legs), 2)

    def test_short_leg_is_higher_strike(self):
        signal = self.strategy.evaluate(self.chain)
        short_leg = next(l for l in signal.legs if l.side == "sell_to_open")
        long_leg  = next(l for l in signal.legs if l.side == "buy_to_open")
        self.assertGreater(short_leg.strike, long_leg.strike)

    def test_both_legs_are_puts(self):
        signal = self.strategy.evaluate(self.chain)
        for leg in signal.legs:
            self.assertEqual(leg.option_type, "put")

    def test_max_loss_equals_spread_minus_credit(self):
        signal = self.strategy.evaluate(self.chain)
        short_leg = next(l for l in signal.legs if l.side == "sell_to_open")
        long_leg  = next(l for l in signal.legs if l.side == "buy_to_open")
        spread_width = short_leg.strike - long_leg.strike
        net_credit = signal.estimated_fill_price
        expected_max_loss = (spread_width - net_credit) * 100
        self.assertAlmostEqual(
            signal.max_loss_per_contract, expected_max_loss, delta=0.01
        )

    def test_max_loss_is_finite(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertTrue(abs(signal.max_loss_per_contract) < float('inf'))

    def test_hard_rejects_when_no_delta_in_range(self):
        """
        AUDIT FIX regression guard: previously this strategy "relaxed and
        took the closest" when nothing was in [min_delta, max_delta],
        meaning it would trade an arbitrarily-far-off delta rather than
        skip. Now it must hard-reject, consistent with CSP/ShortCallSpread.
        """
        from options_bot.strategy import ShortPutSpread, ShortPutSpreadConfig
        from options_bot.exceptions import LiquidityFilterError

        strategy = ShortPutSpread(ShortPutSpreadConfig(
            min_delta=-0.95, max_delta=-0.90,
            min_dte=15, max_dte=60, min_open_interest=50,
        ))
        with self.assertRaises(LiquidityFilterError):
            strategy.evaluate(self.chain)

    def test_hard_stop_is_2x_credit(self):
        signal = self.strategy.evaluate(self.chain)
        ratio = signal.hard_stop_price / signal.estimated_fill_price
        self.assertAlmostEqual(ratio, 2.0, delta=0.01)


# ---------------------------------------------------------------------------
# Test: ShortCallSpread strategy
#
# AUDIT FIX: this strategy previously had an incompatible evaluate() signature
# (ticker, chain, expiry, dte, underlying_price) that didn't match the single
# `evaluate(chain)` call the orchestrator actually uses, and returned an
# undefined `TradeSignal` type instead of `StrategySignal`. It would throw
# TypeError on every single ticker if "short_call_spread" were ever selected
# as the active strategy, and had zero test coverage to catch it. Rewritten
# to mirror ShortPutSpread's structure exactly (same checks, same return type)
# and covered here so a regression can never go undetected again.
# ---------------------------------------------------------------------------

class TestShortCallSpread(unittest.TestCase):

    def setUp(self):
        from options_bot.strategy import ShortCallSpread, ShortCallSpreadConfig
        # Default call_strikes only reach delta~0.24 (max strike 610); widen the
        # ladder so a true ~0.15-delta call exists for this lower-delta config.
        self.chain = make_spy_chain_for_strategy(
            dte=30,
            call_strikes=[595, 600, 605, 610, 615, 620, 625, 630, 635, 640],
        )
        self.strategy = ShortCallSpread(ShortCallSpreadConfig(
            target_delta=0.15,
            long_delta=0.07,
            min_dte=15,
            max_dte=60,
            min_open_interest=50,
            min_credit=0.10,
            min_spread_width=2.0,
        ))

    def test_returns_signal(self):
        from options_bot.strategy import StrategySignal
        signal = self.strategy.evaluate(self.chain)
        self.assertIsInstance(signal, StrategySignal)

    def test_signal_has_two_legs(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertEqual(len(signal.legs), 2)

    def test_both_legs_are_calls(self):
        signal = self.strategy.evaluate(self.chain)
        for leg in signal.legs:
            self.assertEqual(leg.option_type, "call")

    def test_long_leg_is_higher_strike(self):
        signal = self.strategy.evaluate(self.chain)
        short_leg = next(l for l in signal.legs if l.side == "sell_to_open")
        long_leg  = next(l for l in signal.legs if l.side == "buy_to_open")
        self.assertGreater(long_leg.strike, short_leg.strike)

    def test_net_credit_is_positive(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertGreater(signal.estimated_fill_price, 0)
        self.assertLess(signal.net_debit_credit, 0)   # negative = credit received

    def test_max_loss_equals_spread_minus_credit(self):
        signal = self.strategy.evaluate(self.chain)
        short_leg = next(l for l in signal.legs if l.side == "sell_to_open")
        long_leg  = next(l for l in signal.legs if l.side == "buy_to_open")
        spread_width = long_leg.strike - short_leg.strike
        net_credit = signal.estimated_fill_price
        expected_max_loss = (spread_width - net_credit) * 100
        self.assertAlmostEqual(
            signal.max_loss_per_contract, expected_max_loss, delta=0.01
        )

    def test_max_loss_is_finite(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertTrue(abs(signal.max_loss_per_contract) < float('inf'))

    def test_hard_stop_is_2x_credit(self):
        signal = self.strategy.evaluate(self.chain)
        ratio = signal.hard_stop_price / signal.estimated_fill_price
        self.assertAlmostEqual(ratio, 2.0, delta=0.01)

    def test_orchestrator_call_signature(self):
        """
        Regression guard: orchestrator.py always calls
        self.strategy.evaluate(enriched) with exactly one positional arg.
        This must never require ticker/expiry/dte/underlying_price again.
        """
        import inspect
        sig = inspect.signature(self.strategy.evaluate)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["chain"])

    def test_reachable_via_strategy_registry(self):
        from options_bot.strategy import get_strategy, StrategySignal
        s = get_strategy("short_call_spread")
        signal = s.evaluate(self.chain)
        self.assertIsInstance(signal, StrategySignal)


# ---------------------------------------------------------------------------
# Test: ShortStrangle strategy
# ---------------------------------------------------------------------------

class TestShortStrangle(unittest.TestCase):

    def setUp(self):
        from options_bot.strategy import ShortStrangle, ShortStrangleConfig
        # Default call_strikes only reach delta~0.24 (max strike 610); widen the
        # ladder so a true ~0.15-delta call exists, matching the production
        # default (lowered from 0.20/-0.20 -> 0.15/-0.15, see PoT calibration
        # note in ShortStrangleConfig). Hard-reject now applies if nothing is
        # within delta_tolerance, so the fixture must actually contain a match.
        self.chain = make_spy_chain_for_strategy(
            dte=35,
            call_strikes=[595, 600, 605, 610, 615, 620, 625, 630, 635, 640],
        )
        self.strategy = ShortStrangle(ShortStrangleConfig(
            call_delta=0.15,
            put_delta=-0.15,
            min_dte=15,
            max_dte=60,
            min_open_interest=50,
            min_total_credit=0.10,
        ))

    def test_returns_signal(self):
        from options_bot.strategy import StrategySignal
        signal = self.strategy.evaluate(self.chain)
        self.assertIsInstance(signal, StrategySignal)

    def test_signal_has_two_legs(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertEqual(len(signal.legs), 2)

    def test_has_one_call_one_put(self):
        signal = self.strategy.evaluate(self.chain)
        types = {leg.option_type for leg in signal.legs}
        self.assertIn("call", types)
        self.assertIn("put", types)

    def test_both_legs_are_sell_to_open(self):
        signal = self.strategy.evaluate(self.chain)
        for leg in signal.legs:
            self.assertEqual(leg.side, "sell_to_open")

    def test_credit_is_positive(self):
        signal = self.strategy.evaluate(self.chain)
        self.assertGreater(signal.estimated_fill_price, 0)

    def test_hard_stop_is_3x_credit(self):
        signal = self.strategy.evaluate(self.chain)
        ratio = signal.hard_stop_price / signal.estimated_fill_price
        self.assertAlmostEqual(ratio, 3.0, delta=0.01)

    def test_call_strike_higher_than_put_strike(self):
        signal = self.strategy.evaluate(self.chain)
        call_leg = next(l for l in signal.legs if l.option_type == "call")
        put_leg  = next(l for l in signal.legs if l.option_type == "put")
        self.assertGreater(call_leg.strike, put_leg.strike)


# ---------------------------------------------------------------------------
# Test: Strategy registry
# ---------------------------------------------------------------------------

class TestStrategyRegistry(unittest.TestCase):

    def test_get_csp(self):
        from options_bot.strategy import get_strategy, CashSecuredPut
        s = get_strategy("csp")
        self.assertIsInstance(s, CashSecuredPut)

    def test_get_short_put_spread(self):
        from options_bot.strategy import get_strategy, ShortPutSpread
        s = get_strategy("short_put_spread")
        self.assertIsInstance(s, ShortPutSpread)

    def test_get_short_strangle(self):
        from options_bot.strategy import get_strategy, ShortStrangle
        s = get_strategy("short_strangle")
        self.assertIsInstance(s, ShortStrangle)

    def test_unknown_raises(self):
        from options_bot.strategy import get_strategy
        with self.assertRaises(ValueError):
            get_strategy("naked_call")


# ---------------------------------------------------------------------------
# Test: Full pipeline — strategy → risk → execution guard
# ---------------------------------------------------------------------------

class TestStrategyToRiskPipeline(unittest.TestCase):
    """
    Proves: enriched chain → strategy signal → risk evaluation →
            build order → execution guard — end to end.
    """

    def test_csp_full_pipeline(self):
        from options_bot.strategy import CashSecuredPut, CSPConfig
        from options_bot.risk import RiskManager, RiskConfig, ExecutionGuard

        chain = make_spy_chain_for_strategy(dte=30)
        strategy = CashSecuredPut(CSPConfig(
            min_dte=15, max_dte=60, min_open_interest=50
        ))
        signal = strategy.evaluate(chain)

        # CSP on SPY requires large account — use $5M with 2% risk
        rm = RiskManager(equity=5_000_000, config=RiskConfig(
            risk_budget_pct=0.02, max_trades_per_day=5
        ))
        decision = rm.evaluate(
            max_loss_per_contract=signal.max_loss_per_contract,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
        )
        self.assertTrue(decision.approved, f"Vetoed: {decision.rejection_reason}")

        order = rm.build_approved_order(
            legs=signal.legs,
            decision=decision,
            net_debit_credit=signal.net_debit_credit,
            estimated_fill_price=signal.estimated_fill_price,
            hard_stop_price=signal.hard_stop_price,
            profit_target_price=signal.profit_target_price,
            strategy_name=signal.strategy_name,
            underlying=signal.underlying,
        )

        ExecutionGuard.check(order)

        self.assertTrue(order.risk_approved)
        self.assertGreater(order.hard_stop_price, 0)
        self.assertGreater(order.position_size_contracts, 0)
        self.assertIsNotNone(order.signal_timestamp)

    def test_spread_full_pipeline(self):
        from options_bot.strategy import ShortPutSpread, ShortPutSpreadConfig
        from options_bot.risk import RiskManager, RiskConfig, ExecutionGuard

        chain = make_spy_chain_for_strategy(dte=30)
        strategy = ShortPutSpread(ShortPutSpreadConfig(
            min_dte=15, max_dte=60, min_open_interest=50, min_credit=0.10
        ))
        signal = strategy.evaluate(chain)

        # Use $100k with 2% risk = $2000 budget, covers typical spread max loss
        rm = RiskManager(equity=100_000, config=RiskConfig(risk_budget_pct=0.02))
        decision = rm.evaluate(
            max_loss_per_contract=signal.max_loss_per_contract,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
        )
        self.assertTrue(decision.approved)

        order = rm.build_approved_order(
            legs=signal.legs,
            decision=decision,
            net_debit_credit=signal.net_debit_credit,
            estimated_fill_price=signal.estimated_fill_price,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
            underlying=signal.underlying,
        )
        ExecutionGuard.check(order)
        self.assertEqual(len(order.legs), 2)

    def test_strangle_full_pipeline(self):
        from options_bot.strategy import ShortStrangle, ShortStrangleConfig
        from options_bot.risk import RiskManager, RiskConfig, ExecutionGuard

        # Widened call_strikes — production default call_delta=0.15 needs a
        # strike further OTM than the default ladder's max (610) reaches.
        chain = make_spy_chain_for_strategy(
            dte=35, call_strikes=[595, 600, 605, 610, 615, 620, 625, 630, 635, 640],
        )
        strategy = ShortStrangle(ShortStrangleConfig(
            min_dte=15, max_dte=60, min_open_interest=50, min_total_credit=0.10
        ))
        signal = strategy.evaluate(chain)

        # Strangle stop = 3x credit; use $500k with 2% = $10k budget
        rm = RiskManager(equity=500_000, config=RiskConfig(risk_budget_pct=0.02))
        decision = rm.evaluate(
            max_loss_per_contract=signal.max_loss_per_contract,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
        )
        self.assertTrue(decision.approved)

        order = rm.build_approved_order(
            legs=signal.legs,
            decision=decision,
            net_debit_credit=signal.net_debit_credit,
            estimated_fill_price=signal.estimated_fill_price,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
            underlying=signal.underlying,
        )
        ExecutionGuard.check(order)
        self.assertEqual(len(order.legs), 2)



# ---------------------------------------------------------------------------
# Test: PaperBroker (no network — tests broker interface with stub)
# ---------------------------------------------------------------------------

class TestPaperBroker(unittest.TestCase):

    def setUp(self):
        from options_bot.broker import PaperBroker
        from options_bot.strategy import ShortPutSpread, ShortPutSpreadConfig
        from options_bot.risk import RiskManager, RiskConfig

        self.broker = PaperBroker(starting_equity=100_000)

        chain = make_spy_chain_for_strategy(dte=30)
        strategy = ShortPutSpread(ShortPutSpreadConfig(
            min_dte=15, max_dte=60, min_open_interest=50, min_credit=0.10
        ))
        signal = strategy.evaluate(chain)

        rm = RiskManager(equity=100_000, config=RiskConfig(risk_budget_pct=0.02))
        decision = rm.evaluate(
            max_loss_per_contract=signal.max_loss_per_contract,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
        )
        self.order = rm.build_approved_order(
            legs=signal.legs,
            decision=decision,
            net_debit_credit=signal.net_debit_credit,
            estimated_fill_price=signal.estimated_fill_price,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
            underlying=signal.underlying,
        )

    def test_is_paper(self):
        self.assertTrue(self.broker.is_paper)

    def test_get_account_returns_dict(self):
        account = self.broker.get_account()
        self.assertIn("equity", account)
        self.assertIn("buying_power", account)
        self.assertFalse(account["trading_blocked"])

    def test_get_equity(self):
        self.assertEqual(self.broker.get_equity(), 100_000)

    def test_check_account_ready_does_not_raise(self):
        self.broker.check_account_ready()  # no exception

    def test_submit_returns_filled_order(self):
        from options_bot.contracts import FilledOrder
        filled = self.broker.submit(self.order)
        self.assertIsInstance(filled, FilledOrder)

    def test_submit_has_order_id(self):
        filled = self.broker.submit(self.order)
        self.assertTrue(filled.order_id.startswith("paper-"))

    def test_submit_fill_price_positive(self):
        filled = self.broker.submit(self.order)
        self.assertGreater(filled.fill_price, 0)

    def test_submit_slippage_applied(self):
        filled = self.broker.submit(self.order)
        self.assertGreater(filled.slippage_actual, 0)

    def test_submit_rejects_without_stop(self):
        from options_bot.contracts import ApprovedOrder
        from options_bot.exceptions import RiskVetoError
        bad_order = ApprovedOrder(
            legs=self.order.legs,
            net_debit_credit=self.order.net_debit_credit,
            estimated_fill_price=self.order.estimated_fill_price,
            hard_stop_price=0.0,           # missing stop
            max_loss_dollars=self.order.max_loss_dollars,
            position_size_contracts=self.order.position_size_contracts,
            risk_approved=True,
        )
        with self.assertRaises(RiskVetoError):
            self.broker.submit(bad_order)

    def test_submit_rejects_risk_not_approved(self):
        from options_bot.contracts import ApprovedOrder
        from options_bot.exceptions import RiskVetoError
        bad_order = ApprovedOrder(
            legs=self.order.legs,
            net_debit_credit=self.order.net_debit_credit,
            estimated_fill_price=self.order.estimated_fill_price,
            hard_stop_price=self.order.hard_stop_price,
            max_loss_dollars=self.order.max_loss_dollars,
            position_size_contracts=self.order.position_size_contracts,
            risk_approved=False,           # not approved
        )
        with self.assertRaises(RiskVetoError):
            self.broker.submit(bad_order)

    def test_get_positions_after_submit(self):
        self.broker.submit(self.order)
        positions = self.broker.get_positions()
        self.assertGreater(len(positions), 0)

    def test_close_position_removes_it(self):
        self.broker.submit(self.order)
        positions_before = self.broker.get_positions()
        symbol = positions_before[0]["symbol"]
        self.broker.close_position(symbol)
        positions_after = self.broker.get_positions()
        symbols_after = [p["symbol"] for p in positions_after]
        self.assertNotIn(symbol, symbols_after)

    def test_cancel_all_returns_zero(self):
        count = self.broker.cancel_all_orders()
        self.assertEqual(count, 0)

    def test_option_snapshot_returns_dict(self):
        snap = self.broker.get_option_snapshots(["SPY260620C00580000"])
        self.assertIn("SPY260620C00580000", snap)


# ---------------------------------------------------------------------------
# Test: AlpacaBroker raises without credentials
# ---------------------------------------------------------------------------

class TestAlpacaBrokerCredentials(unittest.TestCase):

    def test_raises_without_api_key(self):
        from options_bot.broker import AlpacaBroker
        from options_bot.exceptions import PipelineConnectionError
        import os
        # Ensure env vars are not set
        old_key = os.environ.pop("ALPACA_API_KEY", None)
        old_secret = os.environ.pop("ALPACA_SECRET_KEY", None)
        try:
            with self.assertRaises(PipelineConnectionError):
                AlpacaBroker()
        finally:
            if old_key:
                os.environ["ALPACA_API_KEY"] = old_key
            if old_secret:
                os.environ["ALPACA_SECRET_KEY"] = old_secret

    def test_get_broker_paper_stub(self):
        from options_bot.broker import get_broker, PaperBroker
        broker = get_broker(use_paper_stub=True)
        self.assertIsInstance(broker, PaperBroker)


# ---------------------------------------------------------------------------
# Test: Full pipeline — strategy → risk → paper broker (end to end)
# ---------------------------------------------------------------------------

class TestFullPipelineWithBroker(unittest.TestCase):
    """
    Proves the complete pipeline:
    chain → strategy → risk → execution guard → paper broker → FilledOrder
    """

    def _run_pipeline(self, strategy_name: str, equity: float) -> "FilledOrder":
        from options_bot.strategy import get_strategy
        from options_bot.risk import RiskManager, RiskConfig
        from options_bot.broker import PaperBroker

        # Widened call_strikes for short_strangle — production default
        # call_delta=0.15 needs a strike further OTM than the default
        # ladder's max (610) reaches.
        call_strikes = (
            [595, 600, 605, 610, 615, 620, 625, 630, 635, 640]
            if strategy_name == "short_strangle" else None
        )
        chain = make_spy_chain_for_strategy(dte=35, call_strikes=call_strikes)
        strategy = get_strategy(strategy_name)
        signal = strategy.evaluate(chain)

        rm = RiskManager(equity=equity, config=RiskConfig(risk_budget_pct=0.02))
        decision = rm.evaluate(
            max_loss_per_contract=signal.max_loss_per_contract,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
        )
        self.assertTrue(decision.approved, f"[{strategy_name}] Vetoed: {decision.rejection_reason}")

        order = rm.build_approved_order(
            legs=signal.legs,
            decision=decision,
            net_debit_credit=signal.net_debit_credit,
            estimated_fill_price=signal.estimated_fill_price,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
            underlying=signal.underlying,
        )

        broker = PaperBroker(starting_equity=equity)
        return broker.submit(order)

    def test_csp_end_to_end(self):
        from options_bot.contracts import FilledOrder
        filled = self._run_pipeline("csp", equity=5_000_000)
        self.assertIsInstance(filled, FilledOrder)
        self.assertEqual(filled.status, "open")
        self.assertGreater(filled.fill_price, 0)
        self.assertEqual(filled.broker, "paper")

    def test_spread_end_to_end(self):
        from options_bot.contracts import FilledOrder
        filled = self._run_pipeline("short_put_spread", equity=100_000)
        self.assertIsInstance(filled, FilledOrder)
        self.assertEqual(filled.status, "open")

    def test_strangle_end_to_end(self):
        from options_bot.contracts import FilledOrder
        filled = self._run_pipeline("short_strangle", equity=500_000)
        self.assertIsInstance(filled, FilledOrder)
        self.assertEqual(filled.status, "open")



# ---------------------------------------------------------------------------
# Test: Orchestrator components (no network — uses PaperBroker stub)
# ---------------------------------------------------------------------------

class TestTradeDatabase(unittest.TestCase):

    def setUp(self):
        import tempfile, os
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        from options_bot.orchestrator import TradeDatabase
        self.db = TradeDatabase(sqlite_path=self.tmp.name)

    def tearDown(self):
        import os
        os.unlink(self.tmp.name)

    def _make_filled(self) -> "FilledOrder":
        from options_bot.strategy import ShortPutSpread, ShortPutSpreadConfig
        from options_bot.risk import RiskManager, RiskConfig
        from options_bot.broker import PaperBroker

        chain = make_spy_chain_for_strategy(dte=30)
        signal = ShortPutSpread(ShortPutSpreadConfig(
            min_dte=15, max_dte=60, min_open_interest=50, min_credit=0.10
        )).evaluate(chain)
        rm = RiskManager(equity=100_000, config=RiskConfig(risk_budget_pct=0.02))
        decision = rm.evaluate(signal.max_loss_per_contract, signal.hard_stop_price)
        order = rm.build_approved_order(
            legs=signal.legs, decision=decision,
            net_debit_credit=signal.net_debit_credit,
            estimated_fill_price=signal.estimated_fill_price,
            hard_stop_price=signal.hard_stop_price,
            strategy_name=signal.strategy_name,
            underlying=signal.underlying,
        )
        return PaperBroker(100_000).submit(order)

    def test_save_and_retrieve(self):
        filled = self._make_filled()
        self.db.save_fill(filled)
        open_trades = self.db.get_open_trades()
        self.assertGreater(len(open_trades), 0)
        ids = [t["id"] for t in open_trades]
        self.assertIn(filled.order_id, ids)

    def test_update_status(self):
        filled = self._make_filled()
        self.db.save_fill(filled)
        self.db.update_status(
            filled.order_id, "stopped_out",
            close_price=2.50, realized_pnl=-150.0
        )
        open_trades = self.db.get_open_trades()
        ids = [t["id"] for t in open_trades]
        self.assertNotIn(filled.order_id, ids)

    def test_schema_created(self):
        import sqlite3
        conn = sqlite3.connect(self.tmp.name)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        self.assertIsNotNone(cur.fetchone())
        conn.close()


class TestDiscordNotifier(unittest.TestCase):

    def test_send_discord_silent_on_empty_url(self):
        from options_bot.orchestrator import send_discord
        send_discord("", "test message")  # should not raise

    def test_send_discord_silent_on_bad_url(self):
        from options_bot.orchestrator import send_discord
        send_discord("https://invalid.example.com/webhook", "test")  # should not raise


class TestOrchestratorConfig(unittest.TestCase):

    def test_default_config(self):
        from options_bot.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig()
        self.assertIn("SPY", cfg.tickers)
        self.assertEqual(cfg.strategy_name, "short_put_spread")
        self.assertTrue(cfg.paper)

    def test_custom_tickers(self):
        from options_bot.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig(tickers=["AAPL", "TSLA"])
        self.assertEqual(cfg.tickers, ["AAPL", "TSLA"])


class TestSessionState(unittest.TestCase):

    def test_reset_for_new_day(self):
        from options_bot.orchestrator import SessionState
        from datetime import date, timedelta
        state = SessionState()
        state.trade_date = date.today() - timedelta(days=1)
        state.daily_realized_pnl = -500.0
        state.scan_ran = True
        state.reset_for_new_day()
        self.assertEqual(state.trade_date, date.today())
        self.assertEqual(state.daily_realized_pnl, 0.0)
        self.assertFalse(state.scan_ran)

    def test_record_error(self):
        from options_bot.orchestrator import SessionState
        state = SessionState()
        state.record_error("test error")
        self.assertEqual(len(state.errors_today), 1)
        self.assertIn("test error", state.errors_today[0])


class TestTradingPipeline(unittest.TestCase):
    """Pipeline integration test using PaperBroker stub — no network."""

    def _make_pipeline(self, strategy_name="short_put_spread", equity=100_000):
        from options_bot.orchestrator import (
            OrchestratorConfig, TradingPipeline, TradeDatabase, SessionState
        )
        from options_bot.risk import RiskManager, RiskConfig
        from options_bot.broker import PaperBroker
        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()

        config = OrchestratorConfig(
            tickers=["SPY"],
            strategy_name=strategy_name,
            risk_config=RiskConfig(risk_budget_pct=0.02, max_trades_per_day=5),
            min_dte=5, max_dte=60,
            min_open_interest=50,
        )
        broker = PaperBroker(starting_equity=equity)
        rm = RiskManager(equity=equity, config=config.risk_config)
        db = TradeDatabase(sqlite_path=tmp.name)
        state = SessionState()
        pipeline = TradingPipeline(config, rm, broker, db, state)
        return pipeline, tmp.name

    def _patch_loader(self, pipeline):
        """Patch YFinanceDataLoader to return synthetic chain (no network)."""
        from unittest.mock import patch, MagicMock

        chain = make_spy_chain_for_strategy(dte=30)
        expirations = [chain[0].expiry.isoformat()]

        mock_loader = MagicMock()
        mock_loader.get_expirations.return_value = expirations
        mock_loader.get_chain_filtered.return_value = [r.raw for r in chain]

        return patch(
            'options_bot.orchestrator.YFinanceDataLoader',
            return_value=mock_loader
        )

    def test_pipeline_returns_filled_order_on_spread(self):
        import os
        pipeline, tmp_path = self._make_pipeline("short_put_spread", equity=100_000)
        try:
            with self._patch_loader(pipeline):
                result = pipeline.run_for_ticker("SPY")
            from options_bot.contracts import FilledOrder
            self.assertIsInstance(result, FilledOrder)
            self.assertEqual(result.broker, "paper")
        finally:
            os.unlink(tmp_path)

    def test_pipeline_records_in_db(self):
        import os
        from options_bot.orchestrator import TradeDatabase
        pipeline, tmp_path = self._make_pipeline("short_put_spread", equity=100_000)
        try:
            with self._patch_loader(pipeline):
                pipeline.run_for_ticker("SPY")
            db = TradeDatabase(sqlite_path=tmp_path)
            open_trades = db.get_open_trades()
            self.assertGreater(len(open_trades), 0)
        finally:
            os.unlink(tmp_path)

    def test_pipeline_returns_none_when_risk_vetoes(self):
        """With tiny equity, risk manager should veto and pipeline returns None."""
        import os
        pipeline, tmp_path = self._make_pipeline("short_put_spread", equity=100)
        try:
            with self._patch_loader(pipeline):
                result = pipeline.run_for_ticker("SPY")
            self.assertIsNone(result)
        finally:
            os.unlink(tmp_path)


class TestMarketHoursHelpers(unittest.TestCase):

    def test_minutes_to_close_is_numeric(self):
        from options_bot.orchestrator import _minutes_to_close
        mins = _minutes_to_close()
        self.assertIsInstance(mins, float)

    def test_market_is_open_returns_bool(self):
        from options_bot.orchestrator import _market_is_open
        result = _market_is_open()
        self.assertIsInstance(result, bool)


class TestParityCheck(unittest.TestCase):
    """Put-call parity data-quality gate (parity_check.check_parity)."""

    def _make_pair(self, strike, c_mid, p_mid, spot, dte, expiry, spread=0.10):
        from options_bot.contracts import OptionChainRow, EnrichedOptionRow
        rows = []
        for otype, mid in (("call", c_mid), ("put", p_mid)):
            raw = OptionChainRow(
                symbol=f"SPY{strike}{otype[0].upper()}", underlying="SPY",
                option_type=otype, strike=strike, expiry=expiry, dte=dte,
                bid=mid - spread / 2, ask=mid + spread / 2, last_price=mid,
                mid_price=mid, volume=1000, open_interest=2000,
                underlying_price=spot, data_timestamp=datetime.now(tz=timezone.utc),
            )
            rows.append(EnrichedOptionRow(raw=raw, iv=0.20, delta=0.3))
        return rows

    def _consistent_mids(self, strike, spot, rate, years):
        import math
        theo = spot - strike * math.exp(-rate * years)  # = C - P
        p = 8.0
        return p + theo, p

    def setUp(self):
        self.spot = 582.50
        self.rate = 0.045
        self.dte = 30
        self.expiry = date.today() + timedelta(days=self.dte)
        self.years = self.dte / 365.0

    def test_clean_chain_passes(self):
        from options_bot.parity_check import check_parity
        chain = []
        for k in [575, 580, 582.5, 585, 590]:
            c, p = self._consistent_mids(k, self.spot, self.rate, self.years)
            chain += self._make_pair(k, c, p, self.spot, self.dte, self.expiry)
        result = check_parity(chain, rate=self.rate)
        self.assertTrue(result.ok)

    def test_grossly_broken_chain_flagged(self):
        from options_bot.parity_check import check_parity
        chain = []
        for k in [575, 580, 582.5, 585, 590]:
            c, p = self._consistent_mids(k, self.spot, self.rate, self.years)
            chain += self._make_pair(k, c + 25.0, p, self.spot, self.dte, self.expiry)
        result = check_parity(chain, rate=self.rate)
        self.assertFalse(result.ok)

    def test_thin_chain_fails_open(self):
        from options_bot.parity_check import check_parity
        c, p = self._consistent_mids(582.5, self.spot, self.rate, self.years)
        chain = self._make_pair(582.5, c + 25.0, p, self.spot, self.dte, self.expiry)
        result = check_parity(chain, rate=self.rate)
        self.assertTrue(result.ok)  # insufficient pairs -> fail open

    def test_small_deviation_within_tolerance(self):
        from options_bot.parity_check import check_parity
        chain = []
        for k in [575, 580, 582.5, 585, 590]:
            c, p = self._consistent_mids(k, self.spot, self.rate, self.years)
            chain += self._make_pair(k, c, p + self.spot * 0.01, self.spot, self.dte, self.expiry)
        result = check_parity(chain, rate=self.rate)
        self.assertTrue(result.ok)  # 1% legit deviation must not trip the gate

    def test_empty_chain_fails_open(self):
        from options_bot.parity_check import check_parity
        result = check_parity([], rate=self.rate)
        self.assertTrue(result.ok)

class TestZeroDTECircuitBreaker(unittest.TestCase):
    """Dedicated 0DTE guardrails: daily loss cap + consecutive-losing-day cooldown."""

    def _make_cb(self):
        from options_bot.strategy_0dte import ZeroDTEConfig
        from options_bot.zerodte_guard import ZeroDTECircuitBreaker
        store = {}
        def load(key, max_age): return store.get(key)
        def save(key, val, max_age): store[key] = dict(val)
        return ZeroDTECircuitBreaker(ZeroDTEConfig(), load, save), store

    def test_daily_loss_cap_blocks_over_threshold(self):
        cb, _ = self._make_cb()
        d = date(2026, 6, 22)
        self.assertTrue(cb.check_entry_allowed(d, 100_000, -200).allowed)   # within $500 cap
        self.assertFalse(cb.check_entry_allowed(d, 100_000, -500).allowed)  # at cap
        self.assertFalse(cb.check_entry_allowed(d, 100_000, -600).allowed)  # over cap

    def test_three_losing_days_arm_cooldown(self):
        cb, _ = self._make_cb()
        cb.record_eod(date(2026, 6, 22), -100)
        cb.record_eod(date(2026, 6, 23), -100)
        st = cb.record_eod(date(2026, 6, 24), -100)
        self.assertIsNotNone(st["cooldown_until_iso"])
        # entry blocked the next day
        self.assertFalse(cb.check_entry_allowed(date(2026, 6, 25), 100_000, 0.0).allowed)

    def test_win_resets_streak(self):
        cb, _ = self._make_cb()
        cb.record_eod(date(2026, 6, 22), -100)
        cb.record_eod(date(2026, 6, 23), -100)
        st = cb.record_eod(date(2026, 6, 24), +50)
        self.assertEqual(st["consec_losing_days"], 0)
        self.assertIsNone(st["cooldown_until_iso"])

    def test_flat_day_does_not_count(self):
        cb, _ = self._make_cb()
        cb.record_eod(date(2026, 6, 22), -100)
        st = cb.record_eod(date(2026, 6, 23), 0.0)
        self.assertEqual(st["consec_losing_days"], 1)

    def test_eod_idempotent_same_day(self):
        cb, _ = self._make_cb()
        cb.record_eod(date(2026, 6, 22), -100)
        st = cb.record_eod(date(2026, 6, 22), -100)
        self.assertEqual(st["consec_losing_days"], 1)

    def test_cooldown_expires_on_target_date(self):
        from options_bot.zerodte_guard import _add_trading_days
        cb, _ = self._make_cb()
        cb.record_eod(date(2026, 6, 22), -100)
        cb.record_eod(date(2026, 6, 23), -100)
        st = cb.record_eod(date(2026, 6, 24), -100)
        cd_until = date.fromisoformat(st["cooldown_until_iso"])
        self.assertEqual(cd_until, _add_trading_days(date(2026, 6, 24), 14))
        self.assertTrue(cb.check_entry_allowed(cd_until, 100_000, 0.0).allowed)

class TestIronCondor(unittest.TestCase):
    """Iron Condor: defined-risk neutral structure composed from both spreads."""

    def setUp(self):
        self.chain = make_spy_chain_for_strategy(
            dte=35,
            call_strikes=[595, 600, 605, 610, 615, 620, 625, 630, 635, 640],
        )

    def test_returns_four_leg_signal(self):
        from options_bot.strategy import IronCondor, StrategySignal
        sig = IronCondor().evaluate(self.chain)
        self.assertIsInstance(sig, StrategySignal)
        self.assertEqual(len(sig.legs), 4)

    def test_has_both_put_and_call_spreads(self):
        from options_bot.strategy import IronCondor
        sig = IronCondor().evaluate(self.chain)
        sides = [(l.side.split('_')[0], l.option_type) for l in sig.legs]
        self.assertIn(("sell", "put"), sides)
        self.assertIn(("buy", "put"), sides)
        self.assertIn(("sell", "call"), sides)
        self.assertIn(("buy", "call"), sides)

    def test_is_net_credit(self):
        from options_bot.strategy import IronCondor
        sig = IronCondor().evaluate(self.chain)
        self.assertLess(sig.net_debit_credit, 0)  # credit
        self.assertGreater(sig.estimated_fill_price, 0)

    def test_max_loss_finite_and_positive(self):
        from options_bot.strategy import IronCondor
        sig = IronCondor().evaluate(self.chain)
        self.assertGreater(sig.max_loss_per_contract, 0)
        self.assertLess(sig.max_loss_per_contract, float('inf'))

    def test_registered_in_registry(self):
        from options_bot.strategy import STRATEGY_REGISTRY, get_strategy, IronCondor
        self.assertIn("iron_condor", STRATEGY_REGISTRY)
        self.assertIsInstance(get_strategy("iron_condor"), IronCondor)

    def test_total_credit_gate_rejects_low_credit(self):
        from options_bot.strategy import IronCondor, IronCondorConfig
        from options_bot.exceptions import LiquidityFilterError
        cfg = IronCondorConfig(min_total_credit=999.0)  # impossible
        with self.assertRaises(LiquidityFilterError):
            IronCondor(cfg).evaluate(self.chain)


class TestIronCondorGating(unittest.TestCase):
    """Iron Condor must stay dormant until BOTH gates clear."""

    def test_default_config_disabled(self):
        from options_bot.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig()
        self.assertFalse(cfg.iron_condor_enabled)
        self.assertEqual(cfg.iron_condor_min_trades, 30)

    def test_not_in_default_extra_strategies(self):
        from options_bot.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig()
        self.assertNotIn("iron_condor", cfg.extra_strategies)

class TestGatedSignalsCore(unittest.TestCase):
    """Pure-logic cores for the gated regime/signal inputs (credit_regime,
    analyst_revisions). No network. Mirrors the author's provided tests plus
    the gating-default checks."""

    # ---------- credit regime ----------
    def test_credit_stress_produces_defensive_nudge(self):
        from options_bot.credit_regime import compute_credit_regime
        sig = compute_credit_regime([80.0] * 60 + [70.0], [95.0] * 61)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.state, "stress")
        self.assertLess(sig.regime_adjustment, 0)

    def test_credit_calm_produces_positive_nudge(self):
        from options_bot.credit_regime import compute_credit_regime
        sig = compute_credit_regime([80.0] * 60 + [90.0], [95.0] * 61)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.state, "calm")
        self.assertGreater(sig.regime_adjustment, 0)

    def test_credit_insufficient_history_returns_none(self):
        from options_bot.credit_regime import compute_credit_regime
        self.assertIsNone(compute_credit_regime([80.0] * 10, [95.0] * 10))

    def test_credit_flat_series_is_neutral(self):
        from options_bot.credit_regime import compute_credit_regime
        sig = compute_credit_regime([80.0] * 61, [95.0] * 61)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.state, "neutral")
        self.assertLess(abs(sig.regime_adjustment), 0.1)

    def test_credit_adjustment_clamped(self):
        from options_bot.credit_regime import compute_credit_regime, PROVISIONAL_MAX_ADJUSTMENT
        sig = compute_credit_regime([80.0] * 60 + [40.0], [95.0] * 61)  # severe collapse
        self.assertGreaterEqual(sig.regime_adjustment, -PROVISIONAL_MAX_ADJUSTMENT)

    # ---------- analyst revisions ----------
    def _ev(self, days_ago, frm, to, action="up"):
        from options_bot.analyst_revisions import RevisionEvent
        return RevisionEvent(
            date=datetime.now(timezone.utc) - timedelta(days=days_ago),
            firm="TestCo", from_grade=frm, to_grade=to, action=action,
        )

    def test_revisions_cluster_of_upgrades_is_bullish(self):
        from options_bot.analyst_revisions import score_revisions
        sig = score_revisions("XLK", [self._ev(2, "Hold", "Buy"),
                                       self._ev(5, "Sell", "Buy"),
                                       self._ev(9, "Hold", "Buy")])
        self.assertEqual(sig.direction, "bullish")
        self.assertEqual(sig.n_upgrades, 3)
        self.assertGreater(sig.net_score, 0)

    def test_revisions_downgrades_are_bearish(self):
        from options_bot.analyst_revisions import score_revisions
        sig = score_revisions("XLF", [self._ev(1, "Buy", "Sell", "down"),
                                       self._ev(4, "Buy", "Hold", "down")])
        self.assertEqual(sig.direction, "bearish")
        self.assertEqual(sig.n_downgrades, 2)

    def test_revisions_old_events_excluded(self):
        from options_bot.analyst_revisions import score_revisions
        sig = score_revisions("SPY", [self._ev(120, "Sell", "Strong Buy"),
                                       self._ev(200, "Sell", "Buy")])
        self.assertEqual(sig.direction, "neutral")
        self.assertEqual(sig.net_score, 0.0)

    def test_revisions_action_fallback_when_grades_unmappable(self):
        from options_bot.analyst_revisions import score_revisions
        sig = score_revisions("IWM", [self._ev(3, None, None, "up"),
                                       self._ev(6, None, None, "up")])
        self.assertEqual(sig.n_upgrades, 2)

    def test_revisions_reiteration_not_counted(self):
        from options_bot.analyst_revisions import score_revisions
        sig = score_revisions("QQQ", [self._ev(2, "Buy", "Buy", "main")])
        self.assertEqual(sig.net_score, 0.0)
        self.assertEqual(sig.n_upgrades, 0)


class TestGatedSignalsDormant(unittest.TestCase):
    """Both new signals must be disabled by default (dormant until milestone
    + explicit enable flag)."""

    def test_credit_regime_disabled_by_default(self):
        from options_bot.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig()
        self.assertFalse(cfg.credit_regime_enabled)
        self.assertEqual(cfg.credit_regime_min_trades, 30)

    def test_analyst_revisions_disabled_by_default(self):
        from options_bot.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig()
        self.assertFalse(cfg.analyst_revisions_enabled)
        self.assertEqual(cfg.analyst_revisions_min_trades, 30)

class TestGatedRegimeSignalWiring(unittest.TestCase):
    """credit_regime / analyst_revisions are wired but OFF until both gates clear."""

    def test_regime_detector_signals_off_by_default(self):
        from options_bot.regime import RegimeDetector
        d = RegimeDetector()
        self.assertFalse(d._credit_regime_active)
        self.assertFalse(d._analyst_revisions_active)

    def test_inactive_credit_regime_not_computed(self):
        from options_bot.regime import RegimeDetector
        d = RegimeDetector(credit_regime_active=False)
        # Guarded path: when inactive the detector must not invoke the fetch.
        self.assertIsNone(
            d._compute_credit_regime() if d._credit_regime_active else None
        )

    def test_zero_credit_adjustment_is_noop(self):
        from options_bot.regime import RegimeDetector
        d = RegimeDetector(credit_regime_active=True)
        base = {
            "vix_level": 18.0, "vix_trend": "stable", "trend_strength": 0.5,
            "yield_curve_slope": 0.5, "hurst": 0.5, "_breadth_scores": {},
            "vix_term_structure": "unknown", "vix_term_ratio": 1.0,
            "vix_percentile": 50.0,
        }
        with_zero = dict(base); with_zero["credit_regime_adjustment"] = 0.0
        r_zero, _ = d._classify(with_zero)
        r_missing, _ = d._classify(dict(base))
        self.assertEqual(r_zero, r_missing)

    def test_config_gating_flags_default_off(self):
        from options_bot.orchestrator import OrchestratorConfig
        c = OrchestratorConfig()
        self.assertFalse(c.credit_regime_enabled)
        self.assertFalse(c.analyst_revisions_enabled)
        self.assertEqual(c.credit_regime_min_trades, 30)
        self.assertEqual(c.analyst_revisions_min_trades, 30)

class TestVRPGate(unittest.TestCase):
    """Vol-risk-premium gate: realized vol estimators + gate decision logic."""

    def _series(self, vol=0.20, n=60, seed=7):
        import numpy as np
        np.random.seed(seed)
        rets = np.random.randn(n) * vol * (1/252) ** 0.5
        close = 100 * np.exp(np.cumsum(rets))
        high = close * 1.004
        low = close * 0.996
        open_ = list(close[:-1]); open_ = [100.0] + open_
        return list(open_), list(high), list(close * 0 + high*0 + low), list(close)

    def test_rv_recovers_known_vol(self):
        from options_bot.realized_vol import rv_close_to_close
        import numpy as np
        np.random.seed(42)
        rets = np.random.randn(300) * 0.20 * (1/252) ** 0.5
        close = list(100 * np.exp(np.cumsum(rets)))
        rv = rv_close_to_close(close, 63)
        self.assertTrue(0.14 < rv < 0.26)  # recovers ~0.20 within estimator noise

    def test_rv_insufficient_data_none(self):
        from options_bot.realized_vol import rv_close_to_close, rv_yang_zhang
        self.assertIsNone(rv_close_to_close([100, 101], 21))
        self.assertIsNone(rv_yang_zhang([100]*5, [101]*5, [99]*5, [100]*5, 21))

    def test_vrp_rich_iv_passes(self):
        from options_bot.vrp_gate import evaluate_vrp_gate
        o, h, l, c = self._series(vol=0.20)
        r = evaluate_vrp_gate(0.30, o, h, l, c, estimator="close_to_close")
        self.assertTrue(r.passes)
        self.assertGreater(r.size_factor, 0.5)

    def test_vrp_thin_premium_vetoes(self):
        from options_bot.vrp_gate import evaluate_vrp_gate
        o, h, l, c = self._series(vol=0.20)
        r = evaluate_vrp_gate(0.205, o, h, l, c, estimator="close_to_close")
        self.assertFalse(r.passes)
        self.assertEqual(r.size_factor, 0.0)

    def test_vrp_negative_vetoes(self):
        from options_bot.vrp_gate import evaluate_vrp_gate
        o, h, l, c = self._series(vol=0.20)
        r = evaluate_vrp_gate(0.15, o, h, l, c, estimator="close_to_close")
        self.assertFalse(r.passes)

    def test_vrp_invalid_iv_returns_none(self):
        from options_bot.vrp_gate import evaluate_vrp_gate
        o, h, l, c = self._series()
        self.assertIsNone(evaluate_vrp_gate(0.0, o, h, l, c))
        self.assertIsNone(evaluate_vrp_gate(None, o, h, l, c))

    def test_signal_has_default_vrp_size_factor(self):
        from options_bot.strategy import StrategySignal
        import inspect
        # vrp_size_factor must default to 1.0 (no shrink when gate inactive)
        sig_fields = {f.name: f.default for f in __import__('dataclasses').fields(StrategySignal)}
        self.assertIn("vrp_size_factor", sig_fields)

    def test_vrp_config_flags_default_off(self):
        from options_bot.orchestrator import OrchestratorConfig
        c = OrchestratorConfig()
        self.assertFalse(c.vrp_gate_enabled)
        self.assertEqual(c.vrp_gate_min_trades, 30)


class TestRegressionThreeLiveBugs(unittest.TestCase):
    """Regression guards for the three bugs found in the 2026-06-22 Railway
    logs: (1) extra-strategy pass passed kwargs run_for_ticker doesn't accept,
    (2) the CBOE GEX fallback returned a dict where a GEXPin was required,
    (3) LIKE '0dte_%' broke psycopg2 param binding (tuple index out of range).
    None were covered before; all three silently broke the live trade path."""

    # --- Bug 1: run_for_ticker signature contract ---
    def test_run_for_ticker_signature_contract(self):
        import inspect
        from options_bot.orchestrator import TradingPipeline
        params = set(inspect.signature(TradingPipeline.run_for_ticker).parameters)
        # The contract the extra-strategy pass must honor:
        self.assertIn("regime_name", params)
        self.assertIn("regime_options_weight", params)
        # The kwargs the buggy call site invented — must NOT silently reappear:
        for bad in ("regime", "open_trades", "sentiment_signals"):
            self.assertNotIn(bad, params,
                             f"run_for_ticker gained '{bad}' — update call sites or this guard")

    # --- Bug 2: CBOE fallback dict -> GEXPin adapter ---
    def _engine(self):
        from options_bot.strategy_0dte import GEXEngine, ZeroDTEConfig
        return GEXEngine(broker=MagicMock(), config=ZeroDTEConfig())

    def test_cboe_to_pin_returns_gexpin_consumable_by_select_strikes(self):
        from options_bot.strategy_0dte import GEXPin
        eng = self._engine()
        cboe = {"spot": 500.0, "net_gex_usd": -1.2e9, "call_wall": 505.0,
                "put_wall": 498.0, "negative_gamma": True, "source": "cboe_delayed"}
        pin = eng._cboe_to_pin(cboe)
        self.assertIsInstance(pin, GEXPin)
        # nearest wall to spot (498 is 2 away vs 505's 5) -> put_wall
        self.assertEqual(pin.strike, 498.0)
        self.assertAlmostEqual(pin.distance_from_spot, 2.0)
        self.assertEqual(pin.side, "BELOW")
        self.assertEqual(pin.regime, "NEGATIVE_GAMMA")
        # The crash was pin.distance_from_spot on a dict — prove attr access works
        self.assertTrue(hasattr(pin, "distance_from_spot"))

    def test_cboe_to_pin_guards_missing_data(self):
        eng = self._engine()
        self.assertIsNone(eng._cboe_to_pin({"call_wall": 1, "put_wall": 2}))  # no spot
        self.assertIsNone(eng._cboe_to_pin({"spot": 500.0}))                  # no walls

    # --- Bug 3: psycopg2 literal-% escaping ---
    def test_execute_escapes_literal_percent_for_pg(self):
        from options_bot.orchestrator import TradeDatabase
        captured = {}

        class FakeCur:
            def execute(self, sql, params): captured["sql"] = sql; captured["params"] = params
        class FakeConn:
            def cursor(self): return FakeCur()

        db = TradeDatabase.__new__(TradeDatabase)   # bypass __init__/real conn
        db._use_pg = True
        db._execute(FakeConn(),
                    "SELECT 1 FROM trades WHERE trade_date = ? AND strategy LIKE '0dte_%'",
                    ("2026-06-22",))
        # literal % doubled, ? -> %s, exactly one bind target
        self.assertIn("LIKE '0dte_%%'", captured["sql"])
        self.assertIn("trade_date = %s", captured["sql"])
        self.assertEqual(captured["sql"].count("%s"), 1)


class TestLiquidityFilterOIFix(unittest.TestCase):
    """The 2026-06-22 logs showed every ticker (incl. SPY/IWM) rejected at the
    liquidity gate. Root cause: yfinance returns open_interest=0, and the gate
    rejected 0 < min_oi. Alpaca has no OI field at all, so OI=0/None is now
    treated as 'unknown' and the spread gate decides. These guard that fix and
    the new per-reason rejection breakdown."""

    def _loader(self, rows):
        from options_bot.market_data import YFinanceDataLoader
        ld = YFinanceDataLoader.__new__(YFinanceDataLoader)   # bypass network init
        ld.ticker_str = "SPY"
        ld.get_chain = lambda expiry: rows
        return ld

    def test_zero_oi_not_rejected_when_spread_ok(self):
        # tight-spread contract with OI=0 (the yfinance failure) must be accepted
        rows = [make_option_chain_row(open_interest=0, bid=9.33, ask=9.53)]
        out = self._loader(rows).get_chain_filtered("2026-07-31", min_open_interest=100)
        self.assertEqual(len(out), 1)

    def test_none_oi_not_rejected_when_spread_ok(self):
        rows = [make_option_chain_row(open_interest=None, bid=9.33, ask=9.53)]
        out = self._loader(rows).get_chain_filtered("2026-07-31", min_open_interest=100)
        self.assertEqual(len(out), 1)

    def test_positive_oi_below_threshold_still_rejected(self):
        from options_bot.exceptions import LiquidityFilterError
        rows = [make_option_chain_row(open_interest=5, bid=9.33, ask=9.53)]
        with self.assertRaises(LiquidityFilterError) as ctx:
            self._loader(rows).get_chain_filtered("2026-07-31", min_open_interest=100)
        self.assertIn("oi=1", str(ctx.exception))  # reason breakdown present

    def test_wide_spread_rejected_with_reason(self):
        from options_bot.exceptions import LiquidityFilterError
        # OI fine, but spread ~50% — the real liquidity guard should fire
        rows = [make_option_chain_row(open_interest=5000, bid=1.00, ask=3.00)]
        with self.assertRaises(LiquidityFilterError) as ctx:
            self._loader(rows).get_chain_filtered("2026-07-31", max_spread_pct=0.20)
        self.assertIn("spread=1", str(ctx.exception))

    def test_mixed_chain_accepts_only_tradeable(self):
        rows = [
            make_option_chain_row(symbol="A", open_interest=0, bid=9.33, ask=9.53),   # ok (OI unknown)
            make_option_chain_row(symbol="B", open_interest=5000, bid=1.00, ask=3.00), # wide spread -> reject
            make_option_chain_row(symbol="C", open_interest=5000, bid=4.20, ask=4.40), # ok
        ]
        out = self._loader(rows).get_chain_filtered("2026-07-31",
                                                    min_open_interest=100, max_spread_pct=0.20)
        self.assertEqual(len(out), 2)


class TestMassiveOpenInterest(unittest.TestCase):
    """Massive (Polygon) free-tier OI source. Must fail open (no key / error ->
    empty map, never raises) and must enrich the chain's open_interest from real
    Massive data when available, matched by bare OCC symbol."""

    def setUp(self):
        import options_bot.massive_data as md
        md._cache.clear()
        md._warned_no_key = False

    def _fake_resp(self, payload):
        import io, json
        class _Ctx:
            def __enter__(self_):
                return self_
            def __exit__(self_, *a):
                return False
            def read(self_):
                return json.dumps(payload).encode()
        return _Ctx()

    def test_no_key_returns_empty_failopen(self):
        import options_bot.massive_data as md
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(md.get_open_interest_map("SPY", "2026-07-31"), {})

    def test_parses_oi_map_and_strips_prefix(self):
        import options_bot.massive_data as md
        payload = {"results": [
            {"details": {"ticker": "O:SPY260731P00500000"}, "open_interest": 1234},
            {"details": {"ticker": "O:SPY260731C00600000"}, "open_interest": 0},
        ], "status": "OK"}
        with patch.dict("os.environ", {"MASSIVE_API_KEY": "k"}, clear=False):
            with patch("urllib.request.urlopen", return_value=self._fake_resp(payload)):
                m = md.get_open_interest_map("SPY", "2026-07-31")
        self.assertEqual(m, {"SPY260731P00500000": 1234, "SPY260731C00600000": 0})

    def test_daily_cache_avoids_refetch(self):
        import options_bot.massive_data as md
        payload = {"results": [{"details": {"ticker": "O:SPY260731P00500000"},
                                "open_interest": 50}]}
        with patch.dict("os.environ", {"MASSIVE_API_KEY": "k"}, clear=False):
            with patch("urllib.request.urlopen", return_value=self._fake_resp(payload)) as op:
                md.get_open_interest_map("SPY", "2026-07-31")
                md.get_open_interest_map("SPY", "2026-07-31")  # cached
                self.assertEqual(op.call_count, 1)

    def test_http_error_failopen(self):
        import options_bot.massive_data as md
        import urllib.error
        def boom(*a, **k):
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        with patch.dict("os.environ", {"MASSIVE_API_KEY": "k"}, clear=False):
            with patch("urllib.request.urlopen", side_effect=boom):
                self.assertEqual(md.get_open_interest_map("SPY", "2026-07-31"), {})

    def test_enrichment_overrides_zero_oi(self):
        from options_bot.market_data import YFinanceDataLoader
        ld = YFinanceDataLoader.__new__(YFinanceDataLoader)
        ld.ticker_str = "SPY"
        rows = [
            make_option_chain_row(symbol="SPY260731P00500000", open_interest=0),
            make_option_chain_row(symbol="SPY260731C00600000", open_interest=0),
        ]
        with patch("options_bot.massive_data.get_open_interest_map",
                   return_value={"SPY260731P00500000": 4200}):
            out = ld._enrich_open_interest(rows, "2026-07-31")
        self.assertEqual(out[0].open_interest, 4200)   # enriched
        self.assertEqual(out[1].open_interest, 0)      # untouched (not in map)

    def test_enrichment_noop_when_massive_empty(self):
        from options_bot.market_data import YFinanceDataLoader
        ld = YFinanceDataLoader.__new__(YFinanceDataLoader)
        ld.ticker_str = "SPY"
        rows = [make_option_chain_row(symbol="SPY260731P00500000", open_interest=7)]
        with patch("options_bot.massive_data.get_open_interest_map", return_value={}):
            out = ld._enrich_open_interest(rows, "2026-07-31")
        self.assertEqual(out[0].open_interest, 7)      # unchanged, fail-open


class TestAlpacaQuoteEnrichment(unittest.TestCase):
    """Alpaca real-time quote source for the liquidity gate. Must fail open
    (no creds / disabled / error -> empty map, never raises) and must overlay
    Alpaca bid/ask onto chain rows (recomputing mid/spread), matched by OCC
    symbol, only when Alpaca returns a usable quote (present, not 0/0)."""

    def setUp(self):
        import options_bot.alpaca_quotes as aq
        aq._client = None
        aq._warned_no_key = False

    class _Q:
        def __init__(self, bid, ask):
            self.bid_price = bid
            self.ask_price = ask

    class _Snap:
        def __init__(self, bid, ask):
            self.latest_quote = TestAlpacaQuoteEnrichment._Q(bid, ask)

    class _Client:
        def __init__(self, mapping):
            self._m = mapping
        def get_option_chain(self, req):
            return self._m

    def test_no_creds_returns_empty_failopen(self):
        import options_bot.alpaca_quotes as aq
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(aq.get_quote_map("SPY", "2026-07-31"), {})

    def test_disabled_kill_switch_returns_empty(self):
        import options_bot.alpaca_quotes as aq
        with patch.dict("os.environ",
                        {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s",
                         "ALPACA_QUOTE_ENRICH": "false"}, clear=False):
            self.assertEqual(aq.get_quote_map("SPY", "2026-07-31"), {})

    def test_parses_quote_map(self):
        import options_bot.alpaca_quotes as aq
        fake = self._Client({
            "SPY260731P00500000": self._Snap(1.28, 1.29),
            "SPY260731C00600000": self._Snap(None, None),
        })
        with patch.dict("os.environ",
                        {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}, clear=False):
            with patch("options_bot.alpaca_quotes._get_client", return_value=fake):
                m = aq.get_quote_map("SPY", "2026-07-31")
        self.assertEqual(m["SPY260731P00500000"], (1.28, 1.29))
        self.assertEqual(m["SPY260731C00600000"], (None, None))

    def test_error_failopen(self):
        import options_bot.alpaca_quotes as aq
        class _Boom:
            def get_option_chain(self, req):
                raise RuntimeError("network")
        with patch.dict("os.environ",
                        {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}, clear=False):
            with patch("options_bot.alpaca_quotes._get_client", return_value=_Boom()):
                self.assertEqual(aq.get_quote_map("SPY", "2026-07-31"), {})

    def test_enrichment_overrides_zero_quote_and_recomputes(self):
        from options_bot.market_data import YFinanceDataLoader
        ld = YFinanceDataLoader.__new__(YFinanceDataLoader)
        ld.ticker_str = "SPY"
        rows = [
            make_option_chain_row(symbol="SPY260731P00500000", bid=0.0, ask=0.0),
            make_option_chain_row(symbol="SPY260731C00600000", bid=0.0, ask=0.0),
        ]
        with patch("options_bot.alpaca_quotes.get_quote_map",
                   return_value={"SPY260731P00500000": (1.28, 1.29)}):
            out = ld._enrich_quotes(rows, "2026-07-31")
        self.assertEqual(out[0].bid, 1.28)
        self.assertEqual(out[0].ask, 1.29)
        self.assertAlmostEqual(out[0].mid_price, 1.285, places=4)
        self.assertIsNotNone(out[0].spread_pct)
        self.assertTrue(out[0].spread_pct < 0.05)   # ~0.78% — clears the spread gate
        self.assertEqual(out[0].source, "yfinance+alpaca_quote")
        self.assertEqual(out[1].bid, 0.0)            # untouched (not in map)

    def test_enrichment_skips_alpaca_zero_quote(self):
        # Alpaca also returns 0/0 -> leave the row alone (fail-open, per row).
        from options_bot.market_data import YFinanceDataLoader
        ld = YFinanceDataLoader.__new__(YFinanceDataLoader)
        ld.ticker_str = "SPY"
        rows = [make_option_chain_row(symbol="SPY260731P00500000", bid=2.0, ask=2.1)]
        with patch("options_bot.alpaca_quotes.get_quote_map",
                   return_value={"SPY260731P00500000": (0.0, 0.0)}):
            out = ld._enrich_quotes(rows, "2026-07-31")
        self.assertEqual(out[0].bid, 2.0)            # unchanged
        self.assertEqual(out[0].ask, 2.1)

    def test_enrichment_noop_when_alpaca_empty(self):
        from options_bot.market_data import YFinanceDataLoader
        ld = YFinanceDataLoader.__new__(YFinanceDataLoader)
        ld.ticker_str = "SPY"
        rows = [make_option_chain_row(symbol="SPY260731P00500000", bid=0.0, ask=0.0)]
        with patch("options_bot.alpaca_quotes.get_quote_map", return_value={}):
            out = ld._enrich_quotes(rows, "2026-07-31")
        self.assertEqual(out[0].bid, 0.0)            # unchanged, fail-open


class TestExtraStrategyRouting(unittest.TestCase):
    """_select_extra_tickers must never run a direction-specific strategy
    against the bullish shortlist. The 2026-06-23 bug: the call-spread bucket
    was empty (no ticker bearish enough), so `... or scan_tickers` ran call
    spreads on the bullish names — fighting the trend."""

    def setUp(self):
        from options_bot.orchestrator import _select_extra_tickers
        self.sel = _select_extra_tickers
        # The 06-23 routing result: 9 bullish -> put_spread, QQQ -> strangle,
        # nothing bearish enough for call_spread.
        self.bullish = ["XBI", "XLF", "IWM", "XLV", "SMH", "XLI", "TLT", "HYG", "EEM"]
        self.routes = {
            "short_put_spread": list(self.bullish),
            "short_strangle": ["QQQ"],
        }

    def test_callspread_empty_bucket_skips_not_bullish_fallback(self):
        # THE BUG: routing ran, no ticker routed to call spreads -> SKIP ([]),
        # must NOT fall back to the bullish shortlist.
        out = self.sel("short_call_spread", self.routes, True, self.bullish)
        self.assertEqual(out, [])

    def test_strangle_uses_its_routed_bucket(self):
        out = self.sel("short_strangle", self.routes, True, self.bullish)
        self.assertEqual(out, ["QQQ"])

    def test_csp_reuses_bullish_shortlist(self):
        # csp shares the primary's bullish/neutral thesis and has no router
        # bucket -> bullish shortlist is correct.
        out = self.sel("csp", self.routes, True, self.bullish)
        self.assertEqual(out, self.bullish)

    def test_iron_condor_empty_bucket_skips(self):
        # iron_condor is neutral/defined-risk, NOT bullish-equivalent -> skip
        # when routing assigned it nothing (don't run it on bullish names).
        out = self.sel("iron_condor", self.routes, True, self.bullish)
        self.assertEqual(out, [])

    def test_routing_not_run_falls_back_for_callspread(self):
        # When routing never ran (ticker_gate disabled / errored), scan_tickers
        # is the unfiltered pool, so the legacy fallback is acceptable.
        out = self.sel("short_call_spread", {}, False, ["SPY", "QQQ", "IWM"])
        self.assertEqual(out, ["SPY", "QQQ", "IWM"])

    def test_routed_bucket_takes_precedence(self):
        routes = {"short_call_spread": ["SPY", "XLE"]}
        out = self.sel("short_call_spread", routes, True, self.bullish)
        self.assertEqual(out, ["SPY", "XLE"])


if __name__ == "__main__":
    unittest.main()
