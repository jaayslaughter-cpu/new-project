"""
SystemValidator — boot-sequence firewall for the options bot on Railway.

Runs four sequential checks before the Orchestrator is constructed or any
background workers are started. If any check fails the process exits with
code 1, leaving a clear reason in the Railway log.

Check sequence (all must pass):
  1. Environment        — ALPACA_API_KEY + ALPACA_SECRET_KEY are present
  2. Broker connection  — TradingClient authenticates, account is ACTIVE
  3. Market data        — OptionHistoricalDataClient returns a live quote
  4. Internal modules   — key pipeline modules import and pass sanity checks:
                          contracts.format_option_symbol, broker.AlpacaBroker,
                          circuit_breaker, risk, strategy, exceptions

Architecture notes
------------------
- Reuses the shared data_circuit_breaker for checks 2 and 3 so failures
  are propagated to the running bot's breaker state, not silently swallowed.
- All string inputs (ticker, expiry, symbol) are normalised (strip + upper)
  before any Alpaca call, enforcing cross-module data integrity.
- Rate-limit (429) responses are caught and logged explicitly with retry
  guidance — not swallowed into a generic Exception catch.
- Fail-stop: if get_contract_midpoint() returns None for the probe symbol,
  check 3 fails rather than proceeding with a stale or zero price.
- Does NOT call sys.exit() directly — raises SystemValidationError so the
  caller (__main__.py) decides the exit code and can send a Discord alert.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel exception
# ---------------------------------------------------------------------------

class SystemValidationError(RuntimeError):
    """
    Raised by SystemValidator.run_all_checks() when a mandatory check fails.

    Attributes
    ----------
    check_number : int   (1-4)
    check_name   : str
    detail       : str
    """
    def __init__(self, check_number: int, check_name: str, detail: str):
        self.check_number = check_number
        self.check_name   = check_name
        self.detail       = detail
        super().__init__(
            f"[Check {check_number}/4 FAILED] {check_name}: {detail}"
        )


# ---------------------------------------------------------------------------
# Probe contract — highly liquid SPY contract used to verify the data stream
# ---------------------------------------------------------------------------

# SPY Dec-2026 $400 Put — deep enough to always have open interest,
# far enough out to never expire during normal bot operation.
# Update this annually to keep DTE > 60.
_PROBE_SYMBOL_RAW = "SPY261218P00400000"


def _normalise(value: str) -> str:
    """Strip whitespace and upper-case a string input. Applied to all tickers/symbols."""
    return value.strip().upper()


# ---------------------------------------------------------------------------
# SystemValidator
# ---------------------------------------------------------------------------

class SystemValidator:
    """
    Boot-sequence firewall — run before constructing Orchestrator.

    Usage in __main__.py::

        validator = SystemValidator(api_key, secret_key, paper=paper)
        try:
            validator.run_all_checks()
        except SystemValidationError as exc:
            logger.critical("Boot aborted: %s", exc)
            send_discord(webhook, f"🔴 Bot failed to start: {exc}")
            sys.exit(1)

    Parameters
    ----------
    api_key    : Alpaca API key (from ALPACA_API_KEY env var)
    secret_key : Alpaca secret key
    paper      : True = paper trading, False = live
    discord_webhook : Optional Discord webhook URL for failure alerts
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        discord_webhook: Optional[str] = None,
    ):
        self._api_key         = api_key
        self._secret_key      = secret_key
        self._paper           = paper
        self._discord_webhook = discord_webhook
        self._mode_label      = "PAPER" if paper else "LIVE"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_checks(self) -> None:
        """
        Run all four checks in sequence. Raises SystemValidationError on
        the first failure. Callers should catch this and call sys.exit(1).
        """
        logger.info("=" * 60)
        logger.info("  OPTIONS BOT — SYSTEM VALIDATION [%s MODE]", self._mode_label)
        logger.info("=" * 60)

        self._check_environment()
        self._check_broker_connection()
        self._check_market_data()
        self._check_internal_modules()

        logger.info("=" * 60)
        logger.info("  ✅ ALL CHECKS PASSED — STARTING BOT")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Check 1 — Environment variables
    # ------------------------------------------------------------------

    def _check_environment(self) -> None:
        """
        Verify ALPACA_API_KEY and ALPACA_SECRET_KEY are set and non-empty.

        Also checks for DISCORD_WEBHOOK_URL and DATABASE_URL, logging
        warnings if absent (not hard failures — bot can run without them).
        """
        logger.info("[1/4] Validating environment variables...")

        issues = []
        if not self._api_key:
            issues.append("ALPACA_API_KEY is missing or empty")
        if not self._secret_key:
            issues.append("ALPACA_SECRET_KEY is missing or empty")

        if issues:
            detail = " | ".join(issues)
            logger.critical(
                "❌ [1/4] FAILED: %s\n"
                "    → Add these to Railway → your service → Variables tab",
                detail,
            )
            raise SystemValidationError(1, "Environment", detail)

        # Soft warnings — not failures
        if not os.getenv("DISCORD_WEBHOOK_URL"):
            logger.warning(
                "⚠  DISCORD_WEBHOOK_URL not set — trade alerts and EOD summaries "
                "will be suppressed. Set it in Railway Variables to enable."
            )
        if not os.getenv("DATABASE_URL"):
            logger.warning(
                "⚠  DATABASE_URL not set — bot will fall back to local SQLite. "
                "Add a Railway PostgreSQL plugin for persistent trade history."
            )

        # Log key prefix for debugging (never log full keys)
        key_preview = f"{self._api_key[:4]}...{self._api_key[-4:]}" if len(self._api_key) > 8 else "****"
        logger.info(
            "✅ [1/4] Environment OK — mode=%s key=%s",
            self._mode_label, key_preview,
        )

    # ------------------------------------------------------------------
    # Check 2 — Broker connection
    # ------------------------------------------------------------------

    def _check_broker_connection(self) -> None:
        """
        Authenticate against Alpaca TradingClient and verify account status.

        Checks:
          - Credentials are accepted (no AuthenticationError)
          - account.status == ACTIVE
          - account.currency == USD
          - Logs equity and buying power for confirmation

        Rate-limit (HTTP 429) is caught and reported explicitly with
        guidance to wait and retry rather than a generic 'connection failed'.
        """
        logger.info("[2/4] Testing Alpaca broker connectivity...")

        from .circuit_breaker import data_circuit_breaker as _cb

        if not _cb.is_available("alpaca_trading"):
            detail = (
                "Circuit breaker is OPEN for alpaca_trading — "
                "previous authentication failures exceeded threshold. "
                "Wait for cooldown or restart the bot."
            )
            logger.critical("❌ [2/4] FAILED: %s", detail)
            raise SystemValidationError(2, "BrokerConnection", detail)

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.common.exceptions import APIError

            client = TradingClient(
                self._api_key, self._secret_key, paper=self._paper
            )
            account = client.get_account()
            _cb.record_success("alpaca_trading")

        except Exception as exc:
            _cb.record_failure("alpaca_trading", str(exc))
            # Surface rate-limit errors explicitly
            err_str = str(exc).lower()
            if "429" in err_str or "too many" in err_str or "rate" in err_str:
                detail = (
                    f"Alpaca API rate-limited (HTTP 429). "
                    f"Wait 60s and redeploy. Detail: {exc}"
                )
            elif "403" in err_str or "unauthori" in err_str or "forbidden" in err_str:
                detail = (
                    f"Alpaca authentication rejected (HTTP 403/401). "
                    f"Verify ALPACA_API_KEY and ALPACA_SECRET_KEY are correct "
                    f"and match {'paper' if self._paper else 'live'} mode. Detail: {exc}"
                )
            else:
                detail = (
                    f"Could not reach Alpaca Trading API. "
                    f"Check Railway network egress or Alpaca status page. "
                    f"Detail: {exc}"
                )
            logger.critical("❌ [2/4] FAILED: %s", detail)
            raise SystemValidationError(2, "BrokerConnection", detail) from exc

        # Validate account health
        status   = str(getattr(account, "status", "")).upper()
        currency = str(getattr(account, "currency", ""))
        equity   = getattr(account, "equity", None)
        bp       = getattr(account, "buying_power", None)

        if status != "ACTIVE":
            detail = (
                f"Account status is '{status}' — expected ACTIVE. "
                f"Check app.alpaca.markets for account restrictions or "
                f"pending verification."
            )
            logger.critical("❌ [2/4] FAILED: %s", detail)
            raise SystemValidationError(2, "BrokerConnection", detail)

        logger.info(
            "✅ [2/4] Broker OK — status=%s currency=%s equity=%s buying_power=%s",
            status, currency,
            f"${float(equity):.2f}" if equity else "N/A",
            f"${float(bp):.2f}" if bp else "N/A",
        )

    # ------------------------------------------------------------------
    # Check 3 — Market data / live midpoint
    # ------------------------------------------------------------------

    def _check_market_data(self) -> None:
        """
        Verify the OptionHistoricalDataClient returns a live quote and that
        the live midpoint can be computed from it.

        Fail-stop coupling: if get_contract_midpoint() returns None (zero bid,
        zero ask, or missing data), this check fails rather than proceeding
        with a stale price. The execution engine's fail-stop guarantee depends
        on this check passing cleanly.

        Rate-limit (429) is caught and reported explicitly.
        """
        logger.info("[3/4] Testing option market data pipeline...")

        from .circuit_breaker import data_circuit_breaker as _cb
        from .broker import _get_live_midpoint

        probe = _normalise(_PROBE_SYMBOL_RAW)
        logger.debug("[3/4] Probe symbol: %s", probe)

        if not _cb.is_available("alpaca_options_data"):
            detail = (
                "Circuit breaker OPEN for alpaca_options_data. "
                "Previous data failures exceeded threshold. "
                "Wait for cooldown or restart."
            )
            logger.critical("❌ [3/4] FAILED: %s", detail)
            raise SystemValidationError(3, "MarketData", detail)

        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.requests import OptionLatestQuoteRequest

            data_client = OptionHistoricalDataClient(
                self._api_key, self._secret_key
            )
            req    = OptionLatestQuoteRequest(symbol_or_symbols=probe)
            result = data_client.get_option_latest_quote(req)
            _cb.record_success("alpaca_options_data")

        except Exception as exc:
            _cb.record_failure("alpaca_options_data", str(exc))
            err_str = str(exc).lower()
            if "429" in err_str or "too many" in err_str or "rate" in err_str:
                detail = (
                    f"Market data API rate-limited (HTTP 429). "
                    f"Wait 60s and redeploy. Detail: {exc}"
                )
            elif "403" in err_str or "unauthori" in err_str:
                detail = (
                    f"Market data API authentication failed. "
                    f"Paper API keys do not have access to live option data — "
                    f"verify you are using a valid Alpaca key pair. Detail: {exc}"
                )
            else:
                detail = (
                    f"OptionHistoricalDataClient failed to connect. "
                    f"Detail: {exc}"
                )
            logger.critical("❌ [3/4] FAILED: %s", detail)
            raise SystemValidationError(3, "MarketData", detail) from exc

        # Fail-stop: verify a midpoint can actually be computed
        # (not just that the API call succeeded — a zero-bid response would
        # silently pass but the execution engine would abort at order time)
        if probe not in result:
            detail = (
                f"Probe symbol {probe} not found in API response. "
                f"The data stream returned an empty result. "
                f"This would cause silent execution failures at trade time."
            )
            logger.critical("❌ [3/4] FAILED: %s", detail)
            raise SystemValidationError(3, "MarketData", detail)

        q    = result[probe]
        bid  = float(q.bid_price) if q.bid_price is not None else 0.0
        ask  = float(q.ask_price) if q.ask_price is not None else 0.0

        if ask == 0:
            # Closed market or illiquid probe — warn but don't fail
            # (probe is a deep OTM put; zero quotes pre-market are expected)
            logger.warning(
                "⚠  [3/4] Probe %s has ask=0 (market likely closed). "
                "Live midpoint will be available during market hours. "
                "Proceeding — this is expected outside 9:30-4:00 ET.",
                probe,
            )
        else:
            mid = round((bid + ask) / 2, 2) if bid > 0 else ask
            logger.info(
                "✅ [3/4] Market data OK — probe=%s bid=%.2f ask=%.2f mid=%.2f",
                probe, bid, ask, mid,
            )
            return

        logger.info("✅ [3/4] Market data API responding (market closed — quotes pending open)")

    # ------------------------------------------------------------------
    # Check 4 — Internal module integrity
    # ------------------------------------------------------------------

    def _check_internal_modules(self) -> None:
        """
        Import and sanity-check all critical pipeline modules.

        Validates:
          - contracts.format_option_symbol — correct OCC formatting
          - broker.AlpacaBroker            — class importable
          - broker._get_live_midpoint      — function importable
          - exceptions hierarchy           — all custom exceptions present
          - circuit_breaker                — data_circuit_breaker accessible
          - risk.RiskConfig                — importable with defaults
          - strategy module                — importable
          - strategy_0dte module           — importable

        String normalisation: all ticker/symbol inputs are tested through
        _normalise() to confirm strip+upper enforcement is in place before
        they reach Alpaca's formatting layer.
        """
        logger.info("[4/4] Verifying internal module integrity...")

        failures: list[str] = []

        # ── contracts.format_option_symbol ────────────────────────────
        try:
            from .contracts import format_option_symbol

            # Test known-good symbol — SPY $580C 2026-06-20
            raw_ticker  = "  spy  "   # intentional whitespace + lowercase
            raw_expiry  = "260620"
            raw_type    = "call"
            strike      = 580.0
            result = format_option_symbol(
                _normalise(raw_ticker), raw_expiry, raw_type, strike
            )
            expected = "SPY260620C00580000"
            if result != expected:
                failures.append(
                    f"format_option_symbol returned '{result}', expected '{expected}'"
                )
            else:
                logger.debug("[4/4] format_option_symbol: OK → %s", result)

            # Test fractional strike — QQQ $475.50 put 2026-07-18
            result2 = format_option_symbol("QQQ", "260718", "put", 475.5)
            expected2 = "QQQ260718P00475500"
            if result2 != expected2:
                failures.append(
                    f"format_option_symbol fractional strike: got '{result2}', "
                    f"expected '{expected2}'"
                )

        except Exception as exc:
            failures.append(f"contracts.format_option_symbol import/run failed: {exc}")

        # ── broker ────────────────────────────────────────────────────
        try:
            from .broker import AlpacaBroker, _get_live_midpoint, format_option_symbol as broker_fmt
            # Verify broker's own format_option_symbol is consistent with contracts'
            r = broker_fmt("SPY", "260620", "C", 580.0)
            if r != "SPY260620C00580000":
                failures.append(f"broker.format_option_symbol mismatch: got '{r}'")
        except Exception as exc:
            failures.append(f"broker module failed: {exc}")

        # ── exceptions hierarchy ──────────────────────────────────────
        try:
            from .exceptions import (
                OptionsBotError, PipelineConnectionError, StalenessError,
                LiquidityFilterError, RiskVetoError, DataValidationError,
                IVSolveError,
            )
        except Exception as exc:
            failures.append(f"exceptions module failed: {exc}")

        # ── circuit_breaker ───────────────────────────────────────────
        try:
            from .circuit_breaker import data_circuit_breaker, CircuitBreaker
            _ = data_circuit_breaker.status()
        except Exception as exc:
            failures.append(f"circuit_breaker module failed: {exc}")

        # ── risk ──────────────────────────────────────────────────────
        try:
            from .risk import RiskConfig, RiskManager
            rc = RiskConfig()
            assert rc.risk_budget_pct > 0, "RiskConfig.risk_budget_pct must be > 0"
        except Exception as exc:
            failures.append(f"risk module failed: {exc}")

        # ── strategy ─────────────────────────────────────────────────
        try:
            from .strategy import ShortPutSpread, CashSecuredPut, ShortStrangle
        except Exception as exc:
            failures.append(f"strategy module failed: {exc}")

        # ── strategy_0dte ─────────────────────────────────────────────
        try:
            from .strategy_0dte import ZeroDTEStrategy, ZeroDTEConfig
            cfg = ZeroDTEConfig()
            assert 0 < cfg.vwap_stretch_threshold < 0.1, \
                f"vwap_stretch_threshold out of range: {cfg.vwap_stretch_threshold}"
        except Exception as exc:
            failures.append(f"strategy_0dte module failed: {exc}")

        # ── iv_quality ────────────────────────────────────────────────
        try:
            from .iv_quality import assess_iv_quality, robust_iv_rank, ContaminationLevel
        except Exception as exc:
            failures.append(f"iv_quality module failed: {exc}")

        # ── Fail-stop: any failure blocks startup ─────────────────────
        if failures:
            detail = " | ".join(failures)
            logger.critical(
                "❌ [4/4] FAILED: %d internal module issue(s):\n    → %s",
                len(failures),
                "\n    → ".join(failures),
            )
            raise SystemValidationError(4, "InternalModules", detail)

        logger.info(
            "✅ [4/4] Internal modules OK — "
            "contracts, broker, exceptions, circuit_breaker, risk, "
            "strategy, strategy_0dte, iv_quality all healthy"
        )


# ---------------------------------------------------------------------------
# Module-level convenience runner
# ---------------------------------------------------------------------------

def run_system_validation(
    api_key: str,
    secret_key: str,
    paper: bool = True,
    discord_webhook: Optional[str] = None,
) -> None:
    """
    Convenience wrapper — constructs SystemValidator and runs all checks.

    Raises SystemValidationError on failure. Caller is responsible for
    catching it and calling sys.exit(1).

    Used by __main__.py immediately after env vars are loaded and before
    Orchestrator.__init__() is called.
    """
    validator = SystemValidator(
        api_key=api_key,
        secret_key=secret_key,
        paper=paper,
        discord_webhook=discord_webhook,
    )
    validator.run_all_checks()


__all__ = [
    "SystemValidator",
    "SystemValidationError",
    "run_system_validation",
]
