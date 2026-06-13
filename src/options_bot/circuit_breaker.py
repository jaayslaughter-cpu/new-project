"""
Circuit breaker — data source failure management.

Prevents the bot from hammering failed external endpoints (yfinance, FRED,
Alpaca data) by tracking consecutive failures and enforcing a cooldown period.

State machine:
  CLOSED   → normal, requests pass through
  OPEN     → failed N times, requests blocked until cooldown expires
  HALF_OPEN → cooldown expired, one probe request allowed
               success → CLOSED | failure → OPEN again

Extracted and rewritten from QuantDinger's circuit_breaker.py
(original concept, rewritten without Chinese comments, typed with dataclasses).

Usage:
    from options_bot.circuit_breaker import CircuitBreaker

    _cb = CircuitBreaker()   # one instance, shared across all data fetches

    if _cb.is_available("yfinance"):
        try:
            result = yf.Ticker("SPY").fast_info
            _cb.record_success("yfinance")
        except Exception as e:
            _cb.record_failure("yfinance", str(e))

Default thresholds (configurable in constructor):
  failure_threshold  = 3 consecutive failures → OPEN
  cooldown_seconds   = 300s (5 min) before HALF_OPEN probe
  half_open_max_calls = 1 probe request before deciding recovery
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED    = "closed"     # normal — requests pass through
    OPEN      = "open"       # tripped — requests blocked
    HALF_OPEN = "half_open"  # probing — one request allowed


@dataclass
class _SourceState:
    state:           CircuitState = CircuitState.CLOSED
    failures:        int          = 0
    last_failure_at: float        = 0.0
    half_open_calls: int          = 0
    last_error:      Optional[str] = None


class CircuitBreaker:
    """
    Per-source circuit breaker for external data dependencies.

    One instance can track multiple named sources independently.
    Thread-safe for read operations; writes use Python's GIL.

    Parameters
    ----------
    failure_threshold : int
        Consecutive failures before tripping (default 3).
    cooldown_seconds : float
        Seconds to wait in OPEN before allowing a probe (default 300).
    half_open_max_calls : int
        Max probe requests in HALF_OPEN before forcing a decision (default 1).
    """

    def __init__(
        self,
        failure_threshold: int   = 3,
        cooldown_seconds:  float = 300.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold   = failure_threshold
        self.cooldown_seconds    = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls
        self._states: Dict[str, _SourceState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self, source: str) -> bool:
        """
        Returns True if a request to `source` should be attempted.
        Call this before every external fetch.
        """
        s = self._get(source)
        now = time.monotonic()

        if s.state == CircuitState.CLOSED:
            return True

        if s.state == CircuitState.OPEN:
            elapsed = now - s.last_failure_at
            if elapsed >= self.cooldown_seconds:
                s.state           = CircuitState.HALF_OPEN
                s.half_open_calls = 0
                logger.info("[CB] %s: cooldown expired → HALF_OPEN (probe allowed)", source)
                return True
            remaining = self.cooldown_seconds - elapsed
            logger.debug("[CB] %s: OPEN, %.0fs remaining in cooldown", source, remaining)
            return False

        # HALF_OPEN
        if s.half_open_calls < self.half_open_max_calls:
            return True
        logger.debug("[CB] %s: HALF_OPEN probe budget exhausted", source)
        return False

    def record_success(self, source: str) -> None:
        """Call after a successful request. Resets the source to CLOSED."""
        s = self._get(source)
        if s.state != CircuitState.CLOSED:
            logger.info("[CB] %s: success → CLOSED (was %s)", source, s.state.value)
        s.state           = CircuitState.CLOSED
        s.failures        = 0
        s.half_open_calls = 0
        s.last_error      = None

    def record_failure(self, source: str, error: Optional[str] = None) -> None:
        """Call after a failed request. May trip the breaker to OPEN."""
        s   = self._get(source)
        now = time.monotonic()

        s.failures       += 1
        s.last_failure_at = now
        s.last_error      = error

        if s.state == CircuitState.HALF_OPEN:
            s.state           = CircuitState.OPEN
            s.half_open_calls = 0
            logger.warning(
                "[CB] %s: HALF_OPEN probe failed → OPEN (cooldown %.0fs) — %s",
                source, self.cooldown_seconds, error or ""
            )
        elif s.failures >= self.failure_threshold:
            s.state = CircuitState.OPEN
            logger.warning(
                "[CB] %s: %d consecutive failures → OPEN (cooldown %.0fs) — %s",
                source, s.failures, self.cooldown_seconds, error or ""
            )
        else:
            logger.debug(
                "[CB] %s: failure %d/%d — %s",
                source, s.failures, self.failure_threshold, error or ""
            )

    def status(self) -> dict:
        """Return a snapshot of all source states for logging/Discord."""
        return {
            src: {
                "state":    info.state.value,
                "failures": info.failures,
                "error":    info.last_error,
            }
            for src, info in self._states.items()
        }

    def reset(self, source: Optional[str] = None) -> None:
        """Manually reset one source or all sources."""
        if source:
            self._states.pop(source, None)
            logger.info("[CB] %s: manually reset", source)
        else:
            self._states.clear()
            logger.info("[CB] all sources reset")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get(self, source: str) -> _SourceState:
        if source not in self._states:
            self._states[source] = _SourceState()
        return self._states[source]


# ---------------------------------------------------------------------------
# Module-level shared instance — import and use directly
# ---------------------------------------------------------------------------

#: Shared circuit breaker used by market_data.py and regime.py.
#: Stricter thresholds for real-time data: 2 failures → 3-min cooldown.
data_circuit_breaker = CircuitBreaker(
    failure_threshold   = 2,
    cooldown_seconds    = 180.0,
    half_open_max_calls = 1,
)
