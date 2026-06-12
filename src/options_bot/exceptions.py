"""
Custom exception hierarchy for the options bot pipeline.

Every module boundary raises a named exception from this module rather than
a generic Exception or silent failure. This satisfies the PipelineConnectionError
mandate: if Module A fails to pass data to Module B, the system throws a specific,
trackable error rather than failing silently.

Zero-hallucination rule: if data is missing, raise StalenessError or
LiquidityFilterError — never estimate or fill with a default.
"""


class OptionsBotError(RuntimeError):
    """Base class for all options bot pipeline errors."""
    pass


class PipelineConnectionError(OptionsBotError):
    """
    Raised when a module fails to receive or pass data to the next stage.

    Example: market data module returns None, Greeks layer cannot proceed.
    """
    pass


class StalenessError(OptionsBotError):
    """
    Raised when data is too old to use safely.

    Default threshold: 60 seconds for live quotes, 24 hours for EOD data.
    Never extrapolate from stale data — raise this and let the caller decide.
    """
    def __init__(self, field: str, age_seconds: float, max_age_seconds: float):
        self.field = field
        self.age_seconds = age_seconds
        self.max_age_seconds = max_age_seconds
        super().__init__(
            f"Stale data: '{field}' is {age_seconds:.0f}s old "
            f"(max allowed: {max_age_seconds:.0f}s)"
        )


class LiquidityFilterError(OptionsBotError):
    """
    Raised when a contract fails liquidity requirements.

    Filters enforced:
      - open_interest < min_open_interest
      - bid_ask_spread_pct > max_spread_pct
      - bid or ask is None/zero
      - volume == 0 with no open interest

    Never suggest or execute trades on illiquid contracts.
    """
    def __init__(self, symbol: str, reason: str):
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Liquidity filter rejected '{symbol}': {reason}")


class RiskVetoError(OptionsBotError):
    """
    Raised when an order is blocked by the risk manager.

    Hard stops:
      - No stop-loss defined (naked position)
      - Position size exceeds max risk % of equity
      - Daily loss limit already hit
      - Max daily trade count reached

    Every generated order payload must include a defined stop-loss.
    Never suggest an open-ended, unhedged short options position.
    """
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Risk veto: {reason}")


class DataValidationError(OptionsBotError):
    """
    Raised when incoming data fails schema or type validation.

    Per zero-hallucination policy: if a specific data point is missing,
    corrupted, or unavailable, output an error — never estimate or guess.
    """
    def __init__(self, field: str, issue: str):
        self.field = field
        self.issue = issue
        super().__init__(f"Data validation failed for '{field}': {issue}")


class IVSolveError(OptionsBotError):
    """
    Raised when implied volatility cannot be solved for a contract.

    Per zero-hallucination policy: if IV solve fails, set iv=None and
    raise this error. Never use a fallback estimate for IV.
    """
    def __init__(self, symbol: str, reason: str):
        self.symbol = symbol
        super().__init__(f"IV solve failed for '{symbol}': {reason}")
