"""
Alpaca broker adapter.

Built against the official Alpaca OpenAPI spec (trading-api.json +
market-data-api.json) included in the alpaca-cli repository.

Key spec-verified facts:
  Trading base URL:    https://paper-api.alpaca.markets/v2  (paper)
                       https://api.alpaca.markets/v2        (live)
  Options TIF:         day only (gtc not supported for options)
  Options order class: simple or mleg (bracket/oco/oto are equity-only)
  mleg limit_price:    negative = credit received, positive = debit paid
  mleg stop_loss:      NOT supported — must manage stops via separate orders
  Snapshot Greeks:     delta, gamma, theta, vega, rho  (all from Black-Scholes)
  Snapshot quote keys: bp=bid_price, ap=ask_price, bs=bid_size, as=ask_size
  Option chain URL:    GET /v1beta1/options/snapshots/{underlying_symbol}
  Chain filter params: type, strike_price_gte, strike_price_lte,
                       expiration_date, expiration_date_gte, expiration_date_lte,
                       root_symbol

Single-leg order flow:
  POST /v2/orders  {symbol, qty, position_intent, type=limit,
                    time_in_force=day, order_class=simple,
                    limit_price, stop_loss: {stop_price}}

Multi-leg order flow (spreads, strangles):
  POST /v2/orders  {qty, type=limit, time_in_force=day, order_class=mleg,
                    limit_price (negative=credit),
                    legs: [{symbol, ratio_qty, position_intent}]}
  NOTE: stop_loss is NOT supported on mleg orders per spec.
        Stops on spreads must be managed via separate monitoring + close orders.

Position exercise/DNE:
  POST /v2/positions/{symbol}/exercise
  POST /v2/positions/{symbol}/do-not-exercise
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from .contracts import ApprovedOrder, FilledOrder
from .exceptions import PipelineConnectionError, RiskVetoError

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT_BUFFER_PCT = 0.02   # 2% buffer on limit price to improve fill rate


class AlpacaBroker:
    """
    Alpaca broker adapter for options order submission.

    Reads credentials from environment variables:
      ALPACA_API_KEY     — Alpaca API key
      ALPACA_SECRET_KEY  — Alpaca secret key
      ALPACA_PAPER       — "true" (default) or "false"

    Never hardcode credentials. The constructor raises PipelineConnectionError
    if ALPACA_API_KEY or ALPACA_SECRET_KEY are not set.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
    ):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.option import OptionHistoricalDataClient
        except ImportError:
            raise PipelineConnectionError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            )

        resolved_key = api_key or os.getenv("ALPACA_API_KEY")
        resolved_secret = secret_key or os.getenv("ALPACA_SECRET_KEY")
        if not resolved_key or not resolved_secret:
            raise PipelineConnectionError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables."
            )

        # Safety default: paper=True unless explicitly set to false
        if paper is None:
            paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"

        self._paper = paper
        self._trading = TradingClient(
            api_key=resolved_key, secret_key=resolved_secret, paper=paper
        )
        self._data = OptionHistoricalDataClient(
            api_key=resolved_key, secret_key=resolved_secret
        )

        mode = "PAPER" if paper else "LIVE"
        logger.info("[AlpacaBroker] Initialized — %s mode", mode)
        if not paper:
            logger.warning(
                "[AlpacaBroker] LIVE mode: real money at risk. "
                "Verify all risk parameters before submitting orders."
            )

    @property
    def is_paper(self) -> bool:
        return self._paper

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """GET /v2/account — returns key account fields."""
        try:
            a = self._trading.get_account()
            return {
                "equity":            float(a.equity),
                "buying_power":      float(a.buying_power),
                "cash":              float(a.cash),
                "portfolio_value":   float(a.portfolio_value),
                "trading_blocked":   bool(a.trading_blocked),
                "account_blocked":   bool(a.account_blocked),
                "pattern_day_trader": bool(a.pattern_day_trader),
            }
        except Exception as exc:
            raise PipelineConnectionError(f"GET /v2/account failed: {exc}") from exc

    def get_equity(self) -> float:
        return self.get_account()["equity"]

    def get_buying_power(self) -> float:
        return self.get_account()["buying_power"]

    def check_account_ready(self) -> None:
        """Raises if account is blocked. Call before each trading session."""
        info = self.get_account()
        if info["trading_blocked"]:
            raise PipelineConnectionError(
                "Account trading_blocked=True. Check Alpaca dashboard."
            )
        if info["account_blocked"]:
            raise PipelineConnectionError(
                "Account account_blocked=True. Contact Alpaca support."
            )
        logger.info(
            "[AlpacaBroker] Account OK — equity=$%.2f buying_power=$%.2f",
            info["equity"], info["buying_power"]
        )

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit(self, order: ApprovedOrder) -> FilledOrder:
        """
        Submit an ApprovedOrder to Alpaca.

        Runs ExecutionGuard (raises RiskVetoError on any violation) then
        routes to single-leg or multi-leg submission.

        Returns a FilledOrder with Alpaca order ID. Fill price is an estimate
        until confirmed — poll wait_for_fill() to get actual avg fill price.
        """
        from .risk import ExecutionGuard
        ExecutionGuard.check(order)

        logger.info(
            "[AlpacaBroker] Submitting %s %s — %d contracts "
            "net=%.2f stop=%.2f max_loss=$%.2f",
            order.underlying, order.strategy_name,
            order.position_size_contracts,
            order.net_debit_credit,
            order.hard_stop_price,
            order.max_loss_dollars,
        )

        if len(order.legs) == 1:
            return self._submit_single(order)
        else:
            return self._submit_mleg(order)

    def _submit_single(self, order: ApprovedOrder) -> FilledOrder:
        """
        Single-leg option order.

        Spec: POST /v2/orders
          order_class = simple
          type        = limit
          time_in_force = day  (only valid TIF for options per spec)
          position_intent = sell_to_open | buy_to_open | etc.
          stop_loss.stop_price = hard stop level (exchange-managed)
        """
        from alpaca.trading.requests import LimitOrderRequest, StopLossRequest
        from alpaca.trading.enums import TimeInForce, OrderClass

        leg = order.legs[0]
        qty = order.position_size_contracts
        limit_price = _limit_price_single(order.estimated_fill_price, leg.side)

        stop = StopLossRequest(stop_price=round(order.hard_stop_price, 2))

        client_id = f"optbot-{order.strategy_name}-{uuid.uuid4().hex[:8]}"

        req = LimitOrderRequest(
            symbol=leg.symbol,
            qty=qty,
            position_intent=_position_intent(leg.side),
            time_in_force=TimeInForce.DAY,   # only valid TIF for options
            order_class=OrderClass.SIMPLE,
            limit_price=round(limit_price, 2),
            stop_loss=stop,
            client_order_id=client_id,
        )
        logger.debug(
            "[AlpacaBroker] single-leg: %s qty=%d limit=%.2f stop=%.2f",
            leg.symbol, qty, limit_price, order.hard_stop_price
        )
        return self._post_order(req, order, client_id)

    def _submit_mleg(self, order: ApprovedOrder) -> FilledOrder:
        """
        Multi-leg option order (spreads, strangles).

        Spec: POST /v2/orders
          order_class  = mleg
          type         = limit
          time_in_force = day
          qty          = number of spread units
          limit_price  = net price (NEGATIVE = credit per spec)
          legs         = [{symbol, ratio_qty, position_intent}]

        IMPORTANT per spec: stop_loss is NOT supported on mleg orders.
        Stops must be managed externally — monitor positions and submit
        separate closing orders when stop level is reached.
        """
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
        from alpaca.trading.enums import TimeInForce, OrderClass

        qty = order.position_size_contracts

        # Build legs — ratio_qty=1 for standard 1:1 spreads
        alpaca_legs = [
            OptionLegRequest(
                symbol=leg.symbol,
                ratio_qty=1,
                position_intent=_position_intent(leg.side),
            )
            for leg in order.legs
        ]

        # mleg limit_price: negative = credit (spec-verified)
        # net_debit_credit is already negative for credit strategies
        net = order.net_debit_credit
        # Buffer: accept slightly less credit / pay slightly more debit
        if net < 0:
            limit_price = net * (1 - _DEFAULT_LIMIT_BUFFER_PCT)   # e.g. -1.50 → -1.47
        else:
            limit_price = net * (1 + _DEFAULT_LIMIT_BUFFER_PCT)

        client_id = f"optbot-{order.strategy_name}-{uuid.uuid4().hex[:8]}"

        req = LimitOrderRequest(
            qty=qty,
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
            legs=alpaca_legs,
            client_order_id=client_id,
        )

        logger.debug(
            "[AlpacaBroker] mleg: %d legs qty=%d limit_price=%.2f (credit=%.2f)",
            len(alpaca_legs), qty, limit_price, abs(net)
        )
        logger.info(
            "[AlpacaBroker] NOTE: mleg stop_loss not supported by API. "
            "Monitor position and submit closing order at stop level %.2f manually.",
            order.hard_stop_price
        )
        return self._post_order(req, order, client_id)

    def _post_order(self, req, order: ApprovedOrder, client_id: str) -> FilledOrder:
        """POST /v2/orders — shared submission path."""
        submit_time = datetime.now(tz=timezone.utc)
        try:
            resp = self._trading.submit_order(order_data=req)
        except Exception as exc:
            raise PipelineConnectionError(
                f"POST /v2/orders failed: {exc}"
            ) from exc

        order_id = str(resp.id)
        status   = str(resp.status)
        logger.info(
            "[AlpacaBroker] Order submitted: id=%s status=%s client_id=%s",
            order_id, status, client_id
        )
        return FilledOrder(
            order_id=order_id,
            approved_order=order,
            fill_price=order.estimated_fill_price,   # estimated until confirmed
            fill_timestamp=submit_time,
            slippage_actual=0.0,
            status="open",
            broker="alpaca",
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def get_order_status(self, order_id: str) -> dict:
        """GET /v2/orders/{order_id}"""
        try:
            o = self._trading.get_order_by_id(order_id)
            return {
                "id":                str(o.id),
                "status":            str(o.status),
                "filled_qty":        float(o.filled_qty or 0),
                "filled_avg_price":  float(o.filled_avg_price or 0),
                "client_order_id":   str(o.client_order_id or ""),
            }
        except Exception as exc:
            raise PipelineConnectionError(
                f"GET /v2/orders/{order_id} failed: {exc}"
            ) from exc

    def wait_for_fill(
        self,
        order_id: str,
        timeout_seconds: float = 60.0,
        poll_interval: float = 2.0,
    ) -> dict:
        """Poll until filled, cancelled, expired, or rejected."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            s = self.get_order_status(order_id)
            status = s["status"]
            if status in ("filled", "partially_filled"):
                logger.info(
                    "[AlpacaBroker] Filled: id=%s qty=%.0f avg_price=%.4f",
                    order_id, s["filled_qty"], s["filled_avg_price"]
                )
                return s
            if status in ("cancelled", "expired", "rejected"):
                logger.warning("[AlpacaBroker] Order %s ended: %s", order_id, status)
                return s
            time.sleep(poll_interval)

        raise PipelineConnectionError(
            f"Order {order_id} did not fill within {timeout_seconds:.0f}s"
        )

    def cancel_order(self, order_id: str) -> None:
        """DELETE /v2/orders/{order_id}"""
        try:
            self._trading.cancel_order_by_id(order_id)
            logger.info("[AlpacaBroker] Cancelled: %s", order_id)
        except Exception as exc:
            raise PipelineConnectionError(
                f"DELETE /v2/orders/{order_id} failed: {exc}"
            ) from exc

    def cancel_all_orders(self) -> int:
        """DELETE /v2/orders — cancel all open orders."""
        try:
            cancelled = self._trading.cancel_orders()
            count = len(cancelled) if cancelled else 0
            logger.info("[AlpacaBroker] Cancelled %d orders", count)
            return count
        except Exception as exc:
            raise PipelineConnectionError(
                f"DELETE /v2/orders failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """GET /v2/positions"""
        try:
            positions = self._trading.get_all_positions()
            return [
                {
                    "symbol":          pos.symbol,
                    "qty":             float(pos.qty),
                    "side":            str(pos.side),
                    "market_value":    float(pos.market_value or 0),
                    "unrealized_pl":   float(pos.unrealized_pl or 0),
                    "unrealized_plpc": float(pos.unrealized_plpc or 0),
                    "avg_entry_price": float(pos.avg_entry_price or 0),
                    "current_price":   float(pos.current_price or 0),
                    "asset_class":     str(pos.asset_class),
                }
                for pos in positions
            ]
        except Exception as exc:
            raise PipelineConnectionError(
                f"GET /v2/positions failed: {exc}"
            ) from exc

    def close_position(self, symbol: str, qty: Optional[int] = None) -> None:
        """DELETE /v2/positions/{symbol_or_asset_id}"""
        try:
            if qty is not None:
                self._trading.close_position(symbol, qty=qty)
            else:
                self._trading.close_position(symbol)
            logger.info("[AlpacaBroker] Closed position: %s qty=%s", symbol, qty)
        except Exception as exc:
            raise PipelineConnectionError(
                f"DELETE /v2/positions/{symbol} failed: {exc}"
            ) from exc

    def exercise_option(self, symbol: str) -> None:
        """POST /v2/positions/{symbol}/exercise"""
        try:
            self._trading.exercise_option(symbol_or_contract_id=symbol)
            logger.info("[AlpacaBroker] Exercised: %s", symbol)
        except Exception as exc:
            raise PipelineConnectionError(
                f"POST /v2/positions/{symbol}/exercise failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_option_chain(
        self,
        underlying_symbol: str,
        option_type: Optional[str] = None,          # "call" or "put"
        expiration_date: Optional[str] = None,      # "YYYY-MM-DD"
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
    ) -> dict:
        """
        GET /v1beta1/options/snapshots/{underlying_symbol}

        Returns dict: symbol → snapshot with keys:
          greeks:            {delta, gamma, theta, vega, rho}
          impliedVolatility: float
          latestQuote:       {bp, ap, bs, as, t, ...}
          latestTrade:       {p, s, t, x, ...}
          dailyBar:          {o, h, l, c, v, vw, ...}
        """
        from alpaca.data.requests import OptionChainRequest
        try:
            req = OptionChainRequest(
                underlying_symbol=underlying_symbol,
                type=option_type,
                expiration_date=expiration_date,
                expiration_date_gte=expiration_date_gte,
                expiration_date_lte=expiration_date_lte,
                strike_price_gte=strike_price_gte,
                strike_price_lte=strike_price_lte,
            )
            chain = self._data.get_option_chain(req)
            return _parse_chain(chain)
        except Exception as exc:
            raise PipelineConnectionError(
                f"GET /v1beta1/options/snapshots/{underlying_symbol} failed: {exc}"
            ) from exc

    def get_option_snapshots(self, symbols: list[str]) -> dict:
        """
        GET /v1beta1/options/snapshots?symbols=...

        Returns dict: symbol → snapshot (same schema as get_option_chain).
        Use for marking-to-market specific known symbols.
        """
        from alpaca.data.requests import OptionSnapshotRequest
        try:
            req = OptionSnapshotRequest(symbol_or_symbols=symbols)
            snaps = self._data.get_option_snapshot(req)
            return _parse_snapshots(snaps)
        except Exception as exc:
            raise PipelineConnectionError(
                f"GET /v1beta1/options/snapshots failed: {exc}"
            ) from exc

    def get_latest_quotes(self, symbols: list[str]) -> dict:
        """
        GET /v1beta1/options/quotes/latest

        Returns dict: symbol → {bp, ap, bs, as, t}
        Quote fields per spec: bp=bid_price, ap=ask_price, bs=bid_size, as=ask_size
        """
        from alpaca.data.requests import OptionLatestQuoteRequest
        try:
            req = OptionLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self._data.get_option_latest_quote(req)
            result = {}
            for sym, q in quotes.items():
                result[sym] = {
                    "bid":      float(q.bid_price) if q.bid_price is not None else None,
                    "ask":      float(q.ask_price) if q.ask_price is not None else None,
                    "bid_size": int(q.bid_size)    if q.bid_size  is not None else None,
                    "ask_size": int(q.ask_size)    if q.ask_size  is not None else None,
                    "timestamp": str(q.timestamp),
                }
            return result
        except Exception as exc:
            raise PipelineConnectionError(
                f"GET /v1beta1/options/quotes/latest failed: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Paper broker — same interface, no network
# ---------------------------------------------------------------------------

class PaperBroker:
    """
    In-memory paper broker for tests, CI, and dry-run mode.
    Identical interface to AlpacaBroker — swap with get_broker(use_paper_stub=True).
    """

    def __init__(self, starting_equity: float = 100_000):
        self._equity = starting_equity
        self._positions: list[dict] = []
        self._order_counter = 0
        logger.info("[PaperBroker] Initialized: equity=$%.2f", starting_equity)

    @property
    def is_paper(self) -> bool:
        return True

    def get_account(self) -> dict:
        return {
            "equity": self._equity, "buying_power": self._equity * 4,
            "cash": self._equity, "portfolio_value": self._equity,
            "trading_blocked": False, "account_blocked": False,
            "pattern_day_trader": False,
        }

    def get_equity(self) -> float:
        return self._equity

    def get_buying_power(self) -> float:
        return self._equity * 4

    def check_account_ready(self) -> None:
        logger.info("[PaperBroker] Account ready")

    def submit(self, order: ApprovedOrder) -> FilledOrder:
        from .risk import ExecutionGuard
        ExecutionGuard.check(order)

        self._order_counter += 1
        order_id = f"paper-{self._order_counter:04d}"
        fill_price = order.estimated_fill_price
        slippage = fill_price * _DEFAULT_LIMIT_BUFFER_PCT

        for leg in order.legs:
            self._positions.append({
                "symbol":          leg.symbol,
                "qty":             order.position_size_contracts * leg.quantity,
                "side":            leg.side,
                "avg_entry_price": fill_price,
                "strategy":        order.strategy_name,
                "market_value":    fill_price * order.position_size_contracts * 100,
                "unrealized_pl":   0.0,
                "unrealized_plpc": 0.0,
                "current_price":   fill_price,
                "asset_class":     "us_option",
            })

        logger.info(
            "[PaperBroker] Fill: %s id=%s fill=%.4f slippage=%.4f",
            order.strategy_name, order_id, fill_price + slippage, slippage
        )
        return FilledOrder(
            order_id=order_id,
            approved_order=order,
            fill_price=fill_price + slippage,
            fill_timestamp=datetime.now(tz=timezone.utc),
            slippage_actual=slippage,
            status="open",
            broker="paper",
        )

    def get_order_status(self, order_id: str) -> dict:
        return {"id": order_id, "status": "filled", "filled_qty": 1, "filled_avg_price": 0}

    def wait_for_fill(self, order_id: str, **_) -> dict:
        return self.get_order_status(order_id)

    def cancel_order(self, order_id: str) -> None:
        logger.info("[PaperBroker] Cancel (sim): %s", order_id)

    def cancel_all_orders(self) -> int:
        return 0

    def get_positions(self) -> list[dict]:
        return list(self._positions)

    def close_position(self, symbol: str, qty=None) -> None:
        self._positions = [p for p in self._positions if p["symbol"] != symbol]

    def exercise_option(self, symbol: str) -> None:
        self.close_position(symbol)

    def get_option_chain(self, underlying_symbol: str, **_) -> dict:
        return {}

    def get_option_snapshots(self, symbols: list[str]) -> dict:
        return {s: {"bid": None, "ask": None, "iv": None, "delta": None,
                    "gamma": None, "theta": None, "vega": None} for s in symbols}

    def get_latest_quotes(self, symbols: list[str]) -> dict:
        return {s: {"bid": None, "ask": None, "bid_size": None,
                    "ask_size": None, "timestamp": None} for s in symbols}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_broker(paper: Optional[bool] = None, use_paper_stub: bool = False):
    """
    Return broker instance.
      use_paper_stub=True  → PaperBroker (no network, for tests/CI)
      use_paper_stub=False → AlpacaBroker (requires env vars)
    """
    if use_paper_stub:
        return PaperBroker()
    return AlpacaBroker(paper=paper)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _position_intent(side: str):
    from alpaca.trading.enums import PositionIntent
    mapping = {
        "buy_to_open":   PositionIntent.BUY_TO_OPEN,
        "buy_to_close":  PositionIntent.BUY_TO_CLOSE,
        "sell_to_open":  PositionIntent.SELL_TO_OPEN,
        "sell_to_close": PositionIntent.SELL_TO_CLOSE,
    }
    if side not in mapping:
        raise PipelineConnectionError(
            f"Unknown side '{side}'. Expected: {list(mapping)}"
        )
    return mapping[side]


def _limit_price_single(estimated_fill: float, side: str) -> float:
    """For single-leg orders, widen limit slightly to improve fill rate."""
    if "sell" in side:
        return estimated_fill * (1 - _DEFAULT_LIMIT_BUFFER_PCT)
    return estimated_fill * (1 + _DEFAULT_LIMIT_BUFFER_PCT)


def _parse_chain(chain) -> dict:
    """Parse alpaca-py chain response into consistent dict format."""
    result = {}
    if chain is None:
        return result
    snapshots = getattr(chain, 'snapshots', chain) if not isinstance(chain, dict) else chain
    for symbol, snap in (snapshots.items() if hasattr(snapshots, 'items') else {}.items()):
        result[symbol] = _parse_one_snapshot(snap)
    return result


def _parse_snapshots(snaps) -> dict:
    """Parse alpaca-py snapshot response."""
    result = {}
    if snaps is None:
        return result
    for symbol, snap in (snaps.items() if hasattr(snaps, 'items') else {}.items()):
        result[symbol] = _parse_one_snapshot(snap)
    return result


def _parse_one_snapshot(snap) -> dict:
    """
    Parse one option_snapshot object into a flat dict.
    Spec-verified field names:
      latestQuote: {bp, ap, bs, as}  (bid_price, ask_price, bid_size, ask_size)
      greeks:      {delta, gamma, theta, vega, rho}
      impliedVolatility: float
    """
    result = {
        "bid": None, "ask": None, "bid_size": None, "ask_size": None,
        "iv": None, "delta": None, "gamma": None,
        "theta": None, "vega": None, "rho": None,
    }
    if snap is None:
        return result

    # Latest quote — spec keys: bp, ap (bid_price, ask_price)
    quote = getattr(snap, 'latest_quote', None)
    if quote is not None:
        result["bid"]      = _f(getattr(quote, 'bid_price', None))
        result["ask"]      = _f(getattr(quote, 'ask_price', None))
        result["bid_size"] = _i(getattr(quote, 'bid_size',  None))
        result["ask_size"] = _i(getattr(quote, 'ask_size',  None))

    # Implied volatility
    result["iv"] = _f(getattr(snap, 'implied_volatility', None))

    # Greeks — spec: delta, gamma, theta, vega, rho (all from Black-Scholes)
    greeks = getattr(snap, 'greeks', None)
    if greeks is not None:
        result["delta"] = _f(getattr(greeks, 'delta', None))
        result["gamma"] = _f(getattr(greeks, 'gamma', None))
        result["theta"] = _f(getattr(greeks, 'theta', None))
        result["vega"]  = _f(getattr(greeks, 'vega',  None))
        result["rho"]   = _f(getattr(greeks, 'rho',   None))

    return result


def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        import math
        r = float(v)
        return None if math.isnan(r) else r
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
