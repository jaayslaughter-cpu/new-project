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


# ---------------------------------------------------------------------------
# OCC symbol formatting
# ---------------------------------------------------------------------------

def format_option_symbol(
    ticker: str,
    expiry_yymmdd: str,
    option_type: str,
    strike: float,
) -> str:
    """
    Format an option contract symbol in strict Alpaca OCC format (no spaces).

    Pattern: {TICKER}{YYMMDD}{C|P}{strike_8digit}

    Strike encoding: integer cents × 10, zero-padded to 8 digits.
    e.g. SPY $580.00 call expiring 2026-06-20 → SPY260620C00580000
         SPY $522.50 put  expiring 2026-06-20 → SPY260620P00522500

    Args:
        ticker:       Underlying symbol, e.g. "SPY"
        expiry_yymmdd: 6-digit string "YYMMDD", e.g. "260620"
        option_type:  "C" or "call", "P" or "put" (case-insensitive)
        strike:       Strike price as a float, e.g. 580.0 or 522.5

    Returns:
        OCC-formatted symbol string, e.g. "SPY260620C00580000"

    Raises:
        ValueError: if option_type is not C/P or strike is non-positive.
    """
    ot = option_type.upper()
    if ot not in ("C", "P", "CALL", "PUT"):
        raise ValueError(f"option_type must be C/P/CALL/PUT, got {option_type!r}")
    ot_char = "C" if ot in ("C", "CALL") else "P"

    if strike <= 0:
        raise ValueError(f"Strike must be positive, got {strike}")

    padded_strike = f"{int(round(strike * 1000)):08d}"
    return f"{ticker.upper()}{expiry_yymmdd}{ot_char}{padded_strike}"


# ---------------------------------------------------------------------------
# Live midpoint fetcher (Alpaca OptionLatestQuoteRequest)
# ---------------------------------------------------------------------------

def _get_live_midpoint(
    data_client,
    symbol: str,
    fallback: Optional[float] = None,
) -> Optional[float]:
    """
    Fetch the live bid/ask midpoint for a single option contract from Alpaca.

    Used by _submit_single and _submit_mleg to replace the strategy-layer
    estimated_fill_price with a real-time midpoint at the moment of order
    submission — tighter pricing, better fills.

    Fallback chain:
        1. Live (bid + ask) / 2 from OptionLatestQuoteRequest
        2. ask_price alone if bid is zero (deep OTM / wide market)
        3. fallback argument (strategy-layer mid estimate)
        4. None → caller must abort the order

    Zero bids are common in illiquid options — asking for the ask is correct
    because it represents the actual market offer.

    Args:
        data_client: OptionHistoricalDataClient instance (from AlpacaBroker._data)
        symbol:      OCC symbol string e.g. "SPY260620C00580000"
        fallback:    Strategy-layer estimated mid, used if Alpaca quote fails.

    Returns:
        Midpoint as float rounded to 2dp, or None if all sources fail.
    """
    try:
        from alpaca.data.requests import OptionLatestQuoteRequest
        req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
        result = data_client.get_option_latest_quote(req)
        if symbol not in result:
            logger.warning("[Broker] Live quote missing for %s — using fallback", symbol)
            return fallback

        q = result[symbol]
        bid = float(q.bid_price) if q.bid_price is not None else 0.0
        ask = float(q.ask_price) if q.ask_price is not None else 0.0

        if ask == 0:
            logger.warning("[Broker] Zero ask for %s — using fallback", symbol)
            return fallback

        if bid == 0:
            # Deep OTM / illiquid: use ask as the price to pay/receive
            logger.debug("[Broker] Zero bid for %s — using ask %.2f", symbol, ask)
            return round(ask, 2)

        mid = round((bid + ask) / 2.0, 2)
        logger.debug("[Broker] Live mid for %s: bid=%.2f ask=%.2f mid=%.2f", symbol, bid, ask, mid)

        # Slippage deviation check — warn if live mid differs materially from strategy estimate
        if fallback is not None and fallback > 0:
            deviation_pct = abs(mid - fallback) / fallback
            if deviation_pct > 0.15:
                logger.warning(
                    "[Broker] SLIPPAGE WARNING: live mid %.2f deviates %.0f%% from "
                    "strategy estimate %.2f for %s. Market may have moved — "
                    "fill economics differ from what risk system approved.",
                    mid, deviation_pct * 100, fallback, symbol,
                )

        return mid

    except Exception as exc:
        logger.warning("[Broker] Live quote fetch failed for %s (%s) — using fallback", symbol, exc)
        return fallback


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
            from alpaca.data.historical.stock import StockHistoricalDataClient
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
        self._stock_data = StockHistoricalDataClient(
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

        Limit price: fetched live from Alpaca OptionLatestQuoteRequest at
        submission time. Fallback to strategy-layer estimated_fill_price if
        the live quote is unavailable (e.g. pre-market, network hiccup).

        client_order_id includes int(time.time()) so Railway container
        restarts cannot accidentally re-submit a duplicate order.
        """
        from alpaca.trading.requests import LimitOrderRequest, StopLossRequest
        from alpaca.trading.enums import TimeInForce, OrderClass
        from alpaca.common.exceptions import APIError

        leg = order.legs[0]
        qty = order.position_size_contracts

        # Live midpoint — tighter than strategy-layer estimate
        live_mid = _get_live_midpoint(
            self._data, leg.symbol, fallback=order.estimated_fill_price
        )
        if live_mid is None:
            raise PipelineConnectionError(
                f"Cannot submit order for {leg.symbol}: "
                f"live midpoint unavailable and no fallback price."
            )
        limit_price = _limit_price_single(live_mid, leg.side)

        # Fail-stop: limit_price must be positive and non-zero before submission.
        # _limit_price_single applies a small buffer to live_mid; if live_mid was
        # near-zero (deep OTM, stale quote) the result could be <= 0, which Alpaca
        # would reject with a cryptic API error. Catch it here with a clear message.
        if not limit_price or limit_price <= 0:
            raise PipelineConnectionError(
                f"Invalid limit_price={limit_price!r} for {leg.symbol} "
                f"(live_mid={live_mid}). Aborting order — check quote freshness."
            )

        stop = StopLossRequest(stop_price=round(order.hard_stop_price, 2))

        # Timestamp + uuid4 suffix: Railway container restarts get a new
        # timestamp, preventing duplicate orders from identical restarts.
        client_id = (
            f"opt_{order.underlying}_{order.strategy_name}"
            f"_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        )

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
            "[AlpacaBroker] single-leg: %s qty=%d live_mid=%.2f limit=%.2f stop=%.2f",
            leg.symbol, qty, live_mid, limit_price, order.hard_stop_price
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
        Stops must be managed externally — the PositionMonitor submits
        separate closing orders when the stop level is reached.

        Live midpoint: each leg's midpoint is fetched from Alpaca at
        submission time. The net is recomputed from live quotes. This
        replaces the strategy-layer estimated net, which may be stale
        by the time the order reaches execution. Falls back to
        net_debit_credit from the ApprovedOrder if live quotes fail.

        client_order_id includes int(time.time()) for Railway restart safety.
        """
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
        from alpaca.trading.enums import TimeInForce, OrderClass
        from alpaca.common.exceptions import APIError

        qty = order.position_size_contracts

        # ── Live net price from per-leg midpoints ──────────────────────────
        # For a credit spread: short leg receives premium (positive mid),
        # long leg costs premium (negative mid). Net = short_mid - long_mid.
        # Sign convention: negative net = credit (we receive), per Alpaca spec.
        live_net = self._compute_live_net(order)
        if live_net is None:
            logger.warning(
                "[AlpacaBroker] mleg live net unavailable for %s — "
                "falling back to estimated net %.2f",
                order.underlying, order.net_debit_credit,
            )
            live_net = order.net_debit_credit

        # Buffer: accept slightly less credit / pay slightly more debit
        if live_net < 0:
            limit_price = live_net * (1 - _DEFAULT_LIMIT_BUFFER_PCT)  # e.g. -1.50 → -1.47
        else:
            limit_price = live_net * (1 + _DEFAULT_LIMIT_BUFFER_PCT)

        # Build legs — ratio_qty=1 for standard 1:1 spreads
        alpaca_legs = [
            OptionLegRequest(
                symbol=leg.symbol,
                ratio_qty=1,
                position_intent=_position_intent(leg.side),
            )
            for leg in order.legs
        ]

        client_id = (
            f"opt_{order.underlying}_{order.strategy_name}"
            f"_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        )

        req = LimitOrderRequest(
            qty=qty,
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
            legs=alpaca_legs,
            client_order_id=client_id,
        )

        logger.debug(
            "[AlpacaBroker] mleg: %d legs qty=%d live_net=%.2f limit=%.2f (credit=%.2f)",
            len(alpaca_legs), qty, live_net, limit_price, abs(live_net),
        )
        logger.debug(
            "[AlpacaBroker] mleg: stop managed by PositionMonitor at %.2f",
            order.hard_stop_price
        )
        return self._post_order(req, order, client_id)

    def _compute_live_net(self, order: ApprovedOrder) -> Optional[float]:
        """
        Recompute the spread's net credit/debit from live per-leg midpoints.

        For each leg: sell_to_open = receive premium (positive contribution),
        buy_to_open = pay premium (negative contribution).
        Returns None if any leg's midpoint cannot be fetched.
        """
        net = 0.0
        for leg in order.legs:
            mid = _get_live_midpoint(self._data, leg.symbol)
            if mid is None:
                logger.warning(
                    "[AlpacaBroker] No live mid for leg %s — aborting live net calc",
                    leg.symbol,
                )
                return None
            # sell_to_open / sell_to_close → receive → positive credit
            # buy_to_open  / buy_to_close  → pay     → negative contribution
            if leg.side in ("sell_to_open", "sell_to_close"):
                net -= mid   # Alpaca mleg spec: net negative = credit received
            else:
                net += mid
        return round(net, 2)

    def _post_order(self, req, order: ApprovedOrder, client_id: str) -> FilledOrder:
        """
        POST /v2/orders — shared submission path with retry.

        Retry policy:
          - 4xx (APIError): no retry — bad request, retrying won't help
          - 5xx / network timeout: up to 2 retries, exponential backoff (2s, 4s)
          - After all retries exhausted: raise PipelineConnectionError
        """
        from alpaca.common.exceptions import APIError
        submit_time = datetime.now(tz=timezone.utc)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._trading.submit_order(order_data=req)
                break   # success — exit retry loop
            except APIError as api_err:
                # 4xx: Alpaca rejected the request — retrying won't help
                err_msg = getattr(api_err, "message", str(api_err))
                raise PipelineConnectionError(
                    f"Alpaca API rejected order (client_id={client_id}): {err_msg}"
                ) from api_err
            except Exception as exc:
                if attempt == max_attempts:
                    raise PipelineConnectionError(
                        f"POST /v2/orders failed after {max_attempts} attempts "
                        f"(client_id={client_id}): {exc}"
                    ) from exc
                wait = 2 ** attempt   # 2s, 4s
                logger.warning(
                    "[AlpacaBroker] Order submit attempt %d/%d failed (%s) — "
                    "retrying in %ds (client_id=%s)",
                    attempt, max_attempts, exc, wait, client_id,
                )
                time.sleep(wait)

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

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 400,
    ) -> list[dict]:
        """
        Fetch historical bars for a stock/ETF symbol via Alpaca's Stock Data API.

        Used by VWAPStretchFilter to compute intraday VWAP from 1-minute SPY bars.

        Parameters
        ----------
        symbol    : Ticker symbol (e.g. "SPY")
        timeframe : Alpaca timeframe string — "1Min", "5Min", "1Hour", "1Day"
        start     : ISO date string "YYYY-MM-DD" (defaults to today)
        end       : ISO date string "YYYY-MM-DD" (defaults to today)
        limit     : Max number of bars to return

        Returns
        -------
        list of dicts with keys: t, o, h, l, c, v, vw
          t  = timestamp (str)
          o  = open
          h  = high
          l  = low
          c  = close
          v  = volume
          vw = vwap (volume-weighted price for the bar)
        """
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError:
            raise PipelineConnectionError(
                "alpaca-py not installed or outdated. Run: pip install alpaca-py>=0.26.0"
            )

        # Parse timeframe string into alpaca TimeFrame object
        _tf_map = {
            "1Min":  TimeFrame(1, TimeFrameUnit.Minute),
            "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day":  TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = _tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))

        from datetime import date as _date, datetime as _datetime
        today = _date.today().isoformat()
        start_dt = start or today
        end_dt   = end   or today

        # Alpaca requires datetime objects with timezone for intraday requests
        # Use market open (09:30 ET) to market close (16:00 ET) window
        import pytz as _pytz
        et = _pytz.timezone("US/Eastern")
        start_aware = et.localize(_datetime.fromisoformat(f"{start_dt}T09:30:00"))
        end_aware   = et.localize(_datetime.fromisoformat(f"{end_dt}T16:00:00"))

        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_aware,
                end=end_aware,
                limit=limit,
                feed="iex",  # IEX feed — free tier; switch to "sip" for paid data
            )
            resp = self._stock_data.get_stock_bars(req)
            bars_obj = resp.data.get(symbol, [])
            result = []
            for bar in bars_obj:
                result.append({
                    "t":  str(bar.timestamp),
                    "o":  float(bar.open),
                    "h":  float(bar.high),
                    "l":  float(bar.low),
                    "c":  float(bar.close),
                    "v":  float(bar.volume),
                    "vw": float(bar.vwap) if bar.vwap is not None else float(bar.close),
                })
            logger.debug("[AlpacaBroker] get_bars(%s, %s): %d bars", symbol, timeframe, len(result))
            return result
        except Exception as exc:
            raise PipelineConnectionError(
                f"get_bars({symbol}, {timeframe}) failed: {exc}"
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

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 400,
    ) -> list[dict]:
        """PaperBroker stub — returns empty list; VWAPStretchFilter will bypass gracefully."""
        return []


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
