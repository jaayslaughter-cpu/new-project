# Options Bot

Automated options trading via Alpaca. Runs as a Railway worker.

## Stack
- **Strategies**: Cash-secured put, short put spread, short strangle
- **Broker**: Alpaca (paper + live) via `alpaca-py`
- **Scheduling**: APScheduler — scan at 9:45 AM PT, monitor every 15 min, EOD cleanup at 3:45 PM PT
- **Database**: PostgreSQL on Railway, SQLite locally
- **Notifications**: Discord webhook

## Setup

### 1. Clone and install
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in your Alpaca API keys
```

### 3. Test locally (no API keys needed)
```bash
python -m options_bot --dry-run
```

### 4. Run single scan (paper mode)
```bash
python -m options_bot --once
```

### 5. Run full scheduler (paper mode)
```bash
python -m options_bot
```

## Railway Deployment

1. Push this repo to GitHub
2. New Railway service → deploy from GitHub
3. Add PostgreSQL plugin (Railway injects DATABASE_URL automatically)
4. Set environment variables from `.env.example`
5. Service type: **Worker** (not Web)

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | Yes | Alpaca API key |
| `ALPACA_SECRET_KEY` | Yes | Alpaca secret key |
| `ALPACA_PAPER` | No | `true` (default) or `false` |
| `DISCORD_WEBHOOK_URL` | No | Discord webhook for notifications |
| `DATABASE_URL` | No | PostgreSQL URL (Railway injects this) |
| `LOG_LEVEL` | No | `INFO` (default) |

## CLI Options

```
python -m options_bot [options]

--tickers SPY QQQ IWM     Tickers to scan (default: SPY QQQ IWM)
--strategy short_put_spread  Strategy: csp | short_put_spread | short_strangle
--risk-pct 0.02           Risk per trade as fraction of equity (default: 2%)
--max-trades 5            Max trades per day
--max-positions 5         Max concurrent open positions
--scan-time 09:45         Scan time in PT (HH:MM)
--once                    Single scan then exit
--dry-run                 Use paper stub (no API calls)
--live                    Live trading (overrides paper default)
```

## Module Map

```
exceptions.py    — PipelineConnectionError, RiskVetoError, LiquidityFilterError, etc.
contracts.py     — OptionChainRow, EnrichedOptionRow, ApprovedOrder, FilledOrder
market_data.py   — YFinanceDataLoader (yfinance chain fetch + liquidity filter)
greeks.py        — Black-Scholes IV solver + Greeks + Treasury rate fetcher
strategy.py      — CashSecuredPut, ShortPutSpread, ShortStrangle
risk.py          — RiskManager, RiskConfig, ExecutionGuard
broker.py        — AlpacaBroker, PaperBroker (alpaca-py SDK)
orchestrator.py  — Orchestrator, TradingPipeline, PositionMonitor, TradeDatabase
__main__.py      — CLI entry point
```

## Tests

```bash
pytest tests/test_integration.py -v   # 129 tests, no API calls needed
```
