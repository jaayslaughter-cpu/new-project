# Options Bot

Production-grade automated short-premium options trading bot. Deployed on Railway, trading against Alpaca paper ($100k equity). Capital preservation first, profitability second, scale only after proven edge.

**Go-live gate:** 60–90 days of paper trading with 30+ closed trades (walk-forward milestone).

---

## Contents

- [Architecture](#architecture)
- [Strategies](#strategies)
- [Signal Stack](#signal-stack)
- [Risk Management](#risk-management)
- [Gated Features](#gated-features)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)
- [Local Development](#local-development)
- [Scripts](#scripts)
- [Module Map](#module-map)
- [Research](#research)
- [Key Invariants](#key-invariants)

---

## Architecture

```
Railway (Docker worker)
├── APScheduler
│   ├── Daily scan         06:45 PT / 09:45 ET  (Mon–Fri)
│   ├── Position monitor   every 15 min
│   ├── 0DTE scan          every 2 min  06:32–11:00 PT
│   ├── 0DTE monitor       every 15 sec
│   └── EOD cleanup        12:45 PT / 15:45 ET
├── PostgreSQL             Trade persistence (Railway plugin)
└── Discord webhooks       Alerts, daily/weekly summaries
```

**Broker:** Alpaca Algo Trader Plus (paper) — options TIF `day` only, multi-leg complex orders for spreads, stops managed by PositionMonitor (not broker OCO).

**Data sources:** Alpaca snapshots, yfinance (OHLCV + chains), Alpha Vantage (sector), FRED (yield curve, VIX3M via `^VIX3M`), CBOE CDN (direct VIX3M), Quiver Quant (congressional trades).

---

## Strategies

### Active (paper trading)

| Strategy | Description | Delta targets |
|---|---|---|
| `ShortPutSpread` | Sell OTM put, buy further OTM put | Short −0.25, Long −0.10 |
| `ShortCallSpread` | Sell OTM call, buy further OTM call | Short +0.15, Long +0.07 |
| `CashSecuredPut` | Sell OTM put, cash secured | Short −0.25 |
| `ShortStrangle` | Sell OTM put + OTM call simultaneously | Put −0.25, Call +0.15 |
| `ZeroDTEStrategy` | GEX-anchored 0DTE scalp on SPY | Gamma wall detection |

### Dormant (gated)

| Strategy | Gate condition |
|---|---|
| `IronCondor` | 30 closed trades **AND** `iron_condor_enabled=True` in Railway env |

When Iron Condor activates it **replaces** ShortStrangle on neutral-direction tickers — not additive.

### Universe (15 ETFs)

```
SPY  QQQ  IWM  TLT  XLF  XLK  XLE  XLV  XLI  GLD  EEM  HYG  SMH  VXX  XBI
```

### Spread width (budget-fit logic)

The strategy selects the widest spread that fits within the 1% per-trade risk budget. Starting from the natural long-leg (closest to target delta), it walks inward until `max_loss = (width − credit) × 100 ≤ equity × 0.01`. If even a 1-wide spread exceeds budget the ticker is skipped with a clear log message.

---

## Signal Stack

Every ticker passes through this pipeline before an order is submitted. All weights marked `PROVISIONAL` require 3–6 months of trade data to calibrate.

### Regime detection (`regime.py`)

| Signal | Source | Weight |
|---|---|---|
| VIX level + term structure | CBOE `^VIX3M` | PROVISIONAL |
| Yield curve slope | FRED T10Y2Y | PROVISIONAL |
| Market breadth (A/D) | yfinance breadth tickers | PROVISIONAL |
| Hurst exponent | Rolling close prices | PROVISIONAL |
| ADX trend strength | yfinance OHLCV | PROVISIONAL |

Output: `MEAN_REVERTING` or `TRENDING`. Short-premium strategies only trade in mean-reverting regimes. Options weight is reduced in trending regimes.

### FinBERT sentiment (`sentiment.py`)

Model: `ProsusAI/finbert`. Pulls recent news headlines for each ticker, runs batch inference, outputs a sentiment score in [−1, +1].

- Score < −0.15 → SELL signal → ticker skipped for that scan *(threshold PROVISIONAL)*
- Score ≥ −0.15 → passes sentiment gate

### Greeks enrichment (`greeks.py`)

Black-Scholes IV solver, delta, gamma, theta, vega for every option row. Treasury risk-free rate fetched live from FRED.

### IV quality scoring (`iv_quality.py`)

Scores the reliability of each option's implied volatility based on bid-ask spread, open interest, volume, and model consistency. Low-quality IV rows are excluded before delta filtering.

### Volume profile + HVN (`volume_profile.py`)

Identifies high-volume nodes (HVN) in the underlying's recent price history. Short strike must be within 1.5% of an HVN for confirmation. *(Threshold PROVISIONAL)*

### VRP gate (`vrp_gate.py`) — dormant

Measures the spread between implied vol and realized vol. Positive VRP confirms a short-premium edge. Gated behind 30-trade milestone + `vrp_gate_enabled=True`. Uses 5 vol estimators (Yang-Zhang, Garman-Klass, Parkinson, Rogers-Satchell, Close-to-Close).

### GEX analysis (`gex_analysis.py`)

Gamma exposure wall detection used by the 0DTE scalper. Identifies price levels where dealer gamma flips from positive to negative — high-probability mean-reversion zones for intraday spreads.

### Confidence scoring (`confidence_score.py`)

Aggregates all signals into a 0–100 composite score:

```
REGIME   — regime alignment and strength
SIGNAL   — delta, PoP, IV quality, HVN confirmation
ENTRY    — spread math, credit quality, DTE
STRATEGY — position sizing, width, risk/reward
RISK     — drawdown headroom, daily loss budget
EXEC     — liquidity, bid-ask, fill probability
TRACK    — closed-trade win rate (zero until baseline complete)
```

Scores below 60 are vetoed by the RiskManager.

### Congressional trades (`sec_signals.py`)

Quiver Quant API. Flags tickers with recent congressional purchases as bullish context for short-put positioning, recent sales as bearish context for short-call positioning.

### Piotroski F-Score (`parity_check.py`)

Fundamental quality filter. ETFs pass automatically; single stocks (if added later) require F ≥ 6.

### Earnings + macro blackout (`earnings_calendar.py`, `macro_blackout.py`)

No entries within 5 days of earnings. Macro blackout on FOMC, CPI, NFP dates.

---

## Risk Management

### Per-trade

| Parameter | Value |
|---|---|
| Risk budget | 1% of equity (~$1,000 at $100k) |
| Hard stop | 2× net credit received |
| Profit target | 50% of credit (closes at half-DTE) |
| Max loss | Capped by defined-risk spread width |

Stops are managed by PositionMonitor polling every 15 minutes — not broker OCO orders.

### Portfolio

| Parameter | Value |
|---|---|
| Max trades per day | 5 |
| Max concurrent positions | 5 |
| Daily loss halt | −3% NLV |
| Max drawdown halt | −8% from peak |

### 0DTE circuit breaker (`zerodte_guard.py`)

- Dedicated 0.5% daily loss cap (separate from main strategies)
- 3 consecutive losing days → arms a 14-trading-day disable
- Fail-closed on any error

### Stress testing (`stress_testing.py`)

7 scenarios run at EOD: Flash Crash, Vol Spike, Sector Rotation, Correlation Breakdown, Liquidity Crunch, Gap Risk, Tail Risk. Bot must survive all 7 at current equity level.

---

## Gated Features

All gated features require **two conditions**: (1) trade-count milestone AND (2) explicit human enable flag in Railway env. Never activate by milestone alone.

| Feature | Milestone | Enable flag | Status |
|---|---|---|---|
| Iron Condor | 30 closed trades | `iron_condor_enabled=True` | Built, dormant |
| Adaptive tuner | 30 closed trades | `adaptive_tuner_enabled=True` | Built, dormant |
| VRP gate | 30 closed trades | `vrp_gate_enabled=True` | Built, dormant |
| Credit regime | 30 closed trades | `credit_regime_enabled=True` | Built, dormant |
| Realized vol filter | 30 closed trades | `realized_vol_enabled=True` | Built, dormant |
| IV term structure | 30 closed trades | `iv_term_structure_enabled=True` | Built, dormant |
| Flow positioning | 30 closed trades | `flow_positioning_enabled=True` | Built, dormant |
| Analyst revisions | 30 closed trades | `analyst_revisions_enabled=True` | Built, dormant |
| Catch-up scan | Always on | `catchup_scan_enabled=True` (default) | Live |
| Orphan adoption | Deploy confirmed | `adopt_orphans_enabled=True` | Off by default |
| Position rolling | Post-milestone + stat validation | Future build | Not built |

**Walk-forward milestone (30 trades):** Triggers Iron Condor activation reminder, adaptive tuner enablement, and stat_validation DSR/Monte Carlo review. After 90 days: breadth weight calibration and adaptive tuner recalibration.

---

## Deployment

### Railway setup

The bot runs as a **Worker** service (not Web). Railway auto-deploys on every push to `main`.

```
Service type:  Worker
Build command: pip install -e .
Start command: python -m options_bot
```

> ⚠️ **Do not push to `main` during active scanning hours.** Railway auto-deploys immediately, restarting the container mid-session.

### Database

Railway PostgreSQL plugin injects `DATABASE_URL` automatically. Falls back to local SQLite (`options_bot.db`) when `DATABASE_URL` is unset.

All DB writes route through `TradeDatabase._execute()` which rewrites `?` placeholders to `%s` for Postgres. Never call `conn.execute()` directly on a `_get_conn()` connection — psycopg2 connections have no `.execute()` method and will silently fail.

### Scheduler timezone

All times are `America/Los_Angeles` (Pacific Time).

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | ✅ | Alpaca API key |
| `ALPACA_SECRET_KEY` | ✅ | Alpaca secret key |
| `ALPACA_PAPER` | | `true` (default) or `false` for live |
| `DISCORD_WEBHOOK_URL` | | Discord webhook for all alerts and summaries |
| `DATABASE_URL` | | PostgreSQL URL (Railway injects automatically) |
| `FRED_API_KEY` | | FRED API key (yield curve, VIX3M) |
| `ALPHA_VANTAGE_API_KEY` | | Alpha Vantage key (sector signals) |
| `QUIVER_API_KEY` | | Quiver Quant key (congressional trades) |
| `LOG_LEVEL` | | `INFO` (default) |
| `iron_condor_enabled` | | `true` to activate Iron Condor (requires 30 closed trades) |
| `catchup_scan_enabled` | | `true` (default) |
| `adopt_orphans_enabled` | | `false` (default) — enable only after deployment confirmed |

---

## Local Development

### Install

```bash
git clone https://github.com/jaayslaughter-cpu/new-project.git
cd new-project
pip install -e .
```

### Run tests

```bash
python -m pytest tests/ -q
# 238 tests, no API keys required
```

### Single scan (paper mode)

```bash
ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python -m options_bot --once
```

### Full scheduler (paper mode)

```bash
ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python -m options_bot
```

### Dry run (no API calls)

```bash
python -m options_bot --dry-run
```

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/backtest_strategy.py` | Backtests ShortPutSpread + ShortCallSpread against Alpaca historical stock bars. Uses real FRED VIX/RVX/GVZ/VXEEM for SPY/IWM/GLD/EEM, real `^VXN` for QQQ, VIX-scaled multipliers for sector ETFs. Outputs VRP ranking, PoP accuracy sweep, filter attribution, and per-ticker Sharpe. |
| `scripts/regime_backtest.py` | Backtests the regime detection signal stack. |
| `scripts/reconcile_iwm_orphan.py` | One-off reconciliation for the 06-24 orphaned IWM short call spread. Inserts the position into the DB so PositionMonitor enforces its stop and profit target. Idempotent — safe to run multiple times. |
| `scripts/screen_etf_universe.py` | Screens the 15-ETF universe for current liquidity and IV quality. |
| `scripts/smoke_test.py` | Integration smoke test against the live Alpaca paper API. |

### Running the backtest

```bash
ALPACA_API_KEY=... ALPACA_SECRET_KEY=... FRED_API_KEY=... \
  python scripts/backtest_strategy.py \
  --start 2024-02-01 \
  --tickers SPY QQQ IWM TLT XLF XLK XLE XLV XLI GLD EEM HYG SMH VXX XBI \
  --output backtest_results/
```

Outputs: `trades.csv`, `vrp_by_ticker.csv`, `pop_sweep.csv`, `filter_attribution.csv`, `ticker_summary.csv`, `summary.txt`.

---

## Module Map

```
src/options_bot/
├── __main__.py           CLI entry point, config parsing, bot startup
├── orchestrator.py       Orchestrator, TradingPipeline, PositionMonitor,
│                           TradeDatabase, catch-up scan, orphan adoption
├── strategy.py           BaseStrategy, ShortPutSpread, ShortCallSpread,
│                           CashSecuredPut, ShortStrangle, IronCondor (dormant)
├── strategy_0dte.py      ZeroDTEStrategy, ZeroDTECircuitBreaker, Kelly sizing
├── risk.py               RiskManager, RiskConfig, ExecutionGuard
├── risk_profiles.py      Conservative / Balanced / Aggressive profile presets
├── broker.py             AlpacaBroker (alpaca-py SDK), multi-leg order builder
├── contracts.py          OptionChainRow, EnrichedOptionRow, ApprovedOrder,
│                           FilledOrder, StrategySignal data classes
├── market_data.py        YFinanceDataLoader, chain fetch, liquidity filter
├── greeks.py             Black-Scholes IV solver, Greeks, PoP, pop_spread,
│                           bs_price, bs_greeks, Treasury rate fetch
├── spread_math.py        bull_put_entry, calc_spread, profit_target_price
├── regime.py             RegimeDetector — VIX, yield curve, breadth, Hurst, ADX
├── sentiment.py          SentimentAnalyzer — ProsusAI/finbert, batch inference
├── iv_quality.py         IV quality scoring — bid-ask, OI, volume, model check
├── volume_profile.py     HVN detection, price-level clustering
├── gex_analysis.py       Gamma exposure wall detection for 0DTE scalper
├── realized_vol.py       5 RV estimators: Yang-Zhang, Garman-Klass, Parkinson,
│                           Rogers-Satchell, Close-to-Close
├── vrp_gate.py           Volatility risk premium gate (dormant, gated)
├── adaptive.py           Adaptive parameter tuner (dormant, gated)
├── stat_validation.py    DSR t-stat, Monte Carlo, Fama-French attribution
├── walk_forward.py       Walk-forward milestone tracking and gate logic
├── confidence_score.py   0–100 composite signal score aggregation
├── circuit_breaker.py    Per-signal circuit breaker with consecutive-fail logic
├── stress_testing.py     7 EOD stress scenarios
├── earnings_calendar.py  Earnings blackout filter
├── macro_blackout.py     FOMC/CPI/NFP macro event blackout
├── scanner.py            Universe scanning orchestration
├── breadth.py            Market breadth (A/D ratio, McClellan oscillator)
├── hurst.py              Hurst exponent estimator
├── tape.py               Order flow and tape reading utilities
├── trendlines.py         Support/resistance trendline detection
├── alpaca_quotes.py      Alpaca real-time snapshot and options chain fetch
├── analyst_revisions.py  Analyst revision signals (dormant, gated)
├── credit_regime.py      Credit market regime via HYG/LQD (dormant, gated)
├── sec_signals.py        Congressional + insider trade signals (Quiver Quant)
├── parity_check.py       Piotroski F-Score fundamental filter
├── massive_data.py       Bulk historical data utilities
├── metrics.py            Sharpe, Calmar, win rate, expectancy, drawdown
├── universe.py           ETF universe definition and liquidity thresholds
├── zerodte_guard.py      0DTE circuit breaker — 0.5% daily cap, 14-day disable
├── system_validator.py   4-check boot validator (env, broker, data, modules)
├── logging_config.py     Structured JSON logging for Railway
└── exceptions.py         PipelineConnectionError, RiskVetoError,
                            LiquidityFilterError, StrategySignalError, etc.
```

---

## Research

```
research/
├── strategy_records.json             19 records (core strategy library)
├── champions_strategy_records.json   12 records (Champions of Trading Profits)
├── elder_1993_strategy_records.json   7 records (Elder 1993 — Trading for a Living)
└── options_strategy_intelligence.json Broader strategy intelligence base
```

**Flagged future builds (post-30-trade milestone):**
- Force Index (2-day EMA) as Screen 2 oscillator — Elder 1993, `elder93_002`
- Bollinger Band width as per-ticker IV proxy for entry timing — Elder 1993, `elder93_005`
- Divergence quality weighting (Class A/B/C) as signal confidence multiplier — Elder 1993, `elder93_004`
- Unusual options flow signal — requires Unusual Whales API (~$50/month), the only real-time source with a proper programmatic API

---

## Key Invariants

These must never be violated regardless of feature flags, configuration, or edits.

**DB writes use `?` placeholders via `_execute()`.**
All SQL routes through `TradeDatabase._execute()` which rewrites `?` → `%s` and escapes literal `%`. Never call `conn.execute()` directly — psycopg2 connections have no `.execute()` method and silently fail on Postgres.

**All `strategy.evaluate()` signatures accept `risk_budget_dollars`.**
The orchestrator passes `risk_budget_dollars=...` uniformly to every strategy. Every concrete strategy class (including CSP and IronCondor) must accept this optional kwarg or it raises `TypeError` and silently skips all tickers for that strategy.

**Critical paths fail loud.**
`save_fill` re-raises on DB error and sends a Discord alert. Silent exception swallowing on trade writes is what caused the 06-24 orphan incident (IWM short call spread dispatched but never persisted).

**Container restart state is persisted.**
`Trades today` counter and `last_scan_date` are stored in the DB — not in memory. APScheduler reschedules all jobs after every restart. Catch-up scan logic handles missed windows.

**Gated feature pattern.**
Every gated feature requires (1) trade-count milestone AND (2) explicit human enable flag. Never activate by trade count alone.

**Iron Condor replaces ShortStrangle.**
When activated at the 30-trade milestone, IronCondor replaces ShortStrangle on neutral-direction tickers — it is not additive to it.

**Position rolling (not yet built).**
When built: max 2 rolls per position, hard DTE floor, rolled risk must stay within original 1% per-trade budget (never increase), rolled trades must have separate accounting so they cannot corrupt win-rate/expectancy stats or the adaptive tuner.

**Push workflow.**
Always: fetch file + SHA → patch locally → `ast.parse` validate → `pytest tests/ -q` (238 passing) → push with freshly fetched SHA → verify on fresh re-clone. Stale-copy push regression is the documented #1 recurring failure mode.

**No pushes to `main` during active market hours.**
Railway auto-deploys on every push to `main`. A mid-session deploy resets the scheduler and any in-progress scan.

---

## Provisional Weights

These thresholds require calibration against real closed-trade data. Do not treat them as validated until 3–6 months of live data confirms them.

| Parameter | Current value | How to calibrate |
|---|---|---|
| Regime score weights | Various | `scripts/regime_backtest.py` |
| PoP minimum | 65% | `scripts/backtest_strategy.py` PoP sweep |
| Sentiment threshold | −0.15 | PoP sweep + filter attribution |
| HVN distance | 1.5% | Filter attribution by ticker |
| 0DTE bootstrap parameters | Various | Walk-forward on 0DTE closed trades |

Run `scripts/backtest_strategy.py --start 2024-02-01` to get the PoP accuracy sweep and filter attribution before adjusting any of these values.
