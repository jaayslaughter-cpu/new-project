"""
Regime Backtest Data Builder — v2, full signal coverage
==========================================================
Adds SPX, VIX3M (term structure), TLT (stock-bond divergence), and UUP
(dollar stress) to the original VIX + Treasury merge, completing the data
needed for ALL 9 RegimeDetector signals (vs. 4/9 in the first pass).

FIXES vs. your latest snippet:
  1. query1.finance.yahoo.com/v7/finance/download/... is DEAD. Confirmed via
     direct fetch -- Yahoo returns "ROBOTS_DISALLOWED" / 401, not CSV data,
     for ANY ticker on that endpoint (tested with ^GSPC directly). This
     isn't a per-ticker issue, the whole endpoint was retired by Yahoo years
     ago. Replaced with the yfinance package's Ticker().history(), the same
     method the bot already uses successfully in production.

  2. ^VXV is STALE. Confirmed via search: CBOE renamed the ticker from VXV
     to VIX3M on September 18, 2017. ^VXV will return incomplete or no data
     for the modern period. Use ^VIX3M instead -- or better, CBOE's own
     direct CDN endpoint (same pattern as your VIX_History.csv,
     GVZ_History.csv, OVX_History.csv downloads), which needs no auth and
     is more reliable than scraping Yahoo for an index ticker:
         https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv
     This script uses the CBOE direct source for VIX3M.

  3. TLT and UUP are real US-listed ETFs (not indices) -- yfinance handles
     these natively and reliably, no substitution needed there.

Run this somewhere with real internet access (your own machine, or a
one-off Railway job) -- the SPX/TLT/UUP yfinance calls and the VIX3M CBOE
fetch both need outbound network access this sandbox doesn't have.
"""

import urllib.request
import pandas as pd

# ── 1. Load VIX (local file, CBOE format) ────────────────────────────────────
vix_df = pd.read_csv("VIX_History.csv")
vix_df["date"] = pd.to_datetime(vix_df["DATE"], format="%m/%d/%Y")
vix_df = vix_df.rename(columns={"CLOSE": "vix_close", "OPEN": "vix_open",
                                  "HIGH": "vix_high", "LOW": "vix_low"})
vix_df = vix_df[["date", "vix_open", "vix_high", "vix_low", "vix_close"]]

# ── 2. Load Treasury par yield curve (local file, combined 1990-2023) ────────
# KNOWN GAP: stops 2023-12-29. Download an updated CSV from Treasury.gov
# covering 2024-present if you need yield-curve data for recent dates.
tsy_df = pd.read_csv("par-yield-curve-rates-1990-2023.csv")
tsy_df["date"] = pd.to_datetime(tsy_df["date"], format="%m/%d/%Y")
tsy_df = tsy_df.rename(columns={c: f"tsy_{c.replace(' ', '')}" for c in tsy_df.columns if c != "date"})

# ── 3. Load supporting vol indices (local files) ──────────────────────────────
def load_simple_vol_index(path, value_col_rename):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    value_col = [c for c in df.columns if c not in ("DATE", "date")][-1]
    df = df.rename(columns={value_col: value_col_rename})
    return df[["date", value_col_rename]]

vvix_df  = load_simple_vol_index("VVIX_History.csv",  "vvix")
gvz_df   = load_simple_vol_index("GVZ_History.csv",   "gvz")
ovx_df   = load_simple_vol_index("OVX_History.csv",   "ovx")
vix9d_df = load_simple_vol_index("VIX9D_History.csv", "vix9d")

# ── 4. VIX3M (term structure) — direct CBOE CDN, NOT Yahoo's stale ^VXV ──────
print("Fetching VIX3M from CBOE direct CDN...")
vix3m_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"
req = urllib.request.Request(vix3m_url, headers={"User-Agent": "OptionsBot regime-backtest/1.0"})
with urllib.request.urlopen(req, timeout=30) as r:
    vix3m_raw = r.read().decode("utf-8")
with open("VIX3M_History.csv", "w") as f:
    f.write(vix3m_raw)
vix3m_df = pd.read_csv("VIX3M_History.csv")
vix3m_df["date"] = pd.to_datetime(vix3m_df["DATE"], format="%m/%d/%Y")
vix3m_df = vix3m_df.rename(columns={"CLOSE": "vix3m_close"})[["date", "vix3m_close"]]
print(f"  VIX3M: {len(vix3m_df):,} rows ({vix3m_df['date'].min().date()} to {vix3m_df['date'].max().date()})")

# ── 5. SPX, TLT, UUP via yfinance (NOT the dead Yahoo CSV endpoint) ──────────
import yfinance as yf

def fetch_via_yfinance(ticker, prefix):
    print(f"Fetching {ticker} via yfinance...")
    hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    df = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["date", f"{prefix}_open", f"{prefix}_high", f"{prefix}_low",
                  f"{prefix}_close", f"{prefix}_volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    print(f"  {ticker}: {len(df):,} rows ({df['date'].min().date()} to {df['date'].max().date()})")
    return df

spx_df = fetch_via_yfinance("^GSPC", "spx")
tlt_df = fetch_via_yfinance("TLT",   "tlt")
uup_df = fetch_via_yfinance("UUP",   "uup")

# ── 6. Merge everything on date (outer join — keep all dates, NaN where missing) ─
merged = vix_df
for df in (tsy_df, vvix_df, gvz_df, ovx_df, vix9d_df, vix3m_df, spx_df, tlt_df, uup_df):
    merged = pd.merge(merged, df, on="date", how="outer")

merged = merged.sort_values("date").reset_index(drop=True)

# ── 7. Report data quality / gaps before saving ──────────────────────────────
print(f"\nMerged: {len(merged):,} rows ({merged['date'].min().date()} to {merged['date'].max().date()})")

tsy_gap_start = tsy_df["date"].max()
vix_end = vix_df["date"].max()
if vix_end > tsy_gap_start:
    gap_days = (vix_end - tsy_gap_start).days
    print(
        f"\n⚠️  TREASURY DATA GAP: no yield-curve data from "
        f"{tsy_gap_start.date()} to {vix_end.date()} ({gap_days} days, "
        f"~{gap_days/365:.1f} years). Download an updated par-yield CSV "
        f"from Treasury.gov covering this period if needed."
    )

tlt_start = tlt_df["date"].min()
print(
    f"\nℹ️  TLT (used for the SPY/TLT divergence signal) only exists from "
    f"{tlt_start.date()} (ETF inception). Backtest rows before this date "
    f"will have null TLT columns -- the divergence signal can only be "
    f"validated from {tlt_start.date()} onward."
)

uup_start = uup_df["date"].min()
print(
    f"ℹ️  UUP (used for the dollar stress signal) only exists from "
    f"{uup_start.date()} (ETF inception). Same caveat applies."
)

merged.to_csv("regime_backtest_data.csv", index=False)
print(f"\nSaved: regime_backtest_data.csv ({len(merged):,} rows, {len(merged.columns)} columns)")
print(f"Columns: {list(merged.columns)}")
