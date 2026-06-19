"""
regime_backtest.py — Historical validation harness for RegimeDetector.

Replays historical market data day-by-day through RegimeDetector._classify()
(the same scoring logic used live) and checks whether the resulting regime
classifications actually preceded the kind of market behavior they claim to
predict — rather than relying solely on synthetic unit tests of individual
signals in isolation.

DATA COVERAGE — READ BEFORE TRUSTING RESULTS
==============================================
_classify() takes 9 signals. As of this version, only 4 can be computed from
real historical data; the remaining 5 are neutralized (set to values that
contribute nothing to the score) and clearly flagged in the output:

  COMPUTED FROM REAL DATA:
    - vix_level             (VIX_History.csv)
    - vix_trend              (5-day SMA of VIX_History.csv)
    - vix_percentile         (252-day rolling window of VIX_History.csv)
    - yield_curve_slope      (10Y - 2Y from Treasury par-yield CSV)

  NEUTRALIZED (no data available yet):
    - trend_strength, hurst   — need SPX/SPY price history (ADX + Hurst both
                                  require daily OHLC, not yet merged in)
    - vix_term_structure      — production uses VXV (3-month VIX), not VIX9D
                                  (9-day VIX) which is what was supplied.
                                  These measure opposite ends of the term
                                  structure curve and are NOT interchangeable
                                  — left as "unknown" rather than substituted.
    - breadth_scores           — needs 100-name constituent universe history,
                                  not a single index. Defaults to neutral.
    - big_blue_day/capitulation/stock_bond_signal_trusted
                                — needs TLT price history (not supplied)
    - dollar_stress_day        — needs UUP price history (not supplied)

This means: regime classifications from this version are driven almost
entirely by the VIX-based signals (level, trend, percentile) and the yield
curve. The "trending" vs "mean_reverting" distinction in particular is
under-powered right now since trend_strength and Hurst — two of the three
signals that actually distinguish those two regimes — are neutralized. The
"high_volatility" classification is the most trustworthy output of this
first-pass version, since 3 of its 4 contributing signals (VIX level, VIX
trend, VIX percentile) are real.

NEXT STEPS to get full signal coverage:
  1. Merge in SPX/SPY daily OHLC (unlocks trend_strength + Hurst)
  2. Get VXV (^VXV) history, not VIX9D (unlocks term structure)
  3. Get TLT history (unlocks the SPY/TLT divergence signal)
  4. Get UUP history (unlocks the dollar stress signal)
  5. Breadth requires either a historical constituent-level dataset or
     accepting it stays neutralized indefinitely (least critical — it's
     1 of 9 signals and the others compensate reasonably)
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

sys.path.insert(0, "src")
from options_bot.regime import RegimeDetector


def build_indicator_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a day-by-day indicators dataframe from the merged historical CSV,
    computing the 4 real signals and neutralizing the other 5.

    Expects columns: date, vix_close, tsy_10yr, tsy_2yr (from the corrected
    build_regime_backtest_data.py merge script).
    """
    df = df.sort_values("date").reset_index(drop=True).copy()

    # --- VIX trend: 5-day SMA comparison, exact production formula ---
    df["vix_sma5"] = df["vix_close"].rolling(5, min_periods=1).mean()
    pct_vs_sma5 = (df["vix_close"] - df["vix_sma5"]) / df["vix_sma5"] * 100
    df["vix_trend"] = np.select(
        [pct_vs_sma5 > 10, pct_vs_sma5 < -10],
        ["rising", "falling"],
        default="stable",
    )

    # --- VIX percentile: 252-day rolling window, exact production formula ---
    vix_vals = df["vix_close"].values
    pct = np.full(len(df), np.nan)
    for i in range(len(df)):
        window = vix_vals[max(0, i - 251) : i + 1]
        if len(window) >= 20:
            pct[i] = percentileofscore(window, vix_vals[i])
    df["vix_percentile"] = pct

    # --- Yield curve slope: 10Y - 2Y, exact production formula ---
    if "tsy_10yr" in df.columns and "tsy_2yr" in df.columns:
        df["yield_curve_slope"] = df["tsy_10yr"] - df["tsy_2yr"]
    else:
        df["yield_curve_slope"] = np.nan

    return df


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replay each day's indicators through the live RegimeDetector._classify()
    and record the resulting regime + confidence.
    """
    detector = RegimeDetector()
    regimes, confidences = [], []

    for _, row in df.iterrows():
        if pd.isna(row["vix_close"]) or pd.isna(row["vix_percentile"]):
            regimes.append(None)
            confidences.append(None)
            continue

        indicators = {
            # --- real signals ---
            "vix_level":          row["vix_close"],
            "vix_trend":          row["vix_trend"],
            "vix_percentile":     row["vix_percentile"],
            "yield_curve_slope":  row["yield_curve_slope"] if not pd.isna(row["yield_curve_slope"]) else 0.5,
            # --- neutralized: contribute nothing / minimal to the score ---
            "trend_strength":     0.5,           # neutral midpoint
            "hurst":               0.5,           # neutral midpoint (random walk)
            "_breadth_scores":     {},            # no contribution
            "vix_term_structure": "unknown",      # no contribution (production also defaults here on failure)
            "vix_term_ratio":      1.0,
            "big_blue_day":        False,
            "capitulation":        False,
            "stock_bond_signal_trusted": False,
            "dollar_stress_day":   False,
        }

        regime, confidence = detector._classify(indicators)
        regimes.append(regime)
        confidences.append(confidence)

    df = df.copy()
    df["regime"] = regimes
    df["confidence"] = confidences
    return df


def validate_against_forward_vix(df: pd.DataFrame, horizons=(5, 10, 20)) -> dict:
    """
    The one validation that's honestly possible with VIX-only data:
    does a 'high_volatility' classification actually precede VIX staying
    elevated or rising further, vs. other regimes? This doesn't require SPX
    realized vol -- it's a direct, real check using only the VIX series.

    For each regime, computes the mean forward VIX change (current VIX vs
    VIX N days later) and the % of days where VIX was higher N days later.
    A genuinely working high_volatility classifier should show: higher mean
    forward VIX change, and a higher % of "VIX stayed/went higher" days,
    than trending or mean_reverting classifications.
    """
    results = {}
    vix = df["vix_close"].values
    n = len(df)

    for horizon in horizons:
        fwd_change = np.full(n, np.nan)
        fwd_higher = np.full(n, np.nan)
        for i in range(n - horizon):
            if not np.isnan(vix[i]) and not np.isnan(vix[i + horizon]):
                fwd_change[i] = (vix[i + horizon] - vix[i]) / vix[i] * 100
                fwd_higher[i] = 1.0 if vix[i + horizon] > vix[i] else 0.0

        tmp = df.copy()
        tmp[f"fwd_vix_chg_{horizon}d"] = fwd_change
        tmp[f"fwd_vix_higher_{horizon}d"] = fwd_higher

        by_regime = tmp.groupby("regime").agg(
            n_days=("regime", "size"),
            mean_fwd_vix_change_pct=(f"fwd_vix_chg_{horizon}d", "mean"),
            pct_days_vix_higher=(f"fwd_vix_higher_{horizon}d", "mean"),
        )
        by_regime["pct_days_vix_higher"] = (by_regime["pct_days_vix_higher"] * 100).round(1)
        by_regime["mean_fwd_vix_change_pct"] = by_regime["mean_fwd_vix_change_pct"].round(2)

        results[horizon] = by_regime

    return results


def main(csv_path: str):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Loaded {len(df):,} rows ({df['date'].min().date()} to {df['date'].max().date()})")

    df = build_indicator_series(df)
    df = run_backtest(df)

    valid = df.dropna(subset=["regime"])
    print(f"\nClassified {len(valid):,} / {len(df):,} days")
    print("\nRegime distribution:")
    print(valid["regime"].value_counts())
    print(f"\nMean confidence: {valid['confidence'].mean():.3f}")

    print("\n" + "=" * 70)
    print("FORWARD VIX VALIDATION (the one honest check possible right now)")
    print("=" * 70)
    results = validate_against_forward_vix(valid)
    for horizon, table in results.items():
        print(f"\n--- {horizon}-day forward horizon ---")
        print(table)

    out_path = csv_path.replace(".csv", "_classified.csv")
    valid.to_csv(out_path, index=False)
    print(f"\nSaved classified output to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest RegimeDetector against historical data")
    parser.add_argument("csv_path", help="Path to merged regime_backtest_data.csv")
    args = parser.parse_args()
    main(args.csv_path)
