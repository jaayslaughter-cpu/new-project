"""
regime_backtest.py — Historical validation harness for RegimeDetector (v2).

Replays historical market data day-by-day through RegimeDetector._classify()
(the same scoring function used live, no live network calls) and checks
whether the resulting regime classifications actually preceded the kind of
market behavior they claim to predict.

v2 vs v1: now computes ALL 9 signals from real data, given a
regime_backtest_data.csv built by build_regime_backtest_data_v2.py
(VIX + Treasury + SPX + VIX3M + TLT + UUP). v1 only had 4/9 (VIX-only).

Signal coverage in this version:
  vix_level, vix_trend, vix_percentile   <- VIX_History.csv (same as v1)
  yield_curve_slope                       <- Treasury par-yield CSV (same as v1)
  trend_strength (ADX-style)              <- SPX OHLC (NEW)
  hurst                                   <- SPX close series (NEW, uses the
                                              exact hurst_exponent() from
                                              options_bot.hurst). Verified
                                              correct directionally against
                                              realistic-noise synthetic data
                                              (trending=0.52, random walk=0.46,
                                              mean-reverting=0.31) -- an
                                              initial test with artificially
                                              noiseless data gave nonsensical
                                              results, but that's a degenerate
                                              edge case the formula's std()-of-
                                              differences method can't handle;
                                              real market data always has
                                              enough noise to avoid it.

  KNOWN CAVEAT -- trend_strength (ADX) has weak discrimination at realistic
  drift magnitudes: a 20-trial synthetic test (0.05%/day drift vs pure
  random walk, both at 1% daily vol, 20-day window -- matching the exact
  production formula in _compute_trend_strength()) found the trending and
  non-trending series statistically indistinguishable (mean 0.260 vs 0.262,
  trending "won" only 9/20 trials). This is the SAME formula currently live
  in production. It may simply need a stronger trend or longer window to
  show real discrimination, or realistic equity drift may genuinely be too
  subtle for a 20-bar ADX window to catch reliably. Worth re-testing against
  real historical trending periods (e.g. 2017 low-vol bull) vs choppy
  periods (e.g. 2022 range-bound bear) once SPX data is available, rather
  than relying on synthetic drift alone -- flagging here rather than silently
  treating this signal as more reliable than this test suggests it is.
  vix_term_structure                      <- VIX3M close vs VIX close (NEW,
                                              exact production formula:
                                              ratio = vix3m/vix, >1=contango)
  big_blue_day / capitulation /
  stock_bond_signal_trusted               <- SPX + TLT (NEW, exact production
                                              formula incl. the ECB-informed
                                              20d correlation trust gate)
  dollar_stress_day                       <- UUP + SPX (NEW, exact production
                                              formula)
  breadth_scores                          <- still neutralized (needs
                                              constituent-level data, not a
                                              single index — lowest priority
                                              of the 9, 1 signal out of 9)

If your CSV is missing spx_close/vix3m_close/tlt_close/uup_close columns
(i.e. you're still on the v1 merge), this script automatically falls back
to the v1-style neutralized indicators for whichever signals are missing —
it won't crash, it'll just tell you what got neutralized and why.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

sys.path.insert(0, "src")
from options_bot.regime import RegimeDetector
from options_bot.hurst import hurst_exponent, classify_regime as hurst_classify


def _adx_trend_strength(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Rolling 20-day ADX-style trend strength, vectorized version of the exact
    formula in RegimeDetector._compute_trend_strength(). Returns an array
    aligned with the input (NaN for the first 21 days where there isn't
    enough history).
    """
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(21, n):
        plus_dm_sum = minus_dm_sum = tr_sum = 0.0
        for j in range(i - 20, i):
            high_i, low_i, close_p = high[j + 1], low[j + 1], close[j]
            high_p, low_p = high[j], low[j]
            tr = max(high_i - low_i, abs(high_i - close_p), abs(low_i - close_p))
            tr_sum += tr
            up_move = high_i - high_p
            down_move = low_p - low_i
            if up_move > down_move and up_move > 0:
                plus_dm_sum += up_move
            if down_move > up_move and down_move > 0:
                minus_dm_sum += down_move
        if tr_sum == 0:
            out[i] = 0.5
            continue
        plus_di = (plus_dm_sum / tr_sum) * 100
        minus_di = (minus_dm_sum / tr_sum) * 100
        di_sum = plus_di + minus_di
        out[i] = min(1.0, max(0.0, abs(plus_di - minus_di) / di_sum)) if di_sum > 0 else 0.5
    return out


def _rolling_hurst(close: np.ndarray, window: int = 252) -> np.ndarray:
    """Rolling Hurst exponent using the exact production hurst_exponent()."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = hurst_exponent(close[i - window : i])
    return out


def build_indicator_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    has_spx = "spx_close" in df.columns and df["spx_close"].notna().sum() > 300
    has_vix3m = "vix3m_close" in df.columns and df["vix3m_close"].notna().sum() > 100
    has_tlt = "tlt_close" in df.columns and df["tlt_close"].notna().sum() > 300
    has_uup = "uup_close" in df.columns and df["uup_close"].notna().sum() > 300

    print("Signal coverage for this run:")
    print(f"  trend_strength / hurst : {'REAL (SPX)' if has_spx else 'NEUTRALIZED (no SPX data)'}")
    print(f"  vix_term_structure     : {'REAL (VIX3M)' if has_vix3m else 'NEUTRALIZED (no VIX3M data)'}")
    print(f"  stock_bond divergence  : {'REAL (TLT)' if (has_spx and has_tlt) else 'NEUTRALIZED (need SPX+TLT)'}")
    print(f"  dollar_stress           : {'REAL (UUP)' if (has_spx and has_uup) else 'NEUTRALIZED (need SPX+UUP)'}")
    print(f"  breadth                 : NEUTRALIZED (always — needs constituent data, not built yet)")
    print()

    # --- VIX trend (same as v1) ---
    df["vix_sma5"] = df["vix_close"].rolling(5, min_periods=1).mean()
    pct_vs_sma5 = (df["vix_close"] - df["vix_sma5"]) / df["vix_sma5"] * 100
    df["vix_trend"] = np.select(
        [pct_vs_sma5 > 10, pct_vs_sma5 < -10], ["rising", "falling"], default="stable",
    )

    # --- VIX percentile (same as v1) ---
    vix_vals = df["vix_close"].values
    pct = np.full(len(df), np.nan)
    for i in range(len(df)):
        window = vix_vals[max(0, i - 251) : i + 1]
        if len(window) >= 20:
            pct[i] = percentileofscore(window, vix_vals[i])
    df["vix_percentile"] = pct

    # --- Yield curve slope (same as v1) ---
    if "tsy_10yr" in df.columns and "tsy_2yr" in df.columns:
        df["yield_curve_slope"] = df["tsy_10yr"] - df["tsy_2yr"]
    else:
        df["yield_curve_slope"] = np.nan

    # --- trend_strength + hurst (NEW) ---
    if has_spx:
        df["trend_strength"] = _adx_trend_strength(
            df["spx_high"].values, df["spx_low"].values, df["spx_close"].values
        )
        df["hurst"] = _rolling_hurst(df["spx_close"].values)
    else:
        df["trend_strength"] = 0.5
        df["hurst"] = 0.5

    # --- VIX term structure via VIX3M (NEW, exact production formula) ---
    if has_vix3m:
        ratio = df["vix3m_close"] / df["vix_close"]
        df["vix_term_ratio"] = ratio
        df["vix_term_structure"] = np.where(ratio > 1.0, "contango",
                                     np.where(ratio < 1.0, "backwardation", "flat"))
    else:
        df["vix_term_ratio"] = 1.0
        df["vix_term_structure"] = "unknown"

    # --- Stock-bond divergence via TLT (NEW, exact production formula incl.
    #     the ECB-informed 20d correlation trust gate) ---
    if has_spx and has_tlt:
        spx_ret = df["spx_close"].pct_change()
        tlt_ret = df["tlt_close"].pct_change()
        spx_vol_avg20 = df["spx_volume"].rolling(20).mean().shift(1)
        spx_vol_ratio = df["spx_volume"] / spx_vol_avg20

        df["big_blue_day"] = (spx_ret < -0.01) & (tlt_ret > 0.01)
        df["capitulation"] = (spx_ret < -0.01) & (tlt_ret < 0.0) & (spx_vol_ratio > 1.5)

        corr_20d = spx_ret.rolling(20).corr(tlt_ret)
        df["stock_bond_correlation_20d"] = corr_20d
        df["stock_bond_signal_trusted"] = corr_20d < 0.0
    else:
        df["big_blue_day"] = False
        df["capitulation"] = False
        df["stock_bond_signal_trusted"] = False

    # --- Dollar stress via UUP (NEW, exact production formula) ---
    if has_spx and has_uup:
        spx_ret = df["spx_close"].pct_change()
        uup_ret = df["uup_close"].pct_change()
        df["dollar_stress_day"] = (uup_ret > 0.005) & (spx_ret < -0.01)
    else:
        df["dollar_stress_day"] = False

    return df


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    detector = RegimeDetector()
    regimes, confidences = [], []

    for _, row in df.iterrows():
        if pd.isna(row["vix_close"]) or pd.isna(row["vix_percentile"]):
            regimes.append(None)
            confidences.append(None)
            continue

        indicators = {
            "vix_level":          row["vix_close"],
            "vix_trend":          row["vix_trend"],
            "vix_percentile":     row["vix_percentile"],
            "yield_curve_slope":  row["yield_curve_slope"] if not pd.isna(row["yield_curve_slope"]) else 0.5,
            "trend_strength":     row["trend_strength"] if not pd.isna(row["trend_strength"]) else 0.5,
            "hurst":               row["hurst"] if not pd.isna(row["hurst"]) else 0.5,
            "_breadth_scores":     {},
            "vix_term_structure": row["vix_term_structure"],
            "vix_term_ratio":      row["vix_term_ratio"] if not pd.isna(row["vix_term_ratio"]) else 1.0,
            "big_blue_day":        bool(row["big_blue_day"]) if not pd.isna(row["big_blue_day"]) else False,
            "capitulation":        bool(row["capitulation"]) if not pd.isna(row["capitulation"]) else False,
            "stock_bond_signal_trusted": bool(row["stock_bond_signal_trusted"]) if not pd.isna(row["stock_bond_signal_trusted"]) else False,
            "dollar_stress_day":   bool(row["dollar_stress_day"]) if not pd.isna(row["dollar_stress_day"]) else False,
        }

        regime, confidence = detector._classify(indicators)
        regimes.append(regime)
        confidences.append(confidence)

    df = df.copy()
    df["regime"] = regimes
    df["confidence"] = confidences
    return df


def validate_against_forward_vix(df: pd.DataFrame, horizons=(5, 10, 20)) -> dict:
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


def validate_against_forward_spx(df: pd.DataFrame, horizons=(5, 10, 20)) -> dict:
    """
    NEW in v2: now that SPX is available, validate trending classification
    against actual forward SPX returns and realized volatility -- the real
    test "trending" should pass that wasn't possible in v1.
    """
    if "spx_close" not in df.columns or df["spx_close"].isna().all():
        return {}
    results = {}
    spx = df["spx_close"].values
    spx_ret = df["spx_close"].pct_change().values
    n = len(df)
    for horizon in horizons:
        fwd_ret = np.full(n, np.nan)
        fwd_vol = np.full(n, np.nan)
        for i in range(n - horizon):
            if not np.isnan(spx[i]) and not np.isnan(spx[i + horizon]):
                fwd_ret[i] = (spx[i + horizon] - spx[i]) / spx[i] * 100
            window = spx_ret[i + 1 : i + 1 + horizon]
            if len(window) == horizon and not np.any(np.isnan(window)):
                fwd_vol[i] = np.std(window) * np.sqrt(252) * 100
        tmp = df.copy()
        tmp[f"fwd_spx_ret_{horizon}d"] = fwd_ret
        tmp[f"fwd_spx_vol_{horizon}d"] = fwd_vol
        by_regime = tmp.groupby("regime").agg(
            n_days=("regime", "size"),
            mean_fwd_spx_return_pct=(f"fwd_spx_ret_{horizon}d", "mean"),
            mean_fwd_realized_vol_annualized_pct=(f"fwd_spx_vol_{horizon}d", "mean"),
        ).round(2)
        results[horizon] = by_regime
    return results


def main(csv_path: str):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df):,} rows ({df['date'].min().date()} to {df['date'].max().date()})\n")

    df = build_indicator_series(df)
    df = run_backtest(df)

    valid = df.dropna(subset=["regime"])
    print(f"Classified {len(valid):,} / {len(df):,} days")
    print("\nRegime distribution:")
    print(valid["regime"].value_counts())
    print(f"\nMean confidence: {valid['confidence'].mean():.3f}")

    print("\n" + "=" * 70)
    print("FORWARD VIX VALIDATION")
    print("=" * 70)
    for horizon, table in validate_against_forward_vix(valid).items():
        print(f"\n--- {horizon}-day forward horizon ---")
        print(table)

    spx_results = validate_against_forward_spx(valid)
    if spx_results:
        print("\n" + "=" * 70)
        print("FORWARD SPX VALIDATION (NEW in v2)")
        print("=" * 70)
        for horizon, table in spx_results.items():
            print(f"\n--- {horizon}-day forward horizon ---")
            print(table)

    out_path = csv_path.replace(".csv", "_classified.csv")
    valid.to_csv(out_path, index=False)
    print(f"\nSaved classified output to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest RegimeDetector against historical data (v2, full signal coverage)")
    parser.add_argument("csv_path", help="Path to merged regime_backtest_data.csv from build_regime_backtest_data_v2.py")
    args = parser.parse_args()
    main(args.csv_path)
