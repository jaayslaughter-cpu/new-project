"""Realized-volatility estimators — annualized RV from OHLC bars.

Reframed onto data we already have: every estimator here needs only OHLC bars,
which we already pull from Alpaca / yfinance. No ORATS, no subscription.

These are the standard public-domain estimators (close-to-close, Parkinson,
Garman-Klass, Rogers-Satchell, Yang-Zhang). Yang-Zhang is the default: it is
drift-independent and handles overnight gaps, which matters for ETFs that gap.

Purpose: give the VRP gate (see vrp_gate.py) a defensible realized-vol baseline
to compare implied vol against. We score IV *quality* today but never check
whether IV is actually rich relative to what the underlying is *realizing*.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
TRADING_DAYS = 252
def _arr(x): return np.asarray(x, dtype=float)
def rv_close_to_close(close, window=21, ann=TRADING_DAYS):
    c = _arr(close)
    if len(c) < window + 1: return None
    rets = np.diff(np.log(c))
    if len(rets) < window: return None
    var = np.sum(rets[-window:] ** 2) * (ann / window)
    return float(np.sqrt(var))
def rv_parkinson(high, low, window=21, ann=TRADING_DAYS):
    h, l = _arr(high), _arr(low)
    if len(h) < window or len(l) < window: return None
    rng = np.log(h / l) ** 2
    var = np.sum(rng[-window:]) * (ann / (4 * np.log(2) * window))
    return float(np.sqrt(var))
def rv_garman_klass(open_, high, low, close, window=21, ann=TRADING_DAYS):
    o, h, l, c = _arr(open_), _arr(high), _arr(low), _arr(close)
    if min(len(o), len(h), len(l), len(c)) < window: return None
    rs = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    var = np.sum(rs[-window:]) * (ann / window)
    return float(np.sqrt(max(var, 0.0)))
def rv_rogers_satchell(open_, high, low, close, window=21, ann=TRADING_DAYS):
    o, h, l, c = _arr(open_), _arr(high), _arr(low), _arr(close)
    if min(len(o), len(h), len(l), len(c)) < window: return None
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    var = np.sum(rs[-window:]) * (ann / window)
    return float(np.sqrt(max(var, 0.0)))
def rv_yang_zhang(open_, high, low, close, window=21, ann=TRADING_DAYS, k=0.34):
    o, h, l, c = _arr(open_), _arr(high), _arr(low), _arr(close)
    if min(len(o), len(h), len(l), len(c)) < window + 1: return None
    oc = np.log(o[1:] / c[:-1]); co = np.log(c[1:] / o[1:])
    rs = (np.log(h[1:] / c[1:]) * np.log(h[1:] / o[1:]) + np.log(l[1:] / c[1:]) * np.log(l[1:] / o[1:]))
    if len(oc) < window: return None
    oc_w, co_w, rs_w = oc[-window:], co[-window:], rs[-window:]
    var_o = np.var(oc_w, ddof=1); var_c = np.var(co_w, ddof=1); var_rs = np.sum(rs_w) / window
    yz_var = (var_o + k * var_c + (1 - k) * var_rs) * ann
    return float(np.sqrt(max(yz_var, 0.0)))
ESTIMATORS = {"close_to_close": rv_close_to_close, "parkinson": rv_parkinson,
    "garman_klass": rv_garman_klass, "rogers_satchell": rv_rogers_satchell, "yang_zhang": rv_yang_zhang}
