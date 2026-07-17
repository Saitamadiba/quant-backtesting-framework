"""Attach regime5 (own-asset + BTC) to ORB+FVG trades. Mirrors ny4h/regimes.

Rule-based classifier on 15m bars, thresholds percentile-calibrated per
asset (p43/p75/p93 of that asset's own atr14_pct). Attachment is causal:
last 15m bar whose close time <= entry time.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from regime_classifier import (  # noqa: E402
    RuleThresholds, classify_rule_based, compute_features,
)
from orb_fvg.engine import load_bars  # noqa: E402

_REGIME_CACHE: dict[str, pd.DataFrame] = {}


def regime_series(symbol: str) -> pd.DataFrame:
    if symbol in _REGIME_CACHE:
        return _REGIME_CACHE[symbol]
    df = load_bars(symbol, "15m")
    if df.empty:
        _REGIME_CACHE[symbol] = pd.DataFrame({"time": [], "regime5": []})
        return _REGIME_CACHE[symbol]
    feats = compute_features(df)
    atrp = feats["atr14_pct"].dropna()
    thr = RuleThresholds(
        quiet_max=float(atrp.quantile(0.43)),
        moderate_max=float(atrp.quantile(0.75)),
        elevated_max=float(atrp.quantile(0.93)),
    )
    labels = classify_rule_based(feats, thr)
    out = pd.DataFrame({"time": df["timestamp"] + pd.Timedelta(minutes=15),
                        "regime5": labels.to_numpy()})
    _REGIME_CACHE[symbol] = out
    return out


def attach_regimes(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    btc = regime_series("BTC").rename(columns={"regime5": "btc_regime5"})
    parts = []
    for sym, grp in trades.groupby("symbol"):
        grp = grp.sort_values("entry_time").copy()
        own = regime_series(sym)
        if own.empty:
            grp["regime5"] = None
        else:
            grp["regime5"] = pd.merge_asof(
                grp[["entry_time"]], own, left_on="entry_time", right_on="time",
                direction="backward")["regime5"].to_numpy()
        if btc.empty:
            grp["btc_regime5"] = None
        else:
            grp["btc_regime5"] = pd.merge_asof(
                grp[["entry_time"]], btc, left_on="entry_time", right_on="time",
                direction="backward")["btc_regime5"].to_numpy()
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)
