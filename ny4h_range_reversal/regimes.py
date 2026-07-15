"""Attach regime5 labels (own-asset + BTC) to NY4H trades.

Uses the canonical `regime_classifier.py` rule-based classifier on 15m
bars, with thresholds percentile-calibrated per asset (p43 / p75 / p93 of
that asset's own atr14_pct distribution — the documented calibration
recipe from the 2026-05-08 regime work). Attachment is causal: each trade
gets the label of the last 15m bar whose CLOSE time <= entry time.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from regime_classifier import (  # noqa: E402
    RuleThresholds,
    classify_rule_based,
    compute_features,
)
from ny4h_range_reversal.engine import load_bars  # noqa: E402

_REGIME_CACHE: dict[str, pd.DataFrame] = {}


def regime_series(symbol: str) -> pd.DataFrame:
    """Per-15m-bar regime5 labels for `symbol`, keyed by bar close time."""
    if symbol in _REGIME_CACHE:
        return _REGIME_CACHE[symbol]
    df = load_bars(symbol, "15m")
    feats = compute_features(df)
    atrp = feats["atr14_pct"].dropna()
    thr = RuleThresholds(
        quiet_max=float(atrp.quantile(0.43)),
        moderate_max=float(atrp.quantile(0.75)),
        elevated_max=float(atrp.quantile(0.93)),
    )
    labels = classify_rule_based(feats, thr)
    out = pd.DataFrame({
        "time": df["timestamp"] + pd.Timedelta(minutes=15),
        "regime5": labels.to_numpy(),
    })
    _REGIME_CACHE[symbol] = out
    return out


def attach_regimes(trades: pd.DataFrame) -> pd.DataFrame:
    """Add `regime5` (own asset) and `btc_regime5` columns to a trade frame."""
    if trades.empty:
        return trades
    btc = regime_series("BTC").rename(columns={"regime5": "btc_regime5"})
    parts = []
    for sym, grp in trades.groupby("symbol"):
        grp = grp.sort_values("entry_time").copy()
        own = regime_series(sym)
        grp["regime5"] = pd.merge_asof(
            grp[["entry_time"]], own, left_on="entry_time", right_on="time",
            direction="backward",
        )["regime5"].to_numpy()
        grp["btc_regime5"] = pd.merge_asof(
            grp[["entry_time"]], btc, left_on="entry_time", right_on="time",
            direction="backward",
        )["btc_regime5"].to_numpy()
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)
