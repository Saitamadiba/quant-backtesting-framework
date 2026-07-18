"""Uniform fair-value context provider.

For each symbol, on a FIXED 15m reference grid (so the context is comparable
across strategies regardless of their exec TF), compute causally:
  - session VWAP (00:00 UTC anchor) + ATR15,
  - prior-session volume profile (POC / VAH / VAL / node-density).
Then annotate ANY strategy trade-set (symbol, entry_time, entry, side) with:
  dist_vwap_atr (signed by side: + = entry beyond VWAP in trade dir),
  dist_pPOC_atr (|.| = how far from the prior-day POC magnet),
  node_density (HVN>1 / LVN<1 / 0 outside prior range),
  vp_zone (in_value / above_value / below_value).

NQ is excluded (corrupt multi-scale data, per the VWAP/VP studies).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
from volume_profile.indicator import load_bars, session_profiles, node_density  # noqa: E402

REF_TF = "15m"
_CTX_CACHE: dict = {}


def _session_vwap(df: pd.DataFrame):
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy(float)
    vol = df["volume"].to_numpy(float)
    grp = pd.factorize(df["timestamp"].dt.floor("D"))[0]
    pv = tp * vol
    out = np.empty(len(df)); cum_pv = 0.0; cum_v = 0.0; cur = -1
    for i in range(len(df)):
        if grp[i] != cur:
            cur = grp[i]; cum_pv = 0.0; cum_v = 0.0
        cum_pv += pv[i]; cum_v += vol[i]
        out[i] = cum_pv / cum_v if cum_v > 0 else tp[i]
    return out


def _provider(symbol: str):
    if symbol in _CTX_CACHE:
        return _CTX_CACHE[symbol]
    df = load_bars(symbol, REF_TF)
    if df.empty:
        _CTX_CACHE[symbol] = None
        return None
    vwap = _session_vwap(df)
    vw = pd.DataFrame({"time": df["timestamp"], "vwap": vwap,
                       "atr": df["atr_14"].to_numpy(float)}).sort_values("time")
    sp = session_profiles(symbol, REF_TF)
    prof = {r["sess"]: r for _, r in sp.iterrows()} if not sp.empty else {}
    _CTX_CACHE[symbol] = (vw, prof)
    return _CTX_CACHE[symbol]


def annotate(trades: pd.DataFrame) -> pd.DataFrame:
    """Return trades (NQ dropped) with normalized net + fair-value context."""
    if trades.empty:
        return trades
    t = trades[trades.symbol != "NQ"].copy()
    # normalize net columns across schemas
    if "net_taker_r" in t:
        t["net"] = t["net_taker_r"]
    elif "net_r" in t:
        t["net"] = t["net_r"]
    else:
        t["net"] = np.nan
    t["net_mm"] = t["net_mm_r"] if "net_mm_r" in t else t["net"]
    t["side_sign"] = np.where(t["side"].astype(str).str.lower().str.startswith("l"), 1, -1)
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)

    parts = []
    for sym, g in t.groupby("symbol"):
        prov = _provider(sym)
        if prov is None:
            continue
        vw, prof = prov
        g = g.sort_values("entry_time").copy()
        m = pd.merge_asof(g[["entry_time"]], vw, left_on="entry_time",
                          right_on="time", direction="backward")
        g["_vwap"] = m["vwap"].to_numpy(); g["_atr"] = m["atr"].to_numpy()
        skey = g["entry_time"].dt.floor("D")  # tz-aware UTC, matches session_profiles keys
        dvw, dppoc, nd, zone = [], [], [], []
        for (_, tr), sk in zip(g.iterrows(), skey):
            a = tr["_atr"]; px = tr["entry"]; ss = tr["side_sign"]
            vwp = tr["_vwap"]
            dvw.append((px - vwp) / a * ss if a and a > 0 and np.isfinite(vwp) else np.nan)
            r = prof.get(sk)
            if r is None or not (a and a > 0):
                dppoc.append(np.nan); nd.append(np.nan); zone.append("none"); continue
            dppoc.append((px - r["pPOC"]) / a)
            nd.append(node_density(px, r))
            zone.append("above_value" if px > r["pVAH"] else ("below_value" if px < r["pVAL"] else "in_value"))
        g["dist_vwap_atr"] = dvw
        g["dist_pPOC_atr"] = dppoc
        g["node_density"] = nd
        g["vp_zone"] = zone
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out
