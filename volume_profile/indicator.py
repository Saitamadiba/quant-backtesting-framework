"""Volume profile — prior-session POC / value area / node density.

Builds a per-session volume-by-price histogram (bar volume spread uniformly
across each bar's [low, high]), then exposes the PRIOR completed session's
profile as today's causal reference: POC (mode of the volume distribution),
value area (70% of volume around POC -> VAH/VAL), and node_density(price)
for HVN (acceptance) / LVN (rejection) classification.

Session anchor = 00:00 UTC (crypto), matching the VWAP study. Everything is
causal: the profile used on session S is built from bars of session S-1.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

DUCKDB_PATH = os.path.join(_BASE, "duckdb_data", "trading_data.duckdb")


def load_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    df = con.execute(
        "SELECT timestamp, open, high, low, close, volume, atr_14, ema_200 "
        "FROM ohlcv_data WHERE symbol = ? AND timeframe = ? ORDER BY timestamp",
        [symbol, timeframe],
    ).df()
    con.close()
    if df.empty:
        return df
    df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
    df["timestamp"] = (pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
                       .astype("datetime64[ns, UTC]"))
    df["volume"] = df["volume"].clip(lower=0).fillna(0.0)
    return df


def _profile_one_session(h, l, v, n_bins, value_frac):
    """Return (poc, vah, val, lo, hi, bin_w, hist) for one session's arrays."""
    lo = float(np.min(l)); hi = float(np.max(h))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_w = (hi - lo) / n_bins
    centers = (edges[:-1] + edges[1:]) / 2
    hist = np.zeros(n_bins)
    for bh, bl, bv in zip(h, l, v):
        if bv <= 0 or not np.isfinite(bh) or not np.isfinite(bl) or bh < bl:
            continue
        lo_i = int(np.clip((bl - lo) / bin_w, 0, n_bins - 1))
        hi_i = int(np.clip((bh - lo) / bin_w, 0, n_bins - 1))
        span = hi_i - lo_i + 1
        hist[lo_i:hi_i + 1] += bv / span
    total = hist.sum()
    if total <= 0:
        return None
    poc_i = int(np.argmax(hist))
    # expand value area from POC until value_frac of volume captured
    lo_i = hi_i = poc_i
    acc = hist[poc_i]
    while acc < value_frac * total and (lo_i > 0 or hi_i < n_bins - 1):
        below = hist[lo_i - 1] if lo_i > 0 else -1
        above = hist[hi_i + 1] if hi_i < n_bins - 1 else -1
        if above >= below:
            hi_i += 1; acc += hist[hi_i]
        else:
            lo_i -= 1; acc += hist[lo_i]
    return {
        "poc": float(centers[poc_i]), "vah": float(edges[hi_i + 1]),
        "val": float(edges[lo_i]), "lo": lo, "hi": hi, "bin_w": bin_w,
        "hist": hist, "centers": centers, "mean_bin": total / max((hist > 0).sum(), 1),
    }


def session_profiles(symbol: str, timeframe: str, n_bins: int = 50,
                     value_frac: float = 0.70) -> pd.DataFrame:
    """One row per UTC session: PRIOR session's profile as the causal reference
    for THIS session. Columns: date, pPOC, pVAH, pVAL, plus the raw prior
    hist/centers/mean_bin for node-density lookups."""
    df = load_bars(symbol, timeframe)
    if df.empty:
        return pd.DataFrame()
    df["sess"] = df["timestamp"].dt.floor("D")
    profs = {}
    for sess, g in df.groupby("sess"):
        p = _profile_one_session(g["high"].to_numpy(float), g["low"].to_numpy(float),
                                 g["volume"].to_numpy(float), n_bins, value_frac)
        if p is not None:
            profs[sess] = p
    sessions = sorted(profs.keys())
    rows = []
    for k in range(1, len(sessions)):
        prev = profs[sessions[k - 1]]
        rows.append({
            "sess": sessions[k], "pPOC": prev["poc"], "pVAH": prev["vah"],
            "pVAL": prev["val"], "plo": prev["lo"], "phi": prev["hi"],
            "pbin_w": prev["bin_w"], "phist": prev["hist"],
            "pcenters": prev["centers"], "pmean_bin": prev["mean_bin"],
        })
    return pd.DataFrame(rows)


def node_density(price: float, row) -> float:
    """Volume in the prior-session bin at `price`, / mean bin volume.
    >1 = HVN (acceptance/magnet); <1 = LVN (rejection); 0 = outside prior
    range (single-print / fast-move territory)."""
    if not np.isfinite(price):
        return np.nan
    lo, hi, bw = row["plo"], row["phi"], row["pbin_w"]
    if price < lo or price >= hi or bw <= 0:
        return 0.0
    hist = row["phist"]
    idx = int(np.clip((price - lo) / bw, 0, len(hist) - 1))
    mb = row["pmean_bin"]
    return float(hist[idx] / mb) if mb > 0 else np.nan


def vp_context(price: float, atr: float, row) -> dict:
    """Prior-session VP context at `price`: distance to pPOC (ATR), value zone,
    node density."""
    ppoc, pvah, pval = row["pPOC"], row["pVAH"], row["pVAL"]
    if price > pvah:
        zone = "above_value"
    elif price < pval:
        zone = "below_value"
    else:
        zone = "in_value"
    return {
        "dist_pPOC_atr": (price - ppoc) / atr if atr > 0 else np.nan,
        "vp_zone": zone,
        "node_density": node_density(price, row),
        "pPOC": ppoc, "pVAH": pvah, "pVAL": pval,
    }
