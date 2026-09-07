"""Evaluation battery for the session-VWAP value-area reversion study.

Day-clustered inference throughout (same-session trades are one bet wearing
several name tags). Wild cluster bootstrap for the family-wise bar.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
from vwap_value_area.engine import DATA_DIR, CRYPTO                      # noqa: E402

TRD = os.path.join(DATA_DIR, "trades")
RNG = np.random.default_rng(20260907)


def load(symbols) -> pd.DataFrame:
    fr = []
    for s in symbols:
        p = os.path.join(TRD, f"{s}.parquet")
        if os.path.exists(p):
            fr.append(pd.read_parquet(p))
    df = pd.concat(fr, ignore_index=True)
    df["cluster"] = df["symbol"].astype(str) + "|" + df["date"].astype(str)
    return df


def clustered_t(x: np.ndarray, cl: np.ndarray):
    """Cluster-robust t for the mean against zero."""
    n = len(x)
    if n < 10:
        return np.nan, np.nan, 0
    m = x.mean(); e = x - m
    s = pd.Series(e).groupby(cl).sum().to_numpy()
    var = (s ** 2).sum() / (n ** 2)
    se = np.sqrt(var)
    return (m / se if se > 0 else np.nan), se, len(s)


def cell_stats(d: pd.DataFrame, col="gross_r"):
    x = d[col].to_numpy(float); cl = d["cluster"].to_numpy()
    t, se, ng = clustered_t(x, cl)
    return dict(n=len(d), n_days=ng, mean=x.mean(), t=t,
                wr=(x > 0).mean() * 100, sum=x.sum())


def grid(df: pd.DataFrame, col="gross_r") -> pd.DataFrame:
    rows = []
    for (cf, tg, bd), d in df.groupby(["confirm", "target", "bands"]):
        r = cell_stats(d, col); r.update(confirm=cf, target=tg, bands=bd)
        rows.append(r)
    return pd.DataFrame(rows)[["confirm", "target", "bands", "n", "n_days",
                               "mean", "t", "wr", "sum"]]


def wild_cluster_max_t(df: pd.DataFrame, cells, col="gross_r", B=2000):
    """Null distribution of max|t| across the arm family (wild cluster bootstrap,
    Rademacher weights at the day level). Gives the family-wise bar."""
    prepared = []
    for key in cells:
        d = df[(df.confirm == key[0]) & (df.target == key[1]) & (df.bands == key[2])]
        x = d[col].to_numpy(float)
        codes, uniq = pd.factorize(d["cluster"])
        e = x - x.mean()
        prepared.append((len(x), e, codes, len(uniq)))
    maxt = np.empty(B)
    for b in range(B):
        best = 0.0
        for n, e, codes, ng in prepared:
            w = RNG.choice([-1.0, 1.0], size=ng)[codes]
            eb = e * w
            m = eb.mean()
            s = np.bincount(codes, weights=eb - m, minlength=ng)
            se = np.sqrt((s ** 2).sum()) / n
            if se > 0:
                best = max(best, abs(m / se))
        maxt[b] = best
    return maxt


def halves(df: pd.DataFrame, col="gross_r"):
    med = pd.to_datetime(df["date"]).median()
    a = df[pd.to_datetime(df["date"]) <= med]; b = df[pd.to_datetime(df["date"]) > med]
    return cell_stats(a, col), cell_stats(b, col)


def er_response(df: pd.DataFrame, col="gross_r", feat="pre_er20", q=5, placebo=False):
    d = df.dropna(subset=[feat]).copy()
    v = d[feat].to_numpy(float)
    if placebo:
        v = RNG.permutation(v)
    d["_bin"] = pd.qcut(v, q, labels=False, duplicates="drop")
    rows = []
    for b, g in d.groupby("_bin"):
        r = cell_stats(g, col); r["bin"] = int(b)
        r["feat_lo"] = g[feat].min() if not placebo else np.nan
        r["feat_hi"] = g[feat].max() if not placebo else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame, cols=None) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
    return d.to_string(index=False)
