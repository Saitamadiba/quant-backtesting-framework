"""Analyze the VWAP-confluence battery. Keystones: (1) does VWAP-anchoring
beat the MA-control? (2) is the real gross edge fee-walled, and does 15m lift
it? (3) does confluence stacking A->B->C help? (4) is reversion dead?
Gross/net decomposition leads; family-wise bar on any winning cell.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
OUT = os.path.join(_BASE, "reports", "vwap_confluence")
NQ = "NQ"


def ct(t, col="gross_r"):
    if t.empty:
        return (np.nan, np.nan, 0)
    g = t[col].to_numpy(float)
    cl = t.groupby([t.symbol, t.ny_date])[col].mean()
    n = len(cl)
    if n < 2 or cl.std() == 0:
        return (float(g.mean()), np.nan, n)
    return (float(g.mean()), float(cl.mean() / (cl.std(ddof=1) / np.sqrt(n))), n)


def line(name, t):
    if t.empty:
        return f"{name:26s} n=0"
    gm, gt, nd = ct(t)
    return (f"{name:26s} n={len(t):6d} gross {gm:+.4f} (t {gt:+.2f}, {nd}d) "
            f"net_tk {t.net_taker_r.mean():+.3f} net_mm {t.net_mm_r.mean():+.3f} "
            f"win {(t.gross_r>0).mean():.3f} tp% {t.exit_reason.isin(['tp','tp_gap']).mean():.3f}")


def auc(x, y):
    m = np.isfinite(x); x, y = x[m], y[m]
    if len(np.unique(y)) < 2 or len(x) < 30:
        return np.nan
    order = np.argsort(x); r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


FEATURES = ["vwap_dist_atr", "vwap_slope_atr", "pullback_depth_atr", "retrace_frac",
            "sd_pos", "confirm_body_ratio", "dist_ema200_pct", "stop_atr",
            "bars_since_anchor", "break_vol_ratio", "entry_hour_ny"]


def load(name):
    p = os.path.join(OUT, f"trades_{name}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()


def report():
    A = load("A"); B = load("B"); C = load("C"); MA = load("MA"); D = load("D"); A15 = load("A15")
    L = []; P = L.append
    P("=" * 100)
    P("VWAP CONFLUENCE (Tom Crown) — results (Tier-3 measurement)")
    P("=" * 100)

    P("\n## Headline — arm A (pullback-to-VWAP + candle, the video's core)")
    P(line("ALL crypto+NQ", A))
    P(line("crypto", A[A.symbol != NQ]))
    P(line("NQ (native)", A[A.symbol == NQ]))
    P(line("  longs", A[A.side == "long"]))
    P(line("  shorts", A[A.side == "short"]))

    P("\n## KEYSTONE 1 — does VWAP-anchoring beat a plain EMA pullback? (A vs MA-control)")
    P(line("A  (pullback to VWAP)", A))
    P(line("MA (pullback to EMA100)", MA))
    dg = A.gross_r.mean() - MA.gross_r.mean()
    P(f"   VWAP edge over MA-control: {dg:+.4f}R gross  (if ~0, VWAP is just a line)")

    P("\n## KEYSTONE 2 — is the gross edge fee-walled? does 15m lift net? (scale)")
    P(line("A  @native (5m alts)", A[A.symbol.isin(A15.symbol.unique())] if not A15.empty else A))
    P(line("A  @15m (10 alts)", A15))

    P("\n## Confluence stacking A -> B -> C (does adding gates lift gross?)")
    P(line("A  VWAP only", A))
    P(line("B  A + HTF trend", B))
    P(line("C  A + weekly-VWAP", C))

    P("\n## Reversion arm D (SD-band fade -> VWAP); prior: fade space closed")
    P(line("D  all", D))
    P(line("D  crypto", D[D.symbol != NQ]))
    P(line("D  NQ", D[D.symbol == NQ]))

    P("\n## Per-asset (arm A)")
    for s in sorted(A.symbol.unique()):
        P(line(s, A[A.symbol == s]))

    P("\n## Regime map (own regime5, arm A, gross R)")
    if A.regime5.notna().any():
        for rg, grp in A[A.regime5.notna()].groupby("regime5"):
            gm, gt, nd = ct(grp)
            P(f"  {rg:16s} n={len(grp):6d} gross {gm:+.4f} (t {gt:+.2f}) net_tk {grp.net_taker_r.mean():+.3f}")

    P("\n## Winner/loser AUC (PRE-fill; arm A; win=tp/tp_gap; near 0.5 = no separation)")
    y = A.exit_reason.isin(["tp", "tp_gap"]).to_numpy().astype(float)
    rows = []
    for f in FEATURES:
        if f not in A:
            continue
        a = auc(A[f].to_numpy(float), y)
        if not np.isnan(a):
            rows.append((abs(a - 0.5), f, a))
    rows.sort(reverse=True)
    for _, f, a in rows:
        P(f"  {f:22s} AUC {a:.3f}")

    P("\n## Half-split stability (arm A, by time)")
    ps = A.sort_values("entry_time"); mid = len(ps) // 2
    for half, sub in [("H1", ps.iloc[:mid]), ("H2", ps.iloc[mid:])]:
        gm, gt, nd = ct(sub)
        P(f"  {half}: gross {gm:+.4f} (t {gt:+.2f}) net_tk {sub.net_taker_r.mean():+.3f}")

    P("\n## Exit-reason mix (arm A): " + str(A.exit_reason.value_counts().to_dict()))
    return "\n".join(L)


def main():
    txt = report()
    print(txt)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
