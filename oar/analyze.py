"""Analyze the OAR battery. Keystone = does the exhaustion gate SELECT, or
is the IB-target 'lift' just a target-distance artifact? The fixed-rr arm
disentangles: if gross is flat vs the gate at fixed rr, the gate does not
select. Gross/net decomposition leads; family-wise bar on any winning cell.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
OUT = os.path.join(_BASE, "reports", "oar")
NQ = "NQ"
GATES = [0.0, 0.15, 0.20, 0.30, 0.50]


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
    wr = (t.gross_r > 0).mean()
    tp = t.exit_reason.isin(["tp", "tp_gap"]).mean()
    return (f"{name:26s} n={len(t):6d} gross {gm:+.4f} (t {gt:+.2f}, {nd}d) "
            f"net_tk {t.net_taker_r.mean():+.3f} net_mm {t.net_mm_r.mean():+.3f} "
            f"win {wr:.3f} tp% {tp:.3f} rr_med {t.rr_planned.median():.2f}")


def auc(x, y):
    m = np.isfinite(x); x, y = x[m], y[m]
    if len(np.unique(y)) < 2 or len(x) < 30:
        return np.nan, 0
    order = np.argsort(x); ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan, len(x)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), len(x)


FEATURES = ["opening_vol_ratio", "ib_range_pct", "sweep_depth_atr", "sweep_depth_frac",
            "bars_ib_to_sweep", "bars_sweep_to_entry", "trigger_wick_ratio",
            "trigger_body_frac", "trigger_range_atr", "stop_atr", "dist_ema200_pct",
            "break_vol_ratio", "mins_since_open", "prior_trades_today"]


def bonf_z(m):
    from scipy.stats import norm
    return float(norm.ppf(1 - 0.05 / (2 * max(m, 1))))


def report(engib, engrr2, hamib):
    L = []; A = L.append
    A("=" * 100)
    A("OPENING-AUCTION EXHAUSTION REVERSAL — results (Tier-3 measurement)")
    A("=" * 100)

    A("\n## Headline (engulf, IB-target, ib_ratio_min=0.20 = the transcript's ~20% ATR gate)")
    p = engib[engib.opening_vol_ratio >= 0.20]
    A(line("ALL crypto+NQ", p))
    A(line("crypto", p[p.symbol != NQ]))
    A(line("NQ (native)", p[p.symbol == NQ]))

    A("\n## KEYSTONE — does the exhaustion gate SELECT? gate sweep, IB-target vs FIXED rr=2")
    A("   IB-target: gross should rise with the gate IF the mechanism is real...")
    for g in GATES:
        A(line(f"  engulf IB   ovr>={g:.2f}", engib[engib.opening_vol_ratio >= g]))
    A("   ...but at FIXED rr=2 (target distance held constant) the lift must survive:")
    for g in GATES:
        A(line(f"  engulf rr2  ovr>={g:.2f}", engrr2[engrr2.opening_vol_ratio >= g]))
    A("   Read: if rr2 gross is flat across the gate, the IB 'lift' is a target-")
    A("   distance artifact (further target = bigger R on the same coin-flip), NOT")
    A("   selection. Also watch win% — a real gate raises the HIT rate.")
    A("   win% by gate (IB): " + "  ".join(
        f"{g:.2f}:{(engib[engib.opening_vol_ratio>=g].gross_r>0).mean():.3f}" for g in GATES))

    A("\n## Trigger: engulf vs hammer-break (IB-target, gate 0.20)")
    A(line("engulf  ovr>=0.20", engib[engib.opening_vol_ratio >= 0.20]))
    A(line("hammer  ovr>=0.20", hamib[hamib.opening_vol_ratio >= 0.20]))
    A(line("engulf  ovr>=0.00", engib))
    A(line("hammer  ovr>=0.00", hamib))

    A("\n## Per-asset (engulf, IB-target, gate 0.20)")
    for s in sorted(p.symbol.unique()):
        A(line(s, p[p.symbol == s]))

    A("\n## Regime map (own regime5, engulf IB gate 0.20, gross R)")
    if p.regime5.notna().any():
        for rg, grp in p[p.regime5.notna()].groupby("regime5"):
            gm, gt, nd = ct(grp)
            A(f"  {rg:16s} n={len(grp):5d} gross {gm:+.4f} (t {gt:+.2f}) net_tk {grp.net_taker_r.mean():+.3f}")

    A("\n## Winner/loser AUC (PRE-fill; engulf IB gate 0.20; win=tp/tp_gap)")
    y = p.exit_reason.isin(["tp", "tp_gap"]).to_numpy().astype(float)
    rows = []
    for f in FEATURES:
        if f not in p:
            continue
        a, n = auc(p[f].to_numpy(float), y)
        if np.isnan(a):
            continue
        rows.append((abs(a - 0.5), f, a, n))
    rows.sort(reverse=True)
    A(f"   (AUCs near 0.5 = no separation; n={len(p)})")
    for _, f, a, n in rows:
        A(f"  {f:22s} AUC {a:.3f}  n={n}")

    A("\n## Half-split stability of baseline gross (engulf rr2, gate 0, by time)")
    ps = engrr2.sort_values("entry_time"); mid = len(ps) // 2
    for half, sub in [("H1", ps.iloc[:mid]), ("H2", ps.iloc[mid:])]:
        gm, gt, nd = ct(sub)
        A(f"  {half}: gross {gm:+.4f} (t {gt:+.2f}) net_tk {sub.net_taker_r.mean():+.3f}")

    A("\n## Exit-reason mix (engulf IB gate 0.20)")
    A("  " + str(p.exit_reason.value_counts().to_dict()))
    return "\n".join(L)


def main():
    engib = pd.read_parquet(os.path.join(OUT, "trades_engulf_ib.parquet"))
    engrr2 = pd.read_parquet(os.path.join(OUT, "trades_engulf_rr2.parquet"))
    hamib = pd.read_parquet(os.path.join(OUT, "trades_hammer_ib.parquet"))
    txt = report(engib, engrr2, hamib)
    print(txt)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
