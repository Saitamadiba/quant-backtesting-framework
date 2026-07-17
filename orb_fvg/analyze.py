"""Analyze the ORB+FVG battery: gross-vs-net decomposition, per-asset,
per-regime, winner/loser AUC, family-wise bar, half-split stability.

Headline discipline (CLAUDE.md): lead with the gross/net split; every
multi-cell scan carries a Bonferroni/permutation bar; label PRE vs POST
fill (all recorded features here are PRE-fill entry-time).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
OUT = os.path.join(_BASE, "reports", "orb_fvg")

NQ = "NQ"


def clustered_t(trades: pd.DataFrame, col: str = "gross_r") -> tuple[float, float, int]:
    """Mean and day-clustered t-stat (each NY day = one cluster / bet)."""
    if trades.empty:
        return (np.nan, np.nan, 0)
    g = trades[col].to_numpy(float)
    cl = trades.groupby([trades.symbol, trades.ny_date])[col].mean()
    n = len(cl)
    if n < 2 or cl.std() == 0:
        return (float(g.mean()), np.nan, n)
    t = cl.mean() / (cl.std(ddof=1) / np.sqrt(n))
    return (float(g.mean()), float(t), n)


def line(name: str, t: pd.DataFrame) -> str:
    if t.empty:
        return f"{name:22s} n=0"
    gm, gt, nd = clustered_t(t, "gross_r")
    ntk = t.net_taker_r.mean()
    nmm = t.net_mm_r.mean()
    wr = (t.gross_r > 0).mean()
    fee = t.fee_r.median()
    tp = (t.exit_reason.isin(["tp", "tp_gap"])).mean()
    return (f"{name:22s} n={len(t):6d}  gross {gm:+.4f} (clust-t {gt:+.2f}, "
            f"{nd}d)  net_tk {ntk:+.3f}  net_mm {nmm:+.3f}  win {wr:.3f}  "
            f"tp% {tp:.3f}  feeR_med {fee:.2f}")


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUC of feature x separating winners (y=1) vs losers (y=0)."""
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if len(np.unique(y)) < 2 or len(x) < 30:
        return np.nan
    order = np.argsort(x)
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


FEATURES = [
    "or_range_pct", "or_range_vs_atr15", "break_close_margin_atr",
    "bars_break_to_fvg", "fvg_height_atr", "fvg_depth_frac",
    "bars_fvg_to_engulf", "retest_leg_bars", "engulf_body_ratio",
    "engulf_range_atr", "stop_pct", "stop_atr", "dist_ema200_pct",
    "break_vol_ratio", "mins_since_open", "prior_trades_today",
]


def report(prim, noeng, scale15, rr2) -> str:
    L = []
    A = L.append
    A("=" * 96)
    A("ORB + FVG-retest + engulfing — results (Tier-3 measurement)")
    A("=" * 96)

    cr = prim[prim.symbol != NQ]
    nq = prim[prim.symbol == NQ]
    A("\n## Headline (primary arm: engulf-confirmed, rr=3)")
    A(line("ALL (crypto+NQ)", prim))
    A(line("crypto (10@5m,ETH/SOL@15m)", cr))
    A(line("NQ @15m (NATIVE asset)", nq))
    A(line("  longs", prim[prim.side == "long"]))
    A(line("  shorts", prim[prim.side == "short"]))

    A("\n## Per-asset (primary)")
    for s in sorted(prim.symbol.unique()):
        A(line(s, prim[prim.symbol == s]))

    A("\n## Prereg Q3 — does the engulfing gate select? (vs no-confirm)")
    A(line("engulf (primary)", prim))
    A(line("no-engulf (retest only)", noeng))
    A(line("  NQ engulf", nq))
    A(line("  NQ no-engulf", noeng[noeng.symbol == NQ]))

    A("\n## Scale-invariance (structure-scalper lesson): 10 alts @15m vs @5m")
    A(line("10 alts @5m (primary)", cr[cr.symbol.isin(scale15.symbol.unique())]))
    A(line("10 alts @15m", scale15))

    A("\n## Target sensitivity: rr=3 vs rr=2")
    A(line("rr=3 (primary)", prim))
    A(line("rr=2", rr2))

    A("\n## Regime map (own regime5, primary, gross R)")
    if "regime5" in prim and prim.regime5.notna().any():
        for rg, grp in prim[prim.regime5.notna()].groupby("regime5"):
            gm, gt, nd = clustered_t(grp, "gross_r")
            A(f"  {rg:16s} n={len(grp):6d}  gross {gm:+.4f} (clust-t {gt:+.2f})  "
              f"net_tk {grp.net_taker_r.mean():+.3f}")

    A("\n## Winner/loser AUC scan (PRE-fill features; gross win = tp/tp_gap)")
    A("   (family-wise bar: Bonferroni |z|>%.2f for %d features @ 0.05 two-sided)"
      % (_bonf_z(len(FEATURES)), len(FEATURES)))
    y = prim.exit_reason.isin(["tp", "tp_gap"]).to_numpy().astype(float)
    rows = []
    for f in FEATURES:
        if f not in prim:
            continue
        a = auc(prim[f].to_numpy(float), y)
        if np.isnan(a):
            continue
        n = np.isfinite(prim[f].to_numpy(float)).sum()
        z = (a - 0.5) * np.sqrt(12 * n)  # approx z for AUC under H0 (uniform ranks)
        rows.append((abs(z), f, a, z, n))
    rows.sort(reverse=True)
    zbar = _bonf_z(len(rows))
    for az, f, a, z, n in rows:
        flag = "  <-- passes FW bar" if az > zbar else ""
        A(f"  {f:22s} AUC {a:.3f}  z {z:+.2f}  n={n}{flag}")

    A("\n## Half-split stability of the top winner-feature (first vs second half by time)")
    if rows:
        topf = rows[0][1]
        ps = prim.sort_values("entry_time")
        mid = len(ps) // 2
        for half, sub in [("H1", ps.iloc[:mid]), ("H2", ps.iloc[mid:])]:
            yy = sub.exit_reason.isin(["tp", "tp_gap"]).to_numpy().astype(float)
            a = auc(sub[topf].to_numpy(float), yy)
            A(f"  {topf} {half}: AUC {a:.3f}  (n={len(sub)})")

    A("\n## Exit-reason mix (primary)")
    A("  " + str(prim.exit_reason.value_counts().to_dict()))
    A("  time-exits booked mark-to-market; median bars_held %.0f" % prim.bars_held.median())
    return "\n".join(L)


def _bonf_z(m: int) -> float:
    from scipy.stats import norm
    m = max(m, 1)
    return float(norm.ppf(1 - 0.05 / (2 * m)))


def main():
    prim = pd.read_parquet(os.path.join(OUT, "trades_primary.parquet"))
    noeng = pd.read_parquet(os.path.join(OUT, "trades_noengulf.parquet"))
    scale15 = pd.read_parquet(os.path.join(OUT, "trades_scale15.parquet"))
    rr2 = pd.read_parquet(os.path.join(OUT, "trades_rr2.parquet"))
    txt = report(prim, noeng, scale15, rr2)
    print(txt)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
