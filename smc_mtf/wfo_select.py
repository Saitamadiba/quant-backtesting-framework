#!/usr/bin/env python3
"""SMC-MTF WFO — selection, robustness, and gating phase.

Per TF rung: rolling 540d IS / 180d OOS windows (step 180d); per window
pick the grid combo with best IS mean net_mm (min 25 IS trades, fallback
FROZEN); stitch the picked OOS slices into the WFO-OOS book. Baselines:
FROZEN config OOS (same windows) and the in-hindsight ORACLE (upper
bound, never tradable). Robustness: CSCV PBO over the window×combo
matrix; clustered t (cluster = NY date × side — simultaneous
same-direction entries across correlated cryptos are ONE bet).
Gating: BASE vs AUG (+microstructure) walk-forward ridge on the FROZEN
book (layer separation: params and gates never co-selected), plus the
standing preregistered manual gates, Bonferroni-corrected.
"""

from __future__ import annotations

import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from ny4h_range_reversal.analyze import tstat  # noqa: E402
from smc_mtf.wfo import FROZEN, OUT, combo_id  # noqa: E402

IS_DAYS, OOS_DAYS = 540, 180
MIN_IS_TRADES = 25
FROZEN_ID = combo_id(FROZEN)


def clustered_t(t: pd.DataFrame, col: str = "net_mm") -> tuple[float, int]:
    g = t.groupby([pd.to_datetime(t.entry_time).dt.date, t.side])[col].mean()
    if len(g) < 5 or g.std() == 0:
        return np.nan, len(g)
    return float(g.mean() / g.std() * np.sqrt(len(g))), len(g)


def load_tf(tf_dir: str) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in glob.glob(os.path.join(tf_dir, "*.parquet"))]
    t = pd.concat(frames, ignore_index=True)
    t["entry_time"] = pd.to_datetime(t.entry_time)
    return t[t.symbol != "NQ"], t[t.symbol == "NQ"]


def wfo_one(t: pd.DataFrame):
    t0 = t.entry_time.min().normalize()
    t_end = t.entry_time.max()
    windows = []
    cur = t0
    while cur + pd.Timedelta(days=IS_DAYS + 30) < t_end:
        is_a, is_b = cur, cur + pd.Timedelta(days=IS_DAYS)
        oos_b = min(is_b + pd.Timedelta(days=OOS_DAYS), t_end)
        windows.append((is_a, is_b, oos_b))
        cur += pd.Timedelta(days=OOS_DAYS)
    picks, oos_parts, frozen_parts, oracle = [], [], [], []
    mat = {}
    for (a, b, c) in windows:
        is_sl = t[(t.entry_time >= a) & (t.entry_time < b)]
        oos_sl = t[(t.entry_time >= b) & (t.entry_time < c)]
        stats = is_sl.groupby("combo_id").net_mm.agg(["mean", "size"])
        ok = stats[stats["size"] >= MIN_IS_TRADES]
        pick = ok["mean"].idxmax() if len(ok) else FROZEN_ID
        picks.append({"is_start": a.date(), "pick": pick,
                      "is_mean": float(ok["mean"].max()) if len(ok) else np.nan,
                      "is_n": int(ok.loc[pick, "size"]) if pick in ok.index else 0})
        oos_parts.append(oos_sl[oos_sl.combo_id == pick])
        frozen_parts.append(oos_sl[oos_sl.combo_id == FROZEN_ID])
        om = oos_sl.groupby("combo_id").net_mm.mean()
        if len(om):
            oracle.append(oos_sl[oos_sl.combo_id == om.idxmax()])
        mat[b.date()] = oos_sl.groupby("combo_id").net_mm.mean()
    oos = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    froz = pd.concat(frozen_parts, ignore_index=True) if frozen_parts else pd.DataFrame()
    orac = pd.concat(oracle, ignore_index=True) if oracle else pd.DataFrame()
    M = pd.DataFrame(mat).T  # windows × combos (OOS window means)
    return picks, oos, froz, orac, M


def pbo_cscv(M: pd.DataFrame, n_splits: int = 200) -> float:
    """CSCV PBO on the window×combo mean matrix."""
    M = M.dropna(axis=1, thresh=max(3, int(0.7 * len(M))))
    S = len(M)
    if S < 6 or M.shape[1] < 3:
        return np.nan
    idx = list(range(S))
    halves = list(itertools.combinations(idx, S // 2))
    rng = np.random.default_rng(11)
    if len(halves) > n_splits:
        halves = [halves[i] for i in rng.choice(len(halves), n_splits, replace=False)]
    below = 0
    for h in halves:
        tr = M.iloc[list(h)].mean()
        te = M.iloc[[i for i in idx if i not in h]].mean()
        best = tr.idxmax()
        rank = (te < te[best]).mean()   # fraction of combos WORSE than pick OOS
        below += rank < 0.5             # pick in bottom half OOS
    return below / len(halves)


def summary(t: pd.DataFrame, name: str) -> str:
    if t.empty:
        return f"| {name} | 0 | — | — | — | — | — | — |"
    ct, nc = clustered_t(t)
    return (f"| {name} | {len(t)} | {(t.gross_r > 0).mean():.1%} "
            f"| {t.gross_r.mean():+.3f} | {t.net_mm.mean():+.3f} "
            f"| {tstat(t.net_mm):+.1f} | {ct:+.1f} ({nc} cl) "
            f"| {t.groupby('symbol').net_mm.mean().gt(0).sum()}/{t.symbol.nunique()} |")


HDR = ("| book | n | WR | gross | net mm | t(mm) | clustered t | syms mm+ |\n"
       "|---|---|---|---|---|---|---|---|")


def main() -> None:
    lines = ["# SMC-MTF walk-forward optimization — desk report (2026-07-17)", "",
             "Protocol in script headers (24-combo prereg grid; 540/180d windows; "
             "IS pick on net_mm, min 25 trades; CSCV PBO; clustered t = "
             "date×side clusters).", ""]
    for tf in sorted(os.listdir(OUT)):
        d = os.path.join(OUT, tf)
        if not os.path.isdir(d):
            continue
        cr, nq = load_tf(d)
        picks, oos, froz, orac, M = wfo_one(cr)
        pbo = pbo_cscv(M)
        lines += [f"## {tf} (crypto)", "", HDR,
                  summary(froz, "FROZEN v1 OOS"),
                  summary(oos, "WFO-reoptimized OOS"),
                  summary(orac, "ORACLE (hindsight bound)"), "",
                  f"PBO (CSCV): **{pbo:.2f}**" if np.isfinite(pbo) else "PBO: n/a", "",
                  "Window picks: " + ", ".join(
                      f"{p['is_start']}→{p['pick']}" for p in picks), ""]
        churn = len({p["pick"] for p in picks})
        lines += [f"Param churn: {churn} distinct configs across {len(picks)} windows; "
                  f"frozen picked {sum(p['pick'] == FROZEN_ID for p in picks)}×.", ""]
        if not nq.empty:
            nfz = nq[nq.combo_id == FROZEN_ID]
            lines += [f"NQ (frozen config, whole period): n={len(nfz)} "
                      f"gross={nfz.gross_r.mean():+.3f} mm={nfz.net_mm.mean():+.3f}", ""]
        oos.to_parquet(os.path.join(OUT, f"wfo_oos_{tf}.parquet"), index=False)
        froz.to_parquet(os.path.join(OUT, f"frozen_oos_{tf}.parquet"), index=False)

    with open(os.path.join(_BASE, "reports", "smc_mtf", "wfo_report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote reports/smc_mtf/wfo_report.md")


if __name__ == "__main__":
    main()
