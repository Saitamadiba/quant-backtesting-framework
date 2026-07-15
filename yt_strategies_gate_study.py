#!/usr/bin/env python3
"""Walk-forward gate study over the three YouTube-strategy trade books.

Question: can a gate built from the RECORDED entry-time features flip any
of the (gross-zero / gross-negative) YouTube strategies to net-positive?

PREREGISTERED PROTOCOL (fixed before any result was seen):
  * Model: Ridge regression on standardized causal features, target =
    gross_r. Predicted net = predicted gross − fee_r (fee_r is known at
    entry). Gate keeps the top 30% / 50% of predicted net, thresholds set
    on TRAIN predictions only.
  * Walk-forward: expanding yearly folds — train on all years < Y,
    evaluate on year Y; first two years are train-only.
  * Mechanical baseline: stop_pct ≥ train-median (pure fee arithmetic).
  * SUCCESS BAR: OOS pooled net_r > 0 AND both OOS halves same sign AND
    ≥ 2/3 of symbols positive. Anything less = gate does NOT rescue.

Slices (chosen a priori as each family's most viable / representative):
  A ny4h 5m           (gross ≈ 0, fee-killed)
  B struct 5m confirm  (gross −0.045, hardest)
  C struct 1h/4h pullback (best fee tier of that family)
  D fib trigger 5m     (faithful video arm)
  E fib limit618 1h    (gross ≈ 0 at the cheapest toll; maker-mixed net
                        also reported)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
R = os.path.join(_BASE, "reports")

ONEWAY_TAKER = {"BTC": 0.00075, "ETH": 0.00075, "SOL": 0.00105}
KEEPS = (0.30, 0.50)
MIN_TRAIN_YEARS = 2


def load_slices() -> dict[str, tuple[pd.DataFrame, list[str], list[str]]]:
    out = {}

    ny = pd.read_parquet(os.path.join(R, "ny4h_range_reversal/trades_all.parquet"))
    ny = ny[ny.exec_tf == "5m"].copy()
    out["A ny4h 5m"] = (ny, [
        "range_pct", "range_vs_atr15", "sweep_depth_pct", "sweep_depth_atr",
        "bars_outside", "reentry_pos", "stop_pct", "tp_room_frac",
        "breakout_vol_ratio", "dist_ema200_pct", "mins_since_range",
        "prior_signals_today", "entry_hour_ny", "fee_r",
    ], ["side", "regime5", "btc_regime5"])

    sfeat = ["bos_dist_atr", "bos_vol_ratio", "bars_to_retest", "bars_to_confirm",
             "retest_depth_atr", "retrace_frac", "confirm_body_frac",
             "confirm_range_atr", "stop_pct", "fee_r", "dist_ema200_pct",
             "hours_since_htf_break", "prior_signals_today", "entry_hour_ny"]
    scat = ["side", "confirm_type", "session", "structure_seq", "regime5", "btc_regime5"]

    st = pd.read_parquet(os.path.join(R, "structure_scalper/trades_all.parquet"))
    st5 = st[(st.exec_tf == "5m") & (st.arm == "confirm")].copy()
    out["B struct 5m confirm"] = (st5, sfeat, scat)

    grid = pd.read_parquet(os.path.join(R, "structure_scalper/trades_tf_grid.parquet"))
    st1h = grid[(grid.exec_tf == "1h") & (grid.htf_tf == "4h")
                & (grid.arm == "pullback")].copy()
    out["C struct 1h/4h pullback"] = (st1h, sfeat, scat)

    ffeat = ["leg_atr", "leg_pct", "entry_fib", "bos_dist_atr", "zone_touch_bars",
             "retrace_depth_frac", "rr_planned", "stop_pct", "fee_r",
             "trigger_vol_ratio", "dist_ema200_pct", "entry_hour_ny",
             "prior_signals_today"]
    fcat = ["side", "session", "regime5", "btc_regime5"]
    fb = pd.read_parquet(os.path.join(R, "fib_bos/trades_all.parquet"))
    out["D fib trigger 5m"] = (
        fb[(fb.exec_tf == "5m") & (fb.arm == "trigger")].copy(),
        ffeat + ["trigger_type_dummy"], fcat + ["trigger_type"])
    lim = fb[(fb.exec_tf == "1h") & (fb.arm == "limit618")].copy()
    ow = lim["symbol"].map(ONEWAY_TAKER).fillna(0.0018)
    is_tp = lim["exit_reason"].isin(["tp", "tp_gap"])
    lim["net_mm"] = lim["gross_r"] - (0.0002 + np.where(is_tp, 0.0002, ow)) / lim["stop_pct"]
    out["E fib limit618 1h"] = (lim, ffeat, fcat)
    return out


def build_xy(t: pd.DataFrame, num: list[str], cat: list[str]):
    num = [f for f in num if f in t.columns]
    X = t[num].astype(float).copy()
    for c in cat:
        if c in t.columns:
            X = pd.concat([X, pd.get_dummies(t[c].astype(str), prefix=c)], axis=1)
    return X.astype(float)


def run_slice(name: str, t: pd.DataFrame, num: list[str], cat: list[str]) -> dict:
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr

    t = t.copy()
    t["year"] = pd.to_datetime(t["entry_time"]).dt.year
    X = build_xy(t, num, cat)
    years = sorted(t["year"].unique())
    oos_mask = np.zeros(len(t), bool)
    pred_net = np.full(len(t), np.nan)
    keep_flags = {k: np.zeros(len(t), bool) for k in KEEPS}
    stop_floor_keep = np.zeros(len(t), bool)

    for yi in range(MIN_TRAIN_YEARS, len(years)):
        y = years[yi]
        tr = t["year"] < y
        te = (t["year"] == y).to_numpy()
        if tr.sum() < 1500 or te.sum() < 100:
            continue
        med = X[tr].median()
        Xtr = X[tr].fillna(med)
        Xte = X[te].fillna(med)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        model = Ridge(alpha=10.0)
        model.fit((Xtr - mu) / sd, t.loc[tr.to_numpy(), "gross_r"])
        p_tr = model.predict((Xtr - mu) / sd) - t.loc[tr.to_numpy(), "fee_r"].to_numpy()
        p_te = model.predict((Xte - mu) / sd) - t.loc[te, "fee_r"].to_numpy()
        oos_mask |= te
        pred_net[te] = p_te
        for k in KEEPS:
            thr = np.quantile(p_tr, 1 - k)
            keep_flags[k][te] = p_te >= thr
        stop_floor_keep[te] = (t.loc[te, "stop_pct"]
                               >= t.loc[tr.to_numpy(), "stop_pct"].median())

    oos = t[oos_mask].copy()
    oos["pred_net"] = pred_net[oos_mask]
    et = pd.to_datetime(oos["entry_time"])
    half = et.quantile(0.5)
    ic = spearmanr(oos["pred_net"], oos["gross_r"], nan_policy="omit").statistic

    def stats(mask):
        g = oos[mask.astype(bool)]
        if len(g) < 50:
            return None
        h1 = g[pd.to_datetime(g.entry_time) <= half]
        h2 = g[pd.to_datetime(g.entry_time) > half]
        sym = g.groupby("symbol").net_r.mean()
        row = {"n": len(g), "gross": g.gross_r.mean(), "net": g.net_r.mean(),
               "net_h1": h1.net_r.mean(), "net_h2": h2.net_r.mean(),
               "sym_pos": f"{(sym > 0).sum()}/{len(sym)}"}
        if "net_mm" in g.columns:
            row["net_mm"] = g.net_mm.mean()
        return row

    res = {"name": name, "oos_n": len(oos), "ic": ic,
           "base": stats(np.ones(len(oos), bool))}
    for k in KEEPS:
        res[f"keep{int(k*100)}"] = stats(keep_flags[k][oos_mask])
    res["stop_floor"] = stats(stop_floor_keep[oos_mask])
    return res


def fmt(row: dict | None) -> str:
    if row is None:
        return "| — | | | | | |"
    mm = f" (mm {row['net_mm']:+.3f})" if "net_mm" in row else ""
    return (f"| {row['n']} | {row['gross']:+.4f} | {row['net']:+.3f}{mm} "
            f"| {row['net_h1']:+.3f} | {row['net_h2']:+.3f} | {row['sym_pos']} |")


def main() -> None:
    lines = ["# YouTube strategies — walk-forward gate study (2026-07-16)", "",
             "Protocol: see script header (preregistered keep-rates 30/50%, "
             "expanding yearly folds, ridge on gross, gate on predicted net).", ""]
    for name, (t, num, cat) in load_slices().items():
        r = run_slice(name, t, num, cat)
        print(f"{name}: OOS n={r['oos_n']} IC={r['ic']:+.3f}")
        lines += [f"## {name} — OOS IC(pred, gross) = {r['ic']:+.3f}", "",
                  "| cohort | n | grossR | netR | net H1 | net H2 | syms+ |",
                  "|---|---|---|---|---|---|---|"]
        for label, key in (("all OOS (base)", "base"), ("gate keep 30%", "keep30"),
                           ("gate keep 50%", "keep50"),
                           ("stop_pct ≥ train median (fee floor)", "stop_floor")):
            lines.append(f"| {label} " + fmt(r[key]))
        lines.append("")
    out = os.path.join(R, "yt_gate_study.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
