#!/usr/bin/env python3
"""Five-state framework — classification run + validation battery.

1) Classify 12 symbols on 15m and 1h; save per-bar states.
2) State anatomy: occupancy, dwell, transition matrix, forward drift/vol,
   false-breakout rate.
3) Framework claim test: per-state gross R of the recorded YouTube books,
   benchmarked against regime5 via eta² (variance explained).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from market_states.engine import STATES, classify_symbol  # noqa: E402

R = os.path.join(_BASE, "reports")
OUT_DIR = os.path.join(R, "market_states")
SYMBOLS = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT",
           "ETH", "LINK", "LTC", "SOL", "XRP"]
FWD = 24  # forward horizon in bars


def classify_all() -> dict[str, pd.DataFrame]:
    out = {}
    meta = []
    for tf in ("15m", "1h"):
        frames = []
        for sym in SYMBOLS:
            t0 = time.time()
            s = classify_symbol(sym, tf)
            if s.empty:
                continue
            s["symbol"] = sym
            frames.append(s)
            meta.append({"tf": tf, "symbol": sym,
                         "n_breakouts": s.attrs["n_breakouts"],
                         "false_break_rate": s.attrs["false_break_rate"]})
            print(f"  {sym:5s} {tf:3s} classified ({time.time()-t0:.0f}s)", flush=True)
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(os.path.join(OUT_DIR, f"states_{tf}.parquet"), index=False)
        out[tf] = df
    pd.DataFrame(meta).to_csv(os.path.join(OUT_DIR, "breakout_meta.csv"), index=False)
    out["meta"] = pd.DataFrame(meta)
    return out


def anatomy(df: pd.DataFrame, tf: str) -> list[str]:
    lines = [f"## State anatomy — {tf} (12 symbols pooled)", ""]
    # occupancy
    occ = df["state"].value_counts(normalize=True)
    # forward drift/vol per state (per symbol to avoid boundary leaks)
    rows = []
    for sym, g in df.groupby("symbol"):
        c = g["close"].to_numpy()
        a = g["atr"].to_numpy()
        d = g["direction"].to_numpy()
        st = g["state"].to_numpy()
        fwd = np.full(len(g), np.nan)
        fwd[:-FWD] = (c[FWD:] - c[:-FWD])
        fwd_atr = fwd / np.where(a > 0, a, np.nan)          # in ATR units
        signed = np.where(d != 0, fwd_atr * d, np.nan)       # along state direction
        rv = pd.Series(np.log(c)).diff().rolling(FWD).std().shift(-FWD).to_numpy()
        rv0 = pd.Series(np.log(c)).diff().rolling(FWD).std().to_numpy()
        for s in STATES:
            m = st == s
            if m.sum() < 100:
                continue
            rows.append({"state": s, "n": int(m.sum()),
                         "fwd_atr": np.nanmean(fwd_atr[m]),
                         "signed_fwd_atr": np.nanmean(signed[m]),
                         "vol_ratio": np.nanmean(rv[m] / rv0[m])})
    agg = (pd.DataFrame(rows).groupby("state")
           .apply(lambda g: pd.Series({
               "n": g.n.sum(),
               "fwd_atr": np.average(g.fwd_atr, weights=g.n),
               "signed_fwd_atr": np.average(g.signed_fwd_atr, weights=g.n),
               "vol_ratio": np.average(g.vol_ratio, weights=g.n)}),
                  include_groups=False))
    # dwell + transitions
    dwell = {}
    trans = pd.DataFrame(0.0, index=STATES, columns=STATES)
    for sym, g in df.groupby("symbol"):
        st = g["state"].to_numpy()
        change = np.flatnonzero(st[1:] != st[:-1]) + 1
        segs = np.split(np.arange(len(st)), change)
        for seg in segs:
            dwell.setdefault(st[seg[0]], []).append(len(seg))
        for i in range(len(change) - 1):
            trans.loc[st[change[i]], st[change[i + 1]]] += 1
    trans = trans.div(trans.sum(axis=1).replace(0, np.nan), axis=0)

    lines += ["| state | share | median dwell (bars) | fwd 24-bar drift (ATR) | signed drift (ATR, along state dir) | fwd/trailing vol |",
              "|---|---|---|---|---|---|"]
    for s in STATES:
        if s not in agg.index:
            continue
        r = agg.loc[s]
        md = int(np.median(dwell.get(s, [np.nan])))
        sd = f"{r.signed_fwd_atr:+.3f}" if np.isfinite(r.signed_fwd_atr) else "—"
        lines.append(f"| {s} | {occ.get(s, 0):.1%} | {md} | {r.fwd_atr:+.3f} "
                     f"| {sd} | {r.vol_ratio:.2f} |")
    lines += ["", f"Transition matrix (row → col, at state changes, {tf}):", "",
              "| from \\ to | " + " | ".join(STATES) + " |",
              "|---|" + "---|" * len(STATES)]
    for s in STATES:
        lines.append(f"| {s} | " + " | ".join(
            f"{trans.loc[s, t]:.2f}" if np.isfinite(trans.loc[s, t]) else "—"
            for t in STATES) + " |")
    lines.append("")
    return lines


def eta2(x: pd.Series, groups: pd.Series) -> float:
    d = pd.DataFrame({"x": x, "g": groups}).dropna()
    gm = d["x"].mean()
    ss_tot = ((d["x"] - gm) ** 2).sum()
    ss_b = d.groupby("g")["x"].agg(["mean", "size"]).pipe(
        lambda t: (t["size"] * (t["mean"] - gm) ** 2).sum())
    return float(ss_b / ss_tot) if ss_tot > 0 else np.nan


def book_test(states: dict[str, pd.DataFrame]) -> list[str]:
    books = []
    ny = pd.read_parquet(os.path.join(R, "ny4h_range_reversal/trades_all.parquet"))
    books.append(("ny4h fade 5m", ny[ny.exec_tf == "5m"], "15m"))
    st = pd.read_parquet(os.path.join(R, "structure_scalper/trades_all.parquet"))
    books.append(("struct cont. 5m confirm", st[(st.exec_tf == "5m") & (st.arm == "confirm")], "15m"))
    grid = pd.read_parquet(os.path.join(R, "structure_scalper/trades_tf_grid.parquet"))
    books.append(("struct cont. 1h/4h pullback",
                  grid[(grid.exec_tf == "1h") & (grid.htf_tf == "4h") & (grid.arm == "pullback")], "1h"))
    fb = pd.read_parquet(os.path.join(R, "fib_bos/trades_all.parquet"))
    books.append(("fib trigger 5m", fb[(fb.exec_tf == "5m") & (fb.arm == "trigger")], "5m->15m"))
    books.append(("fib limit618 1h", fb[(fb.exec_tf == "1h") & (fb.arm == "limit618")], "1h"))

    lines = ["## Framework claim test — per-state gross R of the recorded books", "",
             "States attached causally (state-bar close ≤ entry; 15m states for "
             "5m books, 1h states for 1h books). `aligned` = trade side matches "
             "the state direction (directional states only).", ""]
    for name, book, tfk in books:
        tf = "15m" if "15m" in tfk else "1h"
        sframe = states[tf]
        parts = []
        for sym, grp in book.groupby("symbol"):
            grp = grp.sort_values("entry_time").copy()
            ss = sframe[sframe.symbol == sym].sort_values("time")
            merged = pd.merge_asof(grp, ss[["time", "state", "direction"]],
                                   left_on="entry_time", right_on="time",
                                   direction="backward")
            parts.append(merged)
        b = pd.concat(parts, ignore_index=True)
        side_sign = np.where(b["side"] == "long", 1, -1)
        b["aligned"] = np.where(b["direction"] != 0,
                                side_sign == b["direction"], np.nan)
        e5 = eta2(b["gross_r"], b["state"])
        er = eta2(b["gross_r"], b["regime5"]) if "regime5" in b else np.nan
        lines += [f"### {name} — eta²: 5-state {e5:.5f} vs regime5 {er:.5f}", "",
                  "| state | n | grossR | netR | grossR aligned | grossR counter |",
                  "|---|---|---|---|---|---|"]
        for s in STATES:
            g = b[b.state == s]
            if len(g) < 30:
                continue
            al = g[g.aligned == 1]; ct = g[g.aligned == 0]
            av = f"{al.gross_r.mean():+.3f} (n={len(al)})" if len(al) >= 30 else "—"
            cv = f"{ct.gross_r.mean():+.3f} (n={len(ct)})" if len(ct) >= 30 else "—"
            lines.append(f"| {s} | {len(g)} | {g.gross_r.mean():+.3f} "
                         f"| {g.net_r.mean():+.3f} | {av} | {cv} |")
        lines.append("")
    return lines


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    states = classify_all()
    lines = ["# Five-State Market Framework — validation report (2026-07-16)", ""]
    meta = states.pop("meta")
    for tf in ("15m", "1h"):
        lines += anatomy(states[tf], tf)
        fb_rate = meta[meta.tf == tf].false_break_rate.mean()
        nb = meta[meta.tf == tf].n_breakouts.sum()
        lines += [f"Breakout events ({tf}): {nb}; mean false-break rate "
                  f"(close back inside within 12 bars): {fb_rate:.1%}", ""]
    lines += book_test(states)
    with open(os.path.join(OUT_DIR, "report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", os.path.join(OUT_DIR, "report.md"))


if __name__ == "__main__":
    main()
