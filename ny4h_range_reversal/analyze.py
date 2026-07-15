#!/usr/bin/env python3
"""NY4H range-reversal — per-regime assessment + winner/loser pattern scan.

Reads reports/ny4h_range_reversal/trades_all.parquet, writes report.md.

House standards applied:
  * gross AND net shown everywhere (fee wall is the a-priori killer at 5m)
  * split-half (chronological) sign agreement on every pattern claim
  * quintile monotonicity, not just top-bottom delta
  * multiple-testing caveat: ~20 features scanned -> Bonferroni mindset
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

OUT_DIR = os.path.join(_BASE, "reports", "ny4h_range_reversal")

NUMERIC_FEATURES = [
    "range_pct", "range_vs_atr15", "sweep_depth_pct", "sweep_depth_atr",
    "bars_outside", "reentry_pos", "stop_pct", "tp_room_frac",
    "breakout_vol_ratio", "dist_ema200_pct", "mins_since_range",
    "prior_signals_today", "prior_sameside_today", "entry_hour_ny", "fee_r",
]
BINARY_FEATURES = ["trend_align", "tp_fits_in_range"]


def tstat(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 3 or s.std() == 0:
        return np.nan
    return float(s.mean() / s.std() * np.sqrt(len(s)))


def profit_factor(s: pd.Series) -> float:
    gains = s[s > 0].sum()
    losses = -s[s <= 0].sum()
    return float(gains / losses) if losses > 0 else np.inf


def max_dd(s: pd.Series) -> float:
    eq = s.cumsum()
    return float((eq - eq.cummax()).min())


def auc_win(feature: pd.Series, win: pd.Series) -> float:
    """Rank-based AUC of feature for predicting a win."""
    d = pd.DataFrame({"f": feature, "w": win}).dropna()
    if d["w"].nunique() < 2 or len(d) < 50:
        return np.nan
    r = d["f"].rank()
    n1 = d["w"].sum()
    n0 = len(d) - n1
    u = r[d["w"] == 1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def headline_block(t: pd.DataFrame, label: str) -> list[str]:
    lines = [f"### {label}", "",
             "| symbol | n | trades/day | WR | grossR | netR | t(net) | PF(net) | maxDD(net R) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for sym, g in list(t.groupby("symbol")) + [("POOLED", t)]:
        g = g.sort_values("entry_time")
        lines.append(
            f"| {sym} | {len(g)} | {len(g)/max(g['ny_date'].nunique(),1):.1f} "
            f"| {(g.gross_r > 0).mean():.1%} | {g.gross_r.mean():+.3f} "
            f"| {g.net_r.mean():+.3f} | {tstat(g.net_r):+.1f} "
            f"| {profit_factor(g.net_r):.2f} | {max_dd(g.net_r):.0f} |")
    lines.append("")
    return lines


def yearly_block(t: pd.DataFrame) -> list[str]:
    t = t.copy()
    t["year"] = pd.to_datetime(t["entry_time"]).dt.year
    piv = t.groupby("year").agg(n=("net_r", "size"), gross=("gross_r", "mean"),
                                net=("net_r", "mean"))
    lines = ["### Per-year (pooled, 5m arm)", "",
             "| year | n | grossR | netR |", "|---|---|---|---|"]
    for y, r in piv.iterrows():
        lines.append(f"| {y} | {int(r.n)} | {r.gross:+.3f} | {r.net:+.3f} |")
    lines.append("")
    return lines


def regime_block(t: pd.DataFrame, col: str, title: str) -> list[str]:
    half = pd.to_datetime(t["entry_time"]).quantile(0.5)
    lines = [f"### {title}", "",
             "| regime | n | WR | grossR | netR | t(gross) | gross H1 | gross H2 | same sign? |",
             "|---|---|---|---|---|---|---|---|---|"]
    for reg, g in sorted(t.groupby(col), key=lambda kv: -kv[1]["gross_r"].mean()):
        h1 = g[pd.to_datetime(g["entry_time"]) <= half]["gross_r"].mean()
        h2 = g[pd.to_datetime(g["entry_time"]) > half]["gross_r"].mean()
        same = "YES" if np.sign(h1) == np.sign(h2) and not np.isnan(h1) else "no"
        lines.append(
            f"| {reg} | {len(g)} | {(g.gross_r > 0).mean():.1%} "
            f"| {g.gross_r.mean():+.3f} | {g.net_r.mean():+.3f} "
            f"| {tstat(g.gross_r):+.1f} | {h1:+.3f} | {h2:+.3f} | {same} |")
    lines.append("")
    return lines


def feature_scan(t: pd.DataFrame, target: str = "gross_r",
                 features: list[str] | None = None) -> pd.DataFrame:
    """Quintile scan of every numeric feature vs the target R column."""
    features = features if features is not None else NUMERIC_FEATURES
    half = pd.to_datetime(t["entry_time"]).quantile(0.5)
    is_h1 = pd.to_datetime(t["entry_time"]) <= half
    win = (t["gross_r"] > 0).astype(int)
    rows = []
    for f in features:
        if f not in t.columns:
            continue
        x = t[f]
        ok = x.notna() & t[target].notna()
        if ok.sum() < 500:
            continue
        try:
            q = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        qm = t.loc[ok].groupby(q)[target].mean()
        if len(qm) < 4:
            continue
        delta = qm.iloc[-1] - qm.iloc[0]
        rho = pd.Series(qm.to_numpy()).corr(pd.Series(range(len(qm))), method="spearman")
        # split-half: does the top-bottom delta keep its sign?
        deltas = []
        for mask in (is_h1, ~is_h1):
            sub = t.loc[ok & mask]
            xq = x[ok & mask]
            try:
                qq = pd.qcut(xq, 5, labels=False, duplicates="drop")
                m = sub.groupby(qq)[target].mean()
                deltas.append(m.iloc[-1] - m.iloc[0])
            except (ValueError, IndexError):
                deltas.append(np.nan)
        same = (np.sign(deltas[0]) == np.sign(deltas[1])
                and not any(np.isnan(d) for d in deltas))
        rows.append({
            "feature": f, "n": int(ok.sum()),
            "q1": qm.iloc[0], "q5": qm.iloc[-1], "delta_q5_q1": delta,
            "monotone_rho": rho, "auc_win": auc_win(x, win),
            "delta_h1": deltas[0], "delta_h2": deltas[1],
            "both_halves_same_sign": bool(same),
        })
    return pd.DataFrame(rows).sort_values("delta_q5_q1", key=abs, ascending=False)


def quintile_table(t: pd.DataFrame, feature: str) -> list[str]:
    ok = t[feature].notna()
    q, bins = pd.qcut(t.loc[ok, feature], 5, labels=False, retbins=True,
                      duplicates="drop")
    g = t.loc[ok].groupby(q)
    lines = [f"#### {feature} quintiles", "",
             "| quintile (range) | n | WR | grossR | netR |", "|---|---|---|---|---|"]
    for qi, grp in g:
        lo, hi = bins[int(qi)], bins[int(qi) + 1]
        lines.append(f"| Q{int(qi)+1} [{lo:.4g}, {hi:.4g}] | {len(grp)} "
                     f"| {(grp.gross_r > 0).mean():.1%} "
                     f"| {grp.gross_r.mean():+.3f} | {grp.net_r.mean():+.3f} |")
    lines.append("")
    return lines


def main() -> None:
    trades = pd.read_parquet(os.path.join(OUT_DIR, "trades_all.parquet"))
    t5 = trades[trades["exec_tf"] == "5m"].copy()
    t15 = trades[trades["exec_tf"] == "15m"].copy()

    lines = ["# NY 4H Range Reversal — backtest report", "",
             f"Trades: {len(t5)} (5m arm) + {len(t15)} (ETH/SOL 15m sensitivity arm). "
             f"Period: {t5['entry_time'].min()} → {t5['entry_time'].max()}.", ""]

    lines += headline_block(t5, "Headline — 5m arm (faithful spec)")
    if not t15.empty:
        lines += headline_block(t15, "Sensitivity — ETH/SOL on 15m execution")
    lines += yearly_block(t5)

    lines += regime_block(t5, "regime5", "Per-regime — OWN-asset regime5 (pooled 5m arm, gross basis)")
    lines += regime_block(t5, "btc_regime5", "Per-regime — BTC regime5 (pooled 5m arm)")

    # side / structural splits
    lines += ["### Structural splits (pooled 5m arm)", "",
              "| split | n | WR | grossR | netR |", "|---|---|---|---|---|"]
    for name, mask in [
        ("long", t5["side"] == "long"), ("short", t5["side"] == "short"),
        ("trend_align=True", t5["trend_align"] == True),   # noqa: E712
        ("trend_align=False", t5["trend_align"] == False),  # noqa: E712
        ("tp_fits_in_range=True", t5["tp_fits_in_range"] == True),   # noqa: E712
        ("tp_fits_in_range=False", t5["tp_fits_in_range"] == False),  # noqa: E712
        ("first signal of day", t5["prior_signals_today"] == 0),
        ("4th+ signal of day", t5["prior_signals_today"] >= 3),
    ]:
        g = t5[mask]
        if len(g) == 0:
            continue
        lines.append(f"| {name} | {len(g)} | {(g.gross_r > 0).mean():.1%} "
                     f"| {g.gross_r.mean():+.3f} | {g.net_r.mean():+.3f} |")
    lines.append("")

    # feature scans, gross and net
    for target in ("gross_r", "net_r"):
        scan = feature_scan(t5, target)
        scan.to_csv(os.path.join(OUT_DIR, f"feature_scan_{target}.csv"), index=False)
        lines += [f"### Feature scan — quintile Q5−Q1 delta on {target}", "",
                  "| feature | n | Q1 | Q5 | Δ(Q5−Q1) | mono ρ | AUC(win) | Δ H1 | Δ H2 | both halves |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in scan.iterrows():
            lines.append(
                f"| {r.feature} | {r.n} | {r.q1:+.3f} | {r.q5:+.3f} "
                f"| {r.delta_q5_q1:+.3f} | {r.monotone_rho:+.2f} | {r.auc_win:.3f} "
                f"| {r.delta_h1:+.3f} | {r.delta_h2:+.3f} "
                f"| {'YES' if r.both_halves_same_sign else 'no'} |")
        lines.append("")

    # detailed quintiles for the a-priori interesting features
    for f in ("stop_pct", "sweep_depth_atr", "bars_outside", "tp_room_frac",
              "breakout_vol_ratio", "entry_hour_ny"):
        if f in t5.columns:
            lines += quintile_table(t5, f)

    lines += ["### Multiple-testing note", "",
              f"{len(NUMERIC_FEATURES)} numeric + {len(BINARY_FEATURES)} binary features scanned "
              "on 2 targets — treat any single pattern at p≈0.05 as noise; only "
              "monotone, both-halves-same-sign, mechanism-backed patterns are "
              "candidates, and even those need a forward shadow before use.", ""]

    report = "\n".join(lines)
    path = os.path.join(OUT_DIR, "report.md")
    with open(path, "w") as fh:
        fh.write(report)
    print(f"wrote {path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
