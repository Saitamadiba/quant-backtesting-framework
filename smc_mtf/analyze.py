#!/usr/bin/env python3
"""SMC MTF — full battery: regime filters, feature anatomy, and the
knife-style preregistered walk-forward pre-fill gate.

Net shown three ways: gross, full-taker, maker-mixed (CE entry is a
resting limit; TP is a limit; stops/sess/time pay one-way taker).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from ny4h_range_reversal.analyze import (  # noqa: E402
    feature_scan, quintile_table, regime_block, tstat,
)

OUT_DIR = os.path.join(_BASE, "reports", "smc_mtf")

NUM_FEATURES = ["htf_zone_atr", "zone_age_htf", "zone_pen_frac",
                "choch_leg_atr", "choch_break_margin_atr", "disp_body_frac",
                "disp_range_atr", "ltf_fvg_atr", "entry_leg_retrace",
                "bars_touch_to_choch", "bars_choch_to_fvg", "bars_to_fill",
                "vol_ratio_choch", "dist_ema200_pct", "stop_pct", "fee_r",
                "entry_hour_ny"]
CATS = {"side": ["long", "short"],
        "session": ["london", "ny", "off", "overlap"],
        "combo": ["1h->5m", "4h->15m"],
        "regime5": ["normal_chop", "normal_trend", "quiet_chop",
                    "quiet_trend", "vol_expansion"],
        "btc_regime5": ["normal_chop", "normal_trend", "quiet_chop",
                        "quiet_trend", "vol_expansion"]}
ONEWAY = {"BTC": 0.00075, "ETH": 0.00075, "SOL": 0.00105, "NQ": 0.0004}
MAKER = 0.0002


def add_nets(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    ow = t.symbol.map(ONEWAY).fillna(0.0018)
    is_tp = t.exit_reason.isin(["tp", "tp_gap"])
    t["net_mm"] = t.gross_r - (MAKER + np.where(is_tp, MAKER, ow)) / t.stop_pct
    return t


def head_line(g: pd.DataFrame, name: str) -> str:
    return (f"| {name} | {len(g)} | {(g.gross_r > 0).mean():.1%} "
            f"| {g.gross_r.mean():+.3f} | {tstat(g.gross_r):+.1f} "
            f"| {g.r_be.mean():+.3f} | {g.net_r.mean():+.3f} "
            f"| {g.net_mm.mean():+.3f} |")


HDR = ("| slice | n | WR | gross | t | BE gross | net taker | net mm |\n"
       "|---|---|---|---|---|---|---|---|")


def gate_study(t: pd.DataFrame) -> list[str]:
    """Preregistered walk-forward ridge gate (yt_gate_study protocol)."""
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr

    t = t.copy()
    t["year"] = pd.to_datetime(t.entry_time).dt.year
    X = t[NUM_FEATURES].astype(float).copy()
    for c, cats in CATS.items():
        s = t[c].astype(str)
        for cat in cats:
            X[f"{c}_{cat}"] = (s == cat).astype(float)
    years = sorted(t.year.unique())
    oos = np.zeros(len(t), bool)
    pred = np.full(len(t), np.nan)
    keeps = {k: np.zeros(len(t), bool) for k in (0.3, 0.5)}
    for yi in range(2, len(years)):
        y = years[yi]
        tr = (t.year < y).to_numpy(); te = (t.year == y).to_numpy()
        if tr.sum() < 1500 or te.sum() < 100:
            continue
        med = X[tr].median()
        Xtr, Xte = X[tr].fillna(med), X[te].fillna(med)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        m = Ridge(alpha=10.0).fit((Xtr - mu) / sd, t.gross_r.to_numpy()[tr])
        ptr = m.predict((Xtr - mu) / sd) - t.fee_r.to_numpy()[tr]
        pte = m.predict((Xte - mu) / sd) - t.fee_r.to_numpy()[te]
        oos |= te
        pred[te] = pte
        for k in keeps:
            keeps[k][te] = pte >= np.quantile(ptr, 1 - k)
    g = t[oos].copy()
    g["pred"] = pred[oos]
    ic = spearmanr(g.pred, g.gross_r, nan_policy="omit").statistic
    half = pd.to_datetime(g.entry_time).quantile(0.5)
    lines = [f"### Walk-forward pre-fill gate (crypto book) — OOS n={len(g)}, "
             f"IC={ic:+.3f}", "", HDR]

    def block(mask, name):
        s = g[mask]
        if len(s) < 50:
            return
        h1 = s[pd.to_datetime(s.entry_time) <= half]
        h2 = s[pd.to_datetime(s.entry_time) > half]
        sym = s.groupby("symbol").net_mm.mean()
        lines.append(head_line(s, name)[:-1] +
                     f" H1mm {h1.net_mm.mean():+.3f} / H2mm {h2.net_mm.mean():+.3f} "
                     f"/ syms+ {(sym > 0).sum()}/{len(sym)} |")

    block(np.ones(len(g), bool), "all OOS (base)")
    block(keeps[0.3][oos], "gate keep 30%")
    block(keeps[0.5][oos], "gate keep 50%")
    lines.append("")
    return lines


def main() -> None:
    t = add_nets(pd.read_parquet(os.path.join(OUT_DIR, "trades_all.parquet")))
    cr = t[t.symbol != "NQ"].copy()
    nq = t[t.symbol == "NQ"].copy()

    lines = ["# Multi-TF SMC — backtest report (2026-07-17)", "",
             f"n={len(t)} (crypto {len(cr)} across 1h→5m + 4h→15m; NQ {len(nq)}). "
             f"Period {t.entry_time.min()} → {t.entry_time.max()}.", "",
             "## Headline", "", HDR]
    for combo, g in cr.groupby("combo"):
        lines.append(head_line(g, f"crypto {combo}"))
    lines.append(head_line(nq, "NQ 4h->15m"))
    lines += ["", "### Per symbol (both combos pooled)", "", HDR]
    for sym, g in cr.groupby("symbol"):
        lines.append(head_line(g, sym))
    lines += ["", "### Per year (crypto pooled)", "",
              "| year | n | gross | net mm |", "|---|---|---|---|"]
    cr["year"] = pd.to_datetime(cr.entry_time).dt.year
    for y, g in cr.groupby("year"):
        lines.append(f"| {y} | {len(g)} | {g.gross_r.mean():+.3f} "
                     f"| {g.net_mm.mean():+.3f} |")
    lines.append("")

    # exit decomposition (integrity)
    lines += ["### Exit decomposition (crypto pooled)", "",
              "| reason | n | mean R |", "|---|---|---|"]
    for r, g in cr.groupby("exit_reason"):
        lines.append(f"| {r} | {len(g)} | {g.gross_r.mean():+.2f} |")
    lines.append("")

    lines += regime_block(cr, "regime5", "Per-regime — OWN regime5 (crypto)")
    lines += regime_block(cr, "btc_regime5", "Per-regime — BTC regime5 (crypto)")

    # regime-gated variants (the LR/knife dimmer question)
    lines += ["### Regime-gated variants (crypto, gross / net_mm)", "",
              "| gate | n kept | gross | net mm |", "|---|---|---|---|"]
    for name, mask in [
        ("none", np.ones(len(cr), bool)),
        ("drop quiet_* (own)", ~cr.regime5.isin(["quiet_chop", "quiet_trend"])),
        ("vol_expansion only (own)", cr.regime5 == "vol_expansion"),
        ("drop quiet_* (BTC)", ~cr.btc_regime5.isin(["quiet_chop", "quiet_trend"])),
        ("fib filter ≥0.618", cr.entry_leg_retrace >= 0.618),
        ("fib <0.618", cr.entry_leg_retrace < 0.618),
    ]:
        g = cr[mask]
        lines.append(f"| {name} | {len(g)} | {g.gross_r.mean():+.3f} "
                     f"| {g.net_mm.mean():+.3f} |")
    lines.append("")

    scan = feature_scan(cr, "gross_r", features=NUM_FEATURES)
    scan.to_csv(os.path.join(OUT_DIR, "feature_scan_gross.csv"), index=False)
    lines += ["### Feature scan — Q5−Q1 on gross (crypto)", "",
              "| feature | n | Q1 | Q5 | Δ | mono ρ | AUC | Δ H1 | Δ H2 | both |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in scan.iterrows():
        lines.append(f"| {r.feature} | {r.n} | {r.q1:+.3f} | {r.q5:+.3f} "
                     f"| {r.delta_q5_q1:+.3f} | {r.monotone_rho:+.2f} | {r.auc_win:.3f} "
                     f"| {r.delta_h1:+.3f} | {r.delta_h2:+.3f} "
                     f"| {'YES' if r.both_halves_same_sign else 'no'} |")
    lines.append("")

    for f in ("entry_leg_retrace", "choch_leg_atr", "htf_zone_atr", "stop_pct"):
        lines += quintile_table(cr, f)

    lines += gate_study(cr)

    with open(os.path.join(OUT_DIR, "report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", os.path.join(OUT_DIR, "report.md"))


if __name__ == "__main__":
    main()
