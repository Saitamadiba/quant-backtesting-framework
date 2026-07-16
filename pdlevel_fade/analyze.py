#!/usr/bin/env python3
"""PD-level fade — regime assessment + winner/loser anatomy + the five
preregistered questions from SPEC.md."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from ny4h_range_reversal.analyze import (  # noqa: E402
    feature_scan, headline_block, quintile_table, regime_block, tstat,
)

OUT_DIR = os.path.join(_BASE, "reports", "pdlevel_fade")

FEATURES = ["pd_range_atr", "pd_range_rel20", "gap_open_pct",
            "bars_since_day_open", "sweep_depth_atr", "vol_ratio_touch",
            "dist_ema200_pct", "rsi_entry", "rsi_div_strength", "rr_planned",
            "stop_pct", "fee_r", "entry_hour_ny"]


def variant_block(t: pd.DataFrame, label: str) -> list[str]:
    lines = [f"### {label} — exit-variant comparison (gross)", "",
             "| slice | n | plain | t | scratch | scaleout | fee_r |",
             "|---|---|---|---|---|---|---|"]
    for (tf, arm), g in t.groupby(["exec_tf", "arm"]):
        lines.append(f"| {tf} {arm} | {len(g)} | {g.gross_r.mean():+.3f} "
                     f"| {tstat(g.gross_r):+.1f} | {g.r_scratch.mean():+.3f} "
                     f"| {g.r_scaleout.mean():+.3f} | {g.fee_r.mean():.2f} |")
    lines.append("")
    return lines


def main() -> None:
    t = pd.read_parquet(os.path.join(OUT_DIR, "trades_all.parquet"))
    t5d = t[(t.exec_tf == "5m") & (t.arm == "diverge")].copy()
    t5l = t[(t.exec_tf == "5m") & (t.arm == "limit")].copy()

    lines = ["# Predefined-Level Range Fade — backtest report (2026-07-16)", "",
             f"Rows: {len(t)} (arms × exit variants; 5m 10-sym + 15m 12-sym). "
             f"Period: {t.entry_time.min()} → {t.entry_time.max()}.", ""]

    lines += variant_block(t, "All slices")

    # time-exit decomposition (the integrity check)
    lines += ["### Time-exit decomposition (576/192-bar cap marks)", "",
              "| slice | n | gross | ex-time gross | time n | time mean R | share of total R |",
              "|---|---|---|---|---|---|---|"]
    for (tf, arm), g in t.groupby(["exec_tf", "arm"]):
        ti = g[g.exit_reason == "time"]
        nt = g[g.exit_reason != "time"]
        share = ti.gross_r.sum() / g.gross_r.sum() if g.gross_r.sum() != 0 else np.nan
        lines.append(f"| {tf} {arm} | {len(g)} | {g.gross_r.mean():+.3f} "
                     f"| {nt.gross_r.mean():+.3f} | {len(ti)} "
                     f"| {ti.gross_r.mean():+.2f} | {share:+.1%} |")
    lines.append("")

    for sl, name in ((t5l, "5m limit"), (t5d, "5m diverge")):
        lines += headline_block(sl, f"Headline — {name} (plain exits)")
        lines += regime_block(sl, "regime5", f"Per-regime — {name}")

    lines += regime_block(t5d, "btc_regime5", "Per-regime (BTC) — 5m diverge")

    # paired: same episode booked in both arms
    for tf in ("5m", "15m"):
        a = t[(t.exec_tf == tf) & (t.arm == "limit")]
        b = t[(t.exec_tf == tf) & (t.arm == "diverge")]
        m = a.merge(b, on=["symbol", "ny_date", "side"], suffixes=("_lim", "_div"))
        if len(m):
            lines += [f"### Paired episodes ({tf}, both arms booked, n={len(m)}): "
                      f"limit {m.gross_r_lim.mean():+.3f} vs diverge "
                      f"{m.gross_r_div.mean():+.3f}", ""]

    for target in ("gross_r",):
        for sl, name in ((t5d, "5m_diverge"), (t5l, "5m_limit")):
            scan = feature_scan(sl, target, features=FEATURES)
            scan.to_csv(os.path.join(OUT_DIR, f"feature_scan_{name}.csv"), index=False)
            lines += [f"### Feature scan — Q5−Q1 on gross ({name})", "",
                      "| feature | n | Q1 | Q5 | Δ | mono ρ | AUC | Δ H1 | Δ H2 | both |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in scan.iterrows():
                lines.append(
                    f"| {r.feature} | {r.n} | {r.q1:+.3f} | {r.q5:+.3f} "
                    f"| {r.delta_q5_q1:+.3f} | {r.monotone_rho:+.2f} | {r.auc_win:.3f} "
                    f"| {r.delta_h1:+.3f} | {r.delta_h2:+.3f} "
                    f"| {'YES' if r.both_halves_same_sign else 'no'} |")
            lines.append("")

    for f in ("sweep_depth_atr", "rsi_div_strength", "pd_range_atr"):
        lines += quintile_table(t5d, f)

    with open(os.path.join(OUT_DIR, "report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", os.path.join(OUT_DIR, "report.md"))


if __name__ == "__main__":
    main()
