#!/usr/bin/env python3
"""Structure scalper — per-regime assessment + winner/loser pattern scan.

Reuses the generic table/scan machinery from ny4h_range_reversal.analyze.
Primary basis = 5m arm, stop_mode='confirm' (the video's first-listed
stop); pullback arm reported as a paired comparison.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from ny4h_range_reversal.analyze import (  # noqa: E402
    feature_scan, headline_block, quintile_table, regime_block, yearly_block,
)

OUT_DIR = os.path.join(_BASE, "reports", "structure_scalper")

FEATURES = [
    "bos_dist_atr", "bos_vol_ratio", "bars_to_retest", "bars_to_confirm",
    "retest_depth_atr", "retrace_frac", "confirm_body_frac",
    "confirm_range_atr", "stop_pct", "fee_r", "dist_ema200_pct",
    "hours_since_htf_break", "prior_signals_today", "entry_hour_ny",
]
CATEGORICALS = ["side", "confirm_type", "session", "structure_seq",
                "trend_align", "regime5", "btc_regime5"]


def cat_block(t: pd.DataFrame, col: str) -> list[str]:
    lines = [f"### Split by {col} (5m confirm arm)", "",
             "| value | n | WR | grossR | netR |", "|---|---|---|---|---|"]
    for val, g in sorted(t.groupby(col, dropna=False),
                         key=lambda kv: -kv[1]["gross_r"].mean()):
        lines.append(f"| {val} | {len(g)} | {(g.gross_r > 0).mean():.1%} "
                     f"| {g.gross_r.mean():+.3f} | {g.net_r.mean():+.3f} |")
    lines.append("")
    return lines


def main() -> None:
    trades = pd.read_parquet(os.path.join(OUT_DIR, "trades_all.parquet"))
    t5c = trades[(trades.exec_tf == "5m") & (trades.arm == "confirm")].copy()
    t5p = trades[(trades.exec_tf == "5m") & (trades.arm == "pullback")].copy()
    t15 = trades[(trades.exec_tf == "15m") & (trades.arm == "confirm")].copy()

    lines = ["# HTF-Bias Structure Scalper — backtest report", "",
             f"Entries: {len(t5c)} (5m arm; each booked under 2 stop arms) "
             f"+ {len(t15)} (ETH/SOL 15m sensitivity). "
             f"Period: {t5c['entry_time'].min()} → {t5c['entry_time'].max()}.", ""]

    lines += headline_block(t5c, "Headline — 5m arm, stop = confirmation candle")
    lines += headline_block(t5p, "Headline — 5m arm, stop = pullback extreme (paired)")
    if not t15.empty:
        lines += headline_block(t15, "Sensitivity — ETH/SOL on 15m execution (confirm stop)")
    lines += yearly_block(t5c)

    lines += regime_block(t5c, "regime5",
                          "Per-regime — OWN-asset regime5 (5m confirm arm, gross basis)")
    lines += regime_block(t5c, "btc_regime5", "Per-regime — BTC regime5 (5m confirm arm)")

    for col in ("side", "confirm_type", "session", "structure_seq", "trend_align"):
        lines += cat_block(t5c, col)

    for target in ("gross_r", "net_r"):
        scan = feature_scan(t5c, target, features=FEATURES)
        scan.to_csv(os.path.join(OUT_DIR, f"feature_scan_{target}.csv"), index=False)
        lines += [f"### Feature scan — quintile Q5−Q1 delta on {target} (5m confirm arm)", "",
                  "| feature | n | Q1 | Q5 | Δ(Q5−Q1) | mono ρ | AUC(win) | Δ H1 | Δ H2 | both halves |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in scan.iterrows():
            lines.append(
                f"| {r.feature} | {r.n} | {r.q1:+.3f} | {r.q5:+.3f} "
                f"| {r.delta_q5_q1:+.3f} | {r.monotone_rho:+.2f} | {r.auc_win:.3f} "
                f"| {r.delta_h1:+.3f} | {r.delta_h2:+.3f} "
                f"| {'YES' if r.both_halves_same_sign else 'no'} |")
        lines.append("")

    for f in ("stop_pct", "retest_depth_atr", "retrace_frac", "bars_to_retest",
              "hours_since_htf_break", "bos_dist_atr"):
        lines += quintile_table(t5c, f)

    lines += ["### Multiple-testing note", "",
              f"{len(FEATURES)} numeric + {len(CATEGORICALS)} categorical features "
              "scanned on 2 targets — only monotone, both-halves-same-sign, "
              "mechanism-backed patterns are candidates, and even those need a "
              "forward shadow before any use.", ""]

    path = os.path.join(OUT_DIR, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
