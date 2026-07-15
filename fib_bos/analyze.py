#!/usr/bin/env python3
"""Fib BOS — per-regime assessment + winner/loser patterns, both arms.

Primary slices: trigger arm (faithful video) and limit618 arm (blind-fib
maker probe), each per exec TF. Net shown three ways for the limit arm:
taker RT (standard), and maker-entry mixed-exit counterfactual (2bps
entry; 2bps TP exits, one-way taker on stop exits) — labeled bound.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from ny4h_range_reversal.analyze import (  # noqa: E402
    feature_scan, headline_block, quintile_table, regime_block,
)

OUT_DIR = os.path.join(_BASE, "reports", "fib_bos")

FEATURES = [
    "leg_atr", "leg_pct", "entry_fib", "bos_dist_atr", "zone_touch_bars",
    "retrace_depth_frac", "rr_planned", "stop_pct", "fee_r",
    "trigger_vol_ratio", "dist_ema200_pct", "entry_hour_ny",
    "prior_signals_today",
]

ONEWAY_TAKER = {"BTC": 0.00075, "ETH": 0.00075, "SOL": 0.00105}


def add_maker_mixed(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    ow = t["symbol"].map(ONEWAY_TAKER).fillna(0.0018)
    is_tp = t["exit_reason"].isin(["tp", "tp_gap"])
    t["net_mm"] = t["gross_r"] - (0.0002 + np.where(is_tp, 0.0002, ow)) / t["stop_pct"]
    return t


def cat_block(t: pd.DataFrame, col: str, label: str) -> list[str]:
    lines = [f"### {label} — split by {col}", "",
             "| value | n | WR | grossR | netR |", "|---|---|---|---|---|"]
    for val, g in sorted(t.groupby(col, dropna=False),
                         key=lambda kv: -kv[1]["gross_r"].mean()):
        lines.append(f"| {val} | {len(g)} | {(g.gross_r > 0).mean():.1%} "
                     f"| {g.gross_r.mean():+.3f} | {g.net_r.mean():+.3f} |")
    lines.append("")
    return lines


def main() -> None:
    trades = add_maker_mixed(pd.read_parquet(os.path.join(OUT_DIR, "trades_all.parquet")))

    lines = ["# Fibonacci BOS Continuation — backtest report (causal-fill engine v1.1)", "",
             f"Rows: {len(trades)} across exec TFs 5m/15m/1h × arms trigger/limit618. "
             f"Period: {trades['entry_time'].min()} → {trades['entry_time'].max()}.", "",
             "### Arm × TF summary", "",
             "| exec | arm | n | WR | grossR | netR taker | netR maker-mix (limit only) |",
             "|---|---|---|---|---|---|---|"]
    for (tf, arm), g in trades.groupby(["exec_tf", "arm"]):
        mm = f"{g.net_mm.mean():+.3f}" if arm == "limit618" else "—"
        lines.append(f"| {tf} | {arm} | {len(g)} | {(g.gross_r > 0).mean():.1%} "
                     f"| {g.gross_r.mean():+.3f} | {g.net_r.mean():+.3f} | {mm} |")
    lines.append("")

    for tf in ("5m", "15m", "1h"):
        for arm in ("trigger", "limit618"):
            sl = trades[(trades.exec_tf == tf) & (trades.arm == arm)]
            if len(sl) < 500:
                continue
            lines += headline_block(sl, f"Headline — {tf} {arm}")
            lines += regime_block(sl, "regime5", f"Per-regime (own regime5) — {tf} {arm}")

    t5t = trades[(trades.exec_tf == "5m") & (trades.arm == "trigger")]
    lines += cat_block(t5t, "trigger_type", "5m trigger arm")
    lines += cat_block(t5t, "session", "5m trigger arm")
    lines += cat_block(t5t, "side", "5m trigger arm")
    lines += cat_block(t5t, "prior_was_stopout", "5m trigger arm")

    for tf, arm in (("5m", "trigger"), ("5m", "limit618"), ("1h", "limit618")):
        sl = trades[(trades.exec_tf == tf) & (trades.arm == arm)]
        scan = feature_scan(sl, "gross_r", features=FEATURES)
        scan.to_csv(os.path.join(OUT_DIR, f"feature_scan_{tf}_{arm}.csv"), index=False)
        lines += [f"### Feature scan — Q5−Q1 on gross_r ({tf} {arm})", "",
                  "| feature | n | Q1 | Q5 | Δ | mono ρ | AUC | Δ H1 | Δ H2 | both |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in scan.iterrows():
            lines.append(f"| {r.feature} | {r.n} | {r.q1:+.3f} | {r.q5:+.3f} "
                         f"| {r.delta_q5_q1:+.3f} | {r.monotone_rho:+.2f} | {r.auc_win:.3f} "
                         f"| {r.delta_h1:+.3f} | {r.delta_h2:+.3f} "
                         f"| {'YES' if r.both_halves_same_sign else 'no'} |")
        lines.append("")

    for f in ("leg_atr", "entry_fib", "rr_planned"):
        lines += quintile_table(trades[(trades.exec_tf == "1h") & (trades.arm == "limit618")], f)

    lines += ["### Multiple-testing note", "",
              "13 numeric + 5 categorical features × 3 slices scanned. Only monotone, "
              "both-halves, mechanism-backed patterns are candidates; the leg_atr cell "
              "is post-hoc and decaying — see verdict.", ""]

    with open(os.path.join(OUT_DIR, "report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", os.path.join(OUT_DIR, "report.md"))


if __name__ == "__main__":
    main()
