#!/usr/bin/env python3
"""Run the predefined-level range fade across the local crypto universe."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402
from ny4h_range_reversal.regimes import attach_regimes  # noqa: E402
from pdlevel_fade.engine import PDConfig, run_symbol  # noqa: E402

SYMBOLS_ALL = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT",
               "ETH", "LINK", "LTC", "SOL", "XRP"]
NO_5M = {"ETH", "SOL"}
OUT_DIR = os.path.join(_BASE, "reports", "pdlevel_fade")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for tf in ("5m", "15m"):
        t0 = time.time()
        rows = 0
        for sym in SYMBOLS_ALL:
            if tf == "5m" and sym in NO_5M:
                continue
            cfg = PDConfig(costs=TransactionCosts.for_asset(sym), exec_tf=tf,
                           max_hold_bars=576 if tf == "5m" else 192,
                           min_prevday_bars=200 if tf == "5m" else 70)
            tr = run_symbol(sym, cfg)
            if tr.empty:
                continue
            tr["exec_tf"] = tf
            rows += len(tr)
            frames.append(tr)
        print(f"  {tf:>3s}: {rows:6d} rows ({time.time()-t0:.0f}s)", flush=True)

    trades = pd.concat(frames, ignore_index=True)
    print("attaching regimes…", flush=True)
    trades = attach_regimes(trades)
    trades.to_parquet(os.path.join(OUT_DIR, "trades_all.parquet"), index=False)
    print(f"saved {len(trades)} rows")

    summ = (trades.groupby(["exec_tf", "arm"])
            .agg(n=("gross_r", "size"), plain=("gross_r", "mean"),
                 scratch=("r_scratch", "mean"), scaleout=("r_scaleout", "mean"),
                 wr=("gross_r", lambda s: (s > 0).mean()),
                 med_rr=("rr_planned", "median"), fee_r=("fee_r", "mean"))
            .round(3))
    print(summ.to_string())


if __name__ == "__main__":
    main()
