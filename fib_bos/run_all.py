#!/usr/bin/env python3
"""Run the Fibonacci BOS continuation strategy across the local universe.

TF ladder: 5m (10-symbol universe), 15m and 1h (full 12 incl. ETH/SOL).
Each setup books a `trigger` (faithful) and a `limit618` (blind-fib maker
probe) arm. Bar-denominated geometry constant across TFs.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402
from ny4h_range_reversal.regimes import attach_regimes  # noqa: E402
from fib_bos.engine import FibConfig, run_symbol  # noqa: E402

SYMBOLS_ALL = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT",
               "ETH", "LINK", "LTC", "SOL", "XRP"]
NO_5M = {"ETH", "SOL"}
TFS = ["5m", "15m", "1h"]

OUT_DIR = os.path.join(_BASE, "reports", "fib_bos")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for tf in TFS:
        t0 = time.time()
        rows = 0
        for sym in SYMBOLS_ALL:
            if tf == "5m" and sym in NO_5M:
                continue
            cfg = FibConfig(costs=TransactionCosts.for_asset(sym), exec_tf=tf)
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
    path = os.path.join(OUT_DIR, "trades_all.parquet")
    trades.to_parquet(path, index=False)
    print(f"saved {len(trades)} rows -> {path}")

    summ = (trades.groupby(["exec_tf", "arm"])
            .agg(n=("net_r", "size"), gross=("gross_r", "mean"),
                 net=("net_r", "mean"),
                 wr=("gross_r", lambda s: (s > 0).mean()),
                 med_rr=("rr_planned", "median"),
                 med_stop=("stop_pct", "median"),
                 fee_r=("fee_r", "mean"))
            .round(4))
    summ.to_csv(os.path.join(OUT_DIR, "summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
