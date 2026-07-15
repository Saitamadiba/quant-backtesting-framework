#!/usr/bin/env python3
"""Run the HTF-bias structure scalper across the local crypto universe.

Every entry is booked under BOTH stop arms (confirm-candle / pullback
extreme) as a paired comparison. ETH/SOL run as a 15m-execution
sensitivity arm (no local 5m bars).
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
from structure_scalper.engine import StructConfig, run_symbol  # noqa: E402

SYMBOLS_5M = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "LINK", "LTC", "XRP"]
SYMBOLS_15M = ["ETH", "SOL"]

OUT_DIR = os.path.join(_BASE, "reports", "structure_scalper")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for sym in SYMBOLS_5M:
        t0 = time.time()
        cfg = StructConfig(costs=TransactionCosts.for_asset(sym), exec_tf="5m")
        tr = run_symbol(sym, cfg)
        tr["exec_tf"] = "5m"
        frames.append(tr)
        print(f"  {sym:5s} 5m : {len(tr):6d} rows ({time.time()-t0:.0f}s)", flush=True)
    for sym in SYMBOLS_15M:
        t0 = time.time()
        cfg = StructConfig(costs=TransactionCosts.for_asset(sym), exec_tf="15m",
                           max_wait_retest=16, max_wait_confirm=4, max_hold_bars=192)
        tr = run_symbol(sym, cfg)
        tr["exec_tf"] = "15m"
        frames.append(tr)
        print(f"  {sym:5s} 15m: {len(tr):6d} rows ({time.time()-t0:.0f}s)", flush=True)

    trades = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    print("attaching regimes…", flush=True)
    trades = attach_regimes(trades)

    path = os.path.join(OUT_DIR, "trades_all.parquet")
    trades.to_parquet(path, index=False)
    print(f"saved {len(trades)} rows -> {path}")

    summ = (trades.groupby(["symbol", "exec_tf", "arm"])
            .agg(n=("net_r", "size"), gross_mean=("gross_r", "mean"),
                 net_mean=("net_r", "mean"),
                 win_rate=("gross_r", lambda s: (s > 0).mean()))
            .round(4))
    summ.to_csv(os.path.join(OUT_DIR, "summary_by_symbol.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
