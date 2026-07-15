#!/usr/bin/env python3
"""Run the NY4H range-reversal backtest across the local 5m crypto universe.

Primary arm : 5m execution on every symbol with deep local 5m history.
Sensitivity : 15m execution for ETH / SOL (no local 5m bars) — flagged.

Writes reports/ny4h_range_reversal/trades_all.parquet (+ csv summary).
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402
from ny4h_range_reversal.engine import NY4HConfig, run_symbol  # noqa: E402
from ny4h_range_reversal.regimes import attach_regimes  # noqa: E402

SYMBOLS_5M = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "LINK", "LTC", "XRP"]
SYMBOLS_15M = ["ETH", "SOL"]  # sensitivity arm — no local 5m bars

OUT_DIR = os.path.join(_BASE, "reports", "ny4h_range_reversal")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for sym in SYMBOLS_5M:
        t0 = time.time()
        cfg = NY4HConfig(costs=TransactionCosts.for_asset(sym), exec_tf="5m")
        tr = run_symbol(sym, cfg)
        tr["exec_tf"] = "5m"
        frames.append(tr)
        print(f"  {sym:5s} 5m : {len(tr):5d} trades  ({time.time()-t0:.0f}s)", flush=True)
    for sym in SYMBOLS_15M:
        t0 = time.time()
        cfg = NY4HConfig(costs=TransactionCosts.for_asset(sym), exec_tf="15m",
                         max_hold_bars=192, min_range_bars=14)
        tr = run_symbol(sym, cfg)
        tr["exec_tf"] = "15m"
        frames.append(tr)
        print(f"  {sym:5s} 15m: {len(tr):5d} trades  ({time.time()-t0:.0f}s)", flush=True)

    trades = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    print("attaching regimes…", flush=True)
    trades = attach_regimes(trades)

    path = os.path.join(OUT_DIR, "trades_all.parquet")
    trades.to_parquet(path, index=False)
    print(f"saved {len(trades)} trades -> {path}")

    summ = (trades.groupby(["symbol", "exec_tf"])
            .agg(n=("net_r", "size"), gross_mean=("gross_r", "mean"),
                 net_mean=("net_r", "mean"),
                 win_rate=("gross_r", lambda s: (s > 0).mean()))
            .round(4))
    summ.to_csv(os.path.join(OUT_DIR, "summary_by_symbol.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
