#!/usr/bin/env python3
"""Timeframe grid for the HTF-bias structure scalper.

Bar-denominated geometry is held CONSTANT across TFs (swing N=3, retest
wait 48 bars, confirm wait 12 bars, hold cap 576 bars) so every arm is a
pure rescaling of the same pattern — the scale-invariant reading of the
strategy. Combos:

  exec 15m / HTF 1h    exec 15m / HTF 4h    exec 1h / HTF 4h
  exec 4h  / HTF 1d(resampled)              exec 5m / HTF 4h (control)

All 12 symbols run wherever the exec TF exists locally (ETH/SOL rejoin
at 15m+). Baseline exec 5m / HTF 1h lives in trades_all.parquet.
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

SYMBOLS_ALL = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT",
               "ETH", "LINK", "LTC", "SOL", "XRP"]
NO_5M = {"ETH", "SOL"}

COMBOS = [  # (exec_tf, htf_tf)
    ("15m", "1h"),
    ("15m", "4h"),
    ("1h", "4h"),
    ("4h", "1d"),
    ("5m", "4h"),
]

OUT_DIR = os.path.join(_BASE, "reports", "structure_scalper")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for exec_tf, htf_tf in COMBOS:
        t0 = time.time()
        rows = 0
        for sym in SYMBOLS_ALL:
            if exec_tf == "5m" and sym in NO_5M:
                continue
            cfg = StructConfig(costs=TransactionCosts.for_asset(sym),
                               exec_tf=exec_tf, htf_tf=htf_tf)
            tr = run_symbol(sym, cfg)
            if tr.empty:
                continue
            tr["exec_tf"] = exec_tf
            tr["htf_tf"] = htf_tf
            rows += len(tr)
            frames.append(tr)
        print(f"  {exec_tf:>3s}/{htf_tf:<3s}: {rows:6d} rows ({time.time()-t0:.0f}s)",
              flush=True)

    trades = pd.concat(frames, ignore_index=True)
    print("attaching regimes…", flush=True)
    trades = attach_regimes(trades)
    path = os.path.join(OUT_DIR, "trades_tf_grid.parquet")
    trades.to_parquet(path, index=False)
    print(f"saved {len(trades)} rows -> {path}")

    summ = (trades.groupby(["exec_tf", "htf_tf", "arm"])
            .agg(n=("net_r", "size"), gross=("gross_r", "mean"),
                 net=("net_r", "mean"),
                 wr=("gross_r", lambda s: (s > 0).mean()),
                 med_stop=("stop_pct", "median"),
                 fee_r=("fee_r", "mean"))
            .round(4))
    print(summ.to_string())


if __name__ == "__main__":
    main()
