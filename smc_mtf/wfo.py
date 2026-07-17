#!/usr/bin/env python3
"""SMC-MTF walk-forward optimization — grid phase.

Preregistered grid (24 combos, every lever mechanism-motivated):
  min_zone_atr ∈ {0.15, 0.40}   (HTF zone quality)
  ce_frac      ∈ {0.5, 0.75}    (entry depth in the LTF FVG)
  rr           ∈ {2, 4, 6}      (exit geometry)
  min_leg_retrace ∈ {0, 0.618}  (the video's fib filter)
Fixed: swing_n=3, windows (choch 48 / fvg 12 / fill 24), buffer, floors.

Each (tf-combo, param-combo) runs once over full history; WFO selection
slices the resulting books by calendar window (static params ⇒ one run
serves every window). Usage:
  python3 -m smc_mtf.wfo grid   # run all grids (parallel, hours=0.5)
"""

from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402
from smc_mtf.engine import SMCConfig, run_symbol  # noqa: E402

OUT = os.path.join(_BASE, "reports", "smc_mtf", "wfo")
SYMBOLS = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT",
           "ETH", "LINK", "LTC", "SOL", "XRP"]
NO_5M = {"ETH", "SOL"}
TFS = [("1d", "1h", 240), ("4h", "15m", 192), ("1h", "5m", 576)]

GRID = [dict(min_zone_atr=z, ce_frac=cf, rr=r, min_leg_retrace=fb)
        for z, cf, r, fb in itertools.product(
            [0.15, 0.40], [0.5, 0.75], [2.0, 4.0, 6.0], [0.0, 0.618])]
FROZEN = dict(min_zone_atr=0.15, ce_frac=0.5, rr=4.0, min_leg_retrace=0.0)
ONEWAY = {"BTC": 0.00075, "ETH": 0.00075, "SOL": 0.00105, "NQ": 0.0004}
MAKER = 0.0002


def combo_id(p: dict) -> str:
    return f"z{p['min_zone_atr']}_c{p['ce_frac']}_r{int(p['rr'])}_f{p['min_leg_retrace']}"


def run_one(task):
    htf, ltf, hold, params, cid = task
    frames = []
    syms = SYMBOLS + ["NQ"] if ltf != "5m" else [s for s in SYMBOLS if s not in NO_5M]
    for sym in syms:
        cfg = SMCConfig(htf_tf=htf, ltf_tf=ltf, max_hold_bars=hold,
                        session_gap_exit=(sym == "NQ"),
                        costs=TransactionCosts.for_asset(sym), **params)
        tr = run_symbol(sym, cfg)
        if tr.empty:
            continue
        frames.append(tr)
    if not frames:
        return cid, 0
    t = pd.concat(frames, ignore_index=True)
    ow = t.symbol.map(ONEWAY).fillna(0.0018)
    is_tp = t.exit_reason.isin(["tp", "tp_gap"])
    t["net_mm"] = t.gross_r - (MAKER + np.where(is_tp, MAKER, ow)) / t.stop_pct
    for k, v in params.items():
        t[k] = v
    t["combo_id"] = cid
    d = os.path.join(OUT, f"{htf}-{ltf}")
    os.makedirs(d, exist_ok=True)
    t.to_parquet(os.path.join(d, f"{cid}.parquet"), index=False)
    return f"{htf}-{ltf}/{cid}", len(t)


def main() -> None:
    tasks = [(htf, ltf, hold, p, combo_id(p))
             for (htf, ltf, hold) in TFS for p in GRID]
    print(f"{len(tasks)} grid runs …", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for f in as_completed(futs):
            cid, n = f.result()
            done += 1
            print(f"  [{done}/{len(tasks)}] {cid}: {n} trades", flush=True)
    print("grid complete")


if __name__ == "__main__":
    main()
