"""Run the OAR battery. The ib_ratio gate is a day-level filter on
opening_vol_ratio, so we run gate=0 once per (mode, target) and slice the
gate sweep post-hoc in analyze.py (verified identical to direct gating).

Arms (all: 10 alts @5m + ETH/SOL @15m + NQ @15m, 2021-2026):
  engulf_ib   : engulf trigger, target = opposite IB edge   (faithful)
  engulf_rr2  : engulf trigger, target = fixed 2R            (disentangle)
  hammer_ib   : hammer-break trigger, target = opposite IB edge
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402
from oar.engine import OARConfig, run_symbol  # noqa: E402
from orb_fvg.regimes import attach_regimes  # noqa: E402 (reuse the same attach)

OUT = os.path.join(_BASE, "reports", "oar")
ALTS_5M = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "LINK", "LTC", "XRP"]
NO_5M = ["ETH", "SOL"]


def _run(symbols, exec_tf, **kw):
    parts = []
    for s in symbols:
        cfg = OARConfig(exec_tf=exec_tf, ib_ratio_min=0.0,
                        costs=TransactionCosts.for_asset(s), **kw)
        t = run_symbol(s, cfg)
        if not t.empty:
            parts.append(t)
        print(f"  {s:5s} {exec_tf} n={0 if t.empty else len(t)}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def arm(name, **kw):
    print(f"[{name}] {kw}")
    t = pd.concat([_run(ALTS_5M, "5m", **kw), _run(NO_5M, "15m", **kw),
                   _run(["NQ"], "15m", **kw)], ignore_index=True)
    t = attach_regimes(t)
    t.to_parquet(os.path.join(OUT, f"trades_{name}.parquet"))
    print(f"  -> {name} n={len(t)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    arm("engulf_ib", entry_mode="engulf", target_mode="ib")
    arm("engulf_rr2", entry_mode="engulf", target_mode="fixed", rr=2.0)
    arm("hammer_ib", entry_mode="hammer", target_mode="ib")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
