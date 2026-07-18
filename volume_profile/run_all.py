"""Run standalone VP mechanisms crypto-wide (NQ excluded — corrupt data)."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402
from volume_profile.engine import VPConfig, run_symbol  # noqa: E402

OUT = os.path.join(_BASE, "reports", "volume_profile")
ALTS_5M = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "LINK", "LTC", "XRP"]
NO_5M = ["ETH", "SOL"]


def _run(setup):
    parts = []
    for s, tf in [(x, "5m") for x in ALTS_5M] + [(x, "15m") for x in NO_5M]:
        t = run_symbol(s, VPConfig(exec_tf=tf, setup=setup, costs=TransactionCosts.for_asset(s)))
        if not t.empty:
            parts.append(t)
        print(f"  {s:5s} {tf} {setup} n={0 if t.empty else len(t)}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for su in ["VABREAK", "POCREV", "VAREJ"]:
        print(f"[{su}]", flush=True)
        t = _run(su)
        t.to_parquet(os.path.join(OUT, f"trades_{su}.parquet"))
        print(f"  -> {su} n={len(t)}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
