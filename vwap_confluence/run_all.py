"""Run the VWAP-confluence battery across all assets + NQ.

Arms (native TF: 10 alts @5m, ETH/SOL @15m, NQ @15m):
  A  pullback-continuation to VWAP + candle confirmation (the video's core)
  B  A + HTF (EMA200) trend confluence
  C  A + weekly-VWAP alignment confluence
  MA A but pullback level = EMA(100) instead of VWAP  (the keystone control)
  D  SD-band reversion (fade stretched -> VWAP)
Plus a 15m scale arm (A, 10 alts) to test whether the smaller fee toll at 15m
lets the VWAP gross edge clear net.
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402
from vwap_confluence.engine import VWAPConfig, run_symbol  # noqa: E402
from orb_fvg.regimes import attach_regimes  # noqa: E402

OUT = os.path.join(_BASE, "reports", "vwap_confluence")
ALTS_5M = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "LINK", "LTC", "XRP"]
NO_5M = ["ETH", "SOL"]


def _run(symbols, exec_tf, **kw):
    parts = []
    for s in symbols:
        cfg = VWAPConfig(exec_tf=exec_tf, costs=TransactionCosts.for_asset(s), **kw)
        t = run_symbol(s, cfg)
        if not t.empty:
            parts.append(t)
        print(f"  {s:5s} {exec_tf} arm={kw.get('arm','A')} n={0 if t.empty else len(t)}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def arm(name, exec_map=True, **kw):
    print(f"[{name}] {kw}", flush=True)
    if exec_map:
        t = pd.concat([_run(ALTS_5M, "5m", **kw), _run(NO_5M, "15m", **kw),
                       _run(["NQ"], "15m", **kw)], ignore_index=True)
    else:
        t = _run(ALTS_5M, "15m", **kw)
    t = attach_regimes(t)
    t.to_parquet(os.path.join(OUT, f"trades_{name}.parquet"))
    print(f"  -> {name} n={len(t)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    arm("A", arm="A")
    arm("B", arm="B")
    arm("C", arm="C")
    arm("MA", arm="MA")
    arm("D", arm="D")
    arm("A15", exec_map=False, arm="A")   # 15m scale test, 10 alts
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
