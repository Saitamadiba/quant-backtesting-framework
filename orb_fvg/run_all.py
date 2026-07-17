"""Run the ORB+FVG battery across all assets + NQ and save trade frames.

Primary arm  : 10 alts @ 5m, ETH/SOL @ 15m, NQ @ 15m, engulf-confirmed, rr=3.
Sensitivity  : no-engulf arm; 15m-exec scale test (10 alts); rr=2.
All trades get own+BTC regime5 attached. Outputs parquet to reports/orb_fvg/.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402
from orb_fvg.engine import ORBConfig, run_symbol  # noqa: E402
from orb_fvg.regimes import attach_regimes  # noqa: E402

OUT = os.path.join(_BASE, "reports", "orb_fvg")
ALTS_5M = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "LINK", "LTC", "XRP"]
NO_5M = ["ETH", "SOL"]


def _run(symbols, exec_tf, **kw) -> pd.DataFrame:
    parts = []
    for s in symbols:
        cfg = ORBConfig(exec_tf=exec_tf, costs=TransactionCosts.for_asset(s), **kw)
        t = run_symbol(s, cfg)
        if not t.empty:
            parts.append(t)
        print(f"  {s:5s} {exec_tf} n={0 if t.empty else len(t)}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    print("[primary] 10 alts @5m + ETH/SOL @15m + NQ @15m, engulf, rr=3")
    prim = pd.concat([
        _run(ALTS_5M, "5m"),
        _run(NO_5M, "15m"),
        _run(["NQ"], "15m"),
    ], ignore_index=True)
    prim = attach_regimes(prim)
    prim.to_parquet(os.path.join(OUT, "trades_primary.parquet"))
    print(f"  -> primary n={len(prim)}")

    print("[no-engulf] same universe, enter on first retest close")
    noeng = pd.concat([
        _run(ALTS_5M, "5m", require_engulf=False),
        _run(NO_5M, "15m", require_engulf=False),
        _run(["NQ"], "15m", require_engulf=False),
    ], ignore_index=True)
    noeng = attach_regimes(noeng)
    noeng.to_parquet(os.path.join(OUT, "trades_noengulf.parquet"))
    print(f"  -> no-engulf n={len(noeng)}")

    print("[scale15] 10 alts @15m, engulf, rr=3 (scale-invariance test)")
    sc = _run(ALTS_5M, "15m")
    sc = attach_regimes(sc)
    sc.to_parquet(os.path.join(OUT, "trades_scale15.parquet"))
    print(f"  -> scale15 n={len(sc)}")

    print("[rr2] primary universe, rr=2 (target sensitivity)")
    rr2 = pd.concat([
        _run(ALTS_5M, "5m", rr=2.0),
        _run(NO_5M, "15m", rr=2.0),
        _run(["NQ"], "15m", rr=2.0),
    ], ignore_index=True)
    rr2 = attach_regimes(rr2)
    rr2.to_parquet(os.path.join(OUT, "trades_rr2.parquet"))
    print(f"  -> rr2 n={len(rr2)}")

    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
