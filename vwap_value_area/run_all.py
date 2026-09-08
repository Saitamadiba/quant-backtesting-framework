"""Run the pre-registered arm grid across the crypto panel + NQ futures."""
from __future__ import annotations

import itertools
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts   # noqa: E402
from vwap_value_area.engine import (Cfg, blackout_status, CRYPTO, DATA_DIR, load_cached,      # noqa: E402
                                    load_crypto, run_symbol)

OUT = os.path.join(DATA_DIR, "trades")
os.makedirs(OUT, exist_ok=True)

CONFIRMS = ["S2", "S1", "TOUCH"]
TARGETS = ["T1", "T2", "T3", "SCALE"]
BANDS = ["VWAP", "BB"]


def _cfg_for(sym: str, confirm: str, target: str, bands: str, **kw) -> Cfg:
    if sym in ("NQ", "QQQ"):
        base = dict(anchor="cme" if sym == "NQ" else "rth",
                    cost_kind="points" if sym == "NQ" else "pct",
                    rt_cost_points=1.0,
                    rt_cost_pct=0.0004)          # QQQ: ~1c spread + comms on ~$700
    else:
        base = dict(anchor="utc", cost_kind="pct",
                    rt_cost_pct=TransactionCosts.for_asset(sym).round_trip_cost_pct)
    base.update(kw)
    return Cfg(confirm=confirm, target=target, bands=bands, **base)


def _load(sym: str) -> pd.DataFrame:
    return load_cached(sym) if sym in ("NQ", "QQQ") else load_crypto(sym, "5m")


def work(sym: str) -> str:
    t0 = time.time()
    df = _load(sym)
    if sym in ("NQ", "QQQ"):
        from vwap_value_area.engine import _add_atr
        df = _add_atr(df)
    frames = []
    for confirm, target, bands in itertools.product(CONFIRMS, TARGETS, BANDS):
        r = run_symbol(sym, _cfg_for(sym, confirm, target, bands), df=df.copy())
        if not r.empty:
            frames.append(r)
    if not frames:
        return f"{sym}: EMPTY"
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(os.path.join(OUT, f"{sym}.parquet"))
    return f"{sym}: {len(out):>7,} trades across {len(frames)} cells  ({time.time()-t0:.0f}s)"


if __name__ == "__main__":
    syms = CRYPTO + ["NQ", "QQQ"]
    print(f"news blackout: {blackout_status()}", flush=True)
    with Pool(min(len(syms), os.cpu_count() or 4)) as p:
        for line in p.imap_unordered(work, syms):
            print(line, flush=True)
