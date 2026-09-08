"""Long-history Nasdaq-100 run: validate the CFD volume proxy against real CME
volume on the overlap, then run the pre-registered arm grid with real futures costs."""
from __future__ import annotations
import itertools, os, sys
import numpy as np, pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
from vwap_value_area.engine import (Cfg, blackout_status, DATA_DIR, _add_atr, _anchored_vwap,   # noqa
                                    _sess_groups, _session_sd, run_symbol)

SRC = ("/private/tmp/claude-501/-Users-saitamadiba-Quant-Backtesting/"
       "97d5bad3-07aa-470b-ae76-e4fb993cab68/scratchpad/download/"
       "usatechidxusd-m5-bid-2019-01-01-2026-09-04.csv")


def build():
    d = pd.read_csv(SRC)
    d["timestamp"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    d = d[(d.volume > 0) & (d.high >= d.low) & (d.close > 0)]
    d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    d = _add_atr(d)
    d.to_parquet(os.path.join(DATA_DIR, "NDX.parquet"))
    return d


def validate(d):
    nq = pd.read_parquet(os.path.join(DATA_DIR, "NQ.parquet"))
    m = d.merge(nq[["timestamp", "close", "volume"]], on="timestamp", suffixes=("_cfd", "_nq"))
    if m.empty:
        print("  !! no overlap — cannot validate"); return
    g = _sess_groups(m["timestamp"], "cme"); b = m.groupby(g).cumcount().to_numpy()
    for k in ("cfd", "nq"):
        tp = m[f"close_{k}"].to_numpy(float); v = m[f"volume_{k}"].to_numpy(float)
        m[f"vwap_{k}"] = _anchored_vwap(tp, v, g)
        m[f"sd_{k}"] = _session_sd(tp, v, m[f"vwap_{k}"].to_numpy(), g, b)
    ok = m.dropna(subset=["sd_cfd", "sd_nq"])
    print(f"  overlap bars              : {len(m):,}")
    print(f"  5m-return correlation     : {m.close_cfd.pct_change().corr(m.close_nq.pct_change()):.4f}")
    print(f"  raw volume correlation    : {m.volume_cfd.corr(m.volume_nq):.4f}")
    print(f"  DEVELOPING VWAP corr      : {ok.vwap_cfd.corr(ok.vwap_nq):.6f}")
    print(f"  DEVELOPING SD-band corr   : {ok.sd_cfd.corr(ok.sd_nq):.4f}")
    err = np.abs(ok.vwap_cfd - ok.vwap_nq) / ok.close_nq
    print(f"  median |VWAP| error       : {np.median(err)*1e4:.2f} bps of price")
    print(f"  median SD ratio (cfd/nq)  : {np.median(ok.sd_cfd/ok.sd_nq):.3f}")


if __name__ == "__main__":
    d = build()
    print(f"news blackout: {blackout_status()}", flush=True)
    print(f"NDX 5m: n={len(d):,}  {d.timestamp.min()} -> {d.timestamp.max()}")
    print("\n=== CFD VOLUME-PROXY VALIDATION vs real CME volume ===")
    validate(d)
    print("\n=== ARM GRID — Nasdaq-100, 2019-2026, 1.0-point round-trip toll ===")
    rows = []
    for cf, tg, bd in itertools.product(["S2", "S1", "TOUCH"], ["T1", "T2", "T3", "SCALE"], ["VWAP", "BB"]):
        r = run_symbol("NDX", Cfg(confirm=cf, target=tg, bands=bd, anchor="cme",
                                  cost_kind="points", rt_cost_points=1.0), df=d.copy())
        if r.empty:
            continue
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(os.path.join(DATA_DIR, "trades", "NDX.parquet"))
    print(f"  wrote {len(out):,} trades")
