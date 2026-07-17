#!/usr/bin/env python3
"""Run the multi-TF SMC strategy: 1h->5m (crypto) and 4h->15m (crypto+NQ)."""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402
from ny4h_range_reversal.regimes import attach_regimes  # noqa: E402
from smc_mtf.engine import SMCConfig, run_symbol  # noqa: E402

SYMBOLS_ALL = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT",
               "ETH", "LINK", "LTC", "SOL", "XRP"]
NO_5M = {"ETH", "SOL"}
OUT_DIR = os.path.join(_BASE, "reports", "smc_mtf")

COMBOS = [("1h", "5m", [s for s in SYMBOLS_ALL if s not in NO_5M]),
          ("4h", "15m", SYMBOLS_ALL + ["NQ"])]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for htf, ltf, syms in COMBOS:
        t0 = time.time()
        rows = 0
        for sym in syms:
            cfg = SMCConfig(htf_tf=htf, ltf_tf=ltf,
                            costs=TransactionCosts.for_asset(sym),
                            max_hold_bars=576 if ltf == "5m" else 192,
                            session_gap_exit=(sym == "NQ"))
            tr = run_symbol(sym, cfg)
            if tr.empty:
                continue
            tr["combo"] = f"{htf}->{ltf}"
            rows += len(tr)
            frames.append(tr)
        print(f"  {htf}->{ltf}: {rows:6d} rows ({time.time()-t0:.0f}s)", flush=True)

    trades = pd.concat(frames, ignore_index=True)
    print("attaching regimes…", flush=True)
    crypto = trades[trades.symbol != "NQ"]
    nq = trades[trades.symbol == "NQ"].copy()
    crypto = attach_regimes(crypto)
    nq["regime5"] = None
    nq["btc_regime5"] = None
    trades = pd.concat([crypto, nq], ignore_index=True)
    trades.to_parquet(os.path.join(OUT_DIR, "trades_all.parquet"), index=False)
    print(f"saved {len(trades)} rows")

    summ = (trades.groupby(["combo", trades.symbol == "NQ"])
            .agg(n=("gross_r", "size"), plain=("gross_r", "mean"),
                 be=("r_be", "mean"),
                 wr=("gross_r", lambda s: (s > 0).mean()),
                 med_stop=("stop_pct", "median"), fee_r=("fee_r", "mean"))
            .round(4))
    print(summ.to_string())


if __name__ == "__main__":
    main()
