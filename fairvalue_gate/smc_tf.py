"""Show the full cross-strategy comparison + apply the fair-value / low-volume
-node gate to EVERY SMC timeframe combo (1h->5m, 4h->15m, 1d->1h) to test
whether the 1d->1h low-volume-node hit generalizes across timeframes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
from fairvalue_gate.context import annotate  # noqa: E402


def ct(t, col="gross_r"):
    if len(t) < 2:
        return (t[col].mean() if len(t) else np.nan, np.nan, len(t))
    key = t.ny_date if "ny_date" in t else pd.to_datetime(t.entry_time).dt.date
    cl = t.groupby([t.symbol, key])[col].mean()
    n = len(cl)
    if n < 2 or cl.std() == 0:
        return (t[col].mean(), np.nan, n)
    return (t[col].mean(), cl.mean() / (cl.std(ddof=1) / np.sqrt(n)), n)


def row(label, t):
    gm, gt, nd = ct(t)
    return (f"  {label:18s} n={len(t):6d} gross {gm:+.4f} (t {gt:+.2f}) "
            f"net_tk {t['net'].mean():+.3f}")


def main():
    L = []
    A = L.append

    # ---------- 1) full comparison table ----------
    A("=" * 96)
    A("FULL COMPARISON — fair-value gate on all strategies (favourable gate per family)")
    A("=" * 96)
    comp = pd.read_parquet(os.path.join(_BASE, "reports", "fairvalue_gate", "gate_comparison.parquet"))
    A(f"{'strategy':26s} {'fam':3s} {'base_gross':>10s} {'base_t':>7s} {'base_net':>8s}  "
      f"{'gate':>9s} {'gate_gross':>10s} {'gate_net':>8s} {'Δgross':>8s}")
    for _, r in comp.iterrows():
        A(f"{r.strat:26s}  {r.fam:2s} {r.base_gross:+10.4f} {r.base_t:+7.2f} {r.base_net:+8.3f}  "
          f"{r.gate:>9s} {r.gate_gross:+10.4f} {r.gate_net:+8.3f} {r.d_gross:+8.4f}")

    # ---------- 2) SMC per-timeframe gate ----------
    A("\n" + "=" * 96)
    A("SMC per-TIMEFRAME — does the low-volume-node (LVN) / fair-value gate generalize?")
    A("=" * 96)
    smc = annotate(pd.read_parquet(os.path.join(_BASE, "reports", "smc_mtf", "trades_all.parquet")))
    smc1d = annotate(pd.read_parquet(os.path.join(_BASE, "reports", "smc_mtf", "trades_1d1h.parquet")))
    smc1d["combo"] = "1d->1h"
    allsmc = pd.concat([smc, smc1d], ignore_index=True)

    for combo in ["1h->5m", "4h->15m", "1d->1h"]:
        c = allsmc[allsmc.combo == combo]
        nd = c[c.node_density.notna() & c.dist_pPOC_atr.notna()]
        A(f"\n--- {combo}  (baseline net_tk {c['net'].mean():+.3f}) ---")
        A(row("baseline", c))
        A(row("LVN (dens<0.5)", nd[nd.node_density < 0.5]))
        A(row("  dens<0.25", nd[nd.node_density < 0.25]))
        A(row("HVN (dens>=1.5)", nd[nd.node_density >= 1.5]))
        A(row("near-POC(<0.75)", nd[nd.dist_pPOC_atr.abs() < 0.75]))
        A(row("far-POC(>2)", nd[nd.dist_pPOC_atr.abs() > 2.0]))
        A(row("keystone", nd[(nd.dist_pPOC_atr.abs() < 1.0) & (nd.node_density >= 1.0) & (nd.vp_zone == "in_value")]))
        # LVN half-split
        lvn = nd[nd.node_density < 0.5].sort_values("entry_time")
        if len(lvn) >= 20:
            mid = len(lvn) // 2
            for hnm, sub in [("LVN H1", lvn.iloc[:mid]), ("LVN H2", lvn.iloc[mid:])]:
                gm, gt, n = ct(sub)
                A(f"    {hnm}: n={len(sub)} gross {gm:+.4f} (t {gt:+.2f}) net_tk {sub['net'].mean():+.3f}")

    # LVN gross-lift summary across TFs
    A("\n## Does LVN lift GROSS at every SMC timeframe? (mechanism generalization)")
    for combo in ["1h->5m", "4h->15m", "1d->1h"]:
        c = allsmc[allsmc.combo == combo]
        nd = c[c.node_density.notna()]
        base = c.gross_r.mean()
        lvn = nd[nd.node_density < 0.5].gross_r.mean()
        A(f"  {combo:9s} baseline gross {base:+.4f} -> LVN gross {lvn:+.4f}  (Δ {lvn-base:+.4f}, "
          f"{'LIFTS' if lvn>base else 'no lift'})")

    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(_BASE, "reports", "fairvalue_gate", "SMC_TIMEFRAMES.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
