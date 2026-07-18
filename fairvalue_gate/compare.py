"""Apply the fair-value context gate to every existing strategy and compare.

For each strategy trade-set, annotate with fair-value context (session VWAP +
prior-day POC/node/value-zone) and measure whether conditioning on fair value
lifts gross and net. The keystone gate (near-POC + HVN + in-value) is the
'at established fair value' condition that lifted VWAP; the far-POC cell is the
'stretched from fair value' condition that should favour fades. Comparison
reveals which mechanism families the gate helps, and whether it moves any book
toward clearing the fee wall (esp. smc_1d1h, the deployed net-+ @maker arm).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
from fairvalue_gate.context import annotate  # noqa: E402

OUT = os.path.join(_BASE, "reports", "fairvalue_gate")

# (label, path, family)  family: C=continuation/trend, F=fade/reversion
STRATS = [
    ("vwap_A  (VWAP pullback)", "reports/vwap_confluence/trades_A.parquet", "C"),
    ("smc_all (SMC MTF)", "reports/smc_mtf/trades_all.parquet", "C"),
    ("smc_1d1h(SMC deployed)", "reports/smc_mtf/trades_1d1h.parquet", "C"),
    ("orb     (OR breakout)", "reports/orb_fvg/trades_primary.parquet", "C"),
    ("struct  (BOS-retest)", "reports/structure_scalper/trades_all.parquet", "C"),
    ("fib     (fib BOS)", "reports/fib_bos/trades_all.parquet", "C"),
    ("oar     (opening fade)", "reports/oar/trades_engulf_ib.parquet", "F"),
    ("vp_pocrev(POC reversion)", "reports/volume_profile/trades_POCREV.parquet", "F"),
    ("ny4h    (4H range fade)", "reports/ny4h_range_reversal/trades_all.parquet", "F"),
    ("pdlevel (PD-level fade)", "reports/pdlevel_fade/trades_all.parquet", "F"),
]


def ct(t, col="gross_r"):
    if len(t) < 2:
        return (np.nan, np.nan, len(t))
    if "ny_date" in t:
        cl = t.groupby([t.symbol, t.ny_date])[col].mean()
    else:
        cl = t.groupby([t.symbol, pd.to_datetime(t.entry_time).dt.date])[col].mean()
    n = len(cl)
    if n < 2 or cl.std() == 0:
        return (t[col].mean(), np.nan, n)
    return (t[col].mean(), cl.mean() / (cl.std(ddof=1) / np.sqrt(n)), n)


def cell(t):
    gm, gt, nd = ct(t)
    return dict(n=len(t), gross=gm, t=gt, net=t["net"].mean() if len(t) else np.nan,
                net_mm=t["net_mm"].mean() if len(t) else np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    detail = []
    for label, path, fam in STRATS:
        if not os.path.exists(path):
            continue
        a = annotate(pd.read_parquet(path))
        if a.empty:
            continue
        base = cell(a)
        nd = a[a.node_density.notna() & a.dist_pPOC_atr.notna()]
        near = nd[nd.dist_pPOC_atr.abs() < 0.75]
        far = nd[nd.dist_pPOC_atr.abs() > 2.0]
        hvn = nd[nd.node_density >= 1.5]
        lvn = nd[nd.node_density < 0.5]
        keystone = nd[(nd.dist_pPOC_atr.abs() < 1.0) & (nd.node_density >= 1.0) & (nd.vp_zone == "in_value")]
        c = {k: cell(v) for k, v in [("near", near), ("far", far), ("hvn", hvn),
                                     ("lvn", lvn), ("keystone", keystone)]}
        # favourable gate by family: C -> keystone (at fair value); F -> far (stretched)
        fav = c["keystone"] if fam == "C" else c["far"]
        rows.append(dict(strat=label, fam=fam, base_n=base["n"], base_gross=base["gross"],
                         base_t=base["t"], base_net=base["net"],
                         gate=("keystone" if fam == "C" else "far-POC"),
                         gate_n=fav["n"], gate_gross=fav["gross"], gate_t=fav["t"],
                         gate_net=fav["net"], gate_net_mm=fav["net_mm"],
                         d_gross=fav["gross"] - base["gross"]))
        detail.append((label, fam, base, c))

    df = pd.DataFrame(rows)
    L = []; P = L.append
    P("=" * 108)
    P("FAIR-VALUE GATE (session VWAP + prior-day POC/node/value) applied to ALL strategies")
    P("=" * 108)
    P("\nFavourable gate per family: continuation(C) -> 'keystone' = near-POC + HVN + in-value")
    P("(at established fair value);  fade(F) -> 'far-POC' = |dist to POC| > 2 ATR (stretched).")
    P(f"\n{'strategy':26s} fam {'base_n':>7s} {'base_gr':>8s} {'base_t':>7s} {'base_net':>8s} | "
      f"{'gate':>9s} {'gate_n':>7s} {'gate_gr':>8s} {'gate_t':>7s} {'gate_net':>8s} {'Δgross':>8s}")
    for _, r in df.iterrows():
        P(f"{r.strat:26s}  {r.fam}  {r.base_n:7d} {r.base_gross:+8.4f} {r.base_t:+7.2f} "
          f"{r.base_net:+8.3f} | {r.gate:>9s} {r.gate_n:7d} {r.gate_gross:+8.4f} "
          f"{r.gate_t:+7.2f} {r.gate_net:+8.3f} {r.d_gross:+8.4f}")

    P("\n## Full context breakdown per strategy (gross | net_taker), n in [])")
    for label, fam, base, c in detail:
        P(f"\n{label} [{fam}]  baseline n={base['n']} gross {base['gross']:+.4f} (t {base['t']:+.2f}) net {base['net']:+.3f}")
        for k in ["near", "far", "hvn", "lvn", "keystone"]:
            v = c[k]
            P(f"    {k:9s} n={v['n']:6d} gross {v['gross']:+.4f} (t {v['t']:+.2f}) net {v['net']:+.3f} net_mm {v['net_mm']:+.3f}")

    # family summary
    P("\n## Does the favourable gate LIFT gross? (Δgross by family)")
    for fam, nm in [("C", "continuation"), ("F", "fade")]:
        sub = df[df.fam == fam]
        pos = (sub.d_gross > 0).sum()
        P(f"  {nm:12s}: {pos}/{len(sub)} strategies lifted by their favourable gate; "
          f"mean Δgross {sub.d_gross.mean():+.4f}")
    P("\n## Money test: any gated cell NET-positive at taker?")
    anypos = False
    for label, fam, base, c in detail:
        for k, v in c.items():
            if v["n"] >= 100 and v["net"] > 0:
                P(f"  {label} [{k}] net_taker {v['net']:+.3f} (n={v['n']}, gross {v['gross']:+.4f})")
                anypos = True
    if not anypos:
        P("  NONE — every gated cell (n>=100) remains net-taker-negative. Gate sharpens gross, not net.")

    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "COMPARISON.txt"), "w") as f:
        f.write(txt + "\n")
    df.to_parquet(os.path.join(OUT, "gate_comparison.parquet"))


if __name__ == "__main__":
    main()
