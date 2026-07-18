"""KEYSTONE — does volume-profile structure improve the VWAP signal?

Annotate the VWAP arm-A trades (reports/vwap_confluence/trades_A.parquet) with
PRIOR-session volume-profile context at entry (node density = HVN/LVN, value
zone, distance to prior POC), then slice arm-A gross by that context. If a VWAP
pullback that lands on a high-volume node / near POC beats one that lands in a
low-volume gap, volume profile adds orthogonal information to VWAP — the real
"complement" claim (the user's option 1). Family-wise + half-split guarded.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
from volume_profile.indicator import session_profiles, node_density  # noqa: E402

VWAP_A = os.path.join(_BASE, "reports", "vwap_confluence", "trades_A.parquet")
OUT = os.path.join(_BASE, "reports", "volume_profile")
NQ = "NQ"


def annotate(trades: pd.DataFrame) -> pd.DataFrame:
    """Add prior-session VP context columns to each trade."""
    parts = []
    for sym, g in trades.groupby("symbol"):
        if sym == NQ:  # corrupt multi-scale data, excluded (see VWAP report)
            continue
        tf = "5m" if sym not in ("ETH", "SOL") else "15m"
        sp = session_profiles(sym, tf)
        if sp.empty:
            continue
        sp = sp.set_index(sp["sess"].dt.tz_localize(None) if sp["sess"].dt.tz is not None else sp["sess"])
        g = g.copy()
        g["sess_key"] = pd.to_datetime(g["entry_time"]).dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
        rows = []
        for _, tr in g.iterrows():
            key = tr["sess_key"]
            if key not in sp.index:
                rows.append((np.nan, "none", np.nan)); continue
            r = sp.loc[key]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            price = tr["entry"]; a = tr["entry"] * tr["stop_pct"] / max(tr["stop_atr"], 1e-9)
            # recover atr from stop_pct/stop_atr: risk=stop_pct*entry, atr=risk/stop_atr
            atr = (tr["stop_pct"] * tr["entry"]) / tr["stop_atr"] if tr["stop_atr"] > 0 else np.nan
            nd = node_density(price, r)
            ppoc, pvah, pval = r["pPOC"], r["pVAH"], r["pVAL"]
            zone = "above_value" if price > pvah else ("below_value" if price < pval else "in_value")
            dist = (price - ppoc) / atr if atr and atr > 0 else np.nan
            rows.append((nd, zone, dist))
        g["vp_node_density"] = [x[0] for x in rows]
        g["vp_zone"] = [x[1] for x in rows]
        g["vp_dist_pPOC_atr"] = [x[2] for x in rows]
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def ct(t, col="gross_r"):
    if len(t) < 2:
        return (np.nan, np.nan, 0)
    cl = t.groupby([t.symbol, t.ny_date])[col].mean()
    n = len(cl)
    if n < 2 or cl.std() == 0:
        return (t[col].mean(), np.nan, n)
    return (t[col].mean(), cl.mean() / (cl.std(ddof=1) / np.sqrt(n)), n)


def line(name, t):
    if len(t) == 0:
        return f"{name:30s} n=0"
    gm, gt, nd = ct(t)
    return (f"{name:30s} n={len(t):6d} gross {gm:+.4f} (t {gt:+.2f}) "
            f"net_tk {t.net_taker_r.mean():+.3f} net_mm {t.net_mm_r.mean():+.3f} win {(t.gross_r>0).mean():.3f}")


def report(a):
    L = []; P = L.append
    P("=" * 100)
    P("KEYSTONE — does VOLUME PROFILE improve the VWAP signal? (VWAP arm A + VP context)")
    P("=" * 100)
    P(f"\nbaseline (all annotated crypto arm-A trades): ")
    P(line("  VWAP arm A (baseline)", a))

    P("\n## Slice by node density at entry (HVN=acceptance/magnet vs LVN=rejection)")
    a2 = a[a.vp_node_density.notna()]
    for lab, sub in [("LVN  (density<0.5)", a2[a2.vp_node_density < 0.5]),
                     ("mid  (0.5-1.0)", a2[(a2.vp_node_density >= 0.5) & (a2.vp_node_density < 1.0)]),
                     ("HVN  (1.0-2.0)", a2[(a2.vp_node_density >= 1.0) & (a2.vp_node_density < 2.0)]),
                     ("strong HVN (>=2.0)", a2[a2.vp_node_density >= 2.0]),
                     ("outside prior range (=0)", a2[a2.vp_node_density == 0.0])]:
        P(line("  " + lab, sub))

    P("\n## Slice by value zone (entry inside prior value area vs beyond it)")
    for z in ["in_value", "above_value", "below_value"]:
        P(line(f"  {z}", a[a.vp_zone == z]))

    P("\n## Slice by distance to prior POC (mean~=mode confluence when small)")
    a3 = a[a.vp_dist_pPOC_atr.notna()]
    ad = a3.assign(absd=a3.vp_dist_pPOC_atr.abs())
    for lab, sub in [("|dist|<0.5 ATR (VWAP~=POC)", ad[ad.absd < 0.5]),
                     ("0.5-1.5 ATR", ad[(ad.absd >= 0.5) & (ad.absd < 1.5)]),
                     (">=1.5 ATR (far from POC)", ad[ad.absd >= 1.5])]:
        P(line("  " + lab, sub))

    P("\n## Best confluence cell: HVN AND near-POC AND in-value (does VP stack lift gross?)")
    best = a[(a.vp_node_density >= 1.0) & (a.vp_zone == "in_value") & (a.vp_dist_pPOC_atr.abs() < 1.0)]
    P(line("  HVN + in-value + near-POC", best))
    worst = a[(a.vp_node_density < 0.5) | (a.vp_node_density == 0.0)]
    P(line("  LVN / single-print (avoid?)", worst))

    # family-wise note + half-split of the best cell
    from scipy.stats import norm
    P(f"\n   Bonferroni t-bar for ~10 VP slices @0.05: {norm.ppf(1-0.025/10):.2f}")
    if len(best) > 20:
        bs = best.sort_values("entry_time"); mid = len(bs) // 2
        for half, sub in [("H1", bs.iloc[:mid]), ("H2", bs.iloc[mid:])]:
            gm, gt, nd = ct(sub)
            P(f"   best-cell {half}: gross {gm:+.4f} (t {gt:+.2f}) net_tk {sub.net_taker_r.mean():+.3f}")
    return "\n".join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    trades = pd.read_parquet(VWAP_A)
    a = annotate(trades)
    a.to_parquet(os.path.join(OUT, "vwapA_vp_annotated.parquet"))
    txt = report(a)
    print(txt)
    with open(os.path.join(OUT, "KEYSTONE.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
