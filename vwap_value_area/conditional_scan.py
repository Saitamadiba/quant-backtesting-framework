"""Addendum 1 — conditional scan: which regimes/conditions/assets perform best?

Quintile scan over every PRE-fill feature plus session/dow/side/symbol, with a
wild-cluster-bootstrap family-wise bar across the whole scan, half-split
stability, and a shuffled-label placebo. Produced the "no conditional structure,
one era" result in reports/vwap_value_area/REPORT.md.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vwap_value_area.analyze import *
import pandas as pd, numpy as np
RNGS=np.random.default_rng(7)

def scan(P, feats, cats, label, B=800):
    P=P.copy(); P["dt"]=pd.to_datetime(P["date"]); med=P["dt"].median()
    toll=P.fee_r.mean(); cells=[]
    for f in feats:
        d=P.dropna(subset=[f])
        if d[f].nunique()<5: continue
        try: b=pd.qcut(d[f],5,labels=False,duplicates="drop")
        except Exception: continue
        for q in sorted(pd.unique(b.dropna())):
            g=d[b==q]
            if len(g)<60: continue
            cells.append((f"{f} Q{int(q)+1}", g))
    for f in cats:
        for v,g in P.groupby(f):
            if len(g)<60: continue
            cells.append((f"{f}={v}", g))
    rows=[]
    for nm,g in cells:
        s=cell_stats(g,"gross_r")
        h1=g[g.dt<=med].gross_r; h2=g[g.dt>med].gross_r
        rows.append(dict(cell=nm,n=s["n"],gross=s["mean"],t=s["t"],
                         net=g.net_r.mean(),H1=h1.mean(),H2=h2.mean()))
    R=pd.DataFrame(rows)
    # family-wise bar across the WHOLE scan
    prep=[]
    for nm,g in cells:
        x=g.gross_r.to_numpy(float); codes,uq=pd.factorize(g["cluster"])
        prep.append((len(x),x-x.mean(),codes,len(uq)))
    mx=np.empty(B)
    for b in range(B):
        best=0.0
        for n,e,codes,ng in prep:
            w=RNGS.choice([-1.0,1.0],size=ng)[codes]; eb=e*w; m=eb.mean()
            s_=np.bincount(codes,weights=eb-m,minlength=ng)
            se=np.sqrt((s_**2).sum())/n
            if se>0: best=max(best,abs(m/se))
        mx[b]=best
    bar=np.quantile(mx,0.95)
    R["clears_toll"]=R.gross>toll
    R["both_halves_pos"]=(R.H1>0)&(R.H2>0)
    R["beats_FWER"]=R.t.abs()>bar
    print(f"\n########## {label} ##########")
    print(f"  cells scanned={len(R)}   toll={toll:.4f}R   family-wise |t| bar (95th)={bar:.2f}")
    top=R.sort_values("gross",ascending=False).head(10)
    print(fmt(top[["cell","n","gross","t","net","H1","H2","clears_toll","both_halves_pos","beats_FWER"]]))
    surv=R[(R.gross>toll)&R.both_halves_pos&R.beats_FWER]
    print(f"  --> cells that clear the toll AND hold in both halves AND beat the family-wise bar: {len(surv)}")
    if len(surv): print(fmt(surv[["cell","n","gross","t","net","H1","H2"]]))
    return R

c=load(CRYPTO); P=c[(c.confirm=="S2")&(c.target=="T2")&(c.bands=="VWAP")]
d=load(["NDX"]); Q=d[(d.confirm=="S2")&(d.target=="T2")&(d.bands=="VWAP")]
FE=["pre_er20","pre_er60","pre_atr_pct","pre_band_w_pct","pre_excursion_sd",
    "pre_stop_atr","pre_rr_planned","pre_vol_ratio","pre_setup_bars","pre_bars_since_anchor","pre_sd_pos"]
CA=["pre_session","pre_dow","side"]
scan(Q,FE,CA,"NASDAQ-100 primary arm — conditional scan")
scan(P,FE,CA+["symbol"],"CRYPTO primary arm — conditional scan")
