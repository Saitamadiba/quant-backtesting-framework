"""HMM choppy/trending regime gate for the session-VWAP value-area fade.

FROZEN BAR (stated before any model was fitted):
  A state qualifies as a usable regime gate only if ALL of:
    (a) its gross R clears the panel's toll,
    (b) positive in BOTH halves of the sample,
    (c) beats the family-wise |t| bar across every state x K x panel tested,
    (d) survives pick-on-H1 / read-once-on-H2.
  Anything less is recorded as an ATLAS (a description of the tape), not a gate.

LEAK DISCIPLINE (reusing backtrader_framework/optimization/hmm_regime.py):
  filtered forward probabilities only - never Viterbi, never smoothed;
  rolling refit (fit trailing 90d, filter forward 20d); IS-only standardisation;
  posterior carried across seams; states re-labelled each window by a fixed rule
  so the semantic label is stable across refits.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from backtrader_framework.optimization.hmm_regime import GaussianHMM
from vwap_value_area.engine import load_cached, load_crypto, _add_atr
from vwap_value_area.analyze import load, cell_stats, clustered_t, fmt, CRYPTO
RNG=np.random.default_rng(11)

def hourly(df):
    d=df.set_index("timestamp").resample("1h",label="right",closed="right").agg(
        open=("open","first"),high=("high","max"),low=("low","min"),
        close=("close","last"),volume=("volume","sum")).dropna().reset_index()
    return _add_atr(d)

def features(d, W=96, q=6):
    """Causal choppy/trending features from CLOSED hourly bars only."""
    c=d.close; r=np.log(c).diff()
    rv=r.rolling(W).std()
    path=r.abs().rolling(W).sum()
    er=(np.log(c)-np.log(c).shift(W)).abs()/path            # low = choppy
    v1=r.rolling(W).var()
    vq=r.rolling(q).sum().rolling(W).var()
    vr=vq/(q*v1)                                            # <1 mean-reverting, >1 trending
    atrp=d.atr/c
    F=pd.DataFrame({"rv":rv,"er":er,"vr":vr,"atrp":atrp})
    return F

def rolling_states(d, F, K, train_days=90, apply_days=20):
    ts=d["timestamp"]; n=len(d)
    ok=F.notna().all(axis=1).to_numpy()
    P=np.full((n,K),np.nan); LBL=np.full(n,-1)
    t0=ts.min()+pd.Timedelta(days=train_days)
    starts=pd.date_range(t0, ts.max(), freq=f"{apply_days}D")
    carry=None; fails=[]; mus=[]
    for s in starts:
        tr=(ts>=s-pd.Timedelta(days=train_days))&(ts<s)&ok
        ap=(ts>=s)&(ts<s+pd.Timedelta(days=apply_days))&ok
        if tr.sum()<400 or ap.sum()<10: continue
        Xtr=F[tr].to_numpy(float); mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
        try:
            # vol_feature_index=1 -> the engine relabels states by ascending mean
            # EFFICIENCY RATIO, so state 0 = most CHOPPY, state K-1 = most TRENDING,
            # consistently across every refit.
            m=GaussianHMM(n_states=K,max_iter=120,vol_feature_index=1).fit((Xtr-mu)/sd)
        except Exception as e:
            fails.append(f"{s.date()}: {type(e).__name__}: {e}"); continue
        Xap=(F[ap].to_numpy(float)-mu)/sd
        pr=m.forward_filter(Xap, init_state_probs=carry)
        carry=pr[-1]
        P[np.where(ap)[0]]=pr
        LBL[np.where(ap)[0]]=np.argmax(pr,axis=1)
        mus.append(m.mu[:,1])
    if fails: print(f"    [refit failures: {len(fails)}/{len(starts)}] e.g. {fails[0]}")
    if mus:
        M=np.array(mus)
        print(f"    [state ER means, avg over {len(M)} refits]: "
              + "  ".join(f"S{i}={M[:,i].mean():+.2f}" for i in range(M.shape[1]))
              + "   (ascending = choppy -> trending)")
    return P,LBL

def study(name, bars, trades, K):
    d=hourly(bars); F=features(d)
    P,L=rolling_states(d,F,K)
    lab=pd.DataFrame({"timestamp":d["timestamp"],"state":L})
    lab=lab[lab.state>=0].copy()
    lab["timestamp"]=lab["timestamp"].astype("datetime64[ns, UTC]")
    T=trades.copy(); T["timestamp"]=pd.to_datetime(T.entry_time,utc=True).astype("datetime64[ns, UTC]")
    T=pd.merge_asof(T.sort_values("timestamp"), lab.sort_values("timestamp"),
                    on="timestamp", direction="backward",
                    tolerance=pd.Timedelta("2h")).dropna(subset=["state"])
    T["dt"]=pd.to_datetime(T["date"]); med=T["dt"].median(); toll=T.fee_r.mean()
    names={0:"CHOPPY",K-1:"TRENDING"} if K>1 else {}
    rows=[]
    for st,g in T.groupby("state"):
        s=cell_stats(g,"gross_r")
        rows.append(dict(state=f"S{int(st)} {names.get(int(st),'mid')}",n=s["n"],
                         gross=s["mean"],t=s["t"],net=g.net_r.mean(),
                         H1=g[g.dt<=med].gross_r.mean(),H2=g[g.dt>med].gross_r.mean()))
    R=pd.DataFrame(rows)
    R["clears_toll"]=R.gross>toll; R["both_halves"]=(R.H1>0)&(R.H2>0)
    print(f"\n--- {name}  K={K}   n={len(T):,}  toll={toll:.4f}  "
          f"(unlabelled/dropped: {len(trades)-len(T):,}) ---")
    print(fmt(R))
    return T,R

print(__doc__)
NDX=load_cached("NDX")
tn=load(["NDX"]); tn=tn[(tn.confirm=="S2")&(tn.target=="T2")&(tn.bands=="VWAP")]
res={}
for K in (2,3): res[("NDX",K)]=study("NASDAQ-100", NDX, tn, K)
tc=load(CRYPTO); tc=tc[(tc.confirm=="S2")&(tc.target=="T2")&(tc.bands=="VWAP")]
for sym in ["BTC","ETH"]:
    b=load_crypto(sym,"5m")
    for K in (2,3): res[(sym,K)]=study(f"CRYPTO {sym}", b, tc[tc.symbol==sym], K)

# family-wise bar across every state x K x panel
print("\n=== FAMILY-WISE BAR across all states x K x panels ===")
prep=[]
for (nm,K),(T,R) in res.items():
    for st,g in T.groupby("state"):
        x=g.gross_r.to_numpy(float); codes,uq=pd.factorize(g["cluster"])
        prep.append((len(x),x-x.mean(),codes,len(uq)))
B=2000; mx=np.empty(B)
for b in range(B):
    best=0.0
    for n,e,codes,ng in prep:
        w=RNG.choice([-1.0,1.0],size=ng)[codes]; eb=e*w; m=eb.mean()
        s_=np.bincount(codes,weights=eb-m,minlength=ng); se=np.sqrt((s_**2).sum())/n
        if se>0: best=max(best,abs(m/se))
    mx[b]=best
bar=np.quantile(mx,0.95)
print(f"  cells={len(prep)}   95th pct of max|t| under the null = {bar:.2f}")
best=[]
for (nm,K),(T,R) in res.items():
    for r in R.itertuples():
        if r.clears_toll and r.both_halves and abs(r.t)>bar: best.append((nm,K,r.state,r.gross,r.t))
print(f"  states passing (a) clears toll AND (b) both halves AND (c) beats the bar: {len(best)}")
for b_ in best: print("   ",b_)
