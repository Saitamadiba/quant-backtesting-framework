"""Addendum 2 — session-level ranging-day gate, decided BEFORE the window opens.

Built from the author's own definition of a trending day ("price stays outside
the value areas and rides"): the share of pre-window bars closing inside the
+-1SD value area. Strictly causal — only bars from the session anchor to the last
bar before the first tradeable bar. Produced the "his filter makes it worse"
result in reports/vwap_value_area/REPORT.md.
"""
import sys; sys.path.insert(0,'/Users/saitamadiba/Quant/Backtesting')
import pandas as pd, numpy as np
from zoneinfo import ZoneInfo
from vwap_value_area.engine import (_anchored_vwap,_session_sd,_sess_groups,_add_atr,
                                    load_cached,load_crypto)
from vwap_value_area.analyze import load, cell_stats, fmt, CRYPTO
NY=ZoneInfo("America/New_York")

def day_features(df, anchor, win_start_et):
    df=df.reset_index(drop=True)
    ts=df["timestamp"]; ny=ts.dt.tz_convert(NY)
    hr=ny.dt.hour.to_numpy()+ny.dt.minute.to_numpy()/60.0
    g=_sess_groups(ts,anchor); bis=df.groupby(g).cumcount().to_numpy()
    h=df.high.to_numpy(float); l=df.low.to_numpy(float); c=df.close.to_numpy(float)
    v=df.volume.to_numpy(float); tp=(h+l+c)/3
    mid=_anchored_vwap(tp,v,g); sd=_session_sd(tp,v,mid,g,bis)
    pre = hr < win_start_et                      # bars BEFORE the window opens
    out=[]
    for sess in np.unique(g):
        m=(g==sess)&pre
        if m.sum()<24: continue
        cc=c[m]; mm=mid[m]; ss=sd[m]
        ok=np.isfinite(ss)&(ss>0)
        if ok.sum()<12: continue
        cc,mm,ss=cc[ok],mm[ok],ss[ok]
        inside1=float(np.mean(np.abs(cc-mm)<=ss))          # his criterion
        inside2=float(np.mean(np.abs(cc-mm)<=2*ss))
        path=np.abs(np.diff(cc)).sum()
        er=float(abs(cc[-1]-cc[0])/path) if path>0 else np.nan
        rng=(h[m].max()-l[m].min())
        out.append(dict(sess=sess,
                        date=pd.Timestamp(ts[m].iloc[-1]).tz_convert(NY).date(),
                        time_in_va1=inside1, time_in_va2=inside2,
                        pre_er=er, pre_range_pct=rng/cc[-1]))
    return pd.DataFrame(out)

def report(name, bars, anchor, win, trades):
    F=day_features(bars,anchor,win)
    T=trades.merge(F,on="date",how="inner")
    T["dt"]=pd.to_datetime(T["date"]); med=T["dt"].median(); TOLL=T.fee_r.mean()
    print(f"\n########## {name} ##########")
    print(f"  sessions classified={len(F)}   trades matched={len(T):,}   toll={TOLL:.4f}R")
    for f,lbl,rev in [("time_in_va1","share of pre-window bars INSIDE the 70% value area (his rule)",True),
                      ("pre_er","pre-window efficiency ratio (low = ranging)",False)]:
        d=T.dropna(subset=[f]).copy(); d["q"]=pd.qcut(d[f],5,labels=False,duplicates="drop")
        print(f"\n  {lbl}")
        for q,s in d.groupby("q"):
            st=cell_stats(s,"gross_r")
            h1=s[s.dt<=med].gross_r.mean(); h2=s[s.dt>med].gross_r.mean()
            tag = "RANGING" if (q==4)==rev else ("TRENDING" if (q==0)==rev else "")
            tag = ["Q1","Q2","Q3","Q4","Q5"][q]+(f" {tag}" if tag else "")
            print(f"    {tag:14} n={st['n']:<6} gross={st['mean']:+.4f} t={st['t']:+5.2f} "
                  f"net={s.net_r.mean():+.4f}  H1={h1:+.4f} H2={h2:+.4f}")
    # the gate as he'd trade it: only the most-ranging half of days
    cut=T.time_in_va1.median()
    for nm,sub in [("ALL days",T),("RANGING half only (his gate)",T[T.time_in_va1>cut]),
                   ("TRENDING half only",T[T.time_in_va1<=cut])]:
        st=cell_stats(sub,"gross_r"); h1=sub[sub.dt<=med].gross_r; h2=sub[sub.dt>med].gross_r
        print(f"  {nm:30} n={st['n']:<6} gross={st['mean']:+.4f} t={st['t']:+5.2f} "
              f"net={sub.net_r.mean():+.4f}  H1={h1.mean():+.4f} H2={h2.mean():+.4f}")

nd=_add_atr(load_cached("NDX"))
tn=load(["NDX"]); tn=tn[(tn.confirm=="S2")&(tn.target=="T2")&(tn.bands=="VWAP")]
report("NASDAQ-100 — day-type gate decided before 03:00 ET", nd, "cme", 3.0, tn)

cb=pd.concat([load_crypto(s,"5m").assign(symbol=s) for s in ["BTC","ETH","SOL"]],ignore_index=True)
tc=load(CRYPTO); tc=tc[(tc.confirm=="S2")&(tc.target=="T2")&(tc.bands=="VWAP")&(tc.symbol.isin(["BTC","ETH","SOL"]))]
for s in ["BTC","ETH","SOL"]:
    report(f"CRYPTO {s} — day-type gate decided before 08:00 UTC",
           _add_atr(cb[cb.symbol==s].drop(columns=["symbol"])), "utc", 3.0, tc[tc.symbol==s])
