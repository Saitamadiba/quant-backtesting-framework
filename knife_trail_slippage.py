#!/usr/bin/env python3
"""knife_trail_slippage.py — measure execution slippage + fee-R on live knife trades.

Goal: size the buffer for a fee-aware breakeven profit-lock. We need the empirical
distribution of (a) fee-R per trade (= where breakeven sits, in R) and (b) the slip
below an armed lock level when the synthetic 1s-poll market close fires.

Per-trade reconstruction from funded_trades columns:
    risk_px  = |level - sl|                 (price risk = 1R)
    gross_R  = sign * (level - exit)/risk_px = realized excursion at exit
    net_R    = pnl_usd / risk_usd            (should equal stored r_multiple)
    fee_R    = gross_R - net_R               (the "vig"; breakeven sits at +fee_R gross)
    max_fav_r= stored high-water excursion (peak R reached)
    giveback = max_fav_r - gross_R           (R surrendered from peak to exit)

An "armed" trade is one whose max_fav_r >= TRIGGER_R. For an armed trade that
retraces (gross_R < max_fav_r and gross_R well below TP), the trail market-close
fired; the slip below the lock = LOCK_R - gross_R.

Read-only. Prints distributions; no DB writes.
"""
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DBS = {
    "100k-demo": "dashboard/databases/knife_funded_100k.db",
    "10k-maker": "dashboard/databases/knife_funded_maker.db",
    "taker":     "dashboard/databases/knife_funded_taker.db",
}

# Knife geometry / config (from detector.py + knife_bybit_funded.py defaults).
RR_TP = 1.5            # TP sits at +1.5R gross
TRIGGER_R = 0.80       # user's current arm trigger (was 1.2 default)
LOCK_R = 0.50          # user's current profit-lock level (was 1.2 default)


def load(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT key,symbol,direction,entry_mode,level,sl,tp,qty,risk_usd,"
            "exit_price,max_fav_r,r_multiple,pnl_usd,exit_reason,closed_at_utc "
            "FROM funded_trades WHERE closed_at_utc IS NOT NULL AND exit_price IS NOT NULL",
            con)
    finally:
        con.close()
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("level", "sl", "tp", "qty", "risk_usd", "exit_price",
              "max_fav_r", "r_multiple", "pnl_usd"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    sign = np.where(df["direction"].str.upper() == "SHORT", -1.0, 1.0)
    df["risk_px"] = (df["level"] - df["sl"]).abs()
    df["risk_px_pct"] = df["risk_px"] / df["level"]                # stop width %
    df["gross_R"] = sign * (df["exit_price"] - df["level"]) / df["risk_px"]
    df["net_R_calc"] = df["pnl_usd"] / df["risk_usd"]
    df["fee_R"] = df["gross_R"] - df["net_R_calc"]
    df["giveback_R"] = df["max_fav_r"] - df["gross_R"]
    df["armed"] = df["max_fav_r"] >= TRIGGER_R
    return df


def pctl(s: pd.Series, ps=(10, 25, 50, 75, 90, 95)) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "   (no data)"
    return "  ".join(f"p{p}={np.percentile(s, p):+.3f}" for p in ps)


def report(name: str, df: pd.DataFrame) -> None:
    print(f"\n{'='*78}\n{name}  (n_closed={len(df)})\n{'='*78}")
    if df.empty:
        return
    print(f"date range: {df['closed_at_utc'].min()}  ->  {df['closed_at_utc'].max()}")

    # 0) Sanity: does r_multiple == net_R_calc (pnl/risk_usd)?  Convention check.
    chk = (df["r_multiple"] - df["net_R_calc"]).abs()
    print(f"\n[convention] |r_multiple - pnl/risk_usd|: "
          f"median={chk.median():.4f} max={chk.max():.4f}  "
          f"-> r_multiple is {'NET' if chk.median() < 0.05 else 'NOT net(check!)'}")

    # 1) Fee-R = the vig = where breakeven sits (in gross R).
    print(f"\n[fee-R / the vig]  breakeven sits at +fee_R gross. percentiles:")
    print("  " + pctl(df["fee_R"]))
    print(f"  mean fee_R = {df['fee_R'].mean():+.3f}R   "
          f"stop width %: median={df['risk_px_pct'].median()*100:.3f}%")
    print(f"  by entry_mode:")
    for m, g in df.groupby(df["entry_mode"].fillna("?")):
        print(f"    {m:>7}: n={len(g):3d}  fee_R median={g['fee_R'].median():+.3f}  "
              f"stopwidth%={g['risk_px_pct'].median()*100:.3f}")

    # 2) Armed trades: where do they book vs the LOCK level?
    armed = df[df["armed"]].copy()
    print(f"\n[armed trades]  max_fav_r >= {TRIGGER_R} : n={len(armed)} of {len(df)}")
    if len(armed):
        # classify
        reached_tp = armed["gross_R"] >= RR_TP - 0.15
        wicked = armed["gross_R"] < 0          # blew through to/past SL side
        locked = (~reached_tp) & (~wicked)     # retraced & booked between 0..TP
        print(f"    reached TP(+{RR_TP}R):   {reached_tp.sum()}")
        print(f"    trail-locked (0..TP): {locked.sum()}")
        print(f"    wicked thru to neg:   {wicked.sum()}")
        lk = armed[locked]
        if len(lk):
            print(f"\n  [trail-locked book level]  gross_R at exit (intended lock={LOCK_R}):")
            print("    " + pctl(lk["gross_R"]))
            print(f"\n  [slip below lock]  slip = LOCK_R({LOCK_R}) - gross_R  (>0 = booked under lock):")
            slip = LOCK_R - lk["gross_R"]
            print("    " + pctl(slip))
            print(f"    mean slip={slip.mean():+.3f}R   "
                  f"frac booked under lock={ (slip>0).mean()*100:.0f}%")
            print(f"\n  [giveback from peak]  max_fav_r - gross_R for locked trades:")
            print("    " + pctl(lk["giveback_R"]))
        # net outcome of armed trades
        print(f"\n  [armed net_R outcome]  did 'protection' actually protect?")
        print("    " + pctl(armed["net_R_calc"]))
        print(f"    armed trades net<0: {(armed['net_R_calc']<0).mean()*100:.0f}%  "
              f"mean net_R={armed['net_R_calc'].mean():+.3f}")

    # 3) The core question: with lock at +0.50R gross and fee_R median ~X,
    #    what's the net at the lock?  net_at_lock = LOCK_R - fee_R
    net_at_lock = LOCK_R - df["fee_R"]
    print(f"\n[lock @ +{LOCK_R}R gross -> implied NET if it filled exactly at lock]:")
    print("    " + pctl(net_at_lock))
    print(f"    frac of trades where +{LOCK_R}R lock is a NET LOSS (fee_R>{LOCK_R}): "
          f"{(df['fee_R']>LOCK_R).mean()*100:.0f}%")


def main():
    frames = []
    for name, path in DBS.items():
        p = Path(path)
        if not p.exists():
            print(f"!! missing {path}", file=sys.stderr)
            continue
        df = enrich(load(path))
        df["arm_db"] = name
        report(name, df)
        frames.append(df)
    if frames:
        allf = pd.concat(frames, ignore_index=True)
        report("ALL ARMS POOLED", allf)


if __name__ == "__main__":
    main()
