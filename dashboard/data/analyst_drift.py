"""WS7 — the analyst-drift seat, read honestly at very small n.

The plan scopes this workstream deliberately small: WS1's posterior on hit rate
and M today, a VIX-band atlas if the data allows, and WS4/WS6 **deferred until
n ≥ 300**. The seat's own frozen review gate (n ≥ 1,000 member events;
M ≥ +0.05pp ∧ day-clustered t ≥ 1.5 ∧ paper net P&L > 0) is untouched, and
nothing here shortens it.

**The finding that matters is a counting one.** The store holds 22 closed rows
with 22 distinct `news_id` — but only **13 distinct (symbol, eff_date)** pairs,
and `m_gross` is identical inside each pair. Five separate headlines about SNOW
on 2026-09-03 are five rows, five news ids, and **one bet on one stock's
overnight drift on one day**. Counting rows runs the seat's gate **1.69× too
fast**.

The seat's prereg already specifies a *day-clustered t*, so the clustering is
anticipated for the statistic — what is ambiguous is the **n**. Raising that at
n=22 costs nothing; discovering it at n≈1,000 would mean the gate opened on a
sample 40% smaller than it claimed. This module therefore reports both countings
side by side and defaults to the clustered one.

*Picture: five reporters filing the same story is five bylines and one event.*
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
EVENTS = _ROOT / "flow_aux_data" / "analyst" / "events.parquet"

GATE_N = 1000            # the seat's own frozen review gate
GATE_M_PP = 0.05         # official-print forward gross M, in percentage points
WS4_WS6_N = 300          # the plan's floor before ML / tuning may touch this seat


def load_events() -> pd.DataFrame:
    """The seat's event store, pulled locally by `backfill_analyst_events.py`."""
    if not EVENTS.exists():
        return pd.DataFrame()
    df = pd.read_parquet(EVENTS)
    for c in ("created_utc", "ts_et", "resolved_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, format="mixed")
    return df


def closed_events(df: Optional[pd.DataFrame] = None,
                  cluster: bool = True) -> pd.DataFrame:
    """Closed, priced rows — collapsed to one row per BET when `cluster`.

    A bet is (symbol, eff_date): the same stock's same-day overnight drift. The
    seat can hold several `news_id` rows for it when more than one headline
    lands, and each is filled separately, but they all wager on one move.
    `m_gross` is the event's drift and is identical across the group; `m_fill`
    is per-fill and is averaged; P&L sums.
    """
    d = df if df is not None else load_events()
    if d.empty:
        return pd.DataFrame()
    c = d[(d.get("status") == "closed") & d.get("pnl_usd").notna()].copy()
    if c.empty or not cluster:
        return c
    return (c.groupby(["symbol", "eff_date"], as_index=False)
            .agg(m_gross=("m_gross", "first"), m_fill=("m_fill", "mean"),
                 pnl_usd=("pnl_usd", "sum"), fills=("news_id", "size"),
                 sent=("sent", "first"), pit_rank=("pit_rank", "first"),
                 beta=("beta", "first"), oc_spy=("oc_spy", "first")))


def counting_report(df: Optional[pd.DataFrame] = None) -> Dict:
    """Both readings of "n", side by side, with the ratio between them.

    The seat's gate says "n ≥ 1,000 member events" without defining whether a
    member event is a row or a bet. At the moment those differ by 1.69x.
    """
    d = df if df is not None else load_events()
    rows = closed_events(d, cluster=False)
    bets = closed_events(d, cluster=True)
    if rows.empty:
        return {"status": "no closed rows"}
    return {"status": "ok",
            "closed_rows": int(len(rows)),
            "distinct_news_id": int(rows["news_id"].nunique()),
            "distinct_bets": int(len(bets)),
            "trading_days": int(rows["eff_date"].nunique()),
            "overcount_ratio": float(len(rows) / max(len(bets), 1)),
            "gate_progress_by_bets": float(len(bets) / GATE_N),
            "gate_progress_by_rows": float(len(rows) / GATE_N),
            "bets_to_gate": int(GATE_N - len(bets)),
            "bets_to_ws4_ws6": int(WS4_WS6_N - len(bets))}


def posterior(df: Optional[pd.DataFrame] = None, column: str = "m_gross",
              prior: str = "skeptical") -> Dict:
    """Beta-Binomial on the drift-sign hit rate and Student-t on M (in pp).

    Reported against the seat's own gate level (+0.05pp) rather than against
    zero, because +0.05pp is the number the seat has to clear to graduate.
    At this n the honest headline is that **the prior still dominates**: the
    posterior hit rate is pulled most of the way back to 50% from whatever the
    sample happens to show.
    """
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from backtrader_framework.optimization.bayesian_edge import (  # noqa: PLC0415
        BayesianEdgeEstimator)

    bets = closed_events(df, cluster=True)
    if bets.empty or column not in bets.columns:
        return {"status": "no closed bets"}
    v = pd.to_numeric(bets[column], errors="coerce").dropna() * 100.0   # → pp
    if len(v) < 2:
        return {"status": f"only {len(v)} bets"}
    est = BayesianEdgeEstimator().fit(v.to_numpy(), prior=prior)
    s = est.summary(threshold=GATE_M_PP)
    mr, wr = s["mean_r"], s["win_rate"]
    return {"status": "ok", "column": column, "n_bets": int(len(v)),
            "raw_hits": int((v > 0).sum()), "raw_hit_rate": float((v > 0).mean()),
            "raw_mean_pp": float(v.mean()), "raw_median_pp": float(v.median()),
            "post_mean_pp": float(mr["posterior_mean"]),
            "ci_lo_pp": float(mr["credible_interval_95"][0]),
            "ci_hi_pp": float(mr["credible_interval_95"][1]),
            "p_above_zero": float(mr["p_positive"]),
            "p_above_gate": float(mr["p_above_threshold"]),
            "post_hit_rate": float(wr["posterior_mean"]),
            "hit_ci": (float(wr["credible_interval_95"][0]),
                       float(wr["credible_interval_95"][1])),
            "p_hit_above_50": float(wr["p_above_50"]),
            "prior_dominates": bool(len(v) < WS4_WS6_N)}


def vol_band_atlas(df: Optional[pd.DataFrame] = None) -> Dict:
    """The VIX-band read the plan asks for — and why it cannot be produced yet.

    Two independent blockers, both stated rather than worked around:

    1. **No VIX series.** `^VIX` is not in the local store (only the VIX ETFs
       VIXY/VIXM/SVIX/UVIX, which track rolling futures and carry a structural
       decay that makes their *level percentile* meaningless), and yfinance is
       rate-limited — the same `YFRateLimitError` that blocked the NQ refresh.
       Substituting an ETF here would produce a band that looks like a fear
       gauge and is really a roll-yield gauge.
    2. **Four trading days.** Even with a perfect VIX series, the closed sample
       spans four distinct `eff_date`s, so a percentile band would have at most
       four observations to place — and every event on a given day shares that
       day's band, so the effective n per band is 1.

    Returned as a refusal with its reasons, not as an empty table that reads
    like a null result.
    """
    bets = closed_events(df, cluster=True)
    days = int(bets["eff_date"].nunique()) if not bets.empty else 0
    return {"status": "not askable",
            "trading_days": days,
            "reasons": [
                "no ^VIX series locally; yfinance rate-limited (YFRateLimitError)",
                "VIX ETFs track rolling futures — their level percentile is a "
                "roll-yield gauge, not a fear gauge",
                f"only {days} distinct trading day(s): every event on a day "
                "shares its band, so effective n per band is 1"]}


def readiness(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Distance to every gate that applies to this seat, in BETS."""
    cr = counting_report(df)
    if cr.get("status") != "ok":
        return pd.DataFrame()
    n = cr["distinct_bets"]
    rows = [
        {"gate": "seat review gate (frozen)", "needs": GATE_N, "have": n,
         "to_go": GATE_N - n, "pct": n / GATE_N},
        {"gate": "WS4 / WS6 (plan floor)", "needs": WS4_WS6_N, "have": n,
         "to_go": WS4_WS6_N - n, "pct": n / WS4_WS6_N},
    ]
    return pd.DataFrame(rows)


def equities_block(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The spine row shape for this seat: one row per BET, causal columns only.

    `sent`, `pit_rank` and `beta` are known before the open; `oc_spy` is the
    session's own SPY move and is therefore **contemporaneous, not causal** —
    it is kept because the drift measure is computed net of it, and flagged so
    it is never read as a pre-trade feature.
    """
    bets = closed_events(df, cluster=True)
    if bets.empty:
        return bets
    out = bets.copy()
    out["dow"] = pd.to_datetime(out["eff_date"], errors="coerce").dt.dayofweek
    out = out.rename(columns={"sent": "pre__sent", "pit_rank": "pre__pit_rank",
                              "beta": "pre__beta", "dow": "pre__dow",
                              "oc_spy": "post__oc_spy"})
    return out
