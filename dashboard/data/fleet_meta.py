"""WS5 — is there a better allocation rule than the conductor's, and can we test one?

The plan asked for a meta-strategy classifier: features = the causal market
state, label = the fleet family with the best forward R, then dynamic allocation
against three baselines. **The first thing this module does is refuse the
classifier**, and it says why in numbers rather than in prose.

The fleet's Tier-1/2 books produce **131 distinct trading days, of which 82 have
two or more families active at all**. The engine's own walk-forward wants
`min_train_days=60`, which leaves **22 evaluable days** and **one** retrain, to
choose between **10** family labels — about **2.2 evaluated days per class**.
A random forest fitted on that will return an accuracy, and the accuracy will be
noise wearing a decimal point. That is precisely the failure the 2026-09-03
engine review found in this same file: a walk-forward scoring **62% "accuracy"
on data with no signal in it**, because overlapping labels let the model read
returns it was about to be graded on.

So WS5 delivers the two things the sample CAN support:

1. **The allocation comparison** — the plan's "only thing that matters" — which
   needs no model at all: the conductor's frozen rules against static
   equal-weight, against best-single-family hindsight, on identical fills.
2. **A regime x family heatmap**, labelled in-sample, that the conductor's
   operator can read directly.

Both feed `meta_conductor/PREREG_v2.md`: candidate rules fitted in-sample here,
frozen with a falsifiable bar, and read only on a forward record-only window.
Deriving a rule from history is allowed; *believing* it before the forward window
is not.

*Picture: you may study last season to write the team sheet. You may not also
declare yourself champion on the strength of the study.*
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]

# The conductor's frozen lane assignment (meta_conductor/PREREG.md, 2026-08-24).
# PATIENT = desk-demo, retest-demo · RAID = lrr-short, ofcs-demo, london-demo,
# lr-signal-shadow, lrr-paper · KNIFE = the four funded arms.
# Mapped to FAMILY here because WS5 allocates across families, not books. The
# Liquidity Raid entry is the one extension: the prereg names lr-signal-shadow
# (Tier 3) rather than the Tier-1/2 LR seats, and it is flagged so a reader does
# not mistake it for something the conductor itself gates today.
LANES = {"Desk": "patient", "Retest": "patient",
         "LRR": "raid", "OFCS": "raid", "London Raid": "raid",
         "Liquidity Raid": "raid*",
         "Knife": "knife"}
UNGATED = ("Depth", "Ferryman", "SMC")
PATIENT_RV_FRAC = 0.85
KNIFE_DAILY_CAP = 8


# ══════════════════════════════════════════════════════════════════════════════
#  The daily panel
# ══════════════════════════════════════════════════════════════════════════════
def extreme_r_report(fills: pd.DataFrame, max_abs_r: float = 5.0) -> pd.DataFrame:
    """Rows whose |R| is too large to be a trade outcome.

    Measured on this fleet: the 0.1st percentile of Tier-1/2 R is −2.26 and the
    maximum is +2.61, so anything past ±5 is a broken record, not a bad trade.
    Two such rows exist in 5,139 (`lr-paper-bch` at **−56.9R**, `retest-demo` at
    −6.2R) and between them they move the fleet's mean R per day from −0.0014 to
    −0.2187. Reported, never silently dropped: a number that survives only
    because a row was quietly deleted is not a better number.
    """
    if fills.empty or "r" not in fills.columns:
        return pd.DataFrame()
    ext = fills[fills["r"].abs() > max_abs_r]
    cols = [c for c in ("book", "family", "fill_ts", "r") if c in ext.columns]
    return ext[cols].sort_values("r").reset_index(drop=True)


def daily_family_panel(tiers: Sequence[int] = (1, 2),
                       fills: Optional[pd.DataFrame] = None,
                       max_abs_r: Optional[float] = None) -> pd.DataFrame:
    """One row per (day, family): R per BET, and how many bets stood behind it.

    Per bet, not per row — the `fleet_edge` clustering haircut, so a family that
    re-records one resting setup does not out-vote a family that took one clean
    trade.

    `max_abs_r` excludes implausible records (see `extreme_r_report`). It
    defaults to None — the raw tape — because the honest default is to show what
    was recorded and let the sensitivity be stated, not to clean quietly.
    """
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.vol_atlas import fleet_fills_with_dvol                  # noqa: PLC0415

    f = fills if fills is not None else fleet_fills_with_dvol(tiers=tuple(tiers))
    if f.empty:
        return pd.DataFrame()
    f = f.dropna(subset=["day", "r"])
    if max_abs_r is not None:
        dropped = int((f["r"].abs() > max_abs_r).sum())
        if dropped:
            logger.warning("daily_family_panel: excluding %d row(s) with |R| > %s",
                           dropped, max_abs_r)
        f = f[f["r"].abs() <= max_abs_r]
    bets = f.groupby(["day", "family", "cluster"])["r"].mean().reset_index()
    daily = (bets.groupby(["day", "family"])
             .agg(r=("r", "mean"), bets=("r", "size")).reset_index())
    return daily.sort_values(["day", "family"]).reset_index(drop=True)


def feasibility(daily: pd.DataFrame, min_train_days: int = 60,
                retrain_every: int = 20) -> Dict:
    """The arithmetic that decides whether a classifier is even askable.

    Reported as a first-class output, not a footnote: a reader who sees only an
    accuracy has no way to know it rested on two days per class.
    """
    if daily.empty:
        return {"status": "no data"}
    per_day = daily.groupby("day")["family"].nunique()
    usable = int((per_day >= 2).sum())
    classes = int(daily["family"].nunique())
    evaluable = max(usable - min_train_days, 0)
    return {"status": "ok",
            "days_total": int(daily["day"].nunique()),
            "days_multi_family": usable,
            "classes": classes,
            "min_train_days": min_train_days,
            "evaluable_days": evaluable,
            "retrains": evaluable // max(retrain_every, 1),
            "evaluated_days_per_class": (evaluable / classes) if classes else float("nan"),
            "classifier_supportable": bool(evaluable >= 10 * classes)}


# ══════════════════════════════════════════════════════════════════════════════
#  The allocation comparison — the only thing that matters, and it needs no model
# ══════════════════════════════════════════════════════════════════════════════
def _state_by_day(daily: pd.DataFrame) -> pd.DataFrame:
    """The conductor's own state variables, per day, from the WS3 feature frame.

    Read at the day's START (the last bar closed before 00:00 UTC), so a rule
    evaluated for day D uses only what was knowable when D opened.
    """
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.regime_hmm import btc_features                          # noqa: PLC0415

    feat = btc_features()
    if feat.empty or daily.empty:
        return pd.DataFrame()
    days = pd.DatetimeIndex(sorted(daily["day"].unique()))
    fe = feat.dropna(subset=["close_ts"]).sort_values("close_ts")
    _ns = lambda x: pd.DatetimeIndex(x).tz_convert("UTC").tz_localize(None).astype("int64").to_numpy()  # noqa: E731
    idx = np.searchsorted(_ns(fe["close_ts"]), _ns(days), side="right") - 1
    ok = idx >= 0
    out = pd.DataFrame({"day": days})
    for c in ("rv24", "rv24_med30", "conductor_quiet"):
        v = fe[c].to_numpy()
        out[c] = np.where(ok, v[np.clip(idx, 0, None)], None)
    close = fe["close"].to_numpy()
    out["ret24"] = np.where(ok, close[np.clip(idx, 0, None)], np.nan)
    out["rv24"] = pd.to_numeric(out["rv24"], errors="coerce")
    out["rv24_med30"] = pd.to_numeric(out["rv24_med30"], errors="coerce")
    return out


def allocation_comparison(daily: Optional[pd.DataFrame] = None,
                          hmm_states: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """R per day under four allocations, on the identical set of fills.

    * **equal weight** — hold every family that traded that day. The honest
      default: no view at all.
    * **conductor** — the frozen PATIENT rule (`rv24 <= 0.85 x med30`) applied to
      the patient lane; every other lane held as it is today. This is the rule
      that is actually deployed, so it is the bar.
    * **hmm calm** — hold only in WS3's calmest filtered state. Included because
      WS3 produced the state path; WS3's own kill bar already failed, so this is
      a comparison arm, not a candidate.
    * **hindsight best family** — the single family with the best total R over
      the whole window, held every day. Not achievable; it is the ceiling that
      says how much an allocator could possibly have won.

    Every arm earns the SAME per-family daily R; they differ only in which
    families they were holding.
    """
    daily = daily if daily is not None else daily_family_panel()
    if daily.empty:
        return pd.DataFrame()
    st = _state_by_day(daily)
    d = daily.merge(st, on="day", how="left")

    if hmm_states is not None and not hmm_states.empty:
        h = hmm_states.copy()
        h["day"] = pd.to_datetime(h["close_ts"], utc=True).dt.floor("D")
        calm = (h.groupby("day")["state"].agg(lambda s: int(s.mode().iloc[0]))
                .rename("hmm_state").reset_index())
        d = d.merge(calm, on="day", how="left")
    else:
        d["hmm_state"] = np.nan

    best_family = (daily.groupby("family")["r"].sum().idxmax()
                   if not daily.empty else None)
    d["lane"] = d["family"].map(LANES).fillna("ungated")

    rows = []
    for day, sub in d.groupby("day"):
        quiet = sub["conductor_quiet"].iloc[0]
        conductor_hold = sub[(sub["lane"] != "patient") | (sub["conductor_quiet"] == True)]  # noqa: E712
        hmm_hold = sub[sub["hmm_state"] == 0] if sub["hmm_state"].notna().any() else sub.iloc[0:0]
        rows.append({
            "day": day,
            "n_families": int(len(sub)),
            "equal_weight": float(sub["r"].mean()),
            "conductor": (float(conductor_hold["r"].mean())
                          if len(conductor_hold) else 0.0),
            "conductor_n": int(len(conductor_hold)),
            "hmm_calm": (float(hmm_hold["r"].mean()) if len(hmm_hold) else 0.0),
            "hmm_n": int(len(hmm_hold)),
            "hindsight_best": (float(sub.loc[sub["family"] == best_family, "r"].mean())
                               if (sub["family"] == best_family).any() else 0.0),
            "quiet": quiet,
        })
    out = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    out.attrs["best_family"] = best_family
    return out


def summarise_allocations(cmp_df: pd.DataFrame, n_boot: int = 2000,
                          seed: int = 0) -> pd.DataFrame:
    """Mean R per day for each arm, with a day-block bootstrap interval.

    Days are the unit and days are resampled whole, with multiplicity — the
    correction WS3 needed after a boolean mask silently flattened a day drawn
    twice into a day drawn once.
    """
    if cmp_df.empty:
        return pd.DataFrame()
    arms = [c for c in ("equal_weight", "conductor", "hmm_calm", "hindsight_best")
            if c in cmp_df.columns]
    n = len(cmp_df)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    rows = []
    for a in arms:
        v = cmp_df[a].to_numpy(dtype=float)
        boots = v[draws].mean(axis=1)
        rows.append({"arm": a, "days": n, "mean_r_per_day": float(np.nanmean(v)),
                     "total_r": float(np.nansum(v)),
                     "ci_lo": float(np.percentile(boots, 5)),
                     "ci_hi": float(np.percentile(boots, 95)),
                     "t": float(_t(v))})
    out = pd.DataFrame(rows)
    base = out.loc[out["arm"] == "conductor", "mean_r_per_day"]
    if len(base):
        out["vs_conductor"] = out["mean_r_per_day"] - float(base.iloc[0])
    return out


def paired_vs_conductor(cmp_df: pd.DataFrame, arm: str = "equal_weight",
                        n_boot: int = 2000, seed: int = 0) -> Dict:
    """The PAIRED difference against the conductor, day by day.

    Comparing two independent means throws away the fact that both arms traded
    the same days; the paired difference is what a switch would actually have
    bought, and its interval is far tighter for the same data.
    """
    if cmp_df.empty or arm not in cmp_df.columns:
        return {"status": "n/a"}
    diff = (cmp_df[arm] - cmp_df["conductor"]).to_numpy(dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 10:
        return {"status": "too few days"}
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return {"status": "ok", "arm": arm, "days": int(len(diff)),
            "mean_diff_r_per_day": float(diff.mean()),
            "ci_lo": float(np.percentile(boots, 5)),
            "ci_hi": float(np.percentile(boots, 95)),
            "t": float(_t(diff))}


def _t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


# ══════════════════════════════════════════════════════════════════════════════
#  The heatmap the operator can read — IN SAMPLE, and labelled so
# ══════════════════════════════════════════════════════════════════════════════
def regime_family_heatmap(daily: Optional[pd.DataFrame] = None,
                          min_days: int = 5) -> pd.DataFrame:
    """Mean R per day, per (family x conductor state).

    **In-sample and descriptive.** It is the table the conductor's operator can
    read to see where each family's days actually fell; it is not a fitted rule
    and every cell is a hindsight average. A cell backed by fewer than
    `min_days` days is returned as NaN rather than as a number nobody should use.
    """
    daily = daily if daily is not None else daily_family_panel()
    if daily.empty:
        return pd.DataFrame()
    d = daily.merge(_state_by_day(daily), on="day", how="left")
    d["state"] = np.where(d["conductor_quiet"] == True, "quiet",            # noqa: E712
                          np.where(d["conductor_quiet"] == False, "busy", None))  # noqa: E712
    g = (d.dropna(subset=["state"]).groupby(["family", "state"])
         .agg(mean_r=("r", "mean"), median_r=("r", "median"),
              days=("day", "nunique")).reset_index())
    g.loc[g["days"] < min_days, ["mean_r", "median_r"]] = np.nan
    out = g.pivot(index="family", columns="state", values=["mean_r", "median_r", "days"])
    # The MEDIAN column is not decoration: `Liquidity Raid` reads a mean of
    # -2.668 in quiet against a median of -0.004, because ONE row carries -56.9R.
    # A heatmap showing only means invites exactly that misreading.
    return out
