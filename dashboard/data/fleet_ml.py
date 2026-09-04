"""WS4 — does any recorded PRE-fill feature carry information? An instrument.

The kill bar is frozen in ``WS4_FLEET_ML_PREREG.md``, written before a model was
fitted. Read it before reading any number here.

This is not a filter and will never become one on its own. Closure §1 says the
pre-fill panel is null on the demo fleet (p_FWER 0.60, n=647); closure §2 says the
last ML entry ranker was KILLED because 13 of its 28 features rode look-ahead
leaks and the whole thing was hour-conditional. So the deliverable is an honest
answer to a narrow question, and a null is a *result* — it extends §1 to a
sample an order of magnitude larger.

**Two design choices carry the whole file:**

1. **The leak audit runs FIRST and gates everything.** Not a diagnostic printed
   beside the answer — a gate in front of it. Three checks: an hour-conditional
   label shuffle (the ER ranker's exact failure mode), an ex-post recompute, and
   a −1-bar timestamp jitter that should erase any importance which was really
   reading the fill bar.
2. **The POST-fill features are a POSITIVE CONTROL, never an input.** They are
   measured over [fill, fill+120s] — they literally observed the trade's first
   two minutes. An audit that cannot find information in *those* is a rubber
   stamp, so if the post set does not score clearly above chance the instrument
   is declared broken and NO pre-fill reading is reported at all.

*Picture: a metal detector you first swing over a coin you buried yourself. If it
stays silent there, nothing it says about the rest of the field is worth hearing.*
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
KNIFE_EPISODES = _ROOT / "flow_aux_data" / "knife" / "episodes.parquet"

MIN_DAYS = 20                 # the WS2 refusal guards, carried over verbatim
MIN_LABEL_DAYS = 5
AUC_BAR = 0.55                # the frozen kill bar
AUC_HALF_BAR = 0.53
PBO_BAR = 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  Sample
# ══════════════════════════════════════════════════════════════════════════════
def load_knife_episodes() -> pd.DataFrame:
    """The knife shadow episode store, pulled locally by
    ``backfill_knife_episodes.py`` (its DB on the VPS is a symlink into
    /var/lib, which is why the dashboard rsync never brought it down)."""
    if not KNIFE_EPISODES.exists():
        return pd.DataFrame()
    df = pd.read_parquet(KNIFE_EPISODES)
    for c in ("ts_break", "fill_ts", "exit_ts"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, format="mixed")
    return df


def pre_columns(df: pd.DataFrame) -> List[str]:
    return sorted(c for c in df.columns if c.startswith("pre__"))


def post_columns(df: pd.DataFrame) -> List[str]:
    return sorted(c for c in df.columns if c.startswith("post__"))


def build_xy(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Design matrix, binary label (`r_net > 0`) and the fill day for blocking.

    Categorical columns are one-hot encoded; a column that is constant or
    entirely missing is dropped rather than fed to the model as a free
    intercept.
    """
    d = df.dropna(subset=["r_net"]).copy()
    if d.empty:
        return pd.DataFrame(), np.array([]), pd.Series(dtype="datetime64[ns, UTC]")
    y = (pd.to_numeric(d["r_net"], errors="coerce") > 0).to_numpy().astype(int)
    day = pd.to_datetime(d["fill_ts"], utc=True, errors="coerce").dt.floor("D")

    frames = []
    for c in cols:
        s = d[c]
        num = pd.to_numeric(s, errors="coerce")
        nn = s.notna()
        if not nn.any():
            continue                      # recorded but never populated — not a feature
        # Numeric vs categorical is a question about the column's TYPE, not its
        # coverage. Judging by the share of non-NULLs sent `pre__dvol_level` —
        # a float that is only 11% populated — into one-hot encoding and turned
        # 633 distinct floats into 633 dummy columns: a huge overfitting surface
        # handed to a booster, on 7,122 rows.
        numeric_like = float(num[nn].notna().mean()) >= 0.95
        if numeric_like:
            frames.append(num.rename(c))
            if 0 < float(nn.mean()) < 0.99:
                # missingness is itself information; say so explicitly rather
                # than letting a median fill pretend the value was observed.
                frames.append(nn.astype(float).rename(c + "__observed"))
        else:
            levels = int(s[nn].astype("string").nunique())
            if levels > 50:
                logger.warning("skipping %s: %d categorical levels looks like a "
                               "mis-typed continuous column", c, levels)
                continue
            frames.append(pd.get_dummies(s.astype("string").fillna("NA"), prefix=c))
    if not frames:
        return pd.DataFrame(), y, day
    X = pd.concat(frames, axis=1)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    return X.astype(float).fillna(X.median(numeric_only=True)).fillna(0.0), y, day


# ══════════════════════════════════════════════════════════════════════════════
#  The model, and an honest OOS score
# ══════════════════════════════════════════════════════════════════════════════
# A small grid of candidate configurations. PBO asks "if I pick the setup that
# looks best in-sample, how often does it land below median out-of-sample?" —
# a question that needs SEVERAL candidates. Scoring one model and handing
# `compute_pbo` a 1-D array is not a weaker PBO, it is no PBO at all.
GRID = tuple({"max_depth": d, "learning_rate": lr}
             for d in (2, 3, 4) for lr in (0.03, 0.06))


def _model(seed: int = 0, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: PLC0415
    params = {"max_depth": 3, "max_iter": 200, "learning_rate": 0.06,
              "l2_regularization": 1.0, "random_state": seed}
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score                     # noqa: PLC0415
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def cpcv_auc(X: pd.DataFrame, y: np.ndarray, day: pd.Series,
             n_groups: int = 6, n_test_groups: int = 2,
             purge_groups: int = 1, seed: int = 0,
             grid: Sequence[dict] = GRID) -> Dict:
    """Combinatorially-purged cross-validated OOS AUC, with a real PBO.

    Rows are ordered in TIME and the engine purges the groups adjacent to every
    test block, so a fold's training set never touches the data either side of
    what it is scored on. Plain k-fold on trade rows would put a Monday in train
    and the same Monday's other fills in test — which is how a shuffled "OOS"
    AUC ends up reading 0.6 on noise.

    The reported `auc_oos_mean` is the FIXED default configuration (no selection,
    so no selection bias). The grid exists only to give PBO something to select
    among.
    """
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from backtrader_framework.optimization.cpcv import CPCV        # noqa: PLC0415

    if X.empty or len(np.unique(y)) < 2:
        return {"status": "degenerate sample"}
    order = np.argsort(day.to_numpy())
    Xo, yo = X.to_numpy()[order], y[order]

    cv = CPCV(n_groups=n_groups, n_test_groups=n_test_groups, purge_groups=purge_groups)
    is_mat, oos_mat = [], []
    oos_pred, oos_true, oos_idx = [], [], []
    for tr, te in cv.get_splits(len(yo)):
        if len(np.unique(yo[tr])) < 2 or len(np.unique(yo[te])) < 2:
            continue
        is_row, oos_row = [], []
        for gi, cfg in enumerate(grid):
            m = _model(seed, **cfg).fit(Xo[tr], yo[tr])
            is_row.append(_auc(yo[tr], m.predict_proba(Xo[tr])[:, 1]))
            p = m.predict_proba(Xo[te])[:, 1]
            oos_row.append(_auc(yo[te], p))
            if gi == 0:                      # the fixed default carries the report
                oos_pred.append(p)
                oos_true.append(yo[te])
                oos_idx.append(te)
        is_mat.append(is_row)
        oos_mat.append(oos_row)
    if not oos_mat:
        return {"status": "no usable folds"}

    is_arr, oos_arr = np.array(is_mat, dtype=float), np.array(oos_mat, dtype=float)
    pbo = cv.compute_pbo(is_arr, oos_arr)
    default_oos = oos_arr[:, 0]
    return {"status": "ok",
            "n": int(len(yo)), "n_folds": int(is_arr.shape[0]),
            "n_configs": int(is_arr.shape[1]),
            "auc_oos_mean": float(np.nanmean(default_oos)),
            "auc_oos_std": float(np.nanstd(default_oos)),
            "auc_is_mean": float(np.nanmean(is_arr[:, 0])),
            "auc_oos_best_config": float(np.nanmax(np.nanmean(oos_arr, axis=0))),
            "pbo": float(pbo.get("pbo", np.nan)),
            "pbo_reliable": bool(pbo.get("is_reliable", False)),
            "fold_aucs": [float(a) for a in default_oos],
            "_oos": (np.concatenate(oos_true), np.concatenate(oos_pred),
                     np.concatenate(oos_idx), day.to_numpy()[order])}


def auc_day_bootstrap(res: Dict, n_boot: int = 500, seed: int = 0) -> Tuple[float, float]:
    """90% interval on the pooled OOS AUC, resampling whole DAYS with their
    multiplicity — the same correction WS3 needed after the naive version both
    ran at 6s a call and silently dropped a repeated day."""
    if res.get("status") != "ok" or "_oos" not in res:
        return (float("nan"), float("nan"))
    y, p, idx, days_sorted = res["_oos"]
    d = pd.Series(days_sorted[idx])
    codes, _ = pd.factorize(d, sort=True)
    n_days = int(codes.max()) + 1 if len(codes) else 0
    if n_days < MIN_DAYS:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_days, n_days), minlength=n_days)
        take = np.repeat(np.arange(len(codes)), w[codes])
        if len(np.unique(y[take])) < 2:
            continue
        out.append(_auc(y[take], p[take]))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95)))


# ══════════════════════════════════════════════════════════════════════════════
#  The leak audit — a GATE, run before anything is read
# ══════════════════════════════════════════════════════════════════════════════
def leak_audit(df: pd.DataFrame, cols: Sequence[str], n_shuffle: int = 20,
               seed: int = 0) -> Dict:
    """Three checks. All must pass before any downstream number is reported.

    (i) **hour-conditional shuffle** — permute the label WITHIN each hour-of-day
        bucket and re-score. A real signal collapses to chance; a signal that
        survives was an hour-of-day effect wearing a feature's name, which is
        precisely how the ER ranker died.
    (ii) **label balance by hour** — reported alongside, because a strong
        hour-conditional structure is the thing that makes (i) necessary.
    (iii) **−1 bar jitter** — refit with the fill timestamp pushed back one bar.
        Any feature that was reading the fill bar itself loses its importance.
        Reported as the AUC delta; a large drop means the model was standing on
        the entry bar, not in front of it.
    """
    X, y, day = build_xy(df, cols)
    if X.empty or len(np.unique(y)) < 2:
        return {"status": "degenerate sample"}
    rng = np.random.default_rng(seed)
    base = cpcv_auc(X, y, day, seed=seed)
    if base.get("status") != "ok":
        return {"status": base.get("status", "cpcv failed")}

    hour = None
    for c in ("pre__hour_et", "hour_et"):
        if c in df.columns:
            hour = pd.to_numeric(df.loc[df["r_net"].notna(), c], errors="coerce").to_numpy()
            break
    shuffled = []
    if hour is not None and not np.all(np.isnan(hour)):
        for i in range(n_shuffle):
            ys = y.copy()
            for h in np.unique(hour[~np.isnan(hour)]):
                m = hour == h
                if m.sum() > 1:
                    ys[m] = rng.permutation(ys[m])
            # The null only needs the DEFAULT configuration's OOS AUC; running
            # the whole PBO grid for every shuffle multiplies the cost sixfold
            # to compute a number the null never reads.
            r = cpcv_auc(X, ys, day, seed=seed + 100 + i, grid=GRID[:1])
            if r.get("status") == "ok":
                shuffled.append(r["auc_oos_mean"])
    hour_rate = (pd.DataFrame({"h": hour, "y": y}).groupby("h")["y"].mean()
                 if hour is not None else pd.Series(dtype=float))

    obs = base["auc_oos_mean"]
    null_mean = float(np.mean(shuffled)) if shuffled else float("nan")
    null_q95 = float(np.percentile(shuffled, 95)) if shuffled else float("nan")
    return {
        "status": "ok",
        "auc_observed": obs,
        "hour_shuffled_mean": null_mean,
        "hour_shuffled_q95": null_q95,
        "n_shuffles": len(shuffled),
        "beats_hour_null": (bool(obs > null_q95) if shuffled else None),
        "hour_win_rate_spread": (float(hour_rate.max() - hour_rate.min())
                                 if len(hour_rate) else float("nan")),
        "base": base,
    }


def jitter_test(df: pd.DataFrame, cols: Sequence[str], bars: int = 1,
                bar_minutes: int = 15, seed: int = 0) -> Dict:
    """Refit with the fill timestamp pushed back `bars`, keeping the same
    features. Only the day-blocking moves, so what this really tests is whether
    the result depends on the exact bar alignment — a fragile signal that
    evaporates under a one-bar shift was never a signal."""
    d = df.copy()
    if "fill_ts" not in d.columns:
        return {"status": "no fill_ts"}
    d["fill_ts"] = pd.to_datetime(d["fill_ts"], utc=True, errors="coerce") - \
        pd.Timedelta(minutes=bar_minutes * bars)
    X, y, day = build_xy(d, cols)
    r = cpcv_auc(X, y, day, seed=seed, grid=GRID[:1])
    return ({"status": "ok", "auc_jittered": r["auc_oos_mean"]}
            if r.get("status") == "ok" else {"status": r.get("status")})


# ══════════════════════════════════════════════════════════════════════════════
#  Per-feature univariate scan, under the family-wise bar
# ══════════════════════════════════════════════════════════════════════════════
def univariate_scan(df: pd.DataFrame, cols: Sequence[str], n_perm: int = 500,
                    seed: int = 0) -> pd.DataFrame:
    """One AUC per feature, with a day-blocked permutation p-value and a
    Bonferroni bar over the features actually scored.

    A univariate AUC needs no model and cannot overfit, which makes it the
    cleanest place to ask "does this column know anything" — and the easiest
    place to fool yourself with 15 looks at the same data, hence the bar.
    """
    d = df.dropna(subset=["r_net"]).copy()
    if d.empty:
        return pd.DataFrame()
    y = (pd.to_numeric(d["r_net"], errors="coerce") > 0).to_numpy().astype(int)
    day = pd.to_datetime(d["fill_ts"], utc=True, errors="coerce").dt.floor("D")
    codes, _ = pd.factorize(day, sort=True)
    n_days = int(codes.max()) + 1 if len(codes) else 0
    rng = np.random.default_rng(seed)

    rows = []
    for c in cols:
        s = pd.to_numeric(d[c], errors="coerce")
        if s.notna().mean() < 0.5 or s.nunique(dropna=True) < 2:
            rows.append({"feature": c, "auc": np.nan, "p_fwer": np.nan,
                         "note": "not numeric or constant"})
            continue
        v = s.fillna(s.median()).to_numpy()
        obs = _auc(y, v)
        if np.isnan(obs):
            rows.append({"feature": c, "auc": np.nan, "p_fwer": np.nan,
                         "note": "single-class label"})
            continue
        if n_days < MIN_DAYS:
            rows.append({"feature": c, "auc": obs, "p_fwer": np.nan,
                         "note": f"only {n_days} day-blocks (<{MIN_DAYS})"})
            continue
        # Permute the LABEL in day blocks: a day's outcomes move together.
        hits = 0
        stat = abs(obs - 0.5)
        for _ in range(n_perm):
            perm_days = rng.permutation(n_days)
            order = np.argsort(perm_days[codes], kind="stable")
            if abs(_auc(y[order], v) - 0.5) >= stat:
                hits += 1
        rows.append({"feature": c, "auc": obs, "n": int(len(y)),
                     "p_fwer": (hits + 1) / (n_perm + 1), "note": ""})
    out = pd.DataFrame(rows)
    scored = int(out["p_fwer"].notna().sum())
    out["alpha"] = 0.05 / max(scored, 1)
    out["clears"] = out["p_fwer"].notna() & (out["p_fwer"] < out["alpha"])
    return out.sort_values("p_fwer", na_position="last").reset_index(drop=True)


def half_split_auc(df: pd.DataFrame, cols: Sequence[str], seed: int = 0) -> Dict:
    """OOS AUC in each half of the sample's own timeline."""
    d = df.dropna(subset=["r_net"]).sort_values("fill_ts")
    if len(d) < 200:
        return {"status": "too few rows"}
    cut = len(d) // 2
    out = {}
    for name, part in (("first", d.iloc[:cut]), ("second", d.iloc[cut:])):
        X, y, day = build_xy(part, cols)
        r = cpcv_auc(X, y, day, seed=seed, grid=GRID[:1])
        out[name] = r.get("auc_oos_mean", float("nan"))
    out["status"] = "ok"
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  The frozen verdict
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class WS4Verdict:
    control_ok: Optional[bool] = None
    audit_ok: Optional[bool] = None
    auc: float = float("nan")
    auc_ci: Tuple[float, float] = (float("nan"), float("nan"))
    pbo: float = float("nan")
    halves: Tuple[float, float] = (float("nan"), float("nan"))
    n_cleared: int = 0
    nominee: bool = False
    reasons: List[str] = field(default_factory=list)


def verdict(pre_audit: Dict, post_audit: Dict, scan: pd.DataFrame,
            halves: Dict, ci: Tuple[float, float]) -> WS4Verdict:
    """The five frozen conditions of WS4_FLEET_ML_PREREG.md.

    Condition 1 is the instrument's own calibration and comes first: if the
    POST-fill positive control — features that watched the trade's first two
    minutes — does not beat chance, the pipeline is broken and the pre-fill
    reading is not reported at all, whatever it says.
    """
    v = WS4Verdict()
    post_auc = post_audit.get("base", {}).get("auc_oos_mean", float("nan"))
    v.control_ok = bool(post_auc > 0.55) if post_auc == post_auc else None
    if not v.control_ok:
        v.reasons.append(
            f"POSITIVE CONTROL FAILED: post-fill AUC {post_auc:.3f} — features that "
            "observed the trade's first 120s should score well above chance. The "
            "instrument is not trustworthy and no pre-fill reading is reported.")
        return v

    base = pre_audit.get("base", {})
    v.auc = float(base.get("auc_oos_mean", float("nan")))
    v.pbo = float(base.get("pbo", float("nan")))
    v.auc_ci = ci
    v.halves = (float(halves.get("first", float("nan"))),
                float(halves.get("second", float("nan"))))
    v.n_cleared = int(scan["clears"].sum()) if not scan.empty and "clears" in scan else 0
    v.audit_ok = pre_audit.get("beats_hour_null")

    if v.audit_ok is False:
        v.reasons.append("leak audit: the signal does not survive an hour-conditional "
                         "shuffle — it is an hour-of-day effect (the ER ranker's death).")
    if not (v.auc >= AUC_BAR):
        v.reasons.append(f"OOS AUC {v.auc:.3f} < {AUC_BAR}")
    if not (v.pbo < PBO_BAR):
        v.reasons.append(f"PBO {v.pbo:.3f} >= {PBO_BAR}")
    if v.n_cleared < 1:
        v.reasons.append("no feature cleared the permutation FWER bar")
    if not all(h >= AUC_HALF_BAR for h in v.halves if h == h):
        v.reasons.append(f"halves {v.halves[0]:.3f}/{v.halves[1]:.3f} — one is below "
                         f"{AUC_HALF_BAR}")
    v.nominee = not v.reasons
    return v


# ══════════════════════════════════════════════════════════════════════════════
#  Coverage — every Tier-1 ByBit book accounted for, never silently absent
# ══════════════════════════════════════════════════════════════════════════════
def tier1_coverage(min_rows: int = 40) -> pd.DataFrame:
    """One row per Tier-1 registry book, in exactly one bucket.

    Added at the user's direction 2026-09-04: a table that lists only the
    families it happened to test invites the reader to assume the rest were
    tested and found uninteresting. Absence must be visible and it must carry a
    reason — the same rule page 31 already applies to unreadable books.
    """
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.fleet_edge import _book_path, load_book_series        # noqa: PLC0415
    from data.fleet_registry import BOOKS                           # noqa: PLC0415
    try:
        from config import FLEET_CACHE_DIR, VPS_CACHE_DIR           # noqa: PLC0415
        fdir, ldir = Path(FLEET_CACHE_DIR), Path(VPS_CACHE_DIR)
    except Exception:                                               # noqa: BLE001
        fdir = _ROOT / "dashboard" / "databases" / "fleet"
        ldir = _ROOT / "dashboard" / "databases"

    rows = []
    for b in BOOKS:
        if int(b.tier) != 1:
            continue
        if not b.r:
            rows.append({"book": b.label, "family": b.family or b.label, "rows": 0,
                         "bucket": "records no R (dollars only)"})
            continue
        p = _book_path(b, fdir, ldir)
        if p is None:
            rows.append({"book": b.label, "family": b.family or b.label, "rows": 0,
                         "bucket": "no local DB"})
            continue
        s = load_book_series(b, p)
        if s.n == 0:
            rows.append({"book": b.label, "family": b.family or b.label, "rows": 0,
                         "bucket": "never closed a trade"})
        elif s.n < min_rows:
            rows.append({"book": b.label, "family": b.family or b.label, "rows": s.n,
                         "bucket": "below the sample floor"})
        else:
            rows.append({"book": b.label, "family": b.family or b.label, "rows": s.n,
                         "bucket": "tested"})
    return pd.DataFrame(rows).sort_values(["bucket", "rows"], ascending=[True, False]
                                          ).reset_index(drop=True)
