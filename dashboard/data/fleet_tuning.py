"""WS6 — tune the instruments, not the seats.

The kill bar is frozen in ``WS6_TUNING_PREREG.md``: a tuned parameter ships only
if the holdout metric is within one standard error of the search metric, both
halves of the holdout agree, and it beats the current default on the holdout at
all. Otherwise the default stands.

**What may not be tuned is the load-bearing part of this file.** Live seat entry
and exit parameters are off the table: the lattice and gate scans are closed, the
one surviving exit change already has its own shadow-first prereg, and a sampler
pointed at a seat's own history is the WFO overfit trap with better manners.
``FORBIDDEN_TARGETS`` names them and a test asserts this module cannot reach
them.

``bayesian_tuner.tune_classifier`` wants **optuna, which is not installed here**,
so the classifier arm runs a seeded random search over the same space. That
substitution costs nothing the kill bar cares about: TPE finds a good
configuration in fewer trials, it does not make an overfit configuration
generalise. The gap between search and holdout is a property of the data and the
budget, not of the sampler.

*Picture: a tailor may adjust the tape measure. He may not adjust the customer.*
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]

# Named so the prohibition is greppable, not merely intended.
FORBIDDEN_TARGETS = (
    "seat entry parameters", "seat exit parameters", "stop distance",
    "take profit", "risk percent", "position size", "ladder multiplier",
)


# ══════════════════════════════════════════════════════════════════════════════
#  The kill bar, applied to any search
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TuningVerdict:
    winner: Optional[dict] = None
    search_metric: float = float("nan")
    search_se: float = float("nan")
    holdout_metric: float = float("nan")
    default_holdout: float = float("nan")
    halves: Tuple[float, float] = (float("nan"), float("nan"))
    within_one_se: Optional[bool] = None
    halves_agree: Optional[bool] = None
    beats_default: Optional[bool] = None
    ships: bool = False
    reasons: List[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


def apply_kill_bar(winner: dict, search_metric: float, search_se: float,
                   holdout_metric: float, default_holdout: float,
                   halves: Tuple[float, float]) -> TuningVerdict:
    """The three frozen conditions. No partial credit.

    The one-SE test is the one that does the work: a search over N
    configurations reports the MAXIMUM of N noisy estimates, which is biased
    upward by construction. Requiring the frozen holdout to land within one
    standard error of that maximum is what separates "found a better setting"
    from "found the luckiest fold".
    """
    v = TuningVerdict(winner=winner, search_metric=search_metric,
                      search_se=search_se, holdout_metric=holdout_metric,
                      default_holdout=default_holdout, halves=halves)
    v.within_one_se = bool(abs(holdout_metric - search_metric) <= search_se) \
        if search_se == search_se else None
    lo, hi = halves
    v.halves_agree = bool(np.sign(lo - default_holdout) == np.sign(hi - default_holdout)) \
        if (lo == lo and hi == hi) else None
    v.beats_default = bool(holdout_metric > default_holdout)

    if not v.within_one_se:
        v.reasons.append(
            f"holdout {holdout_metric:.4f} is {abs(holdout_metric - search_metric):.4f} "
            f"from the search's {search_metric:.4f} — more than one SE ({search_se:.4f})")
    if not v.halves_agree:
        v.reasons.append(f"the holdout halves disagree ({lo:.4f} / {hi:.4f})")
    if not v.beats_default:
        v.reasons.append(
            f"holdout {holdout_metric:.4f} does not beat the default {default_holdout:.4f}")
    v.ships = not v.reasons
    return v


# ══════════════════════════════════════════════════════════════════════════════
#  (a) the HMM — scored by HELD-OUT log-likelihood per bar
# ══════════════════════════════════════════════════════════════════════════════
HMM_DEFAULT = {"n_states": 2, "train_days": 90, "step_days": 20, "features": "primary"}


def _hmm_feature_sets() -> Dict[str, Tuple[str, ...]]:
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.regime_hmm import PRIMARY_FEATURES                       # noqa: PLC0415
    return {
        "primary": PRIMARY_FEATURES,
        "vol_only": ("rv24", "vol_of_vol"),
        "no_dvol": ("rv24", "abs_ret24", "vol_of_vol"),
        "rv_dvol": ("rv24", "d_dvol24"),
    }


def hmm_cv_loglik(feat: pd.DataFrame, n_states: int, features: Sequence[str],
                  train_days: int, n_folds: int = 5,
                  test_days: int = 30, embargo_days: int = 2) -> Dict:
    """Held-out log-likelihood per bar, over `n_folds` sequential blocks.

    Per BAR, not per fold: configurations with different windows see different
    numbers of held-out bars, and a total would reward whichever happened to be
    scored on more of them. An embargo sits between train and test so the
    trailing rows of the fitting window cannot bleed into the block it is graded
    on.
    """
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from backtrader_framework.optimization.hmm_regime import GaussianHMM   # noqa: PLC0415

    cols = list(features)
    d = feat.dropna(subset=cols).reset_index(drop=True)
    bpd = 96                                          # 15m bars per day
    tr_n, te_n, emb = train_days * bpd, test_days * bpd, embargo_days * bpd
    need = tr_n + emb + te_n
    if len(d) < need:
        return {"status": f"need {need} bars, have {len(d)}"}

    X = d[cols].to_numpy(dtype=float)
    vol_ix = cols.index("rv24") if "rv24" in cols else 0
    starts = np.linspace(0, len(X) - need, n_folds).astype(int)
    scores = []
    for s in starts:
        tr = X[s:s + tr_n]
        te = X[s + tr_n + emb: s + tr_n + emb + te_n]
        mu, sd = tr.mean(axis=0), tr.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        try:
            m = GaussianHMM(n_states=n_states, vol_feature_index=vol_ix)
            m.fit((tr - mu) / sd)
            ll = m.log_likelihood((te - mu) / sd)
        except Exception as e:                                          # noqa: BLE001
            logger.warning("hmm fold failed: %s", e)
            continue
        if np.isfinite(ll):
            scores.append(ll / len(te))
    if not scores:
        return {"status": "no usable folds"}
    return {"status": "ok", "loglik_per_bar": float(np.mean(scores)),
            "se": float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1
            else float("nan"),
            "folds": len(scores), "fold_scores": [float(x) for x in scores]}


def tune_hmm(feat: Optional[pd.DataFrame] = None, n_folds: int = 5,
             holdout_frac: float = 0.25, test_days: int = 30) -> Dict:
    """Search HMM configurations on the first (1 - holdout_frac) of history and
    confirm the winner on a **frozen final holdout** it never saw.

    Comparing a 3-state model to a 2-state one on TRAINING likelihood answers
    "which has more parameters". Only held-out likelihood answers "which
    describes the tape".
    """
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.regime_hmm import btc_features                            # noqa: PLC0415

    feat = feat if feat is not None else btc_features()
    if feat.empty:
        return {"status": "no features"}
    cut = int(len(feat) * (1 - holdout_frac))
    search_df, hold_df = feat.iloc[:cut].copy(), feat.iloc[cut:].copy()
    sets = _hmm_feature_sets()

    rows = []
    for name, cols in sets.items():
        for K in (2, 3):
            for train_days in (60, 90, 120):
                r = hmm_cv_loglik(search_df, K, cols, train_days, n_folds=n_folds,
                                  test_days=test_days)
                if r.get("status") != "ok":
                    continue
                rows.append({"features": name, "n_states": K, "train_days": train_days,
                             "search_loglik": r["loglik_per_bar"], "se": r["se"],
                             "folds": r["folds"]})
    if not rows:
        return {"status": "no configuration scored"}
    tbl = pd.DataFrame(rows).sort_values("search_loglik", ascending=False).reset_index(drop=True)
    best = tbl.iloc[0].to_dict()

    # The holdout must be scored with the SAME fold geometry as the search.
    # Scoring the search at test_days=30 and the holdout at test_days=15 makes
    # the "gap" partly a measure of the configuration difference rather than of
    # overfitting — two metrics computed under different CV geometries are not
    # comparable, and the one-SE test compares them directly.
    def _hold(cfg, df, n_folds=3):
        r = hmm_cv_loglik(df, int(cfg["n_states"]), sets[cfg["features"]],
                          int(cfg["train_days"]), n_folds=n_folds, test_days=test_days)
        return r.get("loglik_per_bar", float("nan"))

    hold_all = _hold(best, hold_df)
    half = len(hold_df) // 2
    halves = (_hold(best, hold_df.iloc[:half], n_folds=2),
              _hold(best, hold_df.iloc[half:], n_folds=2))
    default_hold = _hold({"n_states": HMM_DEFAULT["n_states"],
                          "features": HMM_DEFAULT["features"],
                          "train_days": HMM_DEFAULT["train_days"]}, hold_df)
    v = apply_kill_bar(best, best["search_loglik"], best["se"], hold_all,
                       default_hold, halves)
    return {"status": "ok", "table": tbl, "verdict": v,
            "default": HMM_DEFAULT, "holdout_bars": int(len(hold_df))}


# ══════════════════════════════════════════════════════════════════════════════
#  (b) the classifier — a calibration of the tuner on data whose answer we know
# ══════════════════════════════════════════════════════════════════════════════
CLS_SPACE = {
    "max_depth": (2, 3, 4, 6, 8),
    "learning_rate": (0.02, 0.04, 0.06, 0.1, 0.2),
    "max_iter": (100, 200, 400),
    "l2_regularization": (0.0, 0.5, 1.0, 5.0),
}


def tune_classifier_gap(df: Optional[pd.DataFrame] = None,
                        n_trials: int = 25, holdout_frac: float = 0.3,
                        seed: int = 0) -> Dict:
    """Random-search the WS4 classifier, then confirm on a frozen holdout.

    WS4 already established that this panel is null (OOS AUC 0.5074, and the
    hour-conditional shuffle beat it). So the expected result is a LARGE
    search-vs-holdout gap: the search reports the maximum of many noisy
    estimates and the holdout does not confirm it. Measuring that gap on data
    whose answer is known is a calibration of the tuning procedure — the same
    move as WS4's POST-fill positive control, pointed at the tuner instead of
    the model.

    optuna is absent; this is a seeded random search over the same space, which
    the kill bar does not distinguish (see WS6_TUNING_PREREG.md §2).
    """
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.fleet_ml import (build_xy, cpcv_auc, load_knife_episodes,   # noqa: PLC0415
                               pre_columns)

    df = df if df is not None else load_knife_episodes()
    if df.empty:
        return {"status": "no episodes"}
    d = df.dropna(subset=["r_net"]).sort_values("fill_ts").reset_index(drop=True)
    cut = int(len(d) * (1 - holdout_frac))
    search_df, hold_df = d.iloc[:cut], d.iloc[cut:]
    cols = pre_columns(d)

    rng = np.random.default_rng(seed)
    trials = []
    for i in range(n_trials):
        cfg = {k: v[rng.integers(0, len(v))] for k, v in CLS_SPACE.items()}
        cfg = {k: (float(x) if isinstance(x, float) else int(x)) for k, x in cfg.items()}
        X, y, day = build_xy(search_df, cols)
        r = cpcv_auc(X, y, day, seed=seed + i, grid=[cfg])
        if r.get("status") == "ok":
            trials.append({**cfg, "search_auc": r["auc_oos_mean"],
                           "se": r["auc_oos_std"] / max(np.sqrt(r["n_folds"]), 1)})
    if not trials:
        return {"status": "no trial scored"}
    tbl = pd.DataFrame(trials).sort_values("search_auc", ascending=False).reset_index(drop=True)
    best = tbl.iloc[0].to_dict()
    # `DataFrame.iloc[0].to_dict()` upcasts EVERY numeric column to float64 —
    # a mixed int/float frame has no other common dtype — so `max_depth` comes
    # back as 2.0 and sklearn rejects it. Recast from the space's own types.
    _int = {"max_depth", "max_iter"}
    cfg = {k: (int(best[k]) if k in _int else float(best[k])) for k in CLS_SPACE}

    def _hold(c, part):
        X, y, day = build_xy(part, cols)
        r = cpcv_auc(X, y, day, seed=seed, grid=[c])
        return r.get("auc_oos_mean", float("nan"))

    hold_all = _hold(cfg, hold_df)
    half = len(hold_df) // 2
    halves = (_hold(cfg, hold_df.iloc[:half]), _hold(cfg, hold_df.iloc[half:]))
    default_hold = _hold({"max_depth": 3, "learning_rate": 0.06, "max_iter": 200,
                          "l2_regularization": 1.0}, hold_df)
    v = apply_kill_bar(cfg, best["search_auc"], best["se"], hold_all, default_hold, halves)
    return {"status": "ok", "table": tbl, "verdict": v,
            "n_trials": len(trials), "holdout_rows": int(len(hold_df)),
            "search_rows": int(len(search_df))}
