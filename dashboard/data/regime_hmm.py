"""WS3 — a walk-forward HMM regime path over the fleet's tape, and its sequencing.

Everything runs on THIS machine: local 15-minute bars, the local hourly DVOL
parquet (WS2), and a locally cached 15-minute OI aggregate. No VPS compute, no
VPS cache.

The kill bar this feeds is frozen in ``WS3_HMM_REGIME_PREREG.md`` and was written
before a single model was fitted. Read it before reading any number here.

**The whole design is about one refusal: the state at bar t may not know bar t+1.**
Three separate mechanisms enforce it, because the failure that killed the ER
ranker (2026-08-17) was not a bad model — it was a good model fed the future:

1. **Filtered probabilities only.** ``forward_filter`` gives P(state_t | x_1..t).
   Viterbi and forward-backward smoothing both use later bars, and a "regime"
   that has seen tomorrow will separate anything you ask it to.
2. **Rolling refit.** Fit on a trailing 90 days, filter the next 20, step, repeat.
   The state at bar t comes from parameters estimated before bar t existed. The
   seam carries the previous window's last posterior instead of restarting from
   pi (the 2026-09-03 engine fix), so continuity costs no leak.
3. **A point-in-time fill join.** A fill takes the state of the last bar that
   CLOSED before its entry — never the bar it landed inside.

*Picture: weather regimes. The forecast may use yesterday's sky, never
tomorrow's — and "storm follows calm" is only worth money if the seats that
trade in calm actually lose when the storm arrives.*
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
TIER3_DIR = _ROOT / "flow_aux_data" / "vol_tier3"

BAR_MIN = 15
BARS_PER_DAY = (24 * 60) // BAR_MIN                 # 96
# The conductor's frozen PATIENT rule, copied here as the BASELINE the HMM has to
# beat. Kept literal (not imported) because meta_conductor/ is a deployed bot
# package and this module must never import live-bot code; `test_regime_hmm`
# asserts the constant still matches market_state.py.
PATIENT_RV_FRAC = 0.85
RV_MED_DAYS = 30

PRIMARY_FEATURES = ("rv24", "abs_ret24", "vol_of_vol", "d_dvol24")
EXTENDED_FEATURES = PRIMARY_FEATURES + ("d_oi24",)


# ══════════════════════════════════════════════════════════════════════════════
#  Features — every column readable from bars that have already closed
# ══════════════════════════════════════════════════════════════════════════════
def btc_features(symbol: str = "BTC", start: str = "2021-03-24",
                 with_oi: bool = False) -> pd.DataFrame:
    """The causal feature frame, indexed by the bar's CLOSE time.

    duckdb stores a kline at its OPEN; a 15-minute bar opened at 10:00 is only
    knowable at 10:15. Indexing by close makes every downstream join ("the last
    bar that had closed") a plain comparison rather than a place to be clever
    and get it wrong.
    """
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.vol_atlas import dvol_asof, dvol_features, load_bars   # noqa: PLC0415

    bars = load_bars(symbol, "15m", start)
    if bars.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"close_ts": bars["timestamp"] + pd.Timedelta(minutes=BAR_MIN),
                       "close": bars["close"].to_numpy()})
    lr = np.log(df["close"]).diff()
    df["rv24"] = lr.rolling(BARS_PER_DAY).std()
    df["abs_ret24"] = (df["close"] / df["close"].shift(BARS_PER_DAY) - 1.0).abs()
    # vol-of-vol: how unsettled the volatility itself is over the last day
    rv_short = lr.rolling(BARS_PER_DAY // 4).std()
    df["vol_of_vol"] = rv_short.rolling(BARS_PER_DAY).std()

    # DVOL, read at the bar's close through the WS2 point-in-time helper (which
    # already enforces "the hourly bar must have closed" and a staleness limit).
    dv = dvol_asof(dvol_features("BTC"), df["close_ts"])
    df["dvol"] = dv["dvol"].to_numpy()
    df["d_dvol24"] = df["dvol"] - df["dvol"].shift(BARS_PER_DAY)

    if with_oi:
        oi = load_oi_15m(f"{symbol}USDT")
        if not oi.empty:
            j = oi.set_index("close_ts")["oi"].reindex(df["close_ts"], method="ffill")
            df["oi"] = j.to_numpy()
            df["d_oi24"] = df["oi"] / df["oi"].shift(BARS_PER_DAY) - 1.0
        else:
            df["oi"] = np.nan
            df["d_oi24"] = np.nan

    # The conductor's baseline state, on the same clock as the HMM's.
    med = df["rv24"].rolling(RV_MED_DAYS * BARS_PER_DAY, min_periods=7 * BARS_PER_DAY).median()
    df["rv24_med30"] = med
    # object dtype on purpose: "unknown" is a third value, not False. Assigning
    # None into a bool column silently coerces it to False, which would hand the
    # conductor's baseline a decision it never made.
    quiet = (df["rv24"] <= PATIENT_RV_FRAC * med).astype(object)
    quiet[med.isna() | df["rv24"].isna()] = None
    df["conductor_quiet"] = quiet
    return df


def load_oi_15m(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Locally cached 15-minute open interest (aggregated on the VPS, pulled by
    ``backfill_vol_tier3.py --only oi``). Indexed by the bucket's CLOSE."""
    p = TIER3_DIR / "oi_15m.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df = df[df["symbol"] == symbol]
    if df.empty:
        return df
    out = pd.DataFrame({
        "close_ts": pd.to_datetime(df["bucket_ms"], unit="ms", utc=True)
                    + pd.Timedelta(minutes=BAR_MIN),
        "oi": pd.to_numeric(df["oi_mean"], errors="coerce")})
    return out.sort_values("close_ts").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  The walk-forward state path
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RegimePath:
    """One walk-forward run: the filtered state at every out-of-sample bar."""
    states: pd.DataFrame                     # close_ts, state, p_state, window
    models: List[dict] = field(default_factory=list)
    n_states: int = 2
    features: Tuple[str, ...] = PRIMARY_FEATURES
    train_days: int = 90
    step_days: int = 20

    @property
    def n_windows(self) -> int:
        return len(self.models)


def rolling_states(feat: pd.DataFrame, n_states: int = 2,
                   features: Sequence[str] = PRIMARY_FEATURES,
                   train_days: int = 90, step_days: int = 20,
                   seed: int = 42) -> RegimePath:
    """Fit on the trailing `train_days`, forward-filter the next `step_days`, step.

    Every out-of-sample bar's state comes from a model that never saw it. The
    seam carries the previous window's final posterior into the next filter, so
    the path is continuous without the filter ever looking forward.
    """
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from backtrader_framework.optimization.hmm_regime import GaussianHMM   # noqa: PLC0415

    cols = list(features)
    d = feat.dropna(subset=cols + ["close_ts"]).reset_index(drop=True)
    if d.empty:
        return RegimePath(states=pd.DataFrame(), n_states=n_states,
                          features=tuple(features), train_days=train_days,
                          step_days=step_days)
    train_bars = train_days * BARS_PER_DAY
    step_bars = step_days * BARS_PER_DAY
    if len(d) < train_bars + step_bars:
        return RegimePath(states=pd.DataFrame(), n_states=n_states,
                          features=tuple(features), train_days=train_days,
                          step_days=step_days)

    X_all = d[cols].to_numpy(dtype=float)
    out_rows: List[pd.DataFrame] = []
    models: List[dict] = []
    carry: Optional[np.ndarray] = None
    # `vol_feature_index` tells the engine which column orders the states; rv24
    # is column 0 here, not column 1 as the WFO caller assumes.
    vol_ix = cols.index("rv24") if "rv24" in cols else 0

    start = train_bars
    while start + 1 <= len(d):
        tr = X_all[start - train_bars:start]
        te = X_all[start:start + step_bars]
        if len(te) == 0:
            break
        mu, sd = tr.mean(axis=0), tr.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)                 # IS-only standardisation
        try:
            hmm = GaussianHMM(n_states=n_states, vol_feature_index=vol_ix)
            hmm.fit((tr - mu) / sd)
            probs = hmm.forward_filter((te - mu) / sd, init_state_probs=carry)
        except Exception as e:                             # noqa: BLE001
            logger.warning("HMM window at %s failed: %s", d["close_ts"].iloc[start], e)
            start += step_bars
            carry = None
            continue
        carry = probs[-1]
        seg = d.iloc[start:start + len(probs)][["close_ts"]].copy()
        seg["state"] = probs.argmax(axis=1)
        seg["p_state"] = probs.max(axis=1)
        seg["window"] = len(models)
        out_rows.append(seg)
        models.append({"window": len(models),
                       "fit_from": d["close_ts"].iloc[start - train_bars],
                       "fit_to": d["close_ts"].iloc[start - 1],
                       "applied_from": seg["close_ts"].iloc[0],
                       "applied_to": seg["close_ts"].iloc[-1],
                       **hmm.transition_summary()})
        start += step_bars

    states = (pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame())
    return RegimePath(states=states, models=models, n_states=n_states,
                      features=tuple(features), train_days=train_days,
                      step_days=step_days)


# ══════════════════════════════════════════════════════════════════════════════
#  Sequencing — the atlas half, reported whatever the kill bar says
# ══════════════════════════════════════════════════════════════════════════════
def empirical_transitions(states: pd.DataFrame, n_states: int,
                          n_boot: int = 500, seed: int = 0) -> Dict:
    """P(A→B) from the realised path, with a DAY-BLOCK bootstrap interval.

    Resampling bar by bar would treat a state that persists for hours as dozens
    of independent observations and hand back an interval far too tight. Days
    are resampled whole, **with their multiplicity**: a day drawn twice counts
    twice, which a boolean `isin` mask would silently flatten to once.

    Implementation note, because the naive version is unusably slow rather than
    merely inelegant: `np.isin` against a datetime64 array costs ~6 s per call
    at this length, so 500 draws is ~50 minutes. Days are factorised to integer
    codes ONCE and each draw becomes a `bincount` — microseconds, and it is the
    change that makes the multiplicity weighting natural instead of awkward.
    """
    if states.empty:
        return {}
    s = states.sort_values("close_ts").reset_index(drop=True)
    day = s["close_ts"].dt.floor("D")
    codes, _uniq = pd.factorize(day, sort=True)
    n_days = int(codes.max()) + 1 if len(codes) else 0

    st = s["state"].to_numpy().astype(int)
    a, b = st[:-1], st[1:]
    # A gap between days is not a transition the tape actually made.
    same_day = codes[:-1] == codes[1:]
    a, b, tcodes = a[same_day], b[same_day], codes[:-1][same_day]
    K = int(n_states)

    def _mat(weights: np.ndarray) -> np.ndarray:
        flat = np.bincount(a * K + b, weights=weights, minlength=K * K).reshape(K, K)
        tot = flat.sum(axis=1, keepdims=True)
        return np.divide(flat, tot, out=np.zeros_like(flat), where=tot > 0)

    ones = np.ones(len(a), dtype=float)
    obs = _mat(ones)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_days, size=n_days)          # days, with replacement
        w = np.bincount(pick, minlength=n_days).astype(float)
        wt = w[tcodes]
        if wt.sum() < 50:
            continue
        boots.append(_mat(wt))
    lo = np.percentile(boots, 5, axis=0) if boots else np.full_like(obs, np.nan)
    hi = np.percentile(boots, 95, axis=0) if boots else np.full_like(obs, np.nan)
    dwell = 1.0 / np.maximum(1.0 - np.diag(obs), 1e-9)
    return {"transition": obs, "ci_lo": lo, "ci_hi": hi,
            "dwell_bars": dwell, "dwell_hours": dwell * BAR_MIN / 60.0,
            "n_transitions": int(len(a)), "n_days": n_days,
            "occupancy": (s["state"].value_counts(normalize=True)
                          .reindex(range(K)).fillna(0.0).to_numpy())}


def n_step_ahead(trans: np.ndarray, horizons: Sequence[int] = (1, 2, 4, 8)) -> pd.DataFrame:
    """P(state at t+h | state at t) by repeated multiplication of P(A→B)."""
    rows = []
    for h in horizons:
        m = np.linalg.matrix_power(np.asarray(trans, dtype=float), int(h))
        for i in range(m.shape[0]):
            rows.append({"from_state": i, "h_bars": int(h),
                         **{f"to_{j}": float(m[i, j]) for j in range(m.shape[1])}})
    return pd.DataFrame(rows)


def next_switch_labels(states: pd.DataFrame) -> pd.DataFrame:
    """For every bar, which state the path switched INTO next, and when.

    This is the ONLY forward-looking column in the module and it is never a
    feature — it labels an outcome table ("entries taken in calm, split by
    whether calm broke into turbulence or into something else"). Using it as an
    input would be exactly the leak the rest of the file exists to prevent, so
    it is named `next_switch_*` and kept out of every feature list.
    """
    if states.empty:
        return states
    s = states.sort_values("close_ts").reset_index(drop=True).copy()
    st = s["state"].to_numpy()
    nxt = np.full(len(st), -1, dtype=int)
    nxt_ts = np.full(len(st), np.datetime64("NaT"), dtype="datetime64[ns]")
    ts = s["close_ts"].dt.tz_localize(None).to_numpy()
    run_end = len(st)
    future_state, future_ts = -1, np.datetime64("NaT")
    for i in range(len(st) - 1, -1, -1):
        if i + 1 < len(st) and st[i + 1] != st[i]:
            future_state, future_ts = st[i + 1], ts[i + 1]
        nxt[i] = future_state
        nxt_ts[i] = future_ts
    s["next_switch_state"] = nxt
    s["next_switch_ts"] = pd.to_datetime(nxt_ts, utc=True)
    s["bars_to_switch"] = ((s["next_switch_ts"] - s["close_ts"]).dt.total_seconds()
                           / (BAR_MIN * 60))
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  The fleet join, and the frozen kill bar
# ══════════════════════════════════════════════════════════════════════════════
def fills_with_state(path: RegimePath, feat: Optional[pd.DataFrame] = None,
                     fills: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Tag every fleet fill with the filtered state of the last bar that CLOSED
    before its entry, plus the conductor's baseline state on the same clock.

    The state frame is indexed by close time, so "closed before the fill" is
    `close_ts <= fill_ts` — a bar that closes exactly at the fill instant IS
    knowable (the WS0 spine learned this the hard way: one-tick-too-strict
    refused 98 legitimate bar-close fills).
    """
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.vol_atlas import fleet_fills_with_dvol                  # noqa: PLC0415

    if path.states.empty:
        return pd.DataFrame()
    f = fills if fills is not None else fleet_fills_with_dvol()
    if f.empty or "fill_ts" not in f.columns:
        return pd.DataFrame()
    f = f.dropna(subset=["fill_ts", "r"]).copy()
    f["fill_ts"] = pd.to_datetime(f["fill_ts"], utc=True)
    f = f.sort_values("fill_ts").reset_index(drop=True)

    st = next_switch_labels(path.states).sort_values("close_ts").reset_index(drop=True)
    _ns = lambda x: pd.DatetimeIndex(x).tz_convert("UTC").tz_localize(None).astype("int64").to_numpy()  # noqa: E731
    idx = np.searchsorted(_ns(st["close_ts"]), _ns(f["fill_ts"]), side="right") - 1
    # A state older than a day is not this fill's regime; drop rather than dress up.
    matched_ns = _ns(st["close_ts"])[np.clip(idx, 0, None)]
    age_h = (_ns(f["fill_ts"]) - matched_ns) / 3.6e12
    ok = (idx >= 0) & (age_h <= 24.0)
    for col in ("state", "p_state", "next_switch_state", "bars_to_switch", "window"):
        vals = st[col].to_numpy()
        f[col] = np.where(ok, vals[np.clip(idx, 0, None)], np.nan)
    f["state"] = pd.to_numeric(f["state"], errors="coerce")

    if feat is not None and not feat.empty:
        fe = feat.dropna(subset=["close_ts"]).sort_values("close_ts")
        j = np.searchsorted(_ns(fe["close_ts"]), _ns(f["fill_ts"]), side="right") - 1
        q = fe["conductor_quiet"].to_numpy()
        f["conductor_quiet"] = np.where(j >= 0, q[np.clip(j, 0, None)], None)
    return f.dropna(subset=["state"])


def state_outcome_table(tagged: pd.DataFrame, min_bets: int = 40) -> pd.DataFrame:
    """Per family x state: bets, mean R, t. R is per BET (the fleet_edge haircut)."""
    if tagged.empty:
        return pd.DataFrame()
    rows = []
    for (fam, stt), sub in tagged.groupby(["family", "state"], observed=True):
        bets = sub.groupby("cluster")["r"].mean()
        rows.append({"family": fam, "state": int(stt), "rows": int(len(sub)),
                     "bets": int(len(bets)), "mean_r": float(bets.mean()),
                     "t": float(_t(bets.to_numpy()))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    tot = df.groupby("family")["bets"].transform("sum")
    return df[tot >= min_bets].sort_values(["family", "state"]).reset_index(drop=True)


def conditional_outcome_table(tagged: pd.DataFrame, min_bets: int = 20) -> pd.DataFrame:
    """Per family: R of fills opened in state A, split by the state the tape
    switched INTO next — "what pays when the tape is calm but about to break".

    `next_switch_state` is the one forward-looking column in this module. It
    labels an outcome; it is never a feature, and no gate may read it, because
    at entry time nobody knows which way calm will break.
    """
    if tagged.empty:
        return pd.DataFrame()
    t = tagged[tagged["next_switch_state"] >= 0]
    rows = []
    for (fam, a, b), sub in t.groupby(["family", "state", "next_switch_state"], observed=True):
        bets = sub.groupby("cluster")["r"].mean()
        if len(bets) < min_bets:
            continue
        rows.append({"family": fam, "state": int(a), "next_switch_to": int(b),
                     "bets": int(len(bets)), "mean_r": float(bets.mean()),
                     "median_bars_to_switch": float(sub["bars_to_switch"].median()),
                     "t": float(_t(bets.to_numpy()))})
    return pd.DataFrame(rows).sort_values(["family", "state", "next_switch_to"]).reset_index(drop=True)


def _t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def _best_split_lift(bets: pd.DataFrame, label_col: str) -> Optional[float]:
    """Mean R of the best label minus the family's overall mean — what the split
    would buy per trade if you traded only its best side."""
    if bets.empty or label_col not in bets.columns:
        return None
    g = bets.dropna(subset=[label_col]).groupby(label_col)["r"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    if len(g) < 2:
        return None
    return float(g["mean"].max() - bets["r"].mean())


def evaluate_kill_bar(tagged: pd.DataFrame, n_states: int, n_k_tested: int = 2,
                      min_bets: int = 40, n_perm: int = 2000,
                      lift_required: float = 0.10) -> pd.DataFrame:
    """The four frozen conditions of WS3_HMM_REGIME_PREREG.md, per family.

    1. state split clears the day-blocked permutation at alpha/(families x K)
    2. it beats the conductor's RV24 rule by >= `lift_required` R per trade
    3. the state ordering holds in both halves
    4. >= `min_bets` effective bets

    Nothing is "close enough": all four, or the HMM is an atlas.
    """
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.vol_atlas import _half_split_agrees, band_permutation_test   # noqa: PLC0415

    if tagged.empty:
        return pd.DataFrame()
    rows = []
    for fam, sub in tagged.groupby("family", observed=True):
        bets = (sub.dropna(subset=["state", "day"])
                .groupby("cluster")
                .agg(r=("r", "mean"), band=("state", "first"), day=("day", "first"),
                     quiet=("conductor_quiet", "first")))
        if len(bets) < min_bets:
            continue
        bets["band"] = bets["band"].astype(int).astype(str)
        res = band_permutation_test(bets["r"].to_numpy(), bets["band"].to_numpy(),
                                    bets["day"].to_numpy(), n_perm=n_perm)
        hmm_lift = _best_split_lift(bets, "band")
        cond_lift = _best_split_lift(bets.assign(quiet=bets["quiet"].map(
            {True: "quiet", False: "busy"})), "quiet")
        rows.append({
            "family": fam, "bets": int(len(bets)), "states": int(bets["band"].nunique()),
            "days": int(res.get("n_days", 0)),
            "max_abs_t": res["obs_max_abs_t"], "p_fwer": res["p_fwer"],
            "hmm_lift_r": hmm_lift, "conductor_lift_r": cond_lift,
            "lift_vs_conductor": (None if hmm_lift is None or cond_lift is None
                                  else hmm_lift - cond_lift),
            "half_split": _half_split_agrees(bets),
            "note": res.get("note", "")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    tested = int(df["p_fwer"].notna().sum())
    df["alpha"] = 0.05 / max(tested * max(n_k_tested, 1), 1)
    df["c1_separation"] = df["p_fwer"].notna() & (df["p_fwer"] < df["alpha"])
    df["c2_beats_conductor"] = df["lift_vs_conductor"].fillna(-99) >= lift_required
    df["c3_half_split"] = df["half_split"].eq(True)
    df["c4_bets"] = df["bets"] >= min_bets
    df["GATE_CANDIDATE"] = (df["c1_separation"] & df["c2_beats_conductor"]
                            & df["c3_half_split"] & df["c4_bets"])
    return df.sort_values("p_fwer", na_position="last").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Cache — a walk-forward fit is minutes; a page load is not
# ══════════════════════════════════════════════════════════════════════════════
CACHE_DIR = _ROOT / "flow_aux_data" / "ws3"


def cached_states(n_states: int = 2, features: Sequence[str] = PRIMARY_FEATURES,
                  train_days: int = 90, step_days: int = 20,
                  feat: Optional[pd.DataFrame] = None,
                  rebuild: bool = False) -> RegimePath:
    """Load the walk-forward path from disk, fitting it once if absent.

    ~95 rolling Baum-Welch fits is minutes of CPU — fine for a research run, not for
    a page load. The cache key carries every parameter that changes the path, so
    a different K or window silently reusing another run's states is impossible.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"K{n_states}_{'-'.join(features)}_{train_days}x{step_days}"
    sp = CACHE_DIR / f"states_{tag}.parquet"
    mp = CACHE_DIR / f"models_{tag}.parquet"
    if sp.exists() and not rebuild:
        states = pd.read_parquet(sp)
        states["close_ts"] = pd.to_datetime(states["close_ts"], utc=True)
        models = pd.read_parquet(mp).to_dict("records") if mp.exists() else []
        return RegimePath(states=states, models=models, n_states=n_states,
                          features=tuple(features), train_days=train_days,
                          step_days=step_days)
    if feat is None:
        feat = btc_features()
    path = rolling_states(feat, n_states=n_states, features=features,
                          train_days=train_days, step_days=step_days)
    if not path.states.empty:
        path.states.to_parquet(sp, index=False)
        pd.DataFrame([{k: v for k, v in m.items()
                       if k not in ("transition_matrix",)} for m in path.models]
                     ).to_parquet(mp, index=False)
    return path
