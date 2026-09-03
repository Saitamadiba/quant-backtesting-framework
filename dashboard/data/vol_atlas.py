"""WS2 — the real DVOL series, and what the fleet's assets actually do inside it.

Everything here runs on THIS machine. The inputs are the local duckdb OHLCV
store, the local hourly/daily DVOL parquets (Deribit's public index, fetched by
``backfill_dvol_parquet.py``) and two small side-series pulled read-only off the
VPS by ``backfill_vol_tier3.py``. Nothing is computed or cached on the VPS —
that box is a filing cabinet we read, not a scratch disk.

What it answers, in the order the plan asks:

1. **Where is implied vol, right now and historically?** DVOL level, its 30d and
   90d trailing percentile, Δ1h and Δ24h — all trailing, so a bar never sees its
   own future.
2. **Does high DVOL mean the alts move more, or just BTC?** Realized vol and
   return dispersion per asset, per DVOL band.
3. **How correlated is the fleet's universe week to week?** Rolling 7-day
   correlation of 15-minute returns — the number that decides whether two long
   seats are two bets or one.
4. **Does BTC lead the alts?** Cross-correlation at 1–4 bars, both directions.
5. **Beta and residual vol by band** — does diversification survive a fear spike?
6. **Is skew a directional gauge?** RR25 / skew against the SUBSEQUENT 24h
   return, sign only, and always beside the mirrored bracket: on a driftless
   path any bracket looks asymmetric, so "it hits +x% first" is not a finding
   until the mirror says otherwise (closure §5(f)).
7. **Does the fleet's own R depend on the band?** Per-family R by DVOL band and
   by Δdvol sign, joined at the fill's entry time with a one-bar lag.

**This is an atlas, not a gate.** The standing closure (§1: the pre-fill panel is
NULL at p_FWER 0.60 across 647 demo fills) says the expected answer is *no edge
by band*. Any claim to the contrary here carries a family-wise bar — permutation
over band labels, block-shuffled BY DAY so a day's autocorrelated fills move
together — plus a half-split. The value that does not need a p-value is the
correlation map: *two seats long correlated alts in a 0.9-correlation week are
one bet wearing two tickets.*
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
DVOL_DIR = _ROOT / "flow_aux_data" / "dvol"
TIER3_DIR = _ROOT / "flow_aux_data" / "vol_tier3"
CACHE_DIR = _ROOT / "flow_aux_data" / "ws2"
DUCKDB_PATH = _ROOT / "duckdb_data" / "trading_data.duckdb"

FLEET = ("BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "BCH", "BNB", "DOGE", "DOT", "LINK", "LTC")
# The adapters' absolute bands (dvol_band in lr_faithful_filters); kept so the
# atlas and the live IV gate never disagree about what "HIGH" means.
ABS_BANDS = (("LOW", -np.inf, 45.0), ("MED", 45.0, 65.0), ("HIGH", 65.0, np.inf))
# Percentile bands are the honest comparison across a 5-year sample whose level
# drifts: 40 vol is "high" in a calm year and "low" in a violent one.
PCT_EDGES = (0.0, 0.25, 0.75, 1.0)
PCT_NAMES = ("P_LOW", "P_MID", "P_HIGH")
BAR_MS = 15 * 60 * 1000


# ══════════════════════════════════════════════════════════════════════════════
#  Inputs
# ══════════════════════════════════════════════════════════════════════════════
def load_dvol_hourly(symbol: str = "BTC") -> pd.DataFrame:
    """Hourly DVOL bars, indexed by the bar's OPEN time (Deribit's stamp)."""
    p = DVOL_DIR / f"dvol_hourly_{symbol}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["ts_utc", "dvol"])
    df = pd.read_parquet(p)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df.sort_values("ts_utc").reset_index(drop=True)


def load_rr25() -> pd.DataFrame:
    """ATM IV / 25d risk-reversal / 25d butterfly at the nearest listed expiry."""
    p = TIER3_DIR / "rr25.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df.sort_values("ts_utc").reset_index(drop=True)


def load_vol_series_hourly() -> pd.DataFrame:
    """The recorder's own hourly DVOL/skew means (BTC and ETH), from the VPS."""
    p = TIER3_DIR / "vol_series_hourly.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
    return df.sort_values("hour_utc").reset_index(drop=True)


def load_bars(symbol: str, tf: str = "15m", start: Optional[str] = None) -> pd.DataFrame:
    """OHLCV from the LOCAL duckdb store. Timestamps are naive UTC by
    convention throughout this repo; localized here so every join is tz-aware."""
    try:
        import duckdb
    except ImportError:                                       # pragma: no cover
        return pd.DataFrame()
    if not DUCKDB_PATH.exists():
        return pd.DataFrame()
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        sql = ("SELECT timestamp, open, high, low, close, volume FROM ohlcv_data "
               "WHERE symbol = ? AND timeframe = ?")
        args: list = [symbol, tf]
        if start:
            sql += " AND timestamp >= ?"
            args.append(start)
        df = conn.execute(sql + " ORDER BY timestamp", args).fetchdf()
    finally:
        conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  1 · The DVOL feature frame — every column trailing
# ══════════════════════════════════════════════════════════════════════════════
def dvol_features(symbol: str = "BTC") -> pd.DataFrame:
    """Hourly DVOL with trailing percentiles and deltas.

    `pct30` / `pct90` are the rank of the current level inside the trailing 30 /
    90 days INCLUDING the current bar — a percentile computed over the whole
    sample would let 2021 know about 2026. `d1h` / `d24h` are differences
    against completed bars only.
    """
    df = load_dvol_hourly(symbol)
    if df.empty:
        return df
    out = df[["ts_utc", "dvol"]].copy()
    s = out["dvol"]
    for name, hours in (("pct30", 30 * 24), ("pct90", 90 * 24)):
        out[name] = s.rolling(hours, min_periods=max(24, hours // 10)).apply(
            lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True)
    out["d1h"] = s.diff(1)
    out["d24h"] = s.diff(24)
    out["abs_band"] = pd.cut(s, bins=[b[1] for b in ABS_BANDS] + [ABS_BANDS[-1][2]],
                             labels=[b[0] for b in ABS_BANDS], right=False)
    out["pct_band"] = pd.cut(out["pct30"], bins=PCT_EDGES, labels=list(PCT_NAMES),
                             include_lowest=True)
    return out


def dvol_asof(feat: pd.DataFrame, when: pd.Series | pd.DatetimeIndex,
              lag_hours: int = 1, max_stale_hours: float = 24.0) -> pd.DataFrame:
    """Point-in-time join: the newest DVOL bar that had CLOSED by `when`.

    Deribit stamps a bar at its OPEN, so the bar stamped 10:00 is only complete
    at 11:00. A fill at 10:30 may therefore see the 09:00 bar and no later —
    hence the one-bar lag. Getting this wrong is the same look-ahead the daily
    join was fixed for on 2026-08-17.

    `max_stale_hours` is the other half of the contract. An as-of join with no
    age limit will happily hand a fill the last print before a gap — or the end
    of the series — however old it is, and the row then looks fully featured
    while carrying a number from another week. That is the depth stale-signal
    failure in miniature (a median 358-minute-old signal traded as fresh); here
    an over-age match is dropped rather than dressed up.
    """
    if feat.empty:
        return pd.DataFrame(index=pd.Index(when))
    w = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(when)), utc=True))
    src = feat.copy()
    src["closes_at"] = src["ts_utc"] + pd.Timedelta(hours=lag_hours)
    src = src.sort_values("closes_at")
    # A tz-aware DatetimeIndex hands back OBJECT dtype from .to_numpy(); drop to
    # naive UTC nanoseconds so both the search and the age arithmetic are numeric.
    _ns = lambda x: pd.DatetimeIndex(x).tz_convert("UTC").tz_localize(None).astype("int64").to_numpy()  # noqa: E731
    idx = np.searchsorted(_ns(src["closes_at"]), _ns(w), side="right") - 1
    if max_stale_hours is not None:
        matched = _ns(src["ts_utc"])[np.clip(idx, 0, None)]
        age_h = (_ns(w) - matched) / 3.6e12
        idx = np.where((idx >= 0) & (age_h <= max_stale_hours + lag_hours), idx, -1)
    cols = ["dvol", "pct30", "pct90", "d1h", "d24h", "abs_band", "pct_band"]
    out = pd.DataFrame(index=range(len(w)))
    for c in cols:
        vals = src[c].to_numpy()
        out[c] = np.where(idx >= 0, vals[np.clip(idx, 0, None)], None)
    out["dvol_ts"] = np.where(idx >= 0, src["ts_utc"].to_numpy()[np.clip(idx, 0, None)],
                              np.datetime64("NaT"))
    for c in ("dvol", "pct30", "pct90", "d1h", "d24h"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  2 · Does a fear spike move the alts, or only BTC?
# ══════════════════════════════════════════════════════════════════════════════
def returns_panel(symbols: Sequence[str] = FLEET, tf: str = "15m",
                  start: str = "2021-03-24") -> pd.DataFrame:
    """Log returns of every fleet asset on one aligned index."""
    cols = {}
    for s in symbols:
        b = load_bars(s, tf, start)
        if b.empty:
            continue
        cols[s] = np.log(b.set_index("timestamp")["close"]).diff()
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def realized_by_band(symbols: Sequence[str] = FLEET, tf: str = "15m",
                     band_col: str = "pct_band") -> pd.DataFrame:
    """Per asset, per BTC-DVOL band: realized vol and cross-sectional dispersion.

    Realized vol is annualised from the bar returns; dispersion is the
    interquartile range of the same returns, in basis points — a fat middle, not
    just a fat tail. The band is BTC's, because BTC's DVOL is the fleet's fear
    gauge; the question is whether the alts inherit it.
    """
    ret = returns_panel(symbols, tf)
    if ret.empty:
        return pd.DataFrame()
    feat = dvol_features("BTC")
    joined = dvol_asof(feat, ret.index)
    joined.index = ret.index
    bars_per_year = (365 * 24 * 60) / _tf_minutes(tf)
    rows = []
    for band, idx in joined.groupby(band_col, observed=True).groups.items():
        sub = ret.loc[idx]
        for sym in sub.columns:
            v = sub[sym].dropna()
            if len(v) < 100:
                continue
            rows.append({
                "band": str(band), "symbol": sym, "n_bars": int(len(v)),
                "rv_annual_pct": float(v.std() * np.sqrt(bars_per_year) * 100),
                "iqr_bps": float((v.quantile(0.75) - v.quantile(0.25)) * 1e4),
                "mean_abs_bps": float(v.abs().mean() * 1e4),
            })
    return pd.DataFrame(rows)


def _tf_minutes(tf: str) -> int:
    unit, mult = tf[-1].lower(), int(tf[:-1])
    return mult * {"m": 1, "h": 60, "d": 1440}[unit]


# ══════════════════════════════════════════════════════════════════════════════
#  3 · How correlated is the universe, week by week?
# ══════════════════════════════════════════════════════════════════════════════
def rolling_correlation(symbols: Sequence[str] = FLEET, tf: str = "15m",
                        window_days: int = 7, step_days: int = 1) -> pd.DataFrame:
    """Mean pairwise correlation of 15m returns in each trailing window.

    Stepped by day rather than by bar: 190k overlapping windows would be 190k
    near-identical matrices and an hour of CPU for no extra information.
    Returns one row per window end with the mean, min and max pairwise rho and
    the top pair — the concentration read, not a forecast.
    """
    ret = returns_panel(symbols, tf).dropna(how="all")
    if ret.empty:
        return pd.DataFrame()
    ends = pd.date_range(ret.index.min() + pd.Timedelta(days=window_days),
                         ret.index.max(), freq=f"{step_days}D", tz="UTC")
    rows = []
    for end in ends:
        sub = ret.loc[end - pd.Timedelta(days=window_days):end].dropna(axis=1, thresh=50)
        if sub.shape[1] < 2:
            continue
        c = sub.corr()
        iu = np.triu_indices_from(c.values, k=1)
        vals = c.values[iu]
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            continue
        top = int(np.argmax(vals))
        rows.append({"window_end": end, "n_assets": int(sub.shape[1]),
                     "mean_rho": float(vals.mean()), "min_rho": float(vals.min()),
                     "max_rho": float(vals.max()),
                     "top_pair": f"{c.index[iu[0][top]]}/{c.columns[iu[1][top]]}"})
    return pd.DataFrame(rows)


def correlation_matrix(symbols: Sequence[str] = FLEET, tf: str = "15m",
                       window_days: int = 7, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """The single correlation matrix for the most recent (or given) window."""
    ret = returns_panel(symbols, tf).dropna(how="all")
    if ret.empty:
        return pd.DataFrame()
    end = end or ret.index.max()
    sub = ret.loc[end - pd.Timedelta(days=window_days):end].dropna(axis=1, thresh=50)
    return sub.corr()


# ══════════════════════════════════════════════════════════════════════════════
#  4 · Does BTC lead the alts?
# ══════════════════════════════════════════════════════════════════════════════
def lead_lag(symbols: Sequence[str] = FLEET, tf: str = "15m",
             lags: Sequence[int] = (1, 2, 3, 4)) -> pd.DataFrame:
    """corr(BTC_t, alt_{t+k}) and corr(alt_t, BTC_{t+k}) for k in `lags`.

    Both directions, always. A lead that only exists one way is a candidate; one
    that shows up symmetrically is contemporaneous correlation smeared by the
    bar clock, not information flow. Sample correlations at n≈190k have a
    standard error near 1/sqrt(n) ≈ 0.002, so the reported rho is precise — that
    is not the same as tradeable, since 15m at these sizes pays the toll.
    """
    ret = returns_panel(symbols, tf)
    if ret.empty or "BTC" not in ret.columns:
        return pd.DataFrame()
    btc = ret["BTC"]
    rows = []
    for sym in ret.columns:
        if sym == "BTC":
            continue
        alt = ret[sym]
        pair = pd.concat([btc.rename("btc"), alt.rename("alt")], axis=1).dropna()
        if len(pair) < 500:
            continue
        row = {"symbol": sym, "n": int(len(pair)),
               "rho_0": float(pair["btc"].corr(pair["alt"]))}
        for k in lags:
            row[f"btc_leads_{k}"] = float(pair["btc"].corr(pair["alt"].shift(-k)))
            row[f"alt_leads_{k}"] = float(pair["alt"].corr(pair["btc"].shift(-k)))
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  5 · Beta and residual vol by band
# ══════════════════════════════════════════════════════════════════════════════
def beta_by_band(symbols: Sequence[str] = FLEET, tf: str = "15m",
                 band_col: str = "pct_band") -> pd.DataFrame:
    """Each alt's beta to BTC and its residual vol, inside each DVOL band.

    If beta rises and residual vol falls as fear rises, diversification is
    evaporating exactly when it is needed — every seat becomes the same seat.
    """
    ret = returns_panel(symbols, tf)
    if ret.empty or "BTC" not in ret.columns:
        return pd.DataFrame()
    joined = dvol_asof(dvol_features("BTC"), ret.index)
    joined.index = ret.index
    bars_per_year = (365 * 24 * 60) / _tf_minutes(tf)
    rows = []
    for band, idx in joined.groupby(band_col, observed=True).groups.items():
        sub = ret.loc[idx]
        if "BTC" not in sub.columns:
            continue
        for sym in sub.columns:
            if sym == "BTC":
                continue
            pair = sub[["BTC", sym]].dropna()
            if len(pair) < 200:
                continue
            x, y = pair["BTC"].to_numpy(), pair[sym].to_numpy()
            var = float(np.var(x))
            if var <= 0:
                continue
            beta = float(np.cov(x, y)[0, 1] / var)
            resid = y - beta * x
            rows.append({"band": str(band), "symbol": sym, "n": int(len(pair)),
                         "beta": beta,
                         "resid_vol_annual_pct": float(np.std(resid) * np.sqrt(bars_per_year) * 100),
                         "r2": float(np.corrcoef(x, y)[0, 1] ** 2)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  6 · Is skew a directional gauge? — with the mirrored bracket
# ══════════════════════════════════════════════════════════════════════════════
def skew_direction(asset: str = "BTC", horizon_h: int = 24,
                   bracket_pct: float = 1.0) -> Dict[str, object]:
    """Does today's 25-delta risk reversal say anything about tomorrow's move?

    Two readings, deliberately side by side:

    * **sign test** — the mean and hit rate of the SUBSEQUENT `horizon_h` return,
      split by the sign of RR25. Direction only; magnitude is not the claim.
    * **the mirrored bracket** — of the paths that touch +`bracket_pct`% or
      -`bracket_pct`% first, which comes first, and does the answer flip when the
      bracket is mirrored? On a driftless path the "first touch" rate is a
      function of the bracket, not of the signal (closure §5(f)); an asymmetry
      that survives the mirror is the only kind worth a second look.
    """
    rr = load_rr25()
    if rr.empty:
        return {"status": "no rr25 series"}
    rr = rr[rr["asset"] == asset]
    if rr.empty:
        return {"status": f"no rr25 rows for {asset}"}
    bars = load_bars(asset, "1h", str(rr["ts_utc"].min().date()))
    if bars.empty:
        return {"status": f"no local 1h bars for {asset}"}

    px = bars.set_index("timestamp")
    # one observation per hour: the LAST rr25 print of that hour, read at the
    # hour's close, acting on the NEXT hour's bars.
    rr_h = (rr.set_index("ts_utc")[["rr25", "atm_iv", "bf25"]]
            .resample("1h").last().dropna())
    obs = []
    closes = px["close"]
    highs, lows = px["high"], px["low"]
    for ts, row in rr_h.iterrows():
        entry_ts = ts + pd.Timedelta(hours=1)
        if entry_ts not in closes.index:
            continue
        entry = float(closes.loc[entry_ts])
        fwd = closes.loc[entry_ts:entry_ts + pd.Timedelta(hours=horizon_h)]
        if len(fwd) < 2:
            continue
        win_h = highs.loc[entry_ts:entry_ts + pd.Timedelta(hours=horizon_h)]
        win_l = lows.loc[entry_ts:entry_ts + pd.Timedelta(hours=horizon_h)]
        up, dn = entry * (1 + bracket_pct / 100), entry * (1 - bracket_pct / 100)
        t_up = np.argmax(win_h.to_numpy() >= up) if (win_h >= up).any() else np.inf
        t_dn = np.argmax(win_l.to_numpy() <= dn) if (win_l <= dn).any() else np.inf
        first = "up" if t_up < t_dn else "down" if t_dn < t_up else "neither"
        obs.append({"ts": entry_ts, "rr25": float(row["rr25"]),
                    "atm_iv": float(row["atm_iv"]),
                    "fwd_ret_bps": float((fwd.iloc[-1] / entry - 1) * 1e4),
                    "first_touch": first})
    if len(obs) < 30:
        return {"status": f"only {len(obs)} usable observations"}
    d = pd.DataFrame(obs)
    pos, neg = d[d["rr25"] > 0], d[d["rr25"] <= 0]
    base_up = float((d["first_touch"] == "up").mean())
    out = {
        "status": "ok", "asset": asset, "n": int(len(d)),
        "horizon_h": horizon_h, "bracket_pct": bracket_pct,
        "rr25_positive_n": int(len(pos)), "rr25_negative_n": int(len(neg)),
        "mean_fwd_bps_rr_pos": float(pos["fwd_ret_bps"].mean()) if len(pos) else float("nan"),
        "mean_fwd_bps_rr_neg": float(neg["fwd_ret_bps"].mean()) if len(neg) else float("nan"),
        "up_rate_rr_pos": float((pos["first_touch"] == "up").mean()) if len(pos) else float("nan"),
        "up_rate_rr_neg": float((neg["first_touch"] == "up").mean()) if len(neg) else float("nan"),
        "up_rate_unconditional": base_up,
        "obs": d,
    }
    # The mirror: an unconditional first-touch rate far from 50% means the
    # BRACKET is asymmetric on this tape, and every conditional split inherits
    # that — the split is only informative relative to this baseline.
    out["mirror_note"] = (
        f"unconditional up-first {base_up:.1%}; a conditional rate must be read "
        f"against THAT, not against 50%"
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  7 · The fleet join — R by band, with a family-wise bar
# ══════════════════════════════════════════════════════════════════════════════
def fleet_fills_with_dvol(tiers: Sequence[int] = (1, 2, 3)) -> pd.DataFrame:
    """Every fleet fill's R, tagged with the DVOL state that had CLOSED by entry.

    Reuses `fleet_edge`'s reader so the two pages cannot disagree about what a
    book's rows are, and applies the same effective-n haircut: a cell's `bets`
    column counts clusters, not rows.
    """
    if str(_ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(_ROOT / "dashboard"))
    from data.fleet_edge import _book_path, load_book_series          # noqa: PLC0415
    from data.fleet_registry import BOOKS                             # noqa: PLC0415
    try:
        from config import FLEET_CACHE_DIR, VPS_CACHE_DIR             # noqa: PLC0415
        fdir, ldir = Path(FLEET_CACHE_DIR), Path(VPS_CACHE_DIR)
    except Exception:                                                 # noqa: BLE001
        fdir = _ROOT / "dashboard" / "databases" / "fleet"
        ldir = _ROOT / "dashboard" / "databases"

    feat = dvol_features("BTC")
    frames = []
    for b in BOOKS:
        if int(b.tier) not in tiers:
            continue
        p = _book_path(b, fdir, ldir)
        if p is None:
            continue
        s = load_book_series(b, p)
        if s.error or s.n == 0 or s.first_ts is None:
            continue
        # load_book_series keeps R and clusters aligned; rebuild the timestamps
        # the same way so the join is on the fill, not the close.
        ts = _series_timestamps(b, p)
        if ts is None or len(ts) != s.n:
            continue
        j = dvol_asof(feat, ts)
        j["r"] = s.r
        j["cluster"] = s.clusters
        j["family"] = s.family
        j["tier"] = s.tier
        j["book"] = s.label
        frames.append(j)
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(frames, ignore_index=True)
    # A missing d24h (a fill before the index had 24h of history, or outside it
    # entirely) is NOT "flat" — np.where would happily call NaN flat and put an
    # unknown into a labelled bucket. It stays None so the groupby drops it.
    d24 = allf["d24h"]
    allf["d24h_sign"] = np.where(d24.isna(), None,
                                 np.where(d24 > 0, "rising",
                                          np.where(d24 < 0, "falling", "flat")))
    allf["day"] = pd.to_datetime(allf["dvol_ts"], utc=True, errors="coerce").dt.floor("D")
    return allf


def fleet_r_by_band(tiers: Sequence[int] = (1, 2, 3), band_col: str = "pct_band",
                    min_n: int = 30, fills: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per family x band: rows, effective bets, mean R and its t — the atlas
    table. `t` is against ZERO and therefore mostly reports what page 31 already
    says (the books are negative); the question this workstream actually asks —
    does the BAND change anything — is `fleet_band_fwer` below."""
    allf = fills if fills is not None else fleet_fills_with_dvol(tiers)
    if allf.empty:
        return pd.DataFrame()
    rows = []
    for (fam, band), sub in allf.groupby(["family", band_col], observed=True):
        if len(sub) < min_n:
            continue
        bets = sub.groupby("cluster")["r"].mean()
        rows.append({"family": fam, "band": str(band), "rows": int(len(sub)),
                     "bets": int(len(bets)), "mean_r": float(bets.mean()),
                     "sum_r": float(bets.sum()),
                     "t": float(_tstat(bets.to_numpy()))})
    return pd.DataFrame(rows).sort_values(["family", "band"]).reset_index(drop=True)


def fleet_band_fwer(tiers: Sequence[int] = (1, 2, 3), band_col: str = "pct_band",
                    min_bets: int = 40, n_perm: int = 2000,
                    fills: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Does R depend on the DVOL band? One permutation test per family, plus the
    fleet-wide max across families.

    The statistic is a CONTRAST — each band's mean minus the family's own mean —
    so a book that simply loses in every band (all of them) scores zero here.
    Labels are permuted in day-sized blocks, because the band barely moves inside
    a day: shuffling fill by fill would compare the real, day-clustered data
    against a null that never clusters, and hand back significance for free.
    """
    allf = fills if fills is not None else fleet_fills_with_dvol(tiers)
    if allf.empty:
        return pd.DataFrame()
    out = []
    for fam, sub in allf.groupby("family", observed=True):
        bets = (sub.dropna(subset=[band_col, "day"])
                .groupby(["cluster"])
                .agg(r=("r", "mean"), band=(band_col, "first"), day=("day", "first")))
        if len(bets) < min_bets or bets["band"].nunique() < 2:
            continue
        res = band_permutation_test(bets["r"].to_numpy(),
                                    bets["band"].astype(str).to_numpy(),
                                    bets["day"].to_numpy(), n_perm=n_perm)
        half = _half_split_agrees(bets)
        out.append({"family": fam, "bets": int(len(bets)),
                    "bands": int(bets["band"].nunique()),
                    "days": int(res.get("n_days", 0)),
                    "max_abs_t": res["obs_max_abs_t"], "p_fwer": res["p_fwer"],
                    "half_split": half, "note": res.get("note", "")})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df.sort_values("p_fwer", na_position="last").reset_index(drop=True)
    # Bonferroni over the families that could actually be TESTED — a family the
    # day-block guard refused is not a test and must not make the bar harder for
    # the others, nor may its NaN ever read as "clears".
    tested = int(df["p_fwer"].notna().sum())
    df["alpha_bonferroni"] = 0.05 / max(tested, 1)
    # A survivor owes BOTH: the family-wise bar AND a band ordering that holds in
    # each half of its own history. An effect that only exists in one half is a
    # window, not a mechanism.
    df["clears"] = (df["p_fwer"].notna() & (df["p_fwer"] < df["alpha_bonferroni"])
                    & df["half_split"].eq(True))
    return df


def _half_split_agrees(bets: pd.DataFrame) -> Optional[bool]:
    """Split a family's days down the middle: does the best and worst band keep
    its rank in both halves?

    Cheap, and the point is not the p-value — it is that a real band effect
    should be visible in the first half of the sample and again in the second.
    Returns None when either half is too thin to rank.
    """
    if bets.empty or "day" not in bets.columns:
        return None
    days = np.sort(pd.unique(bets["day"]))
    if len(days) < 4:
        return None
    cut = days[len(days) // 2]
    h1 = bets[bets["day"] < cut]
    h2 = bets[bets["day"] >= cut]
    means = []
    for h in (h1, h2):
        m = h.groupby("band")["r"].agg(["mean", "size"])
        m = m[m["size"] >= 5]["mean"]
        if len(m) < 2:
            return None
        means.append(m)
    common = means[0].index.intersection(means[1].index)
    if len(common) < 2:
        return None
    return bool(means[0][common].idxmax() == means[1][common].idxmax())


def _series_timestamps(book, db_path: Path) -> Optional[pd.Series]:
    """The entry timestamps behind `load_book_series`'s R array, in the same
    order — read through the registry's own query so the two never drift."""
    import sqlite3                                                     # noqa: PLC0415
    from data.fleet_edge import to_utc                                 # noqa: PLC0415
    from data.fleet_registry import build_history_sql                  # noqa: PLC0415
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({book.table})").fetchall()}
            raw = pd.read_sql_query(build_history_sql(book, available_cols=cols), conn)
    except Exception:                                                  # noqa: BLE001
        return None
    if raw.empty:
        return None
    r = pd.to_numeric(raw.get("r"), errors="coerce")
    ts = to_utc(raw.get("entry_ts"))
    ts = ts.where(ts.notna(), to_utc(raw.get("exit_ts")))
    return ts[r.notna()].reset_index(drop=True)


def _tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def band_permutation_test(values: np.ndarray, bands: np.ndarray, days: np.ndarray,
                          n_perm: int = 2000, seed: int = 0,
                          min_days: int = 20, min_band_days: int = 5) -> Dict[str, float]:
    """Family-wise bar for "R depends on the band": permute the BAND LABELS in
    day-sized blocks and ask how often the shuffled data beats the real max |t|.

    Shuffling fill by fill would break the day's autocorrelation and hand back a
    null that is far too easy to beat — the same effective-n mistake, wearing a
    different hat. Blocks are whole days, so a day's fills keep their label
    together. Returns the observed max |t| across bands and its FWER p-value.

    ``min_days`` is the guard that makes the p-value mean anything. The block
    permutation can only produce as many distinct arrangements as the day labels
    allow: with 7 days split 4/3 there are 35, so the smallest honest p-value is
    1/35 — yet a sampler that never happens to match will happily report 1/2001
    and look like a discovery. That is not a hypothetical. On this fleet,
    `halt-shadow(era2)` spans 7 days and returned p=0.0005 on a max |t| of 1.67,
    and `fib618-shadow(taker)` spans ONE day, where permuting the single block
    changes nothing at all and every statistic is "unbeatable". Below `min_days`
    distinct blocks the test refuses to answer instead of inventing one.

    ``min_band_days`` closes the second door into the same trap. If one band is
    carried by only a day or two, most shuffles leave it with too few members to
    score, the permuted statistic collapses to zero, and NOTHING ever beats the
    observed value — so a max |t| of 0.04 comes back at p = 0.0005. That is not
    hypothetical either: it is exactly what the fleet's ABSOLUTE bands did, where
    almost every fill sits in one band. A band that only a handful of days can
    speak for is not a comparison; the test says so and declines.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    b = np.asarray(bands, dtype=object)
    d = np.asarray(days)
    keep = ~np.isnan(v)
    v, b, d = v[keep], b[keep], d[keep]
    n_days = int(len(pd.unique(d)))
    if len(v) < 30 or n_days < min_days:
        return {"obs_max_abs_t": float("nan"), "p_fwer": float("nan"),
                "n_perm": float(n_perm), "n_days": float(n_days),
                "note": ("too few fills" if len(v) < 30
                         else f"only {n_days} day-blocks (<{min_days}) — the "
                              "permutation null is degenerate, no p-value is honest")}
    # Every band must be carried by enough DISTINCT days to survive a shuffle.
    per_band_days = pd.DataFrame({"b": b, "d": d}).groupby("b")["d"].nunique()
    thin = per_band_days[per_band_days < min_band_days]
    if len(per_band_days) - len(thin) < 2:
        return {"obs_max_abs_t": float("nan"), "p_fwer": float("nan"),
                "n_perm": float(n_perm), "n_days": float(n_days),
                "note": (f"fewer than 2 bands carried by >={min_band_days} days "
                         f"({dict(per_band_days)}) — nothing to compare")}
    if len(thin):
        keep2 = ~pd.Series(b).isin(thin.index).to_numpy()
        v, b, d = v[keep2], b[keep2], d[keep2]
        n_days = int(len(pd.unique(d)))

    # The statistic is a CONTRAST: each band's mean against the sample's own
    # mean. Testing band means against ZERO would just re-report that the books
    # lose money — true, already known from page 31, and completely unmoved by
    # shuffling the band labels. What is being asked here is narrower and
    # answerable: does knowing the band tell you anything the overall average
    # does not?
    grand = float(np.mean(v))

    def max_abs_t(labels):
        """Max |t| of a band's contrast, or None when fewer than two bands have
        enough members to score. Returning 0.0 there would silently mean 'the
        shuffle did not beat the observed', which is how a null collapses and a
        t of 0.04 comes back significant."""
        best, scored_bands = 0.0, 0
        for lab in pd.unique(labels):
            sel = labels == lab
            if sel.sum() < 10:
                continue
            t = _tstat(v[sel] - grand)
            if not np.isnan(t):
                scored_bands += 1
                best = max(best, abs(t))
        return best if scored_bands >= 2 else None

    obs = max_abs_t(b)
    if obs is None:
        return {"obs_max_abs_t": float("nan"), "p_fwer": float("nan"),
                "n_perm": float(n_perm), "n_days": float(n_days),
                "note": "fewer than 2 scoreable bands in the observed data"}
    scored = 0
    # one label per day, so the permutation moves whole days
    day_lab = pd.DataFrame({"d": d, "b": b}).groupby("d")["b"].first()
    uniq_days = day_lab.index.to_numpy()
    lab_arr = day_lab.to_numpy()
    day_of = pd.Series(d)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(lab_arr)
        m = dict(zip(uniq_days, perm))
        stat = max_abs_t(day_of.map(m).to_numpy(dtype=object))
        if stat is None:            # a draw that cannot be scored is not evidence
            continue
        if stat >= obs:
            hits += 1
        scored += 1
    if scored < max(200, n_perm // 10):
        return {"obs_max_abs_t": float(obs), "p_fwer": float("nan"),
                "n_perm": float(n_perm), "n_days": float(n_days),
                "note": f"only {scored} of {n_perm} shuffles were scoreable — "
                        "the null is too thin to quote"}
    return {"obs_max_abs_t": float(obs), "n_perm": float(scored),
            "n_days": float(n_days), "note": "",
            "p_fwer": float((hits + 1) / (scored + 1))}
