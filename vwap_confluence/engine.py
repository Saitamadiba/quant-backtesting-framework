"""VWAP Confluence (Tom Crown) — backtest engine.

VWAP = dynamic fair value (volume-weighted, session/weekly/monthly anchored).
Two mechanisms: (A/B/C) pull back to VWAP in a trend and buy the hold with a
confirmation candle (VWAP as dynamic support); (D) fade price stretched from
VWAP back to fair value (SD-band reversion). An MA-control arm swaps VWAP for
a plain EMA to test whether VWAP-anchoring is special.

Session/weekly/monthly VWAP + session SD bands are computed causally from
typical-price×volume cumulative sums. OHLC intrabar resolution, stop-first
pessimistic. All features PRE-fill causal. See SPEC.md.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from backtrader_framework.optimization.wfo_engine import TransactionCosts  # noqa: E402

DUCKDB_PATH = os.path.join(_BASE, "duckdb_data", "trading_data.duckdb")
NY = ZoneInfo("America/New_York")


@dataclass
class VWAPConfig:
    exec_tf: str = "5m"
    arm: str = "A"                  # A pullback | B +HTF | C +weekly | MA control | D reversion
    ma_len: int = 100              # EMA length for the MA-control arm
    ext_atr: float = 0.6           # min extension above VWAP (ATR) before a pullback qualifies
    slope_bars: int = 12           # VWAP slope lookback
    touch_atr: float = 0.15        # how close to VWAP counts as a touch (ATR)
    rr: float = 2.0
    target: str = "fixed"          # "fixed" | "swing"
    sd_k: float = 2.0              # reversion arm: SD distance to fade
    stop_buffer_atr: float = 0.10
    min_stop_atr: float = 0.10
    rr_max: float = 25.0
    max_setup_bars: int = 60
    max_hold_bars: int = 96
    max_trades_per_day: int = 3
    warmup_bars: int = 6           # bars after session anchor before trading
    session_anchor: str = "utc"    # "utc" (00:00 UTC) | "ny" (09:30 ET, NQ)
    costs: TransactionCosts = field(default_factory=TransactionCosts)


def load_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    df = con.execute(
        "SELECT timestamp, open, high, low, close, volume, atr_14, ema_200 "
        "FROM ohlcv_data WHERE symbol = ? AND timeframe = ? ORDER BY timestamp",
        [symbol, timeframe],
    ).df()
    con.close()
    if df.empty:
        return df
    df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
    df["timestamp"] = (pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
                       .astype("datetime64[ns, UTC]"))
    df["volume"] = df["volume"].clip(lower=0).fillna(0.0)
    return df


def _tf_minutes(tf: str) -> int:
    return {"5m": 5, "15m": 15, "1h": 60}[tf]


def _anchored_vwap(tp: np.ndarray, vol: np.ndarray, grp: np.ndarray):
    """Cumulative VWAP within each anchor group `grp` (monotone int id)."""
    pv = tp * vol
    out = np.empty(len(tp))
    cum_pv = 0.0; cum_v = 0.0; cur = -1
    for i in range(len(tp)):
        if grp[i] != cur:
            cur = grp[i]; cum_pv = 0.0; cum_v = 0.0
        cum_pv += pv[i]; cum_v += vol[i]
        out[i] = cum_pv / cum_v if cum_v > 0 else tp[i]
    return out


def _session_sd(tp: np.ndarray, vol: np.ndarray, vwap: np.ndarray, grp: np.ndarray,
                bar_in_grp: np.ndarray):
    """Volume-weighted running std of (tp - vwap) within each session anchor."""
    out = np.full(len(tp), np.nan)
    cum_wv = 0.0; cum_v = 0.0; cur = -1
    for i in range(len(tp)):
        if grp[i] != cur:
            cur = grp[i]; cum_wv = 0.0; cum_v = 0.0
        d = tp[i] - vwap[i]
        cum_wv += vol[i] * d * d; cum_v += vol[i]
        if cum_v > 0 and bar_in_grp[i] >= 3:
            out[i] = np.sqrt(cum_wv / cum_v)
    return out


def run_symbol(symbol: str, cfg: VWAPConfig | None = None) -> pd.DataFrame:
    cfg = cfg or VWAPConfig(costs=TransactionCosts.for_asset(symbol))
    df = load_bars(symbol, cfg.exec_tf)
    if df.empty:
        return pd.DataFrame()

    tf_min = _tf_minutes(cfg.exec_tf)
    ts_ny = df["timestamp"].dt.tz_convert(NY)
    ts_utc = df["timestamp"]
    df["ny_hour"] = ts_ny.dt.hour
    df["ny_min"] = ts_ny.dt.minute
    df["dow"] = ts_ny.dt.dayofweek

    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy(float)
    vol = df["volume"].to_numpy(float)

    # anchor ids
    if cfg.session_anchor == "ny" or symbol == "NQ":
        sess_key = ts_ny.dt.date.astype(str)
        sess_flat_ref = "ny"
    else:
        sess_key = ts_utc.dt.floor("D").astype(str)
        sess_flat_ref = "utc"
    sess_grp = pd.factorize(sess_key)[0]
    iso = ts_utc.dt.isocalendar()
    week_grp = pd.factorize((iso.year.astype(str) + "-" + iso.week.astype(str)))[0]
    month_grp = pd.factorize(ts_utc.dt.strftime("%Y-%m"))[0]

    bar_in_sess = df.groupby(sess_grp).cumcount().to_numpy()

    svwap = _anchored_vwap(tp, vol, sess_grp)
    wvwap = _anchored_vwap(tp, vol, week_grp)
    mvwap = _anchored_vwap(tp, vol, month_grp)
    ssd = _session_sd(tp, vol, svwap, sess_grp, bar_in_sess)

    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    atr = df["atr_14"].to_numpy(float); ema200 = df["ema_200"].to_numpy(float)
    ema_ma = pd.Series(c).ewm(span=cfg.ma_len, adjust=False).mean().to_numpy()
    ny_hour = df["ny_hour"].to_numpy(int); dow_arr = df["dow"].to_numpy(int)
    vol20 = pd.Series(vol).rolling(20).mean().to_numpy()
    n = len(df)

    # the "fair value" line the arm anchors to
    line = ema_ma if cfg.arm == "MA" else svwap
    # VWAP slope over slope_bars (in ATR units per bar)
    slope = np.full(n, np.nan)
    sb = cfg.slope_bars
    slope[sb:] = (line[sb:] - line[:-sb]) / sb

    # session flat index: last bar of each session
    last_in_sess = {}
    for i in range(n):
        last_in_sess[sess_grp[i]] = i

    trades: list[dict] = []
    if cfg.arm == "D":
        _run_reversion(trades, symbol, cfg, df, tf_min, o, h, l, c, atr, svwap,
                       ssd, ema200, ny_hour, dow_arr, vol, vol20, sess_grp,
                       bar_in_sess, last_in_sess, wvwap, mvwap, slope, n)
    else:
        _run_pullback(trades, symbol, cfg, df, tf_min, o, h, l, c, atr, line,
                      svwap, wvwap, mvwap, ssd, ema200, ny_hour, dow_arr, vol,
                      vol20, sess_grp, bar_in_sess, last_in_sess, slope, n)
    return pd.DataFrame(trades)


def _confirm(side, i, o, h, l, c):
    """Bullish/bearish confirmation candle at index i."""
    pj = i - 1
    if side == 1:
        engulf = (c[i] > o[i]) and (c[pj] < o[pj]) and (c[i] >= o[pj]) and (o[i] <= c[pj])
        rng = h[i] - l[i]
        hammer = rng > 0 and (min(o[i], c[i]) - l[i]) >= 2 * abs(c[i] - o[i]) and (h[i] - max(o[i], c[i])) <= 0.4 * rng
        reclaim = (c[i] > o[i]) and (c[i] > h[pj])
        return engulf or hammer or reclaim
    else:
        engulf = (c[i] < o[i]) and (c[pj] > o[pj]) and (c[i] <= o[pj]) and (o[i] >= c[pj])
        rng = h[i] - l[i]
        star = rng > 0 and (h[i] - max(o[i], c[i])) >= 2 * abs(c[i] - o[i]) and (min(o[i], c[i]) - l[i]) <= 0.4 * rng
        reclaim = (c[i] < o[i]) and (c[i] < l[pj])
        return engulf or star or reclaim


def _run_pullback(trades, symbol, cfg, df, tf_min, o, h, l, c, atr, line, svwap,
                  wvwap, mvwap, ssd, ema200, ny_hour, dow_arr, vol, vol20,
                  sess_grp, bar_in_sess, last_in_sess, slope, n):
    state = {1: {"phase": "idle"}, -1: {"phase": "idle"}}
    day_trades = {}
    for i in range(1, n):
        if bar_in_sess[i] < cfg.warmup_bars:
            state = {1: {"phase": "idle"}, -1: {"phase": "idle"}}
            continue
        a = atr[i]
        if not (np.isfinite(a) and a > 0 and np.isfinite(line[i]) and np.isfinite(slope[i])):
            continue
        sg = sess_grp[i]
        dt = day_trades.get(sg, 0)
        for side in (1, -1):
            st = state[side]
            lv = line[i]
            bias = (c[i] > lv) if side == 1 else (c[i] < lv)
            slope_ok = (slope[i] >= 0) if side == 1 else (slope[i] <= 0)
            dist = (c[i] - lv) * side / a  # signed distance in ATR, + = with bias
            ph = st["phase"]
            if ph == "idle":
                # need to be extended in the bias direction to set up a pullback
                if bias and slope_ok and dist >= cfg.ext_atr:
                    st.clear(); st.update(phase="extended", ext_i=i, ext_dist=dist,
                                          ext_px=c[i])
                continue
            if i - st["ext_i"] > cfg.max_setup_bars:
                state[side] = {"phase": "idle"}; continue
            st["ext_dist"] = max(st["ext_dist"], dist)
            st["ext_px"] = max(st["ext_px"], c[i]) if side == 1 else min(st["ext_px"], c[i])
            # pullback: price touches the line (within touch_atr) while holding it
            touched = (l[i] <= lv + cfg.touch_atr * a) if side == 1 else (h[i] >= lv - cfg.touch_atr * a)
            holds = (c[i] >= lv) if side == 1 else (c[i] <= lv)
            if ph == "extended":
                if touched:
                    st["phase"] = "pulled"; st["pb_ext"] = l[i] if side == 1 else h[i]
                    st["pb_i"] = i
                elif not bias:  # lost the line before pulling back cleanly
                    state[side] = {"phase": "idle"}
                continue
            if ph == "pulled":
                st["pb_ext"] = min(st["pb_ext"], l[i]) if side == 1 else max(st["pb_ext"], h[i])
                if not holds and ((c[i] < lv - cfg.ext_atr * a) if side == 1 else (c[i] > lv + cfg.ext_atr * a)):
                    state[side] = {"phase": "idle"}; continue  # broke through the line
                if dt >= cfg.max_trades_per_day:
                    continue
                if _confirm(side, i, o, h, l, c):
                    # confluence gates
                    if cfg.arm == "B" and not ((c[i] > ema200[i]) if side == 1 else (c[i] < ema200[i])):
                        continue
                    if cfg.arm == "C" and not ((c[i] > wvwap[i]) if side == 1 else (c[i] < wvwap[i])):
                        continue
                    _book_pull(trades, symbol, cfg, df, tf_min, side, i, o, h, l, c,
                               atr, line, svwap, wvwap, mvwap, ssd, ema200, ny_hour,
                               dow_arr, vol, vol20, sess_grp, bar_in_sess,
                               last_in_sess, slope, st, dt)
                    day_trades[sg] = dt + 1
                    state[side] = {"phase": "idle"}
                continue


def _book_pull(trades, symbol, cfg, df, tf_min, side, i, o, h, l, c, atr, line,
               svwap, wvwap, mvwap, ssd, ema200, ny_hour, dow_arr, vol, vol20,
               sess_grp, bar_in_sess, last_in_sess, slope, st, dt):
    a = atr[i]
    entry = c[i]
    pb_ext = st["pb_ext"]
    stop = (pb_ext - cfg.stop_buffer_atr * a) if side == 1 else (pb_ext + cfg.stop_buffer_atr * a)
    risk = (entry - stop) * side
    if risk <= 0:
        return
    if risk < cfg.min_stop_atr * a:
        risk = cfg.min_stop_atr * a
        stop = entry - side * risk
    if cfg.target == "swing":
        tp = st.get("ext_px", entry + side * cfg.rr * risk)
        if (tp - entry) * side <= 0:
            tp = entry + side * cfg.rr * risk
    else:
        tp = entry + side * cfg.rr * risk
    if abs(tp - entry) / risk > cfg.rr_max:
        return
    flat_idx = last_in_sess[sess_grp[i]]
    res = _resolve(o, h, l, c, i + 1, side, entry, stop, tp, cfg.max_hold_bars, flat_idx)
    _emit(trades, symbol, cfg, df, tf_min, side, i, entry, stop, tp, risk, res,
          a, line, svwap, wvwap, mvwap, ssd, ema200, ny_hour, dow_arr, vol, vol20,
          bar_in_sess, slope, st, dt, kind="pull")


def _run_reversion(trades, symbol, cfg, df, tf_min, o, h, l, c, atr, svwap, ssd,
                   ema200, ny_hour, dow_arr, vol, vol20, sess_grp, bar_in_sess,
                   last_in_sess, wvwap, mvwap, slope, n):
    day_trades = {}
    for i in range(1, n):
        if bar_in_sess[i] < cfg.warmup_bars:
            continue
        a = atr[i]; sd = ssd[i]
        if not (np.isfinite(a) and a > 0 and np.isfinite(sd) and sd > 0 and np.isfinite(svwap[i])):
            continue
        sg = sess_grp[i]
        dt = day_trades.get(sg, 0)
        if dt >= cfg.max_trades_per_day:
            continue
        sd_pos = (c[i] - svwap[i]) / sd
        # stretched ABOVE +k SD -> fade short toward VWAP; below -k -> fade long
        for side, cond in ((-1, sd_pos >= cfg.sd_k), (1, sd_pos <= -cfg.sd_k)):
            if not cond:
                continue
            if not _confirm(side, i, o, h, l, c):
                continue
            entry = c[i]
            ext = h[i] if side == -1 else l[i]
            stop = (ext + cfg.stop_buffer_atr * a) if side == -1 else (ext - cfg.stop_buffer_atr * a)
            risk = (entry - stop) * side
            if risk <= 0:
                continue
            if risk < cfg.min_stop_atr * a:
                risk = cfg.min_stop_atr * a; stop = entry - side * risk
            tp = svwap[i]  # target = fair value
            if (tp - entry) * side <= 0 or abs(tp - entry) / risk > cfg.rr_max:
                continue
            flat_idx = last_in_sess[sg]
            res = _resolve(o, h, l, c, i + 1, side, entry, stop, tp, cfg.max_hold_bars, flat_idx)
            st = {"ext_dist": abs(sd_pos), "ext_px": ext, "pb_i": i, "ext_i": i}
            _emit(trades, symbol, cfg, df, tf_min, side, i, entry, stop, tp, risk,
                  res, a, svwap, svwap, wvwap, mvwap, ssd, ema200, ny_hour, dow_arr,
                  vol, vol20, bar_in_sess, slope, st, dt, kind="rev")
            day_trades[sg] = dt + 1
            break


def _session(hr):
    if 0 <= hr < 6: return "asia"
    if 6 <= hr < 9: return "london"
    if 9 <= hr < 12: return "ny_am"
    if 12 <= hr < 16: return "ny_pm"
    return "late"


def _emit(trades, symbol, cfg, df, tf_min, side, i, entry, stop, tp, risk, res, a,
          line, svwap, wvwap, mvwap, ssd, ema200, ny_hour, dow_arr, vol, vol20,
          bar_in_sess, slope, st, dt, kind):
    stop_pct = risk / entry
    fee_taker = cfg.costs.round_trip_cost_pct / stop_pct
    half = cfg.costs.round_trip_cost_pct / 2.0
    fee_mm = ((half + 0.0002) / stop_pct) if res["reason"] in ("tp", "tp_gap") else fee_taker
    em = ema200[i]; sd = ssd[i]
    trades.append({
        "symbol": symbol, "arm": cfg.arm, "kind": kind,
        "ny_date": df["timestamp"].iloc[i].tz_convert(NY).date(),
        "entry_time": df["timestamp"].iloc[i] + pd.Timedelta(minutes=tf_min),
        "side": "long" if side == 1 else "short",
        "entry": entry, "stop": stop, "tp": tp,
        "gross_r": res["gross_r"], "net_taker_r": res["gross_r"] - fee_taker,
        "net_mm_r": res["gross_r"] - fee_mm, "fee_taker_r": fee_taker,
        "exit_reason": res["reason"], "bars_held": res["bars_held"],
        "mae_r": res["mae_r"], "mfe_r": res["mfe_r"],
        "same_day_resolve": res["reason"] not in ("time",),
        "session": _session(int(ny_hour[i])), "entry_hour_ny": int(ny_hour[i]),
        "dow": int(dow_arr[i]),
        "vwap_dist_atr": (entry - svwap[i]) / a * side,
        "vwap_slope_atr": slope[i] / a if np.isfinite(slope[i]) else np.nan,
        "pullback_depth_atr": float(st.get("ext_dist", np.nan)),
        "retrace_frac": (abs(entry - st.get("ext_px", entry)) / (abs(st.get("ext_px", entry) - svwap[i]) + 1e-9))
            if kind == "pull" else np.nan,
        "weekly_align": bool((entry > wvwap[i]) if side == 1 else (entry < wvwap[i])),
        "monthly_align": bool((entry > mvwap[i]) if side == 1 else (entry < mvwap[i])),
        "htf_trend_align": bool((entry > em) if side == 1 else (entry < em)) if np.isfinite(em) else None,
        "sd_pos": (entry - svwap[i]) / sd if np.isfinite(sd) and sd > 0 else np.nan,
        "confirm_body_ratio": abs(df["close"].iloc[i] - df["open"].iloc[i])
            / (abs(df["close"].iloc[i - 1] - df["open"].iloc[i - 1]) + 1e-9),
        "dist_ema200_pct": (entry - em) / em * side if np.isfinite(em) else np.nan,
        "stop_atr": risk / a, "stop_pct": stop_pct, "fee_r": fee_taker,
        "rr_planned": abs(tp - entry) / risk,
        "bars_since_anchor": int(bar_in_sess[i]),
        "break_vol_ratio": (vol[i] / vol20[i]) if np.isfinite(vol20[i]) and vol20[i] > 0 else np.nan,
        "prior_trades_today": int(dt),
    })


def _resolve(o, h, l, c, start, side, entry, stop, tp, max_hold, hard_idx):
    risk = (entry - stop) * side
    mae = 0.0; mfe = 0.0
    end = min(len(c) - 1, start + max_hold - 1, hard_idx)
    for j in range(start, end + 1):
        if side == 1:
            bw = (l[j] - entry) / risk; bb = (h[j] - entry) / risk
        else:
            bw = (entry - h[j]) / risk; bb = (entry - l[j]) / risk
        mae = min(mae, bw); mfe = max(mfe, bb)
        gap_stop = (o[j] <= stop) if side == 1 else (o[j] >= stop)
        gap_tp = (o[j] >= tp) if side == 1 else (o[j] <= tp)
        hit_stop = (l[j] <= stop) if side == 1 else (h[j] >= stop)
        hit_tp = (h[j] >= tp) if side == 1 else (l[j] <= tp)
        if gap_stop:
            return {"gross_r": (o[j] - entry) / risk * side, "reason": "stop_gap",
                    "bars_held": j - start + 1, "mae_r": mae, "mfe_r": mfe}
        if gap_tp:
            return {"gross_r": (o[j] - entry) / risk * side, "reason": "tp_gap",
                    "bars_held": j - start + 1, "mae_r": mae, "mfe_r": mfe}
        if hit_stop:
            return {"gross_r": -1.0, "reason": "stop", "bars_held": j - start + 1,
                    "mae_r": -1.0, "mfe_r": mfe}
        if hit_tp:
            return {"gross_r": (tp - entry) / risk * side, "reason": "tp",
                    "bars_held": j - start + 1, "mae_r": mae, "mfe_r": mfe}
    j = end
    return {"gross_r": (c[j] - entry) / risk * side, "reason": "time",
            "bars_held": end - start + 1, "mae_r": mae, "mfe_r": mfe}
