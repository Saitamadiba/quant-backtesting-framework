"""Session-VWAP developing value-area reversion — backtest engine.

Strategy source: Reddit r/tradingmillionaires (see PREREG.md for the frozen
reading). Developing session VWAP with volume-weighted SD bands: centre line =
"POC", +-1 SD = the "70% value area", +-2 SD = the "95% value area". Price
extends beyond 2 SD, then closes back inside 2 SD, then closes back inside 1 SD
-> fade toward fair value. Stop at the swing extreme of the excursion leg;
targets at the POC and the opposite bands; early exit on failed POC acceptance.

Reuses the anchored-VWAP / session-SD construction from vwap_confluence/engine.py
(2026-07-18) and the TransactionCosts model from the WFO framework.

All features are computed causally (bar i uses only data <= i). Swing-extreme
stops use only bars <= the entry bar; resolution starts at entry+1. Intrabar
resolution is stop-first pessimistic.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

NY = ZoneInfo("America/New_York")
DUCKDB_PATH = os.path.join(_BASE, "duckdb_data", "trading_data.duckdb")
DATA_DIR = os.path.join(_BASE, "research_output", "vwap_value_area")

CRYPTO = ["BTC", "ETH", "SOL", "XRP", "ADA", "BNB", "LINK", "LTC",
          "DOGE", "DOT", "AVAX", "BCH"]

try:
    from fomc_shadow.calendar import is_statement_day as _is_fomc
except Exception:                                          # pragma: no cover
    def _is_fomc(d: date) -> bool:
        return False


# ---------------------------------------------------------------- config ----

@dataclass
class Cfg:
    exec_tf: str = "5m"
    confirm: str = "S2"          # S2 strict two-stage | S1 one-stage | TOUCH
    target: str = "T2"           # T1 POC | T2 opp 1SD | T3 opp 2SD | SCALE
    bands: str = "VWAP"          # VWAP developing session | BB Bollinger control
    k_outer: float = 2.0
    k_inner: float = 1.0
    bb_len: int = 20
    anchor: str = "utc"          # utc 00:00 | cme 18:00 ET | rth 09:30 ET
    use_window: bool = True
    win_start_et: float = 3.0    # Asia close
    win_end_et: float = 16.0     # NY close
    news_blackout: bool = True
    warmup_bars: int = 12
    max_setup_bars: int = 48
    max_hold_bars: int = 96
    max_trades_per_day: int = 3
    stop_buf_atr: float = 0.05
    min_stop_atr: float = 0.15
    rr_max: float = 25.0
    min_rr: float = 0.0        # author implies 1.5 is his selection floor
    swing_bars: int = 0        # 0 = extreme of the whole excursion; K = last K bars only
    poc_reject_exit: bool = True
    # costs
    cost_kind: str = "pct"       # pct (crypto perps) | points (index futures)
    rt_cost_pct: float = 0.0015
    rt_cost_points: float = 1.0


# ------------------------------------------------------------------ data ----

def load_crypto(symbol: str, tf: str) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    df = con.execute(
        "SELECT timestamp, open, high, low, close, volume FROM ohlcv_data "
        "WHERE symbol = ? AND timeframe = ? ORDER BY timestamp", [symbol, tf]).df()
    con.close()
    if df.empty:
        return df
    df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    df["volume"] = df["volume"].clip(lower=0).fillna(0.0)
    return _add_atr(df)


def load_cached(name: str) -> pd.DataFrame:
    p = os.path.join(DATA_DIR, f"{name}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def _add_atr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return df


# ----------------------------------------------------------------- bands ----

def _anchored_vwap(tp, vol, grp):
    out = np.empty(len(tp)); cum_pv = 0.0; cum_v = 0.0; cur = -1
    for i in range(len(tp)):
        if grp[i] != cur:
            cur = grp[i]; cum_pv = 0.0; cum_v = 0.0
        cum_pv += tp[i] * vol[i]; cum_v += vol[i]
        out[i] = cum_pv / cum_v if cum_v > 0 else tp[i]
    return out


def _session_sd(tp, vol, vwap, grp, bar_in_grp, min_bars=3):
    out = np.full(len(tp), np.nan); cum_wv = 0.0; cum_v = 0.0; cur = -1
    for i in range(len(tp)):
        if grp[i] != cur:
            cur = grp[i]; cum_wv = 0.0; cum_v = 0.0
        d = tp[i] - vwap[i]
        cum_wv += vol[i] * d * d; cum_v += vol[i]
        if cum_v > 0 and bar_in_grp[i] >= min_bars:
            out[i] = np.sqrt(cum_wv / cum_v)
    return out


def _sess_groups(ts_utc: pd.Series, anchor: str):
    ts_ny = ts_utc.dt.tz_convert(NY)
    if anchor == "cme":
        key = (ts_ny - pd.Timedelta(hours=18)).dt.date.astype(str)
    elif anchor == "rth":
        key = ts_ny.dt.date.astype(str)
    else:
        key = ts_utc.dt.floor("D").astype(str)
    return pd.factorize(key)[0]


# ------------------------------------------------------------- utilities ----

def _efficiency_ratio(c: np.ndarray, k: int) -> np.ndarray:
    """Trailing k-bar efficiency ratio: |net move| / sum|bar moves|. Causal.
    Low = ranging/choppy, high = trending."""
    d = np.abs(np.diff(c, prepend=c[0]))
    csum = np.convolve(d, np.ones(k), mode="full")[:len(c)]
    net = np.abs(c - np.concatenate([np.full(k, c[0]), c[:-k]]))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(csum > 0, net / csum, np.nan)
    out[:k] = np.nan
    return out


def _is_nfp(d: date) -> bool:
    """First Friday of the month (US non-farm payrolls, 08:30 ET)."""
    return d.weekday() == 4 and d.day <= 7


def _blackout(d: date, hr: int, mi: int) -> bool:
    if _is_fomc(d) or _is_nfp(d):
        return True
    t = hr * 60 + mi
    if 8 * 60 + 25 <= t < 9 * 60:      # 08:30 ET macro release window
        return True
    return False


def _session_label(hr: int) -> str:
    if 0 <= hr < 3:  return "asia_late"
    if 3 <= hr < 8:  return "london"
    if 8 <= hr < 12: return "ny_am"
    if 12 <= hr < 16: return "ny_pm"
    return "asia_early"


# ------------------------------------------------------------- resolution ---

def _resolve(o, h, l, c, mid, start, side, entry, stop, targets, weights,
             max_hold, hard_idx, poc_exit):
    """Stop-first pessimistic intrabar resolution with staged targets and the
    author's failed-POC-acceptance early exit on whatever size remains."""
    risk = (entry - stop) * side
    realized = 0.0; remaining = 1.0; ti = 0
    poc_crossed = False; mae = 0.0; mfe = 0.0
    end = min(len(c) - 1, start + max_hold - 1, hard_idx)
    if end < start:
        return {"gross_r": 0.0, "reason": "nobars", "bars_held": 0,
                "mae_r": 0.0, "mfe_r": 0.0, "n_targets": 0}
    for j in range(start, end + 1):
        bw = ((l[j] - entry) if side == 1 else (entry - h[j])) / risk
        bb = ((h[j] - entry) if side == 1 else (entry - l[j])) / risk
        mae = min(mae, bw); mfe = max(mfe, bb)

        gap_stop = (o[j] <= stop) if side == 1 else (o[j] >= stop)
        if gap_stop:
            realized += remaining * (o[j] - entry) / risk * side
            return {"gross_r": realized, "reason": "stop_gap", "bars_held": j - start + 1,
                    "mae_r": mae, "mfe_r": mfe, "n_targets": ti}
        hit_stop = (l[j] <= stop) if side == 1 else (h[j] >= stop)
        if hit_stop:
            realized += remaining * -1.0
            return {"gross_r": realized, "reason": "stop", "bars_held": j - start + 1,
                    "mae_r": min(mae, -1.0), "mfe_r": mfe, "n_targets": ti}

        while ti < len(targets):
            tg = targets[ti]
            hit_tp = (h[j] >= tg) if side == 1 else (l[j] <= tg)
            if not hit_tp:
                break
            realized += weights[ti] * (tg - entry) / risk * side
            remaining -= weights[ti]; ti += 1
        if remaining <= 1e-9:
            return {"gross_r": realized, "reason": "tp", "bars_held": j - start + 1,
                    "mae_r": mae, "mfe_r": mfe, "n_targets": ti}

        if poc_exit:
            m = mid[j]
            if np.isfinite(m):
                if side == 1:
                    if h[j] > m: poc_crossed = True
                    if poc_crossed and c[j] < m:
                        realized += remaining * (c[j] - entry) / risk * side
                        return {"gross_r": realized, "reason": "poc_reject",
                                "bars_held": j - start + 1, "mae_r": mae,
                                "mfe_r": mfe, "n_targets": ti}
                else:
                    if l[j] < m: poc_crossed = True
                    if poc_crossed and c[j] > m:
                        realized += remaining * (c[j] - entry) / risk * side
                        return {"gross_r": realized, "reason": "poc_reject",
                                "bars_held": j - start + 1, "mae_r": mae,
                                "mfe_r": mfe, "n_targets": ti}
    realized += remaining * (c[end] - entry) / risk * side
    return {"gross_r": realized, "reason": "time", "bars_held": end - start + 1,
            "mae_r": mae, "mfe_r": mfe, "n_targets": ti}


# ------------------------------------------------------------------- run ----

def run_symbol(symbol: str, cfg: Cfg, df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_crypto(symbol, cfg.exec_tf)
    if df is None or df.empty or len(df) < 500:
        return pd.DataFrame()
    df = df.reset_index(drop=True)

    ts = df["timestamp"]
    ts_ny = ts.dt.tz_convert(NY)
    hr = ts_ny.dt.hour.to_numpy(); mi = ts_ny.dt.minute.to_numpy()
    dts = np.array(ts_ny.dt.date)
    dow = ts_ny.dt.dayofweek.to_numpy()

    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float);  c = df["close"].to_numpy(float)
    vol = df["volume"].to_numpy(float); atr = df["atr"].to_numpy(float)
    n = len(df)

    sess = _sess_groups(ts, cfg.anchor)
    bar_in_sess = df.groupby(sess).cumcount().to_numpy()
    last_in_sess = pd.Series(np.arange(n)).groupby(sess).transform("max").to_numpy()

    if cfg.bands == "VWAP":
        tp = (h + l + c) / 3.0
        mid = _anchored_vwap(tp, vol, sess)
        sd = _session_sd(tp, vol, mid, sess, bar_in_sess)
    else:                                              # Bollinger control
        s = pd.Series(c)
        mid = s.rolling(cfg.bb_len).mean().to_numpy()
        sd = s.rolling(cfg.bb_len).std(ddof=0).to_numpy()

    er20 = _efficiency_ratio(c, 20)
    er60 = _efficiency_ratio(c, 60)
    vol20 = pd.Series(vol).rolling(20).mean().to_numpy()

    ko, ki = cfg.k_outer, cfg.k_inner
    trades = []
    cur_sess = -1; state = 0; side_setup = 0; swing = np.nan; arm_i = -1
    day_n = 0; busy_until = -1
    n_drop_wrongside = 0; n_drop_rr = 0; n_setups = 0

    for i in range(n):
        if sess[i] != cur_sess:
            cur_sess = sess[i]; state = 0; side_setup = 0; day_n = 0; swing = np.nan
        if i <= busy_until:
            continue
        if bar_in_sess[i] < cfg.warmup_bars:
            continue
        m, s_ = mid[i], sd[i]; a = atr[i]
        if not (np.isfinite(m) and np.isfinite(s_) and s_ > 0 and np.isfinite(a) and a > 0):
            continue
        u2, u1, d1, d2 = m + ko * s_, m + ki * s_, m - ki * s_, m - ko * s_
        ci = c[i]

        # ---------------- TOUCH control: resting limit at the outer band -----
        if cfg.confirm == "TOUCH":
            for sd_side, band, ext in ((-1, u2, h[i]), (1, d2, l[i])):
                hit = (h[i] >= band) if sd_side == -1 else (l[i] <= band)
                if not hit or day_n >= cfg.max_trades_per_day:
                    continue
                if not _entry_allowed(cfg, dts[i], hr[i], mi[i]):
                    continue
                entry = band
                stop = (ext + cfg.stop_buf_atr * a) if sd_side == -1 else (ext - cfg.stop_buf_atr * a)
                r = _emit_trade(trades, symbol, cfg, df, i, sd_side, entry, stop, m, s_, a,
                                o, h, l, c, mid, last_in_sess[i], hr, mi, dts, dow,
                                er20, er60, vol, vol20, bar_in_sess, sd, 0, np.nan)
                if r is not None:
                    busy_until = i + r; day_n += 1
                break
            continue

        # ---------------- state machine: arm beyond the outer band -----------
        if ci > u2:
            if side_setup != -1:
                side_setup = -1; swing = h[i]; arm_i = i; n_setups += 1
            else:
                swing = max(swing, h[i])
            state = 1
            continue
        if ci < d2:
            if side_setup != 1:
                side_setup = 1; swing = l[i]; arm_i = i; n_setups += 1
            else:
                swing = min(swing, l[i])
            state = 1
            continue
        if side_setup == 0 or state == 0:
            continue
        if i - arm_i > cfg.max_setup_bars:
            side_setup = 0; state = 0
            continue
        swing = max(swing, h[i]) if side_setup == -1 else min(swing, l[i])

        inside_outer = (ci < u2) if side_setup == -1 else (ci > d2)
        inside_inner = (ci < u1) if side_setup == -1 else (ci > d1)

        enter = False
        if cfg.confirm == "S2":
            if state == 1 and inside_outer:
                state = 2                      # "closes inside the 95% level"
            elif state == 2 and inside_inner:
                enter = True                   # "closes back inside the 70%"
        else:                                   # S1 permissive
            if inside_inner:
                enter = True
        if not enter:
            continue
        if day_n >= cfg.max_trades_per_day:
            side_setup = 0; state = 0
            continue
        if not _entry_allowed(cfg, dts[i], hr[i], mi[i]):
            continue

        entry = ci
        sw = swing
        if cfg.swing_bars > 0:
            j0 = max(0, i - cfg.swing_bars + 1)
            sw = h[j0:i + 1].max() if side_setup == -1 else l[j0:i + 1].min()
        stop = (sw + cfg.stop_buf_atr * a) if side_setup == -1 else (sw - cfg.stop_buf_atr * a)
        # must still be on the far side of fair value for the reversion premise
        if (m - entry) * side_setup <= 0:
            n_drop_wrongside += 1; side_setup = 0; state = 0
            continue
        r = _emit_trade(trades, symbol, cfg, df, i, side_setup, entry, stop, m, s_, a,
                        o, h, l, c, mid, last_in_sess[i], hr, mi, dts, dow,
                        er20, er60, vol, vol20, bar_in_sess, sd, i - arm_i, sw)
        if r is None:
            n_drop_rr += 1
        else:
            busy_until = i + r; day_n += 1
        side_setup = 0; state = 0

    out = pd.DataFrame(trades)
    if not out.empty:
        out.attrs["n_setups"] = n_setups
        out.attrs["n_drop_wrongside"] = n_drop_wrongside
        out.attrs["n_drop_rr"] = n_drop_rr
    return out


def _entry_allowed(cfg: Cfg, d, hour, minute) -> bool:
    if cfg.use_window:
        t = hour + minute / 60.0
        if not (cfg.win_start_et <= t < cfg.win_end_et):
            return False
    if cfg.news_blackout and _blackout(d, int(hour), int(minute)):
        return False
    return True


def _emit_trade(trades, symbol, cfg, df, i, side, entry, stop, m, s_, a,
                o, h, l, c, mid, hard_idx, hr, mi, dts, dow, er20, er60,
                vol, vol20, bar_in_sess, sd, setup_bars, swing):
    risk = (entry - stop) * side
    if risk <= 0:
        return None
    if risk < cfg.min_stop_atr * a:
        risk = cfg.min_stop_atr * a
        stop = entry - side * risk

    # targets run from the entry side THROUGH fair value to the opposite bands
    t1 = m
    t2 = m + side * cfg.k_inner * s_
    t3 = m + side * cfg.k_outer * s_
    if cfg.target == "T1":
        targets, weights = [t1], [1.0]
    elif cfg.target == "T2":
        targets, weights = [t2], [1.0]
    elif cfg.target == "T3":
        targets, weights = [t3], [1.0]
    else:
        targets, weights = [t1, t2, t3], [1 / 3, 1 / 3, 1 / 3]

    final = targets[-1]
    if (final - entry) * side <= 0:
        return None
    rr = abs(final - entry) / risk
    if rr > cfg.rr_max or rr < cfg.min_rr:
        return None

    res = _resolve(o, h, l, c, mid, i + 1, side, entry, stop, targets, weights,
                   cfg.max_hold_bars, hard_idx, cfg.poc_reject_exit)
    if res["reason"] == "nobars":
        return None

    stop_pct = risk / entry
    if cfg.cost_kind == "points":
        fee_r = cfg.rt_cost_points / risk
    else:
        fee_r = cfg.rt_cost_pct / stop_pct

    trades.append({
        "symbol": symbol, "confirm": cfg.confirm, "target": cfg.target,
        "bands": cfg.bands, "anchor": cfg.anchor,
        "date": dts[i], "entry_time": df["timestamp"].iloc[i],
        "side": "long" if side == 1 else "short",
        "entry": entry, "stop": stop, "tp_final": final,
        "gross_r": res["gross_r"], "fee_r": fee_r,
        "net_r": res["gross_r"] - fee_r,
        "exit_reason": res["reason"], "bars_held": res["bars_held"],
        "n_targets_hit": res["n_targets"],
        "mae_r": res["mae_r"], "mfe_r": res["mfe_r"],
        # ---- PRE-fill features (legitimate gates) ----
        "pre_sd_pos": (entry - m) / s_,
        "pre_band_w_pct": s_ / entry,
        "pre_stop_atr": risk / a,
        "pre_stop_pct": stop_pct,
        "pre_stop_pts": risk,
        "pre_rr_planned": rr,
        "pre_er20": er20[i], "pre_er60": er60[i],
        "pre_atr_pct": a / entry,
        "pre_setup_bars": setup_bars,
        "pre_excursion_sd": abs((swing - m) / s_) if np.isfinite(swing) else np.nan,
        "pre_vol_ratio": vol[i] / vol20[i] if np.isfinite(vol20[i]) and vol20[i] > 0 else np.nan,
        "pre_bars_since_anchor": int(bar_in_sess[i]),
        "pre_hour_et": int(hr[i]), "pre_dow": int(dow[i]),
        "pre_session": _session_label(int(hr[i])),
        # ---- POST-fill (diagnosis only, never a gate) ----
        "post_bars_held": res["bars_held"],
    })
    return max(1, res["bars_held"])
