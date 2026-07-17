"""Opening-Range Breakout + FVG-retest + engulfing — backtest engine.

Faithful conversion of the "only scalping strategy" transcript: mark the
first opening bar's high/low as the range, wait for a body close beyond it
(breakout), a Fair Value Gap in the break direction (displacement), a
retest back into that gap, and an engulfing candle (order-flow shift) as
the entry trigger. Stop = one tick beyond the engulfed (retest) candle;
target = fixed 3R. Passive management (no trail / no partials); max 2
trades per NY day; flat by session end.

OHLC intrabar resolution, stop-first on ties (pessimistic). Every recorded
feature is entry-time causal (PRE-fill). See SPEC.md for every rule and
ambiguity resolution.
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
class ORBConfig:
    exec_tf: str = "5m"
    open_hour_ny: int = 9
    open_min_ny: int = 30
    flat_hour_ny: int = 16          # flatten open positions at/after this NY hour
    entry_cutoff_hour_ny: int = 15  # no NEW entries at/after this NY hour+min
    entry_cutoff_min_ny: int = 30
    rr: float = 3.0                 # fixed take-profit multiple (faithful)
    max_trades_per_day: int = 2     # faithful
    max_setup_bars: int = 78        # reset a side if no entry within N bars of breakout
    max_hold_bars: int = 96         # safety cap on the resolver (session bounds it too)
    fvg_min_atr: float = 0.0        # min FVG height in units of ATR15 (0 = off)
    stop_buffer_atr: float = 0.0    # "one tick" beyond engulfed candle, in ATR15 (0 = literal)
    require_engulf: bool = True     # False = enter on first retest close (no-confirm arm)
    session_flat: bool = True       # flatten at session end (intraday, faithful)
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
    return df


def _tf_minutes(tf: str) -> int:
    return {"5m": 5, "15m": 15, "1h": 60}[tf]


_ATR15_CACHE: dict[str, pd.DataFrame] = {}


def _atr15_series(symbol: str) -> pd.DataFrame:
    """Wilder ATR14 on 15m bars, indexed by bar CLOSE time (causal)."""
    if symbol in _ATR15_CACHE:
        return _ATR15_CACHE[symbol]
    df = load_bars(symbol, "15m")
    if df.empty:
        _ATR15_CACHE[symbol] = pd.DataFrame({"time": [], "atr15": []})
        return _ATR15_CACHE[symbol]
    hh, ll, cc = df["high"], df["low"], df["close"]
    pc = cc.shift(1)
    tr = pd.concat([(hh - ll), (hh - pc).abs(), (ll - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    out = pd.DataFrame({"time": df["timestamp"] + pd.Timedelta(minutes=15),
                        "atr15": atr.to_numpy()})
    _ATR15_CACHE[symbol] = out
    return out


def _session(hr: int) -> str:
    if 0 <= hr < 6:
        return "asia"
    if 6 <= hr < 9:
        return "london"
    if 9 <= hr < 12:
        return "ny_am"
    if 12 <= hr < 16:
        return "ny_pm"
    return "late"


def run_symbol(symbol: str, cfg: ORBConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ORBConfig(costs=TransactionCosts.for_asset(symbol))
    df = load_bars(symbol, cfg.exec_tf)
    if df.empty:
        return pd.DataFrame()

    tf_min = _tf_minutes(cfg.exec_tf)
    ts_ny = df["timestamp"].dt.tz_convert(NY)
    df["ny_date"] = ts_ny.dt.date
    df["ny_hour"] = ts_ny.dt.hour
    df["ny_min"] = ts_ny.dt.minute
    df["dow"] = ts_ny.dt.dayofweek
    df["vol20"] = df["volume"].rolling(20).mean()

    atr15 = _atr15_series(symbol)
    df["atr15"] = pd.merge_asof(
        df[["timestamp"]], atr15, left_on="timestamp", right_on="time",
        direction="backward",
    )["atr15"].to_numpy()

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    vol20 = df["vol20"].to_numpy(float)
    atr15v = df["atr15"].to_numpy(float)
    ema200 = df["ema_200"].to_numpy(float)
    ny_hour = df["ny_hour"].to_numpy(int)
    ny_min = df["ny_min"].to_numpy(int)
    dow_arr = df["dow"].to_numpy(int)
    tmin_arr = ny_hour * 60 + ny_min  # NY minutes since midnight

    open_mins = cfg.open_hour_ny * 60 + cfg.open_min_ny
    flat_mins = cfg.flat_hour_ny * 60
    cutoff_mins = cfg.entry_cutoff_hour_ny * 60 + cfg.entry_cutoff_min_ny

    trades: list[dict] = []
    for day, idx in df.groupby("ny_date").indices.items():
        idx = np.asarray(idx)
        # opening bar = the exec bar that STARTS exactly at the session open
        or_pos = np.where(tmin_arr[idx] == open_mins)[0]
        if len(or_pos) == 0:
            continue
        or_i = int(idx[or_pos[0]])
        rhi = h[or_i]
        rlo = l[or_i]
        if not (np.isfinite(rhi) and np.isfinite(rlo)) or rhi <= rlo:
            continue
        or_sz = rhi - rlo
        a15_or = atr15v[or_i]

        # tradeable bars: after OR bar, before flat time, same day
        trad_idx = idx[(idx > or_i) & (tmin_arr[idx] < flat_mins)]
        if len(trad_idx) < 3:
            continue
        flat_idx = int(trad_idx.max())  # last bar we may hold to (mark-to-market here)

        day_trades = 0
        # per-side state machine. side: +1 long (break UP), -1 short (break DOWN)
        state = {1: {"phase": "idle"}, -1: {"phase": "idle"}}

        for i in trad_idx:
            if day_trades >= cfg.max_trades_per_day:
                break
            ci, oi, hi, li = c[i], o[i], h[i], l[i]
            can_enter = tmin_arr[i] < cutoff_mins

            for side in (1, -1):
                st = state[side]
                lvl = rhi if side == 1 else rlo
                opp = rlo if side == 1 else rhi
                broke = (ci > lvl) if side == 1 else (ci < lvl)
                broke_opp = (ci < opp) if side == 1 else (ci > opp)

                ph = st["phase"]
                if ph == "idle":
                    if broke:
                        st.clear()
                        st["phase"] = "seek_fvg"
                        st["broke_i"] = i
                        st["break_margin"] = (ci - lvl) * side
                    continue

                # armed on some phase — global resets
                if broke_opp:
                    state[side] = {"phase": "idle"}
                    continue
                if i - st["broke_i"] > cfg.max_setup_bars:
                    state[side] = {"phase": "idle"}
                    continue

                if ph == "seek_fvg":
                    # need i-2 valid and at/after breakout; FVG gap on breakout side
                    if i - 2 < st["broke_i"]:
                        continue
                    if side == 1:
                        gap = li - h[i - 2]          # bullish: low[i] > high[i-2]
                        fvg_bot, fvg_top = h[i - 2], li
                        on_side = fvg_bot >= rhi     # gap above OR high
                    else:
                        gap = l[i - 2] - hi          # bearish: high[i] < low[i-2]
                        fvg_bot, fvg_top = hi, l[i - 2]
                        on_side = fvg_top <= rlo     # gap below OR low
                    if gap <= 0 or not on_side:
                        continue
                    if cfg.fvg_min_atr > 0 and np.isfinite(a15_or):
                        if gap < cfg.fvg_min_atr * a15_or:
                            continue
                    st["phase"] = "seek_retest"
                    st["fvg_bot"] = fvg_bot
                    st["fvg_top"] = fvg_top
                    st["fvg_i"] = i
                    st["fvg_height"] = gap
                    continue

                if ph == "seek_retest":
                    ftop, fbot = st["fvg_top"], st["fvg_bot"]
                    # gap fully filled through far edge without a hold -> seek a new FVG
                    if side == 1 and ci < fbot:
                        st["phase"] = "seek_fvg"
                        continue
                    if side == -1 and ci > ftop:
                        st["phase"] = "seek_fvg"
                        continue
                    entered_gap = (li <= ftop) if side == 1 else (hi >= fbot)
                    if not entered_gap:
                        continue
                    # record how deep the retest reached into the gap (0..1)
                    if side == 1:
                        depth = (ftop - li) / (ftop - fbot)
                    else:
                        depth = (hi - fbot) / (ftop - fbot)
                    st["fvg_depth_frac"] = float(np.clip(depth, 0, 2))
                    st["retest_i"] = i
                    if not cfg.require_engulf:
                        # no-confirm arm: enter on this retest bar's close if it
                        # closed back in the trade direction (gap held)
                        held = (ci > fbot) if side == 1 else (ci < ftop)
                        if held and can_enter:
                            _book(trades, symbol, cfg, df, tf_min, side, i, i,
                                  rhi, rlo, or_sz, a15_or, st, o, h, l, c, v,
                                  vol20, atr15v, ema200, ny_hour, dow_arr,
                                  tmin_arr, open_mins, flat_idx, day_trades)
                            day_trades += 1
                            state[side] = {"phase": "idle"}
                        continue
                    st["phase"] = "seek_engulf"
                    continue

                if ph == "seek_engulf":
                    # update deepest retest as more bars probe the gap
                    ftop, fbot = st["fvg_top"], st["fvg_bot"]
                    pj = i - 1
                    prev_down = c[pj] < o[pj]
                    prev_up = c[pj] > o[pj]
                    if side == 1:
                        engulf = (ci > oi) and prev_down and (ci >= o[pj]) and (oi <= c[pj])
                    else:
                        engulf = (ci < oi) and prev_up and (ci <= o[pj]) and (oi >= c[pj])
                    if engulf and can_enter:
                        _book(trades, symbol, cfg, df, tf_min, side, i, pj,
                              rhi, rlo, or_sz, a15_or, st, o, h, l, c, v,
                              vol20, atr15v, ema200, ny_hour, dow_arr,
                              tmin_arr, open_mins, flat_idx, day_trades)
                        day_trades += 1
                        state[side] = {"phase": "idle"}
                    continue

    return pd.DataFrame(trades)


def _book(trades, symbol, cfg, df, tf_min, side, entry_i, engulfed_i,
          rhi, rlo, or_sz, a15_or, st, o, h, l, c, v, vol20, atr15v, ema200,
          ny_hour, dow_arr, tmin_arr, open_mins, flat_idx, day_trades):
    """Compute stop/tp, resolve the trade, and append the feature row."""
    entry = c[entry_i]
    buf = (cfg.stop_buffer_atr * a15_or) if np.isfinite(a15_or) else 0.0
    if side == 1:
        stop = l[engulfed_i] - buf
    else:
        stop = h[engulfed_i] + buf
    risk = (entry - stop) * side
    if risk <= 0:
        return
    tp = entry + side * cfg.rr * risk
    res = _resolve(o, h, l, c, entry_i + 1, side, entry, stop, tp,
                   cfg.max_hold_bars, flat_idx if cfg.session_flat else len(c) - 1)

    stop_pct = risk / entry
    fee_taker = cfg.costs.round_trip_cost_pct / stop_pct
    # maker on the TP leg only (resting limit at a fixed level); entry+stop taker
    half = cfg.costs.round_trip_cost_pct / 2.0
    if res["reason"] in ("tp", "tp_gap"):
        fee_mm = (half + 0.0002) / stop_pct   # taker entry + ~2bps maker exit
    else:
        fee_mm = fee_taker
    a15 = atr15v[entry_i]
    em = ema200[entry_i]
    fvg_h = st.get("fvg_height", np.nan)
    engulf_body = abs(c[entry_i] - o[entry_i])
    engulfed_body = abs(c[engulfed_i] - o[engulfed_i])

    trades.append({
        "symbol": symbol,
        "ny_date": df["ny_date"].iloc[entry_i],
        "entry_time": df["timestamp"].iloc[entry_i] + pd.Timedelta(minutes=tf_min),
        "side": "long" if side == 1 else "short",
        "entry": entry, "stop": stop, "tp": tp,
        "gross_r": res["gross_r"],
        "net_taker_r": res["gross_r"] - fee_taker,
        "net_mm_r": res["gross_r"] - fee_mm,
        "fee_taker_r": fee_taker,
        "exit_reason": res["reason"],
        "bars_held": res["bars_held"],
        "mae_r": res["mae_r"], "mfe_r": res["mfe_r"],
        "same_day_resolve": res["reason"] not in ("time",),
        # --- PRE-fill causal features ---
        "session": _session(int(ny_hour[entry_i])),
        "entry_hour_ny": int(ny_hour[entry_i]),
        "dow": int(dow_arr[entry_i]),
        "mins_since_open": int(tmin_arr[entry_i] - open_mins),
        "or_range_pct": or_sz / rlo,
        "or_range_vs_atr15": or_sz / a15_or if np.isfinite(a15_or) and a15_or > 0 else np.nan,
        "break_close_margin_atr": st.get("break_margin", np.nan) / a15_or
            if np.isfinite(a15_or) and a15_or > 0 else np.nan,
        "bars_break_to_fvg": int(st.get("fvg_i", entry_i) - st.get("broke_i", entry_i)),
        "fvg_height_atr": fvg_h / a15_or if np.isfinite(a15_or) and a15_or > 0 else np.nan,
        "fvg_depth_frac": st.get("fvg_depth_frac", np.nan),
        "bars_fvg_to_engulf": int(entry_i - st.get("fvg_i", entry_i)),
        "retest_leg_bars": int(entry_i - st.get("retest_i", entry_i)),
        "engulf_body_ratio": engulf_body / engulfed_body if engulfed_body > 0 else np.nan,
        "engulf_range_atr": (h[entry_i] - l[entry_i]) / a15_or
            if np.isfinite(a15_or) and a15_or > 0 else np.nan,
        "stop_pct": stop_pct,
        "stop_atr": risk / a15_or if np.isfinite(a15_or) and a15_or > 0 else np.nan,
        "fee_r": fee_taker,
        "rr_planned": cfg.rr,
        "trend_align": bool((entry > em) if side == 1 else (entry < em))
            if np.isfinite(em) else None,
        "dist_ema200_pct": (entry - em) / em * side if np.isfinite(em) else np.nan,
        "break_vol_ratio": (v[st.get("broke_i", entry_i)] / vol20[st.get("broke_i", entry_i)])
            if np.isfinite(vol20[st.get("broke_i", entry_i)]) and vol20[st.get("broke_i", entry_i)] > 0
            else np.nan,
        "prior_trades_today": int(day_trades),
    })


def _resolve(o, h, l, c, start, side, entry, stop, tp, max_hold, hard_idx):
    """Walk bars forward until stop/target/time/session-flat exit.

    Gap-through fills at the open; stop+target in one bar -> stop first
    (pessimistic). `hard_idx` = last bar index we may hold to (session flat).
    """
    risk = (entry - stop) * side
    mae = 0.0
    mfe = 0.0
    end = min(len(c) - 1, start + max_hold - 1, hard_idx)
    for j in range(start, end + 1):
        if side == 1:
            bar_worst = (l[j] - entry) / risk
            bar_best = (h[j] - entry) / risk
        else:
            bar_worst = (entry - h[j]) / risk
            bar_best = (entry - l[j]) / risk
        mae = min(mae, bar_worst)
        mfe = max(mfe, bar_best)
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
            return {"gross_r": -1.0, "reason": "stop",
                    "bars_held": j - start + 1, "mae_r": -1.0, "mfe_r": mfe}
        if hit_tp:
            return {"gross_r": (tp - entry) / risk * side, "reason": "tp",
                    "bars_held": j - start + 1, "mae_r": mae, "mfe_r": mfe}
    j = end
    return {"gross_r": (c[j] - entry) / risk * side, "reason": "time",
            "bars_held": end - start + 1, "mae_r": mae, "mfe_r": mfe}
