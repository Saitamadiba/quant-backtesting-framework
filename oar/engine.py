"""Opening-Auction Exhaustion Reversal — backtest engine.

Fade the overextended open back into the initial balance. Build the IB from
the first 15m candle, gate on opening_vol_ratio = IB_range / daily_ATR, wait
for price to sweep beyond an IB edge, then a candlestick-exhaustion trigger
(hammer-break or engulfing) — enter the reversal, stop beyond the exhaustion
extreme, target the opposite IB edge. Passive management, max 2/day, flat by
session end. OHLC intrabar, stop-first pessimistic. All features PRE-fill
causal. See SPEC.md.
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
class OARConfig:
    exec_tf: str = "5m"
    open_hour_ny: int = 9
    open_min_ny: int = 30
    ib_minutes: int = 15            # initial-balance candle length
    flat_hour_ny: int = 16
    entry_cutoff_hour_ny: int = 15
    entry_cutoff_min_ny: int = 30
    ib_ratio_min: float = 0.20      # opening_vol_ratio gate (IB range / daily ATR)
    entry_mode: str = "engulf"      # "engulf" | "hammer"
    target_mode: str = "ib"         # "ib" (opposite IB edge) | "fixed"
    rr: float = 2.0                 # used when target_mode == "fixed"
    wick_mult: float = 2.0          # hammer lower-wick / body
    body_frac_max: float = 0.4      # hammer body / range ceiling
    wick_opp_frac_max: float = 0.4  # hammer opposite-wick / range ceiling
    stop_buffer_atr: float = 0.0
    min_stop_atr: float = 0.10      # floor on risk (ATR15) — kills tiny-candle degeneracy
    rr_max: float = 25.0            # skip absurd targets (unrealistic reversion distance)
    max_setup_bars: int = 78
    max_hold_bars: int = 96
    max_trades_per_day: int = 2
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
_DAILY_ATR_CACHE: dict[str, pd.DataFrame] = {}


def _atr15_series(symbol: str) -> pd.DataFrame:
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


def _daily_atr_map(symbol: str) -> dict:
    """NY-calendar-day Wilder ATR14, causal (prior completed day). Keyed by
    ny_date -> atr value usable ON that date. Crypto resampled from 1h; NQ
    uses native 1d."""
    if symbol in _DAILY_ATR_CACHE:
        return _DAILY_ATR_CACHE[symbol]
    # Resample from an intraday TF that is SCALE-CONSISTENT with the 15m IB
    # bars. NQ's 1d/1h/4h series are on different price scales than its 15m
    # (data-integrity mismatch), so NQ must resample from 15m; crypto's 1h is
    # consistent with its 5m/15m.
    src_tf = "15m" if symbol == "NQ" else "1h"
    h1 = load_bars(symbol, src_tf)
    if h1.empty:
        _DAILY_ATR_CACHE[symbol] = {}
        return {}
    h1["ny_date"] = h1["timestamp"].dt.tz_convert(NY).dt.date
    day = h1.groupby("ny_date").agg(h=("high", "max"), l=("low", "min"),
                                    c=("close", "last")).reset_index()
    day = day.sort_values("ny_date").reset_index(drop=True)
    pc = day["c"].shift(1)
    tr = pd.concat([(day["h"] - day["l"]), (day["h"] - pc).abs(),
                    (day["l"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    # causal: the ATR value available ON day t is the one computed through t-1
    atr_prior = atr.shift(1)
    out = dict(zip(day["ny_date"], atr_prior))
    _DAILY_ATR_CACHE[symbol] = out
    return out


def _is_hammer(o, h, l, c, cfg) -> bool:
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    return (lower >= cfg.wick_mult * max(body, 1e-12)
            and body <= cfg.body_frac_max * rng
            and upper <= cfg.wick_opp_frac_max * rng)


def _is_star(o, h, l, c, cfg) -> bool:
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return (upper >= cfg.wick_mult * max(body, 1e-12)
            and body <= cfg.body_frac_max * rng
            and lower <= cfg.wick_opp_frac_max * rng)


def run_symbol(symbol: str, cfg: OARConfig | None = None) -> pd.DataFrame:
    cfg = cfg or OARConfig(costs=TransactionCosts.for_asset(symbol))
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
    df["atr15"] = pd.merge_asof(df[["timestamp"]], atr15, left_on="timestamp",
                                right_on="time", direction="backward")["atr15"].to_numpy()
    datr = _daily_atr_map(symbol)

    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float); vol20 = df["vol20"].to_numpy(float)
    atr15v = df["atr15"].to_numpy(float); ema200 = df["ema_200"].to_numpy(float)
    ny_hour = df["ny_hour"].to_numpy(int); ny_min = df["ny_min"].to_numpy(int)
    dow_arr = df["dow"].to_numpy(int)
    tmin_arr = ny_hour * 60 + ny_min

    open_mins = cfg.open_hour_ny * 60 + cfg.open_min_ny
    ib_end_mins = open_mins + cfg.ib_minutes
    flat_mins = cfg.flat_hour_ny * 60
    cutoff_mins = cfg.entry_cutoff_hour_ny * 60 + cfg.entry_cutoff_min_ny
    ib_bars = max(1, cfg.ib_minutes // tf_min)

    trades: list[dict] = []
    for day, idx in df.groupby("ny_date").indices.items():
        idx = np.asarray(idx)
        # IB = bars from the open bar through ib_end (exclusive)
        ib_mask = (tmin_arr[idx] >= open_mins) & (tmin_arr[idx] < ib_end_mins)
        ib_idx = idx[ib_mask]
        if len(ib_idx) < ib_bars or tmin_arr[idx][ib_mask][0] != open_mins:
            continue
        ib_hi = h[ib_idx].max(); ib_lo = l[ib_idx].min()
        if not (np.isfinite(ib_hi) and np.isfinite(ib_lo)) or ib_hi <= ib_lo:
            continue
        ib_range = ib_hi - ib_lo
        da = datr.get(day, np.nan)
        if not np.isfinite(da) or da <= 0:
            continue
        ovr = ib_range / da
        if ovr < cfg.ib_ratio_min:
            continue
        a15_ib = atr15v[ib_idx[-1]]

        trad_idx = idx[(tmin_arr[idx] >= ib_end_mins) & (tmin_arr[idx] < flat_mins)]
        if len(trad_idx) < 3:
            continue
        flat_idx = int(trad_idx.max())
        ib_close_i = int(ib_idx[-1])

        day_trades = 0
        # per side: +1 long (fade down-extension), -1 short (fade up-extension)
        state = {1: {"phase": "idle"}, -1: {"phase": "idle"}}

        for i in trad_idx:
            if day_trades >= cfg.max_trades_per_day:
                break
            ci, oi, hi, li = c[i], o[i], h[i], l[i]
            can_enter = tmin_arr[i] < cutoff_mins
            for side in (1, -1):
                st = state[side]
                edge = ib_lo if side == 1 else ib_hi
                swept_now = (li < ib_lo) if side == 1 else (hi > ib_hi)
                ph = st["phase"]

                if ph == "idle":
                    if swept_now:
                        st.clear()
                        st.update(phase="swept", sweep_i=i,
                                  ext=(li if side == 1 else hi))
                    continue

                # armed — resets
                if i - st["sweep_i"] > cfg.max_setup_bars:
                    state[side] = {"phase": "idle"}
                    continue
                # keep tracking the extension extreme
                st["ext"] = (min(st["ext"], li) if side == 1 else max(st["ext"], hi))

                if cfg.entry_mode == "hammer":
                    if ph == "swept":
                        # look for the exhaustion candle (hammer/star) that swept
                        cand = (_is_hammer(oi, hi, li, ci, cfg) and li <= ib_lo) if side == 1 \
                            else (_is_star(oi, hi, li, ci, cfg) and hi >= ib_hi)
                        if cand:
                            st["phase"] = "armed_hammer"
                            st["trig_hi"] = hi; st["trig_lo"] = li
                            st["trig_i"] = i
                        continue
                    if ph == "armed_hammer":
                        trg = (hi > st["trig_hi"]) if side == 1 else (li < st["trig_lo"])
                        if trg and can_enter:
                            level = st["trig_hi"] if side == 1 else st["trig_lo"]
                            entry = max(oi, level) if side == 1 else min(oi, level)
                            stop_ext = st["trig_lo"] if side == 1 else st["trig_hi"]
                            _book(trades, symbol, cfg, df, tf_min, side, i,
                                  entry, stop_ext, ib_hi, ib_lo, ib_range, ovr,
                                  da, a15_ib, st, o, h, l, c, v, vol20, atr15v,
                                  ema200, ny_hour, dow_arr, tmin_arr, open_mins,
                                  ib_close_i, flat_idx, day_trades)
                            day_trades += 1
                            state[side] = {"phase": "idle"}
                        continue
                else:  # engulf
                    if ph in ("swept",):
                        pj = i - 1
                        if side == 1:
                            eng = (ci > oi) and (c[pj] < o[pj]) and (ci >= o[pj]) and (oi <= c[pj])
                            swept = (li <= ib_lo) or (l[pj] <= ib_lo)
                        else:
                            eng = (ci < oi) and (c[pj] > o[pj]) and (ci <= o[pj]) and (oi >= c[pj])
                            swept = (hi >= ib_hi) or (h[pj] >= ib_hi)
                        if eng and swept and can_enter:
                            entry = ci
                            stop_ext = min(li, l[pj]) if side == 1 else max(hi, h[pj])
                            st["trig_i"] = i
                            st["trig_lo"] = min(li, l[pj]); st["trig_hi"] = max(hi, h[pj])
                            _book(trades, symbol, cfg, df, tf_min, side, i,
                                  entry, stop_ext, ib_hi, ib_lo, ib_range, ovr,
                                  da, a15_ib, st, o, h, l, c, v, vol20, atr15v,
                                  ema200, ny_hour, dow_arr, tmin_arr, open_mins,
                                  ib_close_i, flat_idx, day_trades)
                            day_trades += 1
                            state[side] = {"phase": "idle"}
                        continue

    return pd.DataFrame(trades)


def _book(trades, symbol, cfg, df, tf_min, side, entry_i, entry, stop_ext,
          ib_hi, ib_lo, ib_range, ovr, da, a15_ib, st, o, h, l, c, v, vol20,
          atr15v, ema200, ny_hour, dow_arr, tmin_arr, open_mins, ib_close_i,
          flat_idx, day_trades):
    buf = (cfg.stop_buffer_atr * a15_ib) if np.isfinite(a15_ib) else 0.0
    stop = (stop_ext - buf) if side == 1 else (stop_ext + buf)
    risk = (entry - stop) * side
    if risk <= 0:
        return
    # floor the stop at min_stop_atr (kills tiny-candle micro-stop degeneracy)
    if cfg.min_stop_atr > 0 and np.isfinite(a15_ib) and a15_ib > 0:
        min_risk = cfg.min_stop_atr * a15_ib
        if risk < min_risk:
            risk = min_risk
            stop = entry - side * risk
    if cfg.target_mode == "ib":
        tp = ib_hi if side == 1 else ib_lo
    else:
        tp = entry + side * cfg.rr * risk
    if (tp - entry) * side <= 0:   # no room to target
        return
    if abs(tp - entry) / risk > cfg.rr_max:   # unrealistic reversion distance
        return
    res = _resolve(o, h, l, c, entry_i + 1, side, entry, stop, tp,
                   cfg.max_hold_bars, flat_idx)

    stop_pct = risk / entry
    fee_taker = cfg.costs.round_trip_cost_pct / stop_pct
    half = cfg.costs.round_trip_cost_pct / 2.0
    fee_mm = ((half + 0.0002) / stop_pct) if res["reason"] in ("tp", "tp_gap") else fee_taker
    a15 = atr15v[entry_i]; em = ema200[entry_i]
    trig_lo = st.get("trig_lo", np.nan); trig_hi = st.get("trig_hi", np.nan)
    trig_i = st.get("trig_i", entry_i)
    body = abs(c[trig_i] - o[trig_i]); rng = h[trig_i] - l[trig_i]
    lower = min(o[trig_i], c[trig_i]) - l[trig_i]
    upper = h[trig_i] - max(o[trig_i], c[trig_i])
    wick = (lower if side == 1 else upper)
    ext = st.get("ext", entry)
    sweep_depth = (ib_lo - ext) if side == 1 else (ext - ib_hi)

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
        "exit_reason": res["reason"], "bars_held": res["bars_held"],
        "mae_r": res["mae_r"], "mfe_r": res["mfe_r"],
        "same_day_resolve": res["reason"] not in ("time",),
        # PRE-fill causal features
        "session": _session(int(ny_hour[entry_i])),
        "entry_hour_ny": int(ny_hour[entry_i]),
        "dow": int(dow_arr[entry_i]),
        "mins_since_open": int(tmin_arr[entry_i] - open_mins),
        "opening_vol_ratio": float(ovr),
        "ib_range_pct": ib_range / ib_lo,
        "sweep_depth_atr": sweep_depth / a15_ib if np.isfinite(a15_ib) and a15_ib > 0 else np.nan,
        "sweep_depth_frac": sweep_depth / ib_range if ib_range > 0 else np.nan,
        "bars_ib_to_sweep": int(st.get("sweep_i", entry_i) - ib_close_i),
        "bars_sweep_to_entry": int(entry_i - st.get("sweep_i", entry_i)),
        "trigger_wick_ratio": wick / max(body, 1e-12),
        "trigger_body_frac": body / rng if rng > 0 else np.nan,
        "trigger_range_atr": rng / a15_ib if np.isfinite(a15_ib) and a15_ib > 0 else np.nan,
        "dist_to_target_atr": abs(tp - entry) / a15_ib if np.isfinite(a15_ib) and a15_ib > 0 else np.nan,
        "rr_planned": abs(tp - entry) / risk,
        "stop_pct": stop_pct,
        "stop_atr": risk / a15_ib if np.isfinite(a15_ib) and a15_ib > 0 else np.nan,
        "fee_r": fee_taker,
        "trend_align": bool((entry > em) if side == 1 else (entry < em)) if np.isfinite(em) else None,
        "dist_ema200_pct": (entry - em) / em * side if np.isfinite(em) else np.nan,
        "break_vol_ratio": (v[trig_i] / vol20[trig_i])
            if np.isfinite(vol20[trig_i]) and vol20[trig_i] > 0 else np.nan,
        "prior_trades_today": int(day_trades),
    })


def _session(hr: int) -> str:
    if 0 <= hr < 6: return "asia"
    if 6 <= hr < 9: return "london"
    if 9 <= hr < 12: return "ny_am"
    if 12 <= hr < 16: return "ny_pm"
    return "late"


def _resolve(o, h, l, c, start, side, entry, stop, tp, max_hold, hard_idx):
    risk = (entry - stop) * side
    mae = 0.0; mfe = 0.0
    end = min(len(c) - 1, start + max_hold - 1, hard_idx)
    for j in range(start, end + 1):
        if side == 1:
            bar_worst = (l[j] - entry) / risk; bar_best = (h[j] - entry) / risk
        else:
            bar_worst = (entry - h[j]) / risk; bar_best = (entry - l[j]) / risk
        mae = min(mae, bar_worst); mfe = max(mfe, bar_best)
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
