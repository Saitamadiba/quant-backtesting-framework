"""Predefined-level range fade — backtest engine. See SPEC.md."""

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
from ny4h_range_reversal.engine import load_bars  # noqa: E402

NY = ZoneInfo("America/New_York")


@dataclass
class PDConfig:
    stop_atr_limit: float = 0.35     # arm A stop beyond the level
    stop_atr_diverge: float = 0.25   # arm B stop beyond the episode extreme
    diverge_window: int = 24         # bars after touch to confirm divergence
    cancel_beyond_atr: float = 1.5   # close beyond level -> episode dead
    scratch_bars: int = 5
    scratch_atr: float = 0.4
    min_prevday_bars: int = 200
    max_hold_bars: int = 576
    exec_tf: str = "5m"
    costs: TransactionCosts = field(default_factory=TransactionCosts)


def rsi14(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    dn = (-d.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def resolve_variants(o, h, l, c, start: int, side: int, entry: float,
                     stop: float, tp: float, atr_e: float, cfg: PDConfig) -> dict:
    """One walk, three exit variants (plain / scratch / scaleout).

    Pessimistic intrabar: gap fills at open, stop beats target in a tie.
    Scratch check happens at the CLOSE of the (scratch_bars)-th bar, after
    that bar's stop/target. Scale-out: 50% at +1R then stop->BE.
    """
    n = len(c)
    risk = (entry - stop) * side
    tp1 = entry + side * risk
    end = min(n, start + cfg.max_hold_bars)
    out = {"mae_r": 0.0, "mfe_r": 0.0}
    open_plain = open_scr = open_so = True
    so_phase1 = True  # before the +1R partial
    for j in range(start, end):
        if not (open_plain or open_scr or open_so):
            break
        worst = ((l[j] if side == 1 else h[j]) - entry) * side / risk
        best = ((h[j] if side == 1 else l[j]) - entry) * side / risk
        out["mae_r"] = min(out["mae_r"], worst)
        out["mfe_r"] = max(out["mfe_r"], best)

        def bar_exit(stop_lvl):
            """(r, reason) if this bar exits vs stop_lvl/tp, else None."""
            gap_stop = (o[j] - stop_lvl) * side <= 0
            gap_tp = (o[j] - tp) * side >= 0
            hit_stop = ((l[j] if side == 1 else h[j]) - stop_lvl) * side <= 0
            hit_tp = ((h[j] if side == 1 else l[j]) - tp) * side >= 0
            if gap_stop:
                return (o[j] - entry) * side / risk, "stop_gap"
            if gap_tp:
                return (o[j] - entry) * side / risk, "tp_gap"
            if hit_stop:
                return (stop_lvl - entry) * side / risk, "stop"
            if hit_tp:
                return (tp - entry) * side / risk, "tp"
            return None

        if open_plain:
            ex = bar_exit(stop)
            if ex:
                out["r_plain"], out["reason_plain"] = ex
                out["bars_plain"] = j - start + 1
                open_plain = False
        if open_scr:
            ex = bar_exit(stop)
            if ex:
                out["r_scratch"], out["reason_scratch"] = ex
                open_scr = False
            elif j - start + 1 == cfg.scratch_bars and \
                    (c[j] - entry) * side < cfg.scratch_atr * atr_e:
                out["r_scratch"] = (c[j] - entry) * side / risk
                out["reason_scratch"] = "scratch"
                open_scr = False
        if open_so:
            if so_phase1:
                gap_stop = (o[j] - stop) * side <= 0
                hit_stop = ((l[j] if side == 1 else h[j]) - stop) * side <= 0
                hit_tp1 = ((h[j] if side == 1 else l[j]) - tp1) * side >= 0
                if gap_stop:
                    out["r_scaleout"] = (o[j] - entry) * side / risk
                    out["reason_scaleout"] = "stop"
                    open_so = False
                elif hit_stop and hit_tp1:
                    out["r_scaleout"] = -1.0       # pessimistic: stop first
                    out["reason_scaleout"] = "stop"
                    open_so = False
                elif hit_tp1:
                    so_phase1 = False              # half banked at +1R, BE on
                    # same-bar pessimism: BE beats full target in a tie
                    if ((l[j] if side == 1 else h[j]) - entry) * side <= 0:
                        out["r_scaleout"] = 0.5
                        out["reason_scaleout"] = "be"
                        open_so = False
                    elif ((h[j] if side == 1 else l[j]) - tp) * side >= 0:
                        out["r_scaleout"] = 0.5 + 0.5 * (tp - entry) * side / risk
                        out["reason_scaleout"] = "tp"
                        open_so = False
            else:
                hit_be = ((l[j] if side == 1 else h[j]) - entry) * side <= 0
                hit_tp = ((h[j] if side == 1 else l[j]) - tp) * side >= 0
                gap_tp = (o[j] - tp) * side >= 0
                if gap_tp:
                    out["r_scaleout"] = 0.5 + 0.5 * (o[j] - entry) * side / risk
                    out["reason_scaleout"] = "tp_gap"
                    open_so = False
                elif hit_be:                        # pessimistic: BE first
                    out["r_scaleout"] = 0.5
                    out["reason_scaleout"] = "be"
                    open_so = False
                elif hit_tp:
                    out["r_scaleout"] = 0.5 + 0.5 * (tp - entry) * side / risk
                    out["reason_scaleout"] = "tp"
                    open_so = False
    j = end - 1
    for flag, rk, reask in ((open_plain, "r_plain", "reason_plain"),
                            (open_scr, "r_scratch", "reason_scratch")):
        if flag:
            out[rk] = (c[j] - entry) * side / risk
            out[reask] = "time"
    if open_so:
        mark = (c[j] - entry) * side / risk
        out["r_scaleout"] = mark if so_phase1 else 0.5 + 0.5 * mark
        out["reason_scaleout"] = "time"
    out.setdefault("bars_plain", end - start)
    return out


def run_symbol(symbol: str, cfg: PDConfig | None = None) -> pd.DataFrame:
    cfg = cfg or PDConfig(costs=TransactionCosts.for_asset(symbol))
    df = load_bars(symbol, cfg.exec_tf)
    if df.empty:
        return pd.DataFrame()

    ts_ny = df["timestamp"].dt.tz_convert(NY)
    df["ny_date"] = ts_ny.dt.date
    ny_hour = ts_ny.dt.hour.to_numpy()
    dow = ts_ny.dt.dayofweek.to_numpy()
    tf_min = {"5m": 5, "15m": 15}[cfg.exec_tf]

    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    vol20 = df["volume"].rolling(20).mean().to_numpy()
    atr = df["atr_14"].to_numpy()
    ema200 = df["ema_200"].to_numpy()
    rsi = rsi14(df["close"]).to_numpy()

    # previous-NY-day levels
    daily = df.groupby("ny_date").agg(
        d_high=("high", "max"), d_low=("low", "min"), d_open=("open", "first"),
        d_close=("close", "last"), d_n=("close", "size")).reset_index()
    daily["pdh"] = daily["d_high"].shift(1)
    daily["pdl"] = daily["d_low"].shift(1)
    daily["pd_close"] = daily["d_close"].shift(1)
    daily["pd_n"] = daily["d_n"].shift(1)
    daily["pd_range"] = daily["pdh"] - daily["pdl"]
    daily["pd_range_20"] = daily["pd_range"].rolling(20).mean().shift(0)
    dmap = daily.set_index("ny_date")

    def session(hr: int) -> str:
        if 3 <= hr < 8:
            return "london"
        if 8 <= hr < 11:
            return "overlap"
        if 11 <= hr < 17:
            return "ny"
        return "off"

    trades: list[dict] = []
    grp = df.groupby("ny_date").indices
    for day, idx in grp.items():
        drow = dmap.loc[day]
        if (not np.isfinite(drow["pdh"]) or not np.isfinite(drow["pdl"])
                or drow["pd_n"] < cfg.min_prevday_bars):
            continue
        idx = np.asarray(idx)
        j0 = idx[0]
        pdh, pdl = float(drow["pdh"]), float(drow["pdl"])
        pd_range = pdh - pdl
        if pd_range <= 0:
            continue
        gap_pct = (o[j0] - drow["pd_close"]) / drow["pd_close"] \
            if np.isfinite(drow["pd_close"]) else np.nan
        # first-touch index per side (None if untouched), computed up front
        touch_i: dict[int, int | None] = {}
        for side in (1, -1):
            lvl = pdl if side == 1 else pdh
            ext_arr = l if side == 1 else h
            if (o[j0] - lvl) * side <= 0:
                touch_i[side] = None  # day opened beyond the level
                continue
            hits = np.flatnonzero((ext_arr[idx] - lvl) * side <= 0)
            touch_i[side] = int(idx[hits[0]]) if len(hits) else None

        for side in (1, -1):
            lvl = pdl if side == 1 else pdh
            tp = pdh if side == 1 else pdl
            ext_arr = l if side == 1 else h
            t_i = touch_i[side]
            if t_i is None:
                continue
            t_other = touch_i[-side]
            other_first = t_other is not None and t_other < t_i
            a_t = atr[t_i]
            if not np.isfinite(a_t) or a_t <= 0:
                continue

            common = {
                "symbol": symbol, "ny_date": day,
                "dow": int(dow[t_i]),
                "pd_range_atr": pd_range / a_t,
                "pd_range_rel20": float(drow["pd_range"] / drow["pd_range_20"])
                    if np.isfinite(drow["pd_range_20"]) and drow["pd_range_20"] > 0
                    else np.nan,
                "gap_open_pct": gap_pct * side if np.isfinite(gap_pct) else np.nan,
                "other_side_touched_first": bool(other_first),
                "side": "long" if side == 1 else "short",
            }

            def book(arm, e_i, entry, stop, sweep_depth, rsi_e, div_str):
                risk = (entry - stop) * side
                reward = (tp - entry) * side
                if risk <= 0 or reward <= 0:
                    return
                res = resolve_variants(o, h, l, c, e_i + 1, side, entry, stop,
                                       tp, atr[e_i], cfg)
                stop_pct = risk / entry
                rt = cfg.costs.round_trip_cost_pct
                trades.append({**common, "arm": arm,
                    "entry_time": df["timestamp"].iloc[e_i] + pd.Timedelta(minutes=tf_min),
                    "entry_hour_ny": int(ny_hour[e_i]),
                    "session": session(int(ny_hour[e_i])),
                    "entry": entry, "stop": stop, "tp": tp,
                    "rr_planned": reward / risk, "stop_pct": stop_pct,
                    "fee_r": rt / stop_pct,
                    "bars_since_day_open": int(e_i - j0),
                    "sweep_depth_atr": sweep_depth,
                    "vol_ratio_touch": v[t_i] / vol20[t_i]
                        if np.isfinite(vol20[t_i]) and vol20[t_i] > 0 else np.nan,
                    "dist_ema200_pct": (entry - ema200[e_i]) / ema200[e_i] * side
                        if np.isfinite(ema200[e_i]) else np.nan,
                    "rsi_entry": rsi_e, "rsi_div_strength": div_str,
                    "gross_r": res["r_plain"], "net_r": res["r_plain"] - rt / stop_pct,
                    "exit_reason": res["reason_plain"],
                    "bars_held": res["bars_plain"],
                    "mae_r": res["mae_r"], "mfe_r": res["mfe_r"],
                    "r_scratch": res["r_scratch"],
                    "reason_scratch": res["reason_scratch"],
                    "r_scaleout": res["r_scaleout"],
                    "reason_scaleout": res["reason_scaleout"]})

            # --- arm A: limit at the level ---
            fill = o[t_i] if (o[t_i] - lvl) * side <= 0 else lvl
            stop_a = lvl - side * cfg.stop_atr_limit * a_t
            if (fill - stop_a) * side <= 0:
                pass  # touch bar gapped beyond the stop — no fill to book
            elif ((l[t_i] if side == 1 else h[t_i]) - stop_a) * side <= 0:
                # filled and stopped within the touch bar — pessimistic -1
                risk = (fill - stop_a) * side
                stop_pct = risk / fill
                rt = cfg.costs.round_trip_cost_pct
                trades.append({**common, "arm": "limit",
                    "entry_time": df["timestamp"].iloc[t_i] + pd.Timedelta(minutes=tf_min),
                    "entry_hour_ny": int(ny_hour[t_i]),
                    "session": session(int(ny_hour[t_i])),
                    "entry": fill, "stop": stop_a, "tp": tp,
                    "rr_planned": (tp - fill) * side / risk, "stop_pct": stop_pct,
                    "fee_r": rt / stop_pct,
                    "bars_since_day_open": int(t_i - j0),
                    "sweep_depth_atr": 0.0,
                    "vol_ratio_touch": v[t_i] / vol20[t_i]
                        if np.isfinite(vol20[t_i]) and vol20[t_i] > 0 else np.nan,
                    "dist_ema200_pct": (fill - ema200[t_i]) / ema200[t_i] * side
                        if np.isfinite(ema200[t_i]) else np.nan,
                    "rsi_entry": rsi[t_i], "rsi_div_strength": np.nan,
                    "gross_r": -1.0, "net_r": -1.0 - rt / stop_pct,
                    "exit_reason": "stop_samebar", "bars_held": 0,
                    "mae_r": -1.0, "mfe_r": 0.0,
                    "r_scratch": -1.0, "reason_scratch": "stop_samebar",
                    "r_scaleout": -1.0, "reason_scaleout": "stop_samebar"})
            else:
                book("limit", t_i, fill, stop_a, 0.0, rsi[t_i], np.nan)

            # --- arm B: divergence within the window ---
            ext_val = ext_arr[t_i]
            ext_rsi = rsi[t_i]
            end_w = min(int(idx[-1]), t_i + cfg.diverge_window)
            for j in range(t_i + 1, end_w + 1):
                if (c[j] - (lvl - side * cfg.cancel_beyond_atr * atr[j])) * side < 0 \
                        and np.isfinite(atr[j]):
                    break  # deep close through the level
                opp = pdh if side == 1 else pdl
                if ((h[j] if side == 1 else l[j]) - opp) * side >= 0:
                    break  # opposite level reached first
                if (ext_arr[j] - ext_val) * side < 0:
                    if np.isfinite(rsi[j]) and np.isfinite(ext_rsi) \
                            and (rsi[j] - ext_rsi) * side > 0:
                        stop_b = ext_arr[j] - side * cfg.stop_atr_diverge * atr[j]
                        book("diverge", j, c[j], stop_b,
                             (lvl - ext_arr[j]) * side / atr[j],
                             rsi[j], (rsi[j] - ext_rsi) * side)
                        break
                    ext_val, ext_rsi = ext_arr[j], rsi[j]
    return pd.DataFrame(trades)
