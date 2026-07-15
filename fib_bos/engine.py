"""Fibonacci BOS Continuation — backtest engine.

Micro-trend (HH/HL) -> BOS -> golden-zone (0.50-0.618) retracement of the
impulse leg -> rejection trigger (or resting limit at 0.618) -> target at
the impulse high, stop beyond the impulse origin. See SPEC.md.

Reuses ny4h_range_reversal loaders/resolver and structure_scalper pivots.
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
from ny4h_range_reversal.engine import _resolve, load_bars  # noqa: E402
from structure_scalper.engine import pivot_asof  # noqa: E402

NY = ZoneInfo("America/New_York")


@dataclass
class FibConfig:
    swing_n: int = 3
    zone_hi: float = 0.50        # golden zone shallow edge
    zone_lo: float = 0.618       # golden zone deep edge / limit level
    invalid_fib: float = 0.786   # close beyond -> setup dead
    buffer_atr: float = 0.25
    max_wait_retrace: int = 96
    trigger_window: int = 6      # bars after last zone touch
    momentum_body: float = 0.60
    momentum_range_atr: float = 1.2
    min_rr: float = 0.5   # objective form of the video's "price accelerates -> cancel"
    max_hold_bars: int = 576
    exec_tf: str = "5m"
    costs: TransactionCosts = field(default_factory=TransactionCosts)


IDLE, AWAIT = 0, 1


def run_symbol(symbol: str, cfg: FibConfig | None = None) -> pd.DataFrame:
    cfg = cfg or FibConfig(costs=TransactionCosts.for_asset(symbol))
    df = load_bars(symbol, cfg.exec_tf)
    if df.empty:
        return pd.DataFrame()
    tf_min = {"5m": 5, "15m": 15, "1h": 60}[cfg.exec_tf]

    ts_ny = df["timestamp"].dt.tz_convert(NY)
    ny_date = ts_ny.dt.date.to_numpy()
    ny_hour = ts_ny.dt.hour.to_numpy()
    dow = ts_ny.dt.dayofweek.to_numpy()

    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    vol20 = df["volume"].rolling(20).mean().to_numpy()
    atr = df["atr_14"].to_numpy()
    ema200 = df["ema_200"].to_numpy()
    n = len(df)

    sh_lvl, sh_idx, sl_lvl, sl_idx = pivot_asof(h, l, cfg.swing_n)

    # per-bar micro-trend label from last-2 confirmed swings
    seq = np.full(n, "na", dtype=object)
    highs2: list[float] = []; lows2: list[float] = []
    psh, psl = -1, -1
    for j in range(n):
        if sh_idx[j] != psh and sh_idx[j] >= 0:
            highs2.append(sh_lvl[j]); highs2 = highs2[-2:]; psh = sh_idx[j]
        if sl_idx[j] != psl and sl_idx[j] >= 0:
            lows2.append(sl_lvl[j]); lows2 = lows2[-2:]; psl = sl_idx[j]
        if len(highs2) == 2 and len(lows2) == 2:
            if highs2[1] > highs2[0] and lows2[1] > lows2[0]:
                seq[j] = "up"
            elif highs2[1] < highs2[0] and lows2[1] < lows2[0]:
                seq[j] = "down"
            else:
                seq[j] = "mixed"

    def session(hr: int) -> str:
        if 3 <= hr < 8:
            return "london"
        if 8 <= hr < 11:
            return "overlap"
        if 11 <= hr < 17:
            return "ny"
        return "off"

    def trigger_type(j: int, side: int) -> str | None:
        rng = h[j] - l[j]
        if rng <= 0 or not np.isfinite(atr[j]) or atr[j] <= 0:
            return None
        body = abs(c[j] - o[j])
        if side == 1:
            bull = c[j] > o[j]
            lower_wick = min(o[j], c[j]) - l[j]
            upper_wick = h[j] - max(o[j], c[j])
            if (bull and c[j - 1] < o[j - 1] and o[j] <= c[j - 1]
                    and c[j] >= o[j - 1] and body >= 0.3 * rng):
                return "engulf"
            if (lower_wick >= 0.55 * rng and upper_wick <= 0.25 * rng
                    and lower_wick >= 2 * body and c[j] >= l[j] + 0.5 * rng):
                return "pin"
            if (bull and body >= cfg.momentum_body * rng
                    and rng >= cfg.momentum_range_atr * atr[j]):
                return "momentum"
        else:
            bear = c[j] < o[j]
            upper_wick = h[j] - max(o[j], c[j])
            lower_wick = min(o[j], c[j]) - l[j]
            if (bear and c[j - 1] > o[j - 1] and o[j] >= c[j - 1]
                    and c[j] <= o[j - 1] and body >= 0.3 * rng):
                return "engulf"
            if (upper_wick >= 0.55 * rng and lower_wick <= 0.25 * rng
                    and upper_wick >= 2 * body and c[j] <= h[j] - 0.5 * rng):
                return "pin"
            if (bear and body >= cfg.momentum_body * rng
                    and rng >= cfg.momentum_range_atr * atr[j]):
                return "momentum"
        return None

    trades: list[dict] = []
    day_counts: dict = {}
    mach = {1: {"state": IDLE}, -1: {"state": IDLE}}
    used_piv = {1: -1, -1: -1}

    def book(side: int, j: int, arm: str, entry: float, stop: float, tp: float,
             st: dict, ttype: str | None) -> None:
        risk = (entry - stop) * side
        reward = (tp - entry) * side
        if risk <= 0 or reward <= 0 or reward / risk < cfg.min_rr:
            return
        entry_bar_ext = l[j] if side == 1 else h[j]
        if arm == "limit618" and (entry_bar_ext - stop) * side <= 0:
            # pessimistic: limit filled and stop traded through in the SAME bar
            res = {"gross_r": -1.0, "reason": "stop_samebar", "bars_held": 0,
                   "mae_r": -1.0, "mfe_r": 0.0}
        else:
            res = _resolve(o, h, l, c, j + 1, side, entry, stop, tp, cfg.max_hold_bars)
        stop_pct = risk / entry
        fee_r = cfg.costs.round_trip_cost_pct / stop_pct
        leg = abs(st["impulse"] - st["origin"])
        day = ny_date[j]
        trades.append({
            "symbol": symbol, "ny_date": day, "arm": arm,
            "entry_time": df["timestamp"].iloc[j] + pd.Timedelta(minutes=tf_min),
            "entry_idx": j, "exit_idx": j + res["bars_held"],
            "side": "long" if side == 1 else "short",
            "trigger_type": ttype or "limit",
            "session": session(int(ny_hour[j])),
            "entry_hour_ny": int(ny_hour[j]), "dow": int(dow[j]),
            "entry": entry, "stop": stop, "tp": tp,
            "rr_planned": reward / risk,
            "stop_pct": stop_pct, "fee_r": fee_r,
            "gross_r": res["gross_r"], "net_r": res["gross_r"] - fee_r,
            "exit_reason": res["reason"], "bars_held": res["bars_held"],
            "mae_r": res["mae_r"], "mfe_r": res["mfe_r"],
            "entry_fib": (st["impulse"] - entry) / leg * side if leg > 0 else np.nan,
            "leg_atr": leg / atr[j] if np.isfinite(atr[j]) and atr[j] > 0 else np.nan,
            "leg_pct": leg / entry,
            "bos_dist_atr": st["bos_dist_atr"],
            "zone_touch_bars": (st["touch_i"] - st["bos_i"]) if st["touch_i"] >= 0 else np.nan,
            "retrace_depth_frac": (st["impulse"] - st["deepest"]) / leg * side
                if leg > 0 else np.nan,
            "trigger_vol_ratio": v[j] / vol20[j]
                if np.isfinite(vol20[j]) and vol20[j] > 0 else np.nan,
            "dist_ema200_pct": (entry - ema200[j]) / ema200[j] * side
                if np.isfinite(ema200[j]) else np.nan,
            "prior_signals_today": day_counts.get(day, 0),
        })
        day_counts[day] = day_counts.get(day, 0) + 1

    warm = max(cfg.swing_n * 2 + 2, 21)
    for j in range(warm, n):
        for side in (1, -1):
            st = mach[side]
            trend_ok = seq[j] == ("up" if side == 1 else "down")

            if st["state"] == IDLE:
                if not trend_ok:
                    continue
                lvl = sh_lvl[j] if side == 1 else sl_lvl[j]
                piv = sh_idx[j] if side == 1 else sl_idx[j]
                origin = sl_lvl[j] if side == 1 else sh_lvl[j]
                if (piv < 0 or piv == used_piv[side] or not np.isfinite(lvl)
                        or not np.isfinite(origin)):
                    continue
                broke = c[j] > lvl if side == 1 else c[j] < lvl
                if not broke:
                    continue
                if (origin - lvl) * side >= 0:  # origin must be on the far side
                    continue
                used_piv[side] = piv
                a0 = atr[j] if np.isfinite(atr[j]) and atr[j] > 0 else np.nan
                st.update(state=AWAIT, origin=origin, bos_i=j,
                          impulse=h[j] if side == 1 else l[j],
                          deepest=c[j], touch_i=-1, limit_done=False,
                          trig_done=False,
                          bos_dist_atr=abs(c[j] - lvl) / a0 if np.isfinite(a0) else np.nan)
                continue

            # --- AWAIT retracement / trigger ---
            opp_lvl = sl_lvl[j] if side == 1 else sh_lvl[j]
            opp_break = (np.isfinite(opp_lvl)
                         and (c[j] < opp_lvl if side == 1 else c[j] > opp_lvl))
            if not trend_ok or opp_break or j - st["bos_i"] > cfg.max_wait_retrace:
                st.clear(); st["state"] = IDLE
                continue
            a = atr[j]
            if not np.isfinite(a) or a <= 0:
                st.clear(); st["state"] = IDLE
                continue

            # CAUSALITY: an intrabar limit fill can only rest at a level derived
            # from information available BEFORE this bar — the fib grid as of
            # the previous bar's impulse. The grid from THIS bar's (possibly
            # extended) impulse is close-time knowledge: valid for the trigger
            # arm and invalidation, never for the intrabar fill. (An earlier
            # version priced the fill off the same-bar impulse; the implicit
            # high-before-low assumption fabricated the whole arm's edge.)
            H_prev, O_ = st["impulse"], st["origin"]
            ext = l[j] if side == 1 else h[j]

            leg_prev = (H_prev - O_) * side
            if leg_prev > 0 and not st["limit_done"]:
                f618_prev = H_prev - side * cfg.zone_lo * leg_prev
                if (ext - f618_prev) * side <= 0:
                    fill = o[j] if (o[j] - f618_prev) * side <= 0 else f618_prev
                    stop = O_ - side * cfg.buffer_atr * a
                    st["limit_done"] = True
                    st["deepest"] = (min(st["deepest"], ext) if side == 1
                                     else max(st["deepest"], ext))
                    book(side, j, "limit618", fill, stop, H_prev, st, None)

            # impulse extension resets the fib grid (effective for close-time
            # checks now, and for intrabar fills from the NEXT bar on)
            if side == 1 and h[j] > st["impulse"]:
                st["impulse"] = h[j]
            elif side == -1 and l[j] < st["impulse"]:
                st["impulse"] = l[j]

            H = st["impulse"]
            leg = (H - O_) * side
            if leg <= 0:
                st.clear(); st["state"] = IDLE
                continue
            f50 = H - side * cfg.zone_hi * leg
            f786 = H - side * cfg.invalid_fib * leg

            st["deepest"] = min(st["deepest"], ext) if side == 1 else max(st["deepest"], ext)

            if (ext - f50) * side <= 0:
                st["touch_i"] = j

            # deep-close invalidation
            if (c[j] - f786) * side < 0:
                st.clear(); st["state"] = IDLE
                continue

            # trigger arm: rejection candle within window of last zone touch
            if (not st["trig_done"] and st["touch_i"] >= 0
                    and j - st["touch_i"] <= cfg.trigger_window):
                tt = trigger_type(j, side)
                if tt is not None:
                    stop = O_ - side * cfg.buffer_atr * a
                    st["trig_done"] = True
                    book(side, j, "trigger", c[j], stop, H, st, tt)

            if st["limit_done"] and st["trig_done"]:
                st.clear(); st["state"] = IDLE

    out = pd.DataFrame(trades)
    if out.empty:
        return out
    # prior_was_stopout: most recent same-arm trade fully resolved before entry
    import bisect

    out = out.sort_values("entry_idx").reset_index(drop=True)
    for arm in out["arm"].unique():
        m = out["arm"] == arm
        sub = out[m]
        flags = []
        exits: list[int] = []          # sorted exit_idx of prior trades
        stops_by_exit: list[bool] = []  # parallel was_stop flags
        for eidx, xidx, reason in zip(sub["entry_idx"], sub["exit_idx"],
                                      sub["exit_reason"]):
            pos = bisect.bisect_right(exits, eidx)
            flags.append(bool(stops_by_exit[pos - 1]) if pos else None)
            ins = bisect.bisect_right(exits, int(xidx))
            exits.insert(ins, int(xidx))
            stops_by_exit.insert(ins, reason in ("stop", "stop_gap", "stop_samebar"))
        out.loc[m, "prior_was_stopout"] = flags
    return out
