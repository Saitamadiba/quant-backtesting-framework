"""HTF-Bias Structure Scalper — backtest engine.

1h structural bias -> 5m BOS -> retest of the broken swing -> candlestick
confirmation -> 2R continuation. All pivots consumed at confirmation time
(N-bar lag), so nothing in the state machine sees the future.

Reuses the ny4h_range_reversal resolver / loaders / regime plumbing.
See SPEC.md for every formalization decision.
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

NY = ZoneInfo("America/New_York")


@dataclass
class StructConfig:
    swing_n: int = 3
    htf_swing_n: int = 3
    rr: float = 2.0
    retest_tol_atr: float = 0.25
    buffer_atr: float = 0.25
    max_wait_retest: int = 48
    max_wait_confirm: int = 12
    max_hold_bars: int = 576
    exec_tf: str = "5m"
    htf_tf: str = "1h"
    costs: TransactionCosts = field(default_factory=TransactionCosts)


def pivot_asof(h: np.ndarray, l: np.ndarray, n: int):
    """Confirmed-pivot as-of arrays.

    Returns (sh_lvl, sh_idx, sl_lvl, sl_idx): at bar j, the level/index of
    the most recent swing high/low whose confirmation bar (pivot + n) <= j.
    """
    hs, ls = pd.Series(h), pd.Series(l)
    left_h = hs.rolling(n).max().shift(1)
    right_h = hs[::-1].rolling(n).max().shift(1)[::-1]
    is_ph = (hs >= left_h) & (hs > right_h)
    left_l = ls.rolling(n).min().shift(1)
    right_l = ls[::-1].rolling(n).min().shift(1)[::-1]
    is_pl = (ls <= left_l) & (ls < right_l)

    idx = pd.Series(np.arange(len(h), dtype=float))
    sh_lvl = hs.where(is_ph).shift(n).ffill().to_numpy()
    sh_idx = idx.where(is_ph).shift(n).ffill().fillna(-1).to_numpy().astype(int)
    sl_lvl = ls.where(is_pl).shift(n).ffill().to_numpy()
    sl_idx = idx.where(is_pl).shift(n).ffill().fillna(-1).to_numpy().astype(int)
    return sh_lvl, sh_idx, sl_lvl, sl_idx


def htf_bias_frame(symbol: str, cfg: StructConfig) -> pd.DataFrame:
    """Per-1h-bar bias state, keyed by bar CLOSE time.

    bias: +1 bull / -1 bear / 0 none. Flips on a fresh structural close.
    Also emits the last same-direction break time and the HH/HL vs LH/LL
    sequence label of the most recent confirmed swings.
    """
    if cfg.htf_tf == "1d":
        # no local daily table — resample 4h bars to UTC calendar days
        raw = load_bars(symbol, "4h").set_index("timestamp")
        df = (raw.resample("1D")
              .agg(open=("open", "first"), high=("high", "max"),
                   low=("low", "min"), close=("close", "last"))
              .dropna().reset_index())
    else:
        df = load_bars(symbol, cfg.htf_tf)
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    sh_lvl, sh_idx, sl_lvl, sl_idx = pivot_asof(h, l, cfg.htf_swing_n)

    n = len(df)
    bias = np.zeros(n, dtype=int)
    last_break = np.full(n, -1, dtype=int)
    seq = np.full(n, "na", dtype=object)

    # rolling last-2 confirmed swing levels for the sequence label
    highs2: list[float] = []
    lows2: list[float] = []
    cur_bias, cur_break = 0, -1
    used_hi, used_lo = -1, -1
    prev_sh_idx, prev_sl_idx = -1, -1
    for j in range(n):
        if sh_idx[j] != prev_sh_idx and sh_idx[j] >= 0:
            highs2.append(sh_lvl[j])
            highs2 = highs2[-2:]
            prev_sh_idx = sh_idx[j]
        if sl_idx[j] != prev_sl_idx and sl_idx[j] >= 0:
            lows2.append(sl_lvl[j])
            lows2 = lows2[-2:]
            prev_sl_idx = sl_idx[j]
        if len(highs2) == 2 and len(lows2) == 2:
            if highs2[1] > highs2[0] and lows2[1] > lows2[0]:
                seq[j] = "up"
            elif highs2[1] < highs2[0] and lows2[1] < lows2[0]:
                seq[j] = "down"
            else:
                seq[j] = "mixed"
        bull = sh_idx[j] >= 0 and sh_idx[j] != used_hi and c[j] > sh_lvl[j]
        bear = sl_idx[j] >= 0 and sl_idx[j] != used_lo and c[j] < sl_lvl[j]
        if bull and not bear:
            cur_bias, cur_break, used_hi = 1, j, sh_idx[j]
        elif bear and not bull:
            cur_bias, cur_break, used_lo = -1, j, sl_idx[j]
        bias[j] = cur_bias
        last_break[j] = cur_break

    tfm = {"1h": 60, "4h": 240, "1d": 1440}[cfg.htf_tf]
    close_time = df["timestamp"] + pd.Timedelta(minutes=tfm)
    ct_ns = close_time.astype("int64").to_numpy()
    break_ns = np.full(n, np.nan)
    ok = last_break >= 0
    break_ns[ok] = ct_ns[last_break[ok]]
    return pd.DataFrame({
        "time": close_time,
        "htf_bias": bias,
        "htf_seq": seq,
        "htf_break_ns": break_ns,
    })


IDLE, AWAIT_RETEST, AWAIT_CONFIRM = 0, 1, 2


def run_symbol(symbol: str, cfg: StructConfig | None = None) -> pd.DataFrame:
    cfg = cfg or StructConfig(costs=TransactionCosts.for_asset(symbol))
    df = load_bars(symbol, cfg.exec_tf)
    if df.empty:
        return pd.DataFrame()
    tf_min = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}[cfg.exec_tf]

    htf = htf_bias_frame(symbol, cfg)
    close_time = df["timestamp"] + pd.Timedelta(minutes=tf_min)
    m = pd.merge_asof(pd.DataFrame({"t": close_time}), htf,
                      left_on="t", right_on="time", direction="backward")
    htf_bias = m["htf_bias"].fillna(0).to_numpy().astype(int)
    htf_seq = m["htf_seq"].fillna("na").to_numpy()
    htf_break_ns = m["htf_break_ns"].to_numpy()
    ts_ns = df["timestamp"].astype("int64").to_numpy()

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

    def session(hr: int) -> str:
        if 3 <= hr < 8:
            return "london"
        if 8 <= hr < 11:
            return "overlap"
        if 11 <= hr < 17:
            return "ny"
        return "off"

    def confirm_pattern(j: int, side: int, L: float, touched_this_bar: bool):
        """Return confirm type at bar j for `side`, or None."""
        rng = h[j] - l[j]
        if rng <= 0:
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
            if (touched_this_bar and bull and c[j] > L
                    and c[j] >= l[j] + 0.65 * rng):
                return "reject"
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
            if (touched_this_bar and bear and c[j] < L
                    and c[j] <= h[j] - 0.65 * rng):
                return "reject"
        return None

    trades: list[dict] = []
    day_counts: dict = {}
    mach = {1: {"state": IDLE}, -1: {"state": IDLE}}
    used_piv = {1: -1, -1: -1}

    warm = max(cfg.swing_n * 2 + 2, 21)
    for j in range(warm, n):
        for side in (1, -1):
            st = mach[side]
            bias_ok = htf_bias[j] == side
            if st["state"] == IDLE:
                if not bias_ok:
                    continue
                lvl = sh_lvl[j] if side == 1 else sl_lvl[j]
                piv = sh_idx[j] if side == 1 else sl_idx[j]
                if piv < 0 or piv == used_piv[side] or not np.isfinite(lvl):
                    continue
                broke = c[j] > lvl if side == 1 else c[j] < lvl
                if broke:
                    used_piv[side] = piv
                    st.update(state=AWAIT_RETEST, L=lvl, bos_i=j,
                              impulse=h[j] if side == 1 else l[j],
                              pull=l[j] if side == 1 else h[j],
                              touch_i=-1)
                continue

            # --- AWAIT_RETEST / AWAIT_CONFIRM ---
            opp_lvl = sl_lvl[j] if side == 1 else sh_lvl[j]
            opp_break = (np.isfinite(opp_lvl)
                         and (c[j] < opp_lvl if side == 1 else c[j] > opp_lvl))
            if not bias_ok or opp_break:
                st.clear(); st["state"] = IDLE
                continue

            L = st["L"]
            a = atr[j]
            if not np.isfinite(a) or a <= 0:
                st.clear(); st["state"] = IDLE
                continue
            st["pull"] = min(st["pull"], l[j]) if side == 1 else max(st["pull"], h[j])

            if st["state"] == AWAIT_RETEST:
                st["impulse"] = (max(st["impulse"], h[j]) if side == 1
                                 else min(st["impulse"], l[j]))
                tol = cfg.retest_tol_atr * a
                touched = (l[j] <= L + tol) if side == 1 else (h[j] >= L - tol)
                if touched:
                    st["state"] = AWAIT_CONFIRM
                    st["touch_i"] = j
                elif j - st["bos_i"] > cfg.max_wait_retest:
                    st.clear(); st["state"] = IDLE
                    continue

            if st["state"] == AWAIT_CONFIRM:
                ctype = confirm_pattern(j, side, L, st["touch_i"] == j)
                if ctype is None:
                    if j - st["touch_i"] > cfg.max_wait_confirm:
                        st.clear(); st["state"] = IDLE
                    continue
                # --- ENTRY at this close, both stop arms ---
                entry = c[j]
                buf = cfg.buffer_atr * a
                confirm_ext = l[j] if side == 1 else h[j]
                pull_ext = st["pull"]
                day = ny_date[j]
                nprior = day_counts.get(day, 0)
                since_htf = ((ts_ns[j] - htf_break_ns[j]) / 3.6e12
                             if np.isfinite(htf_break_ns[j]) else np.nan)
                imp_span = abs(st["impulse"] - L)
                common = {
                    "symbol": symbol, "ny_date": day,
                    "entry_time": df["timestamp"].iloc[j] + pd.Timedelta(minutes=tf_min),
                    "side": "long" if side == 1 else "short",
                    "confirm_type": ctype,
                    "session": session(int(ny_hour[j])),
                    "entry_hour_ny": int(ny_hour[j]), "dow": int(dow[j]),
                    "structure_seq": htf_seq[j],
                    "hours_since_htf_break": float(since_htf),
                    "bos_dist_atr": abs(c[st["bos_i"]] - L) / atr[st["bos_i"]]
                        if np.isfinite(atr[st["bos_i"]]) and atr[st["bos_i"]] > 0 else np.nan,
                    "bos_vol_ratio": v[st["bos_i"]] / vol20[st["bos_i"]]
                        if np.isfinite(vol20[st["bos_i"]]) and vol20[st["bos_i"]] > 0 else np.nan,
                    "bars_to_retest": int(st["touch_i"] - st["bos_i"]),
                    "bars_to_confirm": int(j - st["touch_i"]),
                    "retest_depth_atr": (L - pull_ext) / a * side,
                    "retrace_frac": abs(st["impulse"] - pull_ext) / imp_span
                        if imp_span > 0 else np.nan,
                    "confirm_body_frac": abs(c[j] - o[j]) / (h[j] - l[j]),
                    "confirm_range_atr": (h[j] - l[j]) / a,
                    "dist_ema200_pct": (entry - ema200[j]) / ema200[j] * side
                        if np.isfinite(ema200[j]) else np.nan,
                    "trend_align": bool((entry > ema200[j]) if side == 1
                                        else (entry < ema200[j]))
                        if np.isfinite(ema200[j]) else None,
                    "prior_signals_today": nprior,
                }
                for arm, ext in (("confirm", confirm_ext), ("pullback", pull_ext)):
                    stop = ext - buf * side
                    risk = (entry - stop) * side
                    if risk <= 0:
                        continue
                    tp = entry + side * cfg.rr * risk
                    res = _resolve(o, h, l, c, j + 1, side, entry, stop, tp,
                                   cfg.max_hold_bars)
                    stop_pct = risk / entry
                    fee_r = cfg.costs.round_trip_cost_pct / stop_pct
                    trades.append({**common, "arm": arm,
                                   "entry": entry, "stop": stop, "tp": tp,
                                   "stop_pct": stop_pct, "fee_r": fee_r,
                                   "gross_r": res["gross_r"],
                                   "net_r": res["gross_r"] - fee_r,
                                   "exit_reason": res["reason"],
                                   "bars_held": res["bars_held"],
                                   "mae_r": res["mae_r"], "mfe_r": res["mfe_r"]})
                day_counts[day] = nprior + 1
                st.clear(); st["state"] = IDLE

    return pd.DataFrame(trades)
