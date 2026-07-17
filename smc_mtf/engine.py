"""Multi-TF SMC backtest engine (HTF FVG -> LTF CHOCH -> LTF FVG CE).

See SPEC.md. Reuses ny4h loaders and structure_scalper pivots; every
causality rule from the series is enforced (HTF zones active post-close,
lagged pivots, CE limits live from the bar after the FVG completes).
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
from ny4h_range_reversal.engine import load_bars  # noqa: E402
from structure_scalper.engine import pivot_asof  # noqa: E402

NY = ZoneInfo("America/New_York")


@dataclass
class SMCConfig:
    htf_tf: str = "1h"
    ltf_tf: str = "5m"
    swing_n: int = 3
    min_zone_atr: float = 0.15
    zone_max_age_htf: int = 96
    choch_window: int = 48       # LTF bars after zone touch
    fvg_window: int = 12         # LTF bars after CHOCH
    fill_window: int = 24        # LTF bars for the CE limit
    rr: float = 4.0
    stop_mode: str = "candle"    # "candle" (v1) | "atr" (fixed k x ATR at FVG bar)
    stop_atr_mult: float = 1.0
    ce_frac: float = 0.5         # entry depth in the LTF FVG (0.5 = CE midpoint)
    min_leg_retrace: float = 0.0  # fib filter: skip fills shallower than this
    min_risk_atr: float = 0.10   # skip degenerate stops (< 0.1 x LTF ATR)
    session_gap_exit: bool = False  # flatten before session gaps (NQ)
    max_hold_bars: int = 576
    max_active_zones: int = 12   # per side
    costs: TransactionCosts = field(default_factory=TransactionCosts)


def find_fvgs(df: pd.DataFrame, atr: np.ndarray, min_atr: float):
    """3-candle FVGs. Returns list of dicts with close-time activation."""
    h = df["high"].to_numpy(); l = df["low"].to_numpy()
    out = []
    for i in range(2, len(df)):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_atr * a:
            out.append({"dir": 1, "top": l[i], "bot": h[i - 2],
                        "born_i": i, "size_atr": (l[i] - h[i - 2]) / a})
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_atr * a:
            out.append({"dir": -1, "top": l[i - 2], "bot": h[i],
                        "born_i": i, "size_atr": (l[i - 2] - h[i]) / a})
    return out


def resolve_be(o, h, l, c, start, side, entry, stop, tp, max_hold,
               ts_ns=None, gap_ns=None):
    """One walk, two variants: plain stop/tp and BE-after-1R.

    Pessimistic: gaps fill at open; stop beats tp; BE beats tp. If
    ts_ns/gap_ns given (session-gapped markets like NQ), open positions
    are flattened at the close of the last bar BEFORE a session gap.
    """
    n = len(c)
    risk = (entry - stop) * side
    tp1 = entry + side * risk
    end = min(n, start + max_hold)
    out = {"mae_r": 0.0, "mfe_r": 0.0}
    open_p = open_b = True
    be_armed = False
    for j in range(start, end):
        if not (open_p or open_b):
            break
        if ts_ns is not None and j > start \
                and ts_ns[j] - ts_ns[j - 1] > gap_ns:
            mark = (c[j - 1] - entry) * side / risk
            if open_p:
                out["r_plain"] = mark; out["reason_plain"] = "sess"
                out["bars_plain"] = j - start; open_p = False
            if open_b:
                out["r_be"] = max(mark, 0.0) if be_armed else mark
                out["reason_be"] = "sess"; open_b = False
            break
        worst = ((l[j] if side == 1 else h[j]) - entry) * side / risk
        best = ((h[j] if side == 1 else l[j]) - entry) * side / risk
        out["mae_r"] = min(out["mae_r"], worst)
        out["mfe_r"] = max(out["mfe_r"], best)
        gap_stop = (o[j] - stop) * side <= 0
        gap_tp = (o[j] - tp) * side >= 0
        hit_stop = ((l[j] if side == 1 else h[j]) - stop) * side <= 0
        hit_tp = ((h[j] if side == 1 else l[j]) - tp) * side >= 0
        if open_p:
            if gap_stop:
                out["r_plain"] = (o[j] - entry) * side / risk
                out["reason_plain"] = "stop_gap"; out["bars_plain"] = j - start + 1
                open_p = False
            elif gap_tp:
                out["r_plain"] = (tp - entry) * side / risk  # limit fills AT tp
                out["reason_plain"] = "tp_gap"; out["bars_plain"] = j - start + 1
                open_p = False
            elif hit_stop:
                out["r_plain"] = -1.0
                out["reason_plain"] = "stop"; out["bars_plain"] = j - start + 1
                open_p = False
            elif hit_tp:
                out["r_plain"] = (tp - entry) * side / risk
                out["reason_plain"] = "tp"; out["bars_plain"] = j - start + 1
                open_p = False
        if open_b:
            eff_stop = entry if be_armed else stop
            g_stop = (o[j] - eff_stop) * side <= 0
            h_stop = ((l[j] if side == 1 else h[j]) - eff_stop) * side <= 0
            if g_stop:
                out["r_be"] = (o[j] - entry) * side / risk
                out["reason_be"] = "be_gap" if be_armed else "stop_gap"
                open_b = False
            elif gap_tp:
                out["r_be"] = (tp - entry) * side / risk  # limit fills AT tp
                out["reason_be"] = "tp_gap"; open_b = False
            elif h_stop and not be_armed:
                # could this same bar have armed BE first? pessimistic: no
                out["r_be"] = -1.0; out["reason_be"] = "stop"; open_b = False
            elif be_armed and h_stop:
                out["r_be"] = 0.0; out["reason_be"] = "be"; open_b = False
            elif hit_tp:
                out["r_be"] = (tp - entry) * side / risk
                out["reason_be"] = "tp"; open_b = False
            elif ((h[j] if side == 1 else l[j]) - tp1) * side >= 0:
                be_armed = True
    j = end - 1
    if open_p:
        out["r_plain"] = (c[j] - entry) * side / risk
        out["reason_plain"] = "time"; out["bars_plain"] = end - start
    if open_b:
        mark = (c[j] - entry) * side / risk
        out["r_be"] = max(mark, 0.0) if be_armed else mark
        out["reason_be"] = "time"
    return out


IDLE, AWAIT_CHOCH, AWAIT_FVG, AWAIT_FILL = 0, 1, 2, 3


def run_symbol(symbol: str, cfg: SMCConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SMCConfig(costs=TransactionCosts.for_asset(symbol))
    ltf = load_bars(symbol, cfg.ltf_tf)
    if cfg.htf_tf == "1d":
        raw = load_bars(symbol, "4h").set_index("timestamp")
        htf = (raw.resample("1D")
               .agg(open=("open", "first"), high=("high", "max"),
                    low=("low", "min"), close=("close", "last"))
               .dropna().reset_index())
        pc = htf["close"].shift(1)
        tr = pd.concat([(htf["high"] - htf["low"]), (htf["high"] - pc).abs(),
                        (htf["low"] - pc).abs()], axis=1).max(axis=1)
        htf["atr_14"] = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    else:
        htf = load_bars(symbol, cfg.htf_tf)
    if ltf.empty or htf.empty:
        return pd.DataFrame()
    tf_min = {"5m": 5, "15m": 15, "1h": 60}[cfg.ltf_tf]
    htf_min = {"1h": 60, "4h": 240, "1d": 1440}[cfg.htf_tf]

    htf_atr = htf["atr_14"].to_numpy()
    zones = find_fvgs(htf, htf_atr, cfg.min_zone_atr)
    htf_close_ns = (htf["timestamp"] + pd.Timedelta(minutes=htf_min)).astype("int64").to_numpy()
    for z in zones:
        z["active_ns"] = htf_close_ns[z["born_i"]]
        z["expire_ns"] = z["active_ns"] + cfg.zone_max_age_htf * htf_min * 60 * 10**9

    ts_ns = ltf["timestamp"].astype("int64").to_numpy()
    ts_ny = ltf["timestamp"].dt.tz_convert(NY)
    ny_date = ts_ny.dt.date.to_numpy()
    ny_hour = ts_ny.dt.hour.to_numpy()
    dow = ts_ny.dt.dayofweek.to_numpy()

    o = ltf["open"].to_numpy(); h = ltf["high"].to_numpy()
    l = ltf["low"].to_numpy(); c = ltf["close"].to_numpy()
    v = ltf["volume"].to_numpy()
    vol20 = ltf["volume"].rolling(20).mean().to_numpy()
    atr = ltf["atr_14"].to_numpy()
    ema200 = ltf["ema_200"].to_numpy()
    n = len(ltf)

    sh_lvl, sh_idx, sl_lvl, sl_idx = pivot_asof(h, l, cfg.swing_n)
    # LTF micro-trend seq (same recipe as fib_bos)
    seq = np.full(n, "na", dtype=object)
    highs2: list = []; lows2: list = []
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

    trades: list = []
    active: dict[int, list] = {1: [], -1: []}   # episode dicts per side
    zi = 0
    zones.sort(key=lambda z: z["active_ns"])

    warm = 30
    for j in range(warm, n):
        # activate newly closed HTF zones
        while zi < len(zones) and zones[zi]["active_ns"] <= ts_ns[j - 1]:
            z = zones[zi]; zi += 1
            side = z["dir"]
            eps = active[side]
            eps.append({"z": z, "state": IDLE})
            if len(eps) > cfg.max_active_zones:
                eps.pop(0)

        for side in (1, -1):
            keep = []
            for ep in active[side]:
                z = ep["z"]
                # global invalidations
                if ts_ns[j] > z["expire_ns"]:
                    continue
                filled_thru = (c[j] - z["bot"]) * side < 0 if side == 1 else \
                              (c[j] - z["top"]) * side < 0
                zone_top = z["top"] if side == 1 else z["bot"]
                zone_far = z["bot"] if side == 1 else z["top"]
                if (c[j] - (zone_far - side * atr[j])) * side < 0 \
                        and np.isfinite(atr[j]):
                    continue  # deep close through the zone — dead
                st = ep["state"]

                if st == IDLE:
                    touched = (l[j] <= zone_top) if side == 1 else (h[j] >= zone_top)
                    if touched:
                        if seq[j] == ("down" if side == 1 else "up"):
                            ep.update(state=AWAIT_CHOCH, touch_i=j,
                                      epi_ext=l[j] if side == 1 else h[j])
                        # wrong micro-trend at touch: zone stays; retouch later
                    keep.append(ep)
                    continue

                if st == AWAIT_CHOCH:
                    ep["epi_ext"] = (min(ep["epi_ext"], l[j]) if side == 1
                                     else max(ep["epi_ext"], h[j]))
                    if j - ep["touch_i"] > cfg.choch_window:
                        ep["state"] = IDLE  # window over; allow fresh touch
                        keep.append(ep)
                        continue
                    lvl = sh_lvl[j] if side == 1 else sl_lvl[j]
                    piv = sh_idx[j] if side == 1 else sl_idx[j]
                    if piv >= 0 and np.isfinite(lvl) and \
                            (c[j] - lvl) * side > 0:
                        a = atr[j]
                        ep.update(state=AWAIT_FVG, choch_i=j,
                                  choch_lvl=lvl,
                                  leg_hi=h[j] if side == 1 else l[j],
                                  choch_margin=abs(c[j] - lvl) / a
                                      if np.isfinite(a) and a > 0 else np.nan)
                    keep.append(ep)
                    continue

                if st == AWAIT_FVG:
                    ep["leg_hi"] = (max(ep["leg_hi"], h[j]) if side == 1
                                    else min(ep["leg_hi"], l[j]))
                    if j - ep["choch_i"] > cfg.fvg_window:
                        continue  # setup dead — zone consumed
                    # fresh LTF FVG completing at bar j in trade direction
                    if side == 1 and l[j] > h[j - 2]:
                        ce = l[j] - cfg.ce_frac * (l[j] - h[j - 2])
                        stop = l[j - 1]
                        far = h[j - 2]
                    elif side == -1 and h[j] < l[j - 2]:
                        ce = h[j] + cfg.ce_frac * (l[j - 2] - h[j])
                        stop = h[j - 1]
                        far = l[j - 2]
                    else:
                        keep.append(ep)
                        continue
                    if cfg.stop_mode == "atr":
                        if not (np.isfinite(atr[j]) and atr[j] > 0):
                            keep.append(ep)
                            continue
                        stop = ce - side * cfg.stop_atr_mult * atr[j]
                    if (ce - stop) * side <= 0:
                        keep.append(ep)
                        continue
                    ep.update(state=AWAIT_FILL, fvg_i=j, ce=ce, stop=stop,
                              fvg_far=far,
                              fvg_size_atr=abs(l[j] - h[j - 2]) / atr[j]
                                  if side == 1 else abs(l[j - 2] - h[j]) / atr[j],
                              disp_body=abs(c[j - 1] - o[j - 1]) /
                                  max(h[j - 1] - l[j - 1], 1e-12),
                              disp_range_atr=(h[j - 1] - l[j - 1]) / atr[j]
                                  if np.isfinite(atr[j]) and atr[j] > 0 else np.nan)
                    keep.append(ep)
                    continue

                if st == AWAIT_FILL:
                    # limit lives from the bar AFTER the FVG completed
                    if j == ep["fvg_i"]:
                        keep.append(ep)
                        continue
                    if j - ep["fvg_i"] > cfg.fill_window:
                        continue
                    if (c[j] - ep["fvg_far"]) * side < 0:
                        continue  # LTF FVG invalidated on a close
                    ce, stop = ep["ce"], ep["stop"]
                    touched = (l[j] <= ce) if side == 1 else (h[j] >= ce)
                    if not touched:
                        # leg may keep extending while we wait
                        ep["leg_hi"] = (max(ep["leg_hi"], h[j]) if side == 1
                                        else min(ep["leg_hi"], l[j]))
                        keep.append(ep)
                        continue
                    fill = o[j] if (o[j] - ce) * side <= 0 else ce
                    risk = (fill - stop) * side
                    a_fill = atr[j]
                    if risk <= 0 or not np.isfinite(a_fill) or a_fill <= 0 \
                            or risk < cfg.min_risk_atr * a_fill:
                        continue  # degenerate stop — not the design's intent
                    tp = fill + side * cfg.rr * risk
                    a = atr[j]
                    leg = (ep["leg_hi"] - ep["epi_ext"]) * side
                    retr = ((ep["leg_hi"] - fill) * side / leg) if leg > 0 else np.nan
                    if cfg.min_leg_retrace > 0 and \
                            not (np.isfinite(retr) and retr >= cfg.min_leg_retrace):
                        continue  # fib filter (strategy param, causal at fill)
                    if ((l[j] if side == 1 else h[j]) - stop) * side <= 0:
                        res = {"r_plain": -1.0, "reason_plain": "stop_samebar",
                               "bars_plain": 0, "r_be": -1.0,
                               "reason_be": "stop_samebar",
                               "mae_r": -1.0, "mfe_r": 0.0}
                    else:
                        res = resolve_be(
                            o, h, l, c, j + 1, side, fill, stop, tp,
                            cfg.max_hold_bars,
                            ts_ns=ts_ns if cfg.session_gap_exit else None,
                            gap_ns=4 * tf_min * 60 * 10**9)
                    stop_pct = risk / fill
                    rt = cfg.costs.round_trip_cost_pct
                    trades.append({
                        "symbol": symbol, "ny_date": ny_date[j],
                        "entry_time": ltf["timestamp"].iloc[j] + pd.Timedelta(minutes=tf_min),
                        "side": "long" if side == 1 else "short",
                        "session": session(int(ny_hour[j])),
                        "entry_hour_ny": int(ny_hour[j]), "dow": int(dow[j]),
                        "entry": fill, "stop": stop, "tp": tp,
                        "rr_planned": cfg.rr, "stop_pct": stop_pct,
                        "fee_r": rt / stop_pct,
                        "htf_zone_atr": z["size_atr"],
                        "zone_age_htf": (ts_ns[j] - z["active_ns"]) / (htf_min * 60e9),
                        "zone_pen_frac": (zone_top - ep["epi_ext"]) * side /
                            max((zone_top - zone_far) * side, 1e-12),
                        "choch_leg_atr": leg / a
                            if np.isfinite(a) and a > 0 else np.nan,
                        "choch_break_margin_atr": ep["choch_margin"],
                        "disp_body_frac": ep["disp_body"],
                        "disp_range_atr": ep["disp_range_atr"],
                        "ltf_fvg_atr": ep["fvg_size_atr"],
                        "entry_leg_retrace": retr,
                        "bars_touch_to_choch": ep["choch_i"] - ep["touch_i"],
                        "bars_choch_to_fvg": ep["fvg_i"] - ep["choch_i"],
                        "bars_to_fill": j - ep["fvg_i"],
                        "vol_ratio_choch": v[ep["choch_i"]] / vol20[ep["choch_i"]]
                            if np.isfinite(vol20[ep["choch_i"]])
                            and vol20[ep["choch_i"]] > 0 else np.nan,
                        "dist_ema200_pct": (fill - ema200[j]) / ema200[j] * side
                            if np.isfinite(ema200[j]) else np.nan,
                        "gross_r": res["r_plain"],
                        "net_r": res["r_plain"] - rt / stop_pct,
                        "exit_reason": res["reason_plain"],
                        "bars_held": res.get("bars_plain", 0),
                        "mae_r": res["mae_r"], "mfe_r": res["mfe_r"],
                        "r_be": res["r_be"], "reason_be": res["reason_be"],
                    })
                    continue  # zone consumed by the trade
            active[side] = keep
    return pd.DataFrame(trades)
