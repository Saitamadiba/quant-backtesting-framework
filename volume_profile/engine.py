"""Standalone volume-profile mechanisms (Market Profile / Auction canon).

Uses the PRIOR completed session's profile as today's reference levels
(pPOC / pVAH / pVAL), the canonical "yesterday's value area" approach — fully
causal. Three setups:

  VABREAK  value-area breakout continuation: close ACCEPTS beyond pVAH (long) /
           pVAL (short) after being inside value -> continuation. Stop back
           inside value (pVAH/pVAL - buffer) = WIDER than a candle stop, the one
           shot at a smaller fee toll. Target = fixed RR.
  POCREV   POC reversion (fade): price stretched >= ext_atr from pPOC with a
           reversal candle -> fade to pPOC. Target = pPOC (fade family).
  VAREJ    value-area edge rejection (fade): touch pVAH/pVAL, reject back ->
           fade to pPOC.

OHLC intrabar, stop-first pessimistic. Features PRE-fill causal.
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
from volume_profile.indicator import load_bars, session_profiles, node_density  # noqa: E402

NY = ZoneInfo("America/New_York")


@dataclass
class VPConfig:
    exec_tf: str = "5m"
    setup: str = "VABREAK"          # VABREAK | POCREV | VAREJ
    rr: float = 2.0
    ext_atr: float = 1.0            # POCREV: min stretch from POC (ATR)
    stop_buffer_atr: float = 0.15
    min_stop_atr: float = 0.10
    rr_max: float = 25.0
    max_hold_bars: int = 96
    max_trades_per_day: int = 3
    costs: TransactionCosts = field(default_factory=TransactionCosts)


def _confirm(side, i, o, h, l, c):
    pj = i - 1
    if side == 1:
        eng = (c[i] > o[i]) and (c[pj] < o[pj]) and (c[i] >= o[pj]) and (o[i] <= c[pj])
        rng = h[i] - l[i]
        ham = rng > 0 and (min(o[i], c[i]) - l[i]) >= 2 * abs(c[i] - o[i]) and (h[i] - max(o[i], c[i])) <= 0.4 * rng
        return eng or ham
    eng = (c[i] < o[i]) and (c[pj] > o[pj]) and (c[i] <= o[pj]) and (o[i] >= c[pj])
    rng = h[i] - l[i]
    star = rng > 0 and (h[i] - max(o[i], c[i])) >= 2 * abs(c[i] - o[i]) and (min(o[i], c[i]) - l[i]) <= 0.4 * rng
    return eng or star


def run_symbol(symbol: str, cfg: VPConfig | None = None) -> pd.DataFrame:
    cfg = cfg or VPConfig(costs=TransactionCosts.for_asset(symbol))
    df = load_bars(symbol, cfg.exec_tf)
    if df.empty:
        return pd.DataFrame()
    sp = session_profiles(symbol, cfg.exec_tf)
    if sp.empty:
        return pd.DataFrame()
    prof = {r["sess"]: r for _, r in sp.iterrows()}

    df["sess"] = df["timestamp"].dt.floor("D")
    ny = df["timestamp"].dt.tz_convert(NY)
    df["ny_date"] = ny.dt.date
    df["ny_hour"] = ny.dt.hour
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    atr = df["atr_14"].to_numpy(float); ema200 = df["ema_200"].to_numpy(float)
    ny_hour = df["ny_hour"].to_numpy(int)
    sess_arr = df["sess"].to_numpy()
    n = len(df)
    last_in_sess = {}
    for i in range(n):
        last_in_sess[sess_arr[i]] = i

    trades: list[dict] = []
    day_trades = {}
    state = {1: {"armed": False}, -1: {"armed": False}}
    for i in range(1, n):
        s = sess_arr[i]
        r = prof.get(s)
        a = atr[i]
        if r is None or not (np.isfinite(a) and a > 0):
            continue
        ppoc, pvah, pval = r["pPOC"], r["pVAH"], r["pVAL"]
        dt = day_trades.get(s, 0)
        if dt >= cfg.max_trades_per_day:
            continue
        for side in (1, -1):
            edge = pvah if side == 1 else pval
            sig = False; entry = c[i]; stop = None
            if cfg.setup == "VABREAK":
                # acceptance beyond value edge: prior close inside value, this close beyond
                inside_prev = pval <= c[i - 1] <= pvah
                accept = (c[i] > pvah) if side == 1 else (c[i] < pval)
                if inside_prev and accept:
                    sig = True
                    stop = (edge - cfg.stop_buffer_atr * a) if side == 1 else (edge + cfg.stop_buffer_atr * a)
                    tp = entry + side * cfg.rr * abs(entry - stop)
            elif cfg.setup == "POCREV":
                stretched = ((c[i] - ppoc) * (-side)) >= cfg.ext_atr * a  # below POC->long, above->short
                if stretched and _confirm(side, i, o, h, l, c):
                    sig = True
                    ext = l[i] if side == 1 else h[i]
                    stop = (ext - cfg.stop_buffer_atr * a) if side == 1 else (ext + cfg.stop_buffer_atr * a)
                    tp = ppoc
            elif cfg.setup == "VAREJ":
                touched = (h[i] >= pvah) if side == -1 else (l[i] <= pval)
                # short rejection at VAH, long rejection at VAL
                rej_side = -1 if side == -1 else 1
                if side == -1 and h[i] >= pvah and _confirm(-1, i, o, h, l, c):
                    sig = True; entry = c[i]; stop = h[i] + cfg.stop_buffer_atr * a; tp = ppoc
                elif side == 1 and l[i] <= pval and _confirm(1, i, o, h, l, c):
                    sig = True; entry = c[i]; stop = l[i] - cfg.stop_buffer_atr * a; tp = ppoc
            if not sig or stop is None:
                continue
            risk = (entry - stop) * side
            if risk <= 0:
                continue
            if risk < cfg.min_stop_atr * a:
                risk = cfg.min_stop_atr * a; stop = entry - side * risk
                if cfg.setup in ("POCREV", "VAREJ"):
                    tp = ppoc
                else:
                    tp = entry + side * cfg.rr * risk
            if (tp - entry) * side <= 0 or abs(tp - entry) / risk > cfg.rr_max:
                continue
            res = _resolve(o, h, l, c, i + 1, side, entry, stop, tp, cfg.max_hold_bars, last_in_sess[s])
            stop_pct = risk / entry
            fee_tk = cfg.costs.round_trip_cost_pct / stop_pct
            half = cfg.costs.round_trip_cost_pct / 2.0
            fee_mm = ((half + 0.0002) / stop_pct) if res["reason"] in ("tp", "tp_gap") else fee_tk
            trades.append({
                "symbol": symbol, "setup": cfg.setup, "ny_date": df["ny_date"].iloc[i],
                "entry_time": df["timestamp"].iloc[i], "side": "long" if side == 1 else "short",
                "entry": entry, "stop": stop, "tp": tp, "gross_r": res["gross_r"],
                "net_taker_r": res["gross_r"] - fee_tk, "net_mm_r": res["gross_r"] - fee_mm,
                "fee_r": fee_tk, "exit_reason": res["reason"], "bars_held": res["bars_held"],
                "mae_r": res["mae_r"], "mfe_r": res["mfe_r"],
                "entry_hour_ny": int(ny_hour[i]), "stop_pct": stop_pct, "stop_atr": risk / a,
                "rr_planned": abs(tp - entry) / risk,
                "dist_pPOC_atr": (entry - ppoc) / a, "node_density": node_density(entry, r),
                "prior_trades_today": int(dt),
            })
            day_trades[s] = dt + 1
            break
    return pd.DataFrame(trades)


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
        gs = (o[j] <= stop) if side == 1 else (o[j] >= stop)
        gt = (o[j] >= tp) if side == 1 else (o[j] <= tp)
        hs = (l[j] <= stop) if side == 1 else (h[j] >= stop)
        ht = (h[j] >= tp) if side == 1 else (l[j] <= tp)
        if gs:
            return {"gross_r": (o[j] - entry) / risk * side, "reason": "stop_gap", "bars_held": j - start + 1, "mae_r": mae, "mfe_r": mfe}
        if gt:
            return {"gross_r": (o[j] - entry) / risk * side, "reason": "tp_gap", "bars_held": j - start + 1, "mae_r": mae, "mfe_r": mfe}
        if hs:
            return {"gross_r": -1.0, "reason": "stop", "bars_held": j - start + 1, "mae_r": -1.0, "mfe_r": mfe}
        if ht:
            return {"gross_r": (tp - entry) / risk * side, "reason": "tp", "bars_held": j - start + 1, "mae_r": mae, "mfe_r": mfe}
    j = end
    return {"gross_r": (c[j] - entry) / risk * side, "reason": "time", "bars_held": end - start + 1, "mae_r": mae, "mfe_r": mfe}
