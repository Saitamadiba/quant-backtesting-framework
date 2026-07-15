"""NY 4H Range Reversal — backtest engine.

Builds the first-4H-candle range of each America/New_York day, detects
body-close breakouts followed by body-close re-entries on the execution
timeframe, and simulates fade entries with stop = breakout extreme and a
fixed 2R target. Records a full per-trade feature vector for the regime /
winner-loser pattern analysis.

See SPEC.md for the formal rules and every ambiguity resolution.
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
class NY4HConfig:
    range_start_hour: int = 0        # NY hour the reference 4H window opens
    range_hours: int = 4
    rr: float = 2.0                  # fixed take-profit multiple
    max_hold_bars: int = 576         # 48h on 5m — safety valve, books time-exit R
    min_range_bars: int = 40         # of 48 expected 5m bars in the 4H window
    exec_tf: str = "5m"
    max_stop_atr: float | None = None  # optional cap: skip if stop > X * atr15m
    costs: TransactionCosts = field(default_factory=TransactionCosts)


def load_bars(symbol: str, timeframe: str = "5m") -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    df = con.execute(
        "SELECT timestamp, open, high, low, close, volume, atr_14, ema_200 "
        "FROM ohlcv_data WHERE symbol = ? AND timeframe = ? ORDER BY timestamp",
        [symbol, timeframe],
    ).df()
    con.close()
    df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
    df["timestamp"] = (pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
                       .astype("datetime64[ns, UTC]"))
    return df


def _tf_minutes(tf: str) -> int:
    return {"5m": 5, "15m": 15, "1h": 60}[tf]


def run_symbol(symbol: str, cfg: NY4HConfig | None = None) -> pd.DataFrame:
    """Run the full backtest for one symbol. Returns the per-trade DataFrame."""
    cfg = cfg or NY4HConfig(costs=TransactionCosts.for_asset(symbol))
    df = load_bars(symbol, cfg.exec_tf)
    if df.empty:
        return pd.DataFrame()

    tf_min = _tf_minutes(cfg.exec_tf)
    bars_per_range = cfg.range_hours * 60 // tf_min
    min_range_bars = min(cfg.min_range_bars, bars_per_range - 2)

    ts_ny = df["timestamp"].dt.tz_convert(NY)
    df["ny_date"] = ts_ny.dt.date
    df["ny_hour"] = ts_ny.dt.hour
    df["dow"] = ts_ny.dt.dayofweek
    df["vol20"] = df["volume"].rolling(20).mean()

    # 15m ATR for dimensionless features (works for any exec TF)
    atr15 = _atr15_series(symbol)
    df["atr15"] = pd.merge_asof(
        df[["timestamp"]], atr15, left_on="timestamp", right_on="time", direction="backward"
    )["atr15"].to_numpy()

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    vol20 = df["vol20"].to_numpy()
    atr15v = df["atr15"].to_numpy()
    ema200 = df["ema_200"].to_numpy()
    ny_hour = df["ny_hour"].to_numpy()
    dow_arr = df["dow"].to_numpy()
    n = len(df)

    range_end_hour = cfg.range_start_hour + cfg.range_hours
    in_window = (ny_hour >= cfg.range_start_hour) & (ny_hour < range_end_hour)

    trades: list[dict] = []
    for day, idx in df.groupby("ny_date").indices.items():
        idx = np.asarray(idx)
        rng_idx = idx[in_window[idx]]
        trad_idx = idx[ny_hour[idx] >= range_end_hour]
        if len(rng_idx) < min_range_bars or len(trad_idx) < 2:
            continue
        rhi = h[rng_idx].max()
        rlo = l[rng_idx].min()
        if not np.isfinite(rhi) or not np.isfinite(rlo) or rhi <= rlo:
            continue
        rng_sz = rhi - rlo
        range_close_i = rng_idx.max()

        day_signals = 0
        side_signals = {1: 0, -1: 0}
        # state per side: None or dict(armed bar, extreme)
        state = {1: None, -1: None}  # 1 = long (break below low), -1 = short

        for i in trad_idx:
            ci = c[i]
            inside = rlo < ci < rhi
            for side in (1, -1):
                st = state[side]
                lvl = rlo if side == 1 else rhi
                broke_out = ci < rlo if side == 1 else ci > rhi
                if st is None:
                    if broke_out:
                        state[side] = {
                            "arm_i": i,
                            "ext": l[i] if side == 1 else h[i],
                        }
                    continue
                # armed: update excursion extreme (inclusive of this bar)
                st["ext"] = min(st["ext"], l[i]) if side == 1 else max(st["ext"], h[i])
                if broke_out:
                    continue  # still outside
                if not inside:
                    # closed beyond the OPPOSITE level — setup invalidated
                    state[side] = None
                    continue
                # TRIGGER: closed back inside — enter at this close
                entry = ci
                stop = st["ext"]
                risk = (entry - stop) * side
                if risk <= 0:
                    state[side] = None
                    continue
                tp = entry + side * cfg.rr * risk
                a15 = atr15v[i]
                if cfg.max_stop_atr and np.isfinite(a15) and risk > cfg.max_stop_atr * a15:
                    state[side] = None
                    continue
                res = _resolve(o, h, l, c, i + 1, side, entry, stop, tp, cfg.max_hold_bars)
                stop_pct = risk / entry
                fee_r = cfg.costs.round_trip_cost_pct / stop_pct
                arm_i = st["arm_i"]
                sweep_depth = (lvl - stop) * side
                trades.append({
                    "symbol": symbol,
                    "ny_date": day,
                    "entry_time": df["timestamp"].iloc[i] + pd.Timedelta(minutes=tf_min),
                    "side": "long" if side == 1 else "short",
                    "entry": entry, "stop": stop, "tp": tp,
                    "gross_r": res["gross_r"],
                    "net_r": res["gross_r"] - fee_r,
                    "fee_r": fee_r,
                    "exit_reason": res["reason"],
                    "bars_held": res["bars_held"],
                    "mae_r": res["mae_r"], "mfe_r": res["mfe_r"],
                    "same_day_resolve": res["bars_held"] <= (idx.max() - i),
                    # --- entry-time features ---
                    "entry_hour_ny": int(ny_hour[i]),
                    "dow": int(dow_arr[i]),
                    "mins_since_range": (i - range_close_i) * tf_min,
                    "range_pct": rng_sz / rlo,
                    "range_vs_atr15": rng_sz / a15 if np.isfinite(a15) and a15 > 0 else np.nan,
                    "sweep_depth_pct": sweep_depth / entry,
                    "sweep_depth_atr": sweep_depth / a15 if np.isfinite(a15) and a15 > 0 else np.nan,
                    "bars_outside": int(i - arm_i),
                    "reentry_pos": (entry - rlo) / rng_sz,
                    "stop_pct": stop_pct,
                    "tp_fits_in_range": (tp <= rhi) if side == 1 else (tp >= rlo),
                    "tp_room_frac": ((rhi - entry) if side == 1 else (entry - rlo)) / (cfg.rr * risk),
                    "breakout_vol_ratio": (v[arm_i] / vol20[arm_i])
                        if np.isfinite(vol20[arm_i]) and vol20[arm_i] > 0 else np.nan,
                    "trend_align": bool((entry > ema200[i]) if side == 1 else (entry < ema200[i]))
                        if np.isfinite(ema200[i]) else None,
                    "dist_ema200_pct": (entry - ema200[i]) / ema200[i] * side
                        if np.isfinite(ema200[i]) else np.nan,
                    "prior_signals_today": day_signals,
                    "prior_sameside_today": side_signals[side],
                })
                day_signals += 1
                side_signals[side] += 1
                state[side] = None  # re-arm from scratch

    out = pd.DataFrame(trades)
    return out


def _resolve(o, h, l, c, start: int, side: int, entry: float, stop: float,
             tp: float, max_hold: int) -> dict:
    """Walk bars forward from `start` until stop/target/time exit.

    Gap-through fills at the open; stop+target in one bar -> stop first
    (pessimistic).
    """
    n = len(c)
    risk = (entry - stop) * side
    mae = 0.0
    mfe = 0.0
    end = min(n, start + max_hold)
    for j in range(start, end):
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
        if hit_stop:  # pessimistic: stop wins ties
            return {"gross_r": -1.0, "reason": "stop",
                    "bars_held": j - start + 1, "mae_r": -1.0, "mfe_r": mfe}
        if hit_tp:
            return {"gross_r": (tp - entry) / risk * side, "reason": "tp",
                    "bars_held": j - start + 1, "mae_r": mae, "mfe_r": mfe}
    j = end - 1
    return {"gross_r": (c[j] - entry) / risk * side, "reason": "time",
            "bars_held": end - start, "mae_r": mae, "mfe_r": mfe}


_ATR15_CACHE: dict[str, pd.DataFrame] = {}


def _atr15_series(symbol: str) -> pd.DataFrame:
    """Wilder ATR14 on 15m bars, indexed by bar CLOSE time (causal)."""
    if symbol in _ATR15_CACHE:
        return _ATR15_CACHE[symbol]
    df = load_bars(symbol, "15m")
    hh, ll, cc = df["high"], df["low"], df["close"]
    pc = cc.shift(1)
    tr = pd.concat([(hh - ll), (hh - pc).abs(), (ll - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    out = pd.DataFrame({"time": df["timestamp"] + pd.Timedelta(minutes=15),
                        "atr15": atr.to_numpy()})
    _ATR15_CACHE[symbol] = out
    return out
