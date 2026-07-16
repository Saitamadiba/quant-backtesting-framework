"""Five-state market classifier (YouTube #4 framework, formalized).

Emits one causal state per bar: breakout / reversal / strong_trend /
wide_trend / range / unclassified, plus a direction for the directional
states. See SPEC.md for definitions and priority.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from ny4h_range_reversal.engine import load_bars  # noqa: E402
from structure_scalper.engine import pivot_asof  # noqa: E402
from regime_classifier import wilder_adx  # noqa: E402

STATES = ["breakout", "reversal", "strong_trend", "wide_trend", "range",
          "unclassified"]


@dataclass
class StateConfig:
    swing_n: int = 3
    don_win: int = 48
    reg_win: int = 96
    event_window: int = 12
    adx_trend: float = 25.0
    adx_range: float = 20.0
    slope_norm_trend: float = 2.0
    slope_norm_range: float = 1.0
    band_wide_atr: float = 3.0
    min_zero_cross: int = 4
    brk_range_atr: float = 1.5
    brk_body_frac: float = 0.55
    brk_vol_ratio: float = 1.5
    atr_pctile_range: float = 0.5
    compression_atr: float = 12.0


def _rolling_regression(c: np.ndarray, win: int):
    """Trailing-window OLS per bar: normalized slope*win, resid std, zero-crossings."""
    n = len(c)
    slope_tot = np.full(n, np.nan)
    resid_sd = np.full(n, np.nan)
    zcross = np.full(n, np.nan)
    x = np.arange(win, dtype=float)
    x -= x.mean()
    xx = (x * x).sum()
    for j in range(win, n):
        y = c[j - win:j]
        ym = y.mean()
        beta = ((x * (y - ym)).sum()) / xx
        resid = (y - ym) - beta * x
        slope_tot[j] = beta * win
        resid_sd[j] = resid.std()
        s = np.sign(resid)
        zcross[j] = int((s[1:] * s[:-1] < 0).sum())
    return slope_tot, resid_sd, zcross


def classify_symbol(symbol: str, tf: str, cfg: StateConfig | None = None) -> pd.DataFrame:
    cfg = cfg or StateConfig()
    df = load_bars(symbol, tf)
    if df.empty:
        return pd.DataFrame()
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    n = len(df)

    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    ema200 = df["ema_200"].to_numpy()
    adx = wilder_adx(df.rename(columns=str.lower), 14).to_numpy()
    atr = df["atr_14"].to_numpy()
    atr_pct = (pd.Series(atr).rolling(2000, min_periods=500)
               .rank(pct=True).to_numpy())
    vol20 = pd.Series(v).rolling(20).mean().to_numpy()
    slope5 = ema20 - np.roll(ema20, 5)
    slope5[:5] = np.nan

    don_hi = pd.Series(h).rolling(cfg.don_win).max().shift(1).to_numpy()
    don_lo = pd.Series(l).rolling(cfg.don_win).min().shift(1).to_numpy()
    slope_tot, resid_sd, zcross = _rolling_regression(c, cfg.reg_win)

    sh_lvl, sh_idx, sl_lvl, sl_idx = pivot_asof(h, l, cfg.swing_n)
    # HH/HL sequence label
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

    state = np.full(n, "unclassified", dtype=object)
    direction = np.zeros(n, dtype=int)
    brk_until, brk_dir = -1, 0
    rev_until, rev_dir = -1, 0
    used_rev_lo, used_rev_hi = -1, -1
    brk_events, false_breaks = [], []

    warm = max(cfg.reg_win, 300)
    for j in range(warm, n):
        a = atr[j]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(adx[j]):
            continue
        rng = h[j] - l[j]
        body = abs(c[j] - o[j])
        stack = 1 if (ema20[j] > ema50[j] > ema200[j]) else (
            -1 if (ema20[j] < ema50[j] < ema200[j]) else 0)

        # --- breakout event ---
        vol_ok = np.isfinite(vol20[j]) and vol20[j] > 0 and v[j] >= cfg.brk_vol_ratio * vol20[j]
        big = rng >= cfg.brk_range_atr * a and body >= cfg.brk_body_frac * max(rng, 1e-12)
        if np.isfinite(don_hi[j]) and big and vol_ok:
            if c[j] > don_hi[j]:
                brk_until, brk_dir = j + cfg.event_window, 1
                brk_events.append((j, 1, don_hi[j],
                                   (don_hi[j] - don_lo[j]) / a <= cfg.compression_atr))
            elif c[j] < don_lo[j]:
                brk_until, brk_dir = j + cfg.event_window, -1
                brk_events.append((j, -1, don_lo[j],
                                   (don_hi[j] - don_lo[j]) / a <= cfg.compression_atr))

        # --- reversal event (CHOCH against a stacked trend) ---
        if stack == 1 and sl_idx[j] >= 0 and sl_idx[j] != used_rev_lo and c[j] < sl_lvl[j]:
            rev_until, rev_dir = j + cfg.event_window, -1
            used_rev_lo = sl_idx[j]
        elif stack == -1 and sh_idx[j] >= 0 and sh_idx[j] != used_rev_hi and c[j] > sh_lvl[j]:
            rev_until, rev_dir = j + cfg.event_window, 1
            used_rev_hi = sh_idx[j]

        sn = slope_tot[j] / a if np.isfinite(slope_tot[j]) else np.nan
        band = 2 * resid_sd[j] / a if np.isfinite(resid_sd[j]) else np.nan

        if j <= brk_until:
            state[j], direction[j] = "breakout", brk_dir
        elif j <= rev_until:
            state[j], direction[j] = "reversal", rev_dir
        elif (stack != 0 and adx[j] >= cfg.adx_trend
              and seq[j] == ("up" if stack == 1 else "down")
              and np.isfinite(slope5[j]) and np.sign(slope5[j]) == stack):
            state[j], direction[j] = "strong_trend", stack
        elif (np.isfinite(sn) and abs(sn) >= cfg.slope_norm_trend
              and np.isfinite(band) and band >= cfg.band_wide_atr
              and np.isfinite(zcross[j]) and zcross[j] >= cfg.min_zero_cross):
            state[j], direction[j] = "wide_trend", int(np.sign(sn))
        elif (adx[j] < cfg.adx_range and np.isfinite(sn)
              and abs(sn) < cfg.slope_norm_range
              and np.isfinite(atr_pct[j]) and atr_pct[j] < cfg.atr_pctile_range):
            state[j], direction[j] = "range", 0

    # false-breakout audit: close back through the broken level within window
    for (j, d, lvl, _comp) in brk_events:
        end = min(n, j + cfg.event_window + 1)
        back = any((c[k] - lvl) * d < 0 for k in range(j + 1, end))
        false_breaks.append(back)

    tfm = {"5m": 5, "15m": 15, "1h": 60}[tf]
    out = pd.DataFrame({
        "time": df["timestamp"] + pd.Timedelta(minutes=tfm),
        "state": state, "direction": direction, "close": c, "atr": atr,
    })
    out.attrs["false_break_rate"] = (float(np.mean(false_breaks))
                                     if false_breaks else np.nan)
    out.attrs["n_breakouts"] = len(brk_events)
    return out
