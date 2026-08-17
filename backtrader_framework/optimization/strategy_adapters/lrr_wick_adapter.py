"""LRR wick-reversal adapter — the canonical single-bar wick+reclaim detector
under WFO, with frozen forward-validated overlay gates.

The detection logic is IMPORTED from the canonical module
(`Liquidity_Raid_Reversal.core.detector.detect_reversals`) — one source of
truth, parity-pinned by its own test suite. Every detector input is bar-t
information available at bar-t close: no session levels, no StructureBias,
no daily rows, no DVOL — none of the 2026-08-17 look-ahead classes applies.

Entry = signal-bar close (taker). SL = wick extreme -/+ sl_buffer x ATR.
TP = entry +/- rr x SL-distance (single hard target). Max hold is the WFO
config's max_trade_bars (48 per the study prereg), timeout at close — the
same conventions as the detector's own simulate_outcomes.

WFO params (all geometry/population, 32-combo full grid):
    rr, sl_buffer, vol_mult
Fixed detector params: lookback 8, wick_floor 0.6, prior_move 0.01.

Overlay gates (FROZEN, ablation-toggled, never in the grid):
    admissibility  own-asset regime5 x direction matrix for the LRR family
                   (quiet blocked; normal_trend WITH-HTF only, HTF = 4h
                   EMA50 vs EMA200 from CLOSED bins; normal_chop and
                   vol_expansion both directions). NA regime fails open.
    fomc_veto      entry (bar close) inside 10:00-13:59 ET on an FOMC
                   statement day is dropped.

`mirror=True` flips surviving signals at the same bracket (control only).
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_adapter import ParamSpec, Signal
from .syncretic_lr_adapter import _fomc_days, _ET

_REPO = Path(__file__).resolve().parents[3]

_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from Liquidity_Raid_Reversal.core.detector import detect_reversals
        _DETECTOR = detect_reversals
    return _DETECTOR


_QUIET = ("quiet_chop", "quiet_trend")


class LrrWickAdapter:
    """Canonical LRR detector -> geometry -> frozen admissibility gates."""

    def __init__(self, symbol: str = "BTC", *,
                 admissibility: bool = True,
                 fomc_veto: bool = True,
                 mirror: bool = False):
        self.symbol = symbol
        self.admissibility = bool(admissibility)
        self.fomc_veto = bool(fomc_veto)
        self.mirror = bool(mirror)
        self._fomc = _fomc_days() if self.fomc_veto else frozenset()
        # caches keyed by id(df): detector output per vol_mult, regime/HTF arrays
        self._sig_cache: Dict[tuple, pd.DataFrame] = {}
        self._ctx_cache: Dict[int, dict] = {}

    # ── adapter surface ─────────────────────────────────────────────
    @property
    def name(self) -> str:
        if self.admissibility and self.fomc_veto:
            tag = "adm"
        elif not (self.admissibility or self.fomc_veto):
            tag = "geo"
        else:
            tag = "geo" + ("+adm" if self.admissibility else "") \
                        + ("+fomc" if self.fomc_veto else "")
        if self.mirror:
            tag += "_mirror"
        return f"LrrWick_{tag}"

    @property
    def default_timeframes(self) -> List[str]:
        return ["15m"]

    def get_param_space(self) -> List[ParamSpec]:
        return [
            ParamSpec("rr",        1.5, 1.5, 3.0, 0.5),
            ParamSpec("sl_buffer", 0.5, 0.25, 1.0, 0.25),
            ParamSpec("vol_mult",  2.0, 1.5, 2.0, 0.5),
        ]

    def get_default_params(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.get_param_space()}

    def execute_signals(self, *args, **kwargs):
        return None

    def begin_window(self, *args, **kwargs):
        return None

    def end_window(self, *args, **kwargs):
        return None

    # ── context (regime5 + HTF), computed once per frame ────────────
    def _ctx(self, df: pd.DataFrame) -> dict:
        key = id(df)
        ctx = self._ctx_cache.get(key)
        if ctx is not None:
            return ctx
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from regime_classifier import (wilder_atr, wilder_adx,
                                       choppiness_index, classify_rule_based,
                                       RuleThresholds)
        lc = pd.DataFrame({
            "high": df["High"].to_numpy(dtype=float),
            "low": df["Low"].to_numpy(dtype=float),
            "close": df["Close"].to_numpy(dtype=float),
        })
        feats = pd.DataFrame({
            "atr14_pct": wilder_atr(lc) / lc["close"],
            "choppiness_14": choppiness_index(lc),
            "adx14": wilder_adx(lc),
        })
        regime = classify_rule_based(
            feats, RuleThresholds.for_asset(self.symbol)).to_numpy(dtype=object)

        # HTF direction from CLOSED 4h bins only. A bin labeled T covers
        # [T, T+4h) and completes at T+4h; a 15m signal bar opening at ts
        # decides at ts+15m, so usable bins satisfy T+4h <= ts+15m.
        idx_utc = pd.to_datetime(df.index, utc=True)
        h4_close = (df["Close"].set_axis(idx_utc)
                    .resample("4h", label="left", closed="left").last().dropna())
        e50 = h4_close.ewm(span=50, adjust=False).mean()
        e200 = h4_close.ewm(span=200, adjust=False).mean()
        h4_dir = np.sign((e50 - e200).to_numpy())
        h4_done_ns = (h4_close.index + pd.Timedelta(hours=4)).asi8
        ctx = {"regime": regime, "h4_dir": h4_dir, "h4_done_ns": h4_done_ns,
               "idx_ns": idx_utc.asi8}
        if len(self._ctx_cache) > 4:
            self._ctx_cache.clear()
        self._ctx_cache[key] = ctx
        return ctx

    def _signals_frame(self, df: pd.DataFrame, vol_mult: float) -> pd.DataFrame:
        key = (id(df), float(vol_mult))
        sig = self._sig_cache.get(key)
        if sig is None:
            sig = _detector()(df, vol_mult=float(vol_mult))
            # positions of signal bars in the frame
            pos = df.index.get_indexer(sig.index)
            sig = sig.assign(_pos=pos)
            if len(self._sig_cache) > 8:
                self._sig_cache.clear()
            self._sig_cache[key] = sig
        return sig

    # ── signal pipeline ─────────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any],
                         scan_start_idx: int, scan_end_idx: int) -> List[Signal]:
        rr = float(params.get("rr", 1.5))
        sl_buffer = float(params.get("sl_buffer", 0.5))
        vol_mult = float(params.get("vol_mult", 2.0))

        sig = self._signals_frame(df, vol_mult)
        if sig.empty:
            return []
        in_scan = (sig["_pos"] >= scan_start_idx) & (sig["_pos"] < scan_end_idx)
        sig = sig[in_scan]
        if sig.empty:
            return []

        ctx = self._ctx(df) if self.admissibility else None

        out: List[Signal] = []
        for ts, row in sig.iterrows():
            i = int(row["_pos"])
            direction = row["direction"]
            entry = float(row["entry"])
            atr = float(row["atr"])

            tss = pd.Timestamp(ts)
            if tss.tzinfo is None:
                tss = tss.tz_localize("UTC")

            if self.fomc_veto:
                close_et = (tss + timedelta(minutes=15)).tz_convert(_ET)
                if close_et.date() in self._fomc and 10 <= close_et.hour < 14:
                    continue

            if self.admissibility:
                reg = ctx["regime"][i]
                if isinstance(reg, str):
                    if reg in _QUIET:
                        continue
                    if reg == "normal_trend":
                        # WITH-HTF only: need a KNOWN aligned 4h direction.
                        p = np.searchsorted(ctx["h4_done_ns"],
                                            ctx["idx_ns"][i] + 900_000_000_000,
                                            side="right") - 1
                        d = ctx["h4_dir"][p] if p >= 0 else 0.0
                        want = 1.0 if direction == "LONG" else -1.0
                        if d != want:
                            continue
                # NA regime: fail open (pre-warmup only)

            if direction == "LONG":
                stop = float(row["bar_low"]) - sl_buffer * atr
                risk = entry - stop
                tp = entry + rr * risk
            else:
                stop = float(row["bar_high"]) + sl_buffer * atr
                risk = stop - entry
                tp = entry - rr * risk
            if risk <= 0:
                continue

            s = Signal(
                idx=i, time=df.index[i], direction=direction,
                entry_price=entry, stop_loss=stop,
                take_profit_1=tp, take_profit_2=tp,
                risk=risk, confidence=float(row.get("wick_ratio", 0.5)),
                bias="COUNTER", atr=atr,
                metadata={
                    "volume_ratio": float(row["volume_ratio"]),
                    "wick_ratio": float(row["wick_ratio"]),
                    "prior_move_pct": float(row["prior_move_pct"]),
                    "level": float(row["lookback_low"] if direction == "LONG"
                                   else row["lookback_high"]),
                    "regime": (ctx["regime"][i] if ctx is not None else None),
                },
            )
            out.append(self._mirrored(s) if self.mirror else s)
        return out

    @staticmethod
    def _mirrored(sig: Signal) -> Signal:
        entry = sig.entry_price
        risk = sig.risk
        if sig.direction == "LONG":
            direction, stop = "SHORT", entry + risk
            tp = entry - (sig.take_profit_1 - entry)
        else:
            direction, stop = "LONG", entry - risk
            tp = entry + (entry - sig.take_profit_1)
        return Signal(
            idx=sig.idx, time=sig.time, direction=direction,
            entry_price=entry, stop_loss=stop,
            take_profit_1=tp, take_profit_2=tp,
            risk=risk, confidence=sig.confidence, bias=sig.bias,
            atr=sig.atr, metadata=dict(sig.metadata or {}, mirrored=True),
        )
