"""Syncretic LR adapter — deep-sweep, wide-bracket LR with frozen overlay gates.

Composition of independently-validated pieces (see the study's PREREG for the
evidence chain); the WFO tunes GEOMETRY ONLY (3 params, 36-combo full grid):

    rr                  single hard target (min_rr == max_rr == rr)
    atr_sl_multiplier   stop distance in ATR from entry (uncapped)
    pen_min             sweep-penetration floor in ATR (deep-sweep filter,
                        applied via the base adapter's min_sweep_atr_override)

Signal universe = FaithfulLiquidityRaidAdapter (live filter stack: MTF >= 50,
IV-MED band blocked, counter-trend blocked, ML off) — running the bare
universe is directionally wrong, per the two-pipelines pitfall documented in
lr_faithful_filters.py.

Overlay gates (FROZEN constants, never in the WFO grid; each drops signals,
never resizes them):

    regime_block  drop signals whose concurrent BTC 15m regime5 label is
                  quiet_chop/quiet_trend (BTC-frame gate; fail-open on NA)
    drift_drop    drop signals in the bottom decile of the with-drift
                  composite (frozen Q10 from Liquidity_Raid.core.drift_context;
                  fail-open when the composite cannot be computed)
    fomc_veto     drop signals whose entry (bar close) lands 10:00-13:59 ET on
                  an FOMC statement day (frozen calendar; the calendar starts
                  2021 so earlier history is a structural no-op)

`mirror=True` flips every surviving signal's direction at the SAME bracket
(entry unchanged, stop reflected, target reflected) — the first-passage /
bracket-asymmetry control, never a tradeable variant.

All gates read only data at or before the signal bar's close. Entry is the
confirmation-bar close (the same convention the underlying evidence was
measured at); no touch=fill assumption anywhere.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .base_adapter import ParamSpec, Signal
from .lr_faithful_filters import FaithfulLiquidityRaidAdapter

_ET = ZoneInfo("America/New_York")
_REPO = Path(__file__).resolve().parents[3]

# ── FOMC statement days ──────────────────────────────────────────────
# Prefer the maintained frozen calendar; fall back to a vendored copy of the
# same public Fed dates so this module stays importable standalone.
_FOMC_FALLBACK = frozenset(date(y, m, d) for y, m, d in [
    (2021, 1, 27), (2021, 3, 17), (2021, 4, 28), (2021, 6, 16),
    (2021, 7, 28), (2021, 9, 22), (2021, 11, 3), (2021, 12, 15),
    (2022, 1, 26), (2022, 3, 16), (2022, 5, 4), (2022, 6, 15),
    (2022, 7, 27), (2022, 9, 21), (2022, 11, 2), (2022, 12, 14),
    (2023, 2, 1), (2023, 3, 22), (2023, 5, 3), (2023, 6, 14),
    (2023, 7, 26), (2023, 9, 20), (2023, 11, 1), (2023, 12, 13),
    (2024, 1, 31), (2024, 3, 20), (2024, 5, 1), (2024, 6, 12),
    (2024, 7, 31), (2024, 9, 18), (2024, 11, 7), (2024, 12, 18),
    (2025, 1, 29), (2025, 3, 19), (2025, 5, 7), (2025, 6, 18),
    (2025, 7, 30), (2025, 9, 17), (2025, 10, 29), (2025, 12, 10),
    (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17),
    (2026, 7, 29), (2026, 9, 16), (2026, 10, 28), (2026, 12, 9),
])


def _fomc_days() -> frozenset:
    try:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from fomc_shadow.calendar import STATEMENT_DAYS
        return frozenset(STATEMENT_DAYS)
    except Exception:
        return _FOMC_FALLBACK


# ── BTC regime5 arrays (module-level, one load per process) ──────────
_BTC_REGIME: Optional[tuple] = None  # (open_ts_ns ascending, labels object[])


def _btc_regime_arrays() -> tuple:
    global _BTC_REGIME
    if _BTC_REGIME is None:
        import duckdb
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from regime_classifier import (wilder_atr, wilder_adx,
                                       choppiness_index, classify_rule_based)
        con = duckdb.connect(str(_REPO / "duckdb_data" / "trading_data.duckdb"),
                             read_only=True)
        try:
            btc = con.execute(
                "SELECT timestamp, open, high, low, close, volume "
                "FROM ohlcv_data WHERE symbol='BTC' AND timeframe='15m' "
                "ORDER BY timestamp"
            ).fetchdf()
        finally:
            con.close()
        feats = pd.DataFrame({
            "atr14_pct": wilder_atr(btc) / btc["close"],
            "choppiness_14": choppiness_index(btc),
            "adx14": wilder_adx(btc),
        })
        labels = classify_rule_based(feats)  # default thresholds = BTC row
        ts = pd.to_datetime(btc["timestamp"], utc=True).astype("int64").to_numpy()
        _BTC_REGIME = (ts, labels.to_numpy(dtype=object))
    return _BTC_REGIME


# ── drift composite (frozen constants live in the private core module) ─
_DRIFT: Optional[tuple] = None  # (drift_context fn, DRIFT_Q10)


def _drift_fn() -> tuple:
    global _DRIFT
    if _DRIFT is None:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        try:
            from Liquidity_Raid.core.drift_context import (drift_context,
                                                           DRIFT_Q10)
        except ImportError as e:  # fail CLOSED: V1 without drift is not V1
            raise RuntimeError(
                "SyncreticLRAdapter(drift_drop=True) needs "
                "Liquidity_Raid.core.drift_context (frozen composite). "
                "Run with drift_drop=False only as the declared ablation."
            ) from e
        _DRIFT = (drift_context, DRIFT_Q10)
    return _DRIFT


_QUIET = ("quiet_chop", "quiet_trend")
_DRIFT_WINDOW = 600  # trailing 15m bars handed to the composite (>= its 250 min)


class SyncreticLRAdapter:
    """Delegation wrapper: faithful LR universe -> geometry -> frozen gates."""

    def __init__(self, symbol: str = "BTC", *,
                 regime_block: bool = True,
                 drift_drop: bool = True,
                 fomc_veto: bool = True,
                 mirror: bool = False):
        self.faithful = FaithfulLiquidityRaidAdapter(symbol=symbol)
        self.symbol = symbol
        self.regime_block = bool(regime_block)
        self.drift_drop = bool(drift_drop)
        self.fomc_veto = bool(fomc_veto)
        self.mirror = bool(mirror)
        self._fomc = _fomc_days() if self.fomc_veto else frozenset()
        self._lc_cache: Dict[int, pd.DataFrame] = {}
        self._drift_cache: Dict[tuple, Optional[float]] = {}

    # ── adapter surface ─────────────────────────────────────────────
    @property
    def name(self) -> str:
        if self.regime_block and self.drift_drop and self.fomc_veto:
            tag = "full"
        elif not (self.regime_block or self.drift_drop or self.fomc_veto):
            tag = "geo"
        else:
            tag = "geo" + ("+reg" if self.regime_block else "") \
                        + ("+drift" if self.drift_drop else "") \
                        + ("+fomc" if self.fomc_veto else "")
        if self.mirror:
            tag += "_mirror"
        return f"SyncreticLR_{tag}"

    @property
    def default_timeframes(self) -> List[str]:
        return ["15m"]

    def get_param_space(self) -> List[ParamSpec]:
        return [
            ParamSpec("rr",                3.0, 2.0, 4.0, 1.0),
            ParamSpec("atr_sl_multiplier", 2.5, 2.0, 3.0, 0.5),
            ParamSpec("pen_min",           1.0, 0.6, 1.2, 0.2),
        ]

    def get_default_params(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.get_param_space()}

    def execute_signals(self, *args, **kwargs):
        return None  # always use generate_signals + TradeSimulator

    def begin_window(self, *args, **kwargs):
        return None

    def end_window(self, *args, **kwargs):
        return None

    # ── signal pipeline ─────────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any],
                         scan_start_idx: int, scan_end_idx: int) -> List[Signal]:
        rr = float(params.get("rr", 3.0))
        p = dict(params)
        p["min_rr"] = rr
        p["max_rr"] = rr
        p["min_sweep_atr_override"] = float(params.get("pen_min", 1.0))
        p.setdefault("atr_sl_multiplier", 2.5)

        sigs = self.faithful.generate_signals(df, p, scan_start_idx,
                                              scan_end_idx)
        if not sigs:
            return sigs

        if self.regime_block:
            btc_ts, btc_lbl = _btc_regime_arrays()

        out: List[Signal] = []
        for sig in sigs:
            ts = pd.Timestamp(sig.time)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            # FOMC veto — entry moment is the 15m bar's CLOSE.
            if self.fomc_veto:
                close_et = (ts + timedelta(minutes=15)).tz_convert(_ET)
                if close_et.date() in self._fomc and 10 <= close_et.hour < 14:
                    continue

            # BTC regime5 quiet-block. The BTC bar whose open <= this bar's
            # open closes no later than this bar's close -> causal.
            if self.regime_block:
                pos = np.searchsorted(btc_ts, ts.value, side="right") - 1
                if pos >= 0:
                    lbl = btc_lbl[pos]
                    if isinstance(lbl, str) and lbl in _QUIET:
                        continue
                # pos < 0 or NA label: fail-open (pre-history)

            # Drift bottom-decile drop.
            if self.drift_drop:
                score = self._drift_score(df, sig, ts)
                if score is not None and score <= _drift_fn()[1]:
                    continue
                # None: fail-open (composite unavailable)

            out.append(self._mirrored(sig) if self.mirror else sig)
        return out

    # ── helpers ─────────────────────────────────────────────────────
    def _lc_for(self, df: pd.DataFrame) -> pd.DataFrame:
        key = id(df)
        lc = self._lc_cache.get(key)
        if lc is None:
            idx = pd.to_datetime(df.index, utc=True)
            lc = pd.DataFrame({
                "timestamp": idx,
                "open": df["Open"].to_numpy(dtype=float),
                "high": df["High"].to_numpy(dtype=float),
                "low": df["Low"].to_numpy(dtype=float),
                "close": df["Close"].to_numpy(dtype=float),
                "volume": df["Volume"].to_numpy(dtype=float),
            })
            if len(self._lc_cache) > 4:
                self._lc_cache.clear()
            self._lc_cache[key] = lc
        return lc

    def _drift_score(self, df: pd.DataFrame, sig: Signal,
                     ts: pd.Timestamp) -> Optional[float]:
        key = (ts.value, sig.direction)
        if key in self._drift_cache:
            return self._drift_cache[key]
        drift_context, _ = _drift_fn()
        lc = self._lc_for(df)
        i = int(sig.idx)
        window = lc.iloc[max(0, i - _DRIFT_WINDOW + 1): i + 1]
        res = drift_context(window, sig.direction, float(sig.atr or 0.0))
        score = None if res is None else float(res["drift_score"])
        self._drift_cache[key] = score
        return score

    @staticmethod
    def _mirrored(sig: Signal) -> Signal:
        entry = sig.entry_price
        risk = sig.risk
        if sig.direction == "LONG":
            direction, stop = "SHORT", entry + risk
            tp1 = entry - (sig.take_profit_1 - entry)
            tp2 = entry - (sig.take_profit_2 - entry)
        else:
            direction, stop = "LONG", entry - risk
            tp1 = entry + (entry - sig.take_profit_1)
            tp2 = entry + (entry - sig.take_profit_2)
        return Signal(
            idx=sig.idx, time=sig.time, direction=direction,
            entry_price=entry, stop_loss=stop,
            take_profit_1=tp1, take_profit_2=tp2,
            risk=risk, confidence=sig.confidence, bias=sig.bias,
            atr=sig.atr, metadata=dict(sig.metadata or {}, mirrored=True),
        )
