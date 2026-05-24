"""Live-faithful filter stack for the FVG WFO adapter.

Same pattern as :mod:`lr_faithful_filters` and :mod:`mm_faithful_filters`,
applied to FVG — see memory note an internal pipeline-consistency study.

FVG's live filter surface is *smaller* than LR/MM. The only live entry-
side gate is the IV-MED block on **FVG ETH only** (per
``the internal FVG live config``: ``IV_REGIME_FILTER_ENABLED=True``,
``IV_REGIME_BLOCKED_REGIMES=["MED"]``). FVG BTC and NQ run with the
default unfiltered IV stack live. The new5 regime gate is *not* enabled
on any FVG asset live (an internal regime-conditioning study studied it
but the gate was kept off; the live decision was not to deploy after internal review).

Helper to pick the right config per asset:

>>> FaithfulFVGAdapter.for_asset('ETH')   # IV-MED blocked
>>> FaithfulFVGAdapter.for_asset('BTC')   # no-op vs bare adapter

Not lifted (state-/exit-side or non-active live):
* :class:`FVGZoneTracker` (``core/zone_tracker.py``) — stateful zone
  cooldown/expiry; lives across bars and trades. Bake into the simulator,
  not a filter.
* :class:`AdaptiveRiskManager` (``core/adaptive_risk_manager.py``) —
  sizing, not entry filtering.
* :class:`PrematureEntryAnalyzer` — entry-timing analytics; complex stateful
  logic. Defer.
* Funding-agree gate — **log-only live** per the internal shadow-gate logging policy
  (WFO/live divergence flagged in internal review; kept log-only). Intentionally NOT lifted —
  the WFO should reflect what live actually does, not what shadow tests.
* TAAPI external indicators — runtime API dependency; out of scope.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd

# Reuse from the LR module — same DVOL parquet, same band thresholds
# (FVG live config_base inherits IV_HIGH_THRESHOLD=65 / IV_MED_THRESHOLD=45).
from .lr_faithful_filters import attach_dvol, dvol_band  # noqa: F401
from .mm_faithful_filters import compute_regimes_new5, NEW5_AVAILABLE  # noqa: F401


# ── FVG-specific live defaults per asset ─────────────────────────────

_LIVE_PRESETS: dict[str, dict] = {
    "BTC": {"blocked_regimes": (), "block_bands": ()},                # BTC: no live gates
    "ETH": {"blocked_regimes": (), "block_bands": ("MED",)},          # ETH: IV-MED block (live)
    "NQ":  {"blocked_regimes": (), "block_bands": ()},                # NQ:  no live gates
}


# ── Wrapper adapter: FVG faithful drop-in ────────────────────────────

class FaithfulFVGAdapter:
    """Drop-in replacement for :class:`FVGAdapter` that applies the live FVG
    entry filter stack before returning signals.

    Use :meth:`for_asset` to get an instance configured to match the live
    per-asset settings; ``__init__`` exposes the raw knobs for research.

    Composition-not-subclassing, same caching trick as the LR/MM siblings.
    """

    def __init__(self, symbol: str = "ETH",
                 base: Optional[Any] = None,
                 *, blocked_regimes: Iterable[str] = (),
                 block_bands: Iterable[str] = ()):
        if base is None:
            from .fvg_adapter import FVGAdapter as _Base
            base = _Base()
        self.base = base
        self.symbol = symbol
        self.blocked_regimes = tuple(blocked_regimes)
        self.block_bands = tuple(block_bands)
        self._cache: dict = {}

    @classmethod
    def for_asset(cls, symbol: str, **overrides) -> "FaithfulFVGAdapter":
        """Construct an adapter pre-configured for ``symbol`` (BTC/ETH/NQ).

        Mirrors the live per-asset config. ``overrides`` lets research
        callers tweak individual knobs (e.g., ``blocked_regimes=('quiet_chop',)``
        to study a what-if regime gate).
        """
        preset = dict(_LIVE_PRESETS.get(symbol, {}))
        preset.update(overrides)
        return cls(symbol=symbol, **preset)

    # ── Adapter surface delegation ──────────────────────────────────
    @property
    def name(self) -> str:
        return f"{self.base.name}_faithful"

    @property
    def default_timeframes(self):
        return self.base.default_timeframes

    def get_param_space(self):
        return self.base.get_param_space()

    def __getattr__(self, name):
        """Forward any unforwarded attribute to the wrapped adapter — the
        WFO engine reaches for ``get_default_params``, ``begin_window`` /
        ``end_window``, ``execute_signals`` etc. Guards recursion before
        ``self.base`` is set during ``__init__``."""
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    # ── Cache + filter ───────────────────────────────────────────────
    def _ctx_for(self, df: pd.DataFrame):
        key = id(df)
        ctx = self._cache.get(key)
        if ctx is not None:
            return ctx
        # Same notna() probe as LR/MM — IndicatorEngine adds an empty
        # DVOL placeholder; only use df's DVOL if it has actual data.
        if "DVOL" in df.columns and df["DVOL"].notna().any():
            df_dv = df
        else:
            df_dv = attach_dvol(df.drop(columns=["DVOL"], errors="ignore"),
                                self.symbol)
        # Normalise index dtype/tz — see lr_faithful_filters._ctx_for for why.
        if str(df_dv.index.dtype) != "datetime64[ns, UTC]":
            df_dv = df_dv.copy()
            df_dv.index = pd.to_datetime(df_dv.index, utc=True)
        # Skip the proprietary regime compute if no regimes are blocked —
        # live FVG doesn't gate by regime, so this is a no-op for all
        # default-configured FVG runs.
        regime_arr = (compute_regimes_new5(df_dv, self.symbol)
                      if self.blocked_regimes else None)
        ctx = {"df": df_dv, "regime_arr": regime_arr}
        self._cache[key] = ctx
        return ctx

    def generate_signals(self, df: pd.DataFrame, params: dict,
                         scan_start_idx: int, scan_end_idx: int):
        signals = self.base.generate_signals(df, params, scan_start_idx, scan_end_idx)
        if not signals:
            return signals
        # Fast path: nothing to filter — match the bare adapter exactly.
        if not (self.blocked_regimes or self.block_bands):
            return signals
        ctx = self._ctx_for(df)
        df_dv = ctx["df"]
        regime_arr = ctx["regime_arr"]
        blocked_regs = set(self.blocked_regimes)
        blocked_bands = set(self.block_bands)
        survivors = []
        for sig in signals:
            # Regime gate (off by default for FVG; available for research).
            if blocked_regs and regime_arr is not None:
                idx = getattr(sig, "idx", -1)
                if 0 <= idx < len(regime_arr) and str(regime_arr[idx]) in blocked_regs:
                    continue
            # IV band gate (ETH live blocks MED).
            if blocked_bands:
                ets = pd.Timestamp(getattr(sig, "time"))
                if ets.tzinfo is None:
                    ets = ets.tz_localize("UTC")
                try:
                    dvol_at = float(df_dv.loc[df_dv.index <= ets, "DVOL"].iloc[-1])
                except (IndexError, KeyError):
                    dvol_at = float("nan")
                if dvol_band(dvol_at) in blocked_bands:
                    continue
                md = getattr(sig, "metadata", None) or {}
                sig.metadata = dict(md, dvol=dvol_at, band=dvol_band(dvol_at))
            survivors.append(sig)
        return survivors
