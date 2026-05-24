"""Live-faithful filter stack for the Momentum Mastery WFO adapter.

Same architectural pattern as :mod:`lr_faithful_filters`, applied to MM —
see memory note an internal pipeline-consistency study for the why. The MM bare
adapter is missing several live gates (premature soft-gate, vol-cap,
new5 regime, DVOL/IV gate, partial-TP + BE-after-move). This module
ports the **entry-side** ones (regime + DVOL) into the WFO so the
adapter sees the same signal universe live does.

MM-specific gates NOT yet in this module
----------------------------------------
* **Premature soft-gate** lives in ``the internal position-manager module``
  and runs at trade-management time (it tracks a sliding rate of premature
  classifications); a WFO lift requires simulator-level changes, not just
  signal filtering. Deferred.
* **Vol-cap sweep guard** — same story (position-manager side).
* **Partial TP + move-SL-to-BE** — exit-side, simulator change required.
  See the internal partial-exit replay study for the live-replay impact
  (positive live-replay impact on both assets) — meaningful enough
  that it should be lifted properly in a follow-up.

Regime classifier
-----------------
MM live uses the proprietary :mod:`regime_classifier` module (5-label
"new5": ``quiet_chop`` / ``quiet_trend`` / ``normal_chop`` / ``normal_trend``
/ ``vol_expansion``). When the proprietary module is importable, we use
it directly. When it isn't (public clones of the framework), the regime
filter degrades to a no-op rather than guessing — better silent-off than
silently wrong, since the new5 labels MM blocks on (``quiet_trend``,
``quiet_chop``) don't exist in the 4-label fallback.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

# Reused helpers from the LR module — same DVOL data, same band thresholds
# as live MM config_base (IV_MED_THRESHOLD=45 / IV_HIGH_THRESHOLD=65).
from .lr_faithful_filters import attach_dvol, dvol_band, MIN_MTF_SCORE  # noqa: F401


# ── Proprietary new5 regime classifier (optional) ────────────────────

try:  # pragma: no cover - depends on user env
    import sys as _sys
    from pathlib import Path as _Path
    # The classifier lives at repo root; ensure it's importable.
    _ROOT = _Path(__file__).resolve().parents[3]
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))
    from regime_classifier import (  # type: ignore
        compute_features as _rc_compute_features,
        classify_rule_based as _rc_classify_rule_based,
        RuleThresholds as _RuleThresholds,
    )
    NEW5_AVAILABLE = True
except Exception:
    _rc_compute_features = None
    _rc_classify_rule_based = None
    _RuleThresholds = None
    NEW5_AVAILABLE = False


def compute_regimes_new5(df: pd.DataFrame, symbol: str = "BTC") -> Optional[np.ndarray]:
    """Per-bar new5 regime label (proprietary classifier), or ``None`` if unavailable.

    Returns an object array aligned with ``df.index``. Possible labels:
    ``quiet_chop``, ``quiet_trend``, ``normal_chop``, ``normal_trend``,
    ``vol_expansion`` (the live MM regime gate blocks the first two on BTC).
    """
    if not NEW5_AVAILABLE:
        return None
    # The classifier expects lowercase column names (live live convention).
    lc = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    needed = ["open", "high", "low", "close", "volume"]
    feats = _rc_compute_features(lc[needed])
    thr = _RuleThresholds.for_asset(symbol)
    series = _rc_classify_rule_based(feats, thr)
    return series.to_numpy(dtype=object)


# ── Wrapper adapter: MM faithful drop-in ─────────────────────────────

class FaithfulMMAdapter:
    """Drop-in replacement for :class:`MMAdapter` that applies the live entry
    gate stack (new5 regime block + IV-MED block) before returning signals.

    Defaults match live MM BTC (``Momentum_Mastery/BTC/btc_momentum_mastery_v2.py``):
    ``REGIME_GATE_BLOCKED_REGIMES = ["quiet_trend", "quiet_chop"]``,
    ``IV_REGIME_BLOCKED_REGIMES = []`` (BTC doesn't block any IV band live;
    pass ``block_bands=("MED",)`` to match ETH live).

    Composition-not-subclassing — same reasoning as
    :class:`FaithfulLiquidityRaidAdapter`: the WFO engine only reads the
    public adapter surface, and inheritance would couple us to mutable
    state on the parent.
    """

    def __init__(self, symbol: str = "BTC",
                 base: Optional[Any] = None,
                 *, blocked_regimes: Iterable[str] = ("quiet_trend", "quiet_chop"),
                 block_bands: Iterable[str] = (),
                 partial_exit_pct: float = 0.5,
                 ):
        """
        partial_exit_pct : fraction of position closed at TP1 (default 0.5
            mirrors live ``PARTIAL_EXIT_PCT`` on the BTC config). 0.0 disables
            partial-TP entirely (signals exit the simulator's existing
            trail-after-TP1 path). The simulator reads this from each signal's
            ``metadata['partial_exit_pct']`` (see TradeSimulator.simulate).
            Live BTC PARTIAL_EXIT_RR_FRAC=0.5 vs adapter's tp1 at 1.0×rr is a
            known approximation — see the internal partial-exit study.
        """
        if base is None:
            from .mm_adapter import MomentumMasteryAdapter as _Base
            base = _Base()
        self.base = base
        self.symbol = symbol
        self.blocked_regimes = tuple(blocked_regimes)
        self.block_bands = tuple(block_bands)
        self.partial_exit_pct = float(partial_exit_pct)
        # Caches keyed by id(df) — same trick as the LR faithful adapter.
        self._cache: dict = {}

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
        # DVOL — same notna() probe as the LR adapter to dodge the
        # IndicatorEngine placeholder-column bug.
        if "DVOL" in df.columns and df["DVOL"].notna().any():
            df_dv = df
        else:
            df_dv = attach_dvol(df.drop(columns=["DVOL"], errors="ignore"),
                                self.symbol)
        # Normalise index dtype/tz — see lr_faithful_filters._ctx_for for why.
        if str(df_dv.index.dtype) != "datetime64[ns, UTC]":
            df_dv = df_dv.copy()
            df_dv.index = pd.to_datetime(df_dv.index, utc=True)
        ctx = {
            "df": df_dv,
            "regime_arr": compute_regimes_new5(df_dv, self.symbol),
        }
        self._cache[key] = ctx
        return ctx

    def generate_signals(self, df: pd.DataFrame, params: dict,
                         scan_start_idx: int, scan_end_idx: int):
        signals = self.base.generate_signals(df, params, scan_start_idx, scan_end_idx)
        if not signals:
            return signals
        ctx = self._ctx_for(df)
        df_dv = ctx["df"]
        regime_arr = ctx["regime_arr"]
        blocked_regs = set(self.blocked_regimes)
        blocked_bands = set(self.block_bands)
        survivors = []
        for sig in signals:
            # Regime gate (new5).
            if regime_arr is not None and blocked_regs:
                idx = getattr(sig, "idx", -1)
                if 0 <= idx < len(regime_arr):
                    if str(regime_arr[idx]) in blocked_regs:
                        continue
            # IV band gate (off by default for MM BTC; ETH live blocks MED).
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
                # Stash for telemetry.
                md = getattr(sig, "metadata", None) or {}
                sig.metadata = dict(md, dvol=dvol_at, band=dvol_band(dvol_at))
            # Wire partial-TP into the simulator via metadata. Read by
            # TradeSimulator.simulate() to do frac × R_at_TP1 + (1-frac) × R_at_remaining.
            if self.partial_exit_pct > 0:
                md = getattr(sig, "metadata", None) or {}
                sig.metadata = dict(md, partial_exit_pct=self.partial_exit_pct)
            survivors.append(sig)
        return survivors
