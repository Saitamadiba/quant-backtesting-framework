"""Live-faithful filter stack for the Vol Edge WFO adapter.

Same pattern as the LR / MM / FVG faithful modules — see memory note
an internal pipeline-consistency study.

Vol Edge's situation differs from the others. The bare adapter generates
signals for **two distinct edges**:

* **Edge B** — IV-Spike Spot Reversal (long-only directional play).
* **Edge C** — Short-Vol Synthetic (DVOL low + VRP positive).

But the **live fleet only trades a single edge** — the short-vol straddle
("Edge S" in ``Vol_Edge/Straddle_V1/{btc,eth}_straddle.py``), which is
structurally Edge C with the live thresholds:

* ``EDGE_S_DVOL_MAX = 50.0`` (live)  vs adapter default 45.0
* ``EDGE_S_MIN_VRP  = -5.0`` (live)  vs adapter historic hardcoded ``> 0.0``

So a faithful Vol Edge WFO needs two things:

1. **Drop Edge B signals** — the live straddle bots don't trade them, so
   evaluating the strategy with Edge B mixed in overstates the universe.
2. **Use the live Edge C/S thresholds** — the VRP floor parameter
   (``edge_c_min_vrp``) was added to the adapter alongside this module
   (default 0.0 for backward compat, set to -5.0 here to match live).

Helper:

>>> FaithfulVolEdgeAdapter()                  # Edge-C-only, live thresholds
>>> FaithfulVolEdgeAdapter(force_live_thresholds=False)
                                              # Edge-C-only, WFO-tunable thresholds

Everything else (options pricer, straddle exit logic, position sizing) is
the live bot's runtime concern, not an entry-side filter — those don't
belong in the WFO signal-generation path.
"""
from __future__ import annotations

from typing import Any, Optional


# Live straddle thresholds — see Vol_Edge/Straddle_V1/{btc,eth}_straddle.py.
LIVE_EDGE_S_DVOL_MAX = 50.0
LIVE_EDGE_S_MIN_VRP = -5.0


class FaithfulVolEdgeAdapter:
    """Drop-in replacement for :class:`VolEdgeAdapter` that returns only the
    Edge C / Edge S signals the live straddle bot would generate.

    By default ``force_live_thresholds=True`` overrides ``edge_c_iv_low_threshold``
    and ``edge_c_min_vrp`` in the params dict to the live values, so the WFO
    evaluates the live configuration directly. Pass ``force_live_thresholds=False``
    to let the WFO optimise those thresholds freely (research / sensitivity
    studies).

    Composition-not-subclassing, same as the LR/MM/FVG siblings.
    """

    def __init__(self, base: Optional[Any] = None,
                 *, edge: str = "C",
                 force_live_thresholds: bool = True,
                 dvol_max: float = LIVE_EDGE_S_DVOL_MAX,
                 min_vrp: float = LIVE_EDGE_S_MIN_VRP):
        if base is None:
            from .vol_edge_adapter import VolEdgeAdapter as _Base
            base = _Base()
        self.base = base
        self.edge = edge  # filter to this metadata['edge'] tag
        self.force_live_thresholds = bool(force_live_thresholds)
        self.dvol_max = float(dvol_max)
        self.min_vrp = float(min_vrp)

    # ── Adapter surface delegation ──────────────────────────────────
    @property
    def name(self) -> str:
        return f"{self.base.name}_faithful"

    @property
    def default_timeframes(self):
        return self.base.default_timeframes

    def get_param_space(self):
        return self.base.get_param_space()

    def generate_signals(self, df, params, scan_start_idx, scan_end_idx):
        if self.force_live_thresholds:
            # Don't mutate the caller's dict — WFO engines may reuse it.
            params = dict(params)
            params["edge_c_iv_low_threshold"] = self.dvol_max
            params["edge_c_min_vrp"] = self.min_vrp
        signals = self.base.generate_signals(df, params, scan_start_idx, scan_end_idx)
        if self.edge:
            signals = [s for s in signals
                       if (getattr(s, "metadata", None) or {}).get("edge") == self.edge]
        return signals
