"""Cost decomposition: what the signal must earn before execution can help.

Execution cost is not a haircut applied to a conclusion. It is a hurdle the raw
signal must clear before the idea is worth engineering at all, and it can be
computed *a priori* — before any code is written.

See ``docs/RESEARCH_METHOD.md`` §3.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "fee_in_r_wall",
    "decompose_gross_net",
    "toll_neutral_threshold",
]


def fee_in_r_wall(
    round_trip_bps: float,
    stop_frac: float,
) -> float:
    """Round-trip cost expressed in R, given a stop distance.

    ``toll_R = (round_trip_bps / 10_000) / stop_frac``

    The point of computing this first: costs are roughly constant per trade
    while the stop distance shrinks with the sampling interval, so the *same*
    fee is a very different fraction of R at each timeframe. For one mechanism
    the toll worked out to roughly 0.24R / 0.53R / 1.03R at hourly / 15-minute /
    5-minute bars — the signal survived scaling down, the economics did not.

    There is a timeframe below which a given fee tier cannot support a
    mechanism, and this one-liner finds it before you build anything.

    Examples
    --------
    >>> round(fee_in_r_wall(15.4, 0.0064), 3)   # ~1x ATR stop, hourly
    0.241
    >>> round(fee_in_r_wall(15.4, 0.0015), 3)   # ~1x ATR stop, 5-minute
    1.027
    """
    if stop_frac <= 0:
        raise ValueError("stop_frac must be positive")
    return float((round_trip_bps / 10_000.0) / stop_frac)


def decompose_gross_net(
    gross_r: Sequence[float],
    cost_r: Sequence[float] | float,
) -> dict:
    """Separate gross edge from the execution toll, and report the ratio.

    The verdict-forming number is ``toll_multiple = toll / gross``. Above 1.0 no
    fill improvement can rescue the entry — the boat costs more than the cargo.
    One book measured gross ``+0.0647R`` against a toll of ``0.1919R``, a
    multiple of 3.0, which closed it regardless of execution work.

    A companion diagnostic: if ``gross_mean <= 0`` the candidate has **no gross
    to erode**, and optimising execution is pure motion. A zero-cost backtest is
    the cheapest diagnostic in the toolkit; run it first, always.

    Returns
    -------
    dict with ``gross_mean``, ``toll_mean``, ``net_mean``, ``toll_multiple``,
    ``toll_share_of_gross`` (clipped to [0, inf)), ``n``, and
    ``verdict`` in {"no_gross", "toll_dominated", "viable"}.
    """
    g = np.asarray(gross_r, dtype=float)
    c = (
        np.full_like(g, float(cost_r))
        if np.isscalar(cost_r)
        else np.asarray(cost_r, dtype=float)
    )
    if g.shape != c.shape:
        raise ValueError("gross_r and cost_r must have the same length")

    mask = ~(np.isnan(g) | np.isnan(c))
    g, c = g[mask], c[mask]
    if g.size == 0:
        raise ValueError("no valid observations")

    gross_mean = float(g.mean())
    toll_mean = float(c.mean())
    net_mean = float((g - c).mean())

    if gross_mean <= 0:
        verdict, multiple = "no_gross", float("inf")
    else:
        multiple = toll_mean / gross_mean
        verdict = "toll_dominated" if multiple >= 1.0 else "viable"

    return {
        "gross_mean": gross_mean,
        "toll_mean": toll_mean,
        "net_mean": net_mean,
        "toll_multiple": float(multiple),
        "toll_share_of_gross": float(max(multiple, 0.0)),
        "n": int(g.size),
        "verdict": verdict,
    }


def toll_neutral_threshold(
    round_trip_cost: float,
    per_unit_move: float,
) -> float:
    """How far a trigger must sit before firing it is worth its own cost.

    For any mechanism that pays a round trip to act — a protective stop on an
    always-on book, a re-entry, a rebalance — the trigger distance must be at
    least ``round_trip_cost / per_unit_move`` units, or the action costs more
    than the adverse move it avoids.

    This is the algebra I skipped once, at real expense: a stop set at 3 units
    of dispersion needed roughly **14** to break even, so every firing was a
    guaranteed net loss. A stop is only a stop if the flat state persists;
    otherwise it is a subscription (``RESEARCH_METHOD.md`` §3.5).

    Parameters
    ----------
    round_trip_cost : cost of acting, in the same currency as ``per_unit_move``.
    per_unit_move : value of one unit of the trigger variable (e.g. one standard
        deviation of the spread being stopped on).

    Returns
    -------
    Minimum trigger distance, in units of the trigger variable.
    """
    if per_unit_move <= 0:
        raise ValueError("per_unit_move must be positive")
    return float(round_trip_cost / per_unit_move)
