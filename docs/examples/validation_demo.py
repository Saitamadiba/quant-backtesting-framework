#!/usr/bin/env python3
"""End-to-end demonstration of the validation harness on synthetic data.

Run it:

    python docs/examples/validation_demo.py

Three candidates go through the identical battery:

  A. **No edge, and it looks like none.** Random entries on a driftless walk.
     Dies at the first gate: there is no gross to erode.
  B. **No edge, but it looks profitable.** Random entries on a *drifting* walk.
     Gross is strongly positive, cost-viable, clustered t > 3, stable across
     both halves — and it is still not an edge, because random entries on the
     same path earn just as much. Only the random-bar control catches it.
     This is the most instructive case in the file.
  C. **A genuine conditional edge.** The path is driftless *unconditionally*,
     but drifts after an observable state that the signal enters on. Random
     entries get the unconditional zero; the signal gets the conditional drift.

A harness that passes everything is decoration; one that kills everything is
useless. The value is the separation — and B is where most backtests die.
Everything below is reproducible from this file with no private data.

Reference: docs/RESEARCH_METHOD.md
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtrader_framework.validation import (  # noqa: E402
    clustered_tstat,
    decompose_gross_net,
    effective_sample_size,
    excess_over_control,
    fee_in_r_wall,
    half_split_stability,
    mirrored_bracket_control,
    random_bar_control,
    signed_permutation_maxt,
    walk_bracket,
)

SEED = 20260817
STOP_FRAC = 0.010
TARGET_FRAC = 0.020
N_TRADES = 240
ROUND_TRIP_BPS = 15.4          # a representative taker round trip


def synth_path(n=12_000, sigma=0.004, drift=0.0, seed=SEED):
    """Bars from a (optionally drifting) random walk."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, sigma, n)
    close = 100 * np.exp(np.cumsum(steps))
    wick = np.abs(rng.normal(0, sigma / 2, n)) * close
    return close + wick, close - wick, close


def state_dependent_path(
    n=12_000, sigma=0.004, drift_when_flagged=0.0010, lookback=20,
    threshold=-0.015, hold=40, seed=SEED,
):
    """A path that is driftless on average but drifts after an observable state.

    The state is a ``lookback``-bar return below ``threshold``, evaluated using
    only bars at or before the flagging bar; the drift then applies to the
    *following* ``hold`` bars. So a signal that enters on the flag has no
    look-ahead, and yet captures a real conditional effect — which is exactly
    what a genuine edge looks like, and what the random-bar control should be
    able to detect as an excess over the unconditional baseline.
    """
    rng = np.random.default_rng(seed)
    logp = np.zeros(n)
    flags = np.zeros(n, dtype=bool)
    drift_left = 0

    for i in range(1, n):
        if i > lookback and drift_left == 0:
            past_return = logp[i - 1] - logp[i - 1 - lookback]
            if past_return < threshold:
                flags[i - 1] = True        # observable at bar i-1's close
                drift_left = hold
        mu = drift_when_flagged if drift_left > 0 else 0.0
        if drift_left > 0:
            drift_left -= 1
        logp[i] = logp[i - 1] + rng.normal(mu, sigma)

    close = 100 * np.exp(logp)
    wick = np.abs(rng.normal(0, sigma / 2, n)) * close
    return close + wick, close - wick, close, np.flatnonzero(flags)


def take_trades(highs, lows, closes, idx):
    """Walk the bracket at each entry; return R-multiples."""
    return np.array(
        [
            walk_bracket(
                highs, lows, int(i), float(closes[i]),
                STOP_FRAC, TARGET_FRAC, direction=1, max_bars=400,
            ).r_multiple
            for i in idx
        ],
        dtype=float,
    )


def hr(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def evaluate(label, highs, lows, closes, idx, day_of):
    r = take_trades(highs, lows, closes, idx)

    hr(f"CANDIDATE {label}   (n={len(r)}, raw mean {r.mean():+.4f}R)")

    # ── gate 1: is there gross edge above the toll at all? ───────────────────
    toll = fee_in_r_wall(ROUND_TRIP_BPS, STOP_FRAC)
    costs = decompose_gross_net(r, toll)
    print(f"1. COST DECOMPOSITION      toll {toll:.4f}R at a {STOP_FRAC:.3%} stop")
    print(f"   gross {costs['gross_mean']:+.4f}R   net {costs['net_mean']:+.4f}R"
          f"   toll multiple {costs['toll_multiple']:.2f}x  -> {costs['verdict'].upper()}")

    # ── gate 2: does it beat a random entry with the same bracket? ───────────
    ctl = random_bar_control(
        highs, lows, n_trades=len(r), stop_frac=STOP_FRAC, target_frac=TARGET_FRAC,
        n_replicates=120, warmup=100, seed=SEED + 1,
    )
    exc = excess_over_control(r, ctl)
    print(f"2. RANDOM-BAR CONTROL      baseline {ctl['mean_r']:+.4f}R "
          f"[{ctl['ci95'][0]:+.3f}, {ctl['ci95'][1]:+.3f}]")
    print(f"   excess {exc['excess_r']:+.4f}R   z vs control {exc['z_vs_control']:+.2f}")

    # ── gate 3: is the result just first-passage geometry? ───────────────────
    mir = mirrored_bracket_control(highs, lows, idx, closes[idx],
                                  STOP_FRAC, TARGET_FRAC)
    print(f"3. MIRRORED BRACKET        as-designed {mir['as_designed']['mean_r']:+.4f}R "
          f"(win {mir['as_designed']['win_rate']:.1%})   "
          f"mirrored {mir['mirrored']['mean_r']:+.4f}R "
          f"(win {mir['mirrored']['win_rate']:.1%})")
    print(f"   R gap {mir['r_gap']:+.4f}   win-rate gap {mir['win_rate_gap']:+.1%}"
          f"   -> {'orientation-specific' if abs(mir['r_gap']) > 0.1 else 'geometry, not prediction'}")

    # ── gate 4: how much of n is real? ──────────────────────────────────────
    clu = clustered_tstat(r, day_of)
    eff = effective_sample_size(r, day_of)
    print(f"4. CLUSTERED INFERENCE     naive t {clu['t_naive']:+.2f}  ->  "
          f"clustered t {clu['t_clustered']:+.2f}  (inflation {clu['inflation']:.2f}x, "
          f"df {clu['df']})")
    print(f"   n {eff['n']} -> n_eff {eff['n_eff']:.0f}  "
          f"(ICC {eff['icc']:.2f}, haircut {eff['haircut']:.2f})")

    # ── gate 5: does it hold in both halves? ────────────────────────────────
    hs = half_split_stability(r, min_t=1.0)
    print(f"5. HALF-SPLIT STABILITY    first {hs['first_half']['mean']:+.4f}R "
          f"(t {hs['first_half']['t']:+.2f})   "
          f"second {hs['second_half']['mean']:+.4f}R (t {hs['second_half']['t']:+.2f})"
          f"   -> {'STABLE' if hs['stable'] else 'NOT STABLE'}")

    verdict = (
        costs["verdict"] == "viable"
        and exc["z_vs_control"] > 2
        and clu["t_clustered"] > 2
        and hs["stable"]
    )
    print(f"\n   VERDICT: {'PASSES the battery' if verdict else 'KILLED'}")
    return verdict


def grid_scan_demo():
    """A 40-cell scan of pure noise, which must not clear a family-wise bar."""
    hr("MULTIPLICITY: a 40-cell parameter scan over pure noise")
    rng = np.random.default_rng(SEED + 9)
    cells = [rng.normal(0, 1, 220) for _ in range(40)]
    res = signed_permutation_maxt(cells, n_permutations=2000, seed=SEED + 10)
    best_naive = res["per_cell_t"].max()
    solo_p = 2 * (1 - 0.5 * (1 + math.erf(best_naive / np.sqrt(2))))
    print(f"   best cell's own t-stat        {best_naive:+.2f}   "
          f"(would read as p ~ {solo_p:.3f} if reported alone)")
    print(f"   family-wise bar (signed max-t) p_FWER = {res['p_fwer']:.3f}")
    print(f"   null max-t quantiles          q95 {res['null_quantiles']['q95']:+.2f}   "
          f"q99 {res['null_quantiles']['q99']:+.2f}")
    print(f"\n   The best of 40 noise cells looks significant in isolation and is"
          f"\n   correctly rejected once the search is priced in.")


def main() -> int:
    highs, lows, closes = synth_path(drift=0.0)
    rng = np.random.default_rng(SEED + 2)

    # entries arrive in bursts, as live signals do -> clustered observations
    burst_days = rng.integers(0, 120, size=N_TRADES)
    idx = np.sort(rng.integers(200, len(closes) - 500, size=N_TRADES))

    hr("VALIDATION HARNESS DEMO — synthetic data, fully reproducible")
    print(f"   seed {SEED} | bracket {STOP_FRAC:.2%} stop / {TARGET_FRAC:.2%} target "
          f"(RR {TARGET_FRAC / STOP_FRAC:.1f}) | round trip {ROUND_TRIP_BPS} bps")

    a = evaluate("A — random entries, driftless path (no edge, and it shows)",
                 highs, lows, closes, idx, burst_days)

    dh, dl, dc = synth_path(drift=0.00035, seed=SEED + 5)
    didx = np.sort(rng.integers(200, len(dc) - 500, size=N_TRADES))
    b = evaluate("B — random entries, DRIFTING path (profitable, still not an edge)",
                 dh, dl, dc, didx, burst_days)

    ch, cl, cc, cidx = state_dependent_path(seed=SEED + 6)
    cidx = cidx[(cidx > 200) & (cidx < len(cc) - 500)]
    c_days = rng.integers(0, 120, size=len(cidx))
    c = evaluate(f"C — conditional signal on a state-dependent path "
                 f"({len(cidx)} flagged entries)",
                 ch, cl, cc, cidx, c_days)

    grid_scan_demo()

    hr("SUMMARY")
    rows = [
        ("A", "no edge, looks like none", a, False),
        ("B", "no edge, looks profitable", b, False),
        ("C", "genuine conditional edge", c, True),
    ]
    for tag, desc, got, want in rows:
        mark = "correct" if got == want else "WRONG"
        print(f"   {tag}  {desc:<28} -> {'PASSED' if got else 'KILLED':>6}   [{mark}]")

    print("\n   B is the case worth studying: gross positive, cost-viable,"
          "\n   clustered t above 3, stable in both halves — and still not an edge,"
          "\n   because random entries on the same path earn the same. Four gates"
          "\n   passed it. The random-bar control is the one that caught it.\n")
    return 0 if all(g == w for _, _, g, w in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
