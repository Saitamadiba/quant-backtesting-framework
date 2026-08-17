"""Tests for backtrader_framework.validation.

These go beyond smoke-testing: each group asserts the *statistical property* the
tool exists to provide — a correct null on random data, a detected effect on
planted data, and the conservative behaviour at the edges. A validation harness
that is itself wrong is the worst failure mode in the stack, because it reports
confidently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtrader_framework.validation import (
    clustered_tstat,
    decompose_gross_net,
    drawdown_matched_control,
    effective_sample_size,
    excess_over_control,
    fee_in_r_wall,
    half_split_stability,
    holm_bonferroni,
    mirrored_bracket_control,
    one_sample_t,
    placebo_pvalue,
    placebo_schedule_control,
    random_bar_control,
    signed_permutation_maxt,
    toll_neutral_threshold,
    walk_bracket,
    weighting_disagreement,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _random_walk(n=4000, sigma=0.004, seed=0):
    """Driftless GBM-ish bars. Any 'edge' found here is an artefact."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    wig = np.abs(rng.normal(0, sigma / 2, n)) * close
    return close + wig, close - wig, close   # highs, lows, closes


# ─────────────────────────────────────────────────────────────────────────────
# walk_bracket
# ─────────────────────────────────────────────────────────────────────────────
def test_bracket_books_target_and_stop_at_correct_r():
    highs = np.array([100.0, 100.0, 103.0])
    lows = np.array([100.0, 100.0, 100.0])
    out = walk_bracket(highs, lows, 0, 100.0, stop_frac=0.01, target_frac=0.02)
    assert out.outcome == "target"
    assert out.r_multiple == pytest.approx(2.0)

    lows2 = np.array([100.0, 100.0, 98.0])
    highs2 = np.array([100.0, 100.0, 100.0])
    out2 = walk_bracket(highs2, lows2, 0, 100.0, stop_frac=0.01, target_frac=0.02)
    assert out2.outcome == "stop"
    assert out2.r_multiple == pytest.approx(-1.0)


def test_same_bar_tie_resolves_against_the_trade():
    """The conservative tie is load-bearing: the optimistic one fakes edge."""
    highs = np.array([100.0, 105.0])     # spans target
    lows = np.array([100.0, 95.0])       # and stop
    out = walk_bracket(highs, lows, 0, 100.0, stop_frac=0.01, target_frac=0.02)
    assert out.outcome == "stop"
    assert out.r_multiple == pytest.approx(-1.0)


def test_short_direction_is_mirrored():
    highs = np.array([100.0, 100.0, 100.0])
    lows = np.array([100.0, 100.0, 97.0])
    out = walk_bracket(highs, lows, 0, 100.0, 0.01, 0.02, direction=-1)
    assert out.outcome == "target"
    assert out.r_multiple == pytest.approx(2.0)


def test_bracket_rejects_bad_inputs():
    h = l = np.ones(5) * 100
    with pytest.raises(ValueError):
        walk_bracket(h, l, 0, 100.0, 0.0, 0.02)
    with pytest.raises(ValueError):
        walk_bracket(h, l, 0, 100.0, 0.01, 0.02, direction=0)


# ─────────────────────────────────────────────────────────────────────────────
# random-bar control — the null must be centred near zero on a driftless path
# ─────────────────────────────────────────────────────────────────────────────
def test_random_bar_control_is_near_zero_on_driftless_path():
    highs, lows, _ = _random_walk(seed=1)
    ctl = random_bar_control(
        highs, lows, n_trades=120, stop_frac=0.01, target_frac=0.02,
        n_replicates=60, warmup=50, seed=7,
    )
    # a symmetric bracket on a martingale earns ~0R; allow generous slack for
    # the finite sample and the conservative same-bar tie (which biases down).
    assert -0.25 < ctl["mean_r"] < 0.10
    assert ctl["ci95"][0] < ctl["mean_r"] < ctl["ci95"][1]


def test_excess_over_control_detects_a_planted_edge_and_not_noise():
    highs, lows, _ = _random_walk(seed=2)
    ctl = random_bar_control(
        highs, lows, n_trades=100, stop_frac=0.01, target_frac=0.02,
        n_replicates=80, warmup=50, seed=3,
    )
    # a "signal" that is really just the control -> excess ~0
    noise = np.full(100, ctl["mean_r"])
    assert abs(excess_over_control(noise, ctl)["excess_r"]) < 1e-9

    # a genuinely better book -> large positive z
    planted = np.full(100, ctl["mean_r"] + 0.5)
    assert excess_over_control(planted, ctl)["z_vs_control"] > 3


# ─────────────────────────────────────────────────────────────────────────────
# mirrored bracket — R-expectancy should barely move, win rate should swing
# ─────────────────────────────────────────────────────────────────────────────
def test_mirrored_bracket_moves_win_rate_far_more_than_r():
    highs, lows, closes = _random_walk(n=6000, seed=4)
    rng = np.random.default_rng(5)
    idx = rng.integers(50, 5800, size=250)
    res = mirrored_bracket_control(
        highs, lows, idx, closes[idx], stop_frac=0.01, target_frac=0.02,
    )
    # This is the whole lesson: on a driftless path the 1:2 and 2:1 brackets
    # have very different win rates but similar R-expectancy.
    assert abs(res["win_rate_gap"]) > 0.15
    assert abs(res["r_gap"]) < abs(res["win_rate_gap"]) * 4


# ─────────────────────────────────────────────────────────────────────────────
# drawdown matching
# ─────────────────────────────────────────────────────────────────────────────
def test_drawdown_matched_controls_match_on_drawdown():
    rng = np.random.default_rng(11)
    dd = rng.uniform(0, 0.5, 3000)
    events = np.array([10, 500, 1200, 2500])
    ctl = drawdown_matched_control(events, dd, tolerance=0.01,
                                  n_per_event=1, exclude_window=5, seed=2)
    assert ctl.size > 0
    for c in ctl:
        assert np.min(np.abs(dd[events] - dd[c])) <= 0.01


def test_drawdown_matching_excludes_the_event_neighbourhood():
    dd = np.linspace(0, 1, 1000)
    events = np.array([500])
    ctl = drawdown_matched_control(events, dd, tolerance=1.0,
                                  n_per_event=20, exclude_window=25, seed=1)
    assert not np.any((ctl >= 475) & (ctl <= 525))


# ─────────────────────────────────────────────────────────────────────────────
# placebo schedules
# ─────────────────────────────────────────────────────────────────────────────
def test_placebo_schedules_preserve_count_and_weekday():
    cand = pd.date_range("2024-01-01", periods=600, freq="D")
    real = cand[[3, 45, 90, 200, 380]]
    plac = placebo_schedule_control(real, cand, n_placebos=25,
                                    preserve_weekday=True, seed=4)
    assert len(plac) == 25
    real_dow = sorted(pd.DatetimeIndex(real).dayofweek)
    for p in plac:
        assert len(p) == len(real)
        assert sorted(pd.DatetimeIndex(p).dayofweek) == real_dow


def test_placebo_pvalue_bounds_and_direction():
    # real score beats every placebo -> smallest attainable p, never 0
    assert placebo_pvalue(10.0, [1.0] * 99) == pytest.approx(1 / 100)
    # real score is unremarkable -> p near 1
    assert placebo_pvalue(0.0, [1.0] * 99) == pytest.approx(100 / 100)


# ─────────────────────────────────────────────────────────────────────────────
# multiplicity
# ─────────────────────────────────────────────────────────────────────────────
def test_signed_maxt_null_is_calibrated_on_pure_noise():
    """A 40-cell scan of pure noise must NOT clear a 5% family-wise bar."""
    rng = np.random.default_rng(21)
    cells = [rng.normal(0, 1, 200) for _ in range(40)]
    res = signed_permutation_maxt(cells, n_permutations=800, seed=1)
    assert res["p_fwer"] > 0.05
    assert res["n_cells"] == 40


def test_signed_maxt_detects_one_genuinely_strong_cell():
    rng = np.random.default_rng(22)
    cells = [rng.normal(0, 1, 200) for _ in range(19)]
    cells.append(rng.normal(0.6, 1, 200))       # planted effect
    res = signed_permutation_maxt(cells, n_permutations=800, seed=2)
    assert res["p_fwer"] < 0.05
    assert res["best_cell"] == 19


def test_signed_maxt_is_stricter_than_absolute_maxt_would_be():
    """The point of §5.1: an all-negative family must not score well when the
    claim is positive."""
    rng = np.random.default_rng(23)
    cells = [rng.normal(-0.6, 1, 200) for _ in range(10)]
    res = signed_permutation_maxt(cells, n_permutations=600, direction=1, seed=3)
    assert res["p_fwer"] > 0.5                     # nothing positive here
    flipped = signed_permutation_maxt(cells, n_permutations=600, direction=-1, seed=3)
    assert flipped["p_fwer"] < 0.05                # the negative claim is real


def test_holm_is_monotone_and_no_weaker_than_bonferroni():
    p = np.array([0.001, 0.02, 0.03, 0.7])
    res = holm_bonferroni(p, alpha=0.05)
    adj = res["adjusted"]
    # monotone in the sorted order
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)
    # never rejects something Bonferroni wouldn't, never rejects less
    assert res["n_reject"] >= int(np.sum(p * p.size <= 0.05))
    assert np.all(adj <= 1.0)


def test_half_split_flags_a_one_regime_effect():
    values = np.concatenate([np.full(60, 0.5), np.full(60, -0.5)])
    # add jitter so the t-stats are finite
    values = values + np.random.default_rng(9).normal(0, 0.1, 120)
    res = half_split_stability(values)
    assert not res["signs_agree"]
    assert not res["stable"]


def test_half_split_passes_a_consistent_effect():
    values = np.random.default_rng(10).normal(0.4, 0.5, 200)
    res = half_split_stability(values, min_t=1.0)
    assert res["signs_agree"] and res["stable"]


def test_one_sample_t_degenerate_cases():
    assert np.isnan(one_sample_t(np.array([1.0])))
    assert np.isnan(one_sample_t(np.array([2.0, 2.0, 2.0])))


# ─────────────────────────────────────────────────────────────────────────────
# clustered inference
# ─────────────────────────────────────────────────────────────────────────────
def test_clustered_t_is_smaller_than_naive_when_clusters_are_correlated():
    """The headline claim: perfectly duplicated bets must not add significance."""
    rng = np.random.default_rng(31)
    day_effect = rng.normal(0.1, 0.4, 40)
    values, clusters = [], []
    for d, eff in enumerate(day_effect):
        for _ in range(8):                 # 8 near-identical bets per day
            values.append(eff + rng.normal(0, 0.01))
            clusters.append(d)
    res = clustered_tstat(values, clusters)
    assert res["n"] == 320 and res["n_clusters"] == 40
    assert abs(res["t_clustered"]) < abs(res["t_naive"])
    assert res["inflation"] > 2.0
    assert res["df"] == 39


def test_clustered_t_matches_naive_when_one_obs_per_cluster():
    rng = np.random.default_rng(32)
    x = rng.normal(0.2, 1.0, 200)
    res = clustered_tstat(x, np.arange(200), small_sample_correction=False)
    assert res["t_clustered"] == pytest.approx(res["t_naive"], rel=0.02)


def test_effective_n_collapses_toward_cluster_count_when_icc_is_high():
    rng = np.random.default_rng(33)
    values, clusters = [], []
    for d in range(30):
        base = rng.normal(0, 1)
        for _ in range(10):
            values.append(base + rng.normal(0, 1e-3))   # ICC ~ 1
            clusters.append(d)
    res = effective_sample_size(values, clusters)
    assert res["n"] == 300
    assert res["icc"] > 0.95
    assert res["n_eff"] < 40          # ~= the 30 clusters, not 300
    assert res["haircut"] < 0.15


def test_effective_n_is_near_n_when_clusters_carry_no_information():
    rng = np.random.default_rng(34)
    x = rng.normal(0, 1, 300)
    res = effective_sample_size(x, rng.integers(0, 30, 300))
    assert res["n_eff"] > 0.6 * res["n"]


def test_weighting_disagreement_detects_a_sign_flip():
    # one frantic day dominates the observation-weighted mean
    values = list(np.full(100, 0.2)) + [-1.0, -1.0, -1.0]
    clusters = [0] * 100 + [1, 2, 3]
    res = weighting_disagreement(values, clusters)
    assert res["obs_weighted_mean"] > 0
    assert res["cluster_weighted_mean"] < 0
    assert res["signs_disagree"]


def test_clustering_requires_two_clusters():
    with pytest.raises(ValueError):
        clustered_tstat([1.0, 2.0, 3.0], [0, 0, 0])


# ─────────────────────────────────────────────────────────────────────────────
# costs
# ─────────────────────────────────────────────────────────────────────────────
def test_fee_wall_scales_inversely_with_stop_distance():
    assert fee_in_r_wall(15.4, 0.0064) == pytest.approx(0.2406, abs=1e-3)
    assert fee_in_r_wall(15.4, 0.0015) == pytest.approx(1.0267, abs=1e-3)
    # halving the stop doubles the toll in R
    assert fee_in_r_wall(10, 0.005) == pytest.approx(2 * fee_in_r_wall(10, 0.01))
    with pytest.raises(ValueError):
        fee_in_r_wall(10, 0)


def test_decompose_flags_toll_dominated_and_no_gross():
    # the documented case: gross +0.0647 against a 0.1919 toll => 3.0x
    res = decompose_gross_net(np.full(200, 0.0647), 0.1919)
    assert res["toll_multiple"] == pytest.approx(2.965, abs=0.01)
    assert res["verdict"] == "toll_dominated"
    assert res["net_mean"] < 0

    none = decompose_gross_net(np.full(50, -0.01), 0.05)
    assert none["verdict"] == "no_gross"
    assert np.isinf(none["toll_multiple"])

    ok = decompose_gross_net(np.full(50, 0.40), 0.10)
    assert ok["verdict"] == "viable"
    assert ok["net_mean"] == pytest.approx(0.30)


def test_decompose_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        decompose_gross_net([0.1, 0.2, 0.3], [0.05, 0.05])


def test_toll_neutral_threshold_reproduces_the_documented_ratio():
    # a stop whose round trip costs 31.1 units against 2.24 per sigma needs ~14
    assert toll_neutral_threshold(31.1, 2.24) == pytest.approx(13.9, abs=0.1)
    with pytest.raises(ValueError):
        toll_neutral_threshold(1.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# The property that matters most: does the battery SEPARATE?
# A harness that passes everything is decoration; one that kills everything is
# useless. These two tests pin the separation documented in
# docs/examples/validation_demo.py.
# ─────────────────────────────────────────────────────────────────────────────
def test_random_bar_control_neutralises_market_drift():
    """A profitable book on a drifting path is not an edge.

    This is the case that slips past every other gate: gross is positive, costs
    are covered, the t-stat is large and it is stable across halves — because
    the market went up. The random-bar control is the only gate that catches it,
    since random entries on the same path collect the same drift.
    """
    rng = np.random.default_rng(41)
    n = 8000
    close = 100 * np.exp(np.cumsum(rng.normal(0.00035, 0.004, n)))
    wick = np.abs(rng.normal(0, 0.002, n)) * close
    highs, lows = close + wick, close - wick

    idx = np.sort(rng.integers(200, n - 500, size=200))
    r = np.array([
        walk_bracket(highs, lows, int(i), float(close[i]), 0.01, 0.02).r_multiple
        for i in idx
    ])

    # looks good in isolation
    assert r.mean() > 0
    assert one_sample_t(r) > 2

    # ...and vanishes against the control
    ctl = random_bar_control(highs, lows, n_trades=len(r), stop_frac=0.01,
                            target_frac=0.02, n_replicates=80, warmup=100, seed=42)
    exc = excess_over_control(r, ctl)
    assert abs(exc["z_vs_control"]) < 2.0, (
        "random entries on a drifting path must not show excess over a "
        "random-bar control on the same path"
    )


def test_conditional_edge_survives_the_control():
    """A genuine state-conditional effect must clear the same control.

    Path is driftless unconditionally but drifts for a fixed window after an
    observable state; the signal enters on that state using only past bars. The
    control samples uniformly and therefore collects the unconditional zero, so
    a real excess should appear.
    """
    rng = np.random.default_rng(43)
    n, sigma, hold = 9000, 0.004, 40
    logp = np.zeros(n)
    flags = np.zeros(n, dtype=bool)
    left = 0
    for i in range(1, n):
        if i > 20 and left == 0 and (logp[i - 1] - logp[i - 21]) < -0.015:
            flags[i - 1] = True          # decided from bars <= i-1: no look-ahead
            left = hold
        mu = 0.0010 if left > 0 else 0.0
        left = max(0, left - 1)
        logp[i] = logp[i - 1] + rng.normal(mu, sigma)

    close = 100 * np.exp(logp)
    wick = np.abs(rng.normal(0, sigma / 2, n)) * close
    highs, lows = close + wick, close - wick

    idx = np.flatnonzero(flags)
    idx = idx[(idx > 200) & (idx < n - 500)]
    assert idx.size >= 30, "fixture produced too few flagged entries"

    r = np.array([
        walk_bracket(highs, lows, int(i), float(close[i]), 0.01, 0.02).r_multiple
        for i in idx
    ])
    ctl = random_bar_control(highs, lows, n_trades=len(r), stop_frac=0.01,
                            target_frac=0.02, n_replicates=80, warmup=100, seed=44)
    assert excess_over_control(r, ctl)["z_vs_control"] > 2.0
