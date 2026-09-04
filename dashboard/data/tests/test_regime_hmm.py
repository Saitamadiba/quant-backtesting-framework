"""Hermetic tests for WS3's walk-forward HMM regime path.

No network, no VPS, no duckdb. The point of almost every test here is the same
one: **the state at bar t may not know bar t+1.** The ER ranker died of exactly
that in 2026-08-17, so the leak paths are tested directly rather than assumed.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DASH = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DASH))
sys.path.insert(0, str(_ROOT))

from data import regime_hmm as rh                            # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  Leak discipline — the whole reason the module exists
# ══════════════════════════════════════════════════════════════════════════════
def test_the_oos_path_never_uses_viterbi_or_smoothing():
    """`predict` is Viterbi (backward pass) and is IS-labelling only. If it ever
    appears in this module, a 'regime' has seen the future."""
    src = Path(rh.__file__).read_text()
    assert ".predict(" not in src
    assert "forward_filter" in src
    assert "smooth" not in src.lower().replace("smoothing", "")  # only the warning text


def test_rolling_refit_fits_strictly_before_it_applies():
    """Every window's `applied_from` must be after its `fit_to` — the model that
    labels a bar was estimated on bars that ended before it."""
    feat = _synthetic_features(days=200)
    path = rh.rolling_states(feat, n_states=2, train_days=90, step_days=20)
    assert path.n_windows >= 2
    for m in path.models:
        assert m["applied_from"] > m["fit_to"], m


def test_windows_do_not_overlap_in_what_they_label():
    feat = _synthetic_features(days=200)
    path = rh.rolling_states(feat, n_states=2, train_days=90, step_days=20)
    ranges = [(m["applied_from"], m["applied_to"]) for m in path.models]
    for (a1, b1), (a2, _b2) in zip(ranges, ranges[1:]):
        assert a2 > b1, "a bar must be labelled by exactly one model"


def test_standardisation_uses_the_fitting_window_only():
    import inspect
    src = inspect.getsource(rh.rolling_states)
    assert "tr.mean(axis=0)" in src and "tr.std(axis=0)" in src
    assert "X_all.mean" not in src, "whole-sample stats would leak the future"


def test_the_seam_carries_the_previous_posterior():
    import inspect
    src = inspect.getsource(rh.rolling_states)
    assert "init_state_probs=carry" in src, (
        "restarting from pi at every seam throws away the filter's memory")


def test_next_switch_is_never_a_feature():
    """`next_switch_state` is forward-looking by construction. It may label an
    outcome table and nothing else."""
    assert not any("next_switch" in f for f in rh.PRIMARY_FEATURES)
    assert not any("next_switch" in f for f in rh.EXTENDED_FEATURES)
    import inspect
    src = inspect.getsource(rh.evaluate_kill_bar)
    assert "next_switch" not in src, "the kill bar must not read a future column"


# ══════════════════════════════════════════════════════════════════════════════
#  Features — causal by construction
# ══════════════════════════════════════════════════════════════════════════════
def _synthetic_features(days=200, seed=0):
    n = days * rh.BARS_PER_DAY
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    vol = np.where((np.arange(n) // (30 * rh.BARS_PER_DAY)) % 2 == 0, 0.001, 0.004)
    lr = rng.normal(0, vol)
    close = 30000 * np.exp(np.cumsum(lr))
    df = pd.DataFrame({"close_ts": ts, "close": close})
    s = pd.Series(lr)
    df["rv24"] = s.rolling(rh.BARS_PER_DAY).std().to_numpy()
    df["abs_ret24"] = np.abs(pd.Series(close).pct_change(rh.BARS_PER_DAY).to_numpy())
    df["vol_of_vol"] = (s.rolling(rh.BARS_PER_DAY // 4).std()
                        .rolling(rh.BARS_PER_DAY).std().to_numpy())
    df["d_dvol24"] = rng.normal(0, 1, n)
    return df.dropna().reset_index(drop=True)


def test_features_are_indexed_by_the_bars_close_not_its_open():
    import inspect
    src = inspect.getsource(rh.btc_features)
    assert 'pd.Timedelta(minutes=BAR_MIN)' in src and "close_ts" in src, (
        "duckdb stamps a kline at its OPEN; a bar is only knowable at its close")


def test_rv24_at_a_bar_uses_no_later_bar():
    """Truncating the frame must not change the last surviving value."""
    f = _synthetic_features(days=40)
    cut = len(f) - 50
    a = f["rv24"].iloc[cut - 1]
    lr = np.log(f["close"]).diff()
    b = lr.iloc[:cut].rolling(rh.BARS_PER_DAY).std().iloc[-1]
    assert a == pytest.approx(b, rel=1e-9)


def test_conductor_baseline_matches_the_deployed_constant():
    """The baseline the HMM must beat is the conductor's frozen PATIENT rule.
    If market_state.py's constant moves, this copy is stale and the comparison
    is against a rule nothing is running."""
    ms = (_ROOT / "meta_conductor" / "market_state.py")
    if not ms.exists():
        pytest.skip("meta_conductor not present in this checkout")
    m = re.search(r"PATIENT_RV_FRAC\s*=\s*([0-9.]+)", ms.read_text())
    assert m and float(m.group(1)) == rh.PATIENT_RV_FRAC


def test_unknown_conductor_state_is_none_not_false():
    """Assigning None into a bool column coerces it to False, which would hand
    the baseline a decision it never made."""
    import inspect
    src = inspect.getsource(rh.btc_features)
    assert ".astype(object)" in src


# ══════════════════════════════════════════════════════════════════════════════
#  Sequencing
# ══════════════════════════════════════════════════════════════════════════════
def _path(states, start="2026-01-01"):
    ts = pd.date_range(start, periods=len(states), freq="15min", tz="UTC")
    return pd.DataFrame({"close_ts": ts, "state": states,
                         "p_state": np.ones(len(states)), "window": 0})


def test_transition_rows_sum_to_one():
    rng = np.random.default_rng(3)
    st = rng.integers(0, 2, 4000)
    out = rh.empirical_transitions(_path(st), 2, n_boot=50)
    assert np.allclose(out["transition"].sum(axis=1), 1.0)


def test_dwell_time_is_longer_for_a_stickier_state():
    st = np.repeat([0, 1], [3000, 300])
    st = np.concatenate([st, st, st])
    out = rh.empirical_transitions(_path(st), 2, n_boot=50)
    assert out["dwell_bars"][0] > out["dwell_bars"][1]


def test_transitions_never_span_a_day_boundary():
    import inspect
    src = inspect.getsource(rh.empirical_transitions)
    assert "same_day" in src, (
        "a gap between days is not a transition the tape actually made")


def test_bootstrap_resamples_whole_days():
    import inspect
    src = inspect.getsource(rh.empirical_transitions)
    assert "rng.integers(0, n_days" in src and "w[tcodes]" in src, (
        "bar-by-bar resampling treats an hours-long state as many observations "
        "and returns an interval far too tight")


def test_bootstrap_keeps_a_days_multiplicity():
    """A day drawn twice must count twice. A boolean `isin` mask — the obvious
    implementation — silently flattens it to once, which is a different (and
    narrower) bootstrap than the one being claimed."""
    import inspect
    src = inspect.getsource(rh.empirical_transitions)
    assert "np.bincount(pick" in src and "weights=" in src
    body = src.split('"""')[2]          # skip the docstring, which names the trap
    assert "np.isin" not in body, (
        "isin drops multiplicity, and on datetime64 it costs ~6s per call")


def test_bootstrap_interval_brackets_the_observed_matrix():
    rng = np.random.default_rng(8)
    st = (rng.random(6000) < 0.05).cumsum() % 2
    out = rh.empirical_transitions(_path(st.astype(int)), 2, n_boot=200)
    d = np.diag(out["transition"])
    assert np.all(out["ci_lo"].diagonal() <= d + 1e-9)
    assert np.all(out["ci_hi"].diagonal() >= d - 1e-9)


def test_bootstrap_is_fast_enough_to_run_on_a_page():
    """The naive version took ~50 minutes on the real path; the page calls this."""
    import time
    rng = np.random.default_rng(9)
    st = rng.integers(0, 2, 60000)
    t0 = time.time()
    rh.empirical_transitions(_path(st), 2, n_boot=200)
    assert time.time() - t0 < 20, "the bootstrap must not be the page's bottleneck"


def test_n_step_ahead_is_the_matrix_power():
    m = np.array([[0.9, 0.1], [0.2, 0.8]])
    out = rh.n_step_ahead(m, horizons=(2,))
    got = out[["to_0", "to_1"]].to_numpy()
    assert np.allclose(got, np.linalg.matrix_power(m, 2))


def test_next_switch_labels_look_forward_only_where_intended():
    st = np.array([0, 0, 0, 1, 1, 2, 2])
    out = rh.next_switch_labels(_path(st))
    assert list(out["next_switch_state"])[:3] == [1, 1, 1]
    assert list(out["next_switch_state"])[3:5] == [2, 2]
    assert out["next_switch_state"].iloc[-1] == -1, "no switch after the last run"


def test_bars_to_switch_counts_down():
    out = rh.next_switch_labels(_path(np.array([0, 0, 0, 1])))
    assert list(out["bars_to_switch"])[:3] == [3.0, 2.0, 1.0]


# ══════════════════════════════════════════════════════════════════════════════
#  The fill join
# ══════════════════════════════════════════════════════════════════════════════
def _tagged(n=200, seed=1, states=(0, 1)):
    rng = np.random.default_rng(seed)
    day = pd.to_datetime(pd.date_range("2026-01-01", periods=n // 4, tz="UTC")
                         .repeat(4)[:n])
    st = rng.integers(0, len(states), n)
    return pd.DataFrame({
        "family": ["F"] * n, "cluster": [f"c{i}" for i in range(n)],
        "r": rng.normal(-0.1, 1.0, n), "state": st.astype(float),
        "day": day, "conductor_quiet": rng.random(n) > 0.5,
        "next_switch_state": rng.integers(0, len(states), n),
        "bars_to_switch": rng.integers(1, 40, n).astype(float)})


def test_a_bar_closing_exactly_at_the_fill_is_knowable():
    """The WS0 spine learned this the hard way: one-tick-too-strict refused 98
    legitimate bar-close fills. `side='right'` is deliberate."""
    import inspect
    src = inspect.getsource(rh.fills_with_state)
    assert 'side="right"' in src


def test_fill_join_drops_a_state_older_than_a_day():
    import inspect
    src = inspect.getsource(rh.fills_with_state)
    assert "age_h <= 24.0" in src, "a week-old regime is not this fill's regime"


def test_outcome_tables_count_bets_not_rows():
    import inspect
    for fn in (rh.state_outcome_table, rh.conditional_outcome_table):
        assert 'groupby("cluster")["r"].mean()' in inspect.getsource(fn)


def test_state_outcome_table_shapes():
    out = rh.state_outcome_table(_tagged(200))
    assert set(out.columns) >= {"family", "state", "bets", "mean_r", "t"}
    assert out["bets"].sum() <= 200


# ══════════════════════════════════════════════════════════════════════════════
#  The kill bar — all four, or it is an atlas
# ══════════════════════════════════════════════════════════════════════════════
def test_kill_bar_requires_all_four_conditions():
    import inspect
    src = inspect.getsource(rh.evaluate_kill_bar)
    for cond in ("c1_separation", "c2_beats_conductor", "c3_half_split", "c4_bets"):
        assert cond in src
    assert re.search(r"GATE_CANDIDATE.*=.*c1_separation.*&.*c2_beats_conductor",
                     src, re.S), "a partial pass is not a pass"


def test_alpha_divides_by_families_times_k():
    import inspect
    assert "tested * max(n_k_tested, 1)" in inspect.getsource(rh.evaluate_kill_bar), (
        "fitting K=2 and K=3 is two looks at the same data")


def test_noise_does_not_become_a_gate_candidate():
    out = rh.evaluate_kill_bar(_tagged(400, seed=7), n_states=2, n_perm=300)
    assert not out.empty
    assert not out["GATE_CANDIDATE"].any()


def test_a_family_too_small_is_not_tested():
    out = rh.evaluate_kill_bar(_tagged(20), n_states=2, n_perm=100)
    assert out.empty or not out["GATE_CANDIDATE"].any()


def test_lift_is_measured_against_the_conductor_not_against_zero():
    import inspect
    src = inspect.getsource(rh.evaluate_kill_bar)
    assert "conductor_lift" in src and "hmm_lift - cond_lift" in src, (
        "the conductor's rule is already deployed and costs nothing new")


def test_best_split_lift_needs_two_populated_labels():
    bets = pd.DataFrame({"r": np.arange(20.0), "band": ["A"] * 20})
    assert rh._best_split_lift(bets, "band") is None


def test_best_split_lift_is_relative_to_the_family_mean():
    bets = pd.DataFrame({"r": [1.0] * 10 + [-1.0] * 10, "band": ["A"] * 10 + ["B"] * 10})
    assert rh._best_split_lift(bets, "band") == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Local-only, like WS2
# ══════════════════════════════════════════════════════════════════════════════
def test_module_never_touches_the_vps():
    src = Path(rh.__file__).read_text()
    for forbidden in ("scp ", "rsync", "ssh ", "sudo"):
        assert forbidden not in src
