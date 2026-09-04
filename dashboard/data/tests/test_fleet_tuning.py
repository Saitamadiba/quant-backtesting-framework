"""Hermetic tests for WS6's tuning kill bar.

The bar exists because a search reports the MAXIMUM of many noisy estimates,
which is biased upward by construction. Every test here is about refusing that
maximum unless a frozen holdout confirms it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DASH = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DASH))
sys.path.insert(0, str(_ROOT))

from data import fleet_tuning as ft                           # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  What may NOT be tuned
# ══════════════════════════════════════════════════════════════════════════════
def test_live_seat_parameters_are_named_as_forbidden():
    assert "seat entry parameters" in ft.FORBIDDEN_TARGETS
    assert "stop distance" in ft.FORBIDDEN_TARGETS
    assert "risk percent" in ft.FORBIDDEN_TARGETS


def test_no_search_space_reaches_a_seat_parameter():
    """A sampler pointed at a seat's own history is the WFO overfit trap with
    better manners."""
    spaces = {**ft.CLS_SPACE}
    banned = ("stop", "sl", "tp", "risk", "size", "ladder", "rr")
    for key in spaces:
        assert not any(b == key.lower() for b in banned), key


def test_the_module_tunes_no_live_bot_module():
    import ast
    tree = ast.parse(Path(ft.__file__).read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert not imported & {"HyroTrader", "Liquidity_Raid", "FVG_Strategy",
                           "meta_conductor", "subprocess", "paramiko"}


# ══════════════════════════════════════════════════════════════════════════════
#  The kill bar
# ══════════════════════════════════════════════════════════════════════════════
def test_a_confirmed_improvement_ships():
    v = ft.apply_kill_bar({"cfg": 1}, search_metric=0.60, search_se=0.02,
                          holdout_metric=0.59, default_holdout=0.55,
                          halves=(0.58, 0.60))
    assert v.ships and not v.reasons


def test_a_holdout_beyond_one_se_is_refused():
    """The classic overfit signature: the search looks great, the holdout does
    not follow."""
    v = ft.apply_kill_bar({"cfg": 1}, search_metric=0.65, search_se=0.01,
                          holdout_metric=0.52, default_holdout=0.50,
                          halves=(0.52, 0.52))
    assert not v.ships
    assert any("more than one SE" in r for r in v.reasons)


def test_disagreeing_halves_are_refused():
    v = ft.apply_kill_bar({"cfg": 1}, search_metric=0.60, search_se=0.05,
                          holdout_metric=0.58, default_holdout=0.55,
                          halves=(0.62, 0.51))
    assert not v.ships
    assert any("halves disagree" in r for r in v.reasons)


def test_failing_to_beat_the_default_is_refused():
    v = ft.apply_kill_bar({"cfg": 1}, search_metric=0.56, search_se=0.05,
                          holdout_metric=0.54, default_holdout=0.56,
                          halves=(0.54, 0.54))
    assert not v.ships
    assert any("does not beat the default" in r for r in v.reasons)


def test_every_condition_is_required_not_just_a_majority():
    v = ft.apply_kill_bar({"cfg": 1}, search_metric=0.60, search_se=0.001,
                          holdout_metric=0.59, default_holdout=0.50,
                          halves=(0.58, 0.60))
    assert v.beats_default and v.halves_agree
    assert not v.within_one_se and not v.ships, "two of three is not a pass"


def test_the_verdict_records_the_numbers_it_judged():
    v = ft.apply_kill_bar({"cfg": 7}, 0.6, 0.02, 0.59, 0.55, (0.58, 0.60))
    assert v.winner == {"cfg": 7}
    assert v.search_metric == 0.6 and v.holdout_metric == 0.59


# ══════════════════════════════════════════════════════════════════════════════
#  Held-out log-likelihood — the only fair way to compare 2 states with 3
# ══════════════════════════════════════════════════════════════════════════════
def test_hmm_log_likelihood_matches_brute_force_enumeration():
    """The scaled forward recursion must equal summing over every state path."""
    import itertools
    from backtrader_framework.optimization.hmm_regime import GaussianHMM, _logsumexp
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.5, (400, 2)), rng.normal(3, 0.5, (400, 2))])
    m = GaussianHMM(n_states=2, vol_feature_index=0).fit(X)
    Z = X[:7]
    lb, la, lp = m._log_emission(Z), np.log(m.A), np.log(m.pi)
    tot = []
    for path in itertools.product(range(m.K), repeat=len(Z)):
        s = lp[path[0]] + lb[0, path[0]]
        for t in range(1, len(Z)):
            s += la[path[t - 1], path[t]] + lb[t, path[t]]
        tot.append(s)
    assert m.log_likelihood(Z) == pytest.approx(_logsumexp(np.array(tot)), abs=1e-8)


def test_hmm_log_likelihood_needs_a_fitted_model():
    from backtrader_framework.optimization.hmm_regime import GaussianHMM
    with pytest.raises(ValueError):
        GaussianHMM(n_states=2).log_likelihood(np.zeros((5, 2)))


def test_hmm_cv_is_scored_per_bar_not_per_fold():
    """Configurations with different windows see different numbers of held-out
    bars; a total would reward whichever was graded on more of them."""
    import inspect
    src = inspect.getsource(ft.hmm_cv_loglik)
    assert "ll / len(te)" in src


def test_hmm_cv_embargoes_between_train_and_test():
    import inspect
    src = inspect.getsource(ft.hmm_cv_loglik)
    assert "emb" in src and "tr_n + emb" in src


def test_hmm_cv_refuses_a_window_longer_than_the_data():
    feat = pd.DataFrame({"rv24": np.random.default_rng(0).normal(size=500),
                         "abs_ret24": np.random.default_rng(1).normal(size=500)})
    out = ft.hmm_cv_loglik(feat, 2, ("rv24", "abs_ret24"), train_days=90)
    assert out["status"].startswith("need")


def test_hmm_default_is_the_ws3_deployed_configuration():
    assert ft.HMM_DEFAULT["n_states"] == 2
    assert ft.HMM_DEFAULT["train_days"] == 90 and ft.HMM_DEFAULT["step_days"] == 20


# ══════════════════════════════════════════════════════════════════════════════
#  The optuna substitution is declared, not hidden
# ══════════════════════════════════════════════════════════════════════════════
def test_the_missing_optuna_substitution_is_documented():
    src = Path(ft.__file__).read_text()
    assert "optuna" in src and "random search" in src, (
        "a silent substitution is discovered later in the code; a declared one is not")


def test_the_classifier_space_is_hyperparameters_only():
    assert set(ft.CLS_SPACE) == {"max_depth", "learning_rate", "max_iter",
                                 "l2_regularization"}
