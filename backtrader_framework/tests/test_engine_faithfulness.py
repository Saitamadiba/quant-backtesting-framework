"""Faithfulness review of the research engines (2026-09-03, fleet plan WS1/3/5).

Each test pins a behaviour that, before the review, would have let an engine
report something about a strategy that the strategy had not earned:

  * hmm_regime      — K>2 silently broken; vol column assumed at index 1; the
                      OOS filter forgot the IS window at the seam.
  * bayesian_edge   — plug-in variance overconfident at small n; the "skeptical"
                      prior centred on zero (no toll); every fill independent.
  * meta_strategy   — overlapping-label leak into the walk-forward; each realized
                      day counted H times; Sharpe annualised from H-day sums.
  * bayesian_tuner  — CV folds with no embargo; accuracy on imbalanced labels.
  * shap_analysis   — multiclass insights that never said which class.

Synthetic data throughout; sklearn is required for the meta-strategy tests,
optuna/shap are optional and skipped when absent.
"""
import numpy as np
import pandas as pd
import pytest

from backtrader_framework.optimization.hmm_regime import GaussianHMM, HMMRegimeAssessor, _logsumexp
from backtrader_framework.optimization.bayesian_edge import BayesianEdgeEstimator, PRIORS


# ═══════════════════════════════════════════════════════════════════════════════
#  HMM
# ═══════════════════════════════════════════════════════════════════════════════
def _simulate_hmm(rng, T, mu, sigma, A, pi):
    K, D = mu.shape
    states = np.zeros(T, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
    X = mu[states] + sigma[states] * rng.standard_normal((T, D))
    return X, states


def _naive_forward_filter(hmm, X):
    """Reference forward filter with explicit per-state loops (the pre-review
    implementation), used to prove the vectorised recursion is identical."""
    T = X.shape[0]
    log_B = hmm._log_emission(X)
    log_alpha = np.zeros((T, hmm.K))
    log_alpha[0] = np.log(hmm.pi + 1e-300) + log_B[0]
    log_alpha[0] -= _logsumexp(log_alpha[0])
    for t in range(1, T):
        for j in range(hmm.K):
            log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + np.log(hmm.A[:, j] + 1e-300)) + log_B[t, j]
        log_alpha[t] -= _logsumexp(log_alpha[t])
    return np.exp(log_alpha)


class TestGaussianHMM:
    def test_recovers_two_state_parameters(self):
        rng = np.random.default_rng(7)
        mu = np.array([[0.0, 0.5], [0.0, 2.0]])            # vol in column 1
        sigma = np.array([[0.5, 0.15], [1.5, 0.4]])
        A = np.array([[0.95, 0.05], [0.10, 0.90]])
        X, _ = _simulate_hmm(rng, 4000, mu, sigma, A, np.array([0.5, 0.5]))
        hmm = GaussianHMM(n_states=2, vol_feature_index=1).fit(X)
        assert hmm.mu[0, 1] < hmm.mu[1, 1]                  # state 0 = calm
        assert np.allclose(np.diag(hmm.A), np.diag(A), atol=0.04)
        assert np.allclose(hmm.mu[:, 1], mu[:, 1], atol=0.15)

    def test_three_states_are_real_and_ordered_by_volatility(self):
        """Before the review a 3-state fit carried a zero-mean, 1e-6-sigma ghost."""
        rng = np.random.default_rng(3)
        mu = np.array([[0.0, 0.5], [0.0, 1.5], [0.0, 3.0]])
        sigma = np.array([[0.4, 0.15], [0.9, 0.3], [2.0, 0.5]])
        A = np.array([[0.92, 0.06, 0.02], [0.05, 0.90, 0.05], [0.03, 0.07, 0.90]])
        X, _ = _simulate_hmm(rng, 6000, mu, sigma, A, np.ones(3) / 3)
        hmm = GaussianHMM(n_states=3, vol_feature_index=1).fit(X)
        assert np.all(np.diff(hmm.mu[:, 1]) > 0), hmm.mu[:, 1]
        assert np.all(hmm.sigma > 1e-3)                        # no ghost state
        assert np.allclose(hmm.A.sum(axis=1), 1.0)
        assert np.allclose(hmm.mu[:, 1], mu[:, 1], atol=0.35)

    def test_vol_feature_index_zero_relabels_on_the_right_column(self):
        """Fleet features put realized vol FIRST; the old code sorted on column 1."""
        rng = np.random.default_rng(11)
        mu = np.array([[0.5, 0.0], [2.0, 0.0]])            # vol in column 0
        sigma = np.array([[0.15, 0.5], [0.4, 1.5]])
        A = np.array([[0.95, 0.05], [0.10, 0.90]])
        X, _ = _simulate_hmm(rng, 3000, mu, sigma, A, np.array([0.5, 0.5]))
        hmm = GaussianHMM(n_states=2, vol_feature_index=0).fit(X)
        assert hmm.mu[0, 0] < hmm.mu[1, 0]

    def test_vectorised_forward_filter_equals_the_loop_reference(self):
        rng = np.random.default_rng(5)
        mu = np.array([[0.0, 0.5], [0.0, 2.0]]); sigma = np.array([[0.5, 0.2], [1.5, 0.5]])
        A = np.array([[0.9, 0.1], [0.2, 0.8]])
        X, _ = _simulate_hmm(rng, 800, mu, sigma, A, np.array([0.5, 0.5]))
        hmm = GaussianHMM(n_states=2).fit(X)
        fast = hmm.forward_filter(X[:200])
        slow = _naive_forward_filter(hmm, X[:200])
        assert np.allclose(fast, slow, atol=1e-10)

    def test_oos_filter_continues_the_is_filter_instead_of_restarting(self):
        """Filtering IS+OOS in one pass must equal filtering OOS with the IS
        posterior carried in — and differ from a cold restart from pi."""
        rng = np.random.default_rng(9)
        mu = np.array([[0.0, 0.5], [0.0, 2.0]]); sigma = np.array([[0.5, 0.2], [1.5, 0.5]])
        A = np.array([[0.97, 0.03], [0.05, 0.95]])
        X, _ = _simulate_hmm(rng, 2000, mu, sigma, A, np.array([0.5, 0.5]))
        hmm = GaussianHMM(n_states=2).fit(X[:1500])
        full = hmm.forward_filter(X)[1500:]
        carried = hmm.forward_filter(X[1500:], init_state_probs=hmm.forward_filter(X[:1500])[-1])
        cold = hmm.forward_filter(X[1500:])
        assert np.allclose(full, carried, atol=1e-9)
        # the seam matters: a cold restart disagrees on the first bars
        assert not np.allclose(full[:3], cold[:3], atol=1e-3)

    def test_transition_summary_reports_dwell_and_stationary(self):
        rng = np.random.default_rng(2)
        mu = np.array([[0.0, 0.5], [0.0, 2.0]]); sigma = np.array([[0.5, 0.2], [1.5, 0.5]])
        A = np.array([[0.95, 0.05], [0.10, 0.90]])
        X, _ = _simulate_hmm(rng, 3000, mu, sigma, A, np.array([0.5, 0.5]))
        hmm = GaussianHMM(n_states=2).fit(X)
        s = hmm.transition_summary()
        assert len(s['expected_dwell_bars']) == 2
        assert abs(sum(s['stationary']) - 1.0) < 1e-9
        assert s['expected_dwell_bars'][0] > s['expected_dwell_bars'][1] * 0.8  # calm is stickier here

    def test_assessor_uses_named_vol_feature_and_carries_posterior(self):
        rng = np.random.default_rng(4)
        n = 1500
        vol = np.where(np.arange(n) % 500 < 300, 0.5, 2.0) + 0.1 * rng.standard_normal(n)
        ret = vol * rng.standard_normal(n) * 0.3
        df = pd.DataFrame({'rv24': vol, 'ret': ret})
        a = HMMRegimeAssessor(n_states=2, features=['rv24', 'ret'], vol_feature='rv24')
        assert a.fit_on_is(df.iloc[:1000])
        assert a.is_last_posterior is not None
        probs = a.filter_oos(df.iloc[1000:])
        assert probs.shape == (500, 2)
        assert a.hmm.mu[0, 0] < a.hmm.mu[1, 0]


# ═══════════════════════════════════════════════════════════════════════════════
#  Bayesian edge
# ═══════════════════════════════════════════════════════════════════════════════
def _ci_width(est):
    lo, hi = est.summary()['mean_r']['credible_interval_95']
    return hi - lo


class TestBayesianEdge:
    @pytest.mark.parametrize("n, min_ratio", [(4, 1.15), (16, 1.04)])
    def test_student_t_posterior_is_wider_than_plugin_at_small_n(self, n, min_ratio):
        """The plug-in variance treats the sample s² as known; the t posterior does not."""
        rng = np.random.default_rng(1)
        r = rng.normal(0.1, 1.0, n)
        nig = BayesianEdgeEstimator().fit(r, prior='uninformative', variance='nig')
        plug = BayesianEdgeEstimator().fit(r, prior='uninformative', variance='plugin')
        ratio = _ci_width(nig) / _ci_width(plug)
        assert ratio > min_ratio, ratio
        assert nig.summary()['variance_model'] == 'student_t_nig'
        assert plug.summary()['variance_model'] == 'normal_plugin'

    def test_student_t_and_plugin_agree_at_large_n(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.1, 1.0, 2000)
        nig = BayesianEdgeEstimator().fit(r, prior='uninformative')
        plug = BayesianEdgeEstimator().fit(r, prior='uninformative', variance='plugin')
        assert abs(_ci_width(nig) / _ci_width(plug) - 1.0) < 0.01

    def test_nig_posterior_mean_is_the_conjugate_shrinkage(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0.4, 0.8, 30)
        p = PRIORS['toll']
        est = BayesianEdgeEstimator().fit(r, prior='toll')
        kappa0 = 1.0 / p.r_sigma ** 2                     # NIG_S0=1 → s0²/r_sigma²
        expect = (kappa0 * p.r_mu + 30 * r.mean()) / (kappa0 + 30)
        assert abs(est._r_mu_post - expect) < 1e-9

    def test_plugin_matches_the_known_variance_closed_form(self):
        rng = np.random.default_rng(4)
        r = rng.normal(0.2, 1.0, 50)
        p = PRIORS['skeptical']
        est = BayesianEdgeEstimator().fit(r, prior='skeptical', variance='plugin')
        s2 = np.var(r, ddof=1)
        post_prec = 1 / p.r_sigma ** 2 + 50 / s2
        expect = (p.r_mu / p.r_sigma ** 2 + 50 * r.mean() / s2) / post_prec
        assert abs(est._r_mu_post - expect) < 1e-9

    def test_effective_n_widens_the_interval_and_pulls_toward_the_prior(self):
        rng = np.random.default_rng(5)
        r = rng.normal(0.5, 1.0, 60)
        full = BayesianEdgeEstimator().fit(r, prior='toll')
        clustered = BayesianEdgeEstimator().fit(r, prior='toll', n_eff=20)
        assert _ci_width(clustered) > _ci_width(full) * 1.3
        assert clustered._r_mu_post < full._r_mu_post          # shrinks toward −0.25
        s = clustered.summary()
        assert s['n_eff'] == 20 and s['n_trades'] == 60
        # win-rate pseudo-counts scale too
        assert clustered._wr_alpha_post + clustered._wr_beta_post == pytest.approx(10 + 20)

    def test_n_eff_must_not_exceed_n(self):
        with pytest.raises(ValueError):
            BayesianEdgeEstimator().fit([0.1, -0.2, 0.3], n_eff=5)

    def test_toll_prior_asks_the_fee_paying_question(self):
        """A seat that loses exactly the toll on average: P(>0) ≈ 0, P(>−toll) ≈ ½."""
        rng = np.random.default_rng(6)
        z = rng.standard_normal(200)
        r = -0.25 + 0.05 * (z - z.mean()) / z.std()        # sample mean EXACTLY the toll
        est = BayesianEdgeEstimator().fit(r, prior='toll')
        assert est.p_above(0.0) < 0.01
        assert 0.45 < est.p_above(-0.25) < 0.55            # data and prior agree: a coin flip
        s = est.summary(threshold=-0.25)
        assert s['mean_r']['threshold'] == -0.25
        assert abs(s['mean_r']['p_above_threshold'] - est.p_above(-0.25)) < 1e-9
        assert s['prior']['type'] == 'toll' and s['prior']['r_mu'] == -0.25

    def test_skeptical_prior_is_a_hundred_pseudo_trades(self):
        """Documented, not hidden: Beta(50,50) dominates any book under ~100 fills."""
        p = PRIORS['skeptical']
        assert p.wr_alpha + p.wr_beta == 100
        est = BayesianEdgeEstimator().fit([1.0] * 16, prior='skeptical')
        assert est.summary()['win_rate']['posterior_mean'] < 0.6   # 16/16 wins barely move it

    def test_compare_and_kelly_labels(self):
        rng = np.random.default_rng(8)
        a = BayesianEdgeEstimator().fit(rng.normal(0.3, 1, 80), prior='toll')
        b = BayesianEdgeEstimator().fit(rng.normal(-0.3, 1, 80), prior='toll')
        c = a.compare(b)
        assert c['p_a_better_r'] > 0.95
        k = a.summary()['kelly_fraction']
        assert 'posterior_mean' in k and 'posterior_median' in k


# ═══════════════════════════════════════════════════════════════════════════════
#  Meta-strategy selector — the overlapping-label leak and the H× over-count
# ═══════════════════════════════════════════════════════════════════════════════
sklearn = pytest.importorskip("sklearn")
from backtrader_framework.optimization.meta_strategy_selector import MetaStrategySelector  # noqa: E402


def _synthetic_dataset(n=420, H=7, seed=0):
    """Two strategies with iid daily R (nothing to predict) and PERSISTENT
    features: a slow clock-like drift plus two random walks. Nothing here can
    forecast a return — so any accuracy above chance is a leak. A persistent
    feature is what makes the overlapping-label leak exploitable: the model's
    nearest neighbours in feature space are yesterday's rows, whose labels
    already contain most of today's forward window."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    labels = ["A", "B"]
    daily = pd.DataFrame({l: rng.normal(0.0, 1.0, n) for l in labels}, index=dates)
    feats = pd.DataFrame({
        "f0": np.arange(n) / n + 0.01 * rng.normal(0, 1, n),        # slow drift
        "f1": np.cumsum(rng.normal(0, 1, n)),
        "f2": np.cumsum(rng.normal(0, 1, n)),
    }, index=dates)
    fwd = pd.DataFrame({f"fwd_{l}": daily[l].rolling(H, min_periods=1).sum().shift(-H) for l in labels})
    dly = pd.DataFrame({f"daily_{l}": daily[l] for l in labels})

    def pick(row):
        v = row.values
        if np.all(np.isnan(v)):
            return np.nan
        if np.all(v <= 0):
            return "none"
        return labels[int(np.nanargmax(v))]

    lab = fwd.apply(pick, axis=1).rename("label")
    ds = feats.join(fwd).join(dly).join(lab).dropna(subset=["label"])
    sel = MetaStrategySelector(lookforward_days=H)
    sel.feature_names = list(feats.columns)
    sel.strategy_labels = labels
    return sel, ds


def _leaky_accuracy(ds, feature_names, H):
    """The pre-review walk-forward mechanism: train on rows [:i] (no embargo),
    score day i. Retrained daily with an unregularised forest so the mechanism
    is measured at full strength — with `retrain_every=20` the same leak is
    diluted (only the days right after a refit see the full overlap), which is
    why it hid: a smaller number, not a cleaner one."""
    from sklearn.ensemble import RandomForestClassifier
    X = ds[feature_names].values; y = ds["label"].values
    hits = tot = 0
    for i in range(60, len(ds)):
        model = RandomForestClassifier(n_estimators=40, max_depth=None, bootstrap=False,
                                       random_state=42).fit(X[:i], y[:i])
        hits += int(model.predict(X[i:i + 1])[0] == y[i]); tot += 1
    return hits / tot


class TestMetaStrategyFaithfulness:
    def test_embargoed_backtest_removes_the_overlapping_label_leak(self):
        sel, ds = _synthetic_dataset()
        leaky = _leaky_accuracy(ds, sel.feature_names, H=7)
        res = sel.backtest(ds, min_train_days=60, retrain_every=20)
        assert res["valid"] and res["embargo_rows"] == 7
        honest = res["prediction_accuracy"]
        # nothing is predictable here, so the honest number sits at chance …
        assert honest < 0.5, honest
        # … while the old walk-forward "knew" the answer from overlapping labels
        assert leaky > 0.55, leaky
        assert leaky - honest > 0.15, (leaky, honest)

    def test_each_realized_day_is_booked_exactly_once(self):
        sel, ds = _synthetic_dataset(seed=1)
        res = sel.backtest(ds, min_train_days=60, retrain_every=20)
        booked_dates = pd.to_datetime(res["dates"])
        assert booked_dates.is_unique                       # no day counted twice
        assert res["booking"].startswith("daily")
        # equal-weight total = the mean daily return summed over exactly those days
        expect = ds.loc[booked_dates, ["daily_A", "daily_B"]].mean(axis=1).sum()
        assert res["equal_weight"]["total_r"] == pytest.approx(expect, abs=1e-3)   # metrics are rounded to 4dp
        assert res["equal_weight"]["n_periods"] == len(booked_dates) == res["n_test_days"]
        # the oracle books the better strategy each day — never less than equal weight
        assert res["best_single"]["total_r"] >= res["equal_weight"]["total_r"]

    def test_decisions_are_spaced_by_the_hold_horizon(self):
        sel, ds = _synthetic_dataset(seed=2)
        res = sel.backtest(ds, min_train_days=60, retrain_every=20)
        d = pd.to_datetime([s["date"] for s in res["selection_timeline"]])
        gaps = np.diff(d.values).astype("timedelta64[D]").astype(int)
        assert set(gaps) == {7}
        assert res["hold_days"] == 7 and res["n_decisions"] == len(d)

    def test_train_split_leaves_an_embargo(self):
        sel, ds = _synthetic_dataset(seed=3)
        stats = sel.train(ds, test_size=0.2)
        split = int(len(ds) * 0.8)
        assert stats["n_train"] == split - 7 and stats["embargo_rows"] == 7
        assert stats["n_test"] == len(ds) - split

    def test_regime_heatmap_is_flagged_in_sample(self):
        sel, ds = _synthetic_dataset(seed=4)
        for rc in ("ranging", "trending_up", "trending_down", "volatile"):
            ds[f"regime_{rc}"] = 0.0
        ds["regime_ranging"] = 1.0
        sel.train(ds)
        hm = sel.get_regime_strategy_heatmap(ds)
        assert hm["valid"] and hm["in_sample"] is True


# ═══════════════════════════════════════════════════════════════════════════════
#  Tuner config + SHAP insight wording (no optuna / shap needed)
# ═══════════════════════════════════════════════════════════════════════════════
def test_tuner_defaults_are_honest():
    from backtrader_framework.optimization.bayesian_tuner import TunerConfig
    c = TunerConfig()
    assert c.scoring_metric == "balanced_accuracy"
    assert c.embargo == 0
    c.embargo = 7
    assert c.embargo == 7


def test_shap_insights_name_the_class_they_describe():
    from backtrader_framework.optimization.shap_analysis import SHAPAnalyzer
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    shap_vals = np.column_stack([0.5 * X[:, 0], 0.0 * X[:, 1]])   # feature 0 drives it
    out = SHAPAnalyzer._generate_insights(shap_vals, X, ["rv24", "noise"], ["A", "B"], target="A")
    assert out and all(i["target"] == "A" for i in out)
    assert "P(A)" in out[0]["insight"]
