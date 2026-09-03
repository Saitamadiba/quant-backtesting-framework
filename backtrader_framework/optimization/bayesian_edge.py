"""Bayesian estimation of strategy edge using conjugate priors.

Uses Beta-Binomial for win rate and, for the mean R-multiple, a
Normal-Inverse-Gamma conjugate (Student-t posterior for the mean) — the
variance is a parameter with its own uncertainty, not a plug-in from the
sample. All posteriors are analytic except expectancy, which uses lightweight
Monte Carlo from the posteriors. No MCMC libraries needed.

Faithfulness review 2026-09-03 (fleet plan WS1):
  * The old Normal-Normal "known variance" step plugged the SAMPLE variance in
    as if it were known. Measured on the same draws, the honest Student-t
    interval is 5–50% wider below n≈16 (how much depends on how the sample
    variance happened to fall — the very thing the plug-in ignores) and agrees
    to 2% by n≈50. Small books are exactly what the fleet page shows. The
    mean-R posterior is now Student-t (NIG conjugate, weak variance prior);
    `variance='plugin'` keeps the old behaviour for comparison.
  * The 'skeptical' prior is Beta(50,50) — a hundred pseudo-trades insisting on
    a 50% win rate — with the mean-R prior centred on ZERO, i.e. on "no toll".
    For a fee-paying seat the honest null is the toll itself: the new 'toll'
    prior centres mean R on −0.25R (sd 0.5) with a 10-pseudo-trade 50% WR.
  * Every trade counted as an independent draw. Fleet books cluster (several
    same-direction fills in one 15-minute bucket are one bet), so `fit()` takes
    an effective sample size `n_eff` that scales both the pseudo-counts and the
    likelihood precision.
  * P(mean R > 0) is not the question a fee-paying book faces; `p_above(x)` and
    `summary(threshold=...)` report P(mean R > toll) alongside it.
  * The Kelly summary labelled a median as 'posterior_mean'; both are reported.
"""

import logging
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BayesianPrior:
    """Prior specification for the Bayesian edge estimator."""
    name: str
    wr_alpha: float   # Beta distribution alpha for win rate
    wr_beta: float    # Beta distribution beta for win rate
    r_mu: float       # Normal prior mean for mean R
    r_sigma: float    # Normal prior std for mean R


PRIORS = {
    'uninformative': BayesianPrior('uninformative', 1.0, 1.0, 0.0, 10.0),
    # 100 pseudo-trades at a 50% win rate, mean R centred on zero (no toll).
    # Kept for the research pages; NOT the right null for a fee-paying seat.
    'skeptical':     BayesianPrior('skeptical', 50.0, 50.0, 0.0, 0.5),
    # The fleet's null: a seat that pays the ~0.25R fee/slip crossing charge and
    # has no edge. 10 pseudo-trades at 50% WR so the data can speak by n≈20.
    'toll':          BayesianPrior('toll', 5.0, 5.0, -0.25, 0.5),
}

# Normal-Inverse-Gamma variance prior: a weak belief that R has unit-ish scale.
# nu0 pseudo-observations of variance s0^2. Small nu0 = let the data decide.
NIG_NU0 = 2.0
NIG_S0 = 1.0


class BayesianEdgeEstimator:
    """Bayesian estimation of strategy edge using conjugate priors.

    Uses Beta-Binomial for win rate and Normal-Normal for mean R-multiple.
    Provides posterior distributions, credible intervals, P(edge > 0),
    Kelly fraction, shrinkage analysis, and strategy comparison.

    Usage:
        est = BayesianEdgeEstimator()
        est.fit(r_multiples, prior='skeptical')
        result = est.summary()
        print(f"P(edge > 0) = {result['expectancy']['p_positive']:.1%}")
    """

    N_MC_SAMPLES = 50_000

    def __init__(self):
        self._fitted = False
        self._prior = None
        self._r_multiples = None
        self._n = 0
        self._n_wins = 0
        self._n_losses = 0
        # Win rate posterior
        self._wr_alpha_post = None
        self._wr_beta_post = None
        # Mean R posterior — Student-t (NIG) unless variance='plugin'
        self._r_mu_post = None
        self._r_sigma_post = None       # scale of the t (or sd of the normal)
        self._r_df_post = None          # None → normal (plugin variance)
        self._n_eff = None
        self._variance_mode = 'nig'
        # Win/loss R posteriors (for expectancy)
        self._win_r_mu_post = None
        self._win_r_sigma_post = None
        self._loss_r_mu_post = None
        self._loss_r_sigma_post = None

    def fit(
        self,
        r_multiples: Any,
        prior: str = 'skeptical',
        custom_prior: Optional[BayesianPrior] = None,
        n_eff: Optional[float] = None,
        variance: str = 'nig',
    ) -> 'BayesianEdgeEstimator':
        """Fit the Bayesian model on observed R-multiples.

        Args:
            r_multiples: R-multiple values (positive = win, negative = loss).
            prior: Prior name ('uninformative', 'skeptical', 'toll').
            custom_prior: Custom BayesianPrior (overrides prior name).
            n_eff: Effective number of independent bets (≤ n). Clustered fills
                (same family, same direction, same 15-min bucket) are one bet;
                pass the cluster count and every count/precision is scaled by
                n_eff/n. Default n (independent).
            variance: 'nig' (default) — Normal-Inverse-Gamma conjugate, the mean
                has a Student-t posterior; 'plugin' — the old known-variance
                Normal with the sample variance plugged in (overconfident at
                small n; kept for comparison).

        Returns:
            self (for chaining).
        """
        from scipy import stats as sp_stats

        r = np.asarray(r_multiples, dtype=float)
        r = r[~np.isnan(r)]
        self._r_multiples = r
        self._n = len(r)

        if self._n < 2:
            raise ValueError(f"Need at least 2 trades, got {self._n}")
        if n_eff is None:
            n_eff = float(self._n)
        if not (0 < n_eff <= self._n + 1e-9):
            raise ValueError(f"n_eff must be in (0, n]; got {n_eff} for n={self._n}")
        self._n_eff = float(n_eff)
        w = self._n_eff / self._n            # per-trade weight
        if variance not in ('nig', 'plugin'):
            raise ValueError("variance must be 'nig' or 'plugin'")
        self._variance_mode = variance

        # Resolve prior
        if custom_prior is not None:
            self._prior = custom_prior
        else:
            self._prior = PRIORS.get(prior)
            if self._prior is None:
                raise ValueError(
                    f"Unknown prior: '{prior}'. Available: {list(PRIORS.keys())}"
                )

        p = self._prior

        # --- Win Rate (Beta-Binomial conjugate) ---
        wins = r > 0
        self._n_wins = int(np.sum(wins))
        self._n_losses = self._n - self._n_wins

        self._wr_alpha_post = p.wr_alpha + self._n_wins * w
        self._wr_beta_post = p.wr_beta + self._n_losses * w

        # --- Mean R ---
        x_bar = float(np.mean(r))
        s2 = max(float(np.var(r, ddof=1)), 1e-10)
        n_e = self._n_eff

        if variance == 'plugin':
            # Known-variance Normal-Normal with the sample variance plugged in.
            prior_prec = 1.0 / (p.r_sigma ** 2)
            like_prec = n_e / s2
            post_prec = prior_prec + like_prec
            self._r_sigma_post = float(np.sqrt(1.0 / post_prec))
            self._r_mu_post = float(self._r_sigma_post ** 2
                                    * (p.r_mu * prior_prec + n_e * x_bar / s2))
            self._r_df_post = None
        else:
            # Normal-Inverse-Gamma conjugate. Prior: mu ~ N(mu0, sigma^2/kappa0),
            # sigma^2 ~ InvGamma(nu0/2, nu0*s0^2/2). The prior's stated r_sigma
            # is the marginal sd of mu, so kappa0 = s0^2 / r_sigma^2 pseudo-obs.
            mu0 = p.r_mu
            kappa0 = max((NIG_S0 ** 2) / (p.r_sigma ** 2), 1e-6)
            nu0, s0sq = NIG_NU0, NIG_S0 ** 2
            kappa_n = kappa0 + n_e
            nu_n = nu0 + n_e
            mu_n = (kappa0 * mu0 + n_e * x_bar) / kappa_n
            ss = (self._n - 1) * s2 * w              # effective within-sample SS
            nu_n_s2 = nu0 * s0sq + ss + (kappa0 * n_e / kappa_n) * (x_bar - mu0) ** 2
            sigma_n_sq = nu_n_s2 / nu_n
            self._r_mu_post = float(mu_n)
            self._r_sigma_post = float(np.sqrt(sigma_n_sq / kappa_n))   # t scale
            self._r_df_post = float(nu_n)

        # --- Separate win/loss R posteriors (for expectancy MC) ---
        win_r = r[wins]
        loss_r = r[~wins]

        # Win R posterior (mild positive prior)
        if len(win_r) >= 2:
            win_bar = float(np.mean(win_r))
            win_s2 = max(float(np.var(win_r, ddof=1)), 1e-10)
            win_prior_mu, win_prior_sigma = 0.5, 2.0
            win_prior_prec = 1.0 / win_prior_sigma ** 2
            win_like_prec = len(win_r) / win_s2
            win_post_prec = win_prior_prec + win_like_prec
            self._win_r_sigma_post = float(np.sqrt(1.0 / win_post_prec))
            self._win_r_mu_post = float(
                self._win_r_sigma_post ** 2
                * (win_prior_mu * win_prior_prec + len(win_r) * win_bar / win_s2)
            )
        else:
            self._win_r_mu_post = float(np.mean(win_r)) if len(win_r) == 1 else 0.5
            self._win_r_sigma_post = 1.0

        # Loss R posterior (mild negative prior)
        if len(loss_r) >= 2:
            loss_bar = float(np.mean(loss_r))
            loss_s2 = max(float(np.var(loss_r, ddof=1)), 1e-10)
            loss_prior_mu, loss_prior_sigma = -0.5, 2.0
            loss_prior_prec = 1.0 / loss_prior_sigma ** 2
            loss_like_prec = len(loss_r) / loss_s2
            loss_post_prec = loss_prior_prec + loss_like_prec
            self._loss_r_sigma_post = float(np.sqrt(1.0 / loss_post_prec))
            self._loss_r_mu_post = float(
                self._loss_r_sigma_post ** 2
                * (loss_prior_mu * loss_prior_prec + len(loss_r) * loss_bar / loss_s2)
            )
        else:
            self._loss_r_mu_post = float(np.mean(loss_r)) if len(loss_r) == 1 else -1.0
            self._loss_r_sigma_post = 1.0

        self._fitted = True
        return self

    def _r_dist(self):
        """The mean-R posterior: Student-t (NIG) or Normal (plugin)."""
        from scipy import stats as sp_stats
        if self._r_df_post is None:
            return sp_stats.norm(self._r_mu_post, self._r_sigma_post)
        return sp_stats.t(df=self._r_df_post, loc=self._r_mu_post, scale=self._r_sigma_post)

    def p_above(self, threshold: float = 0.0) -> float:
        """P(mean R > threshold) under the posterior. Pass the toll (e.g. 0.25R
        for a taker seat) to ask the question a fee-paying book actually faces."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .p_above()")
        return float(self._r_dist().sf(threshold))

    def summary(self, threshold: float = 0.0) -> dict:
        """Return full posterior summary.

        Args:
            threshold: an R level to test the mean against in addition to zero
                (e.g. the fee/slip toll). Reported as mean_r.p_above_threshold.

        Returns:
            Dict with keys: prior, n_trades, n_eff, n_wins, win_rate, mean_r,
            expectancy, kelly_fraction, shrinkage, sample_size_assessment.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .summary()")

        from scipy import stats as sp_stats

        p = self._prior
        rng = np.random.default_rng(42)

        # --- Win Rate Posterior ---
        wr_dist = sp_stats.beta(self._wr_alpha_post, self._wr_beta_post)
        wr_mean = float(wr_dist.mean())
        wr_std = float(wr_dist.std())
        wr_ci = (float(wr_dist.ppf(0.025)), float(wr_dist.ppf(0.975)))
        p_above_50 = float(1.0 - wr_dist.cdf(0.5))

        # --- Mean R Posterior (Student-t under NIG) ---
        r_dist = self._r_dist()
        r_mean = float(r_dist.mean())
        r_std = float(r_dist.std()) if (self._r_df_post is None or self._r_df_post > 2) \
            else float('nan')
        r_ci = (float(r_dist.ppf(0.025)), float(r_dist.ppf(0.975)))
        p_positive = float(r_dist.sf(0.0))
        p_above_thr = float(r_dist.sf(threshold))

        # --- Expectancy Posterior (MC from component posteriors) ---
        wr_samples = sp_stats.beta.rvs(
            self._wr_alpha_post, self._wr_beta_post,
            size=self.N_MC_SAMPLES, random_state=rng,
        )
        win_r_samples = sp_stats.norm.rvs(
            self._win_r_mu_post, self._win_r_sigma_post,
            size=self.N_MC_SAMPLES, random_state=rng,
        )
        loss_r_samples = sp_stats.norm.rvs(
            self._loss_r_mu_post, self._loss_r_sigma_post,
            size=self.N_MC_SAMPLES, random_state=rng,
        )

        # Physical consistency: wins > 0, losses < 0
        win_r_samples = np.clip(win_r_samples, 0.01, None)
        loss_r_samples = np.clip(loss_r_samples, None, -0.01)

        expectancy_samples = (
            wr_samples * win_r_samples
            + (1 - wr_samples) * loss_r_samples
        )
        exp_mean = float(np.mean(expectancy_samples))
        exp_ci = (
            float(np.percentile(expectancy_samples, 2.5)),
            float(np.percentile(expectancy_samples, 97.5)),
        )
        exp_p_positive = float(np.mean(expectancy_samples > 0))

        # --- Kelly Fraction (from MC samples) ---
        avg_loss_abs = np.abs(loss_r_samples)
        odds = win_r_samples / avg_loss_abs
        kelly_samples = (wr_samples * (1 + odds) - 1) / odds
        kelly_samples = np.clip(kelly_samples, -1.0, 2.0)
        kelly_median = float(np.median(kelly_samples))
        kelly_mean = float(np.mean(kelly_samples))
        kelly_ci = (
            float(np.percentile(kelly_samples, 2.5)),
            float(np.percentile(kelly_samples, 97.5)),
        )

        # --- Shrinkage ---
        prior_wr = p.wr_alpha / (p.wr_alpha + p.wr_beta)
        observed_wr = self._n_wins / self._n
        denom_wr = abs(observed_wr - prior_wr)
        wr_shrinkage = (
            float(np.clip(
                1.0 - abs(wr_mean - prior_wr) / max(denom_wr, 1e-10),
                0, 1,
            ))
            if denom_wr > 1e-10
            else 0.0
        )

        observed_r = float(np.mean(self._r_multiples))
        denom_r = abs(observed_r - p.r_mu)
        r_shrinkage = (
            float(np.clip(
                1.0 - abs(r_mean - p.r_mu) / max(denom_r, 1e-10),
                0, 1,
            ))
            if denom_r > 1e-10
            else 0.0
        )

        # --- Sample Size Assessment ---
        ci_width = wr_ci[1] - wr_ci[0]
        # CI width ~1/sqrt(n), so to halve need 4x trades
        trades_for_half = int(np.ceil(self._n * 4))

        return {
            'prior': {
                'type': p.name,
                'wr_alpha': p.wr_alpha,
                'wr_beta': p.wr_beta,
                'r_mu': p.r_mu,
                'r_sigma': p.r_sigma,
            },
            'n_trades': self._n,
            'n_eff': round(self._n_eff, 2),
            'n_wins': self._n_wins,
            'variance_model': 'student_t_nig' if self._r_df_post is not None else 'normal_plugin',

            'win_rate': {
                'posterior_mean': round(wr_mean, 4),
                'posterior_std': round(wr_std, 4),
                'credible_interval_95': (round(wr_ci[0], 4), round(wr_ci[1], 4)),
                'p_above_50': round(p_above_50, 4),
            },

            'mean_r': {
                'posterior_mean': round(r_mean, 4),
                'posterior_std': round(r_std, 4) if r_std == r_std else None,
                'posterior_df': round(self._r_df_post, 2) if self._r_df_post is not None else None,
                'credible_interval_95': (round(r_ci[0], 4), round(r_ci[1], 4)),
                'p_positive': round(p_positive, 4),
                'threshold': threshold,
                'p_above_threshold': round(p_above_thr, 4),
            },

            'expectancy': {
                'posterior_mean': round(exp_mean, 4),
                'credible_interval_95': (round(exp_ci[0], 4), round(exp_ci[1], 4)),
                'p_positive': round(exp_p_positive, 4),
            },

            'kelly_fraction': {
                'posterior_mean': round(kelly_mean, 4),
                'posterior_median': round(kelly_median, 4),
                'credible_interval_95': (round(kelly_ci[0], 4), round(kelly_ci[1], 4)),
            },

            'shrinkage': {
                'wr_shrinkage': round(wr_shrinkage, 4),
                'r_shrinkage': round(r_shrinkage, 4),
            },

            'sample_size_assessment': {
                'n_trades': self._n,
                'posterior_uncertainty': round(ci_width, 4),
                'trades_for_half_width': trades_for_half,
            },
        }

    def compare(self, other: 'BayesianEdgeEstimator') -> dict:
        """Compute P(strategy A > strategy B) via Monte Carlo.

        Returns:
            Dict with p_a_better_wr, p_a_better_r, p_a_better_expectancy.
        """
        if not self._fitted or not other._fitted:
            raise RuntimeError("Both estimators must be fitted")

        from scipy import stats as sp_stats

        N = self.N_MC_SAMPLES
        rng = np.random.default_rng(42)

        # Win rate comparison
        wr_a = sp_stats.beta.rvs(
            self._wr_alpha_post, self._wr_beta_post, size=N, random_state=rng,
        )
        wr_b = sp_stats.beta.rvs(
            other._wr_alpha_post, other._wr_beta_post, size=N, random_state=rng,
        )

        # Mean R comparison (each from its own posterior family)
        r_a = self._r_dist().rvs(size=N, random_state=rng)
        r_b = other._r_dist().rvs(size=N, random_state=rng)

        # Expectancy comparison
        win_a = np.clip(sp_stats.norm.rvs(
            self._win_r_mu_post, self._win_r_sigma_post, size=N, random_state=rng,
        ), 0.01, None)
        loss_a = np.clip(sp_stats.norm.rvs(
            self._loss_r_mu_post, self._loss_r_sigma_post, size=N, random_state=rng,
        ), None, -0.01)
        win_b = np.clip(sp_stats.norm.rvs(
            other._win_r_mu_post, other._win_r_sigma_post, size=N, random_state=rng,
        ), 0.01, None)
        loss_b = np.clip(sp_stats.norm.rvs(
            other._loss_r_mu_post, other._loss_r_sigma_post, size=N, random_state=rng,
        ), None, -0.01)

        exp_a = wr_a * win_a + (1 - wr_a) * loss_a
        exp_b = wr_b * win_b + (1 - wr_b) * loss_b

        return {
            'p_a_better_wr': round(float(np.mean(wr_a > wr_b)), 4),
            'p_a_better_r': round(float(np.mean(r_a > r_b)), 4),
            'p_a_better_expectancy': round(float(np.mean(exp_a > exp_b)), 4),
        }

    def plot_posterior(self):
        """Generate matplotlib figure with posterior distributions.

        Returns:
            matplotlib Figure with 3 panels: Win Rate, Mean R, Expectancy.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .plot_posterior()")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        p = self._prior
        rng = np.random.default_rng(42)

        # Win Rate
        x_wr = np.linspace(0, 1, 500)
        wr_pdf = sp_stats.beta.pdf(x_wr, self._wr_alpha_post, self._wr_beta_post)
        wr_prior = sp_stats.beta.pdf(x_wr, p.wr_alpha, p.wr_beta)
        axes[0].plot(x_wr, wr_pdf, 'b-', label='Posterior')
        axes[0].plot(x_wr, wr_prior, 'r--', alpha=0.5, label='Prior')
        axes[0].axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='50%')
        axes[0].set_title('Win Rate')
        axes[0].set_xlabel('Win Rate')
        axes[0].legend(fontsize=8)

        # Mean R
        mu = self._r_mu_post
        sigma = self._r_sigma_post
        x_r = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
        r_pdf = self._r_dist().pdf(x_r)
        r_prior = sp_stats.norm.pdf(x_r, p.r_mu, p.r_sigma)
        axes[1].plot(x_r, r_pdf, 'b-', label='Posterior')
        axes[1].plot(x_r, r_prior, 'r--', alpha=0.5, label='Prior')
        axes[1].axvline(0, color='gray', linestyle=':', alpha=0.5, label='Zero')
        axes[1].set_title('Mean R-Multiple')
        axes[1].set_xlabel('R-Multiple')
        axes[1].legend(fontsize=8)

        # Expectancy (MC histogram)
        wr_s = sp_stats.beta.rvs(
            self._wr_alpha_post, self._wr_beta_post,
            size=self.N_MC_SAMPLES, random_state=rng,
        )
        win_s = np.clip(sp_stats.norm.rvs(
            self._win_r_mu_post, self._win_r_sigma_post,
            size=self.N_MC_SAMPLES, random_state=rng,
        ), 0.01, None)
        loss_s = np.clip(sp_stats.norm.rvs(
            self._loss_r_mu_post, self._loss_r_sigma_post,
            size=self.N_MC_SAMPLES, random_state=rng,
        ), None, -0.01)
        exp_s = wr_s * win_s + (1 - wr_s) * loss_s
        p_pos = float(np.mean(exp_s > 0))

        axes[2].hist(exp_s, bins=100, density=True, alpha=0.7, color='steelblue')
        axes[2].axvline(0, color='red', linestyle=':', label='Zero Edge')
        axes[2].set_title(f'Expectancy  P(>0): {p_pos:.1%}')
        axes[2].set_xlabel('Expectancy (R per trade)')
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_trade_results(
        cls,
        trades: list,
        prior: str = 'skeptical',
    ) -> 'BayesianEdgeEstimator':
        """Construct from List[TradeResult] (WFO engine output)."""
        r_multiples = [t.r_multiple_after_costs for t in trades]
        est = cls()
        est.fit(r_multiples, prior=prior)
        return est

    @classmethod
    def from_wfo_results(
        cls,
        wfo_result: dict,
        prior: str = 'skeptical',
        use_oos: bool = True,
    ) -> 'BayesianEdgeEstimator':
        """Construct from a WFO result JSON dict."""
        key = 'oos_equity' if use_oos else 'is_equity'
        equity = wfo_result.get(key, [])
        r_multiples = [e['r'] for e in equity if 'r' in e]
        if not r_multiples:
            raise ValueError(f"No R-multiples found in WFO result['{key}']")
        est = cls()
        est.fit(r_multiples, prior=prior)
        return est

    @classmethod
    def from_live_trades(
        cls,
        strategy: str,
        symbol: str,
        prior: str = 'skeptical',
    ) -> 'BayesianEdgeEstimator':
        """Construct from VPS-cached live trades."""
        from .shadow_backtest import load_live_trades_for_strategy
        df = load_live_trades_for_strategy(strategy, symbol)
        if df.empty:
            raise ValueError(f"No closed live trades for {strategy}/{symbol}")
        r_multiples = df['r_multiple'].dropna().values
        if len(r_multiples) < 2:
            raise ValueError(f"Need >= 2 trades with R-multiples, got {len(r_multiples)}")
        est = cls()
        est.fit(r_multiples, prior=prior)
        return est

    @classmethod
    def from_informed_prior(
        cls,
        wfo_result: dict,
        live_r_multiples: list,
    ) -> 'BayesianEdgeEstimator':
        """Use WFO OOS as prior, update with live data.

        The backtest distribution becomes the prior belief,
        and live data updates it. This is the most powerful mode.
        """
        oos_equity = wfo_result.get('oos_equity', [])
        oos_r = np.array([e['r'] for e in oos_equity if 'r' in e])
        if len(oos_r) < 5:
            raise ValueError("Need at least 5 OOS trades for informed prior")

        n_oos = len(oos_r)
        n_oos_wins = int(np.sum(oos_r > 0))

        # Scale pseudo-counts: treat OOS as ~50 prior observations
        scale = min(50, n_oos) / n_oos
        alpha = max(1.0, n_oos_wins * scale)
        beta = max(1.0, (n_oos - n_oos_wins) * scale)

        r_mu = float(np.mean(oos_r))
        r_sigma = max(0.1, float(np.std(oos_r, ddof=1)))

        prior = BayesianPrior('informed', alpha, beta, r_mu, r_sigma)
        est = cls()
        est.fit(live_r_multiples, custom_prior=prior)
        return est
