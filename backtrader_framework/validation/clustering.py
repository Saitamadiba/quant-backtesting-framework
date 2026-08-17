"""Clustered inference: what to do when your trades are not independent.

Simultaneous same-direction positions are one bet wearing several name tags.
Counting them as independent observations inflates t-statistics without adding
information — the most common way a book looks more significant than it is.

See ``docs/RESEARCH_METHOD.md`` §5.2-5.3.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "clustered_tstat",
    "effective_sample_size",
    "weighting_disagreement",
]


def clustered_tstat(
    values: Sequence[float],
    clusters: Sequence,
    small_sample_correction: bool = True,
) -> dict:
    """Cluster-robust t-statistic for a mean (CR0/CR1 sandwich estimator).

    Standard use here: cluster by calendar day, because the fleet's positions
    open in bursts driven by shared session structure. Two trades in the same
    burst share most of their information.

    The estimator is
    ``Var(mean) = sum_g (sum_{i in g} e_i)^2 / n^2``
    where ``e_i = x_i - mean``, optionally scaled by ``G/(G-1)`` (CR1). Degrees
    of freedom are ``G-1``, not ``n-1`` — the number of clusters is the real
    sample size for inference purposes.

    Returns
    -------
    dict with ``mean``, ``t_clustered``, ``t_naive``, ``se_clustered``,
    ``se_naive``, ``n``, ``n_clusters``, ``df``, and ``inflation`` (the factor by
    which the naive t overstates the clustered one).
    """
    x = np.asarray(values, dtype=float)
    g = np.asarray(clusters)
    mask = ~np.isnan(x)
    x, g = x[mask], g[mask]
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 observations")

    mean = float(x.mean())
    e = x - mean

    uniq = pd.unique(g)
    G = len(uniq)
    if G < 2:
        raise ValueError("need at least 2 clusters for clustered inference")

    ssum = 0.0
    for c in uniq:
        ssum += float(e[g == c].sum()) ** 2

    var = ssum / (n ** 2)
    if small_sample_correction:
        var *= G / (G - 1)
    se_cl = float(np.sqrt(var))

    sd = x.std(ddof=1)
    se_naive = float(sd / np.sqrt(n)) if sd > 0 else float("nan")

    t_cl = float(mean / se_cl) if se_cl > 0 else float("nan")
    t_nv = float(mean / se_naive) if se_naive and se_naive > 0 else float("nan")

    return {
        "mean": mean,
        "t_clustered": t_cl,
        "t_naive": t_nv,
        "se_clustered": se_cl,
        "se_naive": se_naive,
        "n": int(n),
        "n_clusters": int(G),
        "df": int(G - 1),
        "inflation": float(t_nv / t_cl) if t_cl not in (0.0,) and not np.isnan(t_cl) else float("nan"),
    }


def effective_sample_size(
    values: Sequence[float],
    clusters: Sequence,
) -> dict:
    """Effective n after the design-effect haircut for within-cluster correlation.

    ``n_eff = n / (1 + (m_bar - 1) * rho)`` where ``m_bar`` is mean cluster size
    and ``rho`` is the intra-cluster correlation estimated one-way-ANOVA style.
    At ``rho -> 1`` (trades within a burst are effectively one bet) this
    collapses to the cluster count; at ``rho -> 0`` it returns ``n``.

    Reported alongside every clustered book in this project, because "n=646"
    and "n_eff=41" support very different sentences.

    Returns
    -------
    dict with ``n``, ``n_clusters``, ``mean_cluster_size``, ``icc``, ``n_eff``,
    and ``haircut`` (``n_eff / n``).
    """
    x = np.asarray(values, dtype=float)
    g = np.asarray(clusters)
    mask = ~np.isnan(x)
    x, g = x[mask], g[mask]
    n = x.size
    uniq = pd.unique(g)
    G = len(uniq)
    if n < 2 or G < 1:
        raise ValueError("need at least 2 observations in >=1 cluster")

    sizes = np.array([np.sum(g == c) for c in uniq], dtype=float)
    m_bar = float(sizes.mean())

    grand = x.mean()
    # one-way ANOVA decomposition
    ss_between = float(sum(np.sum(g == c) * (x[g == c].mean() - grand) ** 2 for c in uniq))
    ss_within = float(sum(np.sum((x[g == c] - x[g == c].mean()) ** 2) for c in uniq))

    df_b, df_w = G - 1, n - G
    ms_b = ss_between / df_b if df_b > 0 else 0.0
    ms_w = ss_within / df_w if df_w > 0 else 0.0

    if ms_b + (m_bar - 1) * ms_w <= 0:
        icc = 0.0
    else:
        icc = (ms_b - ms_w) / (ms_b + (m_bar - 1) * ms_w)
    icc = float(min(max(icc, 0.0), 1.0))   # clamp: negative ICC is noise

    design_effect = 1.0 + (m_bar - 1.0) * icc
    n_eff = float(n / design_effect) if design_effect > 0 else float(n)

    return {
        "n": int(n),
        "n_clusters": int(G),
        "mean_cluster_size": m_bar,
        "icc": icc,
        "n_eff": n_eff,
        "haircut": float(n_eff / n),
    }


def weighting_disagreement(
    values: Sequence[float],
    clusters: Sequence,
) -> dict:
    """Compare observation-weighted and cluster-weighted means.

    Observation weighting lets high-activity days dominate; cluster weighting
    treats a quiet day and a frantic one alike. Both are defensible — the trap
    is computing one and reasoning about the other. When they **disagree in
    sign**, that fact deserves a paragraph rather than a silent choice
    (``RESEARCH_METHOD.md`` §5.3).

    Returns
    -------
    dict with both means and ``signs_disagree``.
    """
    x = np.asarray(values, dtype=float)
    g = np.asarray(clusters)
    mask = ~np.isnan(x)
    x, g = x[mask], g[mask]

    uniq = pd.unique(g)
    obs_mean = float(x.mean())
    clu_mean = float(np.mean([x[g == c].mean() for c in uniq]))

    return {
        "obs_weighted_mean": obs_mean,
        "cluster_weighted_mean": clu_mean,
        "signs_disagree": bool(np.sign(obs_mean) != np.sign(clu_mean)),
        "n_clusters": int(len(uniq)),
    }
