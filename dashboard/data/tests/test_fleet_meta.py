"""Hermetic tests for WS5's allocation comparison.

The theme here is refusal: the module must decline to fit a classifier the data
cannot support, must never quietly clean a broken row out of a headline number,
and must compare arms PAIRED rather than as two independent means.
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

from data import fleet_meta as fmeta                          # noqa: E402


def _fills(n_days=40, families=("A", "B"), seed=0, extreme=None):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2026-01-01", periods=n_days, tz="UTC"):
        for f in families:
            for k in range(3):
                rows.append({"day": d, "family": f, "cluster": f"{f}-{d.date()}-{k}",
                             "r": rng.normal(-0.1, 0.8), "book": f"{f}-book",
                             "fill_ts": d + pd.Timedelta(hours=k)})
    df = pd.DataFrame(rows)
    if extreme is not None:
        df.loc[0, "r"] = extreme
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Refusing the classifier
# ══════════════════════════════════════════════════════════════════════════════
def test_feasibility_refuses_a_sample_that_cannot_support_the_classes():
    daily = fmeta.daily_family_panel(fills=_fills(n_days=70, families=tuple("ABCDEFGHIJ")))
    fz = fmeta.feasibility(daily, min_train_days=60)
    assert fz["classes"] == 10
    assert fz["evaluable_days"] == 10
    assert fz["classifier_supportable"] is False


def test_feasibility_accepts_a_sample_that_can():
    daily = fmeta.daily_family_panel(fills=_fills(n_days=400, families=("A", "B")))
    fz = fmeta.feasibility(daily, min_train_days=60)
    assert fz["classifier_supportable"] is True


def test_feasibility_counts_only_days_with_a_choice_to_make():
    """A day with one family active offers the allocator nothing to choose."""
    solo = _fills(n_days=30, families=("A",))
    daily = fmeta.daily_family_panel(fills=solo)
    assert fmeta.feasibility(daily)["days_multi_family"] == 0


def test_the_module_fits_no_classifier():
    src = Path(fmeta.__file__).read_text()
    for banned in ("RandomForest", "fit(", "predict("):
        assert banned not in src, (
            "WS5 refused the classifier on 2.2 evaluated days per class")


# ══════════════════════════════════════════════════════════════════════════════
#  Broken rows are reported, never silently cleaned
# ══════════════════════════════════════════════════════════════════════════════
def test_extreme_rows_are_reported():
    f = _fills(extreme=-56.9)
    ext = fmeta.extreme_r_report(f, max_abs_r=5.0)
    assert len(ext) == 1 and ext["r"].iloc[0] == pytest.approx(-56.9)


def test_the_panel_keeps_broken_rows_by_default():
    """The honest default is the tape as recorded, with the sensitivity stated —
    not a quiet deletion that makes the headline look better."""
    f = _fills(extreme=-56.9)
    raw = fmeta.daily_family_panel(fills=f)
    assert raw["r"].min() < -5


def test_the_panel_can_exclude_on_request_and_it_changes_the_level():
    f = _fills(extreme=-56.9)
    raw = fmeta.daily_family_panel(fills=f)
    clean = fmeta.daily_family_panel(fills=f, max_abs_r=5.0)
    assert clean["r"].min() > -5
    assert abs(raw["r"].mean() - clean["r"].mean()) > 0.05


def test_heatmap_carries_a_median_beside_the_mean():
    """`Liquidity Raid` reads a mean of -2.668 in quiet against a median of
    -0.004 because ONE row carries -56.9R. A means-only heatmap invites that
    misreading."""
    import inspect
    src = inspect.getsource(fmeta.regime_family_heatmap)
    assert "median_r" in src


# ══════════════════════════════════════════════════════════════════════════════
#  The comparison itself
# ══════════════════════════════════════════════════════════════════════════════
def _cmp(n_days=60, seed=1):
    daily = fmeta.daily_family_panel(fills=_fills(n_days=n_days, seed=seed))
    return pd.DataFrame({
        "day": pd.date_range("2026-01-01", periods=n_days, tz="UTC"),
        "equal_weight": np.random.default_rng(seed).normal(-0.1, 0.3, n_days),
        "conductor": np.random.default_rng(seed + 1).normal(-0.1, 0.3, n_days),
        "hmm_calm": np.random.default_rng(seed + 2).normal(-0.1, 0.3, n_days),
        "hindsight_best": np.random.default_rng(seed + 3).normal(0.1, 0.3, n_days),
    })


def test_summary_reports_every_arm_against_the_conductor():
    out = fmeta.summarise_allocations(_cmp())
    assert set(out["arm"]) == {"equal_weight", "conductor", "hmm_calm", "hindsight_best"}
    assert "vs_conductor" in out.columns
    assert out.loc[out["arm"] == "conductor", "vs_conductor"].iloc[0] == 0.0


def test_paired_difference_is_tighter_than_two_independent_means():
    """Both arms traded the same days. The paired difference is what a switch
    would actually have bought, and its interval is far tighter for the data."""
    c = _cmp(n_days=200)
    c["equal_weight"] = c["conductor"] + 0.02       # a small, constant edge
    unpaired = fmeta.summarise_allocations(c)
    paired = fmeta.paired_vs_conductor(c, "equal_weight")
    unpaired_width = float(
        unpaired.loc[unpaired.arm == "equal_weight", "ci_hi"].iloc[0]
        - unpaired.loc[unpaired.arm == "equal_weight", "ci_lo"].iloc[0])
    assert (paired["ci_hi"] - paired["ci_lo"]) < unpaired_width
    assert paired["mean_diff_r_per_day"] == pytest.approx(0.02, abs=1e-9)


def test_paired_refuses_too_few_days():
    c = _cmp(n_days=5)
    assert fmeta.paired_vs_conductor(c, "equal_weight")["status"] == "too few days"


def test_bootstrap_resamples_days_whole():
    import inspect
    src = inspect.getsource(fmeta.summarise_allocations)
    assert "rng.integers(0, n" in src, "the day is the unit of resampling"


# ══════════════════════════════════════════════════════════════════════════════
#  Lane mapping fidelity
# ══════════════════════════════════════════════════════════════════════════════
def test_lane_map_matches_the_deployed_prereg():
    assert fmeta.LANES["Desk"] == "patient" and fmeta.LANES["Retest"] == "patient"
    assert fmeta.LANES["Knife"] == "knife"
    assert fmeta.PATIENT_RV_FRAC == 0.85


def test_the_extended_lane_assignment_is_flagged():
    """The v1 prereg names lr-signal-shadow (Tier 3), not the Tier-1/2 LR seats.
    Mapping the family to `raid` is an extension and must read as one."""
    assert fmeta.LANES["Liquidity Raid"].endswith("*")


def test_ungated_families_are_named_not_assumed():
    assert set(fmeta.UNGATED) and not (set(fmeta.UNGATED) & set(fmeta.LANES))


def test_module_never_touches_the_vps():
    import ast
    tree = ast.parse(Path(fmeta.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"subprocess", "paramiko", "socket", "requests", "urllib"}
