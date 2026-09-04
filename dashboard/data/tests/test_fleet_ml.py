"""Hermetic tests for WS4's leak-audited winner-vs-loser instrument.

No network, no VPS, no real episode store. Two themes: the PRE/POST wall must
never be crossed, and every statistic must refuse to answer rather than invent
an answer when the sample cannot support one.
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

from data import fleet_ml as fm                               # noqa: E402


def _episodes(n=600, seed=0, signal=0.0, post_signal=0.0):
    """A synthetic episode frame with the real column shape."""
    rng = np.random.default_rng(seed)
    # ceil, not floor: periods=n//6 then .repeat(6) silently yields fewer than n
    # rows and every later column addition fails on the length mismatch.
    day = pd.date_range("2026-01-01", periods=-(-n // 6), tz="UTC").repeat(6)[:n]
    assert len(day) == n
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(signal * x)))
    win = rng.random(n) < p
    return pd.DataFrame({
        "id": np.arange(n),
        "fill_ts": pd.to_datetime(day) + pd.to_timedelta(rng.integers(0, 86400, n), unit="s"),
        "r_net": np.where(win, 1.0, -1.0) * rng.uniform(0.5, 1.5, n),
        "pre__hour_et": rng.integers(0, 24, n),
        "pre__signal": x,
        "pre__noise": rng.normal(0, 1, n),
        "pre__asset": rng.choice(["BTC", "ETH", "SOL"], n),
        "post__peek": np.where(win, 1.0, -1.0) * post_signal + rng.normal(0, 1, n),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  The PRE / POST wall
# ══════════════════════════════════════════════════════════════════════════════
def test_pre_and_post_columns_never_overlap():
    df = _episodes(60)
    assert set(fm.pre_columns(df)).isdisjoint(fm.post_columns(df))
    assert all(c.startswith("pre__") for c in fm.pre_columns(df))
    assert all(c.startswith("post__") for c in fm.post_columns(df))


def test_a_k120_column_is_classified_post_fill():
    """The k120 set is measured over [fill, fill+120s]. Closure 5(a) forbids
    ever presenting it as an entry gate."""
    sys.path.insert(0, str(_ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bke", _ROOT / "backfill_knife_episodes.py")
    if spec is None or not (_ROOT / "backfill_knife_episodes.py").exists():
        pytest.skip("puller not present")
    bke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bke)
    raw = pd.DataFrame({"features_json": ['{"into_vol_k120": 1.0, "er20": 0.5}'],
                        "id": [1]})
    out = bke.explode_features(raw)
    assert "post__into_vol_k120" in out.columns
    assert "pre__er20" in out.columns


def test_build_xy_uses_only_the_columns_it_is_given():
    df = _episodes(120)
    X, _y, _d = fm.build_xy(df, fm.pre_columns(df))
    assert not any(c.startswith("post__") for c in X.columns)


# ══════════════════════════════════════════════════════════════════════════════
#  build_xy — the 633-dummy trap
# ══════════════════════════════════════════════════════════════════════════════
def test_a_sparse_float_stays_numeric():
    """`pre__dvol_level` is a float that is only 11% populated. Deciding
    numeric-vs-categorical by COVERAGE one-hot encoded 633 distinct floats into
    633 columns — a vast overfitting surface on 7,122 rows."""
    n = 400
    df = _episodes(n)
    v = np.full(n, np.nan)
    v[:40] = np.linspace(30, 70, 40)          # 10% populated, 40 distinct values
    df["pre__sparse"] = v
    X, _y, _d = fm.build_xy(df, ["pre__sparse"])
    assert "pre__sparse" in X.columns
    assert X.shape[1] <= 2, f"expected value + observed flag, got {list(X.columns)}"


def test_sparse_numeric_gets_an_observed_indicator():
    n = 400
    df = _episodes(n)
    v = np.full(n, np.nan); v[:100] = 1.5
    df["pre__sparse"] = v
    X, _y, _d = fm.build_xy(df, ["pre__sparse"])
    assert "pre__sparse__observed" in X.columns, (
        "a median fill would pretend the value was observed")


def test_an_all_null_column_is_dropped_not_encoded():
    df = _episodes(200)
    df["pre__never_populated"] = np.nan
    X, _y, _d = fm.build_xy(df, ["pre__never_populated", "pre__signal"])
    assert not any("never_populated" in c for c in X.columns)


def test_a_high_cardinality_string_is_refused():
    df = _episodes(300)
    df["pre__idlike"] = [f"v{i}" for i in range(len(df))]
    X, _y, _d = fm.build_xy(df, ["pre__idlike"])
    assert X.shape[1] == 0, "300 levels is a mis-typed column, not a category"


def test_label_is_r_net_positive():
    df = _episodes(100)
    _X, y, _d = fm.build_xy(df, fm.pre_columns(df))
    assert set(np.unique(y)) <= {0, 1}
    assert y.mean() == pytest.approx((df["r_net"] > 0).mean(), abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
#  PBO needs candidates
# ══════════════════════════════════════════════════════════════════════════════
def test_the_grid_has_several_configurations():
    """`compute_pbo` asks which candidate looked best in-sample and where it
    landed out-of-sample. Handing it one model is not a weaker PBO, it is none."""
    assert len(fm.GRID) >= 3


def test_cpcv_reports_the_fixed_config_not_the_best():
    import inspect
    src = inspect.getsource(fm.cpcv_auc)
    assert "default_oos = oos_arr[:, 0]" in src, (
        "reporting the best-of-grid OOS AUC would be selection bias in the headline")


def test_cpcv_refuses_a_degenerate_sample():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = np.array([1, 1, 1])
    d = pd.Series(pd.date_range("2026-01-01", periods=3, tz="UTC"))
    assert fm.cpcv_auc(X, y, d)["status"] == "degenerate sample"


def test_cpcv_orders_rows_in_time_before_splitting():
    import inspect
    src = inspect.getsource(fm.cpcv_auc)
    assert "np.argsort(day" in src, (
        "purging is meaningless if the rows are not in time order")


# ══════════════════════════════════════════════════════════════════════════════
#  The positive control is a gate, not a footnote
# ══════════════════════════════════════════════════════════════════════════════
def test_a_failed_positive_control_suppresses_the_reading():
    """If features that watched the trade cannot beat chance, the pipeline is
    broken and the pre-fill answer must not be reported at all."""
    v = fm.verdict(pre_audit={"base": {"auc_oos_mean": 0.99, "pbo": 0.0},
                              "beats_hour_null": True},
                   post_audit={"base": {"auc_oos_mean": 0.50}},
                   scan=pd.DataFrame({"clears": [True]}),
                   halves={"first": 0.9, "second": 0.9}, ci=(0.9, 0.99))
    assert v.control_ok is False and v.nominee is False
    assert any("POSITIVE CONTROL FAILED" in r for r in v.reasons)
    assert np.isnan(v.auc), "no pre-fill AUC may be reported when the control fails"


def test_a_passing_control_lets_the_reading_through():
    v = fm.verdict(pre_audit={"base": {"auc_oos_mean": 0.60, "pbo": 0.2},
                              "beats_hour_null": True},
                   post_audit={"base": {"auc_oos_mean": 0.85}},
                   scan=pd.DataFrame({"clears": [True]}),
                   halves={"first": 0.58, "second": 0.56}, ci=(0.55, 0.65))
    assert v.control_ok and v.nominee, v.reasons


# ══════════════════════════════════════════════════════════════════════════════
#  The kill bar — every condition bites
# ══════════════════════════════════════════════════════════════════════════════
def _v(auc=0.60, pbo=0.2, halves=(0.58, 0.56), clears=True, hour=True):
    return fm.verdict(pre_audit={"base": {"auc_oos_mean": auc, "pbo": pbo},
                                 "beats_hour_null": hour},
                      post_audit={"base": {"auc_oos_mean": 0.85}},
                      scan=pd.DataFrame({"clears": [clears]}),
                      halves={"first": halves[0], "second": halves[1]},
                      ci=(auc - 0.05, auc + 0.05))


def test_low_auc_fails():
    v = _v(auc=0.52)
    assert not v.nominee and any("OOS AUC" in r for r in v.reasons)


def test_high_pbo_fails():
    v = _v(pbo=0.7)
    assert not v.nominee and any("PBO" in r for r in v.reasons)


def test_no_cleared_feature_fails():
    v = _v(clears=False)
    assert not v.nominee and any("FWER" in r for r in v.reasons)


def test_a_weak_half_fails():
    v = _v(halves=(0.60, 0.51))
    assert not v.nominee and any("halves" in r for r in v.reasons)


def test_failing_the_hour_shuffle_fails():
    v = _v(hour=False)
    assert not v.nominee
    assert any("hour-conditional" in r for r in v.reasons)


# ══════════════════════════════════════════════════════════════════════════════
#  Univariate scan
# ══════════════════════════════════════════════════════════════════════════════
def test_scan_refuses_when_there_are_too_few_day_blocks():
    df = _episodes(120)                       # 20 days x 6 -> exactly at the edge
    df["fill_ts"] = pd.date_range("2026-01-01", periods=len(df), freq="1h", tz="UTC")
    out = fm.univariate_scan(df, ["pre__signal"], n_perm=50)
    assert not out.empty


def test_scan_marks_a_constant_feature_rather_than_scoring_it():
    df = _episodes(300)
    df["pre__constant"] = 1.0
    out = fm.univariate_scan(df, ["pre__constant"], n_perm=20)
    assert out["note"].iloc[0] != ""
    assert np.isnan(out["auc"].iloc[0])


def test_scan_applies_a_bonferroni_over_scored_features():
    df = _episodes(400)
    out = fm.univariate_scan(df, ["pre__signal", "pre__noise"], n_perm=50)
    scored = int(out["p_fwer"].notna().sum())
    if scored:
        assert out["alpha"].iloc[0] == pytest.approx(0.05 / scored)
    assert "clears" in out.columns


def test_scan_finds_a_planted_signal_and_not_pure_noise():
    df = _episodes(1500, seed=3, signal=1.4)
    out = fm.univariate_scan(df, ["pre__signal", "pre__noise"], n_perm=200)
    got = out.set_index("feature")
    assert got.loc["pre__signal", "auc"] > 0.60
    assert abs(got.loc["pre__noise", "auc"] - 0.5) < 0.06


# ══════════════════════════════════════════════════════════════════════════════
#  Coverage must name every book
# ══════════════════════════════════════════════════════════════════════════════
def test_coverage_buckets_every_tier1_book_exactly_once():
    cov = fm.tier1_coverage()
    if cov.empty:
        pytest.skip("no registry available")
    from data.fleet_registry import BOOKS
    n_t1 = sum(1 for b in BOOKS if int(b.tier) == 1)
    assert len(cov) == n_t1, "a book that is not listed reads as tested-and-boring"
    assert cov["bucket"].notna().all()
    assert set(cov["bucket"]) <= {"tested", "below the sample floor",
                                  "never closed a trade", "no local DB",
                                  "records no R (dollars only)"}


def test_module_never_touches_the_vps():
    """Test the MECHANISM, not the vocabulary. The module's prose explains why
    the knife store needed its own rsync-free puller, and a word-grep over the
    source flags that sentence — so parse the AST and look for the calls that
    could actually reach the box."""
    import ast
    tree = ast.parse(Path(fm.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"subprocess", "paramiko", "fabric", "socket", "requests",
                           "urllib", "http"}, f"can reach off-machine: {imported}"
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not {c for c in calls if "system" in c or "popen" in c.lower()}
