"""Hermetic tests for WS2's volatility / cross-asset atlas.

No network, no VPS, no duckdb: every input is a synthetic frame built in the
test, so a broken point-in-time join, a leaking percentile or a permutation that
forgets to block by day fails here rather than on the page.
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

from data import vol_atlas as va                              # noqa: E402


def _dvol_frame(hours=500, start="2026-01-01", seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    lvl = 50 + np.cumsum(rng.normal(0, 0.5, hours))
    return pd.DataFrame({"ts_utc": ts, "dvol": lvl})


# ══════════════════════════════════════════════════════════════════════════════
#  Point-in-time: a bar is not knowable until it closes
# ══════════════════════════════════════════════════════════════════════════════
def test_asof_join_only_sees_bars_that_have_closed():
    """Deribit stamps a bar at its OPEN, so the 10:00 bar is complete at 11:00.
    A fill at 10:30 may see 09:00 and no later."""
    f = pd.DataFrame({"ts_utc": pd.to_datetime(
        ["2026-09-01T09:00Z", "2026-09-01T10:00Z", "2026-09-01T11:00Z"], utc=True),
        "dvol": [10.0, 20.0, 30.0], "pct30": [.1, .2, .3], "pct90": [.1, .2, .3],
        "d1h": [0, 10, 10], "d24h": [0, 0, 0], "abs_band": ["LOW"] * 3,
        "pct_band": ["P_LOW"] * 3})
    w = pd.Series(pd.to_datetime(
        ["2026-09-01T10:30Z", "2026-09-01T11:00Z", "2026-09-01T11:59Z"], utc=True))
    out = va.dvol_asof(f, w)
    assert list(out["dvol"]) == [10.0, 20.0, 20.0]


def test_asof_before_the_series_starts_is_nan_not_the_first_value():
    f = _dvol_frame(10, "2026-05-01")
    f["pct30"] = f["pct90"] = f["d1h"] = f["d24h"] = np.nan
    f["abs_band"] = f["pct_band"] = None
    out = va.dvol_asof(f, pd.Series(pd.to_datetime(["2020-01-01T00:00Z"], utc=True)))
    assert np.isnan(out["dvol"].iloc[0]), "a fill before the index existed knows nothing"


def test_asof_lag_is_configurable_and_zero_lag_is_stricter():
    f = pd.DataFrame({"ts_utc": pd.to_datetime(["2026-09-01T10:00Z"], utc=True),
                      "dvol": [42.0], "pct30": [.5], "pct90": [.5], "d1h": [0.0],
                      "d24h": [0.0], "abs_band": ["LOW"], "pct_band": ["P_MID"]})
    w = pd.Series(pd.to_datetime(["2026-09-01T10:30Z"], utc=True))
    assert np.isnan(va.dvol_asof(f, w, lag_hours=1)["dvol"].iloc[0])
    assert va.dvol_asof(f, w, lag_hours=0)["dvol"].iloc[0] == 42.0


# ══════════════════════════════════════════════════════════════════════════════
#  Percentiles must be trailing, never whole-sample
# ══════════════════════════════════════════════════════════════════════════════
def test_percentile_is_trailing_so_the_past_cannot_see_the_future(monkeypatch):
    """A level that is the highest ever SO FAR must read near 1.0 even if the
    series later goes much higher — a whole-sample rank would say otherwise."""
    hours = 24 * 60
    ts = pd.date_range("2026-01-01", periods=hours, freq="1h", tz="UTC")
    lvl = np.concatenate([np.linspace(20, 60, hours // 2), np.linspace(60, 200, hours - hours // 2)])
    monkeypatch.setattr(va, "load_dvol_hourly",
                        lambda symbol="BTC": pd.DataFrame({"ts_utc": ts, "dvol": lvl}))
    f = va.dvol_features("BTC")
    mid = f.iloc[hours // 2 - 1]
    assert mid["pct30"] > 0.9, "at the midpoint it was the highest level yet"
    assert f["pct30"].max() <= 1.0 and f["pct30"].min() >= 0.0


def test_deltas_use_completed_bars_only(monkeypatch):
    ts = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    lvl = np.arange(48, dtype=float)
    monkeypatch.setattr(va, "load_dvol_hourly",
                        lambda symbol="BTC": pd.DataFrame({"ts_utc": ts, "dvol": lvl}))
    f = va.dvol_features("BTC")
    assert f["d1h"].dropna().eq(1.0).all()
    assert f["d24h"].dropna().eq(24.0).all()
    assert np.isnan(f["d1h"].iloc[0]), "the first bar has no predecessor to difference"


def test_absolute_bands_match_the_live_iv_gate():
    """The atlas and lr_faithful_filters.dvol_band must agree about HIGH."""
    from backtrader_framework.optimization.strategy_adapters.lr_faithful_filters import dvol_band
    ts = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    levels = [30.0, 50.0, 64.9, 80.0]
    f = pd.DataFrame({"ts_utc": ts, "dvol": levels})
    banded = pd.cut(f["dvol"], bins=[b[1] for b in va.ABS_BANDS] + [va.ABS_BANDS[-1][2]],
                    labels=[b[0] for b in va.ABS_BANDS], right=False)
    assert [str(x) for x in banded] == [dvol_band(v) for v in levels]


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-asset mechanics
# ══════════════════════════════════════════════════════════════════════════════
def test_lead_lag_reports_both_directions():
    import inspect
    src = inspect.getsource(va.lead_lag)
    assert "btc_leads_" in src and "alt_leads_" in src, (
        "a one-directional lead test cannot tell information flow from a smeared clock")


def test_tstat_matches_the_textbook():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    expect = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert va._tstat(x) == pytest.approx(expect)
    assert np.isnan(va._tstat(np.array([1.0])))
    assert np.isnan(va._tstat(np.array([2.0, 2.0, 2.0])))


# ══════════════════════════════════════════════════════════════════════════════
#  The family-wise bar — and the block that makes it honest
# ══════════════════════════════════════════════════════════════════════════════
def test_permutation_finds_no_effect_in_pure_noise():
    rng = np.random.default_rng(4)
    n = 900
    days = np.repeat(np.arange(n // 30), 30)
    vals = rng.normal(0, 1, n)
    bands = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    out = va.band_permutation_test(vals, bands, days, n_perm=400, seed=1)
    assert out["p_fwer"] > 0.05, "noise must not clear the bar"


def test_permutation_finds_a_real_band_effect():
    rng = np.random.default_rng(5)
    n = 900
    days = np.repeat(np.arange(n // 30), 30)
    bands = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
    vals = np.where(bands == "A", rng.normal(1.2, 1, n), rng.normal(-1.2, 1, n))
    out = va.band_permutation_test(vals, bands, days, n_perm=400, seed=1)
    assert out["p_fwer"] < 0.05 and out["obs_max_abs_t"] > 3


def test_permutation_blocks_by_day_not_by_fill():
    """Day-correlated noise with a band label that changes only between days is
    the classic false positive. Shuffling fill-by-fill would break the block and
    hand back a null that is far too easy to beat; blocking by day must not."""
    rng = np.random.default_rng(6)
    n_days, per_day = 40, 25
    day_effect = rng.normal(0, 1.0, n_days)
    vals = np.repeat(day_effect, per_day) + rng.normal(0, 0.05, n_days * per_day)
    days = np.repeat(np.arange(n_days), per_day)
    bands = np.repeat(np.where(np.arange(n_days) % 2 == 0, "A", "B"), per_day)
    blocked = va.band_permutation_test(vals, bands, days, n_perm=500, seed=2)
    assert blocked["p_fwer"] > 0.05, (
        "a day-level label on day-level noise carries no fill-level information")


def test_permutation_is_reported_with_its_observed_statistic():
    rng = np.random.default_rng(7)
    n = 900
    out = va.band_permutation_test(rng.normal(0, 1, n),
                                   np.array(["A", "B"])[rng.integers(0, 2, n)],
                                   np.repeat(np.arange(30), 30), n_perm=200)
    assert set(out) >= {"obs_max_abs_t", "p_fwer", "n_perm", "n_days"}
    assert 0 < out["p_fwer"] <= 1


def test_permutation_refuses_a_sample_too_small_to_speak():
    out = va.band_permutation_test(np.arange(10.0), np.array(["A"] * 10),
                                   np.arange(10), n_perm=50)
    assert np.isnan(out["p_fwer"])


# ══════════════════════════════════════════════════════════════════════════════
#  Skew: the mirrored bracket is not optional
# ══════════════════════════════════════════════════════════════════════════════
def test_skew_reports_the_unconditional_touch_rate_as_the_baseline():
    import inspect
    src = inspect.getsource(va.skew_direction)
    assert "up_rate_unconditional" in src
    assert "mirror" in src.lower(), (
        "a conditional first-touch rate is meaningless without its baseline")


def test_skew_direction_fails_soft_without_inputs(monkeypatch):
    monkeypatch.setattr(va, "load_rr25", lambda: pd.DataFrame())
    assert va.skew_direction()["status"] == "no rr25 series"


# ══════════════════════════════════════════════════════════════════════════════
#  Nothing here may write to the VPS
# ══════════════════════════════════════════════════════════════════════════════
def test_module_never_writes_remotely():
    src = (Path(va.__file__)).read_text()
    for forbidden in ("scp ", "rsync", "ssh ", "sudo"):
        assert forbidden not in src, f"WS2 computes locally; found {forbidden!r}"


def test_cache_dir_is_local_and_gitignored():
    assert va.CACHE_DIR.is_relative_to(_ROOT / "flow_aux_data")


def test_permutation_ignores_a_level_shift_that_hits_every_band():
    """A book that loses the same amount in every band has NO band effect. If
    the statistic were band-mean-against-zero, that book would score enormous t
    and read as a discovery; the contrast makes it score nothing."""
    rng = np.random.default_rng(9)
    n = 900
    days = np.repeat(np.arange(n // 30), 30)
    bands = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    vals = rng.normal(-0.5, 1.0, n)          # uniformly negative, no band structure
    out = va.band_permutation_test(vals, bands, days, n_perm=400, seed=3)
    assert out["obs_max_abs_t"] < 3
    assert out["p_fwer"] > 0.05


def test_fleet_band_fwer_applies_a_bonferroni_across_families():
    import inspect
    src = inspect.getsource(va.fleet_band_fwer)
    assert "alpha_bonferroni" in src and "clears" in src
    assert "CONTRAST" in inspect.getsource(va.band_permutation_test)


# ══════════════════════════════════════════════════════════════════════════════
#  The degenerate-null guard — the false positive this WOULD have shipped
# ══════════════════════════════════════════════════════════════════════════════
def test_too_few_day_blocks_refuses_a_p_value():
    """`halt-shadow(era2)` spans 7 days and came back p=0.0005 on |t| = 1.67.
    With 7 blocks split 4/3 there are only 35 arrangements, so the smallest
    honest p-value is 1/35 — the sampler simply never matched. Refuse instead."""
    rng = np.random.default_rng(11)
    per_day = 20
    vals = rng.normal(0, 1, 7 * per_day)
    days = np.repeat(np.arange(7), per_day)
    bands = np.repeat(np.where(np.arange(7) % 2 == 0, "A", "B"), per_day)
    out = va.band_permutation_test(vals, bands, days, n_perm=2000)
    assert np.isnan(out["p_fwer"])
    assert out["n_days"] == 7 and "day-blocks" in out["note"]


def test_a_single_day_block_can_never_be_permuted():
    rng = np.random.default_rng(12)
    vals = rng.normal(0, 1, 140)
    out = va.band_permutation_test(vals, np.array(["A", "B"] * 70), np.zeros(140))
    assert np.isnan(out["p_fwer"]), "permuting one block changes nothing"


def test_min_days_is_configurable_for_a_deliberate_small_sample():
    rng = np.random.default_rng(13)
    per_day = 20
    vals = rng.normal(0, 1, 10 * per_day)
    days = np.repeat(np.arange(10), per_day)
    bands = np.repeat(np.where(np.arange(10) % 2 == 0, "A", "B"), per_day)
    assert np.isnan(va.band_permutation_test(vals, bands, days, n_perm=300)["p_fwer"])
    loose = va.band_permutation_test(vals, bands, days, n_perm=300, min_days=5)
    assert not np.isnan(loose["p_fwer"])


def test_a_refused_family_never_reads_as_clearing():
    import inspect
    src = inspect.getsource(va.fleet_band_fwer)
    assert 'df["p_fwer"].notna() &' in src, "a NaN p-value must not count as a discovery"
    assert "max(tested, 1)" in src, "an untested family must not tighten the bar for the rest"


# ══════════════════════════════════════════════════════════════════════════════
#  Fleet join hygiene
# ══════════════════════════════════════════════════════════════════════════════
def test_fleet_frame_carries_the_day_block_the_permutation_needs():
    import inspect
    src = inspect.getsource(va.fleet_fills_with_dvol)
    assert '["day"]' in src and "floor(\"D\")" in src.replace("'", '"'), (
        "the permutation blocks by day; the frame must carry one")


def test_fleet_r_by_band_counts_bets_not_rows():
    import inspect
    src = inspect.getsource(va.fleet_r_by_band)
    assert 'groupby("cluster")["r"].mean()' in src, (
        "a family that re-records one setup must not vote once per row")


def test_bands_are_read_from_a_closed_bar_in_the_fleet_join():
    import inspect
    src = inspect.getsource(va.fleet_fills_with_dvol)
    assert "dvol_asof" in src, "the fleet join must go through the PIT helper"


def test_a_missing_delta_is_not_called_flat():
    """np.where(NaN > 0, ...) is False and np.where(NaN < 0, ...) is False, so a
    naive chain quietly labels 'unknown' as 'flat' and puts it in a bucket the
    permutation then tests. It must stay None and be dropped."""
    import inspect
    src = inspect.getsource(va.fleet_fills_with_dvol)
    assert "d24.isna()" in src, "an unknown delta must not be labelled"


# ══════════════════════════════════════════════════════════════════════════════
#  Half-split — a survivor owes both bars
# ══════════════════════════════════════════════════════════════════════════════
def _bets(bands, days, r):
    return pd.DataFrame({"band": bands, "day": pd.to_datetime(days, utc=True), "r": r})


def test_half_split_agrees_when_the_ordering_holds():
    n = 200
    bands = np.array(["A", "B"] * (n // 2))
    days = pd.date_range("2026-01-01", periods=n, freq="6h", tz="UTC")
    r = np.where(bands == "A", 1.0, -1.0)
    assert va._half_split_agrees(_bets(bands, days, r)) is True


def test_half_split_rejects_an_effect_that_only_exists_in_one_half():
    n = 200
    bands = np.array(["A", "B"] * (n // 2))
    days = pd.date_range("2026-01-01", periods=n, freq="6h", tz="UTC")
    r = np.concatenate([np.where(bands[:n // 2] == "A", 1.0, -1.0),
                        np.where(bands[n // 2:] == "A", -1.0, 1.0)])
    assert va._half_split_agrees(_bets(bands, days, r)) is False


def test_half_split_declines_when_a_half_is_too_thin():
    bands = np.array(["A", "B", "A"])
    days = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
    assert va._half_split_agrees(_bets(bands, days, np.array([1.0, -1.0, 1.0]))) is None


def test_clearing_requires_the_half_split_too():
    import inspect
    src = inspect.getsource(va.fleet_band_fwer)
    assert 'df["half_split"].eq(True)' in src, (
        "an effect visible in only one half is a window, not a mechanism; and an "
        "undecidable half-split (None) must not pass either")


def test_asof_refuses_a_stale_match():
    """An as-of join with no age limit hands a fill the last print before a gap,
    however old — the row then looks featured while carrying another week's
    number. The depth stale-signal bug, in miniature."""
    f = pd.DataFrame({"ts_utc": pd.to_datetime(["2026-09-01T10:00Z"], utc=True),
                      "dvol": [42.0], "pct30": [.5], "pct90": [.5], "d1h": [0.0],
                      "d24h": [0.0], "abs_band": ["LOW"], "pct_band": ["P_MID"]})
    fresh = pd.Series(pd.to_datetime(["2026-09-01T12:00Z"], utc=True))
    stale = pd.Series(pd.to_datetime(["2026-09-08T12:00Z"], utc=True))
    assert va.dvol_asof(f, fresh)["dvol"].iloc[0] == 42.0
    assert np.isnan(va.dvol_asof(f, stale)["dvol"].iloc[0])
    assert va.dvol_asof(f, stale, max_stale_hours=24 * 30)["dvol"].iloc[0] == 42.0


def test_a_band_only_a_couple_of_days_can_speak_for_is_refused():
    """The fleet's ABSOLUTE bands put nearly every fill in one bucket. Most
    shuffles then leave the rare band unscoreable, the permuted statistic
    collapses, and nothing ever beats the observed — max |t| of 0.04 came back
    at p = 0.0005. Two bands must each be carried by real days, or no answer."""
    rng = np.random.default_rng(21)
    n_days, per_day = 40, 25
    days = np.repeat(np.arange(n_days), per_day)
    bands = np.repeat(np.where(np.arange(n_days) < 2, "RARE", "COMMON"), per_day)
    out = va.band_permutation_test(rng.normal(0, 1, n_days * per_day), bands, days,
                                   n_perm=300)
    assert np.isnan(out["p_fwer"])
    assert "days" in out["note"]


def test_an_unscoreable_shuffle_is_not_counted_as_evidence():
    import inspect
    src = inspect.getsource(va.band_permutation_test)
    assert "scored_bands >= 2" in src, "a one-band statistic is not a comparison"
    assert "is not evidence" in src, "an unscoreable draw must be skipped, not counted"


def test_fleet_frame_blocks_by_the_fills_own_day():
    """The DVOL bar behind a fill can sit in the previous day. The permutation
    block is a trading day's fills, so it must key on the fill, not the bar."""
    import inspect
    src = inspect.getsource(va.fleet_fills_with_dvol)
    assert 'allf["fill_ts"]' in src and "dvol_ts" not in src.split('allf["day"]')[1][:200]
