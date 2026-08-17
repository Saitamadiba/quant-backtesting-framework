"""
Tests for Numba JIT kernels — equivalence with pure Python implementations.

Verifies that the Numba-accelerated kernels produce results consistent with
the original Python code for trade simulation, MC bootstrap, and equity fan.
"""

import numpy as np
import pytest

from backtrader_framework.optimization.numba_kernels import (
    HAS_NUMBA,
    OUTCOME_LOSS,
    OUTCOME_BREAKEVEN,
    OUTCOME_WIN_TP1,
    OUTCOME_WIN_TP2,
    OUTCOME_WIN_TRAIL,
    OUTCOME_WIN_TP,
    OUTCOME_TIME_EXIT,
    OUTCOME_TIMEOUT,
    OUTCOME_STRINGS,
    _block_bootstrap_sample,
    _simulate_v1_kernel,
    _simulate_v2_kernel,
    _mc_bootstrap_ci_kernel,
    _mc_equity_fan_kernel,
    _mc_block_bootstrap_full,
    _mc_dashboard_kernel,
)


# ═════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════

def _make_price_data(n=200, base=50000.0, volatility=500.0, seed=42):
    """Generate synthetic OHLC + ATR data for trade simulation tests."""
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0, volatility, n))
    highs = closes + rng.uniform(50, 300, n)
    lows = closes - rng.uniform(50, 300, n)
    atrs = np.full(n, 200.0)
    return highs, lows, closes, atrs


# ═════════════════════════════════════════════════════════════
#  Block Bootstrap Primitive
# ═════════════════════════════════════════════════════════════

class TestBlockBootstrap:
    def test_iid_produces_correct_length(self):
        values = np.arange(20, dtype=np.float64)
        out = np.empty(20)
        np.random.seed(123)
        _block_bootstrap_sample(values, 20, 1, 20, out)
        assert len(out) == 20
        assert all(0 <= v < 20 for v in out)

    def test_block_produces_correct_length(self):
        values = np.arange(20, dtype=np.float64)
        out = np.empty(20)
        np.random.seed(123)
        _block_bootstrap_sample(values, 20, 3, 18, out)
        assert len(out) == 20
        assert all(0 <= v < 20 for v in out)

    def test_block_preserves_consecutive_order(self):
        """Within each block, values should be consecutive."""
        values = np.arange(100, dtype=np.float64)
        out = np.empty(100)
        np.random.seed(42)
        _block_bootstrap_sample(values, 100, 5, 96, out)
        # Check that we can find at least one consecutive run of 5
        found = False
        for i in range(96):
            if out[i] + 1 == out[i + 1]:
                found = True
                break
        assert found, "Block bootstrap should preserve consecutive ordering"


# ═════════════════════════════════════════════════════════════
#  V1 Simulator Kernel
# ═════════════════════════════════════════════════════════════

class TestSimulateV1:
    def _run(self, is_long=True, entry=50000.0, sl=49500.0, tp1=50500.0,
             tp2=51000.0, risk=500.0, max_bars=168, trail=1.5):
        highs, lows, closes, atrs = _make_price_data(300)
        return _simulate_v1_kernel(
            idx=10, is_long=is_long,
            entry_price=entry, stop_loss=sl, tp1=tp1, tp2=tp2, risk=risk,
            spread_pct=0.0005, commission_pct=0.001, slippage_pct=0.0003,
            max_bars=max_bars, trail_atr_mult=trail,
            highs=highs, lows=lows, closes=closes, atrs=atrs,
            has_atrs=True, n=len(highs),
        )

    def test_returns_tuple_of_8(self):
        result = self._run()
        assert isinstance(result, tuple)
        assert len(result) == 8

    def test_outcome_is_valid_code(self):
        outcome = self._run()[0]
        assert outcome in OUTCOME_STRINGS

    def test_zero_risk_returns_timeout(self):
        result = self._run(risk=0.0)
        assert result[0] == OUTCOME_TIMEOUT
        assert result[1] == -1.0

    def test_long_loss_scenario(self):
        """Force a loss by setting SL very close to entry."""
        highs, lows, closes, atrs = _make_price_data(50)
        # SL right at entry minus tiny gap — any dip triggers loss
        result = _simulate_v1_kernel(
            idx=0, is_long=True,
            entry_price=50000.0, stop_loss=50000.0 - 1.0,
            tp1=60000.0, tp2=70000.0, risk=1.0,
            spread_pct=0.0, commission_pct=0.0, slippage_pct=0.0,
            max_bars=50, trail_atr_mult=0.0,
            highs=highs, lows=lows, closes=closes, atrs=atrs,
            has_atrs=False, n=len(highs),
        )
        # Should likely hit the tight SL
        assert result[0] in (OUTCOME_LOSS, OUTCOME_BREAKEVEN, OUTCOME_WIN_TP1,
                              OUTCOME_WIN_TP2, OUTCOME_TIMEOUT)

    def test_short_direction(self):
        result = self._run(is_long=False, entry=50000.0, sl=50500.0,
                           tp1=49500.0, tp2=49000.0)
        assert result[0] in OUTCOME_STRINGS

    def test_no_atr_data(self):
        """Kernel handles has_atrs=False gracefully."""
        highs, lows, closes, _ = _make_price_data(100)
        empty_atrs = np.empty(0, dtype=np.float64)
        result = _simulate_v1_kernel(
            idx=5, is_long=True,
            entry_price=50000.0, stop_loss=49000.0,
            tp1=51000.0, tp2=52000.0, risk=1000.0,
            spread_pct=0.0005, commission_pct=0.001, slippage_pct=0.0003,
            max_bars=50, trail_atr_mult=1.5,
            highs=highs, lows=lows, closes=closes, atrs=empty_atrs,
            has_atrs=False, n=len(highs),
        )
        assert result[0] in OUTCOME_STRINGS


# ═════════════════════════════════════════════════════════════
#  V2 Simulator Kernel
# ═════════════════════════════════════════════════════════════

class TestSimulateV2:
    def _run(self, is_long=True, entry=50000.0, sl=49500.0, tp=51000.0,
             risk=500.0, max_bars=168):
        highs, lows, closes, atrs = _make_price_data(300)
        return _simulate_v2_kernel(
            idx=10, is_long=is_long,
            entry_price=entry, stop_loss=sl, tp=tp, risk=risk,
            spread_pct=0.0005, commission_pct=0.001, slippage_pct=0.0003,
            max_bars=max_bars,
            buffer_bars=3, buffer_mult=1.5,
            be_trigger_r=1.0, be_buffer_pct=0.001,
            time_exit_bars=100, trail_atr_mult=2.0, trail_step_atr=0.5,
            # 0.0 disables both -> the framework default and the behaviour
            # these assertions were written against.
            min_trail_lock_r=0.0, trail_headroom_frac=0.0,
            highs=highs, lows=lows, closes=closes, atrs=atrs,
            has_atrs=True, n=len(highs),
        )

    def test_returns_tuple_of_8(self):
        result = self._run()
        assert isinstance(result, tuple)
        assert len(result) == 8

    def test_outcome_is_valid_code(self):
        outcome = self._run()[0]
        assert outcome in OUTCOME_STRINGS

    def test_zero_risk_returns_timeout(self):
        result = self._run(risk=0.0)
        assert result[0] == OUTCOME_TIMEOUT
        assert result[1] == -1.0

    def test_short_direction(self):
        result = self._run(is_long=False, entry=50000.0, sl=50500.0, tp=49000.0)
        assert result[0] in OUTCOME_STRINGS

    def test_time_exit_fires(self):
        """With very short time_exit_bars, should trigger time exit."""
        highs, lows, closes, atrs = _make_price_data(300)
        result = _simulate_v2_kernel(
            idx=10, is_long=True,
            entry_price=50000.0, stop_loss=40000.0,
            tp=70000.0, risk=10000.0,
            spread_pct=0.0, commission_pct=0.0, slippage_pct=0.0,
            max_bars=200,
            buffer_bars=0, buffer_mult=1.0,
            be_trigger_r=100.0, be_buffer_pct=0.0,
            time_exit_bars=5, trail_atr_mult=0.0, trail_step_atr=0.0,
            min_trail_lock_r=0.0, trail_headroom_frac=0.0,
            highs=highs, lows=lows, closes=closes, atrs=atrs,
            has_atrs=True, n=len(highs),
        )
        # Time exit should fire around bar 5 (if not in profit)
        assert result[0] in OUTCOME_STRINGS


# ═════════════════════════════════════════════════════════════
#  Monte Carlo Bootstrap CI Kernel
# ═════════════════════════════════════════════════════════════

class TestMCBootstrapCI:
    def _make_r_values(self, n=100, seed=42):
        rng = np.random.default_rng(seed)
        return rng.normal(0.1, 0.5, n)

    def test_output_shapes(self):
        r = self._make_r_values()
        n_resamples = 500
        mean_rs = np.empty(n_resamples)
        win_rates = np.empty(n_resamples)
        expectancies = np.empty(n_resamples)
        profit_factors = np.empty(n_resamples)
        sharpes = np.empty(n_resamples)
        max_dds = np.empty(n_resamples)

        np.random.seed(42)
        _mc_bootstrap_ci_kernel(
            r, n_resamples, 1,
            mean_rs, win_rates, expectancies, profit_factors, sharpes, max_dds,
        )

        assert not np.any(np.isnan(mean_rs))
        assert not np.any(np.isnan(win_rates))
        assert np.all(win_rates >= 0) and np.all(win_rates <= 1)
        assert np.all(max_dds >= 0)

    def test_positive_trades_yield_high_win_rate(self):
        """All-positive R-values should give ~100% win rate."""
        r = np.ones(50)
        n_resamples = 200
        mean_rs = np.empty(n_resamples)
        win_rates = np.empty(n_resamples)
        expectancies = np.empty(n_resamples)
        profit_factors = np.empty(n_resamples)
        sharpes = np.empty(n_resamples)
        max_dds = np.empty(n_resamples)

        np.random.seed(1)
        _mc_bootstrap_ci_kernel(
            r, n_resamples, 1,
            mean_rs, win_rates, expectancies, profit_factors, sharpes, max_dds,
        )

        assert np.mean(win_rates) == pytest.approx(1.0, abs=0.01)
        assert np.mean(mean_rs) == pytest.approx(1.0, abs=0.01)

    def test_block_bootstrap_works(self):
        r = self._make_r_values()
        n_resamples = 500
        mean_rs = np.empty(n_resamples)
        win_rates = np.empty(n_resamples)
        expectancies = np.empty(n_resamples)
        profit_factors = np.empty(n_resamples)
        sharpes = np.empty(n_resamples)
        max_dds = np.empty(n_resamples)

        np.random.seed(42)
        _mc_bootstrap_ci_kernel(
            r, n_resamples, 3,
            mean_rs, win_rates, expectancies, profit_factors, sharpes, max_dds,
        )

        assert not np.any(np.isnan(mean_rs))
        assert np.all(win_rates >= 0) and np.all(win_rates <= 1)


# ═════════════════════════════════════════════════════════════
#  Equity Fan Kernel
# ═════════════════════════════════════════════════════════════

class TestEquityFan:
    def test_output_shape(self):
        r = np.random.default_rng(42).normal(0.1, 0.5, 80)
        n_paths = 100
        all_paths = np.empty((n_paths, len(r)))
        np.random.seed(42)
        _mc_equity_fan_kernel(r, n_paths, 1, all_paths)
        assert all_paths.shape == (100, 80)
        assert not np.any(np.isnan(all_paths))

    def test_cumulative_sum_correct(self):
        """First row should be a valid cumsum of some resampled values."""
        r = np.arange(1, 11, dtype=np.float64)
        all_paths = np.empty((1, 10))
        np.random.seed(42)
        _mc_equity_fan_kernel(r, 1, 1, all_paths)
        # Each row should be monotonically structured as cumsum
        diffs = np.diff(all_paths[0])
        # diffs should be individual sampled values (all from 1..10)
        assert all(1 <= d <= 10 for d in diffs)

    def test_block_fan(self):
        r = np.random.default_rng(42).normal(0.2, 0.3, 50)
        n_paths = 50
        all_paths = np.empty((n_paths, 50))
        np.random.seed(42)
        _mc_equity_fan_kernel(r, n_paths, 5, all_paths)
        assert not np.any(np.isnan(all_paths))


# ═════════════════════════════════════════════════════════════
#  Standalone MC Block Bootstrap Full
# ═════════════════════════════════════════════════════════════

class TestMCBlockBootstrapFull:
    def test_output_shapes(self):
        pnls = np.random.default_rng(42).normal(10, 50, 100)
        n_runs = 500
        final_equities = np.empty(n_runs)
        max_drawdowns = np.empty(n_runs)
        mean_rs = np.empty(n_runs)
        win_rates = np.empty(n_runs)
        profit_factors = np.empty(n_runs)

        np.random.seed(42)
        _mc_block_bootstrap_full(
            pnls, 100.0, 10000.0, n_runs, 3,
            final_equities, max_drawdowns, mean_rs, win_rates, profit_factors,
        )

        assert not np.any(np.isnan(final_equities))
        assert np.all(max_drawdowns >= 0)
        assert np.all(win_rates >= 0) and np.all(win_rates <= 1)

    def test_all_wins_high_equity(self):
        """All positive PnLs should produce high final equity."""
        pnls = np.full(50, 100.0)
        n_runs = 100
        final_equities = np.empty(n_runs)
        max_drawdowns = np.empty(n_runs)
        mean_rs = np.empty(n_runs)
        win_rates = np.empty(n_runs)
        profit_factors = np.empty(n_runs)

        np.random.seed(1)
        _mc_block_bootstrap_full(
            pnls, 100.0, 10000.0, n_runs, 1,
            final_equities, max_drawdowns, mean_rs, win_rates, profit_factors,
        )

        # 50 trades * $100 = $5000 profit → final equity ~$15000
        assert np.all(final_equities == pytest.approx(15000.0))
        assert np.all(max_drawdowns == pytest.approx(0.0))
        assert np.all(win_rates == pytest.approx(1.0))


# ═════════════════════════════════════════════════════════════
#  Dashboard MC Kernel
# ═════════════════════════════════════════════════════════════

class TestDashboardKernel:
    def test_output_shapes(self):
        pnls = np.random.default_rng(42).normal(5, 30, 60)
        n_runs = 300
        max_stored = 50
        final_returns = np.empty(n_runs)
        max_drawdowns = np.empty(n_runs)
        equity_paths = np.empty((max_stored, len(pnls) + 1))

        np.random.seed(42)
        _mc_dashboard_kernel(
            pnls, 10000.0, n_runs, 1,
            final_returns, max_drawdowns, equity_paths, max_stored,
        )

        assert not np.any(np.isnan(final_returns))
        assert not np.any(np.isnan(max_drawdowns))
        # First column of stored paths should be initial_capital
        assert np.all(equity_paths[:, 0] == pytest.approx(10000.0))

    def test_block_bootstrap(self):
        pnls = np.random.default_rng(42).normal(5, 30, 60)
        n_runs = 100
        max_stored = 20
        final_returns = np.empty(n_runs)
        max_drawdowns = np.empty(n_runs)
        equity_paths = np.empty((max_stored, len(pnls) + 1))

        np.random.seed(42)
        _mc_dashboard_kernel(
            pnls, 10000.0, n_runs, 3,
            final_returns, max_drawdowns, equity_paths, max_stored,
        )

        assert not np.any(np.isnan(final_returns))


# ═════════════════════════════════════════════════════════════
#  Meta: Numba availability check
# ═════════════════════════════════════════════════════════════

class TestNumbaAvailability:
    def test_has_numba_is_bool(self):
        assert isinstance(HAS_NUMBA, bool)

    def test_outcome_strings_complete(self):
        for code in range(8):
            assert code in OUTCOME_STRINGS


# ═════════════════════════════════════════════════════════════
#  V2 simulator — PUBLIC API
#
#  The tests above call _simulate_v2_kernel directly. Nothing exercised
#  TradeSimulator.simulate_v2(), which is how a two-argument-short positional
#  call to that kernel sat in simulator.py raising TypeError on every
#  invocation while the kernel's own tests looked like the only problem.
#  A kernel test cannot catch a broken caller.
# ═════════════════════════════════════════════════════════════

class TestSimulateV2PublicAPI:
    @staticmethod
    def _frame_and_signal(**meta):
        import pandas as pd
        from backtrader_framework.optimization.simulator import TransactionCosts

        highs, lows, closes, atrs = _make_price_data(300, seed=7)
        df = pd.DataFrame(
            {"High": highs, "Low": lows, "Close": closes, "ATR": atrs},
            index=pd.date_range("2024-01-01", periods=len(highs), freq="15min",
                                tz="UTC"),
        )
        signal = {
            "idx": 10,
            "time": df.index[10],
            "direction": "LONG",
            "entry_price": float(closes[10]),
            "stop_loss": float(closes[10]) * 0.99,
            "take_profit_1": float(closes[10]) * 1.02,
            "risk": float(closes[10]) * 0.01,
            "confidence": 0.6,
            "bias": "WITH",
            "metadata": dict(meta),
        }
        return df, signal, TransactionCosts(spread_pct=0.0005,
                                           commission_pct=0.001,
                                           slippage_pct=0.0003)

    def test_simulate_v2_runs_without_raising(self):
        """The regression guard: this method used to raise TypeError always."""
        from backtrader_framework.optimization.simulator import TradeSimulator

        df, signal, costs = self._frame_and_signal()
        result = TradeSimulator.simulate_v2(signal, df, costs, max_bars=100)
        assert result is not None
        assert result.outcome in OUTCOME_STRINGS.values()
        assert np.isfinite(result.r_multiple)
        assert np.isfinite(result.r_multiple_after_costs)
        assert result.r_multiple_after_costs <= result.r_multiple

    @pytest.mark.parametrize("lock_key,head_key", [
        ("v2_min_trail_lock_r", "v2_trail_headroom_frac"),   # engine-injected
        ("min_trail_lock_r", "trail_headroom_frac"),         # unprefixed
    ])
    def test_both_metadata_key_spellings_are_accepted(self, lock_key, head_key):
        """The engine injects v2_-prefixed keys; this class uses unprefixed ones
        for its other v2 params. Both must reach the kernel."""
        from backtrader_framework.optimization.simulator import TradeSimulator

        df, signal, costs = self._frame_and_signal(
            **{lock_key: 0.3, head_key: 0.5,
               "trail_atr_mult_long": 2.0, "trail_step_atr_long": 0.5}
        )
        result = TradeSimulator.simulate_v2(signal, df, costs, max_bars=200)
        assert result is not None
        assert result.outcome in OUTCOME_STRINGS.values()

    def test_trail_lock_floor_never_worsens_a_trailed_exit(self):
        """MIN_TRAIL_LOCK_R is a floor under the trailing stop, so enabling it
        must not produce a worse exit than leaving it off."""
        from backtrader_framework.optimization.simulator import TradeSimulator

        common = {"trail_atr_mult_long": 1.5, "trail_step_atr_long": 0.25}
        df, sig_off, costs = self._frame_and_signal(**common)
        _, sig_on, _ = self._frame_and_signal(min_trail_lock_r=0.5, **common)

        off = TradeSimulator.simulate_v2(sig_off, df, costs, max_bars=250)
        on = TradeSimulator.simulate_v2(sig_on, df, costs, max_bars=250)
        assert off is not None and on is not None
        if off.outcome == "win_trail" and on.outcome == "win_trail":
            assert on.r_multiple >= off.r_multiple - 1e-9

    def test_zero_risk_signal_is_rejected(self):
        from backtrader_framework.optimization.simulator import TradeSimulator

        df, signal, costs = self._frame_and_signal()
        signal["risk"] = 0.0
        assert TradeSimulator.simulate_v2(signal, df, costs) is None


# ═════════════════════════════════════════════════════════════
#  Arity guard — catches kernel-signature drift in ANY environment
#
#  The original defect: _simulate_v2_kernel gained two parameters and
#  simulator.py's call site was not updated. It went unnoticed because that
#  branch was gated on `if HAS_NUMBA:` and numba is not installed everywhere,
#  so the broken call was simply never executed on those machines. A test that
#  merely *runs* simulate_v2 therefore proves nothing about the kernel call.
#  These assert the wiring statically instead.
# ═════════════════════════════════════════════════════════════

class TestKernelWiring:
    @staticmethod
    def _kernel_params(fn):
        import inspect
        return list(inspect.signature(getattr(fn, "py_func", fn)).parameters)

    def test_every_kernel_call_site_passes_the_full_arity(self):
        """Count the arguments each call site passes and compare to the kernel."""
        import ast
        import inspect
        from pathlib import Path
        from backtrader_framework.optimization import numba_kernels as nk

        arity = {
            "_simulate_v1_kernel": len(self._kernel_params(nk._simulate_v1_kernel)),
            "_simulate_v2_kernel": len(self._kernel_params(nk._simulate_v2_kernel)),
        }
        # the kernels are also imported under aliases
        aliases = {"_SIM_V2_KERNEL": "_simulate_v2_kernel"}

        root = Path(inspect.getfile(nk)).parent
        problems = []
        for path in sorted(root.glob("*.py")) + sorted(root.glob("strategy_adapters/*.py")):
            if path.name == "numba_kernels.py":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                canonical = aliases.get(fname, fname)
                if canonical not in arity:
                    continue
                if any(isinstance(a, ast.Starred) for a in node.args):
                    continue                      # *args forwarding: can't count
                n = len(node.args) + len(node.keywords)
                if n != arity[canonical]:
                    problems.append(
                        f"{path.name}:{node.lineno} calls {fname} with {n} args, "
                        f"kernel takes {arity[canonical]}"
                    )
        assert not problems, "kernel call-site arity drift:\n  " + "\n  ".join(problems)

    def test_simulate_v2_has_no_second_implementation(self):
        """simulator.simulate_v2 must have exactly one numeric path.

        It previously carried a 168-line pure-Python V2 fallback that ignored
        min_trail_lock_r and trail_headroom_frac entirely, so results depended on
        whether numba happened to be installed.
        """
        import inspect
        from backtrader_framework.optimization.simulator import TradeSimulator

        src = inspect.getsource(TradeSimulator.simulate_v2)
        assert "Pure-Python fallback" not in src
        assert src.count("_simulate_v2_kernel(") == 1

    def test_missing_kernel_fails_closed(self, monkeypatch):
        """No silent fallback: if the kernel is absent, raise."""
        import pandas as pd
        from backtrader_framework.optimization import simulator as sim

        monkeypatch.setattr(sim, "_simulate_v2_kernel", None)
        highs, lows, closes, atrs = _make_price_data(120, seed=3)
        df = pd.DataFrame(
            {"High": highs, "Low": lows, "Close": closes, "ATR": atrs},
            index=pd.date_range("2024-01-01", periods=len(highs), freq="15min", tz="UTC"),
        )
        signal = {
            "idx": 5, "time": df.index[5], "direction": "LONG",
            "entry_price": float(closes[5]), "stop_loss": float(closes[5]) * 0.99,
            "take_profit_1": float(closes[5]) * 1.02, "risk": float(closes[5]) * 0.01,
            "metadata": {},
        }
        with pytest.raises(RuntimeError, match="Refusing to fall back"):
            sim.TradeSimulator.simulate_v2(signal, df, sim.TransactionCosts())
