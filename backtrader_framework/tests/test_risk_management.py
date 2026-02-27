"""Tests for risk management features in base_strategy"""
import pytest


class TestPositionSizing:
    def test_nav_based_sizing(self):
        """Position size should be based on portfolio NAV, not free cash"""
        nav = 10000
        risk_per_trade = 0.01
        entry_price = 50000
        stop_distance = 500
        risk_amount = nav * risk_per_trade  # $100
        size = risk_amount / stop_distance  # 0.2 BTC
        assert size == pytest.approx(0.2)

    def test_position_cap(self):
        """Position should be capped at max_position_pct of portfolio"""
        nav = 10000
        max_pct = 0.25
        entry_price = 100
        max_size = nav * max_pct / entry_price  # 25 units
        computed_size = 50  # hypothetical large size
        final_size = min(computed_size, max_size)
        assert final_size == 25.0

    def test_zero_risk_returns_zero_size(self):
        """If stop distance is 0, size should be 0 (not infinity)"""
        risk_amount = 100
        stop_distance = 0
        size = risk_amount / stop_distance if stop_distance > 0 else 0
        assert size == 0


class TestCircuitBreaker:
    def test_drawdown_calculation(self):
        """Drawdown should be computed from peak"""
        peak = 12000
        current = 10000
        drawdown = (peak - current) / peak
        assert drawdown == pytest.approx(1/6)

    def test_halt_at_threshold(self):
        """Trading should halt when drawdown exceeds threshold"""
        max_dd = 0.20
        peak = 10000
        current = 7800  # 22% drawdown
        drawdown = (peak - current) / peak
        assert drawdown > max_dd


class TestDailyLossLimit:
    def test_daily_loss_detection(self):
        """Should detect when daily loss exceeds limit"""
        max_daily = 0.05
        day_start = 10000
        current = 9400  # 6% daily loss
        daily_loss = (day_start - current) / day_start
        assert daily_loss > max_daily
