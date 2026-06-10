"""
Sanity tests for VolatilityEngine + backtest (PRD v3.1 math).

Run with either:
    python -m pytest tests/ -q
    python tests/test_engine.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (  # noqa: E402
    HISTORICAL_AVG_LOSS,
    HISTORICAL_AVG_WIN,
    HISTORICAL_WIN_RATE,
    VolatilityEngine,
    bs_call,
)
from backtest import BacktestConfig, run_backtest  # noqa: E402


def _recommend_kwargs(**overrides):
    """A clean Recommend setup (mirrors the app's demo ticker)."""
    rng = np.random.default_rng(7)
    base = dict(
        iv_near=0.85, iv_45=0.55, iv_30=0.70, rv_30=0.45,
        avg_30day_volume=4_500_000,
        historical_iv_series=pd.Series(np.clip(rng.normal(0.45, 0.08, 252), 0.15, 0.95)),
        atm_call_price=4.20, atm_put_price=3.90,
        historical_moves=[0.028, -0.034, 0.025, -0.041, 0.030, -0.022],
        spot_price=150.0,
    )
    base.update(overrides)
    return base


def test_core_metrics():
    assert abs(VolatilityEngine.calculate_term_structure_slope(0.85, 0.55) - 0.30) < 1e-9
    assert abs(VolatilityEngine.calculate_iv_rv_ratio(0.70, 0.45) - 1.5555) < 1e-3
    assert VolatilityEngine.calculate_iv_rv_ratio(0.70, 0.0) == 0.0
    assert abs(VolatilityEngine.calculate_expected_move(4.20, 3.90) - 6.885) < 1e-9
    assert VolatilityEngine.calculate_position_sizing(100_000) == 6_000.0


def test_iv_percentile():
    hist = pd.Series([0.30] * 200 + [0.90] * 52)  # 200 of 252 days below 0.5
    pct = VolatilityEngine.calculate_iv_percentile(0.50, hist)
    assert abs(pct - (200 / 252) * 100) < 1e-9
    assert VolatilityEngine.calculate_iv_percentile(0.50, pd.Series(dtype=float)) == 0.0


def test_kelly_matches_prd():
    k = VolatilityEngine.calculate_kelly_fraction()
    # PRD §5: p=0.62, b=0.45/0.35 → f* ≈ 32.44%, applied 10% ≈ 3.24%
    assert abs(k["full_kelly"] - 0.32444) < 1e-3
    assert abs(k["fractional_kelly"] - 0.032444) < 1e-4
    assert abs(k["win_loss_ratio_b"] - HISTORICAL_AVG_WIN / HISTORICAL_AVG_LOSS) < 1e-9


def test_signal_routing():
    # Full pass → Recommend
    res = VolatilityEngine.evaluate_ticker(**_recommend_kwargs())
    assert res["signal"] == "Recommend"
    # Contango → Avoid regardless of everything else
    res = VolatilityEngine.evaluate_ticker(**_recommend_kwargs(iv_near=0.50, iv_45=0.55))
    assert res["signal"] == "Avoid"
    # Backwardation but thin volume → Consider
    res = VolatilityEngine.evaluate_ticker(**_recommend_kwargs(avg_30day_volume=200_000))
    assert res["signal"] == "Consider"
    assert "liquidity" in res["reason"]
    # Backwardation but low IV percentile → Consider
    res = VolatilityEngine.evaluate_ticker(
        **_recommend_kwargs(historical_iv_series=pd.Series([0.95] * 252))
    )
    assert res["signal"] == "Consider"


def test_magnitude_premium_units():
    """§6.3 must compare EM and historical moves in the same unit (% of spot)."""
    res = VolatilityEngine.evaluate_ticker(**_recommend_kwargs())
    m = res["metrics"]
    # EM = $6.885 on a $150 spot → 4.59% vs hist 3.0% × 1.25 = 3.75% → premium
    assert abs(m["expected_move_pct"] - 6.885 / 150.0) < 1e-9
    assert m["magnitude_premium_detected"] is True
    assert res["conviction"] == "High"
    # Bigger historical moves (6% mean → 7.5% bar) kill the premium, not the signal
    res = VolatilityEngine.evaluate_ticker(
        **_recommend_kwargs(historical_moves=[0.06, -0.06, 0.06, -0.06])
    )
    assert res["metrics"]["magnitude_premium_detected"] is False
    assert res["signal"] == "Recommend"
    assert res["conviction"] == "Standard"
    # No spot → cannot assess premium; must not blow up
    res = VolatilityEngine.evaluate_ticker(**_recommend_kwargs(spot_price=0.0))
    assert res["metrics"]["magnitude_premium_detected"] is False


def test_calendar_spread_tilt():
    em = 6.885
    flat = VolatilityEngine.build_calendar_spread("T", 150.0, em, historical_drift=0.005)
    assert flat["strike"] == 150.0 and flat["tilt"] == "none"
    up = VolatilityEngine.build_calendar_spread("T", 150.0, em, historical_drift=0.02)
    assert up["strike"] == round(150.0 + 0.5 * em, 2) and up["tilt"] == "up"
    down = VolatilityEngine.build_calendar_spread("T", 150.0, em, historical_drift=-0.02)
    assert down["strike"] == round(150.0 - 0.5 * em, 2) and down["tilt"] == "down"
    legs = up["legs"]
    assert legs[0]["action"] == "SELL" and legs[1]["action"] == "BUY"
    assert legs[0]["strike"] == legs[1]["strike"]


def test_exit_plan():
    plan = VolatilityEngine.build_exit_plan(entry_debit=3.40, iv_near=0.85, iv_45=0.55)
    assert abs(plan["take_profit_low_value"] - 3.40 * 1.25) < 1e-9
    assert abs(plan["take_profit_high_value"] - 3.40 * 1.35) < 1e-9
    assert abs(plan["expected_iv_crush"] - 0.30) < 1e-9
    # 80% of the 30-pt crush from 85% → 61%
    assert abs(plan["velocity_iv_level"] - 0.61) < 1e-9
    assert plan["max_loss"] == 3.40


def test_position_plan():
    plan = VolatilityEngine.build_position_plan(100_000.0, debit_per_spread=3.40)
    assert plan["max_debit_allocation"] == 6_000.0
    # Kelly suggestion ≈ $3,244 → 9 contracts at $340; cap $6,000 → 17
    assert plan["contracts_suggested"] == 9
    assert plan["contracts_max"] == 17
    assert plan["estimated_cost_max"] <= plan["max_debit_allocation"]
    assert plan["estimated_cost_suggested"] <= plan["suggested_debit_allocation"]
    zero = VolatilityEngine.build_position_plan(100_000.0, debit_per_spread=0.0)
    assert zero["contracts_max"] == 0 and zero["contracts_suggested"] == 0


def test_bs_call():
    # At-the-money call with vol grows with T; intrinsic at expiry
    assert bs_call(100, 100, 0.0, 0.5) == 0.0
    assert bs_call(110, 100, 0.0, 0.5) == 10.0
    c_near = bs_call(100, 100, 7 / 365, 0.85)
    c_far = bs_call(100, 100, 37 / 365, 0.55)
    assert 0 < c_near < c_far  # calendar at these IVs is a valid debit


def test_monte_carlo():
    mc = VolatilityEngine.run_monte_carlo(100_000.0, n_trades=200, n_paths=300, seed=11)
    assert mc["p5"].shape == (201,) and mc["p50"].shape == (201,) and mc["p95"].shape == (201,)
    assert mc["p5"][0] == mc["p95"][0] == 100_000.0
    assert np.all(mc["p5"] <= mc["p50"] + 1e-9) and np.all(mc["p50"] <= mc["p95"] + 1e-9)
    assert 0.0 <= mc["risk_of_ruin"] <= 1.0
    # Positive-edge strategy at 6% sizing should compound upward in the median
    assert mc["median_final_equity"] > 100_000.0
    # Reproducible under the same seed
    mc2 = VolatilityEngine.run_monte_carlo(100_000.0, n_trades=200, n_paths=300, seed=11)
    assert mc["median_final_equity"] == mc2["median_final_equity"]


def test_backtest():
    res = run_backtest(cfg=BacktestConfig(n_events=800, seed=3))
    assert res["n_trades"] > 500
    assert 0.4 < res["win_rate"] < 0.9
    assert res["avg_win"] > 0 and res["avg_loss"] > 0
    assert res["expectancy"] > 0  # the modeled IV-crush edge is positive
    assert res["equity_curve"][0] == BacktestConfig().starting_capital
    assert len(res["equity_curve"]) == res["n_trades"] + 1
    ek = res["empirical_kelly"]
    assert ek["full_kelly"] > 0
    assert res["prd_reference"]["win_rate"] == HISTORICAL_WIN_RATE


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\nAll {len(fns)} test groups passed.")
