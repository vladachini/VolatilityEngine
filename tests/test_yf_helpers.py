"""
Offline tests for the Yahoo provider's pure helpers (no network, no yfinance).

Run with:
    python tests/test_yf_helpers.py
    python -m pytest tests/ -q           # if pytest is installed
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider_yf import (  # noqa: E402
    atm_from_chain,
    earnings_moves,
    interp_iv,
    realized_vol_series,
)


def test_realized_vol_series():
    # Constant +1%/day log-return -> zero rolling std -> zero vol
    closes = pd.Series(100.0 * np.exp(0.01 * np.arange(60)))
    rv = realized_vol_series(closes, window=30).dropna()
    assert len(rv) == 30 and np.allclose(rv, 0.0, atol=1e-12)
    # Alternating ±1% has a known daily std -> annualized by sqrt(252)
    rets = np.tile([0.01, -0.01], 30)
    closes = pd.Series(100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)])))
    rv = realized_vol_series(closes, window=30).dropna()
    expected = np.std([0.01, -0.01] * 15, ddof=1) * math.sqrt(252)
    assert abs(rv.iloc[-1] - expected) < 1e-9


def _chain(strikes, ivs, bids, asks, lasts):
    return pd.DataFrame({
        "strike": strikes, "impliedVolatility": ivs,
        "bid": bids, "ask": asks, "lastPrice": lasts,
    })


def test_atm_from_chain():
    calls = _chain([95, 100, 105], [0.9, 0.8, 0.7], [4.0, 3.0, 2.0],
                   [4.4, 3.4, 2.4], [4.1, 3.1, 2.1])
    puts = _chain([95, 100, 105], [0.95, 0.84, 0.75], [3.0, 2.5, 2.0],
                  [3.4, 2.9, 2.4], [3.2, 2.7, 2.2])
    iv, strike, c_px, p_px = atm_from_chain(calls, puts, spot=101.0)
    assert strike == 100.0
    assert abs(iv - (0.8 + 0.84) / 2) < 1e-12      # mean of call/put IV
    assert abs(c_px - 3.2) < 1e-12                  # bid/ask mid
    assert abs(p_px - 2.7) < 1e-12
    # No bid/ask -> falls back to lastPrice; zero/NaN IV on one side -> other side
    calls2 = _chain([100], [0.0], [0.0], [0.0], [3.1])
    puts2 = _chain([100], [0.84], [0.0], [0.0], [2.7])
    iv2, _, c2, p2 = atm_from_chain(calls2, puts2, spot=100.0)
    assert abs(iv2 - 0.84) < 1e-12 and c2 == 3.1 and p2 == 2.7
    assert atm_from_chain(calls.iloc[0:0], puts.iloc[0:0], 100.0)[0] is None


def test_interp_iv():
    pts = [(7, 0.85), (37, 0.55), (65, 0.50)]
    assert abs(interp_iv(pts, 7) - 0.85) < 1e-12
    assert abs(interp_iv(pts, 22) - 0.70) < 1e-12     # midpoint of 7d..37d
    assert abs(interp_iv(pts, 3) - 0.85) < 1e-12      # clamps below
    assert abs(interp_iv(pts, 90) - 0.50) < 1e-12     # clamps above
    assert interp_iv([(7, 0.0)], 30) is None          # no valid points
    assert abs(interp_iv([(30, 0.6)], 45) - 0.6) < 1e-12


def test_earnings_moves():
    days = pd.bdate_range("2025-01-02", periods=120)
    closes = pd.Series(100.0, index=days)
    closes.iloc[:] = 100.0
    # Earnings after the close on day 49 -> reaction is day 50's close (+8%)
    closes.iloc[50:] = 108.0
    amc = days[49] + pd.Timedelta(hours=20)
    # Earnings before the open on day 80 -> reaction is day 80's close (-5% from 108)
    closes.iloc[80:] = 108.0 * 0.95
    bmo = days[80] + pd.Timedelta(hours=8)
    moves = earnings_moves(closes, [amc, bmo], n=8)
    assert len(moves) == 2
    assert abs(moves[0] - (-0.05)) < 1e-9     # most recent first (BMO event)
    assert abs(moves[1] - 0.08) < 1e-9        # AMC event
    # Timestamps outside the price history are skipped
    assert earnings_moves(closes, [days[0] - pd.Timedelta(days=30)], n=8) == []
    # tz-aware inputs are handled
    from datetime import timezone
    moves_tz = earnings_moves(closes, [amc.tz_localize(timezone.utc)], n=8)
    assert len(moves_tz) == 1 and abs(moves_tz[0] - 0.08) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\nAll {len(fns)} test groups passed.")
