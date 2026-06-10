"""
Strategy backtest for the earnings IV-crush Long Calendar Spread (PRD §5 & §7).

Why this exists
---------------
The PRD hard-codes win rate (62%), avg win (+45%) and avg loss (-35%) from a
72,500-event backtest and feeds them into the Kelly sizing and Monte Carlo. This
module reconstructs that kind of backtest so those numbers are *derived*, not
asserted: it prices an actual ATM calendar spread with Black-Scholes before and
after each earnings event and measures the P&L distribution, then reports the
empirical stats that should drive sizing.

Model
-----
Long calendar at strike K = spot:
    entry debit  = C(S0, K, T_far, iv_far)        - C(S0, K, T_near, iv_near)
    exit value   = C(S1, K, T_far-h, iv_far_post) - C(S1, K, T_near-h, iv_near_post)
    pnl_pct      = (exit_value - entry_debit) / entry_debit
The edge comes from two real effects after earnings:
  * IV crush — the elevated front-week IV collapses toward the back-month level,
    which is pure profit for the calendar (we are net long vega on the back leg
    and the short front leg loses its inflated premium), and
  * the vol-risk premium — realized moves are, on average, a bit smaller than the
    move implied by the straddle.
Large realized moves push the underlying off the strike and turn the tent-shaped
payoff into a loss, which is what produces the loss tail.

Real data: pass `run_backtest(events=df)` where df has columns
[iv_near, iv_far, iv_near_post, iv_far_post, realized_move] (move as a signed
fraction, e.g. 0.04 = +4%). Otherwise a synthetic universe is generated.

This P&L model uses the real-world elevated-front-week IV (backwardation), which now
matches the engine's signal gate: VolatilityEngine.evaluate_ticker recommends when
Slope = IV_near - IV_45 > 0. The backtest validates trade economics + sizing; signal
gating is handled in the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, log, sqrt
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from engine import (
    HISTORICAL_AVG_LOSS,
    HISTORICAL_AVG_WIN,
    HISTORICAL_WIN_RATE,
    MAX_PORTFOLIO_FRACTION,
    VolatilityEngine,
)

TRADING_DAYS = 252.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes European call price. Returns intrinsic value at/after expiry."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


def simulate_calendar_trade(
    iv_near: float,
    iv_far: float,
    iv_near_post: float,
    iv_far_post: float,
    realized_move: float,
    spot: float = 100.0,
    t_near_days: float = 7.0,
    t_far_days: float = 37.0,
    hold_days: float = 1.0,
    r: float = 0.0,
    min_debit_frac: float = 0.004,
) -> Optional[float]:
    """
    P&L of one ATM long calendar spread held through earnings, as a fraction of the
    debit paid. Returns None if the structure is not a tradeable long calendar:
    a non-positive debit, or a debit so small (< `min_debit_frac` of spot) that it
    implies unrealistic leverage no desk would actually put on.
    """
    K = spot
    T_near, T_far = t_near_days / 365.0, t_far_days / 365.0
    entry_debit = bs_call(spot, K, T_far, iv_far, r) - bs_call(spot, K, T_near, iv_near, r)
    if entry_debit <= min_debit_frac * spot:
        return None

    S1 = spot * (1.0 + realized_move)
    T_near2 = max(t_near_days - hold_days, 1e-6) / 365.0
    T_far2 = max(t_far_days - hold_days, 1e-6) / 365.0
    exit_value = bs_call(S1, K, T_far2, iv_far_post, r) - bs_call(S1, K, T_near2, iv_near_post, r)
    return (exit_value - entry_debit) / entry_debit


@dataclass
class BacktestConfig:
    n_events: int = 2000
    starting_capital: float = 100_000.0
    sizing_fraction: float = MAX_PORTFOLIO_FRACTION  # 6% debit per trade
    trades_per_year: int = 50                        # for Sharpe annualization
    ruin_drawdown: float = 0.50
    seed: int = 7
    min_debit_frac: float = 0.004          # skip calendars cheaper than 0.4% of spot
    # synthetic-universe parameters (ignored when real events are supplied)
    front_iv_range: tuple = (0.55, 0.95)   # elevated pre-earnings front-week IV
    back_iv_ratio: tuple = (0.70, 0.92)    # back IV = front IV * U(.) -> elevated but bounded front
    realized_vrp: float = 0.90             # realized move std as fraction of implied (vol-risk premium)
    tail_df: float = 4.0                   # Student-t dof for fat-tailed surprises
    crush_residual: tuple = (0.90, 1.10)   # front IV post = back IV * U(.)  (crush to ~back level)
    back_residual: tuple = (0.92, 1.02)    # back IV post = back IV * U(.)


def generate_synthetic_events(cfg: BacktestConfig) -> pd.DataFrame:
    """Build a realistic universe of earnings events (one calendar trade each)."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_events
    iv_near = rng.uniform(*cfg.front_iv_range, n)
    iv_far = iv_near * rng.uniform(*cfg.back_iv_ratio, n)  # elevated front, bounded gap

    # Front straddle implies a ~1-event expected move; realized is centered smaller
    # (vol-risk premium) with fat tails (occasional large surprises).
    implied_move = 0.85 * (0.80 * iv_near * sqrt(7.0 / 365.0))  # EM as fraction of spot
    t_draw = rng.standard_t(cfg.tail_df, n) / sqrt(cfg.tail_df / (cfg.tail_df - 2.0))
    realized_move = cfg.realized_vrp * implied_move * t_draw

    iv_near_post = iv_far * rng.uniform(*cfg.crush_residual, n)
    iv_far_post = iv_far * rng.uniform(*cfg.back_residual, n)

    return pd.DataFrame(
        {
            "iv_near": iv_near,
            "iv_far": iv_far,
            "iv_near_post": iv_near_post,
            "iv_far_post": iv_far_post,
            "realized_move": realized_move,
            "implied_move": implied_move,
        }
    )


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak)) if len(equity) else 0.0


def run_backtest(
    events: Optional[pd.DataFrame] = None, cfg: Optional[BacktestConfig] = None
) -> Dict[str, Any]:
    """
    Run the calendar-spread backtest and return P&L distribution, equity curve,
    and the empirical stats that should drive Kelly sizing + Monte Carlo.
    """
    cfg = cfg or BacktestConfig()
    if events is None:
        events = generate_synthetic_events(cfg)

    pnls = []
    for row in events.itertuples(index=False):
        pnl = simulate_calendar_trade(
            iv_near=row.iv_near, iv_far=row.iv_far,
            iv_near_post=row.iv_near_post, iv_far_post=row.iv_far_post,
            realized_move=row.realized_move, min_debit_frac=cfg.min_debit_frac,
        )
        if pnl is not None:
            pnls.append(pnl)
    pnl = np.array(pnls, dtype=float)
    if pnl.size == 0:
        raise RuntimeError("No valid calendar trades produced (all entries were credits).")

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = float(len(wins) / len(pnl))
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(-losses.mean()) if losses.size else 0.0  # positive magnitude
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    expectancy = float(pnl.mean())

    # Equity curve under fractional sizing: each trade risks `sizing_fraction` of
    # current equity as debit; pnl is a return on that debit.
    f = cfg.sizing_fraction
    step_returns = f * pnl
    equity = cfg.starting_capital * np.cumprod(1.0 + step_returns)
    equity = np.insert(equity, 0, cfg.starting_capital)
    total_return = float(equity[-1] / cfg.starting_capital - 1.0)
    max_dd = _max_drawdown(equity)
    # The events are a cross-sectional sample, not a 2000-trade timeline; annualize
    # to a CAGR using the assumed trades/year so the growth figure is interpretable.
    n_years = pnl.size / cfg.trades_per_year
    cagr = float((equity[-1] / equity[0]) ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0

    # Sharpe on per-trade portfolio returns, annualized by trades/year.
    sr = step_returns
    sharpe_trade = float(sr.mean() / sr.std(ddof=1)) if sr.std(ddof=1) > 0 else 0.0
    sharpe_annual = sharpe_trade * sqrt(cfg.trades_per_year)

    empirical_kelly = VolatilityEngine.calculate_kelly_fraction(
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss
    )

    return {
        "n_trades": int(pnl.size),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "total_return": total_return,
        "cagr": cagr,
        "n_years": n_years,
        "max_drawdown": max_dd,
        "sharpe_per_trade": sharpe_trade,
        "sharpe_annual": sharpe_annual,
        "empirical_kelly": empirical_kelly,
        "pnl_pct": pnl,
        "equity_curve": equity,
        "prd_reference": {
            "win_rate": HISTORICAL_WIN_RATE,
            "avg_win": HISTORICAL_AVG_WIN,
            "avg_loss": HISTORICAL_AVG_LOSS,
        },
    }


if __name__ == "__main__":  # pragma: no cover - manual run
    res = run_backtest()
    print(f"Trades:          {res['n_trades']}")
    print(f"Win rate:        {res['win_rate']*100:5.1f}%   (PRD 62.0%)")
    print(f"Avg win:         {res['avg_win']*100:5.1f}%   (PRD +45.0%)")
    print(f"Avg loss:        {res['avg_loss']*100:5.1f}%   (PRD -35.0%)")
    print(f"Profit factor:   {res['profit_factor']:.2f}")
    print(f"Expectancy:      {res['expectancy']*100:.2f}% of debit/trade")
    print(f"CAGR:            {res['cagr']*100:.1f}%  (~{res['n_years']:.0f}y @ {BacktestConfig().trades_per_year} trades/yr)")
    print(f"Max drawdown:    {res['max_drawdown']*100:.1f}%")
    print(f"Sharpe (annual): {res['sharpe_annual']:.2f}")
    ek = res["empirical_kelly"]
    print(f"Empirical Kelly: full {ek['full_kelly']*100:.1f}%  applied(10%) {ek['fractional_kelly']*100:.2f}%")
