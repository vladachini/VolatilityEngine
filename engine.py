"""
VolatilityEngine — all backend math for the Earnings IV-Crush strategy (PRD v3.1).

Implements:
  §2   Core metrics: term-structure slope, IV/RV ratio, 30-day ADV.
  §3   Deterministic signal routing (Recommend / Consider / Avoid).
  §4   Long Calendar Spread structuring.
  §5   10% fractional Kelly sizing + 6% max-debit cap.
  §6.1 IV Percentile (252-day window).
  §6.2 Expected Move from the front-week ATM straddle.
  §6.3 Earnings Magnitude filter (EM% vs 1.25x historical mean move).
  §6.4 Multi-strike vega tilt on directional drift.
  §6.5 Dynamic exit protocols (take-profit band + velocity exit).
  §7   Monte Carlo validation (fan chart percentiles + Risk of Ruin).

Sign convention: backwardation (the IV-crush setup) means the front-week IV is
ELEVATED relative to the 45-day IV, i.e. Slope = IV_near - IV_45 > 0. The PRD
text labeled this "negative slope"; the math here uses the corrected gate.
"""
from math import erf, exp, floor, log, sqrt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ==========================================
# STRATEGY CONSTANTS (PRD v3.1, 72.5k-event backtest)
# ==========================================
HISTORICAL_WIN_RATE = 0.62         # p
HISTORICAL_LOSS_RATE = 0.38        # q (1 - p)
HISTORICAL_AVG_WIN = 0.45          # Average gain on debit paid (+45%)
HISTORICAL_AVG_LOSS = 0.35         # Average loss on debit paid (-35%)
MAX_PORTFOLIO_FRACTION = 0.06      # Hard 6% max-debit cap (PRD 5)
KELLY_MULTIPLIER = 0.10            # Fractional Kelly multiplier (PRD 5)
IV_PERCENTILE_FLOOR = 70.0         # Recommend gate (PRD 6.1)
MAGNITUDE_PREMIUM_MULT = 1.25      # EM must exceed hist move x1.25 (PRD 6.3)
TAKE_PROFIT_BAND = (0.25, 0.35)    # Exit alert band on debit (PRD 6.5)
VELOCITY_CRUSH_FRACTION = 0.80     # >=80% of expected crush in 5 min (PRD 6.5)
DRIFT_THRESHOLD = 0.01             # Min |mean quarterly drift| for strike tilt (PRD 6.4)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes European call price. Returns intrinsic value at/after expiry."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


class VolatilityEngine:

    # ------------------------------------------------------------------
    # §2 Core metrics
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_term_structure_slope(iv_near: float, iv_45: float) -> float:
        """
        Slope = IV_near - IV_45.
        Slope > 0  -> backwardation  (front IV elevated; the IV-crush setup we want).
        Slope <= 0 -> contango/flat  (no front-week premium to harvest; avoid).
        """
        return iv_near - iv_45

    @staticmethod
    def calculate_iv_rv_ratio(iv_30: float, rv_30: float) -> float:
        """Ratio = IV_30 / RV_30. > 1.0 means options are priced rich vs realized."""
        if rv_30 == 0:
            return 0.0
        return iv_30 / rv_30

    # ------------------------------------------------------------------
    # §6.1 IV Percentile
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_iv_percentile(current_iv: float, historical_iv_series: pd.Series) -> float:
        """(Days in the trailing 252 where IV_hist < IV_current / window) * 100."""
        window = historical_iv_series.tail(252)
        if len(window) == 0:
            return 0.0
        days_below = np.sum(window < current_iv)
        return float((days_below / len(window)) * 100)

    # ------------------------------------------------------------------
    # §6.2 Expected Move
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_expected_move(atm_call_price: float, atm_put_price: float) -> float:
        """1-sigma Expected Move in dollars: EM ~= 0.85 * (ATM call + ATM put)."""
        return 0.85 * (atm_call_price + atm_put_price)

    # ------------------------------------------------------------------
    # §5 Sizing
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_position_sizing(total_portfolio_value: float) -> float:
        """Max Debit Allocation = Total Portfolio Value * 0.06 (hard cap)."""
        return total_portfolio_value * MAX_PORTFOLIO_FRACTION

    @staticmethod
    def calculate_kelly_fraction(
        win_rate: float = HISTORICAL_WIN_RATE,
        avg_win: float = HISTORICAL_AVG_WIN,
        avg_loss: float = HISTORICAL_AVG_LOSS,
        fractional_multiplier: float = KELLY_MULTIPLIER,
    ) -> Dict[str, float]:
        """
        f* = p - q/b, b = avg_win / avg_loss.
        f_applied = fractional_multiplier * f*  (0.10 -> ~3.25% with PRD stats).
        """
        q = 1.0 - win_rate
        b = avg_win / avg_loss if avg_loss != 0 else 0.0
        full_kelly = win_rate - (q / b) if b != 0 else 0.0
        return {
            "full_kelly": full_kelly,
            "fractional_kelly": fractional_multiplier * full_kelly,
            "win_loss_ratio_b": b,
        }

    @staticmethod
    def estimate_calendar_debit(
        spot_price: float,
        strike: float,
        iv_near: float,
        iv_45: float,
        t_near_days: float = 7.0,
        t_far_days: float = 37.0,
    ) -> float:
        """
        Model-based estimate of the calendar debit per share:
        BS(back leg @ iv_45) - BS(front leg @ iv_near). Floored at 0 — a
        non-positive value means the structure is not a valid debit calendar.
        """
        if spot_price <= 0 or strike <= 0:
            return 0.0
        debit = bs_call(spot_price, strike, t_far_days / 365.0, iv_45) - bs_call(
            spot_price, strike, t_near_days / 365.0, iv_near
        )
        return max(0.0, debit)

    @classmethod
    def build_position_plan(
        cls,
        total_portfolio_value: float,
        debit_per_spread: float,
        contract_multiplier: int = 100,
        win_rate: float = HISTORICAL_WIN_RATE,
        avg_win: float = HISTORICAL_AVG_WIN,
        avg_loss: float = HISTORICAL_AVG_LOSS,
    ) -> Dict[str, Any]:
        """
        Translate the PRD's two sizing numbers into an executable contract count:
          * suggested = applied (10%) Kelly fraction of portfolio  (~3.25%)
          * hard cap  = 6% of portfolio (never exceed, PRD 5)
        Contracts are floored so neither dollar bound is ever breached.
        """
        kelly = cls.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        max_debit = cls.calculate_position_sizing(total_portfolio_value)
        suggested_debit = total_portfolio_value * kelly["fractional_kelly"]
        cost_per_spread = debit_per_spread * contract_multiplier
        if cost_per_spread > 0:
            contracts_suggested = int(floor(suggested_debit / cost_per_spread))
            contracts_max = int(floor(max_debit / cost_per_spread))
        else:
            contracts_suggested = contracts_max = 0
        return {
            "max_debit_allocation": max_debit,
            "suggested_debit_allocation": suggested_debit,
            "full_kelly": kelly["full_kelly"],
            "fractional_kelly": kelly["fractional_kelly"],
            "cost_per_spread": cost_per_spread,
            "contracts_suggested": contracts_suggested,
            "contracts_max": contracts_max,
            "estimated_cost_suggested": contracts_suggested * cost_per_spread,
            "estimated_cost_max": contracts_max * cost_per_spread,
        }

    # ------------------------------------------------------------------
    # §4 + §6.4 Trade structuring
    # ------------------------------------------------------------------
    @staticmethod
    def build_calendar_spread(
        ticker: str,
        spot_price: float,
        expected_move: float,
        historical_drift: float = 0.0,
        drift_threshold: float = DRIFT_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Long Calendar Spread block for a 'Recommend' ticker (PRD 4 & 6.4).
          Leg 1 (Short): sell ATM expiring nearest earnings.
          Leg 2 (Long):  buy ATM (same strike) ~30 days after Leg 1.
        Multi-strike vega tilt: when |mean quarterly drift| exceeds the threshold,
        shift the strike 0.5*EM in the drift direction so the spread's tent sits
        where the underlying tends to land, lowering the max-risk debit. (The PRD
        spells out the upward case; the downward tilt is the symmetric extension.)
        """
        strike = round(spot_price, 2)
        tilt = "none"
        rationale = "ATM calendar (no significant directional drift detected)."
        if expected_move > 0 and historical_drift > drift_threshold:
            strike = round(spot_price + 0.5 * expected_move, 2)
            tilt = "up"
            rationale = (
                f"Upward historical drift ({historical_drift:+.2%}) detected; "
                f"strike shifted +0.5*EM above spot to lower debit."
            )
        elif expected_move > 0 and historical_drift < -drift_threshold:
            strike = round(spot_price - 0.5 * expected_move, 2)
            tilt = "down"
            rationale = (
                f"Downward historical drift ({historical_drift:+.2%}) detected; "
                f"strike shifted -0.5*EM below spot to sit under the drift."
            )
        return {
            "ticker": ticker,
            "strike": strike,
            "tilt": tilt,
            "legs": [
                {
                    "leg": "Leg 1 (Short)",
                    "action": "SELL",
                    "strike": strike,
                    "expiry": "Front-week (nearest earnings)",
                },
                {
                    "leg": "Leg 2 (Long)",
                    "action": "BUY",
                    "strike": strike,
                    "expiry": "~30 days after Leg 1",
                },
            ],
            "rationale": rationale,
        }

    # ------------------------------------------------------------------
    # §6.5 Dynamic exit protocols
    # ------------------------------------------------------------------
    @staticmethod
    def build_exit_plan(
        entry_debit: float,
        iv_near: float,
        iv_45: float,
        take_profit_band: tuple = TAKE_PROFIT_BAND,
        velocity_fraction: float = VELOCITY_CRUSH_FRACTION,
    ) -> Dict[str, Any]:
        """
        Quantifies PRD 6.5 into actionable levels:
          * Take-profit alert: exit when spread value reaches +25%..+35% on debit.
          * Velocity exit: expected crush is the front IV collapsing to the back
            level; if >=80% of that collapse prints in the first 5 minutes after
            the open, exit immediately to dodge intraday directional risk.
          * Max loss on a long calendar = the debit paid.
        """
        tp_low, tp_high = take_profit_band
        expected_crush = max(0.0, iv_near - iv_45)
        velocity_iv_level = iv_near - velocity_fraction * expected_crush
        return {
            "entry_debit": entry_debit,
            "take_profit_low_pct": tp_low,
            "take_profit_high_pct": tp_high,
            "take_profit_low_value": entry_debit * (1.0 + tp_low),
            "take_profit_high_value": entry_debit * (1.0 + tp_high),
            "expected_iv_crush": expected_crush,
            "velocity_fraction": velocity_fraction,
            "velocity_iv_level": velocity_iv_level,
            "max_loss": entry_debit,
        }

    # ------------------------------------------------------------------
    # §7 Monte Carlo validation
    # ------------------------------------------------------------------
    @staticmethod
    def run_monte_carlo(
        starting_capital: float,
        win_rate: float = HISTORICAL_WIN_RATE,
        avg_win: float = HISTORICAL_AVG_WIN,
        avg_loss: float = HISTORICAL_AVG_LOSS,
        sizing_fraction: float = MAX_PORTFOLIO_FRACTION,
        n_trades: int = 500,
        n_paths: int = 1000,
        ruin_drawdown: float = 0.50,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Simulates n_trades sequential trades across n_paths parallel paths, risking
        `sizing_fraction` of current equity per trade (debit at risk). On a win the
        debit returns avg_win; on a loss it returns -avg_loss.
        Returns 5th/50th/95th percentile equity curves, Risk of Ruin (probability a
        path ever draws down >= `ruin_drawdown` from its peak), and the probability
        of finishing below starting capital.
        """
        rng = np.random.default_rng(seed)
        equity = np.full(n_paths, float(starting_capital))
        peak = equity.copy()
        ruined = np.zeros(n_paths, dtype=bool)
        curves = np.empty((n_trades + 1, n_paths), dtype=float)
        curves[0] = equity

        for t in range(1, n_trades + 1):
            wins = rng.random(n_paths) < win_rate
            debit = equity * sizing_fraction
            pnl = np.where(wins, debit * avg_win, -debit * avg_loss)
            equity = equity + pnl
            peak = np.maximum(peak, equity)
            ruined |= equity <= peak * (1.0 - ruin_drawdown)
            curves[t] = equity

        p5 = np.percentile(curves, 5, axis=1)
        p50 = np.percentile(curves, 50, axis=1)
        p95 = np.percentile(curves, 95, axis=1)
        return {
            "trade_index": np.arange(n_trades + 1),
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "risk_of_ruin": float(np.mean(ruined)),
            "median_final_equity": float(np.median(equity)),
            "prob_below_start": float(np.mean(equity < starting_capital)),
            "n_paths": n_paths,
            "n_trades": n_trades,
        }

    # ------------------------------------------------------------------
    # §3 + §6 The recommendation engine
    # ------------------------------------------------------------------
    @classmethod
    def evaluate_ticker(
        cls,
        iv_near: float,
        iv_45: float,
        iv_30: float,
        rv_30: float,
        avg_30day_volume: int,
        historical_iv_series: pd.Series,
        atm_call_price: float,
        atm_put_price: float,
        historical_moves: list,
        spot_price: float = 0.0,
        vol_threshold: int = 1_000_000,
        iv_rv_threshold: float = 1.2,
    ) -> Dict[str, Any]:
        """
        Main recommendation engine: PRD §3 deterministic gate + §6 quant filters.

        Routing (corrected sign convention — backwardation is slope > 0):
          Avoid      slope <= 0 (contango/flat: the IV-crush edge does not exist).
          Recommend  slope > 0 AND ADV > threshold AND IV/RV > threshold
                     AND IV percentile >= 70.
          Consider   slope > 0 but at least one other gate fails.

        Conviction (PRD 6.3 target condition, advisory on top of the §3 gate):
          High when, additionally, the straddle-implied Expected Move is priced
          >= 1.25x the historical mean earnings move — both as % of spot.
        """
        # 1. Core metrics (§2)
        slope = cls.calculate_term_structure_slope(iv_near, iv_45)
        ratio = cls.calculate_iv_rv_ratio(iv_30, rv_30)

        # 2. Advanced filters (§6)
        iv_percentile = cls.calculate_iv_percentile(iv_30, historical_iv_series)
        expected_move = cls.calculate_expected_move(atm_call_price, atm_put_price)
        # EM is in dollars; historical moves are fractional returns. Compare them
        # in the same unit (% of spot) — comparing dollars to fractions made the
        # magnitude filter pass vacuously.
        expected_move_pct = expected_move / spot_price if spot_price > 0 else 0.0
        mean_hist_move = float(np.mean(np.abs(historical_moves))) if len(historical_moves) else 0.0
        magnitude_premium = (
            expected_move_pct > mean_hist_move * MAGNITUDE_PREMIUM_MULT
            if (mean_hist_move > 0 and spot_price > 0)
            else False
        )

        # 3. Deterministic signal routing (§3)
        gates = {
            "backwardation": slope > 0,
            "liquidity": avg_30day_volume > vol_threshold,
            "iv_rv": ratio > iv_rv_threshold,
            "iv_percentile": iv_percentile >= IV_PERCENTILE_FLOOR,
        }
        if not gates["backwardation"]:
            signal = "Avoid"
            reason = ("Term structure is flat or in contango (front IV not elevated). "
                      "IV-crush edge does not exist.")
        elif all(gates.values()):
            signal = "Recommend"
            reason = ("All structural conditions met: backwardation, liquidity, "
                      "overpriced IV, and IV at a historical extreme.")
        else:
            signal = "Consider"
            failed = [k for k, v in gates.items() if not v]
            reason = ("Term structure is in backwardation, but the setup fails: "
                      + ", ".join(failed) + ".")

        conviction = "High" if (signal == "Recommend" and magnitude_premium) else (
            "Standard" if signal == "Recommend" else None
        )

        # Per-gate breakdown for the UI (magnitude is advisory, not a §3 gate).
        checks: List[Dict[str, Any]] = [
            {"label": "Backwardation", "required": True, "passed": gates["backwardation"],
             "detail": f"Slope {slope:+.4f} (need > 0)"},
            {"label": "Liquidity", "required": True, "passed": gates["liquidity"],
             "detail": f"ADV {avg_30day_volume:,} (need > {vol_threshold:,})"},
            {"label": "IV / RV", "required": True, "passed": gates["iv_rv"],
             "detail": f"{ratio:.2f} (need > {iv_rv_threshold:.2f})"},
            {"label": "IV percentile", "required": True, "passed": gates["iv_percentile"],
             "detail": f"{iv_percentile:.0f}% (need ≥ {IV_PERCENTILE_FLOOR:.0f}%)"},
            {"label": "Magnitude premium", "required": False, "passed": magnitude_premium,
             "detail": (f"EM {expected_move_pct:.1%} vs hist "
                        f"{mean_hist_move * MAGNITUDE_PREMIUM_MULT:.1%} (1.25×)")},
        ]

        return {
            "signal": signal,
            "conviction": conviction,
            "reason": reason,
            "checks": checks,
            "metrics": {
                "term_structure_slope": slope,
                "iv_rv_ratio": ratio,
                "iv_percentile": iv_percentile,
                "expected_move_dollars": expected_move,
                "expected_move_pct": expected_move_pct,
                "historical_move_mean": mean_hist_move,
                "magnitude_premium_detected": magnitude_premium,
                "avg_30day_volume": avg_30day_volume,
            },
        }
