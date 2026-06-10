import numpy as np
import pandas as pd
from typing import Dict, Any

# ==========================================
# CONSTANTS & STRATEGY CONSTANTS (PRD v3.0)
# ==========================================
HISTORICAL_WIN_RATE = 0.62         # p
HISTORICAL_LOSS_RATE = 0.38        # q (1 - p)
HISTORICAL_AVG_WIN = 0.45          # Average gain on debit paid (+45%)[cite: 5]
HISTORICAL_AVG_LOSS = 0.35         # Average loss on debit paid (-35%)[cite: 5]
MAX_PORTFOLIO_FRACTION = 0.06      # Capped 10% Kelly strategy assignment (6%)[cite: 5]

class VolatilityEngine:
    
    @staticmethod
    def calculate_term_structure_slope(iv_near: float, iv_45: float) -> float:
        """
        Calculates the difference in implied volatility between near-term
        expiration and the 45+ day expiration.
        Formula: Slope = IV_near - IV_45

        Slope > 0  -> backwardation  (front IV elevated; the IV-crush setup we want).
        Slope <= 0 -> contango/flat  (no front-week premium to harvest; avoid).
        """
        return iv_near - iv_45

    @staticmethod
    def calculate_iv_rv_ratio(iv_30: float, rv_30: float) -> float:
        """
        Compares forward-looking 30-day implied volatility to backward-looking
        30-day realized volatility to determine premium overpricing.
        Formula: Ratio = IV_30 / RV_30[cite: 5]
        """
        if rv_30 == 0:
            return 0.0
        return iv_30 / rv_30

    @staticmethod
    def calculate_iv_percentile(current_iv: float, historical_iv_series: pd.Series) -> float:
        """
        Calculates the IV Percentile over a standard rolling trading year (252 days).
        Formula: (Days where IV_current > IV_historical / 252) * 100[cite: 5]
        """
        # Ensure we are looking strictly at a 252-day window
        window = historical_iv_series.tail(252)
        if len(window) == 0:
            return 0.0
        
        days_below = np.sum(window < current_iv)
        return float((days_below / len(window)) * 100)

    @staticmethod
    def calculate_expected_move(atm_call_price: float, atm_put_price: float) -> float:
        """
        Floor heuristic for the 1-standard-deviation Expected Move (EM) 
        derived from the front-week At-The-Money straddle.
        Formula: EM ≈ 0.85 * (Price_ATM_Call + Price_ATM_Put)[cite: 5]
        """
        return 0.85 * (atm_call_price + atm_put_price)

    @staticmethod
    def calculate_position_sizing(total_portfolio_value: float) -> float:
        """
        Strictly limits trade allocation based on the 10% Kelly Criterion backtest constraints.
        Formula: Max Debit Allocation = Total Portfolio Value * 0.06[cite: 5]
        """
        return total_portfolio_value * MAX_PORTFOLIO_FRACTION

    @staticmethod
    def calculate_kelly_fraction(
        win_rate: float = HISTORICAL_WIN_RATE,
        avg_win: float = HISTORICAL_AVG_WIN,
        avg_loss: float = HISTORICAL_AVG_LOSS,
        fractional_multiplier: float = 0.10,
    ) -> Dict[str, float]:
        """
        Full + fractional Kelly from the 72.5k-event backtest (PRD 5).
        f* = p - q/b,  where b = avg_win / avg_loss, q = 1 - p.
        f_applied = fractional_multiplier * f*  (the strategy uses 0.10 -> ~3.25%).
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
    def build_calendar_spread(
        ticker: str,
        spot_price: float,
        expected_move: float,
        historical_drift: float = 0.0,
        drift_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Builds the Long Calendar Spread block for a 'Recommend' ticker (PRD 4 & 6.4).
        Leg 1 (Short): sell ATM expiring nearest earnings.
        Leg 2 (Long): buy ATM (same strike) ~30 days after Leg 1.
        Multi-strike Vega tilt: if historical drift exceeds the threshold, shift the
        strike +0.5*EM above spot to lower max-risk debit.
        """
        strike = round(spot_price, 2)
        rationale = "ATM calendar (no significant directional drift detected)."
        if historical_drift > drift_threshold and expected_move > 0:
            strike = round(spot_price + 0.5 * expected_move, 2)
            rationale = (
                f"Upward historical drift ({historical_drift:+.2%}) detected; "
                f"strike shifted +0.5*EM above spot to lower debit."
            )
        return {
            "ticker": ticker,
            "strike": strike,
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
        Strategy validation via Monte Carlo (PRD 7).
        Simulates n_trades sequential trades across n_paths parallel paths, risking
        `sizing_fraction` of current equity per trade (debit at risk). On a win the
        debit returns avg_win; on a loss it returns -avg_loss.
        Returns percentile equity curves (5th/50th/95th) and Risk of Ruin: the
        probability that a path ever draws down >= `ruin_drawdown` from its peak.
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
            "n_paths": n_paths,
            "n_trades": n_trades,
        }

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
        vol_threshold: int = 1000000,
        iv_rv_threshold: float = 1.2
    ) -> Dict[str, Any]:
        """
        The Main Recommendation Engine. Combines baseline structural rules 
        with Advanced v3.0 Quantitative Filters to output explicit trade signals.[cite: 5]
        """
        # 1. Compute Base Metrics[cite: 5]
        slope = cls.calculate_term_structure_slope(iv_near, iv_45)
        ratio = cls.calculate_iv_rv_ratio(iv_30, rv_30)
        
        # 2. Compute Advanced Upgrades[cite: 5]
        iv_percentile = cls.calculate_iv_percentile(iv_30, historical_iv_series)
        expected_move = cls.calculate_expected_move(atm_call_price, atm_put_price)
        
        mean_hist_move = np.mean(np.abs(historical_moves)) if historical_moves else 0.0
        earnings_magnitude_premium = expected_move > (mean_hist_move * 1.25) if mean_hist_move > 0 else False
        
        # 3. Deterministic Signal Routing Gate
        # Base requirements: backwardation (rich front-week IV), sufficient volume,
        # overpricing ratio. Backwardation means IV_near > IV_45, i.e. slope > 0 with
        # Slope = IV_near - IV_45 -- the front IV is elevated and primed to crush.
        base_recommend = (slope > 0) and (avg_30day_volume > vol_threshold) and (ratio > iv_rv_threshold)

        if slope <= 0:
            signal = "Avoid"
            reason = "Term structure is flat or in contango (front IV not elevated). IV-crush edge does not exist."
        elif base_recommend and (iv_percentile >= 70.0):
            signal = "Recommend"
            reason = "All structural conditions met: backwardation, liquidity, overpriced IV, and IV at a historical extreme."
        else:
            signal = "Consider"
            reason = "Term structure is in backwardation, but asset fails to clear either volume, IV/RV, or IV Percentile bounds."

        # 4. Compile Output Structuring[cite: 5]
        return {
            "signal": signal,
            "reason": reason,
            "metrics": {
                "term_structure_slope": slope,
                "iv_rv_ratio": ratio,
                "iv_percentile": iv_percentile,
                "expected_move_dollars": expected_move,
                "historical_move_mean": mean_hist_move,
                "magnitude_premium_detected": earnings_magnitude_premium,
                "avg_30day_volume": avg_30day_volume
            }
        }