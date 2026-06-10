"""
Shared demo fixture — the app's "Load demo ticker" setup.

Lives outside app.py so the test suite can assert against the exact same inputs
the demo button seeds (app.py executes Streamlit at import and can't be imported
from tests).
"""
import numpy as np

# A clean "Recommend" setup: backwardation, liquid, rich IV, high percentile,
# and an Expected Move priced > 1.25x the historical mean earnings move.
DEMO = {
    "ticker": "DEMO",
    "portfolio": 100_000.0,
    "spot": 150.0,
    "iv_near": 0.85,      # elevated front-week IV (earnings premium)
    "iv_45": 0.55,        # 45+ day IV -> slope > 0 (backwardation)
    "iv_30": 0.70,
    "rv_30": 0.45,        # IV/RV = 1.55 (> 1.2 threshold)
    "volume": 4_500_000,
    "atm_call": 4.20,
    "atm_put": 3.90,
    "drift": 0.018,       # mild upward historical drift
}

DEFAULT_HIST_MOVES = [0.028, -0.034, 0.025, -0.041, 0.030, -0.022]  # mean |move| ≈ 3%


def demo_iv_history(seed: int = 7) -> list:
    """252-day IV history mostly below the demo's IV_30 -> high percentile."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.45, 0.08, 252), 0.15, 0.95).tolist()
