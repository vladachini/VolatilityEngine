"""
Earnings Volatility (IV Crush) Trading App — Streamlit Frontend
PRD v3.1. Wires the UI to engine.VolatilityEngine.

NOTE: No live market-data provider is wired up yet (PRD 2 lists Tradier/IBKR/
MarketData as future integrations). Until then, metrics are entered manually.
Use the sidebar "Load demo ticker" button to populate a realistic Recommend setup.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from engine import VolatilityEngine, MAX_PORTFOLIO_FRACTION
from backtest import run_backtest, BacktestConfig

st.set_page_config(page_title="Earnings IV Crush Engine", page_icon="📉", layout="wide")

# ------------------------------------------------------------------
# Demo data: a clean "Recommend" setup so the app is testable on first run.
# ------------------------------------------------------------------
DEMO = {
    "ticker": "DEMO",
    "portfolio": 100_000.0,
    "spot": 150.0,
    "iv_near": 0.85,      # elevated front-week IV (earnings premium)
    "iv_45": 0.55,        # 45+ day IV -> slope > 0 (backwardation, the IV-crush setup)
    "iv_30": 0.70,
    "rv_30": 0.45,        # IV/RV = 1.55 (> 1.2 threshold)
    "volume": 4_500_000,
    "atm_call": 4.20,
    "atm_put": 3.90,
    "drift": 0.018,       # mild upward historical drift
}


def _seed_demo():
    for k, v in DEMO.items():
        st.session_state[f"in_{k}"] = v
    # 252-day IV history mostly below current IV_30 -> high percentile
    rng = np.random.default_rng(7)
    hist = np.clip(rng.normal(0.45, 0.08, 252), 0.15, 0.95)
    st.session_state["iv_history"] = hist.tolist()
    # 6 historical earnings moves averaging well below the expected move
    st.session_state["hist_moves"] = [0.04, -0.05, 0.03, -0.06, 0.045, -0.035]


def _get(key, default):
    return st.session_state.get(f"in_{key}", default)


def _seed_from_ibkr(data: dict):
    """Populate the input widgets from an IBKRDataProvider result dict."""
    for k in ("ticker", "spot", "iv_near", "iv_45", "iv_30", "rv_30",
              "volume", "atm_call", "atm_put", "drift"):
        if k in data:
            st.session_state[f"in_{k}"] = data[k]
    if data.get("historical_iv_series"):
        st.session_state["iv_history"] = data["historical_iv_series"]
    if data.get("hist_moves"):
        st.session_state["hist_moves"] = data["hist_moves"]


# ------------------------------------------------------------------
# Sidebar — global inputs (PRD 8)
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Inputs")

source = st.sidebar.radio("Data source", ["Manual / Demo", "IBKR (ib_insync)"], index=0)

if source == "Manual / Demo":
    st.sidebar.button("🎯 Load demo ticker", on_click=_seed_demo, width='stretch')
else:
    with st.sidebar.expander("🔌 IBKR connection", expanded=True):
        ib_symbol = st.text_input("Symbol to fetch", value=_get("ticker", "AAPL")).upper()
        ib_host = st.text_input("Host", value="127.0.0.1")
        ib_port = st.number_input(
            "Port (TWS 7497 paper / 7496 live · GW 4002/4001)", value=7497, step=1
        )
        ib_client = st.number_input("Client ID", value=17, step=1)
        ib_mdt = st.selectbox(
            "Market data type", options=[(3, "Delayed"), (1, "Live"), (4, "Delayed-frozen")],
            format_func=lambda o: o[1], index=0,
        )[0]
        if st.button("📡 Fetch live data", width='stretch'):
            try:
                from data_provider import fetch_ibkr_metrics
                with st.spinner(f"Connecting to IBKR and pulling {ib_symbol}..."):
                    data = fetch_ibkr_metrics(
                        ib_symbol, host=ib_host, port=int(ib_port),
                        client_id=int(ib_client), market_data_type=int(ib_mdt),
                    )
                _seed_from_ibkr(data)
                meta = data.get("_meta", {})
                st.success(
                    f"Fetched {ib_symbol}: ATM {meta.get('atm_strike')} · "
                    f"near {meta.get('near_expiry')} ({meta.get('days_to_near')}d) · "
                    f"45-leg {meta.get('exp_45')} ({meta.get('days_to_45leg')}d)"
                )
                st.rerun()
            except Exception as e:
                st.error(f"IBKR fetch failed: {e}")
                st.caption("Is TWS/IB Gateway running with the API enabled on this port?")

ticker = st.sidebar.text_input("Ticker Symbol", value=_get("ticker", "AAPL")).upper()
portfolio = st.sidebar.number_input(
    "Total Portfolio Value ($)", min_value=0.0, value=float(_get("portfolio", 100_000.0)), step=1000.0
)
spot = st.sidebar.number_input("Current Spot Price ($)", min_value=0.0, value=float(_get("spot", 150.0)), step=0.5)

st.sidebar.markdown("**Signal thresholds**")
vol_threshold = st.sidebar.number_input("Min 30d Avg Volume", min_value=0, value=1_000_000, step=100_000)
iv_rv_threshold = st.sidebar.number_input("Min IV/RV Ratio", min_value=0.0, value=1.2, step=0.05)

st.title("📉 Earnings Volatility — IV Crush Engine")
st.caption("Long Calendar Spread detection · deterministic signal routing · 10% Kelly risk caps (PRD v3.1)")

# ------------------------------------------------------------------
# Metric inputs (would be pulled from a data provider in production)
# ------------------------------------------------------------------
with st.expander("📥 Options-chain & volatility inputs", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        iv_near = st.number_input("IV — near term (front week)", min_value=0.0, value=float(_get("iv_near", 0.85)), step=0.01)
        iv_45 = st.number_input("IV — 45+ days", min_value=0.0, value=float(_get("iv_45", 0.55)), step=0.01)
    with c2:
        iv_30 = st.number_input("IV — 30 day", min_value=0.0, value=float(_get("iv_30", 0.70)), step=0.01)
        rv_30 = st.number_input("RV — 30 day (realized)", min_value=0.0, value=float(_get("rv_30", 0.45)), step=0.01)
    with c3:
        atm_call = st.number_input("ATM Call price ($)", min_value=0.0, value=float(_get("atm_call", 4.20)), step=0.05)
        atm_put = st.number_input("ATM Put price ($)", min_value=0.0, value=float(_get("atm_put", 3.90)), step=0.05)

    c4, c5 = st.columns(2)
    with c4:
        volume = st.number_input("30-day Avg Daily Volume", min_value=0, value=int(_get("volume", 4_500_000)), step=100_000)
    with c5:
        drift = st.number_input("Historical quarterly drift (signed, e.g. 0.02)", value=float(_get("drift", 0.018)), step=0.005, format="%.3f")

# Series-style inputs default to demo if seeded, else synthesized around iv_30.
if "iv_history" in st.session_state:
    iv_history = pd.Series(st.session_state["iv_history"])
else:
    _rng = np.random.default_rng(1)
    iv_history = pd.Series(np.clip(_rng.normal(max(iv_30 * 0.7, 0.1), 0.08, 252), 0.05, 1.5))

hist_moves = st.session_state.get("hist_moves", [0.04, -0.05, 0.03, -0.06, 0.045, -0.035])

# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------
result = VolatilityEngine.evaluate_ticker(
    iv_near=iv_near, iv_45=iv_45, iv_30=iv_30, rv_30=rv_30,
    avg_30day_volume=int(volume), historical_iv_series=iv_history,
    atm_call_price=atm_call, atm_put_price=atm_put, historical_moves=hist_moves,
    vol_threshold=int(vol_threshold), iv_rv_threshold=float(iv_rv_threshold),
)
m = result["metrics"]

# ------------------------------------------------------------------
# Signal banner (PRD 8: green/yellow/red)
# ------------------------------------------------------------------
banner = {
    "Recommend": (st.success, "🟢"),
    "Consider": (st.warning, "🟡"),
    "Avoid": (st.error, "🔴"),
}[result["signal"]]
banner[0](f"{banner[1]} **{result['signal'].upper()} — {ticker}** · {result['reason']}")

# ------------------------------------------------------------------
# Metrics dataframe (PRD 8)
# ------------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.subheader("Core & advanced metrics")
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Term Structure Slope (IV_near − IV_45)",
                "IV / RV Ratio",
                "IV Percentile (252d)",
                "Expected Move ($, 1σ)",
                "Historical Move Mean ($-equiv %)",
                "Earnings Magnitude Premium (EM > hist×1.25)",
                "30-day Avg Daily Volume",
            ],
            "Value": [
                f"{m['term_structure_slope']:+.4f}",
                f"{m['iv_rv_ratio']:.2f}",
                f"{m['iv_percentile']:.1f}%",
                f"${m['expected_move_dollars']:.2f}",
                f"{m['historical_move_mean']:.3f}",
                "✅ Yes" if m["magnitude_premium_detected"] else "❌ No",
                f"{m['avg_30day_volume']:,}",
            ],
        }
    )
    st.dataframe(metrics_df, hide_index=True, width='stretch')

# ------------------------------------------------------------------
# Risk management — 10% Kelly + 6% max debit (PRD 5)
# ------------------------------------------------------------------
with right:
    st.subheader("Position sizing & risk")
    max_debit = VolatilityEngine.calculate_position_sizing(portfolio)
    kelly = VolatilityEngine.calculate_kelly_fraction()
    st.metric("Max Debit Allocation (6% cap)", f"${max_debit:,.2f}")
    kc1, kc2 = st.columns(2)
    kc1.metric("Full Kelly f*", f"{kelly['full_kelly']*100:.2f}%")
    kc2.metric("Applied (10%) Kelly", f"{kelly['fractional_kelly']*100:.2f}%")
    st.warning(f"⚠️ Never exceed **${max_debit:,.2f}** of debit on this trade "
               f"({MAX_PORTFOLIO_FRACTION*100:.0f}% of portfolio).")

# ------------------------------------------------------------------
# Trade structuring — only for Recommend (PRD 4 & 6.4)
# ------------------------------------------------------------------
st.subheader("Proposed trade — Long Calendar Spread")
if result["signal"] == "Recommend":
    spread = VolatilityEngine.build_calendar_spread(
        ticker=ticker, spot_price=spot,
        expected_move=m["expected_move_dollars"], historical_drift=drift,
    )
    st.dataframe(pd.DataFrame(spread["legs"]), hide_index=True, width='stretch')
    st.caption(f"Strike: **${spread['strike']}** · {spread['rationale']}")
else:
    st.info("Trade legs are only generated for **Recommend** signals. "
            "Adjust inputs or thresholds to qualify a setup.")

# ------------------------------------------------------------------
# Backtest — reconstruct the strategy's edge from priced calendar spreads (PRD 5 & 7)
# ------------------------------------------------------------------
st.divider()
st.subheader("📜 Strategy backtest")
st.caption("Prices an ATM calendar spread through each earnings event with Black-Scholes "
           "to *derive* the win rate / avg win / avg loss that drive Kelly sizing — "
           "rather than asserting them.")
bc1, bc2, bc3 = st.columns(3)
bt_events = bc1.slider("Earnings events", 200, 5000, 2000, step=100)
bt_per_year = bc2.slider("Trades / year (for CAGR & Sharpe)", 10, 100, 50, step=5)
run_bt = bc3.button("Run backtest", width='stretch')

if run_bt:
    with st.spinner("Pricing calendar spreads across earnings events..."):
        bt = run_backtest(cfg=BacktestConfig(
            n_events=int(bt_events), starting_capital=portfolio, trades_per_year=int(bt_per_year),
        ))
    st.session_state["bt_result"] = bt

if "bt_result" in st.session_state:
    bt = st.session_state["bt_result"]
    prd = bt["prd_reference"]
    stat_df = pd.DataFrame(
        {
            "Metric": ["Win rate", "Avg win", "Avg loss", "Profit factor",
                       "Expectancy / trade", "CAGR", "Max drawdown", "Sharpe (annual)"],
            "Backtest": [
                f"{bt['win_rate']*100:.1f}%", f"+{bt['avg_win']*100:.1f}%",
                f"-{bt['avg_loss']*100:.1f}%", f"{bt['profit_factor']:.2f}",
                f"{bt['expectancy']*100:.2f}%", f"{bt['cagr']*100:.1f}%",
                f"{bt['max_drawdown']*100:.1f}%", f"{bt['sharpe_annual']:.2f}",
            ],
            "PRD reference": [
                f"{prd['win_rate']*100:.1f}%", f"+{prd['avg_win']*100:.1f}%",
                f"-{prd['avg_loss']*100:.1f}%", "—", "—", "—", "—", "—",
            ],
        }
    )
    bcol, ecol = st.columns([2, 3])
    with bcol:
        st.dataframe(stat_df, hide_index=True, width='stretch')
        ek = bt["empirical_kelly"]
        st.caption(f"Empirical Kelly: full **{ek['full_kelly']*100:.1f}%**, applied 10% → "
                   f"**{ek['fractional_kelly']*100:.2f}%** (vs. {MAX_PORTFOLIO_FRACTION*100:.0f}% sizing cap).")
    with ecol:
        eq = pd.DataFrame({"Equity ($)": bt["equity_curve"]})
        st.line_chart(eq, height=260)
    st.caption(f"{bt['n_trades']:,} valid trades. Backtest stats are independent of the PRD's "
               "hard-coded constants — use them to sanity-check the assumed edge.")

# ------------------------------------------------------------------
# Monte Carlo validation (PRD 7)
# ------------------------------------------------------------------
st.divider()
st.subheader("🎲 Monte Carlo strategy validation")
use_bt = st.checkbox(
    "Use backtest-derived win/loss stats (else PRD constants)",
    value="bt_result" in st.session_state, disabled="bt_result" not in st.session_state,
)
mc1, mc2, mc3 = st.columns(3)
n_trades = mc1.slider("Trades per path", 50, 1000, 500, step=50)
n_paths = mc2.slider("Parallel paths", 100, 5000, 1000, step=100)
run_mc = mc3.button("Run simulation", width='stretch')

if run_mc:
    mc_stats = {}
    if use_bt and "bt_result" in st.session_state:
        bt = st.session_state["bt_result"]
        mc_stats = {"win_rate": bt["win_rate"], "avg_win": bt["avg_win"], "avg_loss": bt["avg_loss"]}
    with st.spinner("Simulating equity paths..."):
        mc = VolatilityEngine.run_monte_carlo(
            starting_capital=portfolio, n_trades=int(n_trades), n_paths=int(n_paths),
            **mc_stats,
        )
    fan = pd.DataFrame(
        {"5th pct": mc["p5"], "50th pct (median)": mc["p50"], "95th pct": mc["p95"]},
        index=mc["trade_index"],
    )
    st.line_chart(fan)
    r1, r2 = st.columns(2)
    r1.metric("Median final equity", f"${mc['median_final_equity']:,.0f}")
    ror = mc["risk_of_ruin"]
    r2.metric("Risk of Ruin (≥50% drawdown)", f"{ror*100:.2f}%",
              delta="acceptable" if ror < 0.05 else "elevated",
              delta_color="normal" if ror < 0.05 else "inverse")

st.caption("Educational tool — not investment advice. Verify all option quotes with your broker.")
