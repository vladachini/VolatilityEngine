# Earnings Volatility (IV Crush) Trading App

Detects, structures, and sizes earnings-based **Long Calendar Spread** options trades
by capitalizing on implied-volatility crush. Implements PRD v3.1.

- `engine.py` — `VolatilityEngine`: all backend math (metrics, signal routing, Kelly,
  trade structuring, Monte Carlo validation).
- `data_provider.py` — `IBKRDataProvider`: live market data via IBKR + `ib_insync`.
- `backtest.py` — Black-Scholes calendar-spread backtest that *derives* the strategy's
  win rate / avg win / avg loss (which then drive Kelly sizing and Monte Carlo).
- `app.py` — Streamlit UI: inputs, signal banner, metric table, trade legs, position
  sizing, backtest, and Monte Carlo fan chart.

## Setup

```bash
cd /Users/vk/StartUps/VolatilityEngine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Click **🎯 Load demo ticker** in the sidebar to
populate a clean "Recommend" setup and exercise every panel immediately.

## Live data via Interactive Brokers (`ib_insync`)

The app can pull live/delayed market data straight from IBKR.

**One-time setup**
1. Install **Trader Workstation (TWS)** or **IB Gateway** and log in (paper or live).
2. Enable the API: *Configure → API → Settings → "Enable ActiveX and Socket Clients"*.
3. Socket ports: TWS `7497` (paper) / `7496` (live); Gateway `4002` / `4001`.
4. Without an OPRA options subscription, use **Delayed** market data (the app's default).

**In the app:** sidebar → *Data source → IBKR (ib_insync)* → set host/port → **📡 Fetch
live data**. Fetched IV term structure, IV/RV, ADV, ATM straddle, and IV history populate
the inputs, then the engine re-evaluates automatically.

**Headless / CLI test** (with TWS or Gateway running):
```bash
python data_provider.py AAPL --port 7497          # paper TWS, delayed data
```

**Cloud deployment:** identical code — run IB Gateway headless (e.g. via IBC) on the
server and point host/port at it. Nothing assumes a GUI.

What gets mapped from IBKR → engine:

| Engine input | IBKR source |
|---|---|
| `iv_near`, `iv_45` | ATM call model-greek IV at near & ~45-day expiries |
| `iv_30`, `historical_iv_series` | `OPTION_IMPLIED_VOLATILITY` daily history (1Y) |
| `rv_30` | `HISTORICAL_VOLATILITY` (latest) |
| `avg_30day_volume` | `TRADES` daily volume, 30-day mean |
| `atm_call`, `atm_put` | front-week ATM straddle quotes |
| `historical_moves` | proxy = N largest absolute daily moves (true earnings dates need a fundamentals subscription) |

## Backtest & Monte Carlo (strategy validation, PRD §5 & §7)

The app has two validation layers under the signal:

1. **Backtest** (`backtest.py`) — prices an ATM long calendar with Black-Scholes
   before and after each earnings event. The edge comes from modeling the two real
   effects: front-week **IV crush** toward the back-month level, and the **vol-risk
   premium** (realized moves average smaller than the straddle-implied move); large
   realized moves push the underlying off-strike and create the loss tail. It outputs
   win rate, avg win/loss, profit factor, expectancy, CAGR, max drawdown, Sharpe, and
   an **empirical Kelly** — independent of the PRD's hard-coded constants, so you can
   sanity-check the assumed edge. Run standalone: `python backtest.py`.
2. **Monte Carlo** (`engine.run_monte_carlo`) — 500 trades × 1000 paths under the 6%
   sizing rule; plots 5th/50th/95th percentile equity curves and Risk of Ruin. It can
   consume the **backtest-derived** win/loss stats instead of the PRD constants (toggle
   in the UI).

Notes on interpretation:
- The backtest's events are a **cross-sectional sample** (distribution of outcomes),
  not a single 2,000-trade timeline — growth is reported as **CAGR** annualized by an
  assumed trades/year, not raw compounded return.
- A trade is skipped if the calendar isn't a valid debit structure (front leg richer
  than back) or the debit is < 0.4% of spot (unrealistic leverage).
- Signal convention: the engine recommends on **backwardation** — `Slope = IV_near -
  IV_45 > 0` (elevated front-week IV, the IV-crush setup). This matches the P&L model
  here. (The PRD text labeled this "negative slope," which was an error; the gate was
  corrected to `slope > 0`.) The backtest validates trade **economics + sizing**;
  signal gating lives in `evaluate_ticker`.

## Notes

- Sizing surfaces both the PRD's applied 10% Kelly (~3.25%) and the hardcoded 6% max
  debit cap, which differ in the PRD by design.
- `historical_moves` is currently a transparent proxy (largest daily moves), since exact
  earnings dates require an IBKR fundamentals subscription. Swap in real earnings dates
  when that data source is available.
- Educational tool — not investment advice.
```
