# Volatility Engine — Earnings IV-Crush Trading App

Detects, structures, and sizes earnings-based **Long Calendar Spread** options trades by
capitalizing on implied-volatility crush. Implements **PRD v3.1** with a dark,
mobile-responsive Streamlit dashboard.

| File | Purpose |
|---|---|
| `engine.py` | `VolatilityEngine` — all strategy math: metrics, signal routing, Kelly sizing, trade structuring, exit protocols, Monte Carlo. |
| `app.py` | Streamlit UI — signal hero + gate pills, metric cards, IV charts, trade plan with payoff diagram, **Upcoming Earnings** calendar, watchlist scanner, live backtest & Monte Carlo. |
| `backtest.py` | Black-Scholes calendar-spread backtest that **derives** win rate / avg win / avg loss (feeding Kelly + Monte Carlo). Accepts real-event CSVs. |
| `data_provider_yf.py` | **Yahoo Finance provider** — free, keyless live data incl. real earnings dates. |
| `data_provider_finnhub.py` | **Finnhub earnings-calendar provider** — free key; market-wide upcoming earnings tagged BMO/AMC, with entry-session timing. |
| `data_provider.py` | `IBKRDataProvider` — live/delayed market data via Interactive Brokers + `ib_insync`. |
| `demo.py` | The demo fixture shared by the app and the tests. |
| `tests/` | Offline sanity tests for the engine, backtest, and the Yahoo + Finnhub helpers. |

---

## Quick start (step by step)

**Prerequisites:** Python **3.10+** (3.11 recommended) and `pip`. Nothing else is needed
for demo mode — live data via IBKR is optional (see below).

```bash
# 1. Clone and enter the repo
git clone https://github.com/vladachini/VolatilityEngine.git
cd VolatilityEngine

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**. On first load it's already evaluating a
clean *Recommend* setup — press **Load demo ticker** in the sidebar at any time to
reset to it. The backtest and Monte Carlo panels compute live; drag their sliders and
everything re-derives instantly.

**Mobile:** the layout is fully responsive — cards reflow, charts resize, and the
sidebar collapses behind the `»` toggle (top-left). To open it from your phone on the
same network: `streamlit run app.py --server.address 0.0.0.0`, then browse to
`http://<your-computer-ip>:8501`.

**Run the tests / standalone backtest:**

```bash
python tests/test_engine.py          # engine math (no extra deps; pytest also works)
python tests/test_yf_helpers.py      # Yahoo data-massaging helpers (offline)
python tests/test_finnhub_helpers.py # earnings-calendar / entry-timing helpers (offline)
python backtest.py                   # prints derived win rate, Kelly, CAGR, Sharpe...
```

---

## The strategy (PRD v3.1)

Ahead of earnings, front-week implied volatility is systematically bid up. A **long
calendar spread** (sell the front-week ATM option, buy the same strike ~30 days out)
harvests the collapse ("IV crush") of that front-week premium after the announcement.

**Signal routing (§3)** — deterministic gates, all surfaced as pills in the UI:

| Signal | Condition |
|---|---|
| **Recommend** | Backwardation **and** ADV > 1M **and** IV/RV > 1.2 **and** IV percentile ≥ 70% |
| **Consider** | Backwardation, but ≥ 1 other gate fails |
| **Avoid** | No backwardation (`Slope = IV_near − IV_45 ≤ 0`) — the edge does not exist |

> Sign convention: backwardation = **elevated front-week IV**, i.e. `Slope > 0`.
> (The PRD prose called this "negative slope" — a labeling error; the math here is verified.)

**Quant filters & risk (§5–§7):**

- **IV percentile (6.1)** — current IV vs a 252-day window; ≥ 70% required.
- **Expected Move (6.2)** — `EM ≈ 0.85 × (ATM call + ATM put)`, shown in $ and % of spot.
- **Magnitude premium (6.3)** — `EM% > 1.25 × mean historical earnings move%` upgrades a
  Recommend to **High conviction**. *(Fixed: this comparison is now done in
  like-for-like units — % of spot — instead of dollars vs fractions.)*
- **Strike tilt (6.4)** — when |mean quarterly drift| > 1% (configurable), the strike is
  shifted ±0.5×EM in the drift direction to cheapen the debit.
- **Exit protocols (6.5)** — quantified in the trade panel: take-profit alerts at
  **+25% / +35%** on debit, a **velocity exit** IV level (if ≥80% of the expected crush
  prints in the first 5 minutes, get out), and max loss = debit paid.
- **Sizing (§5)** — `f* = p − q/b` → 10% fractional Kelly ≈ **3.24%** suggested, under a
  hard **6% max-debit cap**; the UI converts both into whole contract counts.
- **Validation (§7)** — Monte Carlo: 500 trades × 1,000 paths fan chart (5/50/95th
  percentiles), **Risk of Ruin** (P of a ≥50% drawdown), and P(finish below start). You
  can size at the 6% cap or the applied Kelly and compare. Each trade risks a *fixed
  fraction of current equity*, so capital **compounds geometrically** (exponential): the
  median is a straight line on the default **log** axis and an upward-curving sweep on the
  **linear** axis — toggle between them in the tab.

**Backtest (§5/§7)** — `backtest.py` prices an ATM calendar with Black-Scholes before and
after each earnings event. The edge comes from the two real effects (front-week IV crush
toward the back-month level + the vol-risk premium: realized moves average smaller than
implied), while large surprises produce the loss tail. It reports win rate, avg win/loss,
profit factor, expectancy, CAGR, max drawdown, Sharpe, and an **empirical Kelly** — all
independent of the PRD's hard-coded constants, so the assumed edge is *checked*, not
asserted. The Monte Carlo can run on either the PRD constants or these derived stats.
Events are a cross-sectional sample (growth annualized via trades/year); trades are
skipped if the calendar isn't a valid debit ≥ 0.4% of spot.

**Real events:** the Backtest tab accepts a CSV (columns `iv_near, iv_far, iv_near_post,
iv_far_post, realized_move`) and runs the identical pricing on your own historical
earnings data instead of the synthetic universe.

---

## Live data — two options

### Option A: Yahoo Finance (free — no account, no API key)

Works out of the box: sidebar → *Data source → Yahoo (yfinance)* → enter a symbol →
**Fetch from Yahoo**. Spot, ATM IVs (near / interpolated 30d / ~45d), realized vol,
ADV, the front-week straddle, **real past earnings reactions and the next earnings
date** all populate automatically.

The **Scanner** tab scans a whole watchlist through Yahoo and ranks tickers by
signal → conviction → IV percentile, with one-click loading into the engine.

Honest limitations (also in `data_provider_yf.py`):
- Quotes are ~15-min delayed — fine for setup detection, not for the §6.5 velocity exit.
- Yahoo has no IV *history*, so the IV-percentile series uses a trailing 30-day
  realized-vol distribution as a documented proxy. IBKR provides the true series.
- Yahoo rate-limits aggressive IPs; the app caches fetches for 10 minutes.

#### Upcoming Earnings tab — calendar-driven candidates (free Finnhub key)

yfinance has no market-wide *forward* earnings calendar, so the **Upcoming Earnings**
tab pulls one from [Finnhub](https://finnhub.io/register) (free key, 60 calls/min). It
lists every US name reporting in the next *N* days, tagged **BMO/AMC**, runs each through
the engine on Yahoo data, and ranks the setups — each row showing the **session you'd
enter** the calendar:

- **AMC** (reports after the close on day *D*): the crush prints overnight, so you enter
  by the close of *D* → labelled *"Today · AMC"* etc.
- **BMO** (reports before the open on day *D*): you must be in by the prior close → entry
  rolls back to the previous business day (weekends skipped).

Set the key in the sidebar (*Earnings calendar key*) or via the `FINNHUB_API_KEY`
environment variable / Streamlit secret. The calendar is one API call (cached 15 min);
each ranked name is then a Yahoo fetch, so keep *Max names* modest to stay within limits.

```bash
export FINNHUB_API_KEY=your_key
python data_provider_finnhub.py --days 7        # prints the agenda with entry timing
```

### Option B: Interactive Brokers (real-time, OPRA-quality)

The app can pull live/delayed data straight from IBKR with `ib_insync`.

**One-time setup**
1. Install **Trader Workstation (TWS)** or **IB Gateway** and log in (paper or live).
2. Enable the API: *Configure → API → Settings → "Enable ActiveX and Socket Clients"*.
3. Socket ports: TWS `7497` (paper) / `7496` (live); Gateway `4002` / `4001`.
4. Without an OPRA options subscription, use **Delayed** market data (the app's default).

**In the app:** sidebar → *Data source → IBKR (ib_insync)* → set host/port → **Fetch
live data**. The fetched IV term structure, IV/RV, ADV, ATM straddle, and IV history
populate the inputs and the engine re-evaluates automatically.

**Headless / CLI test** (with TWS or Gateway running):
```bash
python data_provider.py AAPL --port 7497          # paper TWS, delayed data
```

**Cloud deployment:** identical code — run IB Gateway headless (e.g. via IBC) on the
server and point host/port at it. Nothing assumes a GUI.

| Engine input | IBKR source |
|---|---|
| `iv_near`, `iv_45` | ATM call model-greek IV at near & ~45-day expiries |
| `iv_30`, `historical_iv_series` | `OPTION_IMPLIED_VOLATILITY` daily history (1Y) |
| `rv_30` | `HISTORICAL_VOLATILITY` (latest) |
| `avg_30day_volume` | `TRADES` daily volume, 30-day mean |
| `atm_call`, `atm_put` | front-week ATM straddle quotes |
| `historical_moves` | proxy = N largest absolute daily moves (true earnings dates need a fundamentals subscription) |

---

## Deploying (free, shareable URL)

[Streamlit Community Cloud](https://share.streamlit.io): point it at this repo,
`app.py` as the entrypoint — the included `.streamlit/config.toml` carries the dark
theme. Works on any phone browser. (For IBKR data in the cloud you'd run IB Gateway
on a reachable host; demo/manual mode needs nothing.)

---

## Notes

- The 6% cap and the ~3.24% applied Kelly differ **by design** (PRD §5): Kelly is the
  suggestion, 6% is the never-exceed ceiling. The UI shows both, in dollars and contracts.
- `historical_moves` from IBKR is a transparent proxy (largest daily moves) until an
  earnings-dates data source is wired in.
- Educational tool — **not investment advice**. Verify every quote with your broker.
