Product Requirements Document (PRD)
Project: Earnings Volatility (IV Crush) Options Trading Application
Version: 3.1 (Markdown Edition - Verified Math)
1. Product Overview
This application automates the detection, structuring, and position sizing for an earnings-based options trading strategy. By capitalizing on implied volatility (IV) crush, the app recommends Long Calendar Spreads on equities approaching earnings announcements. It relies on strict statistical edges and rigorous risk management derived from backtesting 72,500 historical earnings events.
2. Data Ingestion & Core Metric Calculation
The backend must interface with a financial data provider (e.g., Tradier, Interactive Brokers, MarketData) to pull historical and real-time options chain data. The engine computes three foundational variables:
A. Term Structure Slope
We require backwardation (negative slope), meaning short-term certainty is currently overbought.
Slope=IV 
near
​	
 −IV 
45+
​	
 
B. Implied vs. Realized Volatility Ratio
Compares forward-looking 30-day implied volatility to backward-looking 30-day realized volatility. A ratio > 1.0 indicates options are mathematically overpriced compared to recent historical stock movement.
Ratio= 
RV 
30
​	
 
IV 
30
​	
 
​	
 
C. 30-Day Average Daily Volume (ADV)
A simple moving average of the underlying equity's daily volume over the last 30 trading days to guarantee execution liquidity.
3. Recommendation Engine Logic
The filtering system applies deterministic rules to the metrics to output a specific trading signal.
Recommend: Slope<0 AND Volume>Threshold AND Ratio>Threshold AND IV Percentile≥70%
Consider: Slope<0 AND at least one other metric fails the optimal threshold.
Avoid: Slope≥0 (Term structure is in contango or flat; edge does not exist).
4. Trade Structuring: The Long Calendar Spread
For tickers labeled Recommend, the system must automatically build the trade block:
Leg 1 (Short): Sell the At-The-Money (ATM) Call/Put expiring nearest to the earnings date.
Leg 2 (Long): Buy the At-The-Money (ATM) Call/Put (same strike) expiring approximately 30 days after Leg 1.
5. Risk Management: The 10% Kelly Criterion
Position sizing is the most critical component. The traditional Kelly formula determines the optimal fraction (f 
∗
 ) of a bankroll to wager to maximize logarithmic wealth growth:
f 
∗
 =p− 
b
q
​	
 
(Where p = Win probability, q = Loss probability, and b = Average Win / Average Loss)
To smooth the equity curve and protect against black swan earnings reactions, the app must apply a hardcoded fractional multiplier to the historical backtest metrics:
f 
applied
​	
 =0.10×f 
∗
 ≈3.25%
System Requirement for Position Sizing:
To optimize the overall portfolio Sharpe ratio while allowing a margin of safety, the app enforces a strict limit on trade sizing:
Max Debit Allocation=Total Portfolio Value×0.06
The UI must clearly display this exact dollar amount and warn the user never to exceed it.
6. Advanced Quantitative Optimizations (v3.0 Upgrades)
The backend engine must incorporate the following statistical filters to isolate the highest-conviction trading setups.
6.1 IV Percentile / Rank Filter
The system must track the underlying asset's Implied Volatility over a 252-trading-day rolling window to calculate the true historical IV extremity:
IV Percentile=( 
252
Count of Days where IV 
historical
​	
 <IV 
current
​	
 
​	
 )×100
(Requires a result of ≥70% to trigger a standard "Recommend" signal).
6.2 Options Expected Move (EM) Approximation
The backend must compute the market's standard 1-deviation Expected Move using the front-week At-The-Money Straddle price:
EM≈0.85×(Price 
ATM Call
​	
 +Price 
ATM Put
​	
 )
6.3 Earnings Magnitude Filter
The engine will query historical data to calculate the mean absolute percentage move over the last 4 to 8 earnings quarters (Move 
hist
​	
 ).
Target Condition: Look for setups where the market is pricing an Expected Move 25%+ wider than the historical average: EM>Move 
hist
​	
 ×1.25
6.4 Multi-Strike Vega Optimization
When asymmetrical directional drift is detected in historical quarterly performance data, evaluate slightly Out-of-the-Money (OTM) strikes.
Logic: If a stock historically drifts upward, evaluate a strike placed +0.5×EM above current spot price to lower maximum risk debit.
6.5 Dynamic Exit Protocols
Take-Profit: Automatically trigger a manual exit alert if the spread value achieves a +25% to +35% return on the premium paid.
Velocity Exit: Monitor the front-week IV during the first 5 minutes of the opening bell. If ≥80% of the expected IV collapse occurs immediately, trigger a sell signal to avoid intraday directional risk.
7. Strategy Validation: Monte Carlo Simulation Module
To build user confidence, the app must feature a testing environment simulating thousands of sequential trades.
Mechanics: Execute a loop simulating 500 trades across 1,000 parallel paths using the 6% sizing constraint.
Outputs: Plot a fan chart displaying the 5th, 50th, and 95th percentile equity curves and calculate the absolute Risk of Ruin (probability of a 50% drawdown).
8. UI / UX Requirements
Framework: Streamlit (Python).
Inputs: Total Portfolio Value (Numeric), Ticker Symbol (Text).
Outputs: Signal Banner (Green/Yellow/Red), Metric Dataframe (Slope, Volume, Ratios, Expected Move), Proposed Trade Legs, and the Maximum Position Size limit in absolute dollars.