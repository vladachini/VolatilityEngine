"""
IBKR live market-data provider (PRD §2 data ingestion).

Uses ib_insync to connect to a running TWS or IB Gateway instance and pull
everything VolatilityEngine.evaluate_ticker() needs:

  iv_near, iv_45, iv_30, rv_30, avg_30day_volume, historical_iv_series,
  atm_call_price, atm_put_price, historical_moves  (+ spot, drift for sizing/legs)

Prerequisites
-------------
1. Install Trader Workstation (TWS) or IB Gateway and log in.
2. Enable the API:  Configure > API > Settings > "Enable ActiveX and Socket Clients".
3. Note the socket port:
       TWS  live = 7496,  paper = 7497
       Gateway live = 4001, paper = 4002
4. `pip install ib_insync`  (already in requirements.txt)

Market-data subscriptions: live option greeks need an OPRA subscription. Without
one, set market_data_type=3 (delayed) — IBKR returns delayed greeks/quotes, which
is fine for testing the pipeline. This module defaults to delayed.

Local laptop vs. headless cloud server: identical code. On a server, run IB Gateway
headless (e.g. via IBC) and point host/port at it. Nothing here assumes a GUI.

Quick standalone test (with TWS/Gateway running):
    python data_provider.py AAPL --port 7497
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _ensure_event_loop() -> None:
    """ib_insync needs an asyncio loop in the current thread.

    Streamlit runs scripts in a worker thread that has no loop by default, which
    makes ib_insync raise 'There is no current event loop'. Create one if missing.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


class IBKRDataProvider:
    """Thin wrapper around ib_insync that returns engine-ready metric dicts."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 17,
        market_data_type: int = 3,  # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen
        timeout: float = 15.0,
        volume_lot_size: int = 100,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.market_data_type = market_data_type
        self.timeout = timeout
        # IB reports historical STK daily volume in round lots (1 lot = 100 shares).
        # Multiply to recover share counts so the ADV liquidity filter is comparable
        # to the PRD's share-based threshold. Verify against a known ticker and set to
        # 1 if your feed already returns shares.
        self.volume_lot_size = volume_lot_size
        self._ib = None  # lazy import so the app loads even without ib_insync

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self):
        _ensure_event_loop()
        try:
            from ib_insync import IB
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "ib_insync is not installed. Run: pip install ib_insync"
            ) from e

        self._ib = IB()
        self._ib.connect(
            self.host, self.port, clientId=self.client_id, timeout=self.timeout
        )
        self._ib.reqMarketDataType(self.market_data_type)
        return self

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()

    @property
    def ib(self):
        if self._ib is None or not self._ib.isConnected():
            raise RuntimeError("Not connected. Call connect() first.")
        return self._ib

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _days_to(expiry: str) -> int:
        """Calendar days from today to an IBKR 'YYYYMMDD' expiry string."""
        d = datetime.strptime(expiry, "%Y%m%d").date()
        return (d - datetime.now().date()).days

    def _mid_or_last(self, ticker) -> Optional[float]:
        """Best available price: mid, then last, then close."""
        bid, ask = ticker.bid, ticker.ask
        if bid and ask and bid > 0 and ask > 0 and not math.isnan(bid) and not math.isnan(ask):
            return (bid + ask) / 2.0
        for px in (ticker.last, ticker.close):
            if px and not math.isnan(px) and px > 0:
                return float(px)
        return None

    def _hist_series(self, contract, what_to_show: str, duration: str = "1 Y",
                     bar: str = "1 day") -> pd.DataFrame:
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1,
        )
        from ib_insync import util
        df = util.df(bars)
        return df if df is not None else pd.DataFrame()

    def _option_snapshot(self, symbol, expiry, strike, right, exchange="SMART",
                         currency="USD"):
        """Return (price, implied_vol) for a single option leg."""
        from ib_insync import Option
        opt = Option(symbol, expiry, strike, right, exchange, currency=currency)
        qualified = self.ib.qualifyContracts(opt)
        if not qualified:
            return None, None
        t = self.ib.reqMktData(opt, "", False, False)
        # Wait for both model greeks (IV) and a usable price; stop early once we have
        # greeks, else keep waiting up to the timeout in case they arrive late.
        for _ in range(int(self.timeout * 2)):
            self.ib.sleep(0.5)
            if t.modelGreeks is not None and self._mid_or_last(t) is not None:
                break
        iv = None
        if t.modelGreeks is not None and t.modelGreeks.impliedVol:
            iv = float(t.modelGreeks.impliedVol)
        price = self._mid_or_last(t)
        self.ib.cancelMktData(opt)
        return price, iv

    # ------------------------------------------------------------------
    # Public: pull everything the engine needs for one symbol
    # ------------------------------------------------------------------
    def fetch_ticker_metrics(
        self, symbol: str, exchange: str = "SMART", currency: str = "USD",
        n_earnings_moves: int = 6,
    ) -> Dict[str, Any]:
        from ib_insync import Stock

        stock = Stock(symbol.upper(), exchange, currency)
        self.ib.qualifyContracts(stock)

        # --- Spot, daily volume, daily returns (TRADES) ---
        trades = self._hist_series(stock, "TRADES", duration="1 Y", bar="1 day")
        if trades.empty:
            raise RuntimeError(f"No historical TRADES data returned for {symbol}.")
        spot = float(trades["close"].iloc[-1])
        avg_30day_volume = int(trades["volume"].tail(30).mean() * self.volume_lot_size)
        closes = trades["close"].astype(float)
        daily_ret = closes.pct_change().dropna()

        # historical_moves: proxy for earnings reactions = the N largest absolute
        # daily moves over the past year (signed). True earnings dates require a
        # fundamentals subscription; this is a transparent stand-in.
        largest = daily_ret.reindex(daily_ret.abs().sort_values(ascending=False).index)
        historical_moves: List[float] = [float(x) for x in largest.head(n_earnings_moves)]
        drift = float(np.mean(historical_moves)) if historical_moves else 0.0

        # --- IV (underlying 30d) history + realized vol (IBKR-computed) ---
        iv_hist = self._hist_series(stock, "OPTION_IMPLIED_VOLATILITY")
        historical_iv_series = (
            iv_hist["close"].astype(float) if not iv_hist.empty else pd.Series(dtype=float)
        )
        iv_30 = float(historical_iv_series.iloc[-1]) if len(historical_iv_series) else 0.0

        rv_hist = self._hist_series(stock, "HISTORICAL_VOLATILITY")
        rv_30 = float(rv_hist["close"].astype(float).iloc[-1]) if not rv_hist.empty else 0.0

        # --- Option chain: ATM strike, near & ~45d expiries ---
        chains = self.ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0] if chains else None)
        if chain is None:
            raise RuntimeError(f"No option chain found for {symbol}.")

        future_exp = sorted(e for e in chain.expirations if self._days_to(e) >= 0)
        if not future_exp:
            raise RuntimeError(f"No future expirations for {symbol}.")
        near_exp = future_exp[0]
        exp_45 = min(future_exp, key=lambda e: abs(self._days_to(e) - 45))
        atm_strike = min(chain.strikes, key=lambda s: abs(s - spot))

        # ATM call IV for the near expiry and the 45d expiry -> term-structure slope
        near_call_px, iv_near = self._option_snapshot(
            symbol, near_exp, atm_strike, "C", exchange, currency
        )
        _, iv_45 = self._option_snapshot(
            symbol, exp_45, atm_strike, "C", exchange, currency
        )
        # Front-week straddle for the Expected Move
        near_put_px, _ = self._option_snapshot(
            symbol, near_exp, atm_strike, "P", exchange, currency
        )

        return {
            "ticker": symbol.upper(),
            "spot": spot,
            "iv_near": iv_near or 0.0,
            "iv_45": iv_45 or 0.0,
            "iv_30": iv_30,
            "rv_30": rv_30,
            "volume": avg_30day_volume,
            "atm_call": near_call_px or 0.0,
            "atm_put": near_put_px or 0.0,
            "drift": drift,
            "historical_iv_series": [float(x) for x in historical_iv_series.tolist()],
            "hist_moves": historical_moves,
            "_meta": {
                "atm_strike": atm_strike,
                "near_expiry": near_exp,
                "exp_45": exp_45,
                "days_to_near": self._days_to(near_exp),
                "days_to_45leg": self._days_to(exp_45),
                "market_data_type": self.market_data_type,
            },
        }


def fetch_ibkr_metrics(
    symbol: str, host: str = "127.0.0.1", port: int = 7497, client_id: int = 17,
    market_data_type: int = 3,
) -> Dict[str, Any]:
    """One-shot convenience wrapper: connect, fetch, disconnect."""
    provider = IBKRDataProvider(
        host=host, port=port, client_id=client_id, market_data_type=market_data_type
    )
    try:
        provider.connect()
        return provider.fetch_ticker_metrics(symbol)
    finally:
        provider.disconnect()


if __name__ == "__main__":  # pragma: no cover - manual/headless smoke test
    import argparse
    import json

    p = argparse.ArgumentParser(description="Fetch engine metrics from IBKR via ib_insync.")
    p.add_argument("symbol")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497, help="TWS 7497(paper)/7496(live), GW 4002/4001")
    p.add_argument("--client-id", type=int, default=17)
    p.add_argument("--market-data-type", type=int, default=3, help="1 live, 3 delayed")
    args = p.parse_args()

    data = fetch_ibkr_metrics(
        args.symbol, host=args.host, port=args.port,
        client_id=args.client_id, market_data_type=args.market_data_type,
    )
    # trim long series for readable console output
    preview = dict(data)
    preview["historical_iv_series"] = f"<{len(data['historical_iv_series'])} values>"
    print(json.dumps(preview, indent=2, default=str))
